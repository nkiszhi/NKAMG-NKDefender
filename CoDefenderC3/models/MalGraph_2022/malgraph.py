import logging
import sys
import torch
from torch import nn
from torch.nn import Linear
from torch.nn import functional as pt_f
from torch_geometric.data import Batch, Data
from torch_geometric.nn.conv import GCNConv, SAGEConv
from torch_geometric.nn.glob import global_max_pool, global_mean_pool
sys.path.append("..")
from models.MalGraph_2022.ParameterClasses import ModelParams
from models.MalGraph_2022.Vocabulary import Vocab

# 辅助函数
def div_with_small_value(n, d, eps=1e-8):
    """
    处理除法运算，避免分母为零
    :param n: 分子
    :param d: 分母
    :param eps: 小值，避免分母为零
    :return: 除法结果
    """
    d = d * (d > eps).float() + eps * (d <= eps).float()
    return n / d

def padding_tensors(tensor_list):
    """
    对张量列表进行填充，使它们具有相同的长度
    :param tensor_list: 张量列表
    :return: 填充后的张量和对应的掩码
    """
    num = len(tensor_list)
    max_len = max([s.shape[0] for s in tensor_list])
    out_dims = (num, max_len, *tensor_list[0].shape[1:])
    out_tensor = tensor_list[0].data.new(*out_dims).fill_(0)
    mask = tensor_list[0].data.new(*out_dims).fill_(0)
    for i, tensor in enumerate(tensor_list):
        length = tensor.size(0)
        out_tensor[i, :length] = tensor
        mask[i, :length] = 1
    return out_tensor, mask

def inverse_padding_tensors(tensors, masks):
    """
    去除填充的部分，恢复原始的张量
    :param tensors: 填充后的张量
    :param masks: 掩码
    :return: 去除填充后的张量和对应的批次索引列表
    """
    mask_index = torch.sum(masks, dim=-1) / masks.size(-1)
    _out_mask_select = torch.masked_select(tensors, (masks == 1)).view(-1, tensors.size(-1))
    batch_index = torch.sum(mask_index, dim=-1)
    batch_idx_list = []
    for idx, num in enumerate(batch_index):
        batch_idx_list.extend([idx for _ in range(int(num))])
    return _out_mask_select, batch_idx_list

class HierarchicalGraphNeuralNetwork(nn.Module):
    def __init__(self, model_params: ModelParams, external_vocab: Vocab, global_log: logging.Logger):
        super().__init__()
        self.model_params = model_params
        self.external_vocab = external_vocab
        self.global_log = global_log

        # 验证图卷积层类型
        self.conv = model_params.gnn_type.lower()
        if self.conv not in ['graphsage', 'gcn']:
            raise ValueError(f"Unsupported GNN type: {model_params.gnn_type}. Supported types are 'graphsage' and 'gcn'.")

        # 验证全局池化层类型
        self.pool = model_params.pool_type.lower()
        if self.pool not in ["global_max_pool", "global_mean_pool"]:
            raise ValueError(f"Unsupported pooling type: {model_params.pool_type}. Supported types are 'global_max_pool' and 'global_mean_pool'.")

        # 层次 1：控制流图（CFG）嵌入和池化
        cfg_filter_list = self._parse_filter_list(model_params.cfg_filters, model_params.acfg_init_dims)
        self.cfg_conv_layers = self._create_conv_layers(cfg_filter_list, self.conv, 'CFG_gnn_')
        self.dropout = nn.Dropout(p=model_params.dropout_rate)

        # 层次 2：函数调用图（FCG）嵌入和池化
        self.external_embedding_layer = nn.Embedding(
            num_embeddings=external_vocab.max_vocab_size + 2,
            embedding_dim=cfg_filter_list[-1],
            padding_idx=external_vocab.pad_idx
        )
        fcg_filter_list = self._parse_filter_list(model_params.fcg_filters, cfg_filter_list[-1])
        self.fcg_conv_layers = self._create_conv_layers(fcg_filter_list, self.conv, 'FCG_gnn_')

        # 最后投影层
        self.projection_layers = nn.Sequential(
            Linear(in_features=fcg_filter_list[-1], out_features=int(fcg_filter_list[-1] / 2)),
            nn.ReLU(),
            Linear(in_features=int(fcg_filter_list[-1] / 2), out_features=int(fcg_filter_list[-1] / 4)),
            nn.ReLU(),
            Linear(in_features=int(fcg_filter_list[-1] / 4), out_features=1)
        )
        self.last_activation = nn.Sigmoid()

        self.last_fcg_feature = None  # 缓存中间特征

    def _parse_filter_list(self, filters, init_dims):
        """
        解析过滤器列表
        :param filters: 过滤器配置
        :param init_dims: 初始维度
        :return: 过滤器列表
        """
        if isinstance(filters, str):
            filter_list = [int(number_filter) for number_filter in filters.split("-")]
        else:
            filter_list = [int(filters)]
        filter_list.insert(0, init_dims)
        return filter_list

    def _create_conv_layers(self, filter_list, conv_type, layer_prefix):
        """
        创建图卷积层
        :param filter_list: 过滤器列表
        :param conv_type: 图卷积层类型
        :param layer_prefix: 层名称前缀
        :return: 图卷积层模块列表
        """
        conv_params = {
            'graphsage': [
                dict(in_channels=filter_list[i], out_channels=filter_list[i + 1], bias=True)
                for i in range(len(filter_list) - 1)
            ],
            'gcn': [
                dict(in_channels=filter_list[i], out_channels=filter_list[i + 1], cached=False, bias=True)
                for i in range(len(filter_list) - 1)
            ]
        }
        conv_constructor = {
            'graphsage': SAGEConv,
            'gcn': GCNConv
        }
        layers = nn.ModuleList()
        for i, params in enumerate(conv_params[conv_type]):
            layer = conv_constructor[conv_type](**params)
            setattr(self, f"{layer_prefix}{i + 1}", layer)
            layers.append(layer)
        return layers

    def _forward_gnn(self, batch, conv_layers):
        """
        通用的图卷积前向传播方法
        :param batch: 输入批次
        :param conv_layers: 图卷积层列表
        :return: 处理后的批次
        """
        in_x, edge_index = batch.x, batch.edge_index
        for layer in conv_layers:
            out_x = layer(x=in_x, edge_index=edge_index)
            out_x = pt_f.relu(out_x, inplace=True)
            out_x = self.dropout(out_x)
            in_x = out_x
        batch.x = in_x
        return batch

    def _aggregate_pooling(self, batch):
        """
        通用的全局池化方法
        :param batch: 输入批次
        :return: 池化后的结果
        """
        if self.pool == 'global_max_pool':
            return global_max_pool(x=batch.x, batch=batch.batch)
        elif self.pool == 'global_mean_pool':
            return global_mean_pool(x=batch.x, batch=batch.batch)
        else:
            raise ValueError(f"Unsupported pooling type: {self.pool}")

    def forward(self, real_local_batch: Batch, real_bt_positions: list, bt_external_names: list,
                bt_all_function_edges: list, local_device: torch.device):
        # 层次 1：控制流图（CFG）处理
        rtn_local_batch = self._forward_gnn(real_local_batch, self.cfg_conv_layers)
        x_cfg_pool = self._aggregate_pooling(rtn_local_batch)

        # 验证输入长度
        assert len(real_bt_positions) - 1 == len(bt_external_names), "Batch size mismatch for external names."
        assert len(real_bt_positions) - 1 == len(bt_all_function_edges), "Batch size mismatch for function edges."

        # 构建函数调用图（FCG）
        fcg_list = []
        for idx_batch in range(len(real_bt_positions) - 1):
            start_pos, end_pos = real_bt_positions[idx_batch: idx_batch + 2]
            idx_x_cfg = x_cfg_pool[start_pos: end_pos]
            external_names_idx = [self.external_vocab.word2idx(name) for name in bt_external_names[idx_batch]]
            idx_x_external = self.external_embedding_layer(
                torch.tensor([external_names_idx], dtype=torch.long).to(local_device)
            ).squeeze(dim=0)
            idx_x_total = torch.cat([idx_x_cfg, idx_x_external], dim=0)
            idx_function_edge = torch.tensor(bt_all_function_edges[idx_batch], dtype=torch.long).to(local_device)
            idx_graph_data = Data(x=idx_x_total, edge_index=idx_function_edge).to(local_device)
            fcg_list.append(idx_graph_data)
        fcg_batch = Batch.from_data_list(fcg_list)

        # 层次 2：函数调用图（FCG）处理
        rtn_fcg_batch = self._forward_gnn(fcg_batch, self.fcg_conv_layers)
        x_fcg_pool = self._aggregate_pooling(rtn_fcg_batch)

        # 缓存中间特征用于 DFE
        self.last_fcg_feature = x_fcg_pool.detach()

        # 最后投影层
        bt_final_embed = self.projection_layers(x_fcg_pool)
        bt_pred = self.last_activation(bt_final_embed)
        return bt_pred


    def extract_middle_feature(self):
        """
        提取函数调用图池化后的嵌入特征（用于 DFE 动态特征熵计算）
        返回:
            Tensor: [batch_size, feature_dim]
        """
        if self.last_fcg_feature is not None:
            return self.last_fcg_feature.cpu()
        else:
            raise RuntimeError("请先执行 forward()，以缓存中间特征")

"""
Asm2Vec — Ding et al. 2019
论文: "Asm2Vec: Boosting Static Representation Robustness for Binary Clone Search
       against Code Obfuscation and Compiler Optimization"
会议: IEEE S&P 2019
链接: https://doi.org/10.1109/SP.2019.00003

论文原始算法: PV-DM (Paragraph Vector - Distributed Memory)
  1. 在 CFG 上进行随机游走生成指令序列
  2. 用 PV-DM 模型联合训练 函数嵌入 + 指令嵌入
  3. 预测: 给定函数向量 + 上下文指令, 预测下一指令

本实现:
  1. FunctionPVDM: PV-DM 模型 (函数嵌入 + 指令嵌入 → 预测目标指令)
  2. AsmEmbedNet: 聚合函数嵌入 → 注意力BiLSTM → 分类

输入: 函数级嵌入序列 (LongTensor 或 FloatTensor)
输出: (B, 2) 二分类 logits
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FunctionPVDM(nn.Module):
    """
    PV-DM (Paragraph Vector - Distributed Memory)
    联合学习函数嵌入和指令嵌入

    给定函数 f 和上下文指令 [i_{t-w}, ..., i_{t-1}]:
      预测目标指令 i_t
      损失: CrossEntropy(predict(f, context), i_t)
    """

    def __init__(self, n_functions, n_instructions, embed_dim=128, context_size=5):
        """
        参数:
            n_functions: 函数词表大小
            n_instructions: 指令词表大小 (操作码类别数)
            embed_dim: 嵌入维度
            context_size: 上下文窗口大小
        """
        super().__init__()
        self.func_embed = nn.Embedding(n_functions, embed_dim)
        self.inst_embed = nn.Embedding(n_instructions, embed_dim, padding_idx=0)
        # PV-DM: 上下文 + 函数向量 → 预测下一指令
        self.predictor = nn.Linear(embed_dim, n_instructions)
        self.context_size = context_size

    def forward(self, func_ids, context_ids):
        """
        参数:
            func_ids: (B,) 函数索引
            context_ids: (B, context_size) 上下文指令索引
        返回:
            (B, n_instructions) 预测 logits
        """
        f_emb = self.func_embed(func_ids)           # (B, D)
        c_emb = self.inst_embed(context_ids).mean(1) # (B, D) 上下文均值
        combined = f_emb + c_emb                      # PV-DM: 加法组合
        return self.predictor(combined)

    def get_function_embeddings(self):
        """获取所有函数的嵌入矩阵"""
        return self.func_embed.weight.detach()


class AsmEmbedNet(nn.Module):
    """
    Asm2Vec 恶意软件分类器

    对文件中所有函数的嵌入序列进行 LSTM 编码 → 分类
    函数嵌入由 PV-DM 预训练生成 (或外部提供)
    """

    def __init__(self, input_dim=6144, func_embed_dim=128, hidden_dim=256,
                 num_layers=2):
        """
        参数:
            input_dim: 总输入维度 (num_functions × func_embed_dim)
            func_embed_dim: 单个函数嵌入维度 (PV-DM 输出)
            hidden_dim: LSTM 隐藏维度
            num_layers: LSTM 层数
        """
        super().__init__()
        self.func_embed_dim = func_embed_dim

        # 函数嵌入序列 → LSTM 编码
        self.lstm = nn.LSTM(
            func_embed_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=0.3)

        # 注意力聚合 (论文使用注意力加权而非简单均值池化)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1))

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 2),  # logits → CrossEntropyLoss
        )

    def forward(self, x):
        """
        参数:
            x: FloatTensor(B, input_dim) 或 (B, n_funcs, embed_dim)
        返回:
            (B, 2) 二分类 logits
        """
        # 重塑为函数序列
        if x.dim() == 2:
            x = x.float().reshape(x.size(0), -1, self.func_embed_dim)

        # BiLSTM 编码
        lstm_out, _ = self.lstm(x)  # (B, T, hidden*2)

        # 注意力加权聚合
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)  # (B, T, 1)
        context = (attn_weights * lstm_out).sum(dim=1)             # (B, hidden*2)

        return self.classifier(context)

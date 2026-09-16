import os
import json
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from tqdm import tqdm
from models.MalGraph_2022.utils import calculate_file_hash, extract_cfg_and_fcg
from models.MalGraph_2022.Vocabulary import Vocab


class DynamicVocab:
    """动态词表: 增量累积外部函数名频率, 按需生成 Vocab 对象"""

    def __init__(self, vocab_file, max_vocab_size=10000):
        self.vocab_file = vocab_file
        self.max_vocab_size = max_vocab_size
        self.counter = {}
        # 从已有文件恢复
        if os.path.exists(vocab_file):
            try:
                with open(vocab_file, "r", encoding="utf-8") as f:
                    for line in f:
                        obj = json.loads(line)
                        name = obj.get("f_name")
                        cnt = int(obj.get("count", 0))
                        if name:
                            self.counter[name] = self.counter.get(name, 0) + cnt
            except Exception:
                pass

    def update(self, names):
        """累积一批外部函数名"""
        for name in names:
            if name:
                self.counter[name] = self.counter.get(name, 0) + 1

    def save(self):
        """将频率计数持久化到 JSONL 文件"""
        os.makedirs(os.path.dirname(self.vocab_file) or ".", exist_ok=True)
        sorted_items = sorted(self.counter.items(), key=lambda x: -x[1])
        with open(self.vocab_file, "w", encoding="utf-8") as f:
            for name, cnt in sorted_items:
                f.write(json.dumps({"f_name": name, "count": int(cnt)},
                                   ensure_ascii=False) + "\n")

    def get_vocab(self):
        """生成 Vocab 对象 (需先 save)"""
        self.save()
        return Vocab(freq_file=self.vocab_file, max_vocab_size=self.max_vocab_size)


def is_valid_file(path: str) -> bool:
    """快速检测文件是否有效（不做特征提取）"""
    try:
        if not os.path.exists(path):
            return False
        if os.path.getsize(path) == 0:
            return False
        with open(path, "rb") as f:
            f.read(1)
        return True
    except Exception:
        return False


def vocab_index(vocab, name: str) -> int:
    """
    安全取 vocab 索引：
    - Vocab.__getitem__ 已实现 OOV 回退到 <unk>，这里直接用下标即可
    - 对 None / 空串 做一下兜底
    """
    if not name:
        return vocab[vocab.unk_token]
    return vocab[name]


class MalgraphDynamicDataset(Dataset):
    """
    动态从PE样本提取/缓存 MalGraph 特征（.pt/.json），并增量更新外部函数词表
    - 优先加载 .pt 缓存
    - 提取失败：返回 None，让 collate_fn 过滤
    - vocab 只构建一次，避免 "creating vocab..." 重复打印
    """
    def __init__(self, samples, pt_cache_dir, vocab_cache_path,
                 max_vocab_size=10000, force_reextract=False, vocab_save_interval=10):
        self.samples = samples
        self.pt_cache_dir = pt_cache_dir
        self.vocab_cache_path = vocab_cache_path
        self.max_vocab_size = max_vocab_size
        self.force_reextract = force_reextract
        self.vocab_save_interval = vocab_save_interval

        os.makedirs(self.pt_cache_dir, exist_ok=True)

        self.dyn_vocab = DynamicVocab(vocab_file=vocab_cache_path, max_vocab_size=max_vocab_size)
        self._vocab_obj = None            # 缓存最终 Vocab 对象（只创建一次）
        self.counter_since_save = 0       # 控制词表写盘频率

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]

        # 快速校验，无效直接跳过（返回 None）
        if not is_valid_file(file_path):
            # print(f"[SKIP] invalid file: {file_path}")
            return None

        try:
            file_hash = calculate_file_hash(file_path)
            pt_path = os.path.join(self.pt_cache_dir, f"{file_hash}.pt")
            json_path = os.path.join(self.pt_cache_dir, f"{file_hash}.json")

            # 1) 优先读缓存
            if os.path.exists(pt_path) and not self.force_reextract:
                try:
                    data = torch.load(pt_path, weights_only=False)
                    return data, torch.tensor(label, dtype=torch.float)
                except Exception:
                    # 损坏缓存，继续重提取
                    pass

            # 2) 提取特征
            sample_features = extract_cfg_and_fcg(file_path, vocab_dict={})
            if not sample_features \
               or "acfg_list" not in sample_features \
               or "function_names" not in sample_features \
               or "function_edges" not in sample_features:
                # print(f"[SKIP] feature missing: {file_path}")
                return None

            # 3) 更新动态词表（只累积外部函数）
            local_count = len(sample_features["acfg_list"])
            all_names = sample_features["function_names"]
            external_names = all_names[local_count:]
            self.dyn_vocab.update(external_names)

            self.counter_since_save += 1
            if self.counter_since_save >= self.vocab_save_interval:
                self.dyn_vocab.save()
                self.counter_since_save = 0

            # 4) 缓存 JSON（便于排查）
            sample_features["label"] = label
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(sample_features, jf)

            # 5) 构造 PyG Data
            if self._vocab_obj is None:
                self._vocab_obj = self.dyn_vocab.get_vocab()
            vocab_obj = self._vocab_obj

            acfg_list = []
            for one_acfg in sample_features['acfg_list']:
                block_features = one_acfg['block_features']
                block_edges = one_acfg['block_edges']
                acfg_list.append(
                    Data(
                        x=torch.tensor(block_features, dtype=torch.float),
                        edge_index=torch.tensor(block_edges, dtype=torch.long)
                    )
                )

            # 这里使用 vocab 的 __getitem__（支持 OOV 回退）
            external_indices = [vocab_index(vocab_obj, f_name) for f_name in external_names]

            data = Data(
                hash=file_hash,
                local_acfgs=acfg_list,
                external_list=external_indices,
                function_edges=sample_features['function_edges'],
                targets=torch.tensor(label, dtype=torch.float)
            )

            # 6) 缓存 .pt
            torch.save(data, pt_path)
            return data, torch.tensor(label, dtype=torch.float)

        except Exception as e:
            # print(f"[SKIP] error: {file_path} - {e}")
            return None

    def finalize_vocab(self):
        """训练前/后确保词表落盘，并返回最终 Vocab 对象"""
        self.dyn_vocab.save()
        if self._vocab_obj is None:
            self._vocab_obj = self.dyn_vocab.get_vocab()
        return self._vocab_obj


class FilteredDataset(Dataset):
    """
    预扫描阶段做**快速过滤**：
    - 有 .pt 缓存 → 直接保留
    - 否则仅做“存在性 + 可读性 + size>0”检查
    - 不在预扫描阶段跑 extract_cfg_and_fcg（避免慢/崩）
    """
    def __init__(self, base_dataset: MalgraphDynamicDataset):
        self.base = base_dataset
        self.valid_indices = []

        print("预扫描数据集，快速过滤无效样本 ...")
        for i in tqdm(range(len(base_dataset)), desc="Filtering invalid samples"):
            file_path, _ = base_dataset.samples[i]
            if not is_valid_file(file_path):
                continue

            file_hash = calculate_file_hash(file_path)
            pt_path = os.path.join(base_dataset.pt_cache_dir, f"{file_hash}.pt")
            # 有缓存直接保留；无缓存也保留（后续 __getitem__ 提取，失败再被 collate 过滤）
            self.valid_indices.append(i)

        print(f"有效样本: {len(self.valid_indices)} / {len(base_dataset)}")

    def __getitem__(self, idx):
        return self.base[self.valid_indices[idx]]

    def __len__(self):
        return len(self.valid_indices)


def collate_skip_none_to_batch(batch):
    """
    DataLoader 的 collate_fn：
    - 过滤 None
    - 把 (Data, label) → (Batch, labels_tensor)
    - 如果全是 None，返回 None 让训练循环跳过
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    data_list = [d for (d, _) in batch]
    labels = torch.tensor([float(l) for (_, l) in batch], dtype=torch.float)
    pyg_batch = Batch.from_data_list(data_list)
    return pyg_batch, labels

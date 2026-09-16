import os
import json
import traceback
import math
from tqdm import tqdm
from collections import Counter
import torch
from torch_geometric.data import Data
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, TimeoutError, as_completed
import multiprocessing
import gc

from models.MalGraph_2022.Vocabulary import Vocab
from feature_extraction.extract_feature import scan_load_samples
from models.MalGraph_2022.utils import extract_cfg_and_fcg, save_vocab_from_train

# 路径配置
SAVED_DIR = "./saved"
PT_CACHE_DIR = "./malgraph_pt_cache"
VOCAB_FILE = os.path.join(SAVED_DIR, "train_external_function_name_vocab.jsonl")
JSONL_FILE = os.path.join(SAVED_DIR, "features.jsonl")
BAD_FILE_LOG = os.path.join(SAVED_DIR, "bad_files.txt")

# 并行进程数（用于 .pt 转换）
MAX_WORKERS_PT = math.ceil(multiprocessing.cpu_count() / 2)

def get_timeout_for_file(file_path):
    """根据文件大小动态设定超时时间"""
    try:
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
    except OSError:
        return 60
    if size_mb < 5:
        return 60
    elif size_mb < 20:
        return 90
    else:
        return 120

def safe_extract_bn(file_path, vocab_counter, timeout_val):
    """单进程安全执行 BN 提取，带超时控制"""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(extract_cfg_and_fcg, file_path, vocab_counter)
        return future.result(timeout=timeout_val)

def extract_features_with_scan(base_dir, max_vocab_size=1000):
    """两阶段：先单进程提取 JSONL，再多进程转换 .pt"""
    os.makedirs(SAVED_DIR, exist_ok=True)
    os.makedirs(PT_CACHE_DIR, exist_ok=True)

    vocab_counter = Counter()
    bad_files = []

    samples = scan_load_samples(base_dir)
    print(f"Total samples found: {len(samples)}")
    print("[Stage 1] Extracting features with Binary Ninja (single process)")

    # 阶段 1：BN 提取
    with open(JSONL_FILE, "w", encoding="utf-8") as jsonl_file:
        for file_path, label in tqdm(samples, desc="BN Feature Extraction"):
            timeout_val = get_timeout_for_file(file_path)
            try:
                data = safe_extract_bn(file_path, vocab_counter, timeout_val)
                if data is None:
                    bad_files.append((file_path, "打开失败"))
                    continue
                data["label"] = label
                data["file_name"] = os.path.basename(file_path)
                jsonl_file.write(json.dumps(data) + "\n")
            except TimeoutError:
                bad_files.append((file_path, f"超时({timeout_val}s)"))
            except Exception as e:
                bad_files.append((file_path, f"解析异常: {str(e)}"))
                traceback.print_exc()
            finally:
                gc.collect()

    if bad_files:
        with open(BAD_FILE_LOG, "w", encoding="utf-8") as f:
            for bf, reason in bad_files:
                f.write(f"{bf}\t{reason}\n")
        print(f"Bad files logged to {BAD_FILE_LOG} ({len(bad_files)} files)")

    # 保存 vocab
    save_vocab_from_train(vocab_counter, SAVED_DIR)
    print(f"Vocab saved to {VOCAB_FILE}")

    # 阶段 2：并行 JSONL → .pt
    print("[Stage 2] Converting JSONL to .pt files (multi-process)")
    vocab = Vocab(freq_file=VOCAB_FILE, max_vocab_size=max_vocab_size)
    parse_json_list_2_pyg_object(JSONL_FILE, PT_CACHE_DIR, vocab)

def convert_line_to_pt(line, vocab, output_dir):
    """子进程任务：单条 JSON 转 .pt"""
    item = json.loads(line)
    acfg_list = [
        Data(
            x=torch.tensor(acfg['block_features'], dtype=torch.float),
            edge_index=torch.tensor(acfg['block_edges'], dtype=torch.long)
        ) for acfg in item['acfg_list']
    ]
    external_function_index_list = [
        vocab[f_name] for f_name in item['function_names'][len(acfg_list):]
    ]
    base_name = os.path.splitext(item['file_name'])[0] + ".pt"
    pt_path = os.path.join(output_dir, base_name)
    torch.save(
        Data(
            hash=item['hash'],
            local_acfgs=acfg_list,
            external_list=external_function_index_list,
            function_edges=item['function_edges'],
            targets=item['label']
        ),
        pt_path
    )
    return base_name

def parse_json_list_2_pyg_object(jsonl_file, output_dir, vocab):
    """多进程批量转换 JSONL → .pt"""
    os.makedirs(output_dir, exist_ok=True)
    with open(jsonl_file, "r", encoding="utf-8") as file:
        lines = file.readlines()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS_PT) as executor:
        futures = [executor.submit(convert_line_to_pt, line, vocab, output_dir) for line in lines]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="PT Conversion"):
            pass

if __name__ == "__main__":
    base_dir = r"E:/Experimental data/dr_data"
    extract_features_with_scan(base_dir, max_vocab_size=1000)

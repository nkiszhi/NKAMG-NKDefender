import os
import json
from collections import Counter
from tqdm import tqdm

def collect_json_files(root_dir):
    """收集指定目录下的所有 JSON 文件"""
    json_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".json"):
                json_files.append(os.path.join(dirpath, file))
    return json_files

def extract_external_function_names(json_files):
    """从所有 JSON 文件中提取外部函数名称"""
    counter = Counter()
    for json_file in tqdm(json_files, desc="Processing JSON files"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            function_names = data["function_names"]
            local_function_count = len(data["acfg_list"])  # 本地函数数目
            external_function_names = function_names[local_function_count:]  # 外部函数
            counter.update(external_function_names)  # 更新频率计数
    return counter

def save_vocab_to_jsonl(counter, output_file, top_k=10000):
    """保存外部函数词汇表到 JSONL 文件"""
    with open(output_file, "w", encoding="utf-8") as f:
        for func_name, freq in counter.most_common(top_k):
            f.write(json.dumps({"f_name": func_name, "count": freq}) + "\n")

def main():
    # 定义输入目录和输出文件
    benign_dir = "./output/benign_features"
    malicious_dir = "./output/malicious_features"
    output_file = "./train_external_function_name_vocab.jsonl"

    # 收集 JSON 文件
    benign_files = collect_json_files(benign_dir)
    malicious_files = collect_json_files(malicious_dir)
    all_files = benign_files + malicious_files

    print(f"Found {len(all_files)} JSON files.")

    # 提取外部函数名称并统计频率
    function_name_counter = extract_external_function_names(all_files)

    # 保存 TOP-K 外部函数名称到 JSONL 文件
    save_vocab_to_jsonl(function_name_counter, output_file, top_k=10000)
    print(f"Vocabulary saved to {output_file}")

if __name__ == "__main__":
    main()

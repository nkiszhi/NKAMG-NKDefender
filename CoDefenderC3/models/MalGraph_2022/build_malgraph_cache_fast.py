# -*- coding: utf-8 -*-
import os
import sys
import json
import hashlib
import shutil
import logging
from pathlib import Path
from collections import Counter
from datetime import datetime
from tqdm import tqdm
from binaryninja import BinaryViewType, SymbolType, MediumLevelILOperation

# ====== 直接用你已有的功能 ======
from feature_extraction.extract_feature import scan_load_samples
from models.MalGraph_2022.utils import (
    parse_json_list_2_pyg_object, calculate_file_hash,
)
from models.MalGraph_2022.Vocabulary import Vocab

def extract_cfg_and_fcg(sample_path, vocab_dict):
    """
    提取函数调用图（FCG）和控制流图（CFG/ACFG）
    返回:
        {
          "function_edges": [[caller_idx...],[callee_idx...]],
          "acfg_list": [ {block_number, block_edges[[src...],[dst...]], block_features[[...11维]...]} ... ],
          "function_names": [本地函数名..., 导入函数名...],
          "function_number": 总函数数,
          "hash": 文件sha256
        }
    """
    try:
        # ---- BinaryNinja 解析 ----
        file_hash = calculate_file_hash(sample_path)
        bv = BinaryViewType["PE"].open(sample_path)
        if not bv:
            raise ValueError(f"无法打开文件：{sample_path}")
        bv.update_analysis_and_wait()

        # ---- 外部函数统计 & 词频 ----
        # 一次性拿 symbols，避免多次 API 调用
        import_syms = list(bv.get_symbols_of_type(SymbolType.ImportAddressSymbol))
        for s in import_syms:
            # 直接用 Counter 风格累加更快
            vocab_dict[s.name] = vocab_dict.get(s.name, 0) + 1

        # ---- 函数集合 ----
        local_functions = list(bv.functions)  # 本地函数对象
        external_functions = [s.name for s in import_syms]  # 外部函数名
        n_local = len(local_functions)
        n_external = len(external_functions)
        total_fn = n_local + n_external

        # ---- FCG：建索引 ----
        # 本地函数：起始地址 -> 下标；外部函数：名字 -> 下标
        local_idx_by_addr = {f.start: i for i, f in enumerate(local_functions)}
        func_index = dict(local_idx_by_addr)
        for i, name in enumerate(external_functions):
            func_index[name] = n_local + i

        # 导入地址 -> 名字（用于把常量地址映射到导入函数名）
        import_addr_to_name = {s.address: s.name for s in import_syms}

        # ---- 收集调用边（FCG）----
        function_edges_src = []
        function_edges_dst = []
        CALL = MediumLevelILOperation.MLIL_CALL
        TAIL = MediumLevelILOperation.MLIL_TAILCALL
        CONST_PTR = MediumLevelILOperation.MLIL_CONST_PTR
        IMPORT = MediumLevelILOperation.MLIL_IMPORT

        for caller_idx, func in enumerate(local_functions):
            mlil = func.mlil
            if mlil is None:
                continue

            # 局部绑定，少走属性链
            _getattr = getattr
            _func_index_get = func_index.get
            _addr2name_get = import_addr_to_name.get

            for block in mlil:
                for ins in block:
                    op = ins.operation
                    if op != CALL and op != TAIL:
                        continue

                    dest = _getattr(ins, "dest", None)
                    callee_idx = None

                    if dest is None:
                        pass
                    else:
                        dop = dest.operation
                        if dop == CONST_PTR:
                            addr = dest.constant
                            # 先当本地函数地址查，查不到再当导入地址映射到名字
                            callee_idx = _func_index_get(addr)
                            if callee_idx is None:
                                name = _addr2name_get(addr)
                                if name is not None:
                                    callee_idx = _func_index_get(name)
                        elif dop == IMPORT:
                            # IMPORT 直接用当前地址上的符号名兜底
                            sym = bv.get_symbol_at(ins.address)
                            if sym is not None:
                                callee_idx = _func_index_get(sym.name)

                    if callee_idx is not None:
                        function_edges_src.append(caller_idx)
                        function_edges_dst.append(callee_idx)

        function_edges = [function_edges_src, function_edges_dst]

        # ---- ACFG/CFG：基本块特征 + 边 ----
        acfg_list = []
        # 预定义 token 分类集合，避免反复构造 set
        CALL_TOK = {"call"}
        TRANS_TOK = {"jmp", "jz", "jnz", "je", "jne"}
        ARITH_TOK = {"add", "sub", "mul", "div"}
        LOGIC_TOK = {"and", "or", "xor"}
        CMP_TOK = {"cmp"}
        MOVE_TOK = {"mov"}
        TERM_TOK = {"ret", "hlt"}
        DATA_TOK = {"db", "dw", "dd"}
        CONST_TOKEN_TYPES = {"IntegerToken", "StringToken"}

        for func in local_functions:
            blocks = func.basic_blocks
            bcount = len(blocks)

            cfg_data = {
                "block_number": bcount,
                "block_edges": [[], []],  # [[src...], [dst...]]
                "block_features": []      # [[11维]...]
            }

            # 用基本块起始地址做 key（Binary Ninja 的 block 对象不保证恒等）
            block_map = {b.start: i for i, b in enumerate(blocks)}

            # 少做全局查找
            be_src, be_dst = cfg_data["block_edges"]

            for b in blocks:
                # ---- 统计 block 内特征 ----
                call = trans = arith = logic = cmp_ = move = term = data_decl = tot = consts = 0

                # 注意：有些块可能没有反汇编文本，判空
                dt = b.disassembly_text
                if dt is not None:
                    for ins in dt:
                        toks = getattr(ins, "tokens", None)
                        if not toks:
                            continue
                        tot += 1  # 把“有 token 的指令数”作为 total_instructions
                        for t in toks:
                            tt = t.text
                            if tt in CALL_TOK:
                                call += 1
                            elif tt in TRANS_TOK:
                                trans += 1
                            elif tt in ARITH_TOK:
                                arith += 1
                            elif tt in LOGIC_TOK:
                                logic += 1
                            elif tt in CMP_TOK:
                                cmp_ += 1
                            elif tt in MOVE_TOK:
                                move += 1
                            elif tt in TERM_TOK:
                                term += 1
                            elif tt in DATA_TOK:
                                data_decl += 1
                            # 常量：按 Token 类型判断
                            elif getattr(t, "type", None) and t.type.name in CONST_TOKEN_TYPES:
                                consts += 1

                offspring = len(b.outgoing_edges)

                cfg_data["block_features"].append([
                    call, trans, arith, logic, cmp_, move, term,
                    data_decl, tot, consts, offspring
                ])

                # ---- 记录 CFG 边（地址匹配）----
                src_idx = block_map.get(b.start)
                if src_idx is None:
                    continue
                for e in b.outgoing_edges:
                    tgt = e.target
                    if tgt is None:
                        continue
                    dst_idx = block_map.get(tgt.start)
                    if dst_idx is not None:
                        be_src.append(src_idx)
                        be_dst.append(dst_idx)

            acfg_list.append(cfg_data)

        # ---- 汇总 ----
        results = {
            "function_edges": function_edges,
            "acfg_list": acfg_list,
            "function_names": [f.name for f in local_functions] + external_functions,
            "function_number": total_fn,
            "hash": file_hash,
        }
        return results

    except Exception as e:
        print(f"处理文件时出错：{e}")
        return None


# -*- coding: utf-8 -*-
"""
单进程 MalGraph 特征提取与缓存构建：
- 扫描样本 -> 调用已有 extract_cfg_and_fcg 逐个提取 -> 写入 JSONL（断点续跑）
- 同步累计词表计数并周期性落盘 -> 最后生成 Vocab
- 将 JSONL 转为 .pt（逐样本）
鲁棒性：
- 路径归一化，文件存在/非空检查
- 快速 MZ/PE 魔数检查（避免 Binary Ninja 白跑）
- 对失败样本写 bad_files.txt
- 可 resume：已在 JSONL 中出现过的 hash 直接跳过
注意：
- 如果 Binary Ninja native 崩溃（0xC0000005），单进程无法兜底；这是它自己的底层问题。
"""

import os
import json
import gc
import hashlib
from collections import Counter
from tqdm import tqdm

from feature_extraction.extract_feature import scan_load_samples as _scan2
from models.MalGraph_2022.utils import parse_json_list_2_pyg_object
from models.MalGraph_2022.Vocabulary import Vocab

# ========= 可按需修改的配置 =========
BASE_DIR   = r"E:\Experimental data\dr_data"              # 根目录：包含 benign_unpacked/benign 和 malicious_unpacked
OUT_DIR    = r"./malgraph_cache"                          # 输出目录
JSONL_OUT  = os.path.join(OUT_DIR, "malgraph_features.jsonl")
VOCAB_OUT  = os.path.join(OUT_DIR, "train_external_function_name_vocab.jsonl")
PT_DIR     = os.path.join(OUT_DIR, "pt_files")
RESUME     = True                                         # 断点续跑
FLUSH_EVERY = 200                                         # 词表/日志周期性落盘频率（样本数）
# ==================================


def _norm(path: str) -> str:
    return os.path.abspath(os.path.normpath(path))


def _is_probably_pe(path: str) -> bool:
    """快速检查是否 PE：MZ + e_lfanew 指向 'PE\\0\\0'"""
    try:
        with open(path, "rb") as f:
            mz = f.read(2)
            if mz != b"MZ":
                return False
            f.seek(0x3C)
            e_lfanew = int.from_bytes(f.read(4), "little", signed=False)
            if e_lfanew <= 0 or e_lfanew > 10 * 1024 * 1024:
                return False
            f.seek(e_lfanew)
            sig = f.read(4)
            return sig == b"PE\x00\x00"
    except Exception:
        return False


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_cache_once(
    base_dir: str,
    jsonl_out: str,
    vocab_out: str,
    pt_dir: str,
    resume: bool = True,
    flush_every: int = 200,
):
    os.makedirs(os.path.dirname(jsonl_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(vocab_out) or ".", exist_ok=True)
    os.makedirs(pt_dir, exist_ok=True)

    # 已处理样本集合（用 hash 去重）
    done_set = set()
    if resume and os.path.exists(jsonl_out):
        with open(jsonl_out, "r", encoding="utf-8") as fr:
            for line in fr:
                try:
                    obj = json.loads(line)
                    if "hash" in obj:
                        done_set.add(obj["hash"])
                except Exception:
                    continue

    # 词表计数（断点续跑从已有文件恢复）
    vocab_counter = Counter()
    if resume and os.path.exists(vocab_out):
        with open(vocab_out, "r", encoding="utf-8") as fr:
            for line in fr:
                try:
                    o = json.loads(line)
                    name = o.get("f_name")
                    cnt  = int(o.get("count", 0))
                    if name:
                        vocab_counter[name] += cnt
                except Exception:
                    continue

    # 样本列表
    samples = scan_load_samples(base_dir)  # [(path, label), ...]
    if not samples:
        print("未发现样本，结束。")
        return

    bad_log = os.path.join(os.path.dirname(jsonl_out) or ".", "bad_files.txt")
    done, failed, skipped = 0, 0, 0

    with open(jsonl_out, "a", encoding="utf-8") as jw, open(bad_log, "a", encoding="utf-8") as bw:
        pbar = tqdm(samples, desc="提取特征（单进程）", unit="file")
        for spath, label in pbar:
            npath = _norm(spath)

            # 存在/非空
            try:
                if not os.path.exists(npath) or os.path.getsize(npath) == 0:
                    skipped += 1
                    bw.write(f"MISS_OR_EMPTY\t{npath}\n"); bw.flush()
                    pbar.set_postfix(done=done, failed=failed, skipped=skipped)
                    continue
            except Exception as e:
                failed += 1
                bw.write(f"STAT_FAIL\t{npath}\t{repr(e)}\n"); bw.flush()
                pbar.set_postfix(done=done, failed=failed, skipped=skipped)
                continue

            # 快速 PE 筛查
            if not _is_probably_pe(npath):
                skipped += 1
                bw.write(f"NOT_PE\t{npath}\n"); bw.flush()
                pbar.set_postfix(done=done, failed=failed, skipped=skipped)
                continue

            # 计算 hash（用于去重/续跑）
            try:
                file_hash = _sha256(npath)
            except Exception as e:
                failed += 1
                bw.write(f"HASH_FAIL\t{npath}\t{repr(e)}\n"); bw.flush()
                pbar.set_postfix(done=done, failed=failed, skipped=skipped)
                continue

            if file_hash in done_set:
                skipped += 1
                pbar.set_postfix(done=done, failed=failed, skipped=skipped)
                continue

            # 这里调用你已有的提取函数（内部可能会更新 vocab_counter）
            try:
                feat = extract_cfg_and_fcg(npath, vocab_counter)
                if not feat:
                    failed += 1
                    bw.write(f"EXTRACT_NONE\t{npath}\n"); bw.flush()
                    pbar.set_postfix(done=done, failed=failed, skipped=skipped)
                    continue

                # 写入一行 JSONL（加入标签与路径）
                feat_out = dict(feat)
                feat_out["label"] = int(label)
                feat_out["path"]  = npath
                jw.write(json.dumps(feat_out, ensure_ascii=False) + "\n")
                jw.flush()

                done_set.add(file_hash)
                done += 1

            except Exception as e:
                # 提醒：Binary Ninja 如果 native 崩溃，这里捕不到；进程会直接退出。
                failed += 1
                bw.write(f"EXTRACT_ERR\t{npath}\t{repr(e)}\n"); bw.flush()
            finally:
                gc.collect()

            # 周期性落盘词表
            total = done + failed + skipped
            if total % flush_every == 0:
                with open(vocab_out, "w", encoding="utf-8") as vf:
                    for name, cnt in vocab_counter.most_common():
                        vf.write(json.dumps({"f_name": name, "count": int(cnt)}, ensure_ascii=False) + "\n")

            pbar.set_postfix(done=done, failed=failed, skipped=skipped)

    # 最终落盘词表
    with open(vocab_out, "w", encoding="utf-8") as vf:
        for name, cnt in vocab_counter.most_common():
            vf.write(json.dumps({"f_name": name, "count": int(cnt)}, ensure_ascii=False) + "\n")

    # 生成 Vocab 并转换 .pt
    vocab = Vocab(freq_file=vocab_out, max_vocab_size=10000)
    parse_json_list_2_pyg_object(jsonl_file=jsonl_out, vocab=vocab, output_dir=pt_dir)

    print(f"完成：done={done}, failed={failed}, skipped={skipped}")
    print(f"JSONL: {jsonl_out}")
    print(f"Vocab: {vocab_out}")
    print(f"PT目录: {pt_dir}")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print("扫描已缓存JSON以初始化词表: 0it [00:00, ?it/s]")
    build_cache_once(
        base_dir=BASE_DIR,
        jsonl_out=JSONL_OUT,
        vocab_out=VOCAB_OUT,
        pt_dir=PT_DIR,
        resume=RESUME,
        flush_every=FLUSH_EVERY,
    )

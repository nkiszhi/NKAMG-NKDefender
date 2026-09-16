"""
CoDefenderC3 原始 PE 时序数据集构建器
======================================
从已有的 PE 文件构建按月组织的时序数据集, 用于全部 33 个模型。

已有数据:
  恶意: F:/Experimental data/malicious_sample/{2019,2020,...,2024}/  (每年~8万)
  良性: F:/Experimental data/benign_unpacked/benign/  (~8万)

流程:
  1. 扫描所有 PE 文件, 读取 PE header TimeDateStamp → 精确到月
  2. 按月采样 (每月 N 恶意 + N 良性)
  3. 构建 metadata.csv + metadata.json (含 month_offsets, train/eval)
  4. 将采样的 PE 文件链接/复制到 RawPE_data/samples/

产出:
  CoDefenderC3_data/RawPE_data/
      samples/         ← 采样的 PE 文件 (sha256 命名)
      metadata.csv     ← sha256, label, month, appeared, source, orig_path
      metadata.json    ← feature_type: "PE_RAW", train_months, eval_months

用法:
  python build_rawpe.py                  # 构建全部
  python build_rawpe.py --scan-only      # 仅扫描, 不复制文件
  python build_rawpe.py --per-month 500  # 每月每类 500 个
"""
import os, sys, json, time, hashlib, struct, csv, shutil
import numpy as np
from collections import defaultdict

BASE_DIR     = r"F:\Experimental data"
DATA_ROOT    = os.path.join(BASE_DIR, "CoDefenderC3_data")
OUT_DIR      = os.path.join(DATA_ROOT, "RawPE_data")
SAMPLES_DIR  = os.path.join(OUT_DIR, "samples")

# 已有数据路径
MALWARE_ROOT = os.path.join(BASE_DIR, "malicious_sample")   # 子目录: 2019/, 2020/, ...
BENIGN_ROOT  = os.path.join(BASE_DIR, "benign_unpacked", "benign")

PER_MONTH_PER_CLASS = 500  # 每月每类采样数


def pe_timestamp(path):
    """读取 PE TimeDateStamp (编译时间), 返回 (year, month) 或 None"""
    try:
        with open(path, "rb") as f:
            magic = f.read(2)
            if magic != b"MZ":
                return None
            f.seek(0x3C)
            pe_offset = struct.unpack("<I", f.read(4))[0]
            if pe_offset > 0x10000:  # 不合理的偏移
                return None
            f.seek(pe_offset)
            sig = f.read(4)
            if sig != b"PE\x00\x00":
                return None
            # IMAGE_FILE_HEADER: Machine(2) + NumberOfSections(2) + TimeDateStamp(4)
            f.read(4)  # skip Machine + NumberOfSections
            ts = struct.unpack("<I", f.read(4))[0]
            # 合理范围: 2015-2026
            if ts < 1420070400 or ts > 1767225600:
                return None
            t = time.gmtime(ts)
            return (t.tm_year, t.tm_mon)
    except Exception:
        return None


def validate_pe(path):
    """验证 PE 文件结构完整性, 返回 (ok: bool, reason: str)。
    
    检查项:
      1. MZ 魔数
      2. PE 偏移量合理
      3. PE 签名
      4. Machine 类型有效
      5. 节表数量合理 (0-96)
      6. Optional Header 大小合理
      7. 文件大小 >= PE头声明的最小值
      8. 节表不越界
    """
    try:
        size = os.path.getsize(path)
        if size < 64:
            return False, "文件过小"
        if size > 500 * 1024 * 1024:
            return False, "文件过大 (>500MB)"

        with open(path, "rb") as f:
            # 1. MZ
            magic = f.read(2)
            if magic != b"MZ":
                return False, "非MZ文件"

            # 2. PE 偏移
            f.seek(0x3C)
            pe_offset = struct.unpack("<I", f.read(4))[0]
            if pe_offset < 4 or pe_offset > min(size - 4, 0x10000):
                return False, f"PE偏移越界: {pe_offset}"

            # 3. PE 签名
            f.seek(pe_offset)
            sig = f.read(4)
            if sig != b"PE\x00\x00":
                return False, "PE签名无效"

            # 4. IMAGE_FILE_HEADER
            machine = struct.unpack("<H", f.read(2))[0]
            valid_machines = {0x14c, 0x8664, 0x1c0, 0xaa64, 0x1c4}  # i386, AMD64, ARM, ARM64, ARMv7
            if machine not in valid_machines:
                return False, f"Machine类型无效: 0x{machine:x}"

            n_sections = struct.unpack("<H", f.read(2))[0]
            if n_sections > 96:
                return False, f"节表过多: {n_sections}"

            f.read(4)  # TimeDateStamp
            f.read(4)  # PointerToSymbolTable
            f.read(4)  # NumberOfSymbols
            opt_hdr_size = struct.unpack("<H", f.read(2))[0]
            if opt_hdr_size > 0x1000:
                return False, f"OptionalHeader过大: {opt_hdr_size}"

            # 5. 节表范围检查
            section_table_offset = pe_offset + 24 + opt_hdr_size
            section_table_end = section_table_offset + n_sections * 40
            if section_table_end > size:
                return False, "节表越界"

            # 6. 读取节表, 检查基本合理性
            f.seek(section_table_offset)
            for si in range(min(n_sections, 10)):  # 检查前10个节
                sec_data = f.read(40)
                if len(sec_data) < 40:
                    return False, f"节表截断 (节 {si})"
                raw_offset = struct.unpack("<I", sec_data[20:24])[0]
                raw_size = struct.unpack("<I", sec_data[16:20])[0]
                # 节的原始数据不应完全越界
                if raw_offset > size and raw_size > 0:
                    return False, f"节 {si} 偏移越界"

        return True, "ok"

    except Exception as e:
        return False, str(e)


def scan_directory(root, label, source_name):
    """扫描目录下所有 PE 文件, 返回记录列表。
    
    恶意: 文件夹年份为采集年份, 年内均匀分到 12 个月 (TimeDateStamp 不可信)
    良性: PE header TimeDateStamp 为编译时间 (正规软件不伪造)
    """
    records = []
    if not os.path.exists(root):
        print(f"  [跳过] {root} 不存在")
        return records

    if label == 1:
        # ── 恶意: 按年份子目录, 年内均匀分月 ──
        for year_dir in sorted(os.listdir(root)):
            year_path = os.path.join(root, year_dir)
            if not os.path.isdir(year_path):
                continue
            try:
                year = int(year_dir)
            except ValueError:
                continue

            files = [os.path.join(year_path, f) for f in os.listdir(year_path)
                     if os.path.isfile(os.path.join(year_path, f))]
            print(f"    {year}: {len(files)} 文件", end="")

            # PE 结构验证
            pe_files = []
            for fp in files:
                ok, _ = validate_pe(fp)
                if ok:
                    pe_files.append(fp)
            print(f" → {len(pe_files)} 有效PE", end="")

            # 随机打乱后均匀分到 12 个月
            rng = np.random.RandomState(year)
            rng.shuffle(pe_files)
            for i, fp in enumerate(pe_files):
                month = (i % 12) + 1
                appeared = f"{year}-{month:02d}"
                records.append({
                    "filename": os.path.basename(fp),
                    "label": 1,
                    "appeared": appeared,
                    "year": year,
                    "month": month,
                    "source": source_name,
                    "orig_path": fp,
                })
            print(f" → 分配到 {year}-01 ~ {year}-12 ({len(pe_files)} PE)")

    else:
        # ── 良性: TimeDateStamp (编译时间) + PE 结构验证 ──
        all_files = [os.path.join(root, f) for f in os.listdir(root)
                     if os.path.isfile(os.path.join(root, f))]
        print(f"  扫描 {source_name}: {len(all_files)} 文件...")
        t0 = time.time()
        n_valid = 0
        n_rejected = 0
        for i, fp in enumerate(all_files):
            # PE 结构验证
            ok, reason = validate_pe(fp)
            if not ok:
                n_rejected += 1
                continue
            ym = pe_timestamp(fp)
            if ym is not None:
                year, month = ym
                appeared = f"{year}-{month:02d}"
                records.append({
                    "filename": os.path.basename(fp),
                    "label": 0,
                    "appeared": appeared,
                    "year": year,
                    "month": month,
                    "source": source_name,
                    "orig_path": fp,
                })
                n_valid += 1

            if (i + 1) % 10000 == 0:
                elapsed = time.time() - t0
                speed = (i + 1) / max(elapsed, 0.01)
                print(f"\r    {i+1:,}/{len(all_files):,} ({n_valid:,} 有效, "
                      f"{n_rejected:,} 拒绝) {speed:.0f}/s", end="", flush=True)

        print(f"\n    完成: {n_valid:,} 有效, {n_rejected:,} 结构损坏, "
              f"共 {len(all_files):,} ({time.time()-t0:.0f}s)")

    print(f"  {source_name} 总计: {len(records):,} 条记录")
    return records


def sample_by_month(records, per_month, label_name):
    """按月均匀采样"""
    by_month = defaultdict(list)
    for r in records:
        by_month[r["appeared"]].append(r)

    sampled = []
    rng = np.random.RandomState(42)
    for month_key in sorted(by_month.keys()):
        pool = by_month[month_key]
        n = min(len(pool), per_month)
        if n > 0:
            indices = rng.choice(len(pool), n, replace=False)
            for idx in indices:
                sampled.append(pool[idx])

    print(f"  {label_name} 采样: {len(sampled):,} (from {len(records):,})")

    # 月份分布
    dist = defaultdict(int)
    for r in sampled:
        dist[r["appeared"]] += 1
    months = sorted(dist.keys())
    if months:
        print(f"    范围: {months[0]} → {months[-1]} ({len(months)} 月)")
        print(f"    每月: min={min(dist.values())}, max={max(dist.values())}, "
              f"avg={sum(dist.values())/len(dist):.0f}")
    return sampled


def copy_samples(records):
    """验证 + 复制采样的 PE 文件到 samples/ 目录。
    
    每个文件先通过 validate_pe 验证结构完整性, 不合格直接跳过。
    """
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    ok = 0
    rejected = []
    for i, r in enumerate(records):
        src = r["orig_path"]

        # PE 结构验证
        valid, reason = validate_pe(src)
        if not valid:
            rejected.append((os.path.basename(src), reason))
            continue

        # 计算 SHA256 作为文件名
        try:
            with open(src, "rb") as f:
                sha = hashlib.sha256(f.read()).hexdigest()
        except Exception:
            rejected.append((os.path.basename(src), "读取失败"))
            continue
        r["sha256"] = sha
        dest = os.path.join(SAMPLES_DIR, sha)
        if os.path.exists(dest):
            ok += 1
            continue
        try:
            # 优先硬链接 (同驱动器, 零磁盘开销)
            os.link(src, dest)
            ok += 1
        except OSError:
            try:
                shutil.copy2(src, dest)
                ok += 1
            except Exception:
                rejected.append((os.path.basename(src), "复制失败"))

        if (i + 1) % 5000 == 0:
            print(f"\r    复制: {ok:,}/{i+1:,} (拒绝 {len(rejected)})", end="", flush=True)

    print(f"\n    完成: {ok:,} 有效, {len(rejected)} 拒绝 (共 {len(records):,})")
    if rejected:
        print(f"    拒绝原因统计:")
        from collections import Counter
        reason_counts = Counter(r for _, r in rejected)
        for reason, cnt in reason_counts.most_common(10):
            print(f"      {reason}: {cnt}")
        # 保存完整拒绝列表
        reject_path = os.path.join(OUT_DIR, "rejected_samples.txt")
        with open(reject_path, "w", encoding="utf-8") as f:
            for name, reason in rejected:
                f.write(f"{name}\t{reason}\n")
        print(f"    完整列表: {reject_path}")

    return ok


def generate_metadata(all_records):
    """生成 metadata.csv + metadata.json"""
    # 过滤无 sha256 的记录
    all_records = [r for r in all_records if "sha256" in r]

    # 按 appeared 排序
    all_records.sort(key=lambda r: r["appeared"])

    # 分配月份编号
    unique_months = sorted(set(r["appeared"] for r in all_records))
    month_map = {m: i + 1 for i, m in enumerate(unique_months)}
    for r in all_records:
        r["month_id"] = month_map[r["appeared"]]

    # 写 CSV (列名与 PELoader.discover_samples 兼容)
    csv_path = os.path.join(OUT_DIR, "metadata.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "sha256", "path", "label", "month", "appeared", "source"])
        w.writeheader()
        for r in all_records:
            w.writerow({
                "sha256": r["sha256"],
                "path": os.path.join("samples", r["sha256"]),
                "label": r["label"],
                "month": r["month_id"],
                "appeared": r["appeared"],
                "source": r["source"],
            })

    # train/eval: 前 2/3 训练, 后 1/3 评估
    n_months = len(unique_months)
    n_train = max(2, n_months * 2 // 3)
    train_months = list(range(1, n_train + 1))
    eval_months = list(range(n_train + 1, n_months + 1))

    # 月份统计
    month_counts = defaultdict(lambda: {"malware": 0, "benign": 0})
    for r in all_records:
        key = "malware" if r["label"] == 1 else "benign"
        month_counts[r["month_id"]][key] += 1

    n_mal = sum(1 for r in all_records if r["label"] == 1)
    n_ben = sum(1 for r in all_records if r["label"] == 0)

    meta = {
        "name": "RawPE-TimeSeries",
        "source": "Local PE collection (malicious_sample + benign_unpacked)",
        "feature_type": "PE_RAW",
        "n_samples": len(all_records),
        "n_malware": n_mal,
        "n_benign": n_ben,
        "malware_ratio": round(n_mal / max(len(all_records), 1), 4),
        "n_months": n_months,
        "train_months": train_months,
        "eval_months": eval_months,
        "month_labels": {str(v): k for k, v in month_map.items()},
        "month_counts": {str(k): dict(v) for k, v in sorted(month_counts.items())},
    }
    with open(os.path.join(OUT_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n  ═══ 数据集统计 ═══")
    print(f"  总样本: {len(all_records):,} (恶意={n_mal:,}, 良性={n_ben:,})")
    print(f"  月份: {n_months} ({unique_months[0]} → {unique_months[-1]})")
    print(f"  训练: M1-{n_train} ({unique_months[0]} → {unique_months[n_train-1]})")
    print(f"  评估: M{n_train+1}-{n_months} ({unique_months[n_train]} → {unique_months[-1]})")
    print(f"\n  月份分布:")
    for m_id in sorted(month_counts.keys()):
        mc = month_counts[m_id]
        label = unique_months[m_id - 1] if m_id <= len(unique_months) else "?"
        print(f"    M{m_id:>3d} ({label}): mal={mc['malware']:>4d} ben={mc['benign']:>4d}")
    return meta


def binja_prevalidate(records, n_workers=8, batch_size=50):
    """BinaryNinja 预加载验证 + bndb 预生成。
    
    阶段1 的 bndb 生成 = 验证 (能生成 bndb 的 PE 就是有效的)。
    生成的 bndb 同时作为 run_all.py 特征提取的缓存 (不重复分析)。
    
    断点续传: bndb 文件存在即跳过。
    """
    import subprocess as sp
    from concurrent.futures import ThreadPoolExecutor
    import threading

    to_check = [(i, r) for i, r in enumerate(records) if "sha256" in r]
    if not to_check:
        return records

    # bndb 缓存目录
    bndb_dir = os.path.join(OUT_DIR, ".cache", "bndb")
    os.makedirs(bndb_dir, exist_ok=True)

    # 已有 bndb 的跳过
    def _bndb_path(sha):
        """bndb 路径必须与 data_loader._bndb_cache_path 一致 (相同 md5 key)。"""
        pe_path = os.path.join(SAMPLES_DIR, sha)
        return os.path.join(bndb_dir, hashlib.md5(pe_path.encode()).hexdigest() + ".bndb")

    remaining = []
    passed = set()
    for idx, r in to_check:
        bp = _bndb_path(r["sha256"])
        if os.path.exists(bp) and os.path.getsize(bp) > 0:
            passed.add(idx)
        else:
            remaining.append((idx, r))

    total = len(to_check)
    n_done = len(passed)
    if not remaining:
        print(f"  全部 {total} 个样本的 bndb 已存在, 跳过验证")
        clean = [r for i, r in enumerate(records) if i in passed]
        print(f"  最终数据集: {len(clean)} 样本")
        return clean

    print(f"  验证+生成bndb: {len(remaining)} 个 (已有 {n_done}, "
          f"共 {total}, {n_workers} 并行, 每批 {batch_size})...")
    t0 = time.time()

    # 写 worker 脚本
    script_path = os.path.join(OUT_DIR, "_bndb_gen.py")
    with open(script_path, "w") as f:
        f.write('''import sys, gc
try:
    import binaryninja as bn
    try: bn.log.log_to_stderr(bn.LogLevel.ErrorLog)
    except: pass
    try: bn.disable_default_log()
    except: pass
except ImportError:
    for i in range(0, len(sys.argv[1:]), 2):
        print(f"FAIL\\t{sys.argv[1+i]}\\tno_binaryninja", flush=True)
    sys.exit(0)

args = sys.argv[1:]
for i in range(0, len(args), 2):
    pe_path, bndb_path = args[i], args[i+1]
    try:
        bv = bn.load(pe_path, options={
            "analysis.mode": "basic",
            "analysis.linearSweep.autorun": True,
            "analysis.limits.maxFunctionSize": 262144,
        })
        if bv is not None:
            bv.update_analysis_and_wait()
            bv.create_database(bndb_path)
            bv.file.close()
            del bv
            print(f"OK\\t{pe_path}", flush=True)
        else:
            print(f"FAIL\\t{pe_path}\\tload_None", flush=True)
    except Exception as e:
        print(f"FAIL\\t{pe_path}\\t{str(e)[:80]}", flush=True)
    gc.collect()
''')

    lock = threading.Lock()
    failed = []
    progress = [n_done]

    def _run_batch(batch_items):
        args = []
        pe_set = set()
        for idx, r in batch_items:
            pe_path = os.path.join(SAMPLES_DIR, r["sha256"])
            bndb = _bndb_path(r["sha256"])
            args.extend([pe_path, bndb])
            pe_set.add(pe_path)

        idx_map = {os.path.join(SAMPLES_DIR, r["sha256"]): idx for idx, r in batch_items}

        try:
            proc = sp.run([sys.executable, script_path] + args,
                          capture_output=True, timeout=batch_size * 180)
            stdout = (proc.stdout or b"").decode("utf-8", errors="replace")
        except Exception:
            stdout = ""

        reported = set()
        for line in stdout.strip().split("\n"):
            parts = line.split("\t", 2)
            if len(parts) >= 2 and parts[1] in idx_map:
                reported.add(parts[1])
                idx = idx_map[parts[1]]
                with lock:
                    progress[0] += 1
                    if parts[0] == "OK":
                        passed.add(idx)
                    else:
                        failed.append(parts[1])

        for pe_path in pe_set:
            if pe_path not in reported and pe_path in idx_map:
                with lock:
                    progress[0] += 1
                    failed.append(os.path.basename(pe_path)[:16])

        with lock:
            done = progress[0]
            if done % 200 == 0 or done >= total:
                elapsed = time.time() - t0
                rate = max(done - n_done, 1) / max(elapsed, 0.01)
                eta = (total - done) / max(rate, 0.01)
                print(f"\r  [{done}/{total}] {done/total*100:.1f}% "
                      f"({rate:.1f}/s, ETA {eta:.0f}s, "
                      f"pass={len(passed)}, fail={len(failed)})",
                      end="", flush=True)

    batches = [remaining[i:i+batch_size] for i in range(0, len(remaining), batch_size)]
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = [pool.submit(_run_batch, b) for b in batches]
        for f in futs:
            f.result()

    elapsed = time.time() - t0
    print(f"\n  完成: {len(passed)} 通过, {len(failed)} 拒绝 ({elapsed:.0f}s)")

    try: os.remove(script_path)
    except: pass

    if failed:
        reject_path = os.path.join(OUT_DIR, "binja_rejected.txt")
        with open(reject_path, "w") as f:
            for name in failed: f.write(name + "\n")
        print(f"  拒绝列表: {reject_path}")

    clean = [r for i, r in enumerate(records) if i in passed]
    print(f"  最终数据集: {len(clean)} 样本")
    return clean


def _load_records_from_csv():
    """从已有的 metadata.csv 恢复记录列表 (断点续传用)。"""
    csv_path = os.path.join(OUT_DIR, "metadata.csv")
    if not os.path.exists(csv_path):
        return None
    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                "sha256": row["sha256"],
                "label": int(row["label"]),
                "appeared": row["appeared"],
                "month": 0,
                "source": row.get("source", ""),
                "orig_path": "",
            })
    return records


def _rebuild_records_from_samples():
    """从 samples/ 目录 + 重新扫描源目录重建完整记录列表。
    
    当 metadata.csv 被之前的错误运行损坏时使用。
    重新扫描+采样 (相同随机种子 → 相同结果), 只保留 samples/ 中已存在的文件。
    """
    print(f"    重新扫描源目录...")
    mal_records = scan_directory(MALWARE_ROOT, label=1, source_name="malicious_sample")
    ben_records = scan_directory(BENIGN_ROOT, label=0, source_name="benign")

    sampled_mal = sample_by_month(mal_records, PER_MONTH_PER_CLASS, "恶意")
    sampled_ben = sample_by_month(ben_records, PER_MONTH_PER_CLASS, "良性")
    all_sampled = sampled_mal + sampled_ben

    # 计算 sha256, 只保留 samples/ 中已存在的
    existing_shas = set(os.listdir(SAMPLES_DIR)) if os.path.isdir(SAMPLES_DIR) else set()
    matched = []
    for r in all_sampled:
        src = r["orig_path"]
        try:
            with open(src, "rb") as f:
                sha = hashlib.sha256(f.read()).hexdigest()
        except Exception:
            continue
        if sha in existing_shas:
            r["sha256"] = sha
            matched.append(r)

    print(f"    匹配: {len(matched)}/{len(all_sampled)} (samples/ 中 {len(existing_shas)} 文件)")
    return matched if matched else None


def main():
    print("=" * 60)
    print("  CoDefenderC3 原始 PE 时序数据集构建器")
    print(f"  恶意来源: {MALWARE_ROOT}")
    print(f"  良性来源: {BENIGN_ROOT}")
    print(f"  输出: {OUT_DIR}")
    print("=" * 60)

    args = set(sys.argv[1:])
    per_month = PER_MONTH_PER_CLASS
    for a in sys.argv[1:]:
        if a.startswith("--per-month"):
            per_month = int(sys.argv[sys.argv.index(a) + 1])
    scan_only = "--scan-only" in args
    force_rebuild = "--force" in args

    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 断点续传: 如果 Step 1-4 已完成, 跳到 Step 5 ──
    csv_path = os.path.join(OUT_DIR, "metadata.csv")
    n_samples_on_disk = len(os.listdir(SAMPLES_DIR)) if os.path.isdir(SAMPLES_DIR) else 0
    csv_exists = os.path.exists(csv_path)

    print(f"\n  断点检查: metadata.csv={'有' if csv_exists else '无'}, "
          f"samples/={n_samples_on_disk} 文件")

    skip_step1to4 = False
    all_sampled = None

    if n_samples_on_disk > 100 and not force_rebuild:
        if csv_exists:
            all_sampled = _load_records_from_csv()
            if all_sampled and len(all_sampled) > n_samples_on_disk * 0.8:
                skip_step1to4 = True
                print(f"  ✓ metadata.csv 有效 ({len(all_sampled)} 条), 跳过 Step 1-4")
            else:
                # metadata.csv 被损坏 (之前的错误运行覆盖), 从 samples/ 重建
                n_csv = len(all_sampled) if all_sampled else 0
                print(f"  ⚠ metadata.csv ({n_csv} 条) 与 samples/ ({n_samples_on_disk} 文件) 不匹配")
                print(f"    从 samples/ 目录重建 metadata.csv...")
                all_sampled = _rebuild_records_from_samples()
                if all_sampled:
                    generate_metadata(all_sampled)
                    skip_step1to4 = True
                    print(f"    重建完成: {len(all_sampled)} 条记录")
        else:
            # 有 samples/ 但没 csv → 重建
            print(f"  ⚠ samples/ 存在但无 metadata.csv, 重建...")
            all_sampled = _rebuild_records_from_samples()
            if all_sampled:
                generate_metadata(all_sampled)
                skip_step1to4 = True
                print(f"    重建完成: {len(all_sampled)} 条记录")

    if skip_step1to4:
        print(f"    (如需全部重建, 请加 --force 参数)")
    else:
        # Step 1: 扫描恶意样本
        print(f"\n  ── Step 1: 扫描恶意样本 ──")
        mal_records = scan_directory(MALWARE_ROOT, label=1, source_name="malicious_sample")

        # Step 2: 扫描良性样本
        print(f"\n  ── Step 2: 扫描良性样本 ──")
        ben_records = scan_directory(BENIGN_ROOT, label=0, source_name="benign")

        # Step 3: 按月采样
        print(f"\n  ── Step 3: 按月采样 (每月每类 {per_month}) ──")
        sampled_mal = sample_by_month(mal_records, per_month, "恶意")
        sampled_ben = sample_by_month(ben_records, per_month, "良性")
        all_sampled = sampled_mal + sampled_ben

        if scan_only:
            print(f"\n  --scan-only: 跳过文件复制")
            return

        # Step 4: 复制文件 (含 PE 结构验证)
        print(f"\n  ── Step 4: 复制 PE 文件到 {SAMPLES_DIR} (含结构验证) ──")
        copy_samples(all_sampled)

        # 先写一份 metadata.csv (Step 5 之前的 checkpoint)
        generate_metadata(all_sampled)
        print(f"  Step 1-4 checkpoint 已保存")

    # Step 5: BinaryNinja 预加载验证 (自带断点续传: binja_validated.txt)
    print(f"\n  ── Step 5: BinaryNinja 预加载验证 ──")
    all_sampled = binja_prevalidate(all_sampled, n_workers=8, batch_size=50)

    # Step 6: 重新生成 metadata (只含通过验证的样本)
    print(f"\n  ── Step 6: 生成 metadata (仅验证通过样本) ──")
    generate_metadata(all_sampled)

    print("\n" + "=" * 60)
    print(f"  ✓ 完成")
    print(f"  下一步: python run_all.py  (BinaryNinja 提取特征 → 33 模型)")
    print("=" * 60)


if __name__ == "__main__":
    main()

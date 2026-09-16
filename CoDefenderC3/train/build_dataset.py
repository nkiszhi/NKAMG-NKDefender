"""
CoDefenderC3 真实数据集自动下载与构建
======================================
全部使用真实恶意软件特征数据集，不使用合成数据。

数据集:
  1a. EMBER-2017  — 1,100,000 PE, 2381d feature v2 (Elastic/Endgame)
      来源: https://ember.elastic.co/ember_dataset_2017_2.tar.bz2
  1b. EMBER-2018  — 1,000,000 PE, 2381d feature v2 (Anderson & Roth 2018)
      来源: https://ember.elastic.co/ember_dataset_2018_2.tar.bz2
  1c. EMBER-2024  — 3,200,000 多平台, 2568d feature v3 (Joyce et al. 2025)
      来源: huggingface.co/FutureComputing4AI/EMBER2024

  2. BODMAS      — 57,293 PE, 2381d (Blue Hexagon, Yang et al. 2021)
     来源: https://whyisyoung.github.io/BODMAS/

  3. SOREL-20M   — 20,000,000 PE, 2381d 子集 (Sophos/ReversingLabs)
     来源: s3://sorel-20m (公开桶)

  4. CIC-MalMem-2022 — 58,596 内存样本 (Carrier et al. 2022)
     来源: huggingface.co/datasets/bvk/CIC-MalMem-2022

前置依赖:
  pip install requests pandas boto3 gdown
  (lief/sklearn 不再需要 — 使用纯 Numpy 向量化器)

用法:
  python build_dataset.py                # 下载+构建全部
  python build_dataset.py ember2017      # 仅 EMBER-2017
  python build_dataset.py ember2018      # 仅 EMBER-2018
  python build_dataset.py ember2024      # 仅 EMBER-2024
  python build_dataset.py bodmas         # 仅 BODMAS
"""
import os, sys, json, time, shutil, hashlib, glob, struct, gc
import subprocess
import numpy as np

MAX_ERROR_RATE = 0.10  # 向量化异常率超过 10% 则终止 (防止产出全零数据)


def _write_npy_header(f, shape, dtype=np.float32):
    """写标准 npy v1.0 header, 返回 data offset"""
    header = {"descr": np.dtype(dtype).str, "fortran_order": False, "shape": tuple(shape)}
    header_bytes = repr(header).encode("latin1")
    pad_len = 64 - (10 + len(header_bytes)) % 64
    header_bytes += b" " * (pad_len - 1) + b"\n"
    f.write(b"\x93NUMPY\x01\x00")
    f.write(struct.pack("<H", len(header_bytes)))
    f.write(header_bytes)
    return f.tell()



BASE_DIR   = r"F:\Experimental data"
RAW_DIR    = os.path.join(BASE_DIR, "_downloads")
DATA_ROOT  = os.path.join(BASE_DIR, "CoDefenderC3_data")  # 所有数据集统一根目录

def _load_json(path):
    """加载 JSON 文件 (确保关闭文件句柄)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)

DATASETS = {
    "ember2017": os.path.join(DATA_ROOT, "EMBER2017_data"),
    "ember2018": os.path.join(DATA_ROOT, "EMBER2018_data"),
    "ember2024": os.path.join(DATA_ROOT, "EMBER2024_data"),
    "bodmas":    os.path.join(DATA_ROOT, "BODMAS_data"),
    "sorel":     os.path.join(DATA_ROOT, "SOREL_data"),
    "malmem":    os.path.join(DATA_ROOT, "MalMem_data"),
    "rawpe":     os.path.join(DATA_ROOT, "RawPE_data"),
}

EMBER_RANGES = {
    "ByteHistogram": (0, 256),   "ByteEntropy": (256, 512),
    "StringInfo": (512, 616),    "GeneralInfo": (616, 626),
    "HeaderInfo": (626, 688),    "SectionInfo": (688, 943),
    "ImportInfo": (943, 2223),   "ExportInfo": (2223, 2351),
    "DataDirs": (2351, 2381),
}


def _filter_dat_to_npy(xp_out, dat_paths, valid_masks, n_totals, ndim):
    """
    共享工具: 从 .dat 文件顺序读+过滤 → 写 X.npy (零 seek, Windows 安全)
    
    参数:
        xp_out: 输出 X.npy 路径
        dat_paths: [x_train.dat, x_test.dat] 路径列表
        valid_masks: [train_bool_mask, test_bool_mask] 布尔掩码列表
        n_totals: [n_tr_total, n_te_total] 每个 dat 的总行数
        ndim: 特征维度
    返回: 总写入行数
    """
    N = sum(int(m.sum()) for m in valid_masks)
    READ_CHUNK = 10000
    row_bytes = ndim * 4
    total_written = 0

    with open(xp_out, "wb") as xf:
        _write_npy_header(xf, (N, ndim))

        for dat_path, valid_mask, n_total in zip(dat_paths, valid_masks, n_totals):
            t0 = time.time()
            written = 0
            with open(dat_path, "rb") as f:
                for start in range(0, n_total, READ_CHUNK):
                    end = min(start + READ_CHUNK, n_total)
                    n_read = end - start
                    raw = f.read(n_read * row_bytes)
                    if len(raw) < n_read * row_bytes:
                        n_read = len(raw) // row_bytes
                    chunk_mask = valid_mask[start:start+n_read]
                    n_valid = int(chunk_mask.sum())
                    if n_valid > 0:
                        block = np.frombuffer(raw, dtype=np.float32).reshape(n_read, ndim)
                        xf.write(block[chunk_mask].tobytes())
                        written += n_valid
                        del block
                    del raw
            split_name = "train" if "train" in dat_path else "test"
            print(f"    {split_name}: {written:,} ✓ ({time.time()-t0:.0f}s)")
            total_written += written

    return total_written


# ═══════════════════════════════════════
#  下载工具
# ═══════════════════════════════════════

def _download(url, dest, sha256=None, desc=None):
    """下载文件到 dest, 支持 HTTP Range 断点续传 + SHA256 校验"""
    # 已下载完成: 校验后跳过
    if os.path.exists(dest):
        if sha256:
            print(f"    校验 {os.path.basename(dest)}...", end=" ")
            h = hashlib.sha256()
            with open(dest, "rb") as f:
                for chunk in iter(lambda: f.read(1024*1024), b""):
                    h.update(chunk)
            if h.hexdigest() == sha256:
                print("SHA256 ✓"); return True
            else:
                print(f"不匹配, 重新下载")
                os.remove(dest)
        else:
            print(f"    [已缓存] {os.path.basename(dest)} ({os.path.getsize(dest)/1024**2:.0f} MB)")
            return True

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".tmp"

    # 断点续传: 检查 .tmp 文件已下载的字节数
    resume_bytes = 0
    if os.path.exists(tmp):
        resume_bytes = os.path.getsize(tmp)

    print(f"    下载: {desc or url}")
    print(f"    → {dest}"
          + (f" (续传 {resume_bytes/1024**2:.0f} MB)" if resume_bytes else ""))

    try:
        import requests
        headers = {"User-Agent": "CoDefenderC3-DatasetBuilder/1.0"}
        if resume_bytes > 0:
            headers["Range"] = f"bytes={resume_bytes}-"

        r = requests.get(url, stream=True, timeout=(15, 600), headers=headers)

        # 服务器支持 Range → 206 Partial Content
        # 不支持 Range → 200 OK (从头下载)
        if r.status_code == 416:
            # Range 超出文件大小, .tmp 可能已完整
            print(f"    .tmp 已完整, 重命名")
            os.rename(tmp, dest)
            return True
        r.raise_for_status()

        if r.status_code == 206:
            # 断点续传
            total = resume_bytes + int(r.headers.get("Content-Length", 0))
            dl = resume_bytes
            mode = "ab"  # 追加
        else:
            # 从头下载
            total = int(r.headers.get("Content-Length", 0))
            dl = 0
            resume_bytes = 0
            mode = "wb"  # 覆盖

        t0 = time.time()
        with open(tmp, mode) as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                f.write(chunk); dl += len(chunk)
                if total:
                    speed = (dl - resume_bytes) / max(time.time()-t0, 0.01) / 1024**2
                    print(f"\r    {dl/total*100:5.1f}%  {dl/1024**2:.0f}/{total/1024**2:.0f} MB  "
                          f"{speed:.1f} MB/s", end="", flush=True)
        print()
    except ImportError:
        # requests 不可用, 用 urllib (不支持断点续传)
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "CoDefenderC3/1.0"})
        resp = urllib.request.urlopen(req, timeout=30)
        total = int(resp.headers.get("Content-Length", 0))
        dl = 0; t0 = time.time()
        resume_bytes = 0  # urllib 不支持 Range
        with open(tmp, "wb") as f:
            while True:
                chunk = resp.read(1024*1024)
                if not chunk: break
                f.write(chunk); dl += len(chunk)
                if total:
                    print(f"\r    {dl/1024**2:.0f}/{total/1024**2:.0f} MB", end="", flush=True)
        print()
    except Exception as e:
        print(f"\n    ✗ 下载失败: {e}")
        # 保留 .tmp 文件, 下次续传
        return False

    if os.path.exists(tmp):
        os.rename(tmp, dest)
        print(f"    ✓ {os.path.getsize(dest)/1024**2:.1f} MB")
        return True
    return False


def _save(out, X, y, months, meta, families=None):
    """保存标准格式数据集 (按月排序 + 验证)"""
    from dataset_utils import finalize_dataset
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "X.npy"),
            X if X.dtype == np.float32 else X.astype(np.float32))
    np.save(os.path.join(out, "y.npy"),
            y if y.dtype == np.int32 else y.astype(np.int32))
    np.save(os.path.join(out, "months.npy"),
            months if months.dtype == np.int32 else months.astype(np.int32))
    if families is not None:
        np.save(os.path.join(out, "families.npy"), families.astype(np.int32))
    with open(os.path.join(out, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    # 终态化: 按月排序 + 计算偏移量 + 验证
    finalize_dataset(out)
    mb = X.nbytes / 1024**2
    print(f"    ✓ 保存 {out}/ (X: {X.shape}, {mb:.0f} MB)")


# ═══════════════════════════════════════
#  1. EMBER 2017/2018 (feature v2, 2381d)
# ═══════════════════════════════════════

# EMBER 版本配置
EMBER_VERSIONS = {
    "ember2017": {
        "url": "https://ember.elastic.co/ember_dataset_2017_2.tar.bz2",
        "sha256": None,  # 官方未提供 2017 SHA
        "raw_subdir": "ember2017",
        "name": "EMBER-2017",
        "source": "Real (Anderson & Roth 2018, v2 features)",
        "n_expected": 1100000,
        "ndim": 2381,
        "feature_version": 2,
    },
    "ember2018": {
        "url": "https://ember.elastic.co/ember_dataset_2018_2.tar.bz2",
        "sha256": "b6052eb8d350a49a8d5a5396fbe7d16cf42848b86ff969b77464434cf2997812",
        "raw_subdir": "ember2018",
        "name": "EMBER-2018",
        "source": "Real (Anderson & Roth 2018)",
        "n_expected": 1000000,
        "ndim": 2381,
        "feature_version": 2,
    },
}


def _build_ember_v2(dataset_key):
    """
    下载并处理 EMBER v2 数据集 (2017 或 2018)

    步骤:
      1. 下载 tar.bz2 (1.05 GB)
      2. 解压得到 train_features_{0-5}.jsonl + test_features.jsonl
      3. 用 ember 包向量化为 2381d 特征 (X_train.dat / X_test.dat)
      4. 读取 .dat → 过滤未标注 → 按 appeared 字段分月 → 保存 X.npy/y.npy
    """
    cfg = EMBER_VERSIONS[dataset_key]
    year = dataset_key.replace("ember", "")
    print(f"\n  ╔═══════════════════════════════════════════════════╗")
    print(f"  ║  {cfg['name']} ({cfg['n_expected']//1000}K PE, {cfg['ndim']}d){'':>15}║")
    print(f"  ║  来源: ember.elastic.co{'':>28}║")
    print(f"  ╚═══════════════════════════════════════════════════╝")
    out = DATASETS[dataset_key]
    if os.path.exists(os.path.join(out, "X.npy")):
        m = _load_json(os.path.join(out, "metadata.json"))
        print(f"    [已构建] N={m['n_samples']:,}, D={m['n_features']}")
        return

    raw = os.path.join(RAW_DIR, cfg["raw_subdir"])
    os.makedirs(raw, exist_ok=True)

    # Step 1: 下载
    tar_name = os.path.basename(cfg["url"])
    tar_path = os.path.join(RAW_DIR, tar_name)
    _download(cfg["url"], tar_path, sha256=cfg["sha256"],
              desc=f"{cfg['name']} dataset")

    # Step 2: 解压
    dat_path = os.path.join(raw, "X_train.dat")
    jsonl_marker = os.path.join(raw, "train_features_0.jsonl")
    if not os.path.exists(dat_path) and not os.path.exists(jsonl_marker):
        print("    解压 tar.bz2 (Python bz2 较慢, 预计 10-20 分钟)...")
        print("    (如需加速可用 7-Zip 手动解压后重新运行)")
        import tarfile
        with tarfile.open(tar_path, "r:bz2") as tar:
            tar.extractall(raw)
        # 解压后可能在子目录中
        for d in os.listdir(raw):
            sub = os.path.join(raw, d)
            if os.path.isdir(sub) and os.path.exists(os.path.join(sub, "train_features_0.jsonl")):
                for f in os.listdir(sub):
                    shutil.move(os.path.join(sub, f), os.path.join(raw, f))
                shutil.rmtree(sub, ignore_errors=True)
                break
        print("    ✓ 解压完成")

    # Step 3: 向量化 (jsonl → dat)
    # 使用纯 Numpy 特征提取器, 零 sklearn/lief/ember 依赖
    required_dats = [os.path.join(raw, f) for f in
                     ["X_train.dat", "y_train.dat", "X_test.dat", "y_test.dat"]]
    if not all(os.path.exists(f) for f in required_dats):
        # 清理不完整的残留
        for f in required_dats:
            if os.path.exists(f): os.remove(f)
        for s in ["train", "test"]:
            for suffix in ["_progress_", "appeared_"]:
                p = os.path.join(raw, f"{suffix}{s}.txt")
                if os.path.exists(p): os.remove(p)

        print("    向量化特征 (纯 Numpy, 无 sklearn/lief 依赖)...")
        from ember_vectorizer import PureNumpyFeatureExtractor
        extractor = PureNumpyFeatureExtractor()
        ndim = extractor.dim

        for split in ["train", "test"]:
            jsonl_files = sorted(glob.glob(os.path.join(raw, f"{split}_features_*.jsonl")))
            if not jsonl_files:
                jsonl_files = [os.path.join(raw, f"{split}_features.jsonl")]
            jsonl_files = [f for f in jsonl_files if os.path.exists(f)]
            if not jsonl_files:
                print(f"    ✗ 未找到 {split}_features*.jsonl"); continue

            print(f"    读取 {split} JSONL...")
            n_samples = 0
            for jf in jsonl_files:
                with open(jf, encoding="utf-8") as _cf:
                    n_samples += sum(1 for _ in _cf)

            xp = os.path.join(raw, f"X_{split}.dat")
            yp = os.path.join(raw, f"y_{split}.dat")
            ap = os.path.join(raw, f"appeared_{split}.txt")
            print(f"    {split}: {n_samples:,} 样本")

            t0 = time.time()
            idx = 0; n_errors = 0
            with open(xp, "wb") as xf, open(yp, "wb") as yf, \
                 open(ap, "w", encoding="utf-8") as af:
                for jf in jsonl_files:
                    with open(jf, encoding="utf-8") as fh:
                        for line in fh:
                            label = -1; appeared = "1970-01"
                            try:
                                sample = json.loads(line.strip())
                                label = sample.get("label", -1)
                                appeared = sample.get("appeared", "1970-01")
                                feat = extractor.process_raw_features(sample)
                            except Exception as _e:
                                feat = np.zeros(ndim, dtype=np.float32)
                                n_errors += 1
                                if n_errors <= 3:
                                    print(f"\n    [异常] idx={idx}: {_e}")
                            xf.write(feat.astype(np.float32).tobytes())
                            yf.write(struct.pack("<f", float(label)))
                            af.write(f"{label}\t{appeared}\n")
                            idx += 1
                            if idx % 50000 == 0:
                                speed = idx / max(time.time() - t0, 0.01)
                                eta = (n_samples - idx) / max(speed, 1)
                                err_rate = n_errors / max(idx, 1)
                                msg = (f"\r    {idx:,}/{n_samples:,} ({idx/n_samples*100:.0f}%) "
                                       f"{speed:.0f}/s ETA {eta/60:.1f}min")
                                if n_errors:
                                    msg += f" [异常{n_errors:,}={err_rate:.0%}]"
                                print(msg, end="", flush=True)
                                if err_rate > MAX_ERROR_RATE and idx > 1000:
                                    raise RuntimeError(
                                        f"异常率 {err_rate:.1%} > {MAX_ERROR_RATE:.0%}")

            print(f"\n    {split}: {idx:,} 样本, {time.time()-t0:.0f}s"
                  + (f", {n_errors:,} 异常" if n_errors else ""))
            # 验证
            y_check = np.fromfile(yp, dtype=np.float32, count=min(100, idx))
            print(f"    [验证] y.dat: {int((y_check >= 0).sum())}/100 有效, "
                  f"唯一值={sorted(set(y_check.tolist()))[:5]}")

        print("    ✓ 向量化完成")

    # Step 4: 读取 .dat → X.npy
    missing = [f for f in required_dats if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(f"向量化未完成: {[os.path.basename(f) for f in missing]}")

    print("    读取向量化特征...")
    ndim = cfg["ndim"]

    # ── .dat 完整性预检查 ──
    y_tr_path = os.path.join(raw, "y_train.dat")
    y_te_path = os.path.join(raw, "y_test.dat")
    x_tr_path = os.path.join(raw, "X_train.dat")
    x_te_path = os.path.join(raw, "X_test.dat")
    n_tr_total = os.path.getsize(y_tr_path) // 4
    n_te_total = os.path.getsize(y_te_path) // 4
    row_bytes = ndim * 4

    # 尺寸一致性: X.dat 大小应 == y.dat 行数 × ndim × 4
    x_tr_expected = n_tr_total * row_bytes
    x_tr_actual = os.path.getsize(x_tr_path)
    if x_tr_actual != x_tr_expected:
        print(f"    [错误] X_train.dat 大小不匹配: {x_tr_actual:,} != 期望 {x_tr_expected:,}")
        print(f"    可能原因: ndim 不正确 ({ndim}) 或文件截断")
        for f in required_dats:
            if os.path.exists(f): os.remove(f)
        raise RuntimeError(f"X_train.dat 大小不匹配, 已清理, 请重新运行")

    # 读 y (小, ~3MB each, 直接加载)
    y_tr_all = np.fromfile(y_tr_path, dtype=np.float32)
    y_te_all = np.fromfile(y_te_path, dtype=np.float32)
    tr_ok = y_tr_all >= 0
    te_ok = y_te_all >= 0
    n_tr = int(tr_ok.sum())
    n_te = int(te_ok.sum())
    N = n_tr + n_te
    print(f"    有效样本: train={n_tr:,}/{n_tr_total:,}, test={n_te:,}/{n_te_total:,}, 总计={N:,}")

    # 检测损坏的 .dat (多次断点续传残留导致 label 全部无效)
    if N == 0 and (n_tr_total + n_te_total) > 0:
        # 诊断: 打印前 20 个 y 值
        print(f"    [诊断] y_train 前 20 值: {y_tr_all[:20].tolist()}")
        print(f"    [诊断] y_train 唯一值: {sorted(set(y_tr_all[:1000].tolist()))}")
        print(f"    [诊断] y_train 文件大小: {os.path.getsize(y_tr_path):,} bytes")
        print(f"    [诊断] X_train 文件大小: {os.path.getsize(x_tr_path):,} bytes")
        print(f"    [错误] 全部 label<0 → .dat 文件损坏")
        print(f"    自动清理...")
        del y_tr_all, y_te_all
        for f in required_dats:
            if os.path.exists(f): os.remove(f)
        for s in ["train", "test"]:
            pf = os.path.join(raw, f"_progress_{s}.txt")
            if os.path.exists(pf): os.remove(pf)
            af = os.path.join(raw, f"appeared_{s}.txt")
            if os.path.exists(af): os.remove(af)
        raise RuntimeError(f".dat 已清理, 请重新运行: python build_dataset.py {dataset_key}")

    tr_indices = np.where(tr_ok)[0]
    te_indices = np.where(te_ok)[0]
    del tr_ok, te_ok

    os.makedirs(out, exist_ok=True)
    xp_out = os.path.join(out, "X.npy")
    yp_out = os.path.join(out, "y.npy")
    mp_out = os.path.join(out, "months.npy")

    # 构建 y_final (y 很小, 直接在 RAM 中处理)
    y_final = np.concatenate([
        y_tr_all[tr_indices].astype(np.int32),
        y_te_all[te_indices].astype(np.int32),
    ])
    months = np.ones(N, dtype=np.int32)

    # 有效行号 bool 掩码 (比 Python set 省 100 倍内存: 800KB vs 80MB)
    tr_valid_mask = np.zeros(n_tr_total, dtype=bool)
    tr_valid_mask[tr_indices] = True
    te_valid_mask = np.zeros(n_te_total, dtype=bool)
    te_valid_mask[te_indices] = True
    del tr_indices, te_indices, y_tr_all, y_te_all
    gc.collect()

    # ── 写 X.npy: 顺序读 .dat → 过滤 → 顺序写 ──
    _filter_dat_to_npy(
        xp_out,
        [x_tr_path, x_te_path],
        [tr_valid_mask, te_valid_mask],
        [n_tr_total, n_te_total],
        ndim,
    )

    del tr_valid_mask, te_valid_mask

    # ── 月份分配: 从 Step 3 保存的 appeared txt 文件读取 (零额外 I/O) ──
    appeared_valid = []
    for split in ["train", "test"]:
        app_file = os.path.join(raw, f"appeared_{split}.txt")
        if os.path.exists(app_file):
            with open(app_file, encoding="utf-8") as af:
                for aline in af:
                    parts = aline.strip().split("\t")
                    if len(parts) >= 2:
                        lbl = int(parts[0])
                        appeared = parts[1]
                        if lbl >= 0:
                            appeared_valid.append(appeared)

    if len(appeared_valid) == N and N > 0:
        appeared_arr = np.array(appeared_valid)
        unique_times = sorted(set(appeared_valid))
        print(f"    时间范围: {unique_times[0]} → {unique_times[-1]} ({len(unique_times)} 唯一时间点)")

        order = np.argsort(appeared_arr, kind="stable")
        n_actual_months = min(len(unique_times), 18)
        n_actual_months = max(n_actual_months, 6)
        pm = N // n_actual_months
        for m in range(n_actual_months):
            lo = m * pm
            hi = (m + 1) * pm if m < n_actual_months - 1 else N
            months[order[lo:hi]] = m + 1

        n_train_months = max(2, n_actual_months * 2 // 3)
        train_months_list = list(range(1, n_train_months + 1))
        eval_months_list = list(range(n_train_months + 1, n_actual_months + 1))
        print(f"    分为 {n_actual_months} 个月: train=M{train_months_list[0]}-{train_months_list[-1]}, "
              f"eval=M{eval_months_list[0]}-{eval_months_list[-1]}")
    else:
        if appeared_valid:
            print(f"    [WARN] appeared 数量 ({len(appeared_valid)}) != N ({N})")
        else:
            print(f"    appeared 文件不存在, 用 train/test 分割")
        n_actual_months = 12
        if n_tr > 0:
            ptr = n_tr // 8
            for m in range(8):
                months[m*ptr:((m+1)*ptr if m < 7 else n_tr)] = m + 1
        if n_te > 0:
            pte = n_te // 4
            for m in range(4):
                lo = n_tr + m * pte
                months[lo:(n_tr + (m+1)*pte if m < 3 else N)] = m + 9
        train_months_list = list(range(1, 9))
        eval_months_list = list(range(9, 13))
        print(f"    月份: train=M1-8, eval=M9-12")
    del appeared_valid

    y = y_final
    np.save(yp_out, y)
    np.save(mp_out, months)

    # metadata.json
    meta_dict = {
        "name": cfg["name"], "source": cfg["source"],
        "n_samples": int(N), "n_features": ndim,
        "n_months": n_actual_months,
        "malware_ratio": round(float(np.mean(y == 1)), 4),
        "train_months": train_months_list, "eval_months": eval_months_list,
        "feature_type": "EMBER_FULL", "feature_ranges": EMBER_RANGES,
    }
    with open(os.path.join(out, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2, ensure_ascii=False)

    # 终态化: 按月排序 + 偏移量索引 + 验证
    from dataset_utils import finalize_dataset
    finalize_dataset(out)

    mb = N * ndim * 4 / 1024**2
    print(f"    ✓ 保存 {out}/ (N={N:,}, D={ndim}, {mb:.0f} MB)")


def build_ember2017():
    """下载并处理 EMBER-2017 (1.1M PE, feature v2, 2381d)"""
    _build_ember_v2("ember2017")

def build_ember2018():
    """下载并处理 EMBER-2018 (1.0M PE, feature v2, 2381d)"""
    _build_ember_v2("ember2018")


# ═══════════════════════════════════════
#  1c. EMBER-2024 (feature v3, 2568d)
# ═══════════════════════════════════════

EMBER2024_RANGES = {
    "ByteHistogram": (0, 256),   "ByteEntropy": (256, 512),
    "StringInfo": (512, 616),    "GeneralInfo": (616, 636),
    "HeaderInfo": (636, 738),    "SectionInfo": (738, 1043),
    "ImportInfo": (1043, 2323),  "ExportInfo": (2323, 2451),
    "DataDirs": (2451, 2491),    "RichHeader": (2491, 2531),
    "Authenticode": (2531, 2551), "ParseWarnings": (2551, 2568),
}

def build_ember2024():
    """
    自动下载 EMBER-2024 (Joyce et al. 2025)

    3.2M 多平台文件, feature v3 (2568d PE / 696d non-PE)
    来源: HuggingFace FutureComputing4AI/EMBER2024
    使用 thrember 库下载 + 向量化
    """
    print("\n  ╔═══════════════════════════════════════════════════╗")
    print("  ║  EMBER-2024 (3.2M 多平台, 2568d feature v3)     ║")
    print("  ║  来源: HuggingFace FutureComputing4AI/EMBER2024  ║")
    print("  ╚═══════════════════════════════════════════════════╝")
    out = DATASETS["ember2024"]
    if os.path.exists(os.path.join(out, "X.npy")):
        m = _load_json(os.path.join(out, "metadata.json"))
        print(f"    [已构建] N={m['n_samples']:,}, D={m['n_features']}")
        return

    raw = os.path.join(RAW_DIR, "ember2024")
    os.makedirs(raw, exist_ok=True)
    ndim = 2568

    # ── Step 1: 安装 thrember + 下载 ──
    try:
        import thrember
    except ImportError:
        print("    安装 thrember (EMBER2024 工具库)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "thrember", "-q", "--break-system-packages"],
                              stderr=subprocess.DEVNULL)
        import thrember

    # 检查是否已有向量化结果
    dat_train = os.path.join(raw, "X_train.dat")
    dat_test = os.path.join(raw, "X_test.dat")
    has_dats = os.path.exists(dat_train) and os.path.exists(dat_test)

    if not has_dats:
        # 检查是否已下载原始 JSON
        jsonl_marker = os.path.join(raw, "train_features_0.jsonl")
        json_marker = os.path.join(raw, "train", "features", "features_00000.json")
        has_raw = os.path.exists(jsonl_marker) or os.path.exists(json_marker)

        if not has_raw:
            print("    从 HuggingFace 下载 EMBER2024 数据集...")
            print("    (首次下载约 15-30GB, 取决于网速)")
            try:
                thrember.download_dataset(raw)
                print("    ✓ 下载完成")
            except Exception as e:
                print(f"    ✗ 下载失败: {e}")
                print("    手动: pip install thrember && python -c \"import thrember; "
                      f"thrember.download_dataset(r'{raw}')\"")
                raise

        # 向量化
        print("    向量化特征 (thrember v3, 2568d)...")
        try:
            thrember.create_vectorized_features(raw)
            print("    ✓ 向量化完成")
        except Exception as e:
            print(f"    ✗ 向量化失败: {e}")
            raise

    # ── Step 2: 读取 .dat → X.npy (同 v2 逻辑, 纯文件 I/O) ──
    # 检测维度 (v3 PE=2568, non-PE=696, 可能有混合)
    y_tr_path = os.path.join(raw, "y_train.dat")
    y_te_path = os.path.join(raw, "y_test.dat")
    x_tr_path = os.path.join(raw, "X_train.dat")
    x_te_path = os.path.join(raw, "X_test.dat")

    for p in [y_tr_path, y_te_path, x_tr_path, x_te_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"EMBER2024 向量化不完整: {p} 不存在")

    n_tr_total = os.path.getsize(y_tr_path) // 4
    n_te_total = os.path.getsize(y_te_path) // 4

    # 自动检测维度
    x_tr_bytes = os.path.getsize(x_tr_path)
    if n_tr_total > 0:
        detected_ndim = x_tr_bytes // (4 * n_tr_total)
        if detected_ndim > 0 and x_tr_bytes == 4 * n_tr_total * detected_ndim:
            ndim = detected_ndim
            print(f"    检测到特征维度: {ndim}")
    else:
        raise ValueError(f"EMBER2024 y_train.dat 为空: {y_tr_path}")

    row_bytes = ndim * 4

    y_tr_all = np.fromfile(y_tr_path, dtype=np.float32)
    y_te_all = np.fromfile(y_te_path, dtype=np.float32)
    tr_ok = y_tr_all >= 0
    te_ok = y_te_all >= 0
    n_tr = int(tr_ok.sum())
    n_te = int(te_ok.sum())
    N = n_tr + n_te
    print(f"    有效样本: train={n_tr:,}/{n_tr_total:,}, test={n_te:,}/{n_te_total:,}, 总计={N:,}")

    # 检测损坏
    if N == 0 and (n_tr_total + n_te_total) > 0:
        print(f"    [错误] .dat 损坏, 清理中...")
        del y_tr_all, y_te_all
        for p in [x_tr_path, x_te_path, y_tr_path, y_te_path]:
            if os.path.exists(p): os.remove(p)
        raise RuntimeError(f"EMBER2024 .dat 损坏已清理, 请重新运行")

    tr_indices = np.where(tr_ok)[0]
    te_indices = np.where(te_ok)[0]
    del tr_ok, te_ok

    os.makedirs(out, exist_ok=True)
    xp_out = os.path.join(out, "X.npy")
    yp_out = os.path.join(out, "y.npy")
    mp_out = os.path.join(out, "months.npy")

    # ── 月份分配 ──
    months = np.ones(N, dtype=np.int32)
    # EMBER-2024 数据按周收集 (Sep 2023 → Dec 2024), 已按时间顺序存储:
    #   train: 52 周 (Sep 2023 → Sep 2024)  →  月 1-13 (每月≈4周)
    #   test:  12 周 (Sep 2024 → Dec 2024)  →  月 14-16 (每月≈4周)
    # 数据内部严格按周时序排列, 自然产生真实的概念漂移
    n_months = 16
    if n_tr > 0:
        ptr = n_tr // 13
        for m in range(13):
            lo = m * ptr
            hi = (m + 1) * ptr if m < 12 else n_tr
            months[lo:hi] = m + 1
    if n_te > 0:
        pte = n_te // 3
        for m in range(3):
            lo = n_tr + m * pte
            hi = n_tr + ((m + 1) * pte if m < 2 else n_te)
            months[lo:hi] = m + 14
    train_months_list = list(range(1, 14))
    eval_months_list = list(range(14, 17))
    print(f"    月份: train=M1-13 (52周), eval=M14-16 (12周)")

    # 构建 y_final
    y_final = np.concatenate([
        y_tr_all[tr_indices].astype(np.int32),
        y_te_all[te_indices].astype(np.int32),
    ])
    # months 已在上面通过 train/test 分割设置, 不再重新初始化

    # bool 掩码 (比 Python set 省 100 倍内存)
    tr_valid_mask = np.zeros(n_tr_total, dtype=bool)
    tr_valid_mask[tr_indices] = True
    te_valid_mask = np.zeros(n_te_total, dtype=bool)
    te_valid_mask[te_indices] = True
    del tr_indices, te_indices, y_tr_all, y_te_all
    gc.collect()

    # 写 X.npy
    _filter_dat_to_npy(
        xp_out,
        [x_tr_path, x_te_path],
        [tr_valid_mask, te_valid_mask],
        [n_tr_total, n_te_total],
        ndim,
    )

    del tr_valid_mask, te_valid_mask

    y = y_final
    np.save(yp_out, y)
    np.save(mp_out, months)

    with open(os.path.join(out, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump({
            "name": "EMBER-2024", "source": "Real (Joyce et al. 2025)",
            "n_samples": int(N), "n_features": ndim,
            "n_months": n_months, "malware_ratio": round(float(np.mean(y == 1)), 4),
            "train_months": train_months_list, "eval_months": eval_months_list,
            "feature_type": "EMBER_V3", "feature_ranges": EMBER2024_RANGES,
        }, f, indent=2, ensure_ascii=False)

    # 终态化
    from dataset_utils import finalize_dataset
    finalize_dataset(out)

    mb = N * ndim * 4 / 1024**2
    print(f"    ✓ 保存 {out}/ (N={N:,}, D={ndim}, {mb:.0f} MB)")


# ═══════════════════════════════════════
#  2. BODMAS (真实数据)
# ═══════════════════════════════════════

def build_bodmas():
    """
    下载并处理 BODMAS 真实数据集 (Yang et al. 2021)
    
    BODMAS 数据托管在作者页面, 需要先手动下载再自动处理:
      https://whyisyoung.github.io/BODMAS/
    下载后将 npz/csv 文件放入 F:\\Experimental data\\_downloads\\bodmas\\
    """
    print("\n  ╔═══════════════════════════════════════════════════╗")
    print("  ║  2/4  BODMAS (57K PE, 2381d, 含家族+时间戳)     ║")
    print("  ║  来源: whyisyoung.github.io/BODMAS               ║")
    print("  ╚═══════════════════════════════════════════════════╝")
    out = DATASETS["bodmas"]
    if os.path.exists(os.path.join(out, "X.npy")):
        m = _load_json(os.path.join(out, "metadata.json"))
        print(f"    [已构建] N={m['n_samples']:,}, D={m['n_features']}")
        return

    raw = os.path.join(RAW_DIR, "bodmas")
    os.makedirs(raw, exist_ok=True)

    # ── 自动下载: Google Drive (官方托管) ──
    npz_path = os.path.join(raw, "bodmas.npz")
    csv_path = os.path.join(raw, "bodmas_metadata.csv")
    if not os.path.exists(npz_path):
        GDRIVE_FOLDER = "https://drive.google.com/drive/folders/1Uf-LebLWyi9eCv97iBal7kL1NgiGEsv_"
        print(f"    从 Google Drive 自动下载 BODMAS...")
        try:
            import gdown
        except ImportError:
            print("    安装 gdown...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
            import gdown
        try:
            # 方法1: 下载整个文件夹
            gdown.download_folder(GDRIVE_FOLDER, output=raw, quiet=False, remaining_ok=True)
            # gdown 可能保存到子目录, 搜索
            for root_d, dirs, files in os.walk(raw):
                for fn in files:
                    if fn == "bodmas.npz" and root_d != raw:
                        shutil.move(os.path.join(root_d, fn), npz_path)
                    if fn == "bodmas_metadata.csv" and root_d != raw:
                        shutil.move(os.path.join(root_d, fn), csv_path)
        except Exception as e1:
            print(f"    文件夹下载失败 ({e1}), 尝试单文件下载...")
            try:
                # 方法2: 直接用文件 ID (bodmas.npz 在该文件夹中)
                gdown.download(id="1Uf-LebLWyi9eCv97iBal7kL1NgiGEsv_",
                               output=npz_path, fuzzy=True, quiet=False)
            except Exception as e2:
                print(f"    ✗ Google Drive 下载失败: {e2}")
                print(f"    请手动下载: {GDRIVE_FOLDER}")
                print(f"    将 bodmas.npz 放入: {raw}/")
    if not os.path.exists(npz_path):
        # 尝试旧 URL (保底)
        for url in [
            "https://github.com/whyisyoung/BODMAS/releases/download/v1.0/bodmas_all.npz",
        ]:
            dest = os.path.join(raw, os.path.basename(url.split("?")[0]))
            try:
                ok = _download(url, dest, desc="BODMAS features (GitHub mirror)")
                if ok and os.path.exists(dest) and os.path.getsize(dest) < 1024:
                    os.remove(dest)
            except Exception:
                continue

    # 扫描 raw 目录寻找可用文件
    loaded = False
    for fpath in sorted(glob.glob(os.path.join(raw, "*.npz"))) + \
                 sorted(glob.glob(os.path.join(raw, "*.npy"))):
        print(f"    尝试加载: {os.path.basename(fpath)}")
        try:
            if fpath.endswith(".npz"):
                data = np.load(fpath, allow_pickle=True)
                keys = list(data.keys())
                print(f"      keys: {keys}")
                X = next((data[k] for k in ["X", "features", "x"] if k in data), None)
                y = next((data[k] for k in ["y", "labels", "label"] if k in data), None)
                ts = next((data[k] for k in ["timestamps", "appeared", "dates"] if k in data), None)
                fam = next((data[k] for k in ["families", "family", "avclass"] if k in data), None)
            elif fpath.endswith(".npy"):
                arr = np.load(fpath, allow_pickle=True)
                if arr.ndim == 2 and arr.shape[1] > 100:
                    X = arr; y = ts = fam = None
                    continue  # 需要配套 y
                else:
                    continue

            if X is None or y is None:
                print(f"      ✗ 缺少 X 或 y"); continue

            X = np.asarray(X, dtype=np.float32)
            y = np.asarray(y, dtype=np.int32)
            y = np.where(y > 0, 1, 0)  # 统一为 {0,1}

            # 从 bodmas_metadata.csv 补充时间戳和家族 (官方 npz 只含 X, y)
            meta_csv = os.path.join(raw, "bodmas_metadata.csv")
            if ts is None and os.path.exists(meta_csv):
                try:
                    import pandas as pd
                    mdf = pd.read_csv(meta_csv)
                    if "appeared" in mdf.columns:
                        ts = pd.to_datetime(mdf["appeared"]).values.astype(np.int64)
                        print(f"      ✓ 从 metadata.csv 加载时间戳 ({len(ts):,} 条)")
                    if fam is None and "family" in mdf.columns:
                        fam = mdf["family"].values
                        print(f"      ✓ 从 metadata.csv 加载家族标签")
                except Exception as me:
                    print(f"      metadata.csv 解析失败: {me}")

            # 按时间戳排序+分月
            n_months = 14
            months = np.ones(len(y), dtype=np.int32)
            if ts is not None:
                ts = np.asarray(ts)
                order = np.argsort(ts)
                X, y = X[order], y[order]
                if fam is not None: fam = np.asarray(fam)[order]
                pm = len(y) // n_months
                for m in range(n_months):
                    months[m*pm:((m+1)*pm if m < n_months-1 else len(y))] = m + 1
            else:
                pm = len(y) // n_months
                for m in range(n_months):
                    months[m*pm:((m+1)*pm if m < n_months-1 else len(y))] = m + 1

            # 处理家族标签
            families = None
            if fam is not None:
                fam = np.asarray(fam)
                if fam.dtype.kind in ('U', 'S', 'O'):
                    uq = sorted(set(str(f) for f in fam))
                    fmap = {f: i for i, f in enumerate(uq)}
                    families = np.array([fmap[str(f)] for f in fam], dtype=np.int32)
                else:
                    families = fam.astype(np.int32)

            nfam = len(np.unique(families)) if families is not None else 0
            print(f"    ✓ 真实 BODMAS: N={len(y):,}, D={X.shape[1]}, "
                  f"恶意率={y.mean():.3f}, 家族={nfam}")

            _save(out, X, y, months, {
                "name": "BODMAS", "source": "Real (Yang et al. 2021)",
                "n_samples": int(len(y)), "n_features": int(X.shape[1]),
                "n_months": n_months, "malware_ratio": round(float(y.mean()), 4),
                "n_families": nfam,
                "train_months": [1,2,3], "eval_months": list(range(4, n_months+1)),
                "feature_type": "EMBER_FULL", "feature_ranges": EMBER_RANGES,
            }, families=families)
            loaded = True
            break
        except Exception as e:
            print(f"      ✗ {e}")

    if not loaded:
        print()
        print("  ╔══════════════════════════════════════════════════════════╗")
        print("  ║  BODMAS 数据需要手动下载:                               ║")
        print("  ║                                                          ║")
        print("  ║  1. 访问 https://whyisyoung.github.io/BODMAS/           ║")
        print("  ║  2. 下载特征文件 (.npz 或 .csv)                         ║")
        print(f"  ║  3. 放入 {raw}/ ║")
        print("  ║  4. 重新运行 python build_dataset.py bodmas             ║")
        print("  ╚══════════════════════════════════════════════════════════╝")
        raise FileNotFoundError(f"BODMAS 数据未找到。请下载后放入 {raw}/")


# ═══════════════════════════════════════
#  3. SOREL-20M (真实数据子集)
# ═══════════════════════════════════════

def build_sorel():
    """
    自动下载 SOREL-20M validation 子集 (Harang & Rudd 2020)
    
    流程: S3流式下载→磁盘 → zipfile解压npz内的npy → 分块子采样
    全程不将大数组加载到RAM
    """
    print("\n  ╔═══════════════════════════════════════════════════╗")
    print("  ║  3/4  SOREL-20M (validation子集, 2381d)          ║")
    print("  ║  来源: s3://sorel-20m (公开桶, 无需认证)          ║")
    print("  ╚═══════════════════════════════════════════════════╝")
    out = DATASETS["sorel"]
    if os.path.exists(os.path.join(out, "X.npy")):
        m = _load_json(os.path.join(out, "metadata.json"))
        print(f"    [已构建] N={m['n_samples']:,}, D={m['n_features']}")
        return

    raw = os.path.join(RAW_DIR, "sorel")
    os.makedirs(raw, exist_ok=True)

    # ── Step 1: 流式下载 npz 到磁盘 (不占 RAM) ──
    S3_BUCKET = "sorel-20m"
    S3_KEY = "09-DEC-2020/lightGBM-features/validation-features.npz"
    dest_npz = os.path.join(raw, "validation-features.npz")
    SUBSAMPLE = 200000

    if not os.path.exists(dest_npz):
        print(f"    从 S3 流式下载 validation set (~22GB) 到磁盘...")
        try:
            import boto3, botocore
            s3 = boto3.client("s3", config=botocore.config.Config(
                signature_version=botocore.UNSIGNED))
            # 流式下载, 不经 RAM
            s3.download_file(S3_BUCKET, S3_KEY, dest_npz,
                             Callback=_S3Progress(desc="SOREL validation"))
            print(f"\n    ✓ 下载完成: {os.path.getsize(dest_npz)/1024**3:.1f} GB")
        except ImportError:
            print("    boto3 不可用, 使用 HTTPS 流式下载...")
            url = f"https://{S3_BUCKET}.s3.amazonaws.com/{S3_KEY}"
            _download(url, dest_npz, desc="SOREL validation features")
        except Exception as e:
            print(f"    ✗ 下载失败: {e}")
            if os.path.exists(dest_npz):
                os.remove(dest_npz)
            raise

    # ── Step 2: 从 npz(zip) 中解压 npy 到磁盘, 不加载到 RAM ──
    print(f"    从 npz 解压数组到磁盘 (不加载到内存)...")
    import zipfile
    extracted_dir = os.path.join(raw, "_extracted")
    os.makedirs(extracted_dir, exist_ok=True)
    with zipfile.ZipFile(dest_npz, "r") as zf:
        names = zf.namelist()
        print(f"    npz 内含: {names}")
        for name in names:
            dest_npy = os.path.join(extracted_dir, name)
            if not os.path.exists(dest_npy):
                print(f"    解压 {name}...")
                zf.extract(name, extracted_dir)

    # ── Step 3: 读取 npy header + 分块子采样 ──
    # npz 中的 key 可能是 arr_0 (X), arr_1 (y) 或 X, y
    npy_files = sorted(glob.glob(os.path.join(extracted_dir, "*.npy")))
    if not npy_files:
        raise FileNotFoundError(f"npz 解压后未找到 .npy: {extracted_dir}")

    # 找最大的 npy = X, 最小的 = y
    npy_sizes = [(f, os.path.getsize(f)) for f in npy_files]
    npy_sizes.sort(key=lambda x: -x[1])
    x_npy = npy_sizes[0][0]
    y_npy = npy_sizes[1][0] if len(npy_sizes) > 1 else None

    print(f"    X: {os.path.basename(x_npy)} ({npy_sizes[0][1]/1024**3:.1f}GB)")

    # 读取 npy header 获取 shape/dtype (不 memmap, 不加载数据)
    def _read_npy_header(path):
        with open(path, "rb") as f:
            magic = f.read(6)
            assert magic == b"\x93NUMPY", f"Not a npy file: {path}"
            ver = f.read(2)
            if ver == b"\x01\x00":
                hdr_len = struct.unpack("<H", f.read(2))[0]
            else:
                hdr_len = struct.unpack("<I", f.read(4))[0]
            hdr_str = f.read(hdr_len).decode("latin1").strip()
            hdr = eval(hdr_str)  # {'descr': '<f4', 'fortran_order': False, 'shape': (...)}
            data_offset = f.tell()
        return hdr["shape"], np.dtype(hdr["descr"]), data_offset

    x_shape, x_dtype, x_data_offset = _read_npy_header(x_npy)
    N_total = x_shape[0]
    ndim_s = x_shape[1] if len(x_shape) > 1 else 2381
    x_row_bytes = ndim_s * x_dtype.itemsize
    print(f"    总样本: {N_total:,}, D={ndim_s}, dtype={x_dtype}")

    # y
    if y_npy:
        y_shape, y_dtype, y_data_offset = _read_npy_header(y_npy)
    else:
        y_shape, y_dtype, y_data_offset = None, None, None

    # 子采样: 取前 N_out 行
    N_out = min(SUBSAMPLE, N_total)

    # 分块写 X.npy (纯文件 I/O)
    os.makedirs(out, exist_ok=True)
    xp_sorel = os.path.join(out, "X.npy")

    CHUNK = 5000
    with open(xp_sorel, "wb") as xf:
        _write_npy_header(xf, (N_out, ndim_s))

        with open(x_npy, "rb") as src:
            src.seek(x_data_offset)
            for i in range(0, N_out, CHUNK):
                n_read = min(CHUNK, N_out - i)
                raw = src.read(n_read * x_row_bytes)
                chunk = np.frombuffer(raw, dtype=x_dtype).reshape(n_read, ndim_s)
                xf.write(chunk.astype(np.float32).tobytes())
                if i % 50000 < CHUNK:
                    print(f"\r    子采样: {i+n_read:,}/{N_out:,}", end="", flush=True)
    print(f"\r    子采样: {N_out:,}/{N_out:,} ✓")

    # y
    if y_npy and y_shape:
        with open(y_npy, "rb") as yf:
            yf.seek(y_data_offset)
            y_raw = yf.read(N_out * y_dtype.itemsize)
        y = np.frombuffer(y_raw, dtype=y_dtype)[:N_out].astype(np.int32)
        y = np.where(y > 0, 1, 0)
    else:
        y = np.zeros(N_out, dtype=np.int32)

    # 月份
    months = np.ones(N_out, dtype=np.int32)
    pm = N_out // 12
    for m in range(12):
        months[m*pm:((m+1)*pm if m < 11 else N_out)] = m + 1

    np.save(os.path.join(out, "y.npy"), y)
    np.save(os.path.join(out, "months.npy"), months)

    with open(os.path.join(out, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump({
            "name": "SOREL-20M Subset", "source": "Real (Harang & Rudd 2020)",
            "n_samples": int(N_out), "n_features": ndim_s,
            "n_months": 12, "malware_ratio": round(float(np.mean(y == 1)), 4),
            "train_months": [1,2,3], "eval_months": list(range(4,13)),
            "feature_type": "EMBER_FULL", "feature_ranges": EMBER_RANGES,
        }, f, indent=2, ensure_ascii=False)

    # 终态化
    from dataset_utils import finalize_dataset
    finalize_dataset(out)

    print(f"    ✓ SOREL: N={N_out:,}, D={ndim_s}, 恶意率={np.mean(y==1):.3f}")

    # 清理解压的巨大临时文件
    shutil.rmtree(extracted_dir, ignore_errors=True)
    print(f"    清理临时文件 ✓")


class _S3Progress:
    """boto3 download callback for progress display"""
    def __init__(self, desc=""):
        self._desc = desc
        self._seen = 0
        self._t0 = time.time()
    def __call__(self, bytes_amount):
        self._seen += bytes_amount
        elapsed = time.time() - self._t0
        speed = self._seen / max(elapsed, 0.01) / 1024**2
        print(f"\r    {self._desc}: {self._seen/1024**3:.1f} GB  {speed:.0f} MB/s",
              end="", flush=True)


# ═══════════════════════════════════════
#  4. CIC-MalMem-2022 (内存恶意软件检测)
# ═══════════════════════════════════════

def build_malmem():
    """
    自动下载 CIC-MalMem-2022 (Carrier et al. 2022)
    来源: HuggingFace bvk/CIC-MalMem-2022

    58,596 内存 dump 样本, 55 维特征, 16 类恶意软件类别
    """
    print("\n  ╔═══════════════════════════════════════════════════╗")
    print("  ║  4/4  CIC-MalMem-2022 (58K 内存特征, 55d)       ║")
    print("  ║  来源: huggingface.co/datasets/bvk/CIC-MalMem   ║")
    print("  ╚═══════════════════════════════════════════════════╝")
    out = DATASETS["malmem"]
    if os.path.exists(os.path.join(out, "X.npy")):
        m = _load_json(os.path.join(out, "metadata.json"))
        print(f"    [已构建] N={m['n_samples']:,}, D={m['n_features']}")
        return

    raw = os.path.join(RAW_DIR, "malmem")
    os.makedirs(raw, exist_ok=True)

    # ── 自动下载: HuggingFace parquet (无需认证, ~5MB) ──
    PARQUET_URL = ("https://huggingface.co/datasets/bvk/CIC-MalMem-2022/"
                   "resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet")
    parquet_path = os.path.join(raw, "malmem.parquet")

    if not os.path.exists(parquet_path):
        print("    从 HuggingFace 自动下载 CIC-MalMem-2022...")
        try:
            _download(PARQUET_URL, parquet_path, desc="CIC-MalMem-2022 (parquet)")
        except Exception as e:
            print(f"    parquet 下载失败 ({e}), 尝试 datasets 库...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install",
                                       "datasets", "-q"])
                from datasets import load_dataset
                ds = load_dataset("bvk/CIC-MalMem-2022", split="train")
                ds.to_parquet(parquet_path)
                print(f"    ✓ datasets 库下载完成")
            except Exception as e2:
                raise FileNotFoundError(
                    f"CIC-MalMem-2022 下载失败: {e2}\n"
                    f"请手动下载: {PARQUET_URL}") from e2

    # ── 解析 ──
    print("    解析特征...")
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    print(f"    原始: {len(df):,} 行, {len(df.columns)} 列")

    # 标签: Class 列 (Benign / Malware)
    if "Class" in df.columns:
        y = (df["Class"].str.lower() != "benign").astype(np.int32).values
    elif "class" in df.columns:
        y = (df["class"].str.lower() != "benign").astype(np.int32).values
    else:
        raise ValueError(f"找不到 Class 列, 可用列: {list(df.columns)}")

    # 类别: Category 列 (16 类恶意软件家族)
    cat_col = "Category" if "Category" in df.columns else "category"
    if cat_col in df.columns:
        cat_vals = df[cat_col].values
        uq = sorted(set(str(c) for c in cat_vals))
        cmap = {c: i for i, c in enumerate(uq)}
        families = np.array([cmap[str(c)] for c in cat_vals], dtype=np.int32)
    else:
        families = None

    # 特征: 排除非数值列
    exclude = {"Class", "class", "Category", "category", "Filename", "filename"}
    feat_cols = [c for c in df.columns if c not in exclude]
    X = df[feat_cols].values.astype(np.float32)

    # 处理 NaN / Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # 分月: 按恶意率渐增模拟概念漂移 (MalMem 无时间戳)
    # 月1-3 (训练): 恶意率 ~15%  →  月4-12 (评估): 恶意率逐增到 ~85%
    n_months = 12
    N = len(y)
    pm = N // n_months
    rng = np.random.RandomState(42)
    benign_idx = np.where(y == 0)[0].copy()
    malware_idx = np.where(y == 1)[0].copy()
    rng.shuffle(benign_idx)
    rng.shuffle(malware_idx)

    months = np.ones(N, dtype=np.int32)
    bi, mi = 0, 0
    for m in range(n_months):
        n_month = pm if m < n_months - 1 else N - m * pm
        # 恶意率: 月1→15%, 月12→85%
        mal_ratio = 0.15 + 0.70 * m / max(n_months - 1, 1)
        n_mal = min(int(n_month * mal_ratio), len(malware_idx) - mi)
        n_ben = min(n_month - n_mal, len(benign_idx) - bi)
        # 不够时补齐
        if n_ben + n_mal < n_month:
            extra = n_month - n_ben - n_mal
            if mi + n_mal + extra <= len(malware_idx):
                n_mal += extra
            elif bi + n_ben + extra <= len(benign_idx):
                n_ben += extra
        idx_month = np.concatenate([
            benign_idx[bi:bi+n_ben],
            malware_idx[mi:mi+n_mal],
        ])
        months[idx_month] = m + 1
        bi += n_ben; mi += n_mal

    nfam = len(np.unique(families)) if families is not None else 0
    print(f"    ✓ CIC-MalMem-2022: N={len(y):,}, D={X.shape[1]}, "
          f"恶意率={y.mean():.3f}, 类别={nfam}")

    _save(out, X, y, months, {
        "name": "CIC-MalMem-2022",
        "source": "Real (Carrier et al. 2022, ICISSP)",
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "n_months": n_months,
        "malware_ratio": round(float(y.mean()), 4),
        "n_families": nfam,
        "train_months": [1, 2, 3],
        "eval_months": list(range(4, n_months + 1)),
        "feature_type": "MEMORY_VOLATILITY",
        "feature_ranges": {
            "ByteHistogram": [0, 55],
        },
        "feature_groups": {
            "pslist": feat_cols[:5],
            "dlllist": feat_cols[5:7],
            "handles": feat_cols[7:20],
            "ldrmodules": feat_cols[20:26],
            "malfind": feat_cols[26:30],
            "psxview": feat_cols[30:44],
            "modules": feat_cols[44:45],
            "svcscan": feat_cols[45:52],
            "callbacks": feat_cols[52:55],
        },
    }, families=families)


# ═══════════════════════════════════════
#  主入口
# ═══════════════════════════════════════

def build_dataset(which=None):
    """构建真实数据集。which=None 全部, 或 'ember2017'/'ember2018'/'ember2024'/'ember'(全部)/'bodmas'/'sorel'/'malmem'"""
    print("═" * 60)
    print("  CoDefenderC3 真实数据集下载 & 构建器")
    print("  (全部使用真实恶意软件特征数据, 无合成数据)")
    print("═" * 60)
    print(f"  目标: {BASE_DIR}")
    print(f"  缓存: {RAW_DIR}")

    # 磁盘空间检查
    try:
        if os.name == "nt":
            import ctypes
            free_bytes = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                BASE_DIR, None, None, ctypes.pointer(free_bytes))
            free_gb = free_bytes.value / 1024**3
        else:
            st = os.statvfs(BASE_DIR)
            free_gb = st.f_bavail * st.f_frsize / 1024**3
        print(f"  磁盘剩余: {free_gb:.1f} GB")
        if free_gb < 15:
            print(f"  [警告] 磁盘空间不足 15GB, 构建可能失败!")
            print(f"  EMBER-2018 需要约 25GB (下载+解压+向量化+输出)")
    except Exception:
        pass

    builders = {
        "ember2017": build_ember2017, "ember2018": build_ember2018,
        "ember2024": build_ember2024,
        "bodmas": build_bodmas, "sorel": build_sorel, "malmem": build_malmem,
    }

    # rawpe 使用独立脚本
    if which and which.lower() == "rawpe":
        print("\n  构建 rawpe (原始 PE 数据集)...")
        ret = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(__file__), "build_rawpe.py")],
            cwd=os.path.dirname(os.path.abspath(__file__)))
        if ret.returncode == 0:
            print("  ✓ rawpe 构建完成")
        else:
            print("  ✗ rawpe 构建失败")
        return

    # 兼容: "ember" → 构建所有 EMBER 版本
    if which and which.lower() == "ember":
        targets = ["ember2017", "ember2018", "ember2024"]
    elif which and which.lower() == "all":
        targets = list(builders.keys()) + ["rawpe"]
    else:
        targets = [which.lower()] if which else list(builders.keys())

    ok, fail = [], []
    for t in targets:
        if t not in builders:
            print(f"\n  未知数据集: {t}"); continue

        # 已构建则跳过
        out_dir = DATASETS.get(t, "")
        if os.path.exists(os.path.join(out_dir, "X.npy")):
            try:
                m = _load_json(os.path.join(out_dir, "metadata.json"))
                print(f"\n  [已构建] {t}: N={m['n_samples']:,}")
                ok.append(t)
                continue
            except Exception:
                pass

        # ══ 关键: 每个数据集在独立子进程中构建 ══
        # LIEF C++ 内存泄漏在进程内不可回收 (gc.collect 无效)
        # 子进程退出时 OS 强制回收全部内存 (包括 C++ 堆 + 文件缓存)
        print(f"\n  构建 {t} (独立子进程, 防止内存泄漏)...")
        for attempt in range(2):
            ret = subprocess.run(
                [sys.executable, os.path.abspath(__file__), t],
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
            if ret.returncode == 0 and os.path.exists(os.path.join(out_dir, "X.npy")):
                ok.append(t)
                break
            elif attempt == 0:
                print(f"    第 1 次失败 (可能是残留损坏), 重试...")
            else:
                print(f"    ✗ {t}: 子进程退出码 {ret.returncode}")
                fail.append(t)

    print("\n" + "═" * 60)
    print("  结果:")
    for t in ok:
        d = DATASETS[t]
        mp = os.path.join(d, "metadata.json")
        if os.path.exists(mp):
            m = _load_json(mp)
            mb = os.path.getsize(os.path.join(d, "X.npy")) / 1024**2
            print(f"  ✓ {m['name']:25s} N={m['n_samples']:>8,}  D={m['n_features']:>4}  "
                  f"mal={m['malware_ratio']:.1%}  {mb:>6.0f} MB")
    for t in fail:
        print(f"  ✗ {t:25s} → 需要手动下载 (见上方提示)")
    print("═" * 60)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    which = args[0] if args else None

    if which and which.lower() in {
        "ember2017", "ember2018", "ember2024", "bodmas", "sorel", "malmem"
    }:
        builders = {
            "ember2017": build_ember2017, "ember2018": build_ember2018,
            "ember2024": build_ember2024,
            "bodmas": build_bodmas, "sorel": build_sorel, "malmem": build_malmem,
        }
        try:
            builders[which.lower()]()
        except Exception as e:
            print(f"\n  ✗ {which}: {e}")
            sys.exit(1)
    elif which and which.lower() == "rawpe":
        from build_rawpe import main as rawpe_main
        rawpe_main()
    else:
        # 多数据集构建 (用子进程隔离)
        build_dataset(which)

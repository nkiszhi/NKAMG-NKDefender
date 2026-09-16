"""
CoDefenderC3 数据集工具 — 终态化 + 验证 + 月份索引
=====================================================

核心架构优化:
  1. finalize_dataset(): 按月排序 X/y/months + 计算偏移量索引
  2. validate_dataset(): 每步产出完整性校验
  3. 月份偏移量: get_month() 从 O(N) 全盘扫描 → O(1) 定点读取

数据布局 (终态化后):
  X.npy:     [month1_samples | month2_samples | ... | monthK_samples]
  y.npy:     [month1_labels  | month2_labels  | ... | monthK_labels]
  months.npy:[1,1,1,...,1    | 2,2,2,...,2    | ... | K,K,...,K]
  metadata.json: {"month_offsets": {1: {"start":0, "count":44321}, 2: {...}, ...}}
"""
import os, json, struct, hashlib, time
import numpy as np


# ═══════════════════════════════════════
#  月份偏移量计算
# ═══════════════════════════════════════

def compute_month_offsets(months):
    """
    计算每个月在排序后数组中的 [start, count]。
    参数: months — 已按月排序的 int32 数组
    返回: {month: {"start": int, "count": int}}
    """
    if len(months) == 0:
        return {}
    # 向量化: 找到月份变化的位置
    changes = np.where(np.diff(months) != 0)[0] + 1
    starts = np.concatenate([[0], changes])
    ends = np.concatenate([changes, [len(months)]])
    month_vals = months[starts]
    return {int(m): {"start": int(s), "count": int(e - s)}
            for m, s, e in zip(month_vals, starts, ends)}


# ═══════════════════════════════════════
#  数据集终态化 (排序 + 索引 + 验证)
# ═══════════════════════════════════════

def finalize_dataset(out_dir, quiet=False):
    """
    终态化数据集: 按月排序 + 计算偏移量索引 + 验证完整性。
    
    调用时机: 每个 build_XXX() 函数的最后一步。
    
    操作:
      1. 读取 months.npy, 检查是否已排序
      2. 如未排序: 按月排序 X.npy / y.npy / months.npy / families.npy
      3. 计算月份偏移量, 写入 metadata.json
      4. 写 features.jsonl (轻量索引)
      5. 计算 X.npy SHA256 (前 1MB 快速校验)
      6. 全面验证
    
    返回: (n_samples, n_features, n_months) 三元组
    """
    xp = os.path.join(out_dir, "X.npy")
    yp = os.path.join(out_dir, "y.npy")
    mp = os.path.join(out_dir, "months.npy")
    meta_p = os.path.join(out_dir, "metadata.json")
    
    for p in [xp, yp, mp, meta_p]:
        assert os.path.exists(p), f"终态化: 缺少 {p}"
    
    months = np.load(mp)
    y = np.load(yp)
    N = len(y)
    assert len(months) == N, f"months ({len(months)}) != y ({N})"
    
    # ── Step 1: 检查是否已按月排序 (向量化) ──
    is_sorted = bool(np.all(np.diff(months) >= 0)) if N > 1 else True
    
    if not is_sorted:
        if not quiet:
            print(f"    [终态化] X/y/months 未按月排序, 正在排序...")
        t0 = time.time()
        
        # 排序 key: 先按月, 月内保持原序 (stable sort)
        order = np.argsort(months, kind="stable")
        
        # 排序 y, months (小数组, 直接在内存中)
        y = y[order]
        months = months[order]
        np.save(yp, y)
        np.save(mp, months)
        
        # 排序 families (如果存在)
        fp = os.path.join(out_dir, "families.npy")
        if os.path.exists(fp):
            fam = np.load(fp)
            np.save(fp, fam[order])
            del fam
        
        # 排序 X.npy (大文件, 分块读写)
        _sort_x_npy(xp, order)
        
        if not quiet:
            print(f"    [终态化] 排序完成 ({time.time()-t0:.1f}s)")
        del order
    else:
        if not quiet:
            print(f"    [终态化] 数据已按月排序, 跳过排序")
    
    # ── Step 2: 计算月份偏移量 ──
    offsets = compute_month_offsets(months)
    
    # ── Step 3: 更新 metadata.json ──
    meta = {}
    if os.path.exists(meta_p):
        with open(meta_p, encoding="utf-8") as f:
            meta = json.load(f)
    
    # 序列化 offsets (JSON key 必须是 str)
    meta["month_offsets"] = {str(m): v for m, v in offsets.items()}
    meta["sorted_by_month"] = True
    
    # 快速校验: X.npy 前 1MB 的 SHA256
    meta["x_checksum"] = _quick_checksum(xp)
    
    with open(meta_p, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    
    # ── Step 4: 写 features.jsonl (批量, 比逐行快 10x) ──
    jlp = os.path.join(out_dir, "features.jsonl")
    fam_path = os.path.join(out_dir, "families.npy")
    families = np.load(fam_path) if os.path.exists(fam_path) else None
    with open(jlp, "w", encoding="utf-8") as f:
        BATCH = 50000
        for start in range(0, N, BATCH):
            end = min(start + BATCH, N)
            lines = []
            for i in range(start, end):
                rec = {"month": int(months[i]), "label": int(y[i])}
                if families is not None:
                    rec["family"] = int(families[i])
                lines.append(json.dumps(rec))
            f.write("\n".join(lines) + "\n")
    
    # ── Step 5: 验证 ──
    ndim = meta.get("n_features", 0)
    if ndim == 0:
        # 从 X.npy header 读取
        with open(xp, "rb") as f:
            f.read(6)  # magic
            ver = f.read(2)
            hdr_len = struct.unpack("<H" if ver == b"\x01\x00" else "<I",
                                    f.read(2 if ver == b"\x01\x00" else 4))[0]
            hdr = eval(f.read(hdr_len).decode("latin1").strip())
            ndim = hdr["shape"][1]
    
    errors = validate_dataset(out_dir, quiet=True)
    if errors:
        for e in errors:
            print(f"    [验证失败] {e}")
        raise RuntimeError(f"终态化验证失败: {len(errors)} 个错误")
    
    if not quiet:
        n_months = len(offsets)
        print(f"    [终态化] ✓ N={N:,}, D={ndim}, {n_months} 个月, "
              f"偏移量: {{{min(offsets)}: +{offsets[min(offsets)]['count']:,}, "
              f"..., {max(offsets)}: +{offsets[max(offsets)]['count']:,}}}")
    
    return N, ndim, len(offsets)


def _sort_x_npy(xp, order):
    """
    按 order 数组重排 X.npy 的行。

    策略 (全 numpy 向量化, 零 Python 逐行循环):
      1. 顺序读原 X.npy, 按块分发到桶文件 (向量化分桶)
      2. 逐桶读取 → 桶内 argsort → 顺序写新文件
    峰值内存 ≈ CHUNK × ndim × 4 ≈ 190 MB
    """
    import struct as st
    with open(xp, "rb") as f:
        magic = f.read(6); ver = f.read(2)
        hdr_len = st.unpack("<H" if ver == b"\x01\x00" else "<I",
                            f.read(2 if ver == b"\x01\x00" else 4))[0]
        hdr_raw = f.read(hdr_len); data_offset = f.tell()
    hdr = eval(hdr_raw.decode("latin1").strip())
    N, ndim = hdr["shape"]; row_bytes = ndim * 4

    # inv[orig_row] = 排序后的新位置
    inv = np.empty(N, dtype=np.int64)
    inv[order] = np.arange(N, dtype=np.int64)

    BUCKET_SIZE = 50000
    n_buckets = (N + BUCKET_SIZE - 1) // BUCKET_SIZE
    tmp_dir = os.path.dirname(xp)
    bucket_paths = [os.path.join(tmp_dir, f"_sort_bucket_{b}.tmp") for b in range(n_buckets)]

    # 为每个桶打开文件
    bfs = [open(bp, "wb") for bp in bucket_paths]
    try:
        # Pass 1: 顺序读, 向量化分桶
        CHUNK = 10000
        with open(xp, "rb") as src:
            src.seek(data_offset)
            for start in range(0, N, CHUNK):
                end = min(start + CHUNK, N)
                nr = end - start
                raw = src.read(nr * row_bytes)
                if len(raw) < nr * row_bytes:
                    nr = len(raw) // row_bytes
                block = np.frombuffer(raw, dtype=np.float32).reshape(nr, ndim)
                new_positions = inv[start:start + nr]
                bucket_ids = (new_positions // BUCKET_SIZE).astype(np.int32)

                # 按桶分组写入 (每桶一次批量写)
                for b in range(n_buckets):
                    mask = bucket_ids == b
                    cnt = int(mask.sum())
                    if cnt == 0:
                        continue
                    pos_b = new_positions[mask].astype(np.int32)
                    data_b = block[mask]
                    # 向量化交错: [pos0|row0|pos1|row1|...]
                    record_size = 4 + row_bytes
                    buf = np.empty(cnt * record_size, dtype=np.uint8)
                    buf_view = buf.reshape(cnt, record_size)
                    buf_view[:, :4] = pos_b.view(np.uint8).reshape(cnt, 4)
                    buf_view[:, 4:] = data_b.view(np.uint8).reshape(cnt, row_bytes)
                    bfs[b].write(buf.tobytes())
                del block, raw
    finally:
        for bf in bfs:
            bf.close()
    del inv

    # Pass 2: 逐桶读取, 向量化排序, 顺序写
    tmp_path = xp + ".sorting"
    try:
        with open(tmp_path, "wb") as out:
            out.write(magic + ver)
            out.write(st.pack("<H" if ver == b"\x01\x00" else "<I", hdr_len))
            out.write(hdr_raw)

            for b in range(n_buckets):
                bp = bucket_paths[b]
                if not os.path.exists(bp) or os.path.getsize(bp) == 0:
                    if os.path.exists(bp): os.remove(bp)
                    continue
                with open(bp, "rb") as bf:
                    bucket_raw = bf.read()
                os.remove(bp)

                record_size = 4 + row_bytes
                n_rec = len(bucket_raw) // record_size
                # 向量化解析: 把整个桶看作结构化数组
                all_bytes = np.frombuffer(bucket_raw, dtype=np.uint8)
                del bucket_raw
                reshaped = all_bytes.reshape(n_rec, record_size)
                positions = reshaped[:, :4].copy().view(np.int32).ravel()
                data = reshaped[:, 4:].copy().view(np.float32).reshape(n_rec, ndim)
                del all_bytes, reshaped

                sort_idx = np.argsort(positions)
                out.write(data[sort_idx].tobytes())
                del data, positions, sort_idx

        os.replace(tmp_path, xp)
    except Exception:
        # 清理临时文件
        if os.path.exists(tmp_path): os.remove(tmp_path)
        for bp in bucket_paths:
            if os.path.exists(bp): os.remove(bp)
        raise


# ═══════════════════════════════════════
#  数据集验证
# ═══════════════════════════════════════

def validate_dataset(out_dir, quiet=False):
    """
    全面验证数据集完整性。
    
    返回: 错误列表 (空 = 验证通过)
    """
    errors = []
    
    def _check(cond, msg):
        if not cond:
            errors.append(msg)
    
    # 文件存在性
    required = ["X.npy", "y.npy", "months.npy", "metadata.json"]
    for fn in required:
        _check(os.path.exists(os.path.join(out_dir, fn)), f"缺少文件: {fn}")
    if errors:
        return errors
    
    # 加载
    y = np.load(os.path.join(out_dir, "y.npy"))
    months = np.load(os.path.join(out_dir, "months.npy"))
    with open(os.path.join(out_dir, "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    
    N = len(y)
    ndim = meta.get("n_features", 0)
    
    # 基本一致性
    _check(N > 0, "样本数为 0")
    _check(len(months) == N, f"months 长度 ({len(months)}) != y 长度 ({N})")
    _check(ndim > 0, "n_features 为 0")
    
    # X.npy shape
    xp = os.path.join(out_dir, "X.npy")
    x_size = os.path.getsize(xp)
    # npy header 大约 128-256 bytes, X data = N * ndim * 4
    expected_data = N * ndim * 4
    _check(x_size >= expected_data, f"X.npy ({x_size:,} bytes) < 期望 ({expected_data:,} bytes)")
    
    # 标签合理性
    unique_labels = set(y.tolist())
    _check(unique_labels <= {0, 1}, f"标签不是 {{0,1}}: {unique_labels}")
    n_mal = int((y == 1).sum())
    mal_ratio = n_mal / max(N, 1)
    _check(0.01 < mal_ratio < 0.99, f"恶意比例异常: {mal_ratio:.3f}")
    
    # 月份合理性
    unique_months = sorted(set(months.tolist()))
    _check(len(unique_months) >= 2, f"月份数太少: {unique_months}")
    _check(all(m > 0 for m in unique_months), f"月份包含非正值: {unique_months}")
    
    # 月份排序 (向量化)
    is_sorted = bool(np.all(np.diff(months) >= 0)) if N > 1 else True
    _check(is_sorted, "X/y/months 未按月排序 (finalize_dataset 未执行?)")
    
    # 月份偏移量
    offsets = meta.get("month_offsets")
    if offsets:
        total = sum(v["count"] for v in offsets.values())
        _check(total == N, f"month_offsets 总计 ({total}) != N ({N})")
    else:
        errors.append("metadata.json 缺少 month_offsets")
    
    # train/eval 月份
    _check("train_months" in meta, "metadata.json 缺少 train_months")
    _check("eval_months" in meta, "metadata.json 缺少 eval_months")
    
    # 快速校验: 随机抽 10 行检查非零 (用 seek, 不用 mmap)
    if N > 0 and ndim > 0:
        try:
            xp = os.path.join(out_dir, "X.npy")
            # 解析 npy header 获取 data_offset
            with open(xp, "rb") as f:
                f.read(6)  # magic
                ver = f.read(2)
                hl = struct.unpack("<H" if ver == b"\x01\x00" else "<I",
                                   f.read(2 if ver == b"\x01\x00" else 4))[0]
                f.read(hl)
                x_data_offset = f.tell()
            row_bytes = ndim * 4
            rng = np.random.RandomState(42)
            check_idx = rng.choice(N, min(10, N), replace=False)
            n_zero_rows = 0
            with open(xp, "rb") as f:
                for ci in check_idx:
                    f.seek(x_data_offset + int(ci) * row_bytes)
                    row = np.frombuffer(f.read(row_bytes), dtype=np.float32)
                    if np.all(row == 0):
                        n_zero_rows += 1
            _check(n_zero_rows < len(check_idx), f"抽样 {len(check_idx)} 行全为零向量")
        except Exception as e:
            errors.append(f"X.npy 读取失败: {e}")
    
    if not quiet and not errors:
        print(f"    [验证] ✓ N={N:,}, D={ndim}, {len(unique_months)} 月, "
              f"恶意率={mal_ratio:.1%}")
    
    return errors


def _quick_checksum(path, n_bytes=1024*1024):
    """快速 SHA256: 只读前 n_bytes"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(n_bytes))
    return h.hexdigest()[:16]

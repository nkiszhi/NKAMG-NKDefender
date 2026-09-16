"""
Data Loader — 自动发现、自动提取、自动分割
=============================================
自动处理:
  1. 扫描DATA_DIR，识别所有PE样本（支持多种目录结构）
  2. 按时间/月份自动分割训练集和评估集
  3. BinaryNinja提取特征并缓存到CACHE_DIR
  4. 构建视角特征矩阵 {view_name: (N, dim)}
"""
import os, sys, glob, json, hashlib, time
from collections import defaultdict
import numpy as np

# ── 路径引导（2026-09-16 目录整理：本模块自 CoDefenderC3/ 根移入 train/）──
# 本模块依赖仓库根的 config.py 与 feature_extraction/（后者被 engine/ 共用，留在根）。
# 这里做防御式引导，使本模块无论被谁 import 都能解析到仓库根。
_TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TRAIN_DIR)
for _p in (_REPO_ROOT, _TRAIN_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config


PE_EXTENSIONS = {".exe", ".dll", ".bin", ".sys", ".scr", ".pe", ".drv",
                 ".cpl", ".ocx", ".msi", ""}  # "" for extensionless


def _extract_fast_chunk(tasks):
    """多进程 worker: 提取一批文件的快速特征。
    关键: 直接 import 文件, 跳过 __init__.py (避免 import torch)。"""
    import importlib.util, os as _os, numpy as _np
    spec = importlib.util.spec_from_file_location(
        "extract_feature",
        _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                      "feature_extraction", "extract_feature.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ext = mod.BinjaExtractor.__new__(mod.BinjaExtractor)
    ext.bn = None
    ext._headless = True
    ok = 0
    fails = []          # [(pe_path, 原因字符串)]
    dir_created = set()
    for pe_path, npz_path in tasks:
        try:
            d = _os.path.dirname(npz_path)
            if d not in dir_created:
                _os.makedirs(d, exist_ok=True)
                dir_created.add(d)
            feats = ext.extract_fast(pe_path)
            if feats:
                _np.savez(npz_path, **feats)
                ok += 1
            else:
                fails.append((pe_path, "extract_fast 返回空特征"))
        except Exception as e:
            fails.append((pe_path, f"{type(e).__name__}: {e}"))
    return ok, fails


def _check_de_chunk(chunk):
    """检查一批 npz 是否含 D key — 不再使用, 保留签名兼容。"""
    raise NotImplementedError("deprecated")


def _is_pe_file(path):
    """Quick PE header check (MZ magic bytes)."""
    try:
        with open(path, "rb") as f:
            return f.read(2) == b"MZ"
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════
#  Auto-Discovery: find all PE samples in any directory
# ═══════════════════════════════════════════════════════════

def discover_samples(data_dir):
    """
    自动发现PE样本，支持:
      - malware/benign 子目录
      - positive/negative 子目录
      - train/test 子目录
      - metadata.jsonl 标注文件
      - 平铺目录（按MZ头识别PE文件）
    
    返回 [{path, label, month, family}] 列表。
    """
    samples = []

    # 1a. 优先使用 metadata.jsonl
    meta_path = os.path.join(data_dir, "metadata.jsonl")
    if os.path.exists(meta_path):
        print(f"  发现 metadata.jsonl，按标注加载...")
        with open(meta_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line.strip())
                p = rec.get("path", rec.get("file", rec.get("sha256", "")))
                if not os.path.isabs(p):
                    p = os.path.join(data_dir, p)
                samples.append({
                    "path": p,
                    "label": int(rec.get("label", rec.get("y", rec.get("malware", 0)))),
                    "month": int(rec.get("month", rec.get("appeared_month", 0))),
                    "family": rec.get("family", rec.get("avclass", "unknown")),
                })
        if samples:
            return samples

    # 1b. metadata.csv (build_rawpe.py 产出)
    csv_path = os.path.join(data_dir, "metadata.csv")
    if os.path.exists(csv_path):
        import csv as csv_mod
        print(f"  发现 metadata.csv，按标注加载...")
        with open(csv_path, encoding="utf-8") as f:
            reader = csv_mod.DictReader(f)
            for rec in reader:
                p = rec.get("path", rec.get("sha256", ""))
                if not os.path.isabs(p):
                    p = os.path.join(data_dir, p)
                samples.append({
                    "path": p,
                    "label": int(rec.get("label", 0)),
                    "month": int(rec.get("month", 0)),
                    "family": rec.get("family", "unknown"),
                })
        if samples:
            print(f"  metadata.csv: {len(samples)} 样本")
            return samples

    # 2. 子目录结构: malware/benign, positive/negative, 1/0
    label_dirs = [
        (["malware", "malicious", "positive", "mal", "1"], 1),
        (["benign", "clean", "negative", "ben", "goodware", "0"], 0),
    ]
    found_dirs = False
    for names, label in label_dirs:
        for name in names:
            subdir = os.path.join(data_dir, name)
            if os.path.isdir(subdir):
                found_dirs = True
                files = _scan_pe_files(subdir)
                for fp in files:
                    samples.append({"path": fp, "label": label, "month": 0, "family": "unknown"})
                print(f"  {subdir}: {len(files)} PE文件 (label={label})")

    # 3. train/test 子目录
    for split in ["train", "test"]:
        split_dir = os.path.join(data_dir, split)
        if os.path.isdir(split_dir):
            # Recursively scan for malware/benign under train/test
            for names, label in label_dirs:
                for name in names:
                    subdir = os.path.join(split_dir, name)
                    if os.path.isdir(subdir):
                        found_dirs = True
                        files = _scan_pe_files(subdir)
                        for fp in files:
                            samples.append({"path": fp, "label": label, "month": 0, "family": "unknown"})

    if found_dirs and samples:
        return samples

    # 4. 平铺目录: 用MZ头识别PE文件，全部标为未知
    print(f"  无子目录结构，扫描所有文件的MZ头...")
    files = _scan_pe_files(data_dir, check_header=True)
    for fp in files:
        samples.append({"path": fp, "label": -1, "month": 0, "family": "unknown"})
    print(f"  发现 {len(samples)} 个PE文件 (标签未知，需metadata.jsonl)")

    return samples


def _scan_pe_files(directory, check_header=False, max_depth=2):
    """递归扫描目录中的PE文件。"""
    results = []
    for root, dirs, files in os.walk(directory):
        # Limit depth
        depth = root[len(directory):].count(os.sep)
        if depth >= max_depth:
            dirs.clear()
            continue
        for f in files:
            fp = os.path.join(root, f)
            _, ext = os.path.splitext(f)
            if ext.lower() in PE_EXTENSIONS:
                if check_header:
                    if _is_pe_file(fp):
                        results.append(fp)
                else:
                    results.append(fp)
    return results


def assign_months(samples, n_months=12):
    """
    为没有月份信息的样本自动分配月份。
    策略: 按文件修改时间排序，均匀分到12个月。
    """
    no_month = [s for s in samples if s["month"] == 0]
    if not no_month:
        return
    # 按文件mtime排序
    for s in no_month:
        try:
            s["_mtime"] = os.path.getmtime(s["path"])
        except Exception:
            s["_mtime"] = 0
    no_month.sort(key=lambda s: s["_mtime"])
    per_month = max(1, len(no_month) // n_months)
    for i, s in enumerate(no_month):
        s["month"] = min(i // per_month + 1, n_months)
        del s["_mtime"]



# ═══════════════════════════════════════════════════════════
#  PELoader: Raw PE → BinaryNinja → 磁盘缓存 → 按月视角加载
# ═══════════════════════════════════════════════════════════

class PELoader:
    """原始 PE 样本加载器。
    
    缓存架构 (全部在磁盘, 零内存常驻):
      .cache/
        per_sample/{md5}.npz        ← BinaryNinja 单样本特征 (提取后持久化)
        per_month/m{月}_y.npy       ← 标签向量
        per_month/m{月}_{视角}.npy  ← 合并后的视角矩阵
    
    all_views(month) 只读 1 个 .npy / 视角, 而不是 N 个 .npz / 样本。
    """

    def __init__(self, data_dir=None, train_months=None):
        # 规范化 data_dir: 缓存键是 md5(样本绝对路径), 而路径字符串里的分隔符风格
        # (E:\x\y vs E:/x/y) 会改变 md5 → 同一数据集在不同 shell 下会互相看不到缓存,
        # 每次都要重新提取全部特征。这里统一成规范形式, 保证缓存键稳定可复用。
        self.data_dir = os.path.normpath(os.path.abspath(data_dir or config.DATA_DIR))
        self.n_workers = config.N_WORKERS
        self.samples = []
        self.train_months = list(train_months) if train_months else []
        self.eval_months = []
        self.month_idx = {}
        # 缓存目录: 从 data_dir 派生, 避免多数据集冲突
        self.cache_dir = os.path.join(self.data_dir, ".cache")
        self._sample_cache = os.path.join(self.cache_dir, "per_sample")
        self._bndb_cache = os.path.join(self.cache_dir, "bndb")
        self._month_cache = os.path.join(self.cache_dir, "per_month")

    def load(self):
        print(f"[PELoader] 扫描 {self.data_dir}")
        self.samples = discover_samples(self.data_dir)
        if not self.samples:
            raise RuntimeError(f"未在 {self.data_dir} 中发现任何PE样本")

        meta_path = os.path.join(self.data_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            tm = meta.get("train_months")
            if tm:
                self.train_months = list(tm)

        assign_months(self.samples)

        all_months = sorted(set(s["month"] for s in self.samples if s["month"] > 0))
        max_train = max(self.train_months) if self.train_months else 3
        self.train_months = [m for m in all_months if m <= max_train]
        self.eval_months = [m for m in all_months if m > max_train]

        self.month_idx = {}
        for i, s in enumerate(self.samples):
            m = s["month"]
            if m > 0:
                self.month_idx.setdefault(m, []).append(i)

        n_mal = sum(1 for s in self.samples if s["label"] == 1)
        n_ben = sum(1 for s in self.samples if s["label"] == 0)
        print(f"  共 {len(self.samples)} 样本 (恶意={n_mal}, 良性={n_ben})")
        print(f"  训练月份: {self.train_months}, 评估月份: {self.eval_months}")
        return self

    # ── BinaryNinja 提取: 两阶段架构 ──

    def _sample_cache_path(self, path):
        return os.path.join(self._sample_cache, hashlib.md5(path.encode()).hexdigest() + ".npz")

    def _bndb_cache_path(self, path):
        return os.path.join(self._bndb_cache, hashlib.md5(path.encode()).hexdigest() + ".bndb")

    def _is_valid_bndb(self, path):
        """检查 bndb 是否存在。"""
        return os.path.exists(self._bndb_cache_path(path))

    def _has_sample_cache(self, path):
        return os.path.exists(self._sample_cache_path(path))

    def extract_features(self, paths=None):
        """两路并行特征提取:
        
        快速路径 (纯 Python, 多进程并行): A/B/C/F/G/H → 覆盖 11/16 视角
        慢速路径 (BinaryNinja, 子进程并行): D/E → 覆盖剩余 5/16 视角
        
        断点续传:
          - .npz 存在 → 快速路径跳过
          - .npz 含 D01 key → D/E路径跳过 (真正检查npz内容)
        """
        from feature_extraction.extract_feature import BinjaExtractor

        if paths is None:
            paths = [s["path"] for s in self.samples]
        os.makedirs(self._sample_cache, exist_ok=True)

        # ── 快速路径: 纯 Python 提取 (A/B/C/F/G/H) ──
        need_fast = [p for p in paths if not self._has_sample_cache(p)]
        if need_fast:
            self._extract_fast_parallel(need_fast)
        else:
            print(f"  快速特征: 全部 {len(paths)} 个样本已缓存, 跳过")

        # ── D/E 特征 (BinaryNinja): 先生成 bndb, 再提取 ──
        need_binja = self._find_need_de(paths)
        if need_binja:
            os.makedirs(self._bndb_cache, exist_ok=True)
            ext = BinjaExtractor()

            # ── 阶段1: 生成 bndb ──
            bndb_done = self._load_log("bndb_done")
            bndb_fail = self._load_log("bndb_fail")
            need_bndb = [p for p in need_binja
                         if not self._is_valid_bndb(p)
                         and p not in bndb_done
                         and p not in bndb_fail]
            n_have_bndb = sum(1 for p in need_binja if self._is_valid_bndb(p))
            n_skip_fail = sum(1 for p in need_binja if p in bndb_fail)

            # 覆盖率 >98% → 剩余的标记为永久失败, 直接跳过
            if need_bndb and n_have_bndb > 0:
                coverage = n_have_bndb / len(need_binja)
                if coverage > 0.98:
                    print(f"  D/E 阶段1: bndb 覆盖率 {coverage:.1%} ({n_have_bndb}/{len(need_binja)}), "
                          f"标记 {len(need_bndb)} 个为永久失败")
                    self._append_log("bndb_fail", need_bndb)
                    need_bndb = []

            if need_bndb:
                skip_msg = []
                if n_skip_fail: skip_msg.append(f"{n_skip_fail} 已确认失败")
                skip_str = f" (跳过 {', '.join(skip_msg)})" if skip_msg else ""
                print(f"  D/E 阶段1 [生成bndb]: {len(need_bndb)} 个, {self.n_workers} 并行{skip_str}...")
                t0 = time.time()

                # 第一轮 batch_size=5
                ok_set, fail_reasons = ext.generate_bndbs_batch(
                    [(p, self._bndb_cache_path(p)) for p in need_bndb],
                    n_workers=self.n_workers, batch_size=5)
                self._append_log("bndb_done", ok_set)

                # 重试连坐的
                retry = [p for p in need_bndb
                         if p not in ok_set and not os.path.exists(self._bndb_cache_path(p))]
                if retry:
                    print(f"    第一轮: {len(ok_set)} 成功, {len(retry)} 连坐")
                    print(f"    重试 (batch_size=1): {len(retry)} 个...")
                    ok2, fail2 = ext.generate_bndbs_batch(
                        [(p, self._bndb_cache_path(p)) for p in retry],
                        n_workers=self.n_workers, batch_size=1)
                    ok_set.update(ok2)
                    fail_reasons.update(fail2)
                    self._append_log("bndb_done", ok2)
                    print(f"    重试: {len(ok2)} 救回, {len(retry)-len(ok2)} 仍失败")

                # 清理残留 bndb + 记录失败到日志
                final_fails = []
                for p in need_bndb:
                    if p not in ok_set:
                        bp = self._bndb_cache_path(p)
                        if os.path.exists(bp):
                            try: os.remove(bp)
                            except: pass
                        final_fails.append(p)
                self._append_log("bndb_fail", final_fails)

                if final_fails:
                    print(f"    失败 {len(final_fails)} 个 (已记录日志):")
                    for p in final_fails[:5]:
                        print(f"      ✗ {os.path.basename(p)}: {fail_reasons.get(p, '?')}")
                    if len(final_fails) > 5:
                        print(f"      ... 还有 {len(final_fails)-5} 个")
                print(f"    阶段1: {len(ok_set)}/{len(need_bndb)} ({time.time()-t0:.0f}s)")
            else:
                msg = []
                if n_skip_fail: msg.append(f"{n_skip_fail} 失败")
                extra = f" ({', '.join(msg)})" if msg else ""
                print(f"  D/E 阶段1: 全部 bndb 已存在{extra}")

            # ── 阶段2: bndb → D/E (持久化 Worker, 增量断点续传) ──
            have_bndb = [p for p in need_binja if self._is_valid_bndb(p)]
            no_bndb = [p for p in need_binja if not self._is_valid_bndb(p)]

            # 无 bndb 的样本永远不可能提取 D/E → 直接标记永久失败
            if no_bndb:
                self._append_log("de_fail", no_bndb)
                print(f"  D/E: {len(no_bndb)} 个无 bndb → 标记永久失败")

            if have_bndb:
                de_workers = self.n_workers
                print(f"  D/E 阶段2 [bndb→特征]: {len(have_bndb)} 个, {de_workers} 并行...")
                t0 = time.time()

                # 增量日志: 每个文件成功后立即写入 de_done.log (Ctrl+C 不丢进度)
                import threading as _th
                _log_fh = open(self._log_path("de_done"), "a", encoding="utf-8")
                _log_fh_lock = _th.Lock()
                def _on_ok(pe_path):
                    with _log_fh_lock:
                        _log_fh.write(pe_path + "\n")
                        _log_fh.flush()

                try:
                    ok_set, fail_reasons = ext.extract_de_from_bndb_batch(
                        have_bndb,
                        [self._bndb_cache_path(p) for p in have_bndb],
                        [self._sample_cache_path(p) for p in have_bndb],
                        n_workers=de_workers, on_ok=_on_ok)
                finally:
                    _log_fh.close()

                # 检测边界文件: npz 已有 D/E 但 Worker 未报告 OK
                edge_cases = [p for p in have_bndb
                              if p not in ok_set
                              and self._has_de(self._sample_cache_path(p))]
                if edge_cases:
                    self._append_log("de_done", edge_cases)
                    print(f"    边界文件: {len(edge_cases)} 个 (npz 已有 D/E, 已补记日志)")

                # 重试: 真正失败的文件 (npz 没有 D/E key)
                retry = [p for p in have_bndb
                         if p not in ok_set
                         and not self._has_de(self._sample_cache_path(p))]
                if retry:
                    print(f"    第一轮: {len(ok_set)} 成功, {len(retry)} 失败")
                    print(f"    重试: {len(retry)} 个...")
                    _log_fh2 = open(self._log_path("de_done"), "a", encoding="utf-8")
                    def _on_ok2(pe_path):
                        with _log_fh_lock:
                            _log_fh2.write(pe_path + "\n")
                            _log_fh2.flush()
                    try:
                        ok2, fail2 = ext.extract_de_from_bndb_batch(
                            retry,
                            [self._bndb_cache_path(p) for p in retry],
                            [self._sample_cache_path(p) for p in retry],
                            n_workers=de_workers, on_ok=_on_ok2)
                    finally:
                        _log_fh2.close()
                    ok_set.update(ok2)

                    # 重试后仍失败 → 标记永久失败, 下次不再尝试
                    still_fail = [p for p in retry if p not in ok2]
                    if still_fail:
                        self._append_log("de_fail", still_fail)
                        print(f"    重试: {len(ok2)} 救回, {len(still_fail)} 永久失败 (已记录)")
                    else:
                        print(f"    重试: 全部救回")

                print(f"    阶段2: {len(ok_set)}/{len(have_bndb)} ({time.time()-t0:.0f}s)")
            else:
                print(f"  D/E 阶段2: 无有效 bndb 可处理")

            # 最终统计
            de_final = self._load_log("de_done")
            de_fail_final = self._load_log("de_fail")
            n_de = sum(1 for p in paths if p in de_final)
            n_fail = sum(1 for p in paths if p in de_fail_final)
            print(f"  D/E 最终: 完成={n_de}, 永久失败={n_fail}, 总计={len(paths)}")
        else:
            print(f"  D/E: 全部已有, 跳过")

        # 总计
        n_npz = sum(1 for p in paths if self._has_sample_cache(p))
        de_done = self._load_log("de_done")
        de_fail = self._load_log("de_fail")
        n_de = sum(1 for p in paths if p in de_done)
        n_fail = sum(1 for p in paths if p in de_fail)
        print(f"  总计: npz={n_npz}/{len(paths)}, D/E完成={n_de}, D/E失败={n_fail}")

    def _extract_fast_parallel(self, paths):
        """纯 Python 特征提取 — 多进程并行。"""
        from concurrent.futures import ProcessPoolExecutor, as_completed

        total = len(paths)
        print(f"  快速提取 (多进程): {total} 个样本, {self.n_workers} workers...")
        t0 = time.time()

        # 预备: 构造 (path, npz_path) 对
        tasks = [(p, self._sample_cache_path(p)) for p in paths]

        ok_count = 0
        fails_all = []      # [(pe_path, 原因)]
        crashed_chunks = 0

        # 分块: 每个 worker 处理一批文件 (减少 IPC 开销)
        chunk_size = max(50, total // (self.n_workers * 4))
        chunks = [tasks[i:i+chunk_size] for i in range(0, total, chunk_size)]

        with ProcessPoolExecutor(max_workers=self.n_workers) as pool:
            futs = {pool.submit(_extract_fast_chunk, chunk): chunk for chunk in chunks}
            done_total = 0
            for fut in as_completed(futs):
                chunk = futs[fut]
                try:
                    ok, fails = fut.result()
                except Exception as e:
                    # 整个 chunk 抛异常 (worker 崩溃 / BrokenProcessPool / 内存不足):
                    # 不再静默丢弃, 改为在主进程内串行重跑该 chunk
                    crashed_chunks += 1
                    print(f"\n  [WARN] 批次 {len(chunk)} 个样本的子进程异常: "
                          f"{type(e).__name__}: {e}")
                    print(f"         改为主进程串行重跑该批次...")
                    try:
                        ok, fails = _extract_fast_chunk(chunk)
                    except Exception as e2:
                        print(f"  [WARN] 主进程重跑同样失败: "
                              f"{type(e).__name__}: {e2}")
                        ok, fails = 0, [(p, f"重跑失败 {type(e2).__name__}: {e2}")
                                        for p, _ in chunk]
                ok_count += ok
                fails_all.extend(fails)
                done_total += len(chunk)
                elapsed = time.time() - t0
                rate = done_total / max(elapsed, 0.01)
                print(f"\r    [{done_total}/{total}] {rate:.0f}/s "
                      f"(ok={ok_count}, fail={len(fails_all)})",
                      end="", flush=True)

        fail_count = len(fails_all)
        print(f"\n  快速提取完成: {ok_count}/{total} ({time.time()-t0:.0f}s)")
        if crashed_chunks:
            print(f"  [WARN] 有 {crashed_chunks} 个批次曾发生子进程异常 (已串行补偿)")
        if fail_count:
            # 按失败原因聚合, 便于定位是文件问题还是环境问题
            buckets = {}
            for p, why in fails_all:
                buckets.setdefault(why, []).append(p)
            print(f"  失败 {fail_count} 个, 原因分布:")
            for why, ps in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
                print(f"    - {len(ps):4d} 个: {why}")
                for p in ps[:3]:
                    print(f"        ✗ {os.path.basename(p)}")
                if len(ps) > 3:
                    print(f"        ... 还有 {len(ps)-3} 个")

    # ── 日志系统: 断点续传 ──

    def _log_path(self, name):
        """日志路径: .cache/per_sample/{name}.log"""
        return os.path.join(self._sample_cache, f"{name}.log")

    def _load_log(self, name):
        """读取日志 → set, O(1) 查找。"""
        path = self._log_path(name)
        if not os.path.exists(path):
            return set()
        with open(path, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())

    def _append_log(self, name, items):
        """追加到日志。items: set 或 list。"""
        if not items:
            return
        with open(self._log_path(name), "a", encoding="utf-8") as f:
            for item in items:
                f.write(str(item) + "\n")

    @staticmethod
    def _has_de(npz_path):
        """D/E 特征是否存在 — 检查主 npz 是否含 D 开头的 key。"""
        import zipfile
        try:
            with zipfile.ZipFile(npz_path, "r") as zf:
                return any(n.split(".")[0].startswith("D") for n in zf.namelist())
        except Exception:
            return False

    def _find_need_de(self, paths):
        """找需要 D/E 提取的样本。

        排除三类:
          1. de_done: D/E 已成功完成
          2. de_fail: D/E 提取确认失败 (重试后仍 crash)
          3. bndb_fail: bndb 生成失败 (无 bndb 就不可能有 D/E)
        首次启动: 单线程 zipfile 扫描 (~60s for 58k), 结果写入日志。
        后续启动: 读日志 + 抽样验证 (<1s)。
        """
        t0 = time.time()
        cache_paths = {p: self._sample_cache_path(p) for p in paths}

        # ── 读取所有排除日志 ──
        done = self._load_log("de_done")
        de_fail = self._load_log("de_fail")
        bndb_fail = self._load_log("bndb_fail")
        skip = done | de_fail | bndb_fail

        # ── 尝试信任日志 (后续启动 <1s) ──
        if len(done) > 100:
            import random
            check = [p for p in random.sample(list(done), min(100, len(done)))
                     if p in cache_paths][:20]
            if check:
                n_ok = sum(1 for p in check if self._has_de(cache_paths[p]))
                if n_ok == len(check):
                    need = [p for p in paths if p not in skip]
                    print(f"  D/E: {len(done)} 已完成, {len(de_fail)} D/E失败, "
                          f"{len(bndb_fail)} 无bndb, "
                          f"{len(need)} 需提取 ({(time.time()-t0)*1000:.0f}ms)")
                    return need
                print(f"  D/E: 日志抽检 {n_ok}/{len(check)}, 重建...")

        # ── 全量扫描 (仅首次) ──
        print(f"  D/E 扫描 {len(paths)} 个 npz (首次, 请等待)...")
        done_paths, need = [], []
        for i, p in enumerate(paths):
            if p in skip:
                continue  # 已完成/永久失败/无bndb, 跳过
            cp = cache_paths[p]
            if os.path.exists(cp) and self._has_de(cp):
                done_paths.append(p)
            else:
                need.append(p)
            if (i + 1) % 2000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 0.01)
                eta = (len(paths) - i - 1) / max(rate, 0.01)
                print(f"\r    [{i+1}/{len(paths)}] {rate:.0f}/s "
                      f"done={len(done_paths)} need={len(need)} ETA {eta:.0f}s",
                      end="", flush=True)

        # 写日志 (后续启动直接用)
        with open(self._log_path("de_done"), "w", encoding="utf-8") as f:
            for p in done_paths:
                f.write(p + "\n")

        elapsed = time.time() - t0
        print(f"\r  D/E 扫描完成: {len(done_paths)} 已有, "
              f"{len(de_fail)} D/E失败, {len(bndb_fail)} 无bndb, "
              f"{len(need)} 需提取 ({elapsed:.0f}s)")
        return need

    # ── 按月合并缓存: N 个单样本 .npz → 1 个 {视角}.npy ──

    def _month_view_path(self, month, view_name):
        return os.path.join(self._month_cache, f"m{month}_{view_name}.npy")

    def _month_y_path(self, month):
        return os.path.join(self._month_cache, f"m{month}_y.npy")

    def _has_month_cache(self, month):
        yp = self._month_y_path(month)
        try:
            # y.npy 存在且 > numpy header (128B) + 至少 1 个 int32
            #
            # 注意: 这个下限**不能**假设月样本数。曾经写成 200 字节
            # (= 128 + 18*4), 等价于"月样本数必须 >= 18"才认缓存有效。
            # 在那个阈值下, 任何"最小数据集"(每类十几个样本、月样本数 < 18)
            # 的 y.npy 都只有 128 + N*4 < 200 字节 → 缓存永远判定为无效 →
            # 每次 get_month()/all_views() 都重建一遍, 并在重建前删掉旧的
            # m{月}_*.npy (批量删除还可能被 safe-delete 钩子拦下 → 半残状态)。
            # 128 + 4 是真正的"非空"判据: 空数组 npy 刚好 128 字节。
            if os.stat(yp).st_size < 132:
                return False
        except OSError:
            return False
        # 检查至少有一个视角 .npy
        pattern = os.path.join(self._month_cache, f"m{month}_V*.npy")
        return bool(glob.glob(pattern))

    def _drop_unobtainable_views(self, active_views, paths, probe_n=3):
        """探测 .npz 里实际含有哪些 feature_id, 剔除本数据集根本取不到的视角。

        只影响"视角清单", 不丢弃样本; 保留的视角仍按严格模式校验 (任一缺失 → 丢样本)。
        结果按视角清单签名缓存, 避免每个月都重新探测。
        """
        if not hasattr(self, "_view_drop_cache"):
            self._view_drop_cache = {}

        sig = tuple(sorted(active_views))
        if sig in self._view_drop_cache:
            return self._view_drop_cache[sig]

        probe = set()
        n_probe = 0
        for p in paths:
            cp = self._sample_cache_path(p)
            if not os.path.exists(cp):
                continue
            try:
                with np.load(cp, allow_pickle=True) as npz:
                    probe |= set(npz.files)
            except Exception:
                continue
            n_probe += 1
            if n_probe >= probe_n:
                break

        if not probe:
            # 一个可探测的 npz 都没有 (首次运行时 per_month 尚未构建) → 原样返回
            self._view_drop_cache[sig] = active_views
            return active_views

        kept, dropped = {}, []
        for vn, binja_ids in active_views.items():
            if vn == "V1_byte" or (set(binja_ids) & probe):
                kept[vn] = binja_ids
            else:
                dropped.append(vn)

        if dropped:
            print(f"    [视角] 样本中取不到的视角已剔除 ({n_probe} 个 npz 探测): "
                  f"{', '.join(sorted(dropped))}")
            print(f"           原因通常是 BinaryNinja 不可用 (D/E 特征缺失)。"
                  f"样本不丢弃, 这些视角的模型不参与集成。")
            print(f"           保留 {len(kept)}/{len(active_views)} 个视角: "
                  f"{', '.join(sorted(kept))}")

        self._view_drop_cache[sig] = kept
        return kept

    def _build_month_cache(self, month):
        """合并单月所有样本的特征 → 按视角存储 .npy
        
        每个 .npz 只读一次, 一次性提取所有视角的数据。
        缺失任何视角的样本整个排除 (不零填充)。
        已知坏样本记录在 .cache/sample_skip.log, 下次直接跳过。
        """
        if self._has_month_cache(month):
            return
        os.makedirs(self._month_cache, exist_ok=True)

        # 清理旧的无效缓存 (空 y 或无视角)
        for old in glob.glob(os.path.join(self._month_cache, f"m{month}_*")):
            try: os.remove(old)
            except: pass

        samps = [self.samples[i] for i in self.month_idx.get(month, [])]
        if not samps:
            return
        paths = [s["path"] for s in samps]
        labels = [s["label"] for s in samps]

        missing = [p for p in paths if not self._has_sample_cache(p)]
        if missing:
            self.extract_features(missing)

        vg = config.VIEW_GROUPS
        INT_VIEWS = {"V1_byte", "V6_opcode_seq"}
        RAW_LEN = 32768

        # 确定需要构建的视角
        active_views = {}  # vn → binja_ids
        for vn, vi in vg.items():
            binja_ids = vi.get("binja_features", [])
            if vn == "V1_byte" or binja_ids:
                active_views[vn] = binja_ids

        # ── 剔除本数据集"结构上取不到"的视角 ──
        # D/E 视角 (V6_opcode_*, V6_func_embed, V7_graph) 依赖 BinaryNinja 反汇编;
        # BN 不可用时这些 feature_id 永远不会出现在 .npz 里。若仍按"缺视角就丢样本"
        # 的严格模式, 会把每一个样本都排除掉 (train_months 全空 → 无法训练)。
        # 这里改为把取不到的视角直接剔除 (对应模型不参与集成), 样本保留。
        if not getattr(config, "VIEWS_REQUIRE_ALL", False):
            active_views = self._drop_unobtainable_views(active_views, paths)

        # 收集所有需要的 feature IDs (避免加载不需要的大数组)
        needed_fids = set()
        for vn, binja_ids in active_views.items():
            if vn == "V1_byte":
                needed_fids.add("B01")
            else:
                needed_fids.update(binja_ids)

        # ── 读取已知坏样本日志 (持久化, 删 per_month 不影响) ──
        skip_log_path = os.path.join(self.cache_dir, "sample_skip.log")
        known_bad = set()
        if os.path.exists(skip_log_path):
            with open(skip_log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        known_bad.add(line.split("\t")[0])

        # 逐样本: 只加载需要的 keys
        view_rows = {vn: [] for vn in active_views}
        valid_y = []
        skipped = []
        new_bad = []  # 新发现的坏样本, 追加到日志

        for i, p in enumerate(paths):
            basename = os.path.basename(p)

            # 已知坏样本: 直接跳过, 不打开 npz
            if basename in known_bad:
                skipped.append((basename, "已知坏样本"))
                continue

            cp = self._sample_cache_path(p)

            if not os.path.exists(cp):
                skipped.append((basename, "无缓存文件"))
                new_bad.append(f"{basename}\t无缓存文件")
                continue

            # 按需加载: 只读取 needed_fids 中的 key
            feat = {}
            try:
                with np.load(cp, allow_pickle=True) as npz:
                    for fid in needed_fids:
                        if fid in npz:
                            feat[fid] = npz[fid]
            except Exception:
                skipped.append((basename, "npz读取失败"))
                new_bad.append(f"{basename}\tnpz读取失败")
                continue
            feat_keys = set(feat.keys())

            # 逐视角提取
            sample_rows = {}

            for vn, binja_ids in active_views.items():
                if vn == "V1_byte":
                    if "B01" in feat:
                        arr = feat["B01"].flatten()[:RAW_LEN].astype(np.int32)
                        row = np.zeros(RAW_LEN, dtype=np.int32)
                        row[:len(arr)] = arr
                    else:
                        row = np.zeros(RAW_LEN, dtype=np.int32)
                        try:
                            with open(p, "rb") as fh:
                                raw_bytes = fh.read(RAW_LEN)
                            arr = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.int32)
                            row[:len(arr)] = arr
                        except Exception:
                            pass
                    sample_rows[vn] = row
                else:
                    parts = []
                    for fid in binja_ids:
                        if fid in feat:
                            arr = feat[fid].flatten()
                            dt = np.int32 if vn in INT_VIEWS else np.float32
                            parts.append(arr.astype(dt))
                    if parts:
                        sample_rows[vn] = np.concatenate(parts)

            del feat

            # 严格模式: 缺任何视角 → 整个样本排除
            if len(sample_rows) == len(active_views):
                for vn in active_views:
                    view_rows[vn].append(sample_rows[vn])
                valid_y.append(labels[i])
            elif not feat_keys:
                skipped.append((basename, "npz为空"))
                new_bad.append(f"{basename}\tnpz为空")
            else:
                missing_views = sorted(set(active_views) - set(sample_rows))
                reason = "缺视角: " + ",".join(missing_views)
                skipped.append((basename, reason))
                new_bad.append(f"{basename}\t{reason}")

        # ── 追加新发现的坏样本到持久化日志 ──
        if new_bad:
            with open(skip_log_path, "a", encoding="utf-8") as f:
                for entry in new_bad:
                    f.write(entry + "\n")

        # 打印跳过统计
        n_known = sum(1 for _, r in skipped if r == "已知坏样本")
        n_new = len(new_bad)
        if skipped:
            if n_known > 0 and n_new == 0:
                # 全部是已知的, 简短打印
                print(f"    月 {month}: 跳过 {len(skipped)}/{len(paths)} 个已知坏样本")
            else:
                print(f"    月 {month}: 跳过 {len(skipped)}/{len(paths)} 个样本"
                      f" ({n_known} 已知, {n_new} 新发现):")
                for name, reason in [(n, r) for n, r in skipped if r != "已知坏样本"][:5]:
                    print(f"      ✗ {name}: {reason}")
                if n_new > 5:
                    print(f"      ... 还有 {n_new - 5} 个")

        if not valid_y:
            print(f"    月 {month}: 无有效样本, 跳过")
            return

        # 写磁盘 (只保存有实际数据的视角, 全零视角跳过)
        saved_views = []
        for vn, rows in view_rows.items():
            if not rows:
                continue
            max_d = max(r.shape[0] for r in rows)
            dt = np.int32 if vn in INT_VIEWS else np.float32
            mat = np.zeros((len(rows), max_d), dtype=dt)
            for i, r in enumerate(rows):
                mat[i, :len(r)] = r
            # 检查: 如果整个视角全是零 (所有样本都缺该特征), 不保存
            if np.any(mat != 0):
                np.save(self._month_view_path(month, vn), mat)
                saved_views.append(vn)
            del mat
        del view_rows
        if saved_views and len(saved_views) < len(active_views):
            missing = set(active_views) - set(saved_views)
            print(f"    月 {month}: {len(valid_y)} 样本, "
                  f"{len(saved_views)} 视角 (缺 {', '.join(sorted(missing))})")

        y = np.array(valid_y, dtype=np.int32)
        np.save(self._month_y_path(month), y)
        del y

    # ── 公共接口 ──

    def get_month(self, month):
        self._build_month_cache(month)
        yp = self._month_y_path(month)
        if not os.path.exists(yp):
            return np.array([]), np.array([])
        y = np.load(yp)
        return month, y  # 返回 month id (all_views 用它读缓存)

    def get_months(self, start, end):
        from concurrent.futures import ThreadPoolExecutor
        months = [m for m in range(start, end + 1) if m in self.month_idx]
        # 多线程预构建缓存
        with ThreadPoolExecutor(max_workers=max(1, min(4, len(months)))) as pool:
            pool.map(self._build_month_cache, months)
        y_parts = []
        for m in months:
            yp = self._month_y_path(m)
            if os.path.exists(yp):
                y_parts.append(np.load(yp))
        return months, np.concatenate(y_parts) if y_parts else np.array([])

    def all_views(self, month_or_samples):
        """加载单月的所有视角 (每视角 1 次磁盘读取)。
        
        参数: month (int) 或 samples (list of dicts, 兼容旧接口)
        """
        # 确定月份
        if isinstance(month_or_samples, int):
            month = month_or_samples
        elif isinstance(month_or_samples, list) and len(month_or_samples) > 0:
            if isinstance(month_or_samples[0], dict):
                month = month_or_samples[0].get("month", 0)
            else:
                month = 0
        else:
            month = 0

        if month == 0:
            return {}

        self._build_month_cache(month)

        views = {}
        vg = config.VIEW_GROUPS
        for vn in vg:
            fp = self._month_view_path(month, vn)
            if os.path.exists(fp):
                views[vn] = np.load(fp)
        return views



# ═══════════════════════════════════════════════════════════
#  EmberLoader: EMBER预提取特征 → 6视角组
# ═══════════════════════════════════════════════════════════

class EmberLoader:
    """EMBER-2018 预提取特征加载器 — 磁盘懒加载, 不将 X 加载到内存."""

    def __init__(self, data_dir=None, train_months=None):
        self.data_dir = data_dir or config.DATA_DIR
        self.train_months = list(train_months) if train_months else []
        self.eval_months = []
        self.y_all = None      # 标签 (小, 常驻 RAM)
        self.month_idx = {}
        # 磁盘懒加载状态
        self._x_path = None    # X.npy 文件路径
        self._x_offset = 0     # npy header 后数据起始偏移
        self._x_ndim = 2381    # 特征维度
        self._x_n = 0          # 样本数
        self._x_row_bytes = 0  # 单行字节数
        self._active_ranges = config.EMBER_RANGES  # 默认 v2

    def load(self):
        print(f"[EmberLoader] 加载 {self.data_dir}")

        # X.npy + y.npy 必须存在 (由 build_dataset + finalize_dataset 产出)
        xp = os.path.join(self.data_dir, "X.npy")
        yp = os.path.join(self.data_dir, "y.npy")
        if not os.path.exists(xp) or not os.path.exists(yp):
            raise FileNotFoundError(
                f"数据文件不存在: {xp}\n"
                f"请先运行 python build_dataset.py 构建数据集")
        self._setup_lazy_npy(xp)
        self.y_all = np.load(yp).astype(int)

        # ── 读取 metadata.json (必须存在) ──
        self._month_offsets = {}
        meta_path = os.path.join(self.data_dir, "metadata.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"数据目录缺少 metadata.json: {self.data_dir}\n"
                f"请运行 python build_dataset.py 构建数据集")
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        fr = meta.get("feature_ranges")
        if fr and isinstance(fr, dict):
            self._active_ranges = {k: tuple(v) for k, v in fr.items()}
            print(f"  特征范围: {meta.get('feature_type', '?')} ({len(self._active_ranges)} 组)")
        else:
            # 没有 feature_ranges → 根据维度确定 (MalMem 55d 等)
            self._active_ranges = config.get_ember_ranges(self._x_ndim)
            print(f"  特征范围: 从维度 {self._x_ndim} 确定 ({len(self._active_ranges)} 组)")
        tm = meta.get("train_months")
        if tm:
            self.train_months = list(tm)
        mo = meta.get("month_offsets")
        if mo:
            self._month_offsets = mo
            print(f"  月份索引: {len(mo)} 个月 (定点读取模式)")

        # 过滤未标注
        mask = self.y_all >= 0
        n_valid = int(mask.sum())
        if n_valid < len(self.y_all):
            print(f"  过滤未标注: {len(self.y_all)} → {n_valid}")
            valid_idx = np.where(mask)[0]
            self.y_all = self.y_all[mask]
            # 重建索引映射 (懒加载需要原始行号)
            self._valid_idx = valid_idx
        else:
            self._valid_idx = None  # 无需映射

        # 时间戳分割
        self._parse_timestamps()

        all_months = sorted(self.month_idx.keys())
        max_train = max(self.train_months) if self.train_months else 3
        self.train_months = [m for m in all_months if m <= max_train]
        self.eval_months = [m for m in all_months if m > max_train]

        n_mal = int((self.y_all == 1).sum())
        n_ben = int((self.y_all == 0).sum())
        print(f"  共 {len(self.y_all)} 标注样本 (恶意={n_mal}, 良性={n_ben})")
        print(f"  训练月份: {self.train_months}, 评估月份: {self.eval_months}")
        print(f"  X 磁盘懒加载: {self._x_path} ({self._x_n}×{self._x_ndim})")
        return self

    def _setup_lazy_npy(self, npy_path):
        """解析 X.npy header, 不加载数据"""
        import struct as st
        self._x_path = npy_path
        with open(npy_path, "rb") as f:
            magic = f.read(6)
            assert magic == b"\x93NUMPY"
            ver = f.read(2)
            if ver == b"\x01\x00":
                hdr_len = st.unpack("<H", f.read(2))[0]
            else:
                hdr_len = st.unpack("<I", f.read(4))[0]
            hdr_str = f.read(hdr_len).decode("latin1").strip()
            hdr = eval(hdr_str)
            self._x_offset = f.tell()
        shape = hdr["shape"]
        self._x_n = shape[0]
        self._x_ndim = shape[1] if len(shape) > 1 else 2381
        self._x_row_bytes = self._x_ndim * np.dtype(np.float32).itemsize

    def _read_x_rows(self, indices):
        """从磁盘读取 X 的指定行 — 顺序扫描+过滤, 零逐行 seek"""
        if self._valid_idx is not None:
            indices = self._valid_idx[indices]

        n = len(indices)
        if n == 0:
            return np.empty((0, self._x_ndim), dtype=np.float32)

        result = np.empty((n, self._x_ndim), dtype=np.float32)
        wanted = set(indices.tolist())
        # 建立 原始行号 → result 位置 的映射
        row_to_out = {}
        for out_i, row_i in enumerate(indices):
            row_to_out.setdefault(int(row_i), []).append(out_i)

        x_path2 = getattr(self, "_x_path2", None)
        n1 = getattr(self, "_x_n", 0) - getattr(self, "_x_n2", 0) if x_path2 else self._x_n
        SCAN_CHUNK = 20000  # 顺序读 20K 行/块 ≈ 190MB

        def _scan_file(fpath, offset, n_total, row_offset):
            """顺序扫描文件, 提取 wanted 中的行"""
            rb = self._x_row_bytes
            filled = 0
            with open(fpath, "rb") as f:
                f.seek(offset)
                for start in range(0, n_total, SCAN_CHUNK):
                    end = min(start + SCAN_CHUNK, n_total)
                    nr = end - start
                    raw = f.read(nr * rb)
                    if len(raw) < nr * rb:
                        nr = len(raw) // rb
                        if nr == 0:
                            break
                    block = np.frombuffer(raw, dtype=np.float32).reshape(nr, self._x_ndim)
                    for local_i in range(nr):
                        global_row = row_offset + start + local_i
                        if global_row in row_to_out:
                            for oi in row_to_out[global_row]:
                                result[oi] = block[local_i]
                            filled += len(row_to_out[global_row])
                    del block, raw
                    if filled >= n:
                        break  # 已找到所有需要的行

        _scan_file(self._x_path, self._x_offset, n1, 0)
        if x_path2:
            _scan_file(x_path2, 0, getattr(self, "_x_n2", 0), n1)

        return result

    def _parse_timestamps(self):
        self.month_idx = {}
        N = len(self.y_all)

        # 方法 0 (最可靠): 直接读 months.npy (由 build_dataset 保存, 与 X/y 完全对齐)
        months_path = os.path.join(self.data_dir, "months.npy")
        if os.path.exists(months_path):
            months = np.load(months_path).astype(int)
            if len(months) == N:
                for i, m in enumerate(months):
                    if m > 0:
                        self.month_idx.setdefault(int(m), []).append(i)
                if self.month_idx:
                    return
            # 长度不匹配 (可能 X_all 被过滤了), 继续其他方法

        # 方法 1: 读取 build_dataset 生成的单一 features.jsonl (已过滤, 索引对齐)
        single_meta = os.path.join(self.data_dir, "features.jsonl")
        if os.path.exists(single_meta):
            with open(single_meta, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= N:
                        break
                    try:
                        rec = json.loads(line)
                        month = rec.get("appeared_month", rec.get("month"))
                        if month:
                            self.month_idx.setdefault(int(month), []).append(i)
                    except Exception:
                        pass
            if self.month_idx:
                return

        # 方法 2: 分割的 train_features.jsonl + test_features.jsonl
        global_idx = 0
        for split in ["train", "test"]:
            meta = os.path.join(self.data_dir, f"{split}_features.jsonl")
            if not os.path.exists(meta):
                continue
            with open(meta, encoding="utf-8") as f:
                for line in f:
                    if global_idx >= N:
                        break
                    try:
                        rec = json.loads(line)
                        label = rec.get("label", rec.get("y", 0))
                        if int(label) < 0:
                            continue
                        month = rec.get("appeared_month", rec.get("month"))
                        if month:
                            self.month_idx.setdefault(int(month), []).append(global_idx)
                        global_idx += 1
                    except Exception:
                        global_idx += 1

        # 方法 3: 均匀分配到12个月
        if not self.month_idx:
            per_month = N // 12
            for m in range(1, 13):
                start = (m - 1) * per_month
                end = m * per_month if m < 12 else N
                self.month_idx[m] = list(range(start, end))

    def get_month(self, month):
        """读取指定月份的 X, y — 使用月份偏移量定点读取 (O(count) 非 O(N))"""
        # 优先使用偏移量 (终态化后的数据集)
        if self._month_offsets and str(month) in self._month_offsets:
            info = self._month_offsets[str(month)]
            start, count = info["start"], info["count"]
            if count == 0:
                return np.empty((0, self._x_ndim), dtype=np.float32), np.array([])
            X = self._read_x_contiguous(start, count)
            y = self.y_all[start:start+count]
            return X, y

        # 回退: 旧式索引 (未终态化的数据集)
        idx = self.month_idx.get(month, [])
        if not idx:
            return np.empty((0, self._x_ndim), dtype=np.float32), np.array([])
        idx = np.array(idx)
        idx = idx[idx < len(self.y_all)]
        return self._read_x_rows(idx), self.y_all[idx]

    def get_months(self, start_month, end_month):
        """读取连续月份范围的 X, y"""
        # 优先使用偏移量 (连续月份 = 一次连续读取)
        if self._month_offsets:
            first_start = None
            total_count = 0
            for m in range(start_month, end_month + 1):
                info = self._month_offsets.get(str(m))
                if info and info["count"] > 0:
                    if first_start is None:
                        first_start = info["start"]
                    total_count += info["count"]
            if first_start is not None and total_count > 0:
                X = self._read_x_contiguous(first_start, total_count)
                y = self.y_all[first_start:first_start+total_count]
                return X, y
            return np.empty((0, self._x_ndim), dtype=np.float32), np.array([])

        # 回退
        all_idx = []
        for m in range(start_month, end_month + 1):
            all_idx.extend(self.month_idx.get(m, []))
        if not all_idx:
            return np.empty((0, self._x_ndim), dtype=np.float32), np.array([])
        idx = np.array(all_idx)
        idx = idx[idx < len(self.y_all)]
        return self._read_x_rows(idx), self.y_all[idx]

    def _read_x_contiguous(self, start_row, count):
        """从 X.npy 读取连续行块 — 单次 seek + 单次 read, 最快"""
        offset = self._x_offset + start_row * self._x_row_bytes
        nbytes = count * self._x_row_bytes
        with open(self._x_path, "rb") as f:
            f.seek(offset)
            raw = f.read(nbytes)
        assert len(raw) == nbytes, f"X.npy 读取不足: {len(raw)} < {nbytes}"
        return np.frombuffer(raw, dtype=np.float32).reshape(count, self._x_ndim).copy()

    def all_views(self, X):
        return _ember_views(X, self._active_ranges)


def _ember_views(X, active_ranges):
    """EMBER 特征向量 → 视角特征矩阵。
    
    参数:
        X: (N, D) 特征矩阵
        active_ranges: dict, 特征范围名 → (lo, hi) (来自 metadata.json)
    返回:
        views: dict, 视角名 → (N, dim) 特征矩阵
    """
    D = X.shape[1]
    views = {}
    for vn, vi in config.VIEW_GROUPS.items():
        if "ember_ranges" not in vi:
            continue
        range_names = vi["ember_ranges"]
        if range_names == "__ALL__":
            range_names = list(active_ranges.keys())

        parts = []
        for rn in range_names:
            if rn not in active_ranges:
                continue
            lo, hi = active_ranges[rn]
            if lo < D:
                parts.append(X[:, lo:min(hi, D)])
        if parts:
            views[vn] = np.hstack(parts).astype(np.float32)
    return views


# ═══════════════════════════════════════════════════════════
#  create_loader: 基于 metadata.json 选择加载器
# ═══════════════════════════════════════════════════════════

def create_loader(data_dir=None):
    """
    根据 metadata.json 创建数据加载器。

    metadata.json 必须包含 feature_type 字段:
      - "EMBER_FULL", "EMBER_V3", "MEMORY_VOLATILITY" → EmberLoader
      - "PE_RAW" → PELoader (需要 BinaryNinja)
    
    返回 (loader, feature_type_str)
    """
    d = data_dir or config.DATA_DIR
    meta_path = os.path.join(d, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"数据目录缺少 metadata.json: {d}\n"
            f"请先运行 python build_dataset.py 构建数据集")

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    feature_type = meta.get("feature_type", "")
    if not feature_type:
        raise ValueError(f"metadata.json 缺少 feature_type 字段: {meta_path}")

    print(f"\n[数据] 目录: {d}")
    print(f"[数据] 类型: {feature_type}")

    if feature_type == "PE_RAW":
        loader = PELoader(d)
    else:
        loader = EmberLoader(d)
    loader.load()
    return loader, feature_type

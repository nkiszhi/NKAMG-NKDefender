# CoDefenderC3 增量训练 + Checkpoint 机制技术实现方案

- 版本：v2.0（基线更正为项目根 `ensemble.pkl`）
- 日期：2026-09-15
- 基线模型：**`CoDefenderC3/ensemble.pkl`**（834.7 MB，K=33，V=15，33 个模型槽位全部有权重，等权 1/33）
- 基线结构：**已实测拆解**（见 §1.2），工具 `CoDefenderC3/train/_probe_ensemble.py`、`_verify_ensemble_load.py`

> ### 📁 2026-09-16 目录整理说明（**读本文前必看**）
>
> 训练侧脚本已从仓库根平铺状态统一移入 **`CoDefenderC3/train/`**，说明文档移入 **`CoDefenderC3/docs/`**。
>
> **两点影响**：
>
> 1. **命令路径**：文中涉及的 `run_all.py` / `train_minimal.py` / `build_minimal_dataset.py` /
>    `build_flat_dataset_meta.py` / `verify_trained_ensemble.py` / `_probe_ensemble.py` /
>    `_verify_ensemble_load.py` / `_inspect_ensemble.py` 现均在 `CoDefenderC3/train/` 下，
>    命令请从项目根执行并带该前缀。本文已就地更新可执行命令。
> 2. **行号漂移**：文中大量 `文件:行号` 引用（如 `data_loader.py:652`、`run_all.py:211-226`）
>    写于搬迁之前。搬迁时为这三个文件新增了「路径引导」段，**行号整体下移**：
>
>    | 文件 | 新增行数 | 换算 |
>    |---|---|---|
>    | `data_loader.py` | +9 | 旧 `:N` → 新 `:N+9`（`N ≥ 10`） |
>    | `run_all.py` | +9 | 旧 `:N` → 新 `:N+9`（`N ≥ 28`） |
>    | `train_minimal.py` | +1（另重排了 docstring 用法段） | 旧 `:N` → 新 `:N+1`（`N ≥ 54`） |
>    | 其余文件 | 0 | 行号不变 |
>
>    按**符号名搜索**（函数名/变量名）比按行号定位更可靠。

---

## 一、基线与目标

### 1.1 目标

| 编号 | 目标 | 验收标准 |
|---|---|---|
| G1 | 在根 `ensemble.pkl` 权重基础上加入新样本继续训练，**不从头重训** | 旧月份 AUC 掉点 < 1%，新月份 AUC 提升 |
| G2 | 引入训练进度检查点，训练中断可续 | 33 个模型训到第 31 个崩溃 → 重启只训剩余 2 个 |
| G3 | 引入模型版本快照（checkpoint 链），可审计、可回滚 | 任意历史版本一键切回，且**无需重启 uvicorn** |
| G4 | 权重与阈值随 checkpoint 走 | 引擎侧不再硬编码 `> 0.5` 与"等权平均" |

### 1.2 基线模型实测结构（探针输出，非推断）

| 项目 | 实测值 |
|---|---|
| 路径 / 大小 | `CoDefenderC3/ensemble.pkl` / 875,272,232 B（834.7 MB） |
| mtime | 2026-05-25 12:20:58 |
| 顶层 keys | `available_views, model_names, model_types, weights, view_indices, model_kwargs, models, `**`special_state`** |
| schema | **v1**：无 `meta`，无训练元数据（样本数/月份/git/阈值/血统全部缺失） |
| K / V | **33 / 15** |
| available_views | V1_byte, V2_byte_stat, V3_pe_struct, V4_import, V5_string, V6_opcode_seq, V6_opcode_stat, V6_func_embed, V7_graph, V8_gray, V8_color, V8_markov, V8_entropy, V9_hash, V10_metadata（缺 `V11_ensemble`） |
| weights | `shape=(33,) unique=1 sum=1.0` → **完全等权 1/33** |
| `models` 非空数 | **33/33**（0 个 `None`） |
| MalGraph | `models['MalGraph'] = 'SPECIAL_PLACEHOLDER'`（字符串），`model_types='special'` |
| `special_state` | `{'MalGraph': {'model_state_dict': {...}}}` — **当前 `save()` 不会写这个键** |
| `load()` 实测 | ✅ 成功，5.0 s，K=33/V=15，**32/33 模型成功实例化** |
| `predict_all` 实测 | ✅ 通过（2 行伪数据，1.7 s） |

**逐视角特征维度实测**（取自 `model_kwargs` 中 `fit` 阶段记录的实际 `input_dim`）：

| 视角 | 维度 | 来源特征 ID | 与 `extract_feature.py` 维度表核对 |
|---|---|---|---|
| V1_byte | 32768 | B01 | `MAX_BYTES` ✓ |
| V2_byte_stat | **776** | B02+B03+B04+B05 | 256+256+256+8 = 776 ✓ |
| V3_pe_struct | **264** | A01..A05 | 15+7+30+192+20 = 264 ✓ |
| V4_import | **339** | A06+A07+A08_A15 | 256+32+51 = 339 ✓ |
| V5_string | **553** | C01..C04 | 256+10+31+256 = 553 ✓ |
| V6_opcode_seq | 4096 | D01 | `MAX_OPCODES` ✓ |
| V6_opcode_stat | — | D02..D06 | 树模型，未记录 `input_dim` |
| V6_func_embed | 6144 | D07 | `MAX_FUNCTIONS*128` = 48×128 ✓ |
| V7_graph | 22 | E01..E04 | `GRAPH_STAT`=22 ✓ |
| V8_gray / markov / entropy | 65536 | F01 / F03 / F04 | 256×256 ✓ |
| V8_color | 196608 | F02 | 256×256×3 ✓ |
| V9_hash | 512 | G01..G04 | 128×4 ✓ |
| V10_metadata | 60 | H01..H03 | 20×3 ✓ |

### 1.3 结论：这是一个 **PE_RAW（BinaryNinja 48 特征）管线的完整 15 视角产物**

五个视角的维度同时精确吻合 `config.VIEW_GROUPS[*]["binja_features"]` 的维度求和，这是决定性证据——不是 EMBER 预提取特征（EMBER 口径应为 512 / 327 / 1280 / 104 / 158）。

**因此根模型包含 10 个 D/E 依赖模型，且它们的权重已训练：**

```
V6_opcode_seq(2: OpcodeLSTM, OpcodeTransformer) + V6_opcode_stat(1: OpcodeStatNet)
+ V6_func_embed(1: AsmEmbedNet) + V7_graph(6: MalGraph, CFGGAT, CFGGCN, CFGDGCNN,
  CallGraphGNN, GraphStatNet)                                     = 10 个
```

> **⚠️ 重要更正**：先前方案把基线写成 `results/CoDefenderC3_data/ensemble.pkl`（K=23 / V=11 / 706 MB），并断言"D/E 视角永久取不到"。实测表明两者是**两套不同的产物**，差集恰好是上面 10 个 D/E 模型（23 + 10 = 33）。根模型生成于 2026-05-25，当时 D/E 特征是取到的。
> 目前 `results/CoDefenderC3_data/` 目录已不存在，根 `ensemble.pkl` 是全仓唯一的集成模型文件。

### 1.4 当前数据环境实测

| 项目 | 实测值 |
|---|---|
| 数据目录 | `E:\nkproject\test_samples\CoDefenderC3_data` |
| 样本 | `benign/` 100 个（sha256 命名）；`malicious/2024/`、`malicious/2026/` |
| 元数据 | ❌ **`metadata.json` 与 `metadata.jsonl` 均不存在** |
| `.cache/per_month` | m1..m6，**每月 11 个视角**，每视角维度与根模型**逐一吻合** |
| 月度样本 | m1–m4 = 67（恶 50 / 良 17）、m5–m6 = 66（恶 50 / 良 16），**合计 400** |
| 类比 | **≈3:1 偏恶意** —— 与质量报告指出的 FPR 偏高根因一致 |
| `.cache/per_sample` | 703 个 npz |
| `.cache/bndb` | **0 个** → **当前环境 BinaryNinja 不可用** |
| `sample_skip.log` | 仅存 `.stale-20260914-164210`（已失效） |
| 现存 views vs 根模型 | 11 vs 15，**缺 V6_opcode_seq / V6_opcode_stat / V6_func_embed / V7_graph** |

---

## 二、现状硬约束盘点

| # | 约束 | 证据位置 | 对增量训练的影响 |
|---|---|---|---|
| C1 | `save()` 只存模型权重，无训练元数据；**且不透传未知顶层键** | `models/ensemble.py:498-520` | 无法审计/回滚；**重存会静默丢掉 `special_state`** |
| C2 | `saved_views != available_views` 时 `os.remove(model_path)` 重训 | `run_all.py:211-226` | 视角一有增减就丢模型 → 必须改为多态处理 |
| C3 | "numpy 引擎"模型全是**手写纯 NumPy 实现**（非 sklearn），`fit()` 内重置状态 | `DREBIN_2014/drebin.py:64`、`PEMiner_2009/peminer.py:186`、`EmberGBDT_2018/ember_gbdt.py:176`、`MicrosoftBIG_2016/ahmadi.py:164,300` | **33 个模型无一支持续训**，必须逐个加 `warm_start` |
| C4 | numpy 模型以 pickle 存**整个对象**，PyTorch 只存 `state_dict` | `models/ensemble.py:509-517` | 给 numpy 模型加新字段后，旧 pkl 反序列化实例必须用 `getattr(obj, x, default)` 读取 |
| C5 | `_has_month_cache()` 只检查 `y.npy` 存在 + 任一 `V*.npy` 存在，**不校验内容/行数/视角集合** | `data_loader.py:652-662` | ⚠️ 新样本并进已有月份会被**静默忽略** → 新样本必须落新月份号 |
| C6 | `assign_months()` 把 `month==0` 的样本按 mtime 均匀分 12 个月 | `data_loader.py:190-208` | ⚠️ 新样本漏写 `month` 会被塞进旧月份 → 触发 C5 |
| C7 | `sample_skip.log` 按 basename 永久跳过 | `data_loader.py:768-790` | 视角策略变化后必须清理 |
| C8 | 训练数据全量内存 `np.vstack`，单视角超 2 GB 统一子采样 | `run_all.py:236-302` | 增量若走全量重训，内存/耗时随月份线性增长 |
| C9 | PyTorch early stopping **只记 `best_loss`，从不恢复最优权重** | `models/ensemble.py:236-277` | 真 bug：实际用最后一轮权重。微调会放大 |
| C10 | 阈值 `0.5` 硬编码约 10 处 | `engine/app.py:478, 858, 879, 1399, 1708, 1824`、`models/ensemble.py:436` | 阈值必须入库 |
| C11 | `predict_all` 用 `self._model_names.index(mname)` 定位列 | `models/ensemble.py:360` | K 扩展时顺序变动会错位 |
| C12 | `MalGraph` 为 special：**`load()` 后 `_models` 里根本没有这个键** | 实测确认；`models/ensemble.py:548-566` 的 `elif` 链无 `special` 分支 | 推理时恒返回 `0.5` 常量，稀释集成分数 |
| C13 | 视角矩阵按"月份间补零对齐到最大维度"拼接 | `run_all.py:262-273` | 新月份维度更大时旧月份被补零；推理读单月维度小于模型 `input_dim` → 报错或静默降精度 |
| C14 | **引擎有自己的特征提取与聚合路径**：`_extract_views_for_batch` / `_build_batch_views` / `include_de` 开关 | `engine/app.py:1770-1815` | 与 PELoader 并行的第二条链路，需独立同步 |
| C15 | **`predict-directory` 路径不走 `ens.ensemble_predict()`**，而是自算 `valid_scores.mean(axis=1)` 并把 `> 0.5` 写死 | `engine/app.py:1817-1824` | ⚠️ **`ensemble.pkl` 里的 `weights` 存了也不生效** |
| C16 | 引擎的 `valid_k` 只判断 `_models.get(mname) is not None`，**不判断该视角本轮是否真的有数据** | `engine/app.py:1817` | ⚠️ `include_de=False` 时 10 个 D/E 模型的 `0.5` 常量被计入均值 → **集成分数被向 0.5 压缩约 30%** |
| C17 | `predict_all` 对缺视角的模型不产生任务，`_scores[k]` 保持初值 `0.5` | `models/ensemble.py:306-307, 353-361` | 同上，是 C16 的上游根因 |
| C18 | `per_month` 缓存**不记录样本路径清单**，只有 `y.npy` + 视角矩阵 | `data_loader.py:882-905` | metadata 一旦丢失，**无法从缓存反推月份归属** → 不能重算月份，只能冻结已有缓存 |
| C19 | `POST /reload-model?model_path=` 已支持模型热切换（带 `previous` 回滚信息） | `engine/app.py:1299-1352` | ✅ 现成的切版钩子，**优于重建 LATEST 文件** |
| C20 | 根 pkl 是**旧版代码产物**（含 `special_state`；`InceptionV3` 等 `model_kwargs` 为空） | 实测 | 只能"只读兼容"，不能假设与当前 `save()` 完全等价 |

---

## 三、总体设计

### 3.1 两条独立的 checkpoint 轴

| 轴 | 名称 | 粒度 | 生命周期 | 解决什么 |
|---|---|---|---|---|
| **C1** | 训练进度检查点 | 单个子模型 | 一次 run 内，成功后清理 | 断点续训（33 个模型不重来） |
| **C2** | 模型版本快照 | 整个集成 | 永久，跨 run | 血统 / 审计 / 回滚 / 灰度 |

### 3.2 数据流

```
新样本批次 (incoming/)
    │  ⓪ 前置体检: per_month 缓存自洽性 + metadata 恢复 + sha256 去重
    ▼
metadata.jsonl 追加 {path, label, month = 新月份号, family}
    │  ① 更新 metadata.json 的 train_months
    ▼
PELoader._build_month_cache(新月份)          ← 新月份号 → 缓存必然重建 (绕开 C5)
    │  ② 视角维度指纹校验 (C13): 与根模型实测维度逐一比对
    ▼
根 ensemble.pkl ──┐
                  ├─► warm_start_fit(新数据 ⨁ replay 旧数据)   ← C3 各家族续训
旧月份 .npy(replay)┘
    │  ③ 用留出月份重算权重 + 阈值 (C10/C12/C15/C16)
    ▼
checkpoint 快照 + manifest.json + gate 判定
    │  ④ 通过 → POST /reload-model 热切版 (C19)；不通过 → 归档不发布
    ▼
rebuild_calibration() → 失效 preds/scores 缓存 + SDD 重校准
```

### 3.3 目录规范

```
CoDefenderC3/
├── ensemble.pkl                        ← 生产加载路径, 保持不动 (engine/app.py:63)
├── checkpoints/
│   ├── manifest.json                   ← 账本: 全部 checkpoint 元数据 + latest
│   ├── LATEST                          ← 纯文本 checkpoint_id
│   ├── ckpt_0000_baseline_20260525/    ← 根模型回填为 0 号快照 (只登记 meta, 不复制 834 MB)
│   │   └── meta.json
│   ├── ckpt_0001_20260920T101500Z/
│   │   ├── ensemble.pkl
│   │   ├── ensemble.pkl.meta.json      ← 侧车: 供引擎毫秒级读阈值 (免反序列化 834 MB)
│   │   ├── meta.json
│   │   └── replay_pack.npz
│   └── ...
├── runs/<run_id>/partial/<model>.pkl    ← C1 训练进度检查点
└── results/<dataset>/cache/             ← 模型预测缓存 m{月}_{preds,scores,y}.npy
```

---

## 四、C2 模型版本快照

### 4.1 schema v2（向后兼容 + 未知键透传）

```python
SCHEMA_VERSION = 2

def save(self, path, meta=None, atomic=True, passthrough=None):
    """保存集成 (pickle)。

    passthrough: 从旧 pkl 读到的未知顶层键 (如 special_state), 原样带回,
                 否则重存会静默丢失 (C1/C20)。
    """
    state = {
        # ── 原有字段, 一字不改 ──
        "available_views": list(self.view_groups.keys()),
        "model_names": self._model_names,
        "model_types": self._model_types,
        "weights": self._weights,
        "view_indices": self._view_indices,
        "model_kwargs": self._actual_kwargs,
        "models": {},
        # ── 新增 ──
        "meta": {**(getattr(self, "_meta", None) or {}), **(meta or {})},
    }
    # ── 透传旧字段 (special_state 等) ──
    carry = dict(getattr(self, "_passthrough", None) or {})
    carry.update(passthrough or {})
    for k, v in carry.items():
        if k not in state:
            state[k] = v

    for mname, model in self._models.items():
        if model is None:
            state["models"][mname] = None
        elif self._model_types.get(mname) == "numpy":
            state["models"][mname] = model
        elif self._model_types.get(mname) == "pytorch" and HAS_TORCH:
            state["models"][mname] = model.state_dict()
        else:
            state["models"][mname] = None

    # special 模型: 保留占位符, 别让旧键值丢失 (C12)
    for mname, mtype in self._model_types.items():
        if mtype == "special" and mname not in state["models"]:
            state["models"][mname] = "SPECIAL_PLACEHOLDER"

    if not atomic:
        with open(path, "wb") as f:
            pickle.dump(state, f)
        return

    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)          # 834 MB 写到一半崩溃不会毁掉旧模型
```

`load()` 增加三行兜底：

```python
@classmethod
def load(cls, path):
    with open(path, "rb") as f:
        state = pickle.load(f)
    ...
    ens._fitted = True
    ens._meta = state.get("meta", {})
    ens._view_dims = (state.get("meta") or {}).get("view_dims", {})
    # 记录本次未消费的顶层键, 下次 save() 原样带回 (C1/C20)
    _known = {"available_views", "model_names", "model_types", "weights",
              "view_indices", "model_kwargs", "models", "meta", "mode", "view_keys"}
    ens._passthrough = {k: v for k, v in state.items() if k not in _known}
    return ens
```

> 实测根 pkl 走这条路径后，`_passthrough = {"special_state": {...}}` → 增量后重存不会丢 MalGraph 的权重载体。

### 4.2 meta 字段

```python
def build_meta(ens, *, parent_id=None, tag="", train_months=None,
               delta_months=None, n_samples=None, metrics=None,
               threshold=0.5, calibrator=None, view_dims=None, extra=None):
    import datetime, subprocess
    try:
        code = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        code = "unknown"
    m = {
        "schema_version": SCHEMA_VERSION,
        "parent_id": parent_id,
        "tag": tag,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "code_version": code,
        "train_months": list(train_months or []),
        "delta_months": list(delta_months or []),
        "n_samples": n_samples,
        "view_fingerprint": view_fingerprint(ens),
        "view_dims": dict(view_dims or getattr(ens, "_view_dims", {}) or {}),
        "config_snapshot": {k: getattr(config, k) for k in
                            ("SEED", "ALPHA", "DELTA_THR", "WS", "WR", "DACP_ALPHA")},
        "threshold": float(threshold),          # 引擎侧读这个 (C10/C15)
        "weight_mode": "auc_power",             # 供引擎判断聚合方式 (C15)
        "excluded_models": [],                  # 软剔除清单 (C12/C16)
        "frozen_views": [],                     # 本轮无新数据但保留权重的视角
        "pruned_views": [],                     # 确认结构上取不到的视角
        "calibrator": calibrator,
        "metrics": metrics or {},
    }
    if extra:
        m.update(extra)
    return m
```

### 4.3 账本 / 切版 / 保留

`models/checkpoint.py`（新增）：

```python
def promote(ens, ckpt_root, *, tag="", metrics=None, parent_id=None,
            train_months=None, delta_months=None, n_samples=None,
            threshold=None, calibrator=None, keep_last=2, publish_to=None,
            reload_url=None):
    """固化为不可变 checkpoint, 并按需热切换生产模型。"""
    os.makedirs(ckpt_root, exist_ok=True)
    man = _load_manifest(ckpt_root)
    cid = next_checkpoint_id(man)
    cdir = os.path.join(ckpt_root, cid)
    os.makedirs(cdir, exist_ok=True)

    meta = build_meta(...)
    pkl = os.path.join(cdir, "ensemble.pkl")
    ens._meta = meta
    ens.save(pkl, meta=meta)

    # 侧车: 引擎毫秒级读阈值, 不必反序列化 834 MB
    with open(pkl + ".meta.json", "w", encoding="utf-8") as f:
        json.dump({k: meta.get(k) for k in
                   ("checkpoint_id", "threshold", "weight_mode", "K", "V",
                    "excluded_models", "frozen_views", "pruned_views",
                    "train_months", "metrics")},
                  f, ensure_ascii=False, indent=2)
    with open(os.path.join(cdir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    man["checkpoints"].append({...})
    man["latest"] = cid
    _save_manifest(ckpt_root, man)
    with open(os.path.join(ckpt_root, "LATEST"), "w") as f:
        f.write(cid)

    if publish_to:
        publish(pkl, publish_to)              # 硬链接优先, EPERM/EXDEV 回退复制
    if reload_url:
        try:
            reload_engine(reload_url, publish_to or pkl)   # 见下
        except Exception as e:
            print(f"  [engine] 热切换失败({e}); 已落盘, 下次冷启动生效")

    _gc(ckpt_root, keep_last)
    return cid


def reload_engine(base_url, model_path, timeout=300):
    """调用引擎已有的 POST /reload-model (C19), 免重启 uvicorn。"""
    import urllib.parse, urllib.request
    url = "%s/reload-model?%s" % (base_url.rstrip("/"),
                                  urllib.parse.urlencode({"model_path": model_path}))
    req = urllib.request.Request(url, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode())
    print(f"  [engine] 热切换完成: K {body['previous']['model_count']}"
          f" → {body['current']['model_count']}, path={body['current']['model_path']}")
    return body
```

> **注意**：`/reload-model` 内部会 `MultiModelEnsemble.load(target_path)`（834 MB 反序列化，实测 5 s）+ 重置 `_SDD_ENGINE`，所以切版是"秒级但有 5 s 停顿"。建议低峰执行；`reload_url` 不传则仅落盘，下次冷启动生效。
> `reload_engine` 失败（引擎未启动）只打警告，**不回滚磁盘上的 checkpoint** —— 账本已是权威记录。

保留策略：单份 834.7 MB，**`keep_last=2` 即 1.67 GB**（原方案按 740 MB 估的 `keep_last=3` 偏乐观）。

---

## 五、C1 训练进度检查点

`fit()` 目前训完 33 个模型才 `save()` 一次，中途崩溃全丢。改造为"每个子模型完成即落盘"。

```python
def _dump_partial(self, run_id, mname):
    obj = self._models.get(mname)
    rec = {"name": mname,
           "type": self._model_types.get(mname),
           "kwargs": self._actual_kwargs.get(mname, {}),
           "obj": obj.state_dict() if (obj is not None and
                                       self._model_types.get(mname) == "pytorch") else obj}
    fp = os.path.join(self._run_dir(run_id), mname + ".pkl")
    tmp = fp + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(rec, f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, fp)

def _load_partials(self, run_id):
    done = {}
    for fp in glob.glob(os.path.join(self._run_dir(run_id), "*.pkl")):
        try:
            with open(fp, "rb") as f:
                rec = pickle.load(f)
            done[rec["name"]] = rec
        except Exception as e:
            print(f"  [resume] 忽略损坏分片 {os.path.basename(fp)}: {e}")
    return done
```

`fit(self, views, y, run_id=None, resume=False)` 内：
- numpy 任务列表先按 `done` 过滤，命中的直接 `self._models[mn] = _restore_partial(rec)`；
- PyTorch 循环里 `if mname in done: continue`；
- 每个模型完成后 `self._dump_partial(run_id, mname)`。

`run_all.py` 接入：

```python
run_id = os.environ.get("CODEFENDER_RUN_ID") or time.strftime("run_%Y%m%dT%H%M%S")
resume = os.environ.get("CODEFENDER_RESUME", "0") != "0"
ens.fit(views_tr, y_tr, run_id=run_id, resume=resume)
print(f"  [resume] 续训: CODEFENDER_RESUME=1 CODEFENDER_RUN_ID={run_id}")
```

> **顺带**：`_train_numpy` 用 `ThreadPoolExecutor(max_workers=4)` 跑纯 NumPy Python 双层循环，**GIL 不释放**（33 个模型里只有 `NatarajKNN` 用 sklearn 会释放）→ 实际无加速。建议按"是否使用 sklearn"分组，纯 NumPy 串行。

---

## 六、增量训练：数据侧

### 6.0 数据切分口径（**新增，修正原方案的时间假设**）

原方案按"逐月时间流评估"设计门禁（§11.3）与验证表（§14.3）。**实测发现：当前 m1..m6 并不携带任何时间信息**，该前提不成立。

| 事实 | 位置 / 证据 |
|---|---|
| 月份由**分层随机**生成 | `prepare_rawpe_dataset.py:92 stratified_months()`，`month = (i % n_months) + 1`，`month_assignment = "stratified-random(seed=42)"` |
| `train=m1..m3 / eval=m4..m6` 实质 | **随机 50/50 holdout**（分层仅保证正负比例一致） |
| 年度目录名**被代码忽略** | `discover_samples()` 分支 2 只做 `_scan_pe_files(malicious/)` 递归扫平，年度名不进 `month` / `family`，一律 `month=0, family="unknown"` |
| `family` 全库缺失 | 300 恶意 + 100 良性样本 `family` 一律 `"unknown"`，**无 AVClass 标签** |
| 跨年度无精确重复 | 300 个恶意样本 md5 **全唯一**，0 组重复；恶意与良性内容 0 重叠 |

**磁盘 mtime 实测（说明年度目录 ≠ mtime）**：

```
malicious/2024  n=200  mtime 全部 = 2024-12           ← 批次落盘时间，非采集时间
malicious/2026  n=100  mtime 2025-09 … 2026-07 递增    ← 唯一真实的逐月采集序列
benign/         n=100  mtime 2015-01 … 2025-11 跨度11年
```

**结论与修正**：

1. 现有 `train/eval` 划分度量的是**同分布随机泛化**，**不是**时间外推 → §11.3 门禁语义应改称"留出集 AUC"，**不要写"抗漂移"**。
2. `train_months` / `eval_months` 的**机制保留**（它是驱动划分的唯一开关，`data_loader.py:260-263`），但**语义需重新定义**。
3. 改为**双轨评估**，两轨回答不同问题，不可互相替代。

#### 双轨评估口径

| 轨道 | 切分方式 | 能证明 | 前提 / 风险 |
|---|---|---|---|
| **T 轨 · 时间外推** | 按真实时间戳（优先目录名，回退 mtime）派生季度 | 抗概念漂移、部署有效性 | 当前仅 2 个年度桶、良性无时间归属 → 须先补元数据 |
| **F 轨 · 家族隔离** | 按 AVClass `family` 分组后随机切 | 无泄漏的区分能力 | **当前 `family` 全 unknown，必须先打标签** |
| R 轨 · 分层随机（现状） | `stratified_months` | 仅可作快速冒烟 | 不得用于对外报告"抗漂移"结论 |

**T 轨为什么不彻底**：时间切分只是"家族隔离"的**弱代理**。同一家族 / 同一打包器 / 同一次批量编译的变体会随时间延续，仍可能跨 train / eval；长寿命家族（银行木马、远控）尤其如此。要真正量化泄漏，必须补 `family` 标签走 F 轨。

#### 按年度存储的必要性

**当前代码里年度存储的必要性是 0**（`discover_samples` 根本不读它）。但这不是"年度存储无必要"，而是**代码浪费了这个信号**。年度 / 季度目录本该承担四个职责：

1. **真实时间信号的唯一可靠载体**。mtime 会被拷贝、rsync、解压重置 —— `malicious/2024` 全部 mtime=2024-12 正是证据：那是批量落盘时间，不是采集时间。目录名是写死且可审计的。
2. **时间外推评估的前提**。年度是天然 holdout 单元，支撑"训练 ≤ 2025 / 测试 = 2026"这类可写进报告的口径。
3. **增量批次的物理单元**。新批次落成 `malicious/2026Q3/` → 天然派生新月份号（§6.2 铁律 1），**天然绕开 C5 静默忽略**。
4. **血统 / 复现 / 审计**。模型卡需要写明"训练集时间截止"。

**但年度粒度太粗，只适合存储分层、不适合评估切分**：只有 2024 / 2026 两桶 → 单次实验、方差极大、画不出逐月曲线。**评估切分应用季度或月。**

#### 落地改造（P1）

```python
# data_loader.py — discover_samples 分支 2 中解析年度/季度目录名
import re
YEAR_RE = re.compile(r"^(20\d{2})(?:[-_]?[Qq]([1-4]))?(?:[-_]?(\d{2}))?$")

def _temporal_from_path(rel_parts, fallback_mtime):
    """优先取目录名里的年/季/月；回退 mtime。返回 (year, month_of_year)。"""
    for p in rel_parts:
        m = YEAR_RE.match(p)
        if m:
            y = int(m.group(1))
            q = int(m.group(2)) if m.group(2) else None
            mm = int(m.group(3)) if m.group(3) else ((q - 1) * 3 + 1 if q else 1)
            return y, mm
    lt = time.localtime(fallback_mtime)
    return lt.tm_year, lt.tm_mon
```

配套：
- `temporal_index = year * 12 + (month - 1)` 作为**单调时间轴**，替代合成 month 号；
- `metadata.jsonl` 增补 `year` / `temporal_index` / `family` 三字段（`refresh_metadata_jsonl.py` 已保字段，扩展即可）；
- **良性样本必须补时间归属**，否则 T 轨立不起来（当前 `benign/` 100 个全无年度，仅 mtime 可用）；
- 补 `family`：跑 AVClass 或等价工具产出家族标签 —— 这是 F 轨的前置条件。

> ⚠️ 改造 `month` 语义会与既有 `.cache/per_month` 的 m1..m6 冲突。迁移时**先备份 `per_month` 目录**，或把新口径缓存落到独立命名空间（如 `per_tmonth/`），避免 C5 静默污染。

### 6.1 前置体检（**新增，且是必须的第一步**）

当前数据目录**没有 `metadata.json` / `metadata.jsonl`**，`create_loader()` 会直接抛 `FileNotFoundError`。而 `per_month` 缓存又不记录样本路径清单（C18），所以**不能靠重算月份来恢复** —— 重算会把样本塞到不同月份，与已存在的 m1..m6 缓存冲突，而 `_has_month_cache()` 根本发现不了（C5）。

新增 `verify_dataset_cache.py`，做三件事：

```python
# 基线维度指纹 (从根 ensemble.pkl 实测, 见 §1.2)
BASELINE_VIEW_DIMS = {
    "V1_byte": 32768, "V2_byte_stat": 776, "V3_pe_struct": 264, "V4_import": 339,
    "V5_string": 553, "V6_opcode_seq": 4096, "V6_func_embed": 6144, "V7_graph": 22,
    "V8_gray": 65536, "V8_color": 196608, "V8_markov": 65536,
    "V8_entropy": 65536, "V9_hash": 512, "V10_metadata": 60,
}

def audit(data_dir):
    """① 缓存自洽性 ② metadata 与缓存一致性 ③ 视角维度指纹"""
    cache = os.path.join(data_dir, ".cache", "per_month")
    issues = []
    for m in sorted(months_in_cache(cache)):
        y = np.load(os.path.join(cache, f"m{m}_y.npy"))
        for fp in glob.glob(os.path.join(cache, f"m{m}_V*.npy")):
            view = view_of(fp)
            shp = np.load(fp, mmap_mode="r").shape
            if shp[0] != len(y):
                issues.append(f"m{m}: {view} 行数 {shp[0]} != y 长度 {len(y)}")
            expect = BASELINE_VIEW_DIMS.get(view)
            if expect and shp[1] != expect:
                issues.append(f"m{m}: {view} 维度 {shp[1]} != 基线 {expect}")
    return issues
```

**实测结果**：m1..m6 的 11 个视角行数与 `y.npy` 全部一致，维度与基线逐一吻合 → 缓存可用，可跳过重建直接增量。

**metadata 恢复原则**：
- 用 `refresh_metadata_jsonl.py` 的**保月份**逻辑（按 basename 匹配、只改 path、保留 label/month/family）；
- **禁止**用 `prepare_rawpe_dataset.py`（它会重算月份）—— 除非同时删除 `per_month` 重建全部缓存；
- 若备份确实不存在，退路是：从 `benign/` + `malicious/` 目录结构重新生成 metadata，把旧样本月份固定为哨兵值，**只把新批次作为唯一增量月份**，并手工指定 `train_months`。

### 6.2 三条铁律

1. **新样本必须落到"新月份号"**（`month = max(现有月份) + 1 = 7`）。这是绕开 C5 唯一可靠的办法。
2. **必须显式写 `month` 字段**。漏写 → `assign_months()` 按 mtime 重新分配（C6）。
3. **路径写绝对路径**，否则 `data_loader.py:94-95` 按 `data_dir` 拼接，样本一搬就失效。

### 6.3 摄入脚本 `ingest_new_samples.py`

```
python CoDefenderC3/train/ingest_new_samples.py \
    --data-dir  "E:\nkproject\test_samples\CoDefenderC3_data" \
    --incoming  "E:\nkproject\test_samples\incoming_202609" \
    --label-from-subdir --month auto --dry-run
```

流程：扫描 → MZ 头校验 → **sha256 去重（对上 400 个已有样本 + 批次内自去重）** → 追加 `metadata.jsonl`（备份 + 原子替换） → 更新 `metadata.json` 的 `train_months`。

**去重是必需品**：重复样本会同时污染 (a) replay 池、(b) 权重计算用的留出集、(c) 新月份 AUC 指标，三处都会被高估。

### 6.4 平铺目录 + 1:1 配比：可行，但有两个静默陷阱（2026-09-15 实测）

**结论先行**：`malicious/*` + `benign/*` 各平铺一层、配比 1:1 —— **布局本身可行，且比年度嵌套更安全**；但按当前数据目录的现状（新样本 + 旧缓存）直接跑，会**静默产出废模型**。

#### 布局可行性（代码层核实）

| 检查项 | 结论 |
|---|---|
| label 推断 | `discover_samples()` 分支 2 按目录名匹配 `malicious`/`benign` → 自动 0/1 ✓ |
| 会被扫到吗 | `_scan_pe_files(subdir)`，`max_depth=2`，`if depth >= max_depth: dirs.clear(); continue`。平铺 depth=0 ✓；现用的 `malicious/2024/*` depth=1 ✓；**再加一层（如 `malicious/2024/samples/*`）depth=2 → 整层被跳过** ✗ |
| 无扩展名文件 | `PE_EXTENSIONS` 含 `""`，sha256 命名样本可被扫到 ✓ |
| MZ 校验 | 分支 2 调 `_scan_pe_files(subdir)` 时 **`check_header=False`** → 只按扩展名过滤，**不校验 MZ**。非 PE 的 `.bin` 会被收进数据集 ⚠️ |

#### 陷阱 1（致命）：不写 metadata.jsonl → 训练集可能 0 恶意

`discover_samples` 拿不到 month 就置 0，`assign_months()` 随后**按 mtime 排序均匀分月**。实测本数据集按 mtime 排序后的标签序列：

```
BBBBBBBBBBBBBBBBMMMMMMMMMMMMMMMM
前 16 个：恶 0 / 良 16      后 16 个：恶 16 / 良 0
```

原因：`benign/` 的 mtime 集中在 2024-10-08（批量落盘），`malicious/` 分散在 2025-10 → 2026-06 → **月份与标签完全共线**。无 metadata.json 时 `max_train` 兜底为 3，`train = m1..m3` **一个恶意样本都没有**，且**不报错**。

> 原 400 样本集用的是 `prepare_rawpe_dataset.py` 的 `stratified_months`（类内随机），所以没暴露这个问题。一旦走 `discover_samples` + `assign_months` 路径，它立刻出现。

#### 陷阱 2（同样静默）：陈旧 `per_month` 缓存吃掉新样本

实测 `E:\nkproject\test_samples\CoDefenderC3_data`：

| 项 | 实测 |
|---|---|
| 样本 | 32（恶 16 / 良 16，平铺） |
| `.cache/per_month` | **仍是旧的 m1..m6，缓存内 400 个样本** |
| `.cache/per_sample` | 703 个 npz，**键命中 0/32** |
| `.cache/bndb` | 0（空） |
| `per_sample` 占用 | **12 GB**（单个 18.4 MB） |

`_has_month_cache()` 只看文件是否存在 → 直接跑会**读旧 400 样本、完全无视这 32 个新样本**。

**处置**：`mv .cache/per_month .cache/per_month.stale-<ts>`（或脚本 `--purge-per-month`）。`per_sample` 的 703 个 npz 对新样本全是孤儿，可一并归档省 12 GB。

> **顺带一个高价值发现**：`E:\nkproject\test_samples\测试集\` 是原 400 样本 + 原 `.cache` 整体搬过去的。`per_sample` 缓存键是 `md5(样本绝对路径)` → **搬迁后 0/400 命中，12 GB npz 全部作废**。但 `per_month`（655 MB，m1..m6、400 样本、11 视角）**不记录路径，完全可复用** → **不要删它**，否则要重跑 400 个样本的 BN 特征提取。

#### 陷阱 3：1:1 会移动判定面，而阈值写死 0.5

- 集成是**等权平均 + 写死 `>0.5`**（`engine/app.py:1817-1824`）。配比从 3:1 改到 1:1，各模型分数分布整体平移 → **判定面跟着动** → 必须重扫阈值并重建校准器（§十）。
- **配比变更时不要 warm_start**：老模型 3:1 训的、新模型 1:1 训的，分数分布不齐，等权平均语义被破坏 → 走 **L2 全量重训**。
- **报告口径**：1:1 上算出的 FPR 是"50% 患病率下的 FPR"。真实部署恶意占比远低于 50%，须注明 base rate 或用先验修正 `PPV = TPR·π / (TPR·π + FPR·(1-π))`。

#### 样本量：16 vs 16 只够冒烟

良性 16 条 → 1 个误报 = **6.25% FPR**，分辨率不足以支撑任何质量结论。**要出结论至少 100 vs 100。**

#### 无家族标签时的三层降级方案（实测）

**L1 用现成元数据做伪家族**（项目已有 `scaner/staticinfo.py`）：

| 函数 | 实测（32 样本） | 能否做分组键 |
|---|---|---|
| `compute_imphash` | 28/32 成功，**28 个值全不重复**；失败的 4 个是**无导入表的 PE**（MZ 正常、authentihash 正常，导入目录为空） | ✅ 默认键 |
| `compute_authentihash` | **32/32 唯一** | ❌ 只能精确去重 |
| `compute_tlsh` | **26/32 可算（无 256KB 限流）**，但 `scaner/tlsh.py` 的 `Tlsh` 类**只有 `update/final/hexdigest`，没有 `diff()`** | ⚠️ 需自行补标准 TLSH 距离函数或装 `tlsh` 包 |
| `compute_ssdeep` | 受 `FUZZY_DB_MAX_BYTES`（默认 256KB）限流，**19/32 因超限被跳过** | ⚠️ 离线聚类须显式放开上限 |

**L2 相似度聚类**：TLSH/ssdeep 层次聚类切簇当伪家族。

**L3 诚实降级**：都不可行时，报告里**显式声明"未做家族隔离，指标为乐观上界"**，只用 T 轨做现实性证据。

> ⚠️ **分组 ≠ 标签**。分组键只需保证"同组不跨切分"，**不要求组内标签纯**。实测有个 4 成员组含 1 恶 3 良（都是 imphash 为空的 PE）—— 这不影响它当分组键，但说明它代表的是**"解析失败"而非"同家族"**，别拿它当家族标签。
>
> **本批次实测：伪家族分组"做了等于没做"** —— 32 个样本里 28 个 imphash 互不相同、无近重复。该机制在**含变体/重复的大数据集上才体现价值**（原 400 样本集更值得跑一次）。

#### 工具 `build_flat_dataset_meta.py`（`CoDefenderC3/train/`）

扫平铺目录 → 算分组键 → **按 `(label, 组)` 分层、组整体分配月份** → 打印「标签 × 月份」列联表 → 写 `metadata.jsonl`（备份 + 原子替换）+ `metadata.json`。

```bash
python CoDefenderC3/train/build_flat_dataset_meta.py --data-dir <dir> --months 4 --train-months 2 --dry-run
python CoDefenderC3/train/build_flat_dataset_meta.py --data-dir <dir> --months 4 --train-months 2 --purge-per-month
```

实测输出（32 样本 / 4 月）：每月 4 恶 4 良，训练集 8+8、评估集 8+8 —— **不再与标签共线**。

> 脚本会显式写出 `month`，因此**绕开了 `assign_months` 的 mtime 依赖**；同时写入 `capture_date`（mtime 推导，仅供参考）与 `family`（伪家族键），为将来 T 轨改造留字段。

---

## 七、增量训练：模型侧（warm_start）

### 7.1 续训能力矩阵

| 家族 | 模型（个数） | 续训机制 | 难度 | 关键陷阱 |
|---|---|---|---|---|
| 纯 NumPy GBDT | EmberGBDT_Import、OpcodeStatNet（2） | 保留 `trees`/`F0`，对残差**追加树** | ★★ | 新树学习率必须**独立缩放**，不能改 `self.learning_rate`（会把老树等比缩小） |
| 纯 NumPy RF | PEMinerRF、ResourceNet（2） | 保留 `trees`，只追加新树 | ★★ | RF 无"续训"；`rng` 种子必须随已有树数变化，否则新树与旧树 bootstrap 完全重复 |
| 纯 NumPy Linear SVM | DrebinSVM、DrebinImport、DrebinString（3） | 保留 `w`/`b` 继续 SGD，降 lr | ★ | Platt 校准必须**并入锚点 f 值**重拟合，否则概率整体偏移 |
| sklearn kNN | NatarajKNN（1） | `X_train = vstack([旧, 新])` | ★ | 原实现超 `max_train` 时全体随机采样 → **新样本可能被丢光** |
| PyTorch | 24 个 | 载入 `state_dict` + 小 lr 微调 | ★★ | 视角维度变化时首层 shape 不匹配；必须修 C9 |
| special | MalGraph（1） | 不参与（无分数） | — | 权重计算排除（C12） |

### 7.2 EmberGBDT / OpcodeStatNet

```python
def __init__(self, n_estimators=200, max_depth=8, learning_rate=0.05,
             subsample=0.8, min_samples_leaf=50,
             warm_start=False, new_tree_scale=0.5):
    self.warm_start = warm_start
    self.new_tree_scale = new_tree_scale
    self._tree_scales = []            # 向后兼容: 旧 pkl 实例缺此字段 (C4)

def _scales(self):
    s = getattr(self, "_tree_scales", None)
    if not s or len(s) != len(self.trees):
        s = [1.0] * len(self.trees)
        self._tree_scales = s
    return s

def _raw_predict(self, X):
    X = np.asarray(X, dtype=np.float32)
    F = np.full(X.shape[0], self.F0)
    for tree, sc in zip(self.trees, self._scales()):
        F += self.learning_rate * sc * tree.predict(X)
    return F

def fit(self, X, y):
    X = np.asarray(X, dtype=np.float32); y = np.asarray(y, dtype=np.float64)
    N = X.shape[0]; rng = np.random.RandomState(42)
    if self.warm_start and self.trees:
        F = self._raw_predict(X).astype(np.float64)   # F0 不变, 从旧模型原始分数出发
        n_new = self.n_estimators                     # 语义变为"本次新增树数"
    else:
        p_mean = np.clip(np.mean(y), 1e-6, 1 - 1e-6)
        self.F0 = float(np.log(p_mean / (1 - p_mean)))
        F = np.full(N, self.F0); self.trees = []; self._tree_scales = []
        n_new = self.n_estimators
    for m in range(n_new):
        p = np.clip(_sigmoid(F), 1e-10, 1 - 1e-10)
        residuals, hessians = y - p, p * (1 - p)
        n_sub = max(1, int(N * self.subsample))
        sub_idx = rng.choice(N, size=n_sub, replace=False)
        tree = GBDTRegressionTree(max_depth=self.max_depth,
                                  min_samples_leaf=self.min_samples_leaf,
                                  max_features=0.8, random_state=rng.randint(2**31))
        tree.fit(X[sub_idx], residuals[sub_idx], hessians[sub_idx])
        self.trees.append(tree)
        sc = self.new_tree_scale if (self.warm_start and m == 0) else 1.0
        self._tree_scales.append(sc)
        F += self.learning_rate * sc * tree.predict(X)
```

**为什么必须用 `_tree_scales`**：`_raw_predict` 对每棵树统一乘 `self.learning_rate`。若为了"温和学习新数据"把 `learning_rate` 从 0.1 降到 0.02，**老树贡献也被缩到 1/5，等于把旧模型打回欠训练状态**。只有每棵树独立 scale 才能只压新树。

### 7.3 PEMinerRF / ResourceNet

```python
def fit(self, X, y):
    X = np.asarray(X, dtype=np.float32); y = np.asarray(y, dtype=np.int64)
    N = X.shape[0]
    if not (self.warm_start and self.trees):
        self.trees = []
    # 种子随已有树数变化 —— 否则新树 bootstrap 与旧树完全相同, 增量无效
    rng = np.random.RandomState(42 + 1000 * len(self.trees))
    n_add = getattr(self, "n_new_trees", None) or self.n_estimators
    for t in range(n_add):
        boot_idx = rng.choice(N, size=N, replace=True)
        tree = CARTDecisionTree(max_depth=self.max_depth,
                                min_samples_leaf=self.min_samples_leaf,
                                max_features="sqrt",
                                random_state=rng.randint(2**31))
        tree.fit(X[boot_idx], y[boot_idx])
        self.trees.append(tree)
```

> `predict_proba` 是**所有树取平均**，所以新数据的表达力 ∝ `n_add / (旧树数 + n_add)`。基线是 30 棵树，建议 `n_add ≈ 0.3 × 30 ≈ 10`；要更明显需提高 replay 池中新样本占比。

### 7.4 DrebinLinear

```python
def __init__(self, C=1.0, lr=0.001, n_epochs=100, batch_size=256,
             warm_start=False, lr_decay=0.3):
    self.warm_start = warm_start; self.lr_decay = lr_decay

def fit(self, X, y, platt_anchor=None):
    if not (self.warm_start and self.w is not None):
        self.w = np.zeros(D, dtype=np.float64); self.b = 0.0
    lr = self.lr * (self.lr_decay if self.warm_start else 1.0)
    ...  # 原 SGD 循环, 用局部 lr 替换 self.lr
    f_new = self.decision_function(X)
    if platt_anchor is not None:                 # 并入锚点 f 值
        f_all = np.concatenate([platt_anchor[0], f_new])
        y_all = np.concatenate([platt_anchor[1], np.asarray(y)])
    else:
        f_all, y_all = f_new, np.asarray(y)
    self._platt_fit_from_f(f_all, y_all)         # 由 _platt_fit 拆出
```

> `DrebinSVM/DrebinImport/DrebinString` 是 numpy 对象整体 pickle，属性随对象保存。锚点只需把"旧月份样本的 `decision_function` 值 + 标签"存进 checkpoint 的 `replay_pack.npz`（每个 SVM 1 列 float32 × N，体积可忽略）。

### 7.5 NatarajKNN

```python
def fit(self, X, y):
    old_X = getattr(self, "X_train", None)
    if getattr(self, "_append", False) and old_X is not None:
        X = np.vstack([old_X, np.asarray(X, dtype=np.float32)])
        y = np.concatenate([getattr(self, "y_train"), np.asarray(y, dtype=np.int64)])
    if len(X) > self.max_train:
        rng = np.random.RandomState(42)
        n_new = 0
        if getattr(self, "_append", False) and old_X is not None:
            n_new = min(len(X) - len(old_X), self.max_train // 2)
        n_old = self.max_train - n_new
        idx_old = np.arange(len(old_X), dtype=np.int64)
        if n_old < len(idx_old):
            idx_old = rng.choice(idx_old, n_old, replace=False)
        idx_new = np.arange(len(old_X), len(X), dtype=np.int64)
        if n_new < len(idx_new):
            idx_new = rng.choice(idx_new, n_new, replace=False)
        keep = np.concatenate([idx_old, idx_new])
        X, y = X[keep], y[keep]
    self.X_train, self.y_train = X, y
```

> ⚠️ 基线 `NatarajKNN.X_train.shape = (36359, 776)` —— **已远超本数据集的 400 个样本**，说明它的训练点集来自另一批数据或历史累积。增量时若沿用"老样本优先欠采样"，会大幅丢弃这 36k 个点，**可能反而伤害性能**。**建议 kNN 默认冻结（`INCR_SKIP_MODELS` 含 `NatarajKNN`）**，观察后再决定。

### 7.6 PyTorch 24 个（统一函数）

把 `ensemble.py:186-284` 抽成 `_fit_pytorch(mname, cfg, X_view, y, *, init_state=None, lr, n_epochs, patience, run_id, freeze_prefixes)`，四个关键改动：

```python
    # ① 微调时绝不重新拉预训练权重
    if "pretrained" in init_params:
        kwargs["pretrained"] = False

    # ② layer-wise shape 过滤 (C13): 视角维度变化时只跳过不匹配的层
    if init_state is not None:
        cur = model.state_dict()
        ok, dropped = {}, []
        for k, v in init_state.items():
            if k in cur and tuple(cur[k].shape) == tuple(v.shape):
                ok[k] = v
            else:
                dropped.append(k)
        model.load_state_dict(ok, strict=False)
        print(f"      {mname}: warm-start 载入 {len(ok)}/{len(init_state)} 层"
              + (f", 跳过形状不匹配 {dropped}" if dropped else ""))

    # ③ 修复 C9: 真正保存并恢复最优权重
    if v_loss < best_loss - 1e-4:
        best_loss = v_loss
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        wait = 0
        if run_id:
            self._models[mname] = model
            self._dump_partial(run_id, mname)
    else:
        wait += 1
        if wait >= patience:
            break
    ...
    if best_state is not None:
        model.load_state_dict(best_state)      # ← 原代码漏了这一步
```

**④ `input_dim` 必须沿用 checkpoint 记录值，不能从当前月份重新推断。** 实测基线记录：`SaxeBerlinDNN/EmberDNN=776`、`ALOHANet=339`、`DL4MD=264`、`RaffFeatureNet=553`、`MetadataNet=60`、`CFGGAT/CFGGCN/CFGDGCNN/CallGraphGNN/GraphStatNet=22`、`GrayscaleCNN/MarkovCNN/EntropyMapCNN=65536`、`ColorCNN=196608`。一旦缓存重建导致维度变化就重新推断，会静默改变网络结构，让 `load_state_dict` 全层不匹配。

### 7.7 增量统一入口

```python
def warm_start_fit(self, views_new, y_new, *, replay_views=None, replay_y=None,
                   pt_lr=1e-4, pt_epochs=6, pt_patience=2,
                   gbdt_new_trees=40, rf_new_trees=12, knn_append=False,
                   skip_models=(), platt_anchors=None, run_id=None):
    for vname, mnames in self.view_groups.items():
        if vname not in views_new:
            print(f"    [冻结] 视角 {vname} 本轮无新数据, 保留原权重")
            continue                                # ★ 冻结, 不剔除
        X = views_new[vname]
        if replay_views and vname in replay_views and len(replay_views[vname]):
            X = np.vstack([replay_views[vname], X])
            y = np.concatenate([replay_y, y_new])
        else:
            y = y_new

        for mname in mnames:
            if mname in skip_models:
                print(f"    [跳过] {mname} (白名单排除)"); continue
            model = self._models.get(mname)
            if model is None:
                continue                            # special / MalGraph
            try:
                if self._model_types.get(mname) == "numpy":
                    model.warm_start = True
                    if mname == "NatarajKNN":
                        model._append = knn_append
                    if hasattr(model, "trees") and hasattr(model, "learning_rate"):
                        model.n_estimators = gbdt_new_trees
                    elif hasattr(model, "trees"):
                        model.n_new_trees = rf_new_trees
                    anchor = (platt_anchors or {}).get(mname)
                    if anchor is not None and hasattr(model, "platt_A"):
                        model.fit(X, y, platt_anchor=anchor)
                    else:
                        model.fit(X, y)
                    self._models[mname] = model
                elif self._model_types.get(mname) == "pytorch" and HAS_TORCH:
                    m = self._fit_pytorch(mname, get_model_config(mname), X, y,
                                          init_state=model.state_dict(),
                                          lr=pt_lr, n_epochs=pt_epochs,
                                          patience=pt_patience, run_id=run_id)
                    if m is not None:
                        self._models[mname] = m
                print(f"    {mname}: 增量完成")
            except Exception as e:
                print(f"    [警告] {mname} 增量失败, 保留原模型: {e}")
            if run_id:
                self._dump_partial(run_id, mname)

    self._fitted = True
    return self
```

**逐模型 try/except 是必需的**：24 个 PyTorch 模型里（IMCFN 25k 维输入、ColorCNN 196k 维输入）总会有 OOM 或形状异常，不能让单个失败中断整轮。

---

## 八、增量训练：集成侧

### 8.1 有效模型掩码（**P0，修 C16/C17**）

根因：`predict_all` 对缺视角的模型不产生任务，`_scores[k]` 保持初值 `0.5`。引擎的 `valid_k` 又只查 `_models.get(mname) is not None`，于是 **`include_de=False` 时 10 个 D/E 模型的 0.5 常量被计入等权平均** —— 集成分数被向 0.5 压缩约 30%（10/33）。

```python
def predict_all(self, views, return_valid_mask=False):
    ...
    _valid = np.zeros(self.K, dtype=bool)
    def _infer_one(mname, k, X_view):
        ...                                   # 成功产出分数 → _valid[k] = True
    ...
    if return_valid_mask:
        return _preds.T, _scores.T, _valid
    return _preds.T, _scores.T
```

引擎侧同步（`engine/app.py:1817-1824`）：

```python
valid_k = [k for k, mname in enumerate(ens._model_names)
           if ens._models.get(mname) is not None and valid_mask[k]]
```

> 这一条**不依赖任何增量训练改造**，可以立刻单独上线，收益立竿见影。

### 8.2 名字→索引映射（修 C11）

```python
def _rebuild_index(self):
    self._name_to_idx = {n: i for i, n in enumerate(self._model_names)}
    for vname, idxs in self._view_indices.items():
        self.view_groups[vname] = [self._model_names[i] for i in idxs]
```

`predict_all` 中 `self._model_names.index(mname)` → `self._name_to_idx[mname]`。K 扩展时顺序不再敏感。

### 8.3 视角四态处理（修 C2）

`run_all.py:211-226` 现在的两分支（一致 / 零填充）升级为四态：

| 态 | 条件 | 动作 | 风险 |
|---|---|---|---|
| **一致** | `saved == available` | 直接复用 | — |
| **扩展** | `available ⊃ saved` | `expand_views()`：为新视角追加模型，老模型与权重原样保留 | 低（K 增大，需重建 SDD 的 K） |
| **冻结** | `saved ⊃ available`（**默认**） | `freeze_views()`：保留模型与权重，本轮不增量，meta 记 `frozen_views` | 低。**这是根模型的当前场景**（15 vs 11） |
| **裁剪** | 显式 `--prune-views` | `prune_views()`：权重置 0 | **高**：根模型下会丢 10 个模型（30%），需二次确认 |

**根模型当前的真实场景就是"冻结"**：数据环境 11 视角、根模型 15 视角，缺 V6_opcode_seq / V6_opcode_stat / V6_func_embed / V7_graph。**默认冻结这 10 个模型（保权重、不置 0）**，只对 23 个可覆盖的模型做增量。

> ⚠️ **不要把"冻结"实现成"剔除"**。原方案把丢失视角一律 `weights = 0`，这在根模型上等于**自砍 30% 的模型能力**。冻结才是安全默认值：推理环境（Docker 镜像内可能装了 BN）仍能用上这 10 个模型。
> 真正该"裁剪"的只有 `V11_ensemble`（`binja_features: []`，结构上永远取不到），且它本来就不在根模型的 15 视角里。

```python
def freeze_views(self, lost_views):
    """本轮无新数据但保留历史权重 (安全默认)。"""
    ens = deepcopy(self)
    ens._meta["frozen_views"] = sorted(set(ens._meta.get("frozen_views", [])) | set(lost_views))
    return ens
```

### 8.4 权重重算（修 C12/C15）

```python
def update_weights_from_val(self, val_views, y_val, *, min_auc=0.55, power=2.0,
                            exclude=("MalGraph",), mask=None):
    from sklearn.metrics import roc_auc_score
    preds, scores, valid = self.predict_all(val_views, return_valid_mask=True)
    if mask is None:
        mask = valid
    w = np.zeros(self.K, dtype=np.float64)
    report = {}
    for k, name in enumerate(self._model_names):
        if name in exclude or not mask[k]:
            report[name] = {"auc": None, "weight": 0.0, "reason": "special/no-view"}
            continue
        s = scores[:, k]
        if not np.isfinite(s).all() or len(np.unique(s)) < 2:
            report[name] = {"auc": None, "weight": 0.0, "reason": "degenerate"}
            continue
        a = float(roc_auc_score(y_val, s))
        if a < 0.5:
            report[name] = {"auc": a, "weight": 0.0, "reason": "inverted"}; continue
        if a < min_auc:
            report[name] = {"auc": a, "weight": 0.0, "reason": "below_min_auc"}; continue
        w[k] = (a - 0.5) ** power
        report[name] = {"auc": a}
    self._weights = w / w.sum() if w.sum() > 0 else np.ones(self.K) / self.K
    self._excluded = [n for n, r in report.items()
                      if r.get("weight") == 0.0 and r.get("auc") is not None]
    print(f"  [权重] 重算完成; 软剔除 {len(self._excluded)}: {self._excluded}")
    return report
```

> **质量报告里的 5 个废模型是在旧 11 视角 / 23 模型那套上测的，不能直接套用到根模型**（那套目录现已不存在）。**上线前必须对根模型重新做一次质量重放**（§14.2），产出根模型自己的 FPR / 阈值扫描 / 单模型 AUC，再据此设 `min_auc`。

### 8.5 推理侧维度对齐（修 C13）

```python
def _align_view_dims(self, views):
    """把视角矩阵对齐到 checkpoint 记录的维度 (§1.2 实测值)。"""
    dims = getattr(self, "_view_dims", {}) or {}
    out = {}
    for v, X in views.items():
        tgt = dims.get(v)
        if tgt and X.shape[1] != tgt:
            if X.shape[1] < tgt:
                X = np.hstack([X, np.zeros((X.shape[0], tgt - X.shape[1]), dtype=X.dtype)])
                print(f"    [维度对齐] {v} → {tgt} (右侧补零)")
            else:
                print(f"    [维度告警] {v}: 实测 {X.shape[1]} > 模型 {tgt}, 截断")
                X = X[:, :tgt]
        out[v] = X
    return out
```

在 `predict_all` 开头调用一次。**根模型现在没有 `_view_dims`**（v1 schema），首次增量时由 `fit` 写入；对首次增量，用 §1.2 的实测值作为 `checkpoint.BASELINE_VIEW_DIMS` 兜底。

---

## 九、引擎侧接线

增量训练只有在**引擎真正用上**新权重与阈值时才有意义。三条必改：

| # | 位置 | 现状 | 改法 |
|---|---|---|---|
| E1 | `engine/app.py:1817-1824` | 自算 `valid_scores.mean(axis=1)` 等权平均 + `> 0.5` | 改调用 `ens.ensemble_predict(per_model_scores, ens._weights, thr)`，并叠加 `valid_mask`（§8.1） |
| E2 | `engine/app.py:478`、`858`、`879`、`1399`、`1708` | `pred = int(score > 0.5)` | 读 checkpoint 阈值 |
| E3 | 启动期 | 无阈值来源 | 读 `<MODEL_PATH>.meta.json` 侧车（**不要反序列化 834 MB 只为拿一个阈值**） |

```python
# engine/app.py
_ENS_THRESHOLD = 0.5

def _load_model_meta():
    """从侧车读阈值/权重模式, 毫秒级; 侧车缺失则回退 0.5。"""
    global _ENS_THRESHOLD
    sidecar = MODEL_PATH + ".meta.json"
    try:
        with open(sidecar, encoding="utf-8") as f:
            m = json.load(f)
        _ENS_THRESHOLD = float(m.get("threshold", 0.5))
        logging.info("模型侧车: ckpt=%s threshold=%.4f weight_mode=%s",
                     m.get("checkpoint_id"), _ENS_THRESHOLD, m.get("weight_mode"))
    except FileNotFoundError:
        logging.warning("无模型侧车 %s, 阈值回退 0.5", sidecar)
    except Exception:
        logging.exception("读取模型侧车失败, 阈值回退 0.5")
    return _ENS_THRESHOLD
```

并在 `/reload-model` 成功后也调用一次 `_load_model_meta()`（`engine/app.py:1352` 前后）。

---

## 十、校准链路重建

模型变了，`results/<dataset>/cache/m{月}_{preds,scores,y}.npy` 全部失效。当前只在"重训练"分支 `shutil.rmtree(cache_dir)`（`run_all.py:230-232`）—— **增量分支不会清**，会造成"新模型 + 旧预测"混用。

```python
def rebuild_calibration(loader, ens, cache_dir, *, drop_pred_cache=True):
    """只清模型相关缓存; 保留数据目录下的 per_month / per_sample (特征, 与模型无关)。"""
    if drop_pred_cache:
        for pat in ("m*_preds.npy", "m*_scores.npy", "m*_y.npy"):
            for fp in glob.glob(os.path.join(cache_dir, pat)):
                os.remove(fp)
    for m in loader.train_months:
        _predict_month(loader, ens, m, cache_dir)

    sdd = SDDEngine(ens.K, ens.perspective_groups(), alpha=config.ALPHA,
                    ws=config.WS, wr=config.WR)
    bl_p, bl_s, months = [], [], []
    for m in loader.train_months:
        pp, sp, yp = (os.path.join(cache_dir, f"m{m}_preds.npy"),
                      os.path.join(cache_dir, f"m{m}_scores.npy"),
                      os.path.join(cache_dir, f"m{m}_y.npy"))
        if all(map(os.path.exists, (pp, sp, yp))):
            bl_p.append(np.load(pp)); bl_s.append(np.load(sp)); months.append(m)
    if len(bl_p) < 2:
        raise RuntimeError(f"校准需 ≥2 个月数据, 仅 {len(bl_p)} 个月")
    sdd.calibrate(bl_p, bl_s)
    all_s = np.vstack(bl_s)
    sdd.calibrate_sample_baseline(all_s)
    all_y = np.concatenate([np.load(os.path.join(cache_dir, f"m{m}_y.npy")) for m in months])

    baselines = OrderedDict([
        ("Transcend", TranscendBaseline(sig=config.ALPHA)),
        ("CADE", CADEBaseline(latent_frac=0.5)),
        ("HCC", HCCBaseline()),
        ("DroidEvolver", DroidEvolverBaseline()),
        ("DREAM", DREAMBaseline()),
        ("MADCAT", MADCATBaseline()),
        ("Ens.Disagree", EnsembleDisagreementBaseline()),
    ])
    baselines["Transcend"].calibrate(all_s);  baselines["CADE"].fit(all_s)
    baselines["HCC"].calibrate(all_s);        baselines["DroidEvolver"].calibrate(all_s)
    baselines["DREAM"].fit(all_s, all_y);     baselines["MADCAT"].calibrate(all_s)
    baselines["Ens.Disagree"].calibrate(bl_p)

    last_m = loader.train_months[-1]
    return (sdd, baselines,
            np.load(os.path.join(cache_dir, f"m{last_m}_scores.npy")),
            np.load(os.path.join(cache_dir, f"m{last_m}_y.npy")))
```

### 阈值扫描入库（修 C10）

```python
def fit_threshold(scores, y, *, grid=np.arange(0.30, 0.86, 0.01)):
    from sklearn.metrics import f1_score
    best = None
    for thr in grid:
        p = (scores > thr).astype(int)
        tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
        fn = int(((p == 0) & (y == 1)).sum())
        rec = {"thr": float(thr), "f1": float(f1_score(y, p, zero_division=0)),
               "fpr": fp / max((y == 0).sum(), 1), "fp": fp, "fn": fn}
        if best is None or rec["f1"] > best["f1"]:
            best = rec
    return best
```

> ⚠️ **本次留出集的良性样本只有 16~17 条**。逐条错判就是 ±6% 的 FPR 跳变，阈值估计噪声极大。**建议先把良性样本补到 ≥60 条**再定阈值，否则"0.65 最优点"很可能不可复现。

---

## 十一、遗忘防护与门禁

### 11.1 三道防线

| 防线 | 措施 | 参数 |
|---|---|---|
| ① 数据回放 | 每视角从旧月份抽 anchor 子集与新数据混合 | `--replay-ratio 0.5`（新:旧 ≈ 1:0.5） |
| ② 学习率压制 | 新树 `new_tree_scale=0.5`；PyTorch `lr=1e-4`（原 1e-3 的 1/10）；SVM `lr_decay=0.3` | §7 |
| ③ 逐模型白名单 | 只对**该视角有数据**的模型增量；kNN 默认冻结（其 `X_train` 已 36,359 点，见 §7.5） | `INCR_SKIP_MODELS` |

### 11.2 replay 池

```python
def build_replay_pool(loader, views_target, *, months=None, per_month=24, seed=42):
    """按"每月等量分层抽样"构建 anchor 池, 写入 checkpoint 的 replay_pack.npz。"""
```

**按"每月 N 条"而不是"每视角总量 N 条"**：基线每月 67 条，`per_month=24` → 6 个月 × 24 = 144 条，覆盖全部月份且不让某月主导。

体积估算：13 个视角中最重的是 `V8_color`（196608 float32 = 768 KB/样本）→ 144 样本 ≈ 105 MB。`np.savez_compressed` 后预计 20–40 MB。**若嫌大，把视觉类视角（V8_*）的 replay 降到 `per_month=6`**，它们对遗忘不敏感。

### 11.3 门禁

> ⚠️ **术语修正（见 §6.0）**：当前月份是分层随机生成的，`old/new month AUC` 实质是"留出集 AUC"，**不度量时间外推**。此处变量名保留（改动太大），但**报告与结论措辞一律写"留出集"，禁止写"抗漂移 / 时间泛化"**，除非已按 §6.0 完成 T 轨改造。

```python
def gate(parent_metrics, child_metrics, *, max_old_auc_drop=0.01,
         min_new_auc_gain=0.0, max_pruned_models=0):
    reasons, passed = [], True
    for m, pm in parent_metrics.get("per_month", {}).items():
        cm = child_metrics.get("per_month", {}).get(m)
        if pm is None or cm is None: continue
        drop = pm - cm
        ok = drop <= max_old_auc_drop
        reasons.append(("  ✓" if ok else "  ✗") + f" 旧月 {m} AUC 变化 {-drop:+.4f}")
        passed &= ok
    for m, cm in child_metrics.get("delta_month", {}).items():
        gain = cm - parent_metrics.get("delta_month", {}).get(m, 0.0)
        if gain < min_new_auc_gain:
            passed = False
            reasons.append(f"  ✗ 新月份 {m} AUC 未提升 ({cm:.4f})")
    n_pruned = len(child_metrics.get("pruned_models", []))
    if n_pruned > max_pruned_models:
        passed = False
        reasons.append(f"  ✗ 裁剪了 {n_pruned} 个模型 > 上限 {max_pruned_models}")
    return passed, reasons
```

**新增硬门禁**：`frozen_views` 导致的**无新数据模型数**必须显式打印。对根模型而言这个数是 **10**（`frozen_views == {V6_opcode_seq, V6_opcode_stat, V6_func_embed, V7_graph}`），不能静默通过。

**门禁不通过时**：checkpoint 仍归档，但**不 `publish`、不调 `/reload-model`**，并给出建议（提高 replay-ratio、降 lr、补 D/E 特征、或转 L2 全量重训）。

### 11.4 两档执行模式

| 模式 | 触发 | 覆盖范围 | 耗时量级 |
|---|---|---|---|
| **L1 warm_start** | 常规每周/每月新增 | 只构建新月份缓存 | 低 |
| **L2 full retrain** | 视角变化 / 每 N 次 / 门禁连续失败 | 全部月份重新拼接 | 高 |

L2 走 `CODEFENDER_FORCE_RETRAIN=1 python CoDefenderC3/train/run_all.py --single <data_dir>`，改动量为零。
**但注意**：`--single` 会把 `config.MODEL_PATH` 改写为 `results/<dataset>/ensemble.pkl`（`run_all.py:928`），**L2 产出不会自动覆盖根模型**，必须显式 `publish` 回来。这也是 L1 增量器**绝不能走 `--single` 入口**的原因 —— 否则加载与写出都指向别处。

---

## 十二、改造清单

| 文件 | 改动 | 规模 | 关键点 |
|---|---|---|---|
| `models/checkpoint.py` | **新增**：manifest / promote / publish / reload_engine / gc / `BASELINE_VIEW_DIMS` | ~220 行 | 侧车 meta.json + 硬链接回退 |
| `models/ensemble.py` | `save()`/`load()` 加 meta 与**未知键透传**；新增 `warm_start_fit`、`update_weights_from_val`、`_fit_pytorch`、`_dump_partial/_load_partials/_restore_partial`、`freeze_views`、`expand_views`、`prune_views`、`_rebuild_index`、`_align_view_dims`、`predict_all(return_valid_mask)`、`build_meta`、`fit_threshold`；`fit()` 加 `run_id/resume` | ~520 行 | 不动 `predict_all` 主流程，只加钩子 |
| `models/EmberGBDT_2018/ember_gbdt.py` | `warm_start` + `_tree_scales` | ~40 行 | 向后兼容 `getattr` |
| `models/MicrosoftBIG_2016/ahmadi.py` | `OpcodeStatNet`(GBDT) / `ResourceNet`(RF) 各加 warm_start | ~45 行 | 两个类 |
| `models/PEMiner_2009/peminer.py` | `warm_start` + 种子随树数变化 | ~15 行 | 种子必须变 |
| `models/DREBIN_2014/drebin.py` | `warm_start` + `lr_decay` + `_platt_fit_from_f` + `platt_anchor` | ~35 行 | 锚点合并 |
| `models/NatarajKNN_2011/nataraj_knn.py` | `_append` + 老样本优先欠采样（**默认关闭**） | ~25 行 | 见 §7.5 警示 |
| `run_all.py` | 视角四态替换 `os.remove`；`rebuild_calibration()` 抽函数；`fit(run_id, resume)`；checkpoint 落盘 | ~220 行 | 消除 314-368 行重复 |
| `engine/app.py` | 阈值从侧车读（5 处）；`predict-directory` 改用 `ensemble_predict` + `valid_mask`；`/reload-model` 成功后刷新阈值 | ~30 行 | E1/E2/E3 |
| `config.py` | `INCR_*` / `CKPT_*` / `ENS_WEIGHT_*` / `GATE_*` | ~30 行 | 全部 env 可覆盖 |
| `ingest_new_samples.py` | **新增**：摄入脚本 | ~210 行 | 去重 / 备份 / 原子写 |
| `verify_dataset_cache.py` | **新增**：前置体检 | ~140 行 | 缓存自洽 + 维度指纹 |
| `train_incremental.py` | **新增**：增量 CLI | ~330 行 | 体检→摄入→增量→门禁→发布 |
| `verify_incremental_ckpt.py` | **新增**：验证与门禁报告 | ~200 行 | 对标 `verify_trained_model.py` |
| `_probe_ensemble.py` | ✅ 本次已交付 | 已完成 | 容错拆解任意 ensemble.pkl |
| `_verify_ensemble_load.py` | ✅ 本次已交付 | 已完成 | 验证 load / 推理链路 |

### config.py 新增段

```python
# ═══ 增量训练 ═══
INCR_ENABLED          = os.environ.get("CODEFENDER_INCR_ENABLED", "0") != "0"
INCR_REPLAY_RATIO     = float(os.environ.get("CODEFENDER_INCR_REPLAY_RATIO", "0.5"))
INCR_REPLAY_PER_MONTH = int(os.environ.get("CODEFENDER_INCR_REPLAY_PER_MONTH", "24"))
INCR_PT_LR            = float(os.environ.get("CODEFENDER_INCR_PT_LR", "1e-4"))
INCR_PT_EPOCHS        = int(os.environ.get("CODEFENDER_INCR_PT_EPOCHS", "6"))
INCR_GBDT_NEW_TREES   = int(os.environ.get("CODEFENDER_INCR_GBDT_NEW_TREES", "40"))
INCR_RF_NEW_TREES     = int(os.environ.get("CODEFENDER_INCR_RF_NEW_TREES", "12"))
INCR_FULL_EVERY       = int(os.environ.get("CODEFENDER_INCR_FULL_EVERY", "4"))
INCR_SKIP_MODELS      = set(filter(None, os.environ.get(
    "CODEFENDER_INCR_SKIP_MODELS", "MalGraph,NatarajKNN").split(",")))

# ═══ 集成权重 ═══
ENS_WEIGHT_MIN_AUC  = float(os.environ.get("CODEFENDER_ENS_WEIGHT_MIN_AUC", "0.55"))
ENS_WEIGHT_POWER    = float(os.environ.get("CODEFENDER_ENS_WEIGHT_POWER", "2.0"))
ENS_WEIGHT_EXCLUDE  = set(filter(None, os.environ.get(
    "CODEFENDER_ENS_WEIGHT_EXCLUDE", "MalGraph").split(",")))

# ═══ 门禁 ═══
GATE_MAX_OLD_AUC_DROP = float(os.environ.get("CODEFENDER_GATE_MAX_OLD_AUC_DROP", "0.01"))
GATE_MAX_PRUNED       = int(os.environ.get("CODEFENDER_GATE_MAX_PRUNED", "0"))

# ═══ Checkpoint ═══
CKPT_KEEP_LAST = int(os.environ.get("CODEFENDER_CKPT_KEEP_LAST", "2"))   # 834.7MB/份
CKPT_LINK      = os.environ.get("CODEFENDER_CKPT_LINK", "1") != "0"
MAINT_URL      = os.environ.get("CODEFENDER_MAINT_URL", "http://127.0.0.1:8000")
```

---

## 十三、既有缺陷修复清单

| # | 缺陷 | 影响 | 修复 |
|---|---|---|---|
| **B1** | PyTorch early stopping 只记 `best_loss`，从不 `load_state_dict` 恢复（`ensemble.py:236-277`） | 实际用最后一轮（更差）权重；微调场景致命 | §7.6 ③ |
| **B2** | `ThreadPoolExecutor(4)` 跑纯 NumPy Python 循环（`ensemble.py:178`） | GIL 不释放 → 无加速反而有上下文切换开销 | 按是否用 sklearn 分组，纯 NumPy 串行 |
| **B3** ✅已修复(2026-09-15) | `MalGraph` 无分数却以 `0.5` 参与等权平均（`ensemble.py:306-307, 436`） | 稀释集成分数（向 0.5 拉近约 3%） | `ensemble.py` 新增 `valid_model_mask()`：special/skip/缺失槽位不参与聚合，权重在有效槽位重归一化；`ensemble_predict` 显式 weights 路径同样排除。引擎 predict-directory 的 `valid_k` 自算路径本就排除，不受影响 |
| **B4** | 引擎 `valid_k` 不校验视角是否有数据（`app.py:1817`）→ `include_de=False` 时 10 个 D/E 模型的 0.5 被计入均值 | **集成分数被向 0.5 压缩约 30%** | §8.1 |
| **B5** | 引擎 `predict-directory` 不走 `ensemble_predict`，自算等权 + 写死 0.5（`app.py:1817-1824`） | **`weights` 存了不生效**，权重机制形同虚设 | E1 |
| **B6** | `save()` 不透传未知顶层键 | 重存静默丢掉 `special_state`（MalGraph 权重载体） | §4.1 |
| **B7** | `data_loader._has_month_cache()` 用 `y.npy >= 200 字节` 作非空判据（`data_loader.py:652`） | 等价于"月样本数必须 ≥ 18"。**月样本数 < 18 时缓存被永久判为无效**，每次 `get_month()` 都重建并批量删旧文件（实测触发 safe-delete 钩子），训练日志会重复出现"逐模型训练" | **已修**：阈值改 `>= 132`（numpy header 128B + 1×int32）。见 §附 C.4 |
| **B8** | 训练侧给所有 PyTorch 模型都写 `_actual_kwargs["input_dim"]` | `MalConv`/`MalConv2`/`ByteTransformer` 是 RAW_BYTES（V1_byte）模型，`__init__` **不接受** `input_dim` → 产物 `load()` 时 `TypeError`，**候选 pkl 完全不可加载** | **已修**：仅当 `"input_dim" in inspect.signature(cls.__init__).parameters` 时写入。见 §附 C.5 |

**B4 / B5 与增量训练无关，可立即单独上线**，而且 B4 的收益可能比整个增量框架更快见效。

> **B7 的隐蔽性**：`>= 18 个样本/月` 在真实数据集（每月 66~67 样本）上永远成立，所以这个 bug **只在最小数据集 / 小批量增量场景暴露**。任何为增量训练准备的"小月份"都会踩到，必须在做增量前修掉。

---

## 十四、验证方案

### 14.1 三层验证

| 层 | 手段 | 通过标准 |
|---|---|---|
| 单元 | 各家族 `warm_start` 单向性断言：小数据训练 → 增量 → 断言旧月份预测不变或微变、新月份改善 | 单向性成立 |
| 集成 | `verify_incremental_ckpt.py`：加载 parent / child 两个 ckpt，在旧评估月 + 新月份上独立重放 | 旧月 AUC 掉点 < 0.01，新月份提升 |
| 端到端 | `体检 → 摄入 → 增量 → 门禁 → promote → /reload-model → 扫描 20 个样本` | 引擎扫描结果与离线一致，阈值生效 |

### 14.2 首选先做"零风险验证"：根模型质量重放

在动任何增量代码之前，先对根 `ensemble.pkl` 做一次独立质量重放（复用 `verify_trained_model.py` 的模式，从 `E:\...\.cache\per_month` 读 m1..m6，**用 11 视角数据喂 15 视角模型，靠 valid mask 只统计有效模型**）：

- 得到**根模型自己的** per-model AUC、AUC 曲线、阈值扫描、FPR；
- 确认质量报告里的 5 个废模型（ALOHANet 0.5072 / RaffFeatureNet 0.4931 / MetadataNet 0.4565 / DrebinImport 0.4204 / InceptionV3 0.2344）在 33 模型上是否依然成立；
- **这一步的输出是后续所有 `min_auc` / 阈值 / replay-ratio 参数的取值依据。**

### 14.3 输出表（设计）

```
数据集: CoDefenderC3_data
  基线: ckpt_0000_baseline_20260525 (K=33 V=15, 834.7MB)
  增量: ckpt_0001_20261120T101500Z (K=33 V=15, Δmonth=[7])

┌──────┬──────┬──────────────┬──────────────┬────────┬────────┐
│ 月份 │  N   │ 基线 AUC     │ 增量 AUC     │  Δ     │ 判定   │
├──────┼──────┼──────────────┼──────────────┼────────┼────────┤
│  1-3 │ 201  │ (训练集)     │ (训练集)     │   —    │ 参考   │
│  4   │  67  │ 0.9XXX       │ 0.9XXX       │ +0.00X │ ✓      │
│  5   │  66  │ 0.9XXX       │ 0.9XXX       │ -0.00X │ ✓      │
│  6   │  66  │ 0.9XXX       │ 0.9XXX       │ -0.00X │ ✓      │
│  7   │  NN  │  (无)        │ 0.9XXX       │   —    │ 新增   │
└──────┴──────┴──────────────┴──────────────┴────────┴────────┘
冻结视角: V6_opcode_seq, V6_opcode_stat, V6_func_embed, V7_graph → 10 个模型本轮无新数据
裁剪模型: 0
阈值: 0.50 → 0.6X (在 m6 留出集扫描; 良性仅 16 条, 置信度低)
门禁: PASSED / FAILED (逐条原因)
```

---

## 十五、风险与回滚

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| **数据目录无 metadata（当前实况）** | **已发生** | `create_loader()` 直接抛异常；重算月份会与 m1..m6 缓存冲突且无人察觉 | 先跑 `verify_dataset_cache.py`；用保月份的 `refresh_metadata_jsonl.py`；**禁** `prepare_rawpe_dataset.py` |
| 新样本误入旧月份（C5/C6 静默忽略） | **高** | 增量"跑了但没效果"，最难排查 | 摄入脚本强制校验 `month > max(现有月份)`；打印"新建/复用"月份清单 |
| **10 个 D/E 模型本轮无新数据被误当"裁剪"** | **高** | 自砍 30% 模型能力 | 冻结为默认（§8.3）；门禁打印 `frozen_views` 数量 |
| kNN 增量丢失 36k 训练点（§7.5） | 中 | NatarajKNN 性能下降 | 默认冻结并跳过 |
| 留出集良性样本过少（16 条） | **高** | 阈值 / FPR 估计噪声 ±6% | 先补良性样本到 1:1 且 ≥60 条 |
| 灾难性遗忘 | 中 | 旧样本判错率上升 | replay + lr 压制 + 门禁；连续失败转 L2 |
| 视角 / 维度漂移（C13） | 中 | 推理形状报错或静默降精度 | `view_fingerprint` + `_align_view_dims` |
| 磁盘被 834.7 MB × N 撑爆 | 中 | 写入失败 | `keep_last=2`；`--link`；可选 blobs 去重 |
| VMware 共享目录不支持硬链接 | 中 | `os.link` 报 EPERM/EXDEV | `publish()` 自动回退 `shutil.copy2` |
| `/reload-model` 期间 5 s 停顿 | 低 | 扫描请求阻塞 | 低峰执行；或只落盘、下次冷启动生效 |
| `deepcopy(ens)` 内存翻倍（834 MB 模型对象） | 中 | 冻结/扩展时 OOM | 改为原地修改，或先 `del` 未用视角再 deepcopy |

### 回滚 SOP（秒级）

```bash
cd CoDefenderC3
# 1) 看账本
python -c "import json;m=json.load(open('checkpoints/manifest.json',encoding='utf-8'));[print(c['id'],c['train_months'],c.get('metrics',{}).get('overall')) for c in m['checkpoints']]"
# 2) 方式 A: 热切换(不重启) —— 0 号基线就是根 ensemble.pkl 本身
curl -X POST "http://127.0.0.1:8000/reload-model"
# 3) 方式 B: 指向任意历史快照
curl -X POST "http://127.0.0.1:8000/reload-model?model_path=D:/VM_Share/CoDefenderC3_v8/CoDefenderC3/checkpoints/ckpt_0001_20261120T101500Z/ensemble.pkl"
```

> **0 号基线快照只登记 meta、不复制 834 MB 权重** —— 根 `ensemble.pkl` 本身就是那个版本的载体。`POST /reload-model`（不传 `model_path`）即原地重载根路径，等价于回滚到基线。

---

## 十六、实施顺序

| 阶段 | 内容 | 依赖 | 可独立验证 |
|---|---|---|---|
| **P0** | 修 B4/B5（引擎有效掩码 + 走 `ensemble_predict`）。**零风险、收益最大、与增量无关** | 无 | ✅ 同批样本扫描前后对比集成分数分布 |
| **P1** | 根模型质量重放（§14.2），产出基线 AUC / 阈值扫描 / 废模型清单 | P0 | ✅ 独立报告 |
| **P2** | 数据前置体检（`verify_dataset_cache.py`）+ 恢复 `metadata.json/.jsonl` | 无 | ✅ 体检报告无 issue |
| **P3** | 修 B1/B2/B6 + `models/checkpoint.py` + schema v2 + C1 分片检查点 | P2 | ✅ 断点续训演练 + 重存后 `special_state` 仍在 |
| **P4** | 6 个纯 NumPy 家族 `warm_start` | P3 | ✅ 单元级单向性断言 |
| **P5** | PyTorch 统一 `_fit_pytorch`（含形状过滤 + C9 修复） | P3 | ✅ 单模型微调对比 |
| **P6** | `ingest_new_samples.py` + 新月份缓存链路打通 | P4/P5 | ✅ 摄入 10 个样本跑通缓存 |
| **P7** | 集成侧：权重 / 阈值 / 四态视角 / 维度对齐 | P6 | ✅ 权重报告 + 阈值扫描 |
| **P8** | `rebuild_calibration()` 抽函数 + `train_incremental.py` CLI + 门禁 | P7 | ✅ 端到端跑通 |
| **P9** | 引擎侧 E2/E3 阈值接线 + `verify_incremental_ckpt.py` | P8 | ✅ 引擎扫描与离线一致 |
| **P10**（可选） | 内容寻址 blobs 去重 / kNN 增量开关 | P9 | ✅ 磁盘占用对比 |

### 首个增量批次的建议

**补良性样本到 1:1，单次 ≥ 60 条，落到月份 7。**

三个理由：
1. 基线类比 3:1 偏恶意（每月 50 恶 / 16~17 良），是 FPR 偏高的根因；
2. 留出集良性样本必须 ≥60 条，否则阈值与 FPR 估计不可信（§10、§14.3）；
3. 良性样本量大、增量效果最容易观测，适合作为流程首跑的可验证目标。

**若同时想恢复 15 视角**（让 10 个 D/E 模型也吃到增量），新批次必须带上 D/E 特征，即需要一台 BN 可用的机器跑特征提取（当前 `.cache/bndb` 为 0）。否则请接受"冻结 10 个模型"的结果，并把这一条写进门禁报告。

---

## 附 A：`train_incremental.py` CLI 骨架

> ⚠️ **本节是设计稿**：`train_incremental.py` / `ingest_new_samples.py` /
> `verify_incremental_ckpt.py` / `verify_dataset_cache.py` / `models/checkpoint.py`
> 均**尚未实装**。实装时统一放在 `CoDefenderC3/train/`（`models/checkpoint.py` 除外，
> 它属于 `models/` 包）。下面命令已按此约定写好路径前缀。

```bash
# 0) 前置体检 (必须先过)
python CoDefenderC3/train/verify_dataset_cache.py --data-dir "E:\nkproject\test_samples\CoDefenderC3_data"

# 1) 摄入新样本
python CoDefenderC3/train/ingest_new_samples.py \
    --data-dir "E:\nkproject\test_samples\CoDefenderC3_data" \
    --incoming "E:\nkproject\test_samples\incoming_202609" \
    --label-from-subdir --month auto --dry-run

python CoDefenderC3/train/ingest_new_samples.py \
    --data-dir "E:\nkproject\test_samples\CoDefenderC3_data" \
    --incoming "E:\nkproject\test_samples\incoming_202609" \
    --label-from-subdir --month auto

# 2) 增量训练 (L1 warm_start) —— 目标模型是根 ensemble.pkl
python -u CoDefenderC3/train/train_incremental.py \
    --target-model "D:\VM_Share\CoDefenderC3_v8\CoDefenderC3\ensemble.pkl" \
    --data-dir     "E:\nkproject\test_samples\CoDefenderC3_data" \
    --mode warm_start \
    --replay-ratio 0.5 --replay-per-month 24 \
    --pt-lr 1e-4 --pt-epochs 6 \
    --gbdt-new-trees 40 --rf-new-trees 12 \
    --val-month 7 \
    --tag "add-benign-batch-202609" \
    --promote --engine-url http://127.0.0.1:8000

# 3) 验证
python CoDefenderC3/train/verify_incremental_ckpt.py \
    --data-dir "E:\nkproject\test_samples\CoDefenderC3_data" \
    --parent ckpt_0000_baseline_20260525 \
    --child  ckpt_0001_20261120T101500Z \
    --out reports/incremental_0001.md

# 4) 周期性全量校正 (L2) —— --single 会把输出写到 results/<dataset>/
CODEFENDER_FORCE_RETRAIN=1 python CoDefenderC3/train/run_all.py --single "E:\nkproject\test_samples\CoDefenderC3_data"
#    之后需显式 publish 回根模型:
python -c "from models.checkpoint import publish; import config; publish('results/CoDefenderC3_data/ensemble.pkl', config.MODEL_PATH)"
```

> **长任务务必 `python -u`**：stdout 重定向时是块缓冲，非 `flush=True` 的 print 长时间不可见。33 个模型（含 24 个 PyTorch，每个 ≥6 epoch）全流程可能数小时。

## 附 B：本次交付的探针工具

| 脚本 | 用途 |
|---|---|
| `CoDefenderC3/train/_probe_ensemble.py` | 容错拆解任意 `ensemble.pkl`：顶层 keys / K / V / view_indices / weights / 逐模型输入维度 / 未解析类 / 与 `VIEW_GROUPS` 差异。缺模块不中断（自定义 `find_class` 回退占位类） |
| `CoDefenderC3/train/_verify_ensemble_load.py` | 验证当前 `MultiModelEnsemble.load()` 能否加载目标 pkl，并做一次伪数据前向。输出逐模型实例化结果、`_meta`/`_view_dims`、缺视角列 |
| `CoDefenderC3/train/_inspect_ensemble.py` | 只读拆解：逐模型输入特征约定 |

```bash
# 在项目根执行
PYTHONIOENCODING=utf-8 python -u CoDefenderC3/train/_probe_ensemble.py [pkl路径]
PYTHONIOENCODING=utf-8 python -u CoDefenderC3/train/_verify_ensemble_load.py [pkl路径]
```

> Windows 下须带 `PYTHONIOENCODING=utf-8`，否则非 ASCII 输出会崩。

---

## 附 C：最小可训练数据集 + 训练脚本（2026-09-15 实装并实跑验证）

> 目的：在**不触碰生产模型**的前提下，把"增量训练"整条链路用一个 30 秒级别的最小数据集跑通，
> 用来验证「数据构建 → 特征提取 → 视图对齐 → 逐模型训练 → checkpoint → 产物可加载/可热切换」
> 这条链路本身，而不是用来得到质量结论。

### C.1 三个脚本

> **2026-09-16 目录整理**：三个脚本均已移入 `CoDefenderC3/train/`。
> 下列命令已按搬迁后的真实 CLI 校正（旧文档里的 `--malicious-src/--benign-src/--out-dir/--n-per-class/--n-months` 是早期版本参数，现为 `--source/--out/--per-class/--months`；`train_minimal.py` 用 `--dataset` 而非 `--data-dir`）。

| 脚本 | 位置 | 职责 |
|---|---|---|
| `build_minimal_dataset.py` | `CoDefenderC3/train/` | 构建与基线视角严格对齐的最小数据集（取样 → imphash 分组 → 分组感知分月 → 复制 → 特征提取 → D/E 覆盖检查 → metadata/manifest → per_month → 四方校验） |
| `train_minimal.py` | `CoDefenderC3/train/` | 从最小数据集训练全部 33 个模型（可选 warm_start），**逐模型 checkpoint + `--resume` 断点续训** |
| `verify_trained_ensemble.py` | `CoDefenderC3/train/` | 候选 pkl 与基线的等价性全项校验（schema / K,V / available_views / view_indices / 槽位 / weights / 可加载性 / 伪数据前向） |

```bash
# 全部命令在项目根 D:\VM_Share\CoDefenderC3_v8 下执行

# 0) 构建最小数据集（16 恶 / 16 良，4 个月，train=[m1,m2] eval=[m3,m4]）
#    源目录要求：平铺一层，下含 malicious/ 与 benign/ 两个子目录
PYTHONIOENCODING=utf-8 python -u CoDefenderC3/train/build_minimal_dataset.py \
    --source "E:\nkproject\training_data0915" \
    --out    "D:\VM_Share\CoDefenderC3_v8\minimal_dataset" \
    --per-class 16 --months 4 --train-months 2 --views full

# 1) 训练（默认只写 CoDefenderC3/runs/<run_id>/，绝不碰生产 ensemble.pkl）
PYTHONIOENCODING=utf-8 python -u CoDefenderC3/train/train_minimal.py \
    --dataset  "minimal_dataset" \
    --run-id   run-example \
    --mode scratch --epochs 6
#   断点续训：把同一条命令重跑一次，自动从 runs/<id>/partial/ 续上
#   想覆盖生产模型需显式 --in-place（会先备份）

# 2) 复验（必做门禁）
PYTHONIOENCODING=utf-8 python -u CoDefenderC3/train/verify_trained_ensemble.py \
    --candidate CoDefenderC3/runs/run-example/ensemble.pkl
#   --baseline 默认 CoDefenderC3/ensemble.pkl
```

### C.2 最小数据集的"最小"边界在哪

不能小于 **16 恶 / 16 良**，且必须分 ≥2 个月（否则 `train` 或 `eval` 为空）。本配置实测：

| 项 | 值 |
|---|---|
| 样本 | 32（恶 16 / 良 16，严格 1:1） |
| 月份 | 4 个**合成**月份，每月 4 恶 4 良（配比与月份**解耦**，消除"月份 → 标签"的伪相关） |
| 划分 | `train=[m1,m2]`（16 条）/ `eval=[m3,m4]`（16 条） |
| 视角 | **15 / 15 全部对齐基线**，逐视角维度与基线精确吻合 |
| 模型 | 33 槽位全部可训练，**0 个冻结**（含 10 个 D/E 模型） |

### C.3 三个关键设计点（都是踩坑换来的）

**① 权威维度口径的三层判据**（决定"匹配 ensemble 模型"这件事是否真的做到）

```
第一层（最权威）管线定义：data_loader.RAW_LEN（V1_byte=32768）+ config.VIEW_GROUPS[*].binja_features
                          → 对每个视角，从 npz 实测形状按 binja_features 顺序求和
第二层（校核）  基线精确值：pytorch input_dim / 线性层 w.shape[0] / KNN X_train.shape[1]
第三层（兜底）  基线下界：纯 NumPy 树模型 max(feature_idx)+1（只能给下界，树不必用满所有列）
```

实测中 `V6_opcode_stat` 由第三层判据拿到**下界 549**，与第一层推导出的 549 交叉吻合 —— 这是整套维度映射正确的关键证据（此前误记为 512）。

**② 分组感知分月（替代 `assign_months()` 的 mtime 依赖）**

按 `(label, imphash 组)` 分层，**整组**分配到月份。同一 imphash 的变体绝不跨 `train`/`eval`，否则家族变体泄漏会让 eval AUC 虚高到无意义。

**③ `metadata.jsonl` 必须写绝对规范路径**

`discover_samples()` 对相对路径做 `os.path.join(data_dir, rel)` → Windows 上拼出混合分隔符 `...\minimal_dataset\benign/xxx`；而 `PELoader._sample_cache_path()` 直接对该字符串取 md5（**内部不 normpath**）→ 缓存键漂移 → 加载器命中率 0/32，且刚提取好的 D/E 特征会被 `_drop_unobtainable_views()` 判为"取不到"而**静默剔除**。

```
键A(normpath) = ca4d11ce...  存在:True
键B(原样)     = c1b70714...  存在:False
```

写绝对路径后 `os.path.isabs()` 短路 join，问题消失。脚本默认 `--path-style abs`。

### C.4 `_has_month_cache()` 阈值修复（B7）

```python
def _has_month_cache(self, month):
    yp = self._month_y_path(month)
    try:
        # 曾经写成 200 字节 (= 128 + 18*4)，等价于"月样本数必须 >= 18"。
        # 128 + 4 是真正的"非空"判据: 空数组 npy 刚好 128 字节。
        if os.stat(yp).st_size < 132:
            return False
    except OSError:
        return False
    pattern = os.path.join(self._month_cache, f"m{month}_V*.npy")
    return bool(glob.glob(pattern))
```

### C.5 `input_dim` 写入时机修复（B8）

```python
params = inspect.signature(cls.__init__).parameters
if "input_dim" in params:        # MalConv / MalConv2 / ByteTransformer 不满足
    kwargs["input_dim"] = int(Xn.shape[1])
...
akw = (info or {}).get("actual_kwargs")
if akw is not None:              # 用"实际构造参数"覆盖，避免记入未使用的键
    ens._actual_kwargs[mname] = dict(akw)
```

`verify_trained_ensemble.py` 正是在这一步抓到：修复前报
`TypeError: Malconv.__init__() got an unexpected keyword argument 'input_dim'`。

### C.6 实测结果（run-A，修复后正式产物）

> 📌 **2026-09-16 状态更新**：`runs/run-A/` 与 `runs/smoke-scratch-01/`（B8 修复前的失败产物）
> 已按用户决定清理以释放磁盘（两者合计约 1.5 GB）。**当前 `runs/` 下唯一产物是 `pilot500/`**
> —— 它是同一套流程在 491 样本上的验证结果（eval AUC 0.9599 / F1@0.5 0.8855 / FPR 0.225，
> 复验同样全项通过）。本节数值保留为当时的实测记录。

```
数据构建 : 32 样本 / 4 月 / 15 视角全对齐
           BN 可用（binaryninja 5.2.8722），bndb 阶段 89s（32/32，batch=1 重试救回"子进程崩溃"），
           D/E 特征阶段 30s（32/32）。单样本 npz 30 key → 39 key（含 D01–D07、E03、E04）
训练     : 32/32 成功 / 0 失败 / 55.5s / 产物 760.6 MB
复验     : ✓ schema 键集合一致（含 special_state）
           ✓ K=33 V=15、available_views / view_indices / model_names 全一致
           ✓ weights shape=(33,) 等权 sum=1.0，差异槽位 0
           ✓ 候选 load() OK，可实例化 32/33
           ✓ 前向冒烟通过，仅 MalGraph 恒 0.5（special 占位的既有行为）
           → 候选可直接热切换替换基线
```

### C.7 已知局限（**不要用这份数据下质量结论**）

| 现象 | 说明 |
|---|---|
| eval AUC 0.9375 / F1@0.5 0.6957 / FPR 0.875（train AUC 1.0） | **16 个训练样本必然过拟合**。这组数字只能证明"管路通"，不构成任何质量判断 |
| 逐模型分数普遍接近 0.5 | 小数据 + 6 epoch，属预期 |
| 阈值/权重结论 | 需在真实月度缓存（m1..m6，400 样本）上重做，见 §14.2 |

### C.8 BN 实测更正

原 §6.1 记录"BN Personal 许可不支持 headless → D/E 永久取不到、`.cache/bndb` 为 0"。**本轮实测推翻该结论**：`import binaryninja` 成功（5.2.8722），32 个样本的 D/E 特征在 119s 内全部提取成功。

结论修正为：**BN 在本机可用，只是 bndb 阶段偶发"子进程崩溃(unreported)"，把 batch_size 降到 1 即可全部重试救回。** 因此"最小数据集"这条路上不需要接受"冻结 10 个模型"，15 视角是可达的。

---

## 附 D：MalGraph 接入增量训练 —— PyG 适配器实现方案（设计稿，2026-09-15）

> 目标：让 MalGraph 在 `train_minimal.py` 的增量训练里吃到新样本，产出可被引擎
> 热加载的新 `best_model.pt`。设计原则：**最大化复用 MalGraph_2022 自带的训练资产，
> 引擎侧零代码改动**。

### D.0 现状盘点（方案的地基）

MalGraph_2022 目录里其实**已经有一套完整的离线训练管线**，缺的只是与
`train_minimal.py` 的适配层：

| 既有资产 | 位置 | 作用 | 复用方式 |
|---|---|---|---|
| `HierarchicalGraphNeuralNetwork` | `malgraph.py` | 两级 GNN（CFG→FCG），forward 已支持多样本批处理（`real_bt_positions`） | 直接用 |
| `MalgraphDynamicDataset` | `dataset.py:77` | PE → PyG Data，按文件哈希落 `.pt` 缓存 + `DynamicVocab` 增量词表 | **核心复用件** |
| `extract_cfg_and_fcg` | `utils.py:24` | BN 提取 ACFG+FCG（依赖 BinaryViewType，**需 BN 可用**） | 直接用 |
| `1_exec_model.py` | 同目录 | 原训练入口：`safe_extract_bn` 分级超时（60/90/120s 按文件大小）、多进程 `.pt` 转换、bad_files 记账 | 超时/容错策略的参考实现 |
| `Vocab` / `DynamicVocab` | `Vocabulary.py` / `dataset.py:11` | 词表，OOV 回退 `<unk>` | finetune 冻结词表时直接用 |

关键接口事实（适配层必须逐条满足）：

```
forward(real_local_batch: Batch,      # 所有函数级 CFG 拼成的一个大 Batch
        real_bt_positions: list,      # 长度 N+1，每个样本在大 Batch 里的 [起,止)
        bt_external_names: list,      # 每样本的外部函数名列表（FCG 扩展节点）
        bt_all_function_edges: list,  # 每样本的 FCG 边
        local_device)
```

`MalgraphDynamicDataset.__getitem__` 返回的 `Data` 已带全部原料：
`local_acfgs`（该样本所有函数的 `Data(x, edge_index)` 列表）、
`external_list`（外部函数名索引）、`function_edges`、`targets`（标签）。
**适配器的本质 = 把这个 Data 拆开重组为 forward 的五个参数。**

### D.1 数据面：样本从哪来

- **清单**：直接用 `train_minimal.py` 既有的 `loader`（`metadata.jsonl` 的
  path/label/month），只取 `train_months`（eval 月同样取，供指标）。
  不需要新 metadata——**MalGraph 与其他 32 个模型共享同一样本清单**。
- **图缓存**：`MalgraphDynamicDataset(pt_cache_dir=<data_dir>/.cache/malgraph_pt,
  vocab_cache_path=<data_dir>/.cache/malgraph_vocab.jsonl)`。
  缓存键是**文件内容哈希**（`calculate_file_hash`），与 `per_sample` 的路径 md5
  键不同、互不干扰；换目录/搬样本不会导致缓存失效。
- **提取成本**：BN 函数级分析比 bndb 更贵。沿用 `1_exec_model.py` 的分级超时
  （<5MB:60s / <20MB:90s / 其余 120s），失败样本记入 run 目录
  `malgraph_bad_files.txt` 并**跳过不阻断**（与原管线 bad_files.txt 同思路）。
  预算参考：32 样本约 3~5 分钟（BN bndb 实测 89s/32 + 函数级分析放大），
  真实 400 样本需另测。
- **词表策略（关键决策）**：
  - `--mode finetune`（默认建议）：**冻结词表为基线的 1002 token**（embedding
    形状 `(1002, 200)` 与 `best_model.pt` 对齐）。新样本的外部函数名走 OOV→`<unk>`。
    DynamicVocab 照常统计频率并写到 run 目录，但**不回写** `saved/` 的正式词表
    ——若新词频次证明值得扩词表，那是一次显式的"扩词表+重训"操作（见 D.6）。
  - `--mode scratch`：可放开 `max_vocab_size`，全量重建词表与模型。

### D.2 适配层：`train_malgraph.py`（新增，`CoDefenderC3/` 下）

单一职责文件，约 300 行，被 `train_minimal.py` 以可选模块方式调用
（`--train-malgraph` 开关，默认关，不影响既有 32 模型流程）。

```python
def collate_malgraph(batch):
    """[(Data, target), ...] → forward 五参数 + labels"""
    datas, labels = zip(*batch)
    graphs, bt_positions, ext_names, fcg_edges = [], [0], [], []
    for d in datas:
        graphs.extend(d.local_acfgs)              # 函数级 CFG 全部并入大 Batch
        bt_positions.append(bt_positions[-1] + len(d.local_acfgs))
        # external_list 存的是索引；训练时需要名字 → dataset 侧改为同时保留
        # external_names 原文（D.3 改动点 ①），这里直接用名字列表
        ext_names.append(d.external_names)
        fcg_edges.append(d.function_edges)
    big = Batch.from_data_list(graphs)
    return big, bt_positions, ext_names, fcg_edges, torch.stack(labels)
```

训练循环（每 epoch）：

```
for batch in DataLoader(ds, batch_size=B, collate_fn=collate_malgraph,
                        shuffle=True):
    big, pos, names, edges, y = batch
    pred = model(big, pos, names, edges, device)     # (N, 1) sigmoid
    loss = F.binary_cross_entropy(pred.squeeze(-1), y)
    ...
```

- **batch_size**：`real_bt_positions` 机制天然支持多样本/批，但图大小差异极大，
  从 `B=2` 起步（原管线 `train_bs` 也是小批），OOM 再降。
- **早停**：与 `train_minimal.train_torch_model` 同策略（patience=3，**保存并恢复
  最优权重**——避免复刻 B1 缺陷）；eval 集月度标签算 AUC/F1@0.5/FPR。
- **模型实例化**：完全照抄引擎 `_load_malgraph_runtime()` 的推断路径
  （`_infer_malgraph_params_from_state` + `ModelParams` + `Vocab(freq_file,
  max_vocab_size=1002)`）——**训练侧与推理侧用同一套参数推断逻辑**，杜绝
  "训练的形状引擎装不回去"这类 B8 同型事故。

### D.3 MalgraphDynamicDataset 的两处小改（改动点清单）

1. **`external_names` 原文保留**：`__getitem__` 里 `external_indices` 只存了词表
   索引，而 forward 的 `bt_external_names` 需要**名字**（模型内部再走
   `word2idx`）。在 Data 里补存 `external_names=external_names`（.pt 缓存会随之
   变大一点；旧缓存无此字段时回退从 json 缓存读）。
2. **`function_edges` 类型硬化**：`extract_cfg_and_fcg` 边采样自 JSON，可能出
   `[[], []]`/`None` 混态，统一 clamp 成 `[[], []]` + 显式 `dtype=long`（引擎
   `_predict_malgraph_single` 已有同样的防御，照搬）。

### D.4 Checkpoint 与产物落地（与其他 32 个模型的本质差异）

MalGraph 权重**不在 ensemble.pkl 里**（`models['MalGraph']` 只是
`'SPECIAL_PLACEHOLDER'`），引擎从磁盘 `models/MalGraph_2022/best_model.pt` 读。
所以产物面是双轨：

```
runs/<run_id>/
├── partial/MalGraph.pt          # 逐 epoch checkpoint（断点续训粒度）
├── malgraph/
│   ├── best_model.pt            # 训练完成后的最终 state_dict（与引擎同格式）
│   ├── vocab.jsonl              # 本次训练用的词表（随权重走，成对交付）
│   ├── metrics.json             # AUC / F1@0.5 / FPR / bad_files 计数
│   └── bad_files.txt
├── progress.json                # done["MalGraph"] = {"ckpt": "malgraph/best_model.pt", ...}
└── ensemble.pkl                 # 32 个通用模型的产物（MalGraph 槽位仍为占位）
```

- `partial/MalGraph.pt` 存**完整 state_dict**（2.7MB，比通用模型大不了多少，
  不值得为它做增量格式）。
- `save_ensemble_preserving()` 对 MalGraph 槽位照旧透传占位值——**不把新权重塞回
  pkl**，因为没人读那份 `special_state`（死数据，见 §13 B3 澄清）。
- **`--in-place` 安装动作**：备份 `models/MalGraph_2022/best_model.pt` →
  `best_model.pt.bak-<时间戳>`，再拷入新 `.pt`（词表只在 scratch 模式下随同安装；
  finetune 模式词表未变，不覆盖）。
- **引擎热生效**：`POST /reload-model` 已重置 `_MALGRAPH_RUNTIME`
  （`app.py:1307,1342` 实测确认）→ 安装后调一次 reload-model 即可，
  **引擎侧零代码改动**。

### D.5 与 train_minimal.py 的接线（最小侵入）

```
main() 里 special 处理段（现 530-531 行）改为：

passthrough = [m for m in special if m not in ("MalGraph",)]   # 理论上恒空
malgraph_flag = "--train-malgraph" in argv or args.train_malgraph
if "MalGraph" in special:
    if malgraph_flag:
        special_trainable.append("MalGraph")     # 走 D.2 适配层
    else:
        passthrough.append("MalGraph")           # 现状：原样继承
```

- `progress.json` 的 `passthrough` 字段语义不变；MalGraph 进
  `done["MalGraph"]` 时附 `ckpt` / `vocab` / `metrics` 子键。
- `--resume`：`partial/MalGraph.pt` 存在即续训，与通用模型同一套判断。
- 指标汇报：MalGraph 的 AUC/F1 单列一行进 manifest `metrics`，**不并入
  32 模型集成分数**（它本就不在 `ensemble_predict` 的聚合里，B3 修复后口径
  已对齐）。它的定位是"独立结果项 + 特图视角补充信号"，与引擎在线行为一致。

### D.6 边界与风险

| 风险 | 对策 |
|---|---|
| BN 函数级分析慢/崩溃（比 bndb 更贵） | 分级超时 + bad_files 跳过 + `batch_size=1` 重试（§C.8 已验证该策略有效）；`.pt` 内容级缓存保证二次运行近零成本 |
| OOV 率过高使 finetune 退化（大量外部函数映射到 `<unk>`） | run 结束打印 OOV 命中率；>40% 时提示走"扩词表+scratch 重训"（D.1 词表策略的显式升级路径） |
| 图 batch 维度不齐导致 forward assert | `collate_malgraph` 保证 `bt_positions` 单调；空图样本（无函数）在 dataset 层返回 None、DataLoader 用 `collate_skip_none_to_batch`（`dataset.py:223` 已有）过滤 |
| 词表漂移破坏 warm start | finetune 强制 `max_vocab_size=1002` + 词表文件只读使用（不回写 saved/） |
| `torch_geometric` 依赖 | 引擎推理已在用（`_predict_malgraph_single` import Batch/Data），训练侧无新增依赖 |
| 32 样本最小数据集上没有意义 | 每类仅 16 条，图模型必然过拟合——**验收标准是"管线通 + 指标可复现"**，与 §C.7 同口径；真实结论须在 400 样本月度缓存上做 |

### D.7 实施顺序（可分四步独立验收）

1. **数据面**：改 `MalgraphDynamicDataset`（D.3 两处）→ 对 32 样本跑通
   提取+缓存，产出 `.pt`/`.json`/词表统计。
2. **适配器**：`collate_malgraph` + 训练循环 → 用缓存数据空跑 1 epoch 前向/反向。
3. **产物面**：`best_model.pt` 落 run 目录 → `--in-place` 安装 → `reload-model`
   → `predict-directory` 对同样本打分，确认 MalGraph 分数变化且无加载报错。
4. **接线**：`train_minimal.py --train-malgraph` 全流程 + `--resume` 断点续训 +
   manifest 指标。

预计总工作量：第 1、2 步各约半天（代码量小，坑在 BN 提取的脏数据），
第 3、4 步各约 1~2 小时。

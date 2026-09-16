# CoDefenderC3 / train —— 训练侧代码

> 本目录收集 **CoDefenderC3 的训练与数据集构建链路**。
> 2026-09-16 由仓库根平铺状态整理至此；说明文档同期移入 `../docs/`。

---

## 1. 为什么是这些文件搬进来、那些留在根

判断标准只有一条：**是否被推理服务（`engine/`）依赖**。

| 组件 | 去向 | 原因 |
|---|---|---|
| `run_all.py` / `train_minimal.py` / `build_*.py` / `data_loader.py` / 探针脚本 | **→ `train/`** | 仅训练侧使用，`engine/` 不引用（已全仓 grep 核实） |
| `config.py` | **留在仓库根** | `engine/` `core/` `models/` `scaner/` 全部依赖 |
| `model_registry.py` | **留在仓库根** | 被 `models/ensemble.py` 依赖，推理也要读 |
| `feature_extraction/` | **留在仓库根** | 被 `engine/app.py` 直接 import（`BinjaExtractor`） |

本目录脚本通过「把 `train/` 的上一级加回 `sys.path`」继续 `import config` 等，
所以你**必须从项目根执行**，或至少让 `CoDefenderC3/` 可被 Python 找到：

```bash
cd D:/VM_Share/CoDefenderC3_v8          # ← 项目根
python CoDefenderC3/train/<script>.py ...
```

---

## 2. 脚本清单

### 2.1 入口脚本（直接跑）

| 脚本 | 用途 | 何时用 |
|---|---|---|
| `build_minimal_dataset.py` | 构建**与基线视角严格对齐**的数据集：1:1 取样 → imphash 分组 → 整组分月 → 复制 → 特征提取 → D/E 覆盖检查 → per_month → 四方校验 | 有新的一批裸 PE 要训练时，**第一步** |
| `train_minimal.py` | 集成训练 / 微调，**逐模型 checkpoint + `--resume`**，默认不碰生产模型 | 第二步 |
| `verify_trained_ensemble.py` | 候选 `ensemble.pkl` 与基线的等价性全项校验（**上线门禁**） | 第三步，上线前必做 |
| `build_flat_dataset_meta.py` | 轻量版：**只写 metadata**（复用已有特征缓存，不重新提取） | 需要改分月/分组策略时 |
| `run_all.py` | 原完整实验管线（训练 33 模型 + SDD 漂移 + 基线对比 + 消融 + 出图） | 做实验/论文对比时 |

### 2.2 库模块（被上面 import，一般不用直接跑）

| 模块 | 职责 |
|---|---|
| `data_loader.py` | `PELoader`：`per_sample` 增量特征提取、`per_month` 月度缓存、`metadata.jsonl` 解析、月份划分 |
| `dataset_utils.py` | 数据集收尾（`finalize_dataset`），被 `build_dataset.py` 调用 |
| `ember_vectorizer.py` | 纯 NumPy 的 EMBER 2381 维向量化（不依赖 `ember` 包） |
| `build_dataset.py` / `build_rawpe.py` | EMBER-2018 / 裸 PE 数据集构建（`run_all.py` 的依赖） |

### 2.3 诊断探针（只读，排查用）

| 脚本 | 用途 |
|---|---|
| `_probe_ensemble.py` | 容错拆解**任意** `ensemble.pkl`：顶层键 / K / V / `view_indices` / `weights` / 逐模型输入维度 / 未解析类。缺模块不中断（自定义 `find_class` 回退占位类） |
| `_verify_ensemble_load.py` | 验证 `MultiModelEnsemble.load()` 能否真的加载目标 pkl，并做一次伪数据前向 |
| `_inspect_ensemble.py` | 只读拆解各模型的输入特征约定 |
| `_probe_device.py` | 统计 pkl 内所有 torch 张量的**设备标签**，判断能否在纯 CPU 环境 load。全 `cpu` → exit 0；含 `cuda:*` → exit 3 告警 |
| `_cmp_ensemble.py` | 比对两份 pkl 的**数值指纹**（形状 / dtype / 内容 md5），判断能否互换做基线。设备标签不同但 md5 全同 → 数值等价 |

```bash
PYTHONIOENCODING=utf-8 python -u CoDefenderC3/train/_probe_ensemble.py [pkl路径]
PYTHONIOENCODING=utf-8 python -u CoDefenderC3/train/_verify_ensemble_load.py [pkl路径]
PYTHONIOENCODING=utf-8 python -u CoDefenderC3/train/_probe_device.py [pkl路径]
PYTHONIOENCODING=utf-8 python -u CoDefenderC3/train/_cmp_ensemble.py <pklA> <pklB>
```

> **上线到纯 CPU 容器前，先跑 `_probe_device.py`。** 见下文 §5「pkl 设备标签」。

### 2.4 特征提取侧的 BN 工具

| 脚本 | 用途 |
|---|---|
| `apply_bn_optimizations.py` | 给 `../feature_extraction/extract_feature.py` 打 BinaryNinja 分析选项优化补丁（会先备份原文件） |
| `benchmark_bn_optimization.py` | 对比优化前后的 BN 分析耗时 |

> 这两个脚本修改/测量的是 `feature_extraction/`（**留在仓库根，引擎共用**），
> 所以它们的路径常量已改为「仓库根 + `feature_extraction/`」，不是本目录。

### 2.5 `logs/`

历史运行日志（构建 / 训练 / 复验）。纯留档，不被任何代码读取。

---

## 3. 标准工作流

### 三步走：构建 → 训练 → 复验

```bash
cd D:/VM_Share/CoDefenderC3_v8
PY="D:/work/tools/miniconda3/python.exe"

# ── 第 1 步：构建数据集 ───────────────────────────────────────────
#   源目录要求：平铺一层，下含 malicious/ 与 benign/ 两个子目录
#   ⚠ 目录名大小写敏感，且只认：benign|clean|negative|ben|goodware|0
#                                malicious|malware|positive|mal|1
#      （`malicous` 这种拼写变体不在集合内，会被静默忽略）
#   ⚠ 跑之前建议清空 <数据目录>/.cache/sample_skip.log
#      —— 它按 basename 永久跳过"失败过"的样本
CODEFENDER_KEEP_BNDB=1 PYTHONIOENCODING=utf-8 "$PY" -u \
  CoDefenderC3/train/build_minimal_dataset.py \
  --source "E:/nkproject/training_data0915" \
  --out    "D:/nkproject/data_10k" \
  --per-class 5000 --months 4 --train-months 2 --views full

# 先预演（不写文件）确认视角方案与分月：
#   ... --dry-run

# ── 第 2 步：训练 ────────────────────────────────────────────────
#   默认只写 CoDefenderC3/runs/<run_id>/，生产 ensemble.pkl 不受影响
PYTHONIOENCODING=utf-8 "$PY" -u \
  CoDefenderC3/train/train_minimal.py \
  --dataset "D:/nkproject/data_10k" \
  --run-id  run-10k --mode scratch --epochs 20
#   中断后：同一条命令重跑即自动续训（从 runs/<id>/partial/ 续）
#   想按视角分批跑：--models CFGGAT,CFGGCN,...   或   --exclude IMCFN

# ── 第 3 步：复验（门禁） ────────────────────────────────────────
PYTHONIOENCODING=utf-8 "$PY" -u \
  CoDefenderC3/train/verify_trained_ensemble.py \
  --candidate CoDefenderC3/runs/run-10k/ensemble.pkl
#   --baseline 默认 CoDefenderC3/ensemble.pkl

# ── 第 4 步：上线（可选，会先自动备份） ──────────────────────────
#   在 train_minimal.py 上加 --in-place 即可覆盖生产模型，
#   之后调一次 POST /reload-model 让引擎热加载（约 5s 停顿，无需重启）
```

### 只改分月 / 分组策略（不重新提特征）

```bash
python CoDefenderC3/train/build_flat_dataset_meta.py \
    --data-dir "D:/nkproject/pilot500" --months 4 --train-months 2 \
    --group-key imphash --dry-run
#   --purge-per-month 会把陈旧 per_month 缓存**改名归档**（不是删除）
```

---

## 4. 产物与数据放在哪

| 位置 | 内容 | 说明 |
|---|---|---|
| `CoDefenderC3/runs/<run_id>/ensemble.pkl` | 最终集成产物 | schema 与基线一致，可直接热切换 |
| `CoDefenderC3/runs/<run_id>/partial/<model>.pkl` | **逐模型 checkpoint** | 断点续训的唯一依据；占空间大头（主要来自 IMCFN / InceptionV3 / HashEmbedNet） |
| `CoDefenderC3/runs/<run_id>/progress.json` | 续训状态账本 | 记录 `baseline_sha256` / `trainable` / `done[]`，给下一次运行读 |
| `CoDefenderC3/runs/<run_id>/manifest.json` | 训练记录与指标 | 含数据集/基线 sha256、超参、逐模型 AUC，给人看 |
| `<数据目录>/` | 数据集本体（样本 + `metadata.json` + `metadata.jsonl` + `manifest.json`） | 由 `--out` 指定 |
| `<数据目录>/.cache/per_sample/` | 单样本特征 npz | 键 = `md5(normpath(abspath(样本路径)))` |
| `<数据目录>/.cache/per_month/` | 月度视角矩阵 | **不记样本路径**，可跨目录复用，别删 |
| `<数据目录>/.cache/bndb/` | BinaryNinja 数据库中间产物 | 可事后清理（建议**改名**归档，见下） |

> ⚠️ **不要移动数据集目录**。`metadata.jsonl` 写的是**绝对路径**，而 `per_sample`
> 缓存的键是路径的 md5 —— 一旦搬目录，缓存全部失效（需整库重提取）。
> `per_month` 不记路径，可以跟着搬。

---

## 5. 环境变量与运行纪律

| 变量 / 用法 | 作用 |
|---|---|
| `PYTHONIOENCODING=utf-8` | **Windows 下必须**，否则中文输出崩溃（exit 1 且无报错信息） |
| `python -u` | **长任务必须**。stdout 重定向时是块缓冲，非 `flush=True` 的 print 长时间不可见 |
| `CODEFENDER_KEEP_BNDB=1` | 不自动删除 `.cache/bndb`（门控在 `../feature_extraction/extract_feature.py`，共 5 处）。批量提取时**强烈建议开**：崩溃样本的清理动作累计 >50 个文件会触发 safe-delete 钩子，直接把进程杀掉（连 `EXITCODE` 都来不及写） |
| `CODEFENDER_N_WORKERS=8` | 覆盖 BN 并行度（`../config.py:20`，默认 `min(max(cpu,8),16)`）。500+ 样本规模下 16 并行易崩，建议降到 8 |
| `CODEFENDER_VIEWS_REQUIRE_ALL=1` | 严格模式：缺任一视角即丢样本（`../config.py:265`，默认 `0` = 剔除取不到的视角、不丢样本） |
| `CODEFENDER_FORCE_RETRAIN` | ⚠️ **代码里尚未实现**，仅出现在 `docs/增量训练与Checkpoint技术方案.md` 的 L2 设计稿中 |

**清理 bndb 用「改名」而不是删除**，否则又会触发上面那个钩子：

```bash
mv "<数据目录>/.cache/bndb" "<数据目录>/.cache/bndb.stale-<日期>"
```

### pkl 设备标签（上线纯 CPU 容器前必看）

`MultiModelEnsemble.load()`（`../models/ensemble.py:553-556`）用的是**裸 `pickle.load`**，
**没有 `map_location` 参数**。而 pickle 里的 torch 张量带设备标签：

| pkl 内的张量 | 有 GPU 的机器 | 纯 CPU 机器 |
|---|---|---|
| 全 `cpu` | 可 load | 可 load |
| 含 `cuda:*` | 可 load（`load_state_dict` 会跨设备拷贝） | **崩** |

在无 GPU 环境加载 CUDA 版 pkl 的报错是：

```
RuntimeError: Attempting to deserialize object on CUDA device 0 but
torch.cuda.device_count() is 0. Please use torch.load with map_location
to map your storages to an existing device.
```

torch 让你用 `map_location`，但 `load()` 根本没这个参数可传 → **只能改代码，调用方绕不开**。

实测（2026-09-16）：

| 文件 | 大小 | 张量设备 | 纯 CPU 可 load |
|---|---|---|---|
| `CoDefenderC3/ensemble.pkl`（生产基线） | 875.3 MB | 1229 个全 `cpu` | ✓ |
| `D:/VM_Share/ensemble.pkl`（2026-04-21 旧快照） | 1052.6 MB | 1229 个全 `cuda:0` | ✗ |

> **体积差 ≠ 权重不同。** 这两份逐张量 md5 **完全一致**、meta 全同，只差设备标签。
> 不确定两个 pkl 能否互换时，用 `_cmp_ensemble.py` 比指纹，别靠文件大小猜。

**训练产物永远是 CPU 版**，可以放心丢给纯 CPU 容器：
`serialize_model()`（`train_minimal.py:339-347`）与 `save_ensemble_preserving()`
（`:373-409`）对所有 torch `state_dict` 都做了 `v.detach().cpu()`。
所以「用 CPU 还是 CUDA 训练」不影响产物的可移植性。

上线前自检：

```bash
PYTHONIOENCODING=utf-8 python -u CoDefenderC3/train/_probe_device.py <pkl路径>
# 全 cpu → 结论 ✓ exit 0 ；含 cuda → 结论 ⚠ exit 3
# 想模拟纯 CPU 环境验证：前面加 CUDA_VISIBLE_DEVICES=""
```

---

## 6. 与引擎的边界（别搞混）

| | 离线（本目录） | 在线（`engine/app.py`） |
|---|---|---|
| 集成分数 | `MultiModelEnsemble.ensemble_predict()` | **不走它**，自己算 `valid_scores.mean(axis=1)` 且阈值写死 `0.5` |
| `weights` | 生效（B3 修复后还排除了 special 槽位） | **不读**，恒等权 |
| MalGraph | `models['MalGraph']` 只是 `'SPECIAL_PLACEHOLDER'`，权重在磁盘 `models/MalGraph_2022/best_model.pt` | 有独立的实时推理链路（现场提 CFG/FCG），但**不进集成分数** |
| 阈值 | `train_minimal.py` 会做阈值扫描 | 硬编码 `0.5`（全仓约 10 处） |

所以：**离线指标好看 ≠ 线上行为一致**。改阈值/上线前请以 `engine/app.py` 的实际口径复算。

---

## 7. 已知陷阱速查

| 现象 | 根因 | 处置 |
|---|---|---|
| 加载器命中 0/N，刚提取的特征"看不到" | `metadata.jsonl` 写了相对 POSIX 路径 → `os.path.join` 拼出混合分隔符 → 缓存键 md5 不同 | `build_minimal_dataset.py` 默认 `--path-style abs`，别改 |
| 训练日志重复出现"逐模型训练"，`per_month` 反复重建 | `data_loader._has_month_cache()` 阈值 bug（已修为 `>= 132` 字节） | 已修复；若用旧代码，月样本数 < 18 时必现 |
| 产物 `load()` 报 `unexpected keyword argument 'input_dim'` | `MalConv`/`MalConv2`/`ByteTransformer` 是 RAW_BYTES 模型，不接受该参数 | 已修复（仅当 `__init__` 签名含该参数才写入） |
| 进程无故被杀、无 `EXITCODE` | safe-delete 钩子（一轮删除 >50 文件） | 开 `CODEFENDER_KEEP_BNDB=1`；清理用改名 |
| 扫描器说"N 个样本无 bndb、0 个需提取"但盘上明明有 | `bndb_fail` 日志被崩溃记账污染，且条目里的路径字符串与 canonical path 有 md5 错位 | 按 **basename 映射回 canonical path** 再判定，清洗日志 |
| `run_all.py` 无参跑起来在下载 EMBER | 探测不到数据集 → 兜底 `build_dataset("ember2018")` → 输出根硬编码 `F:\Experimental data` | 用 `--single <data_dir>` 显式指定 |
| `build_minimal_dataset.py --help` 崩 `unsupported format character` | argparse help 里的裸 `%` 被当格式符 | 已修复（`100%%`） |
| `load()` 报 `Attempting to deserialize object on CUDA device 0` | 基线 pkl 是 CUDA 标签版，而 `load()` 无 `map_location` | 换 cpu 版基线；先用 `_probe_device.py` 探 |

---

## 8. 相关文档

- `../docs/README.md` —— 项目总览（服务、端点、配置、架构）
- `../docs/增量训练与Checkpoint技术方案.md` —— 增量训练 + checkpoint 完整技术方案（含缺陷清单 B1–B8、最小数据集附录 C、MalGraph 接入附录 D）
- `../docs/TESTING.md`、`../docs/PAPERS.md`、`../docs/BINARYNINJA_OPTIMIZATION_GUIDE.md`

# CoDefenderC3

**Multi-Model Collaborative Malware Concept Drift Defense via Spectral Drift Decomposition and LLM-Assisted Analysis**

多模型集成恶意软件检测系统，基于 PyTorch 的多视角异构模型集成架构，覆盖 PE 文件 **8 大分析维度**，集成 **33 个已发表子模型** + **SDD 谱漂移检测引擎** + **LLM 辅助分析层**。

---

## 目录

- [项目架构](#项目架构)
- [环境要求](#环境要求)
- [开发环境运行](#开发环境运行)
- [生产环境部署 (Docker)](#生产环境部署-docker)
- [API 接口](#api-接口)
- [测试指南](#测试指南)
- [训练管线](#训练管线)
- [配置参考](#配置参考)
- [项目结构](#项目结构)

---

## 项目架构

```
四层架构:
┌─────────────────────────────────────────────────┐
│ Layer 4: 自适应推断层    DSIR + DACP + 策略选择  │
├─────────────────────────────────────────────────┤
│ Layer 3: LLM辅助分析层   ATT&CK映射 + 策略推荐   │
├─────────────────────────────────────────────────┤
│ Layer 2: SDD漂移检测     谱分解 + 漂移证据量化    │
├─────────────────────────────────────────────────┤
│ Layer 1: 多模型集成层    33模型 × 16视角组        │
└─────────────────────────────────────────────────┘
```

**16 个视角组 (View Groups)**:

| 视角 | 特征维度 | 代表模型 |
|------|----------|----------|
| V1_byte | 字节级 | MalConv, MalConv2, ByteTransformer |
| V2_byte_stat | 字节统计 | SaxeBerlinDNN, EmberDNN, NatarajKNN |
| V3_pe_struct | PE 结构 | DrebinSVM, DL4MD, PEMinerRF |
| V4_import | 导入表 | DrebinImport, ALOHANet, EmberGBDT_Import |
| V5_string | 字符串 | RaffFeatureNet, DrebinString |
| V6_opcode_seq/stat/func_embed | 操作码 | OpcodeLSTM, OpcodeStatNet, AsmEmbedNet |
| V7_graph | 图结构 | MalGraph, CFGGAT, CFGGCN, DGCNN, CallGraphGNN |
| V8_gray/color/markov/entropy | 可视化 | GrayscaleCNN, IMCFN, MarkovCNN, EntropyMapCNN |
| V9_hash | 哈希 | HashEmbedNet |
| V10_metadata | 元数据 | MetadataNet, ResourceNet |
| V11_ensemble | 全量 | Ember_NDF, EmberGBDT |

---

## 环境要求

| 组件 | 版本要求 | 说明 |
|------|----------|------|
| Python | ≥ 3.10 | 开发环境当前使用 3.13.13 |
| PyTorch | ≥ 2.0 | CPU-only 节省 ~2GB 空间 |
| scikit-learn | ≥ 1.0 | EmberGBDT (LightGBM)，传统 ML 子模型 |
| FastAPI + uvicorn | ≥ 0.110 / ≥ 0.29 | API 服务层 |
| BinaryNinja | (可选) | PE 深度特征提取（D/E 特征），商业许可证 |
| ensemble.pkl | 834 MB | 预训练的多模型集成权重文件 |

### 依赖安装

```bash
# 1. 安装 PyTorch CPU-only
pip install torch>=2.0 --index-url https://download.pytorch.org/whl/cpu

# 2. 安装其余依赖
pip install -r requirements.txt
```

---

## 开发环境运行

### 1. 确保 ensemble.pkl 就位

```bash
# 检查模型文件是否在项目根目录
ls -lh ensemble.pkl
# 预期: ensemble.pkl (834 MB)

# 或通过环境变量指定路径
export CODEFENDER_MODEL_PATH=/path/to/ensemble.pkl
```

### 2. 启动 API 服务

```bash
# 方式一：在项目根目录执行
uvicorn engine.app:app --host 0.0.0.0 --port 8000

# 方式二：Python 模块方式
python -m uvicorn engine.app:app --host 0.0.0.0 --port 8000

# 方式三：带热重载（开发调试推荐）
uvicorn engine.app:app --host 0.0.0.0 --port 8000 --reload
```

### 3. 验证服务启动

支持swigger UI 文档，访问以下地址查看 API 文档
http://localhost:8000/docs

```bash
# 健康检查
curl http://localhost:8000/health
```

预期返回：

```json
{
  "status": "ok",
  "model_path": "/path/to/ensemble.pkl",
  "model_count": 33,
  "view_count": 16,
  "views": ["V1_byte", "V2_byte_stat", ...],
  "malgraph": { "enabled": true, ... }
}
```

### 4. 开发环境变量参考

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `CODEFENDER_MODEL_PATH` | `./ensemble.pkl` | 模型文件路径 |
| `CODEFENDER_INCLUDE_DE` | `true` | 是否默认启用 D/E 特征提取 |
| `CODEFENDER_ENABLE_MALGRAPH` | `true` | 是否启用 MalGraph 图神经网络 |
| `CODEFENDER_MAX_UPLOAD_BYTES` | `134217728` (128MB) | 最大上传文件大小 |
| `CODEFENDER_MAX_BATCH_FILES` | `500` | 批量扫描最大文件数 |
| `CODEFENDER_MAX_CONCURRENT` | `4` | 预测接口最大并发数 (asyncio.Semaphore) |

---

## 生产环境部署 (Docker)

### 构建镜像

```bash
# 在项目根目录
docker build -t codefender-c3 .
```

**构建说明**:
- 基础镜像: `multiscan_basic_v3` (CentOS, 预装 Python3)
- `ensemble.pkl` (834MB) 已打包进镜像，无需运行时挂载
- supervisor 管理 uvicorn 进程，`autostart=true`
- 健康检查: 每 30s 检查 `/health` 端点

### 启动容器

```bash
# 基础启动
docker run -d -p 8000:8000 --name codefender codefender-c3

# 指定环境变量
docker run -d -p 8000:8000 --name codefender \
  -e CODEFENDER_ENABLE_MALGRAPH=false \
  codefender-c3
```

### 查看日志

```bash
# supervisor 日志
docker logs -f codefender

# 进入容器
docker exec -it codefender bash

# 应用日志
docker exec codefender cat /var/log/supervisor/codefender.log
```

### 管理容器

```bash
docker stop codefender
docker start codefender
docker restart codefender
docker rm -f codefender
```

---

## API 接口

共 8 个端点，3 个探针/指标类 + 3 个业务类 + 2 个管理类。

### 探针与运维端点

#### GET `/livez` — 存活探针

不接触模型，仅检查进程是否存活。适合 K8s `livenessProbe`。

```bash
curl http://localhost:8000/livez
```

**响应示例**:
```json
{
  "alive": true,
  "uptime_sec": 3600,
  "uptime_human": "01:00:00"
}
```

#### GET `/readyz` — 就绪探针

检查模型是否已加载并可服务。未就绪返回 HTTP 503。适合 K8s `readinessProbe`。

```bash
curl http://localhost:8000/readyz
```

**就绪时 (200)**:
```json
{
  "ready": true,
  "model_count": 33,
  "view_count": 16
}
```

**未就绪时 (503)**:
```json
{
  "ready": false,
  "reason": "model not loaded",
  "model_path": "/app/ensemble.pkl"
}
```

#### GET `/metrics` — 运行时指标

返回 Prometheus 兼容的运行时统计，用于监控面板或告警。

```bash
curl http://localhost:8000/metrics
```

**响应示例**:
```json
{
  "total_requests": 142,
  "predict_single_ok": 80,
  "predict_single_error": 3,
  "predict_dir_ok": 5,
  "predict_dir_error": 1,
  "scan_files_total": 520,
  "scan_files_malicious": 410,
  "scan_files_benign": 110,
  "last_predict_ms": 234.5,
  "unhandled_errors": 0,
  "uptime_sec": 3600,
  "models_loaded": true,
  "model_count": 33
}
```

| 指标字段 | 含义 |
|----------|------|
| `total_requests` | 服务启动以来总请求数 |
| `predict_single_ok` | 单文件检测成功次数 |
| `predict_single_error` | 单文件检测失败次数 |
| `predict_dir_ok` | 目录扫描成功次数 |
| `predict_dir_error` | 目录扫描失败次数 |
| `scan_files_total` | 累计扫描文件总数 |
| `scan_files_malicious` | 累计检出恶意文件数 |
| `scan_files_benign` | 累计检出正常文件数 |
| `last_predict_ms` | 最近一次预测耗时 (ms) |
| `unhandled_errors` | 全局异常处理器捕获的未处理错误数 |

---

### 业务端点

#### GET `/health` — 健康检查

```bash
curl http://localhost:8000/health
```

**响应示例**:
```json
{
  "status": "ok",
  "model_path": "/app/ensemble.pkl",
  "model_md5": "a1b2c3d4e5f6...",
  "model_count": 33,
  "view_count": 16,
  "views": ["V1_byte", "V2_byte_stat", "V3_pe_struct", ...],
  "malgraph": {
    "enabled": true,
    "status": "ready",
    "device": "cpu",
    "checkpoint": "/app/models/MalGraph_2022/best_model.pt",
    "vocab": "/app/models/MalGraph_2022/saved/train_external_function_name_vocab.jsonl"
  }
}
```

> `model_md5` 字段用于校验模型文件完整性，由生产加固模块新增。

#### POST `/predict-single` — 单文件检测

```bash
# 快速检测（不提取 D/E 特征）
curl -X POST http://localhost:8000/predict-single \
  -F "file=@sample.exe" \
  -F "include_de=false"

# 完整检测（含反汇编/图特征，需要 BinaryNinja）
curl -X POST http://localhost:8000/predict-single \
  -F "file=@sample.exe" \
  -F "include_de=true"
```

**请求参数**:

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `file` | UploadFile | 是 | — | 上传的 PE 样本文件 |
| `include_de` | bool | 否 | `true` | 是否提取反汇编/图(D/E)特征 |

**响应摘要**:
```json
{
  "FileName": "sample.exe",
  "FileSize": 123456,
  "Status": 1,
  "StatusLabel": "Malicious",
  "IsMalicious": true,
  "EnsembleScore": 0.8723,
  "ModelStats": {
    "TotalModels": 33,
    "OkModels": 30,
    "MaliciousVotes": 28,
    "BenignVotes": 2,
    "AgreementRate": 0.9333
  },
  "DriftEvidence": { "anomaly_score": 0.12, "is_drift_candidate": false },
  "PerModelScores": [...]
}
```

**错误码**:
| HTTP 状态码 | 含义 |
|-------------|------|
| 200 | 检测成功 |
| 400 | 文件为空 / PE 格式无效 |
| 413 | 文件过大 (>64MB) |
| 500 | 服务端内部异常 |

#### POST `/predict-directory` — 目录批量扫描

```bash
# 递归扫描（默认）
curl -X POST "http://localhost:8000/predict-directory?directory=/samples/malware&recursive=true&include_de=false"

# 仅扫描顶层目录
curl -X POST "http://localhost:8000/predict-directory?directory=/samples/malware&recursive=false"
```

**请求参数**:

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `directory` | str | 是 | — | 服务器端目录的绝对路径 |
| `recursive` | bool | 否 | `true` | 是否递归扫描子目录 |
| `include_de` | bool | 否 | `false` | 批量模式默认关闭 D/E 以加速 |

**响应示例**:
```json
{
  "Summary": {
    "TotalFiles": 100,
    "ScannedFiles": 98,
    "MaliciousCount": 85,
    "BenignCount": 13,
    "ErrorCount": 2,
    "Directory": "/samples/malware"
  },
  "Results": [
    { "FileName": "sample_01.exe", "Status": 1, "IsMalicious": true, ... },
    { "FileName": "sample_02.dll", "Status": 0, "IsMalicious": false, ... }
  ]
}
```

---

### 管理端点

#### GET `/model-info` — 模型/病毒库档案

返回当前集成模型的完整档案信息，包括文件完整性校验 (MD5)。

```bash
curl http://localhost:8000/model-info
```

**响应字段**:

| 字段 | 含义 |
|------|------|
| `file.path` | 模型文件路径 |
| `file.size_mb` | 文件大小 (MB) |
| `file.last_modified` | 文件最后修改时间 |
| `file.md5` | 模型文件 MD5 哈希 (完整性校验) |
| `ensemble.model_count` | 集成子模型总数 |
| `ensemble.view_count` | 视角组数量 |
| `models[]` | 逐模型加载状态列表 |
| `views{}` | 视角组详情 (label/ATT&CK映射/模型列表) |
| `sdd.initialized` | SDD 漂移检测引擎是否初始化 |
| `malgraph.*` | MalGraph 图模型状态 |

#### POST `/reload-model` — 病毒库热更新

不传参则原地重载当前模型文件；传参 `model_path` 则切换到新路径并加载。

```bash
# 原地重载
curl -X POST http://localhost:8000/reload-model

# 切换到新模型
curl -X POST "http://localhost:8000/reload-model?model_path=/path/to/new_ensemble.pkl"
```

**请求参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model_path` | str | 否 | 新模型文件绝对路径，不传则重载当前路径 |

---

### 生产加固特性

| 特性 | 机制 | 环境变量 |
|------|------|----------|
| 并发限制 | `asyncio.Semaphore` 包裹 `/predict-single` 和 `/predict-directory` | `CODEFENDER_MAX_CONCURRENT` (默认 4) |
| 请求追踪 | HTTP 中间件注入 `X-Request-ID` 响应头 + 结构化日志 | — |
| 全局异常处理 | `@app.exception_handler(Exception)` 捕获未处理异常 + 计数 | — |
| MD5 完整性校验 | `_file_md5()` → `/health` 和 `/model-info` 均返回 `model_md5`/`md5` | — |
| 安全临时文件 | `_tmp_upload` contextmanager 替代手动 `tempfile`，异常路径下也确保清理 | — |

---

## 测试指南

### 1. 快速功能验证

使用项目自带的 `testsamples/` 目录中的测试样本：

```bash
# 1. 启动服务
uvicorn engine.app:app --host 0.0.0.0 --port 8000

# 2. 健康检查
curl -s http://localhost:8000/health | python -m json.tool

# 3. 单文件检测（使用测试样本）
curl -s -X POST http://localhost:8000/predict-single \
  -F "file=@testsamples/013b241f57afe80f51b5a681774b7aa875b7c676865320ed3ae508a7ba4b9b04" \
  -F "include_de=false" \
  | python -m json.tool
```

### 2. 批量目录扫描测试

```bash
# 扫描整个 testsamples 目录
curl -s -X POST "http://localhost:8000/predict-directory?directory=$(pwd)/testsamples&recursive=false&include_de=false" \
  | python -m json.tool
```

### 3. Python 客户端测试脚本

```python
import requests

BASE = "http://localhost:8000"

# 健康检查
resp = requests.get(f"{BASE}/health")
print(f"Health: {resp.json()['status']}")

# 单文件检测
with open("testsamples/013b241f57afe80f51b5a681774b7aa875b7c676865320ed3ae508a7ba4b9b04", "rb") as f:
    resp = requests.post(
        f"{BASE}/predict-single",
        files={"file": f},
        data={"include_de": "false"},
    )
    result = resp.json()
    print(f"File: {result['FileName']}")
    print(f"Label: {result['StatusLabel']} (score: {result['EnsembleScore']:.4f})")
    print(f"Malicious votes: {result['ModelStats']['MaliciousVotes']}/{result['ModelStats']['OkModels']}")
```

### 4. 压力测试（可选）

```bash
# 使用 Apache Bench
ab -n 100 -c 10 http://localhost:8000/health

# 使用 wrk
wrk -t4 -c10 -d30s http://localhost:8000/health
```

### 5. 预期行为

| 场景 | 预期结果 |
|------|----------|
| 合法 PE 文件 | 返回预测分数 + 标签，per-model scores |
| 非 PE 文件（如 .txt） | HTTP 400: "文件结构不对，或者文件格式损坏" |
| 超过 64MB 文件 | HTTP 413: "file too large" |
| 服务未就绪 | `/health` 返回 `status: "error"` + reason |

---

## 训练管线

> **2026-09-16 目录整理**：训练侧脚本已统一移入 `CoDefenderC3/train/`，
> 说明文档移入 `CoDefenderC3/docs/`。所有命令请从**项目根**执行，
> 并带上新的 `CoDefenderC3/train/` 前缀。详细用法见 `CoDefenderC3/train/README.md`。

`train/run_all.py` 提供完整的实验管线，支持双模式架构：

### 主进程模式（默认）

```bash
python CoDefenderC3/train/run_all.py
```

自动完成:
1. 数据加载（支持年份子目录 `malicious_sample/{2019~2024}/`）
2. 训练 33 个子模型
3. 校准 SDD 漂移检测 + 7 个基线方法
4. 逐月评估 + 漂移检测 + DACP + DSIR
5. 消融实验（可选）
6. 生成图表 + HTML 报告 + JSON/CSV 结果

### 子进程模式

```bash
python CoDefenderC3/train/run_all.py --single <data_dir>
```

⚠ `--single` 会把 `config.MODEL_PATH` 改写成 `results/<dataset>/ensemble.pkl`，
**不会**覆盖仓库根的 `ensemble.pkl`。

### 前置条件

1. 修改 `config.py` 中的 `DATA_DIR` 指向样本目录
2. 确保样本目录结构: `DATA_DIR/{year}/` (如 2019, 2020, ...)
3. 对于 EMBER 数据，需要 `metadata.json` 包含 `feature_ranges` 字段

---

## 配置参考

核心配置文件: `config.py`

```python
# 路径配置
DATA_DIR    = os.environ.get("CODEFENDER_DATA_DIR", r"D:\VM_Share\testsamples\CoDefenderC3_data")
MODEL_PATH  = os.path.join(ROOT, "ensemble.pkl")   # 默认: ./ensemble.pkl

# 实验参数
RUN_ABLATION = True    # 是否运行消融实验
SAVE_MODEL   = True    # 是否保存训练好的模型
N_WORKERS    = 16      # 并行工作线程数
SEED         = 42      # 随机种子

# 视图组定义: VIEW_GROUPS (16 个视角组 × 33 个模型)
```

---

## 项目结构

```
CoDefenderC3_v8/
├── engine/                      # API 服务层
│   ├── app.py                   # FastAPI 应用 (8 个端点 + 生产加固)
│   ├── inference.py             # 推理核心 + PE 验证
│   ├── result_formatter.py      # 结果格式化
│   └── packer_detector.py       # 加壳检测
├── core/                        # SDD 漂移检测引擎
│   ├── sdd_engine.py            # 谱漂移分解核心
│   ├── baselines.py             # 7 个基线方法
│   ├── stats.py                 # 统计分析
│   ├── llm_module.py            # LLM 辅助分析
│   └── slfe_verify.py           # 自验证模块
├── models/                      # 33 个子模型 (模型名_年份/)
│   ├── ensemble.py              # MultiModelEnsemble 集成类
│   ├── MalConv_2017/
│   ├── MalConv2_2021/
│   ├── EmberGBDT_2018/
│   ├── MalGraph_2022/
│   └── ... (共 30+ 模型目录)
├── feature_extraction/          # 特征提取层（引擎与训练共用 → 保留在根）
│   ├── unified.py               # 15 种 FeatureType 统一入口
│   └── extract_feature.py       # BinaryNinja 48 项底层特征
├── train/                       # ← 训练侧（2026-09-16 自仓库根整理至此）
│   ├── run_all.py               # 训练 / 实验主管线
│   ├── train_minimal.py         # 增量训练 + 逐模型 checkpoint
│   ├── build_minimal_dataset.py # 与基线视角对齐的数据集构建
│   ├── verify_trained_ensemble.py  # 候选 pkl 与基线等价性校验（门禁）
│   ├── build_dataset.py         # 数据集构建（EMBER / RAW PE）
│   ├── build_rawpe.py           # 裸 PE 数据集构建
│   ├── data_loader.py           # PELoader：特征提取 / 月度缓存
│   ├── dataset_utils.py         # 数据集收尾工具
│   ├── ember_vectorizer.py      # 纯 NumPy EMBER 向量化
│   ├── _probe_ensemble.py       # 容错拆解 ensemble.pkl
│   ├── _verify_ensemble_load.py # 验证 load / 推理链路
│   ├── _inspect_ensemble.py     # 只读拆解各模型输入约定
│   ├── apply_bn_optimizations.py / benchmark_bn_optimization.py
│   ├── logs/                    # 历史运行日志
│   └── README.md                # 训练侧使用说明
├── docs/                        # ← 说明文档（2026-09-16 自仓库根整理至此）
│   ├── README.md                # 本文件
│   ├── TESTING.md               # 测试说明
│   ├── PAPERS.md                # 模型论文与架构参考
│   ├── BINARYNINJA_OPTIMIZATION_GUIDE.md
│   ├── testing_hash.md / testing_hash_simple.md
│   └── 增量训练与Checkpoint技术方案.md
├── scaner/                      # 哈希库扫描器（与 nkrepo 项目同步）
├── signatures/                  # 哈希签名库分片（sha256 / md5 / fuzzy）
├── runs/                        # 训练产物 runs/<run_id>/
├── config.py                    # 全局配置 (路径/参数/视图组)  ← 引擎与训练共用
├── model_registry.py            # 模型中心注册表               ← 引擎与训练共用
├── ensemble.pkl                 # 预训练模型权重 (834 MB)
├── requirements.txt             # Python 依赖
├── Dockerfile                   # Docker 构建文件
├── supervisord.conf             # Supervisor 主配置
├── codefender.ini               # Supervisor program 配置
└── testsamples/                 # 测试样本 (SHA256 命名)
```

> **为什么 `config.py` / `model_registry.py` / `feature_extraction/` 不搬进 `train/`**：
> 这三者被引擎侧（`engine/app.py`、`models/ensemble.py`）与训练侧共同依赖，
> 属于共享基础设施，移动会同时打断推理服务。`train/` 内的脚本通过
> 「把 `train/` 的上一级加回 `sys.path`」来继续 `import config`。

---

## 常见问题

**Q: 启动时报 `FileNotFoundError: model file not found`**
A: 检查 `ensemble.pkl` 是否在项目根目录，或设置 `CODEFENDER_MODEL_PATH` 环境变量指向正确路径。

**Q: MalGraph 加载失败**
A: 确保 `models/MalGraph_2022/` 下有 `best_model.pt` checkpoint 和 `train_external_function_name_vocab.jsonl` 词汇表文件；或在 Docker 启动时设置 `CODEFENDER_ENABLE_MALGRAPH=false` 禁用它。

**Q: Docker 镜像太大**
A: 主要体积来自 `ensemble.pkl` (834MB) 和 PyTorch (~800MB, CPU-only 已优化)。若需进一步缩小，可考虑将模型文件通过卷挂载而非打包进镜像。

**Q: 如何在 CUDA 环境下使用 GPU 加速？**
A: 将 Dockerfile 中 PyTorch 安装的 `--index-url` 改为 GPU 版本，并设置 `CODEFENDER_MALGRAPH_DEVICE=cuda`。开发环境直接安装 GPU 版 PyTorch 即可。

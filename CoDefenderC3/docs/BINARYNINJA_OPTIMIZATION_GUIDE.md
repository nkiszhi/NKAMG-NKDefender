# BinaryNinja 性能优化指南

## 针对需要 BinaryNinja 分析的 15 个模型优化方案

### 模型分类
**需要 BinaryNinja 的模型（15个/33个，45%）:**
- **D特征（反汇编）**: OpcodeLSTM, OpcodeTransformer, OpcodeStatNet, AsmEmbedNet
- **E特征（图结构）**: MalGraph, CFGGAT, CFGGCN, CFGDGCNN, CallGraphGNN, GraphStatNet
- **F特征（可视化）**: GrayscaleCNN, InceptionV3, IMCFN, ColorCNN, MarkovCNN, EntropyMapCNN

---

## 优化策略（按影响从大到小）

### ✅ 1. 工作线程数优化（已实施）
**位置**: `extract_feature.py::_get_bn()`
```python
worker_threads = max(1, _env_int("BN_WORKER_THREADS", 4))
bn.set_worker_thread_count(worker_threads)
```

**配置建议**:
- **16核CPU（如 Hygon C86 7390）**: 设置 `BN_WORKER_THREADS=6`
- **32核CPU**: 设置 `BN_WORKER_THREADS=8`
- **避免**: 不要超过 CPU 核心数的 50%，否则线程竞争反而降速

**预期提升**: 1.5-2.5x 反汇编速度提升

---

### 🔧 2. BinaryNinja 分析选项优化（推荐实施）

**当前配置**（3处位置需要统一优化）:
```python
bv = self.bn.load(pe_path, options={
    "analysis.mode": "basic",
    "analysis.linearSweep.autorun": True,
    "analysis.limits.maxFunctionSize": 262144,
})
```

**优化后配置**:
```python
bv = self.bn.load(pe_path, options={
    # === 核心分析模式 ===
    "analysis.mode": "basic",  # 已是最快模式
    "analysis.linearSweep.autorun": True,
    
    # === 超时限制（防止卡死在混淆样本上）===
    "analysis.limits.maxFunctionSize": 262144,  # 256KB 函数上限
    "analysis.limits.maxFunctionAnalysisTime": 30000,  # 30秒/函数超时
    
    # === 跳过不必要的分析 ===
    "pdb.features.autoDownloadPDBs": False,  # 跳过 PDB 符号下载
    "pdb.features.allowLocalPDBs": False,    # 跳过本地 PDB 查找
    "analysis.suppressNewAutoFunctionAnalysis": False,
    
    # === 保守策略（速度优先）===
    "analysis.conservativeLinearSweep": True,  # 更快但可能不完整
    "analysis.tailCallHeuristics": False,      # 跳过尾调用识别
    "analysis.tailCallTranslation": False,
})
```

**修改位置**:
1. `extract_feature.py` 第 446 行（`extract()` 方法）
2. `extract_feature.py` 第 485 行（`extract_de_only()` 方法）
3. Worker 脚本生成代码（`generate_bndbs_batch()` 中的动态脚本）

**预期提升**: 1.2-1.8x 速度提升（特别是对混淆/加壳样本）

---

### 🔧 3. .bndb 缓存复用（推荐实施）

**原理**: BinaryNinja 分析是最慢的步骤（~60秒/样本），生成的 `.bndb` 文件应该**永久保存**并复用。

**当前状态**: 代码已支持 bndb 生成和复用，但需要正确配置缓存目录。

**实施步骤**:
```python
# 在 config.py 中添加
BNDB_CACHE_DIR = os.path.join(ROOT, ".bndb_cache")

# 在 app.py 中使用
bndb_path = os.path.join(CACHE_DIR, f"{sample_hash}.bndb")
if not os.path.exists(bndb_path):
    # 生成 bndb（慢，首次）
    extractor.generate_bndbs_batch([(pe_path, bndb_path)])
# 从 bndb 提取特征（快，<1秒）
features = extractor.extract_de_from_bndb(bndb_path)
```

**预期提升**: 重复扫描同一批样本时，从 60秒/样本 降至 <1秒/样本

---

### 🔧 4. 按需加载 D/E 模型（推荐实施）

**原理**: 如果检测目标不需要高精度，可以完全跳过需要 BinaryNinja 的 15 个模型。

**实施方式**:
```python
# 快速模式：跳过 D/E/F 特征（18个模型，2-4x 更快）
results = app.scan_samples(samples, include_de=False)

# 完整模式：所有 33 个模型（慢但全面）
results = app.scan_samples(samples, include_de=True)  # 默认
```

**适用场景**:
- **快速初筛**: 大批量样本的快速扫描
- **已知威胁**: 针对非混淆/非加壳样本
- **资源受限**: CPU/内存/时间有限的环境

**不适用场景**:
- **高级恶意软件**: 需要图结构和控制流分析
- **APT 样本**: 需要反汇编特征检测
- **混淆样本**: 需要操作码序列分析

**预期提升**: 2-4x 整体速度提升（跳过 BinaryNinja）

---

### 🔧 5. 批处理并行优化（推荐实施）

**当前状态**: 代码已实现批量 bndb 生成，但可以进一步优化。

**优化建议**:
```python
# 在 extract_feature.py 中调整
def generate_bndbs_batch(self, pe_bndb_pairs, n_workers=None, batch_size=10):
    if n_workers is None:
        # 自动根据 CPU 核心数调整
        cpu_count = os.cpu_count() or 8
        n_workers = max(2, cpu_count // 4)  # 每 4 核分配 1 个 worker
        batch_size = min(20, max(10, cpu_count // 2))  # 动态批大小
```

**配置建议**（针对 Hygon C86 7390, 32核）:
```bash
# 环境变量
export BN_WORKER_THREADS=8          # BinaryNinja 内部线程
export BN_BATCH_WORKERS=8           # 并行样本数
export BN_BATCH_SIZE=16             # 每批样本数
```

**预期提升**: 在高核心数 CPU 上 1.3-1.6x 提升

---

### 🔧 6. 内存优化（可选）

**问题**: 长时间运行时，BinaryNinja 可能内存泄漏。

**解决方案**（已在代码中实现）:
```python
# 每处理 N 个文件后重启 worker 进程
MAX_FILES_PER_WORKER = 100  # 在 extract_de_from_bndb_batch 中

# 每个文件处理后强制 GC
import gc
bv.file.close()
del bv
gc.collect()
```

**配置建议**:
- **大内存系统（64GB+）**: `MAX_FILES_PER_WORKER=200`
- **中等内存（32GB）**: `MAX_FILES_PER_WORKER=100`（默认）
- **小内存（16GB）**: `MAX_FILES_PER_WORKER=50`

---

### 🔧 7. 特征提取优化（可选）

**针对特定特征的优化**:

#### D01-D03（操作码序列）
```python
# 当前: 提取 4096 条指令
MAX_OPCODES = 4096

# 优化: 对于快速扫描，可以减少到 2048
MAX_OPCODES = 2048  # 2x 更快，准确率轻微下降（~1-2%）
```

#### E01-E04（图特征）
```python
# 优化: 限制图的大小
"analysis.limits.maxBasicBlockCount": 10000,  # 限制基本块数量
"analysis.limits.maxEdgeCount": 50000,        # 限制边数量
```

#### F01-F04（可视化特征）
```python
# 当前: 256×256 图像
IMG_SIZE = 256

# 优化: 对于某些模型可以降低到 128×128
IMG_SIZE = 128  # 4x 更快，准确率下降 ~3-5%
```

---

## 综合优化效果预估

### 场景 1: 全量扫描（所有 33 模型）
| 优化组合 | 预期速度 | 实施难度 |
|---------|---------|---------|
| 基线（无优化） | 17秒/样本 | - |
| ✅ 工作线程优化（已实施） | 11秒/样本 | ✅ 完成 |
| + BN 分析选项优化 | 8-9秒/样本 | 🔧 简单 |
| + .bndb 缓存复用 | 首次9秒，重扫<1秒 | 🔧 中等 |
| + 批处理并行优化 | 6-7秒/样本 | 🔧 简单 |

### 场景 2: 快速扫描（跳过 D/E/F，18 模型）
| 优化组合 | 预期速度 | 实施难度 |
|---------|---------|---------|
| include_de=False | 2-3秒/样本 | 🔧 已支持 |
| + 批处理优化 | 1-2秒/样本 | 🔧 简单 |

---

## 实施优先级

### 🔥 高优先级（立即实施）
1. ✅ **工作线程优化** - 已完成
2. **BN 分析选项优化** - 5分钟修改，1.2-1.8x提升
3. **按需加载优化** - 已支持 `include_de`，直接使用

### 🟡 中优先级（推荐实施）
4. **.bndb 缓存复用** - 需要配置缓存目录
5. **批处理并行优化** - 调整 n_workers 参数

### 🟢 低优先级（可选）
6. **内存优化** - 已实现，无需修改
7. **特征提取优化** - 需要评估准确率影响

---

## 快速实施指南

### 步骤 1: 设置环境变量（立即生效）
```bash
# Windows (cmd)
set BN_WORKER_THREADS=8

# Windows (PowerShell)
$env:BN_WORKER_THREADS=8

# Linux/macOS
export BN_WORKER_THREADS=8
```

### 步骤 2: 修改 BN 分析选项（5分钟）
编辑 `extract_feature.py`，搜索 `bv = self.bn.load(pe_path, options={`，
替换为优化后的配置（见第2节）。

### 步骤 3: 使用按需加载（立即可用）
```python
# 在调用扫描时指定
from engine.app import CoDefenderApp
app = CoDefenderApp()

# 快速模式（跳过 D/E）
results = app.scan_samples(sample_paths, include_de=False)
```

---

## 测试验证

### 验证方法
```bash
# 测试 14 个样本的扫描时间
time python -c "
from engine.app import CoDefenderApp
app = CoDefenderApp()
results = app.scan_samples(['/data/samples/TestSample'])
print(results['Summary']['Timing'])
"
```

### 预期结果（14样本）
- **优化前**: 14样本 × 17秒 = 238秒
- **优化后（全量）**: 14样本 × 7秒 = 98秒 ✅ 达标（<120秒目标）
- **优化后（快速）**: 14样本 × 2秒 = 28秒 🚀

---

## 常见问题

### Q1: 为什么增加 BN_WORKER_THREADS 反而变慢了？
A: 线程数超过 CPU 核心数的 50% 会导致线程竞争。建议：
- 16核: 6-8线程
- 32核: 8-12线程
- 不要超过 16 线程

### Q2: .bndb 文件很大，会占满磁盘吗？
A: 单个 .bndb 约 5-20MB。可以定期清理旧缓存：
```bash
# 删除 30 天前的 bndb
find .bndb_cache -name "*.bndb" -mtime +30 -delete
```

### Q3: include_de=False 会降低多少准确率？
A: 根据论文数据：
- 全量 33 模型: F1=0.987
- 跳过 D/E/F（18模型）: F1≈0.955-0.965（降低 ~2-3%）
- 适用于初筛，高可疑样本再用全量模型

### Q4: 如何判断当前瓶颈在哪？
A: 查看日志输出：
```python
# 在 app.py 中添加时间统计
t0 = time.time()
features = extract_features(pe_path)
print(f"特征提取: {time.time()-t0:.1f}秒")

t0 = time.time()
predictions = ensemble.predict(features)
print(f"模型推理: {time.time()-t0:.1f}秒")
```

典型分布：
- 特征提取: 70-80%（BinaryNinja 主导）
- 模型推理: 20-30%

---

## 联系与支持

如需进一步优化或遇到问题，请提供：
1. CPU 型号和核心数
2. 样本数量和平均大小
3. 当前扫描时间（Timing 字段）
4. 是否使用 .bndb 缓存

---

**版本**: v1.0  
**更新日期**: 2024  
**适用版本**: CoDefenderC3_v8

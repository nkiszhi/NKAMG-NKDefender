# CoDefenderC3

多视图 PE 恶意软件检测系统（33 子模型集成 / 15 特征视角 / BinaryNinja + PyTorch）。

> **说明文档已归档到 [`docs/`](docs/)，训练代码已归档到 [`train/`](train/)（2026-09-16 整理）。**
> 本文件仅作入口索引。

---

## 快速开始

### 推理服务（FastAPI）

```bash
pip install -r requirements.txt
uvicorn engine.app:app --host 0.0.0.0 --port 8000
# 健康检查
curl http://127.0.0.1:8000/health
```

开发环境也支持 `./start_codefender.sh`（Linux）或 Docker：

```bash
docker build -t codefender .
docker run -p 8000:8000 -v /data/signatures:/app/signatures:rw codefender
```

> ⚠️ `signatures/` 是 SQLite 分片库，**严禁挂载 hgfs/VMware 共享目录**（POSIX 锁不兼容）。

### 训练

```bash
PYTHONIOENCODING=utf-8 python -u CoDefenderC3/train/build_minimal_dataset.py --help
PYTHONIOENCODING=utf-8 python -u CoDefenderC3/train/train_minimal.py --help
PYTHONIOENCODING=utf-8 python -u CoDefenderC3/train/verify_trained_ensemble.py --help
```

详见 [`train/README.md`](train/README.md)。

---

## 目录导航

| 路径 | 内容 |
|---|---|
| [`docs/`](docs/) | 全部说明文档（见下方索引） |
| [`train/`](train/) | 训练侧代码：数据集构建、集成训练、checkpoint、复验 |
| `engine/` | FastAPI 推理服务（8 个端点 + 生产加固） |
| `core/` | SDD 谱漂移检测引擎 + 7 个基线方法 |
| `models/` | 33 个子模型实现 + `MultiModelEnsemble` 集成类 |
| `feature_extraction/` | 特征提取层（15 种视角 / BN 48 项底层特征）← 引擎与训练共用 |
| `scaner/` | 哈希库扫描器（与 nkrepo 项目同步） |
| `signatures/` | 哈希签名库分片（sha256 / md5 / fuzzy） |
| `runs/` | 训练产物 `runs/<run_id>/` |
| `config.py` | 全局配置（路径 / 参数 / 视图组）← 引擎与训练共用 |
| `model_registry.py` | 模型中心注册表 ← 引擎与训练共用 |
| `ensemble.pkl` | 预训练集成权重（834 MB，K=33 / V=15） |

### `docs/` 索引

| 文档 | 内容 |
|---|---|
| [`docs/README.md`](docs/README.md) | 项目总览：架构、API 端点、配置参考、常见问题 |
| [`docs/增量训练与Checkpoint技术方案.md`](docs/增量训练与Checkpoint技术方案.md) | 增量训练 + checkpoint 完整技术方案（缺陷清单 B1–B8 / 附录 C 最小数据集 / 附录 D MalGraph 接入） |
| [`docs/TESTING.md`](docs/TESTING.md) | 测试说明 |
| [`docs/PAPERS.md`](docs/PAPERS.md) | 33 个子模型的论文与架构参考 |
| [`docs/BINARYNINJA_OPTIMIZATION_GUIDE.md`](docs/BINARYNINJA_OPTIMIZATION_GUIDE.md) | BinaryNinja 分析选项优化指南 |
| [`docs/testing_hash.md`](docs/testing_hash.md) / [`docs/testing_hash_simple.md`](docs/testing_hash_simple.md) | 哈希库测试记录 |

---

## 运行纪律（Windows 开发环境）

- **必须** `PYTHONIOENCODING=utf-8`：否则中文输出崩溃且无报错信息。
- 长任务加 `python -u`：stdout 重定向时是块缓冲，print 长时间不可见。
- 批量特征提取加 `CODEFENDER_KEEP_BNDB=1`：避免清理动作累计触发安全删除钩子杀进程。
- **不要移动数据集目录**：`metadata.jsonl` 记的是绝对路径，`per_sample` 缓存键是路径的 md5，一搬就全失效。

---

## License

见 [`license.txt`](license.txt)。

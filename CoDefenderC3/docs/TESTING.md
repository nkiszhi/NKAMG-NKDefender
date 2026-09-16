# CoDefenderC3 Docker 部署与接口测试指南

---

## 1. 导入镜像

```bash
docker load -i codefender_latest.tar.gz
```

导入后确认镜像已加载：

```bash
docker images codefender
```

---

## 2. 启动容器

```bash
docker run -d --name codefender \
  -v /mnt/hgfs/VM_Share/testsamples:/data/samples:ro \
  -p 8000:8000 \
  codefender:latest
```

| 参数 | 说明 |
|------|------|
| `-d` | 后台运行 |
| `--name codefender` | 容器名称 |
| `-v /mnt/hgfs/VM_Share/testsamples:/data/samples:ro` | 挂载宿主机样本目录（只读） |
| `-p 8000:8000` | 端口映射 |
| `codefender:latest` | 镜像名:标签 |

### 等待模型加载

模型加载约需 30~60 秒，使用 `/readyz` 就绪探针轮询等待：

```bash
while ! curl -s http://localhost:8000/readyz | grep -q '"ready":true'; do
  echo "waiting for model to load..."
  sleep 5
done
echo "Model ready!"
```

---

## 3. 接口测试

支持swigger UI 文档，访问以下地址查看 API 文档
http://localhost:8000/docs

### 3.1 存活探针 `/livez`

```bash
curl -s http://localhost:8000/livez | python -m json.tool
```

预期输出：
```json
{
  "alive": true,
  "uptime_sec": 60,
  "uptime_human": "00:01:00"
}
```

> 该端点不接触模型，即使模型加载失败也会返回 `alive: true`。仅用于检测进程是否存活。

---

### 3.2 就绪探针 `/readyz`

```bash
curl -s http://localhost:8000/readyz | python -m json.tool
```

**模型已加载时 (HTTP 200)**：
```json
{
  "ready": true,
  "model_count": 33,
  "view_count": 16
}
```

**模型未加载时 (HTTP 503)**：
```json
{
  "ready": false,
  "reason": "model not loaded",
  "model_path": "/app/ensemble.pkl"
}
```

> K8s `readinessProbe` 应指向 `/readyz`，HTTP 503 时调度器不会将流量路由到该 Pod。

---

### 3.3 运行时指标 `/metrics`

```bash
curl -s http://localhost:8000/metrics | python -m json.tool
```

预期输出（字段含义见下表）：

| 字段 | 说明 |
|------|------|
| `total_requests` | 服务启动以来总请求数 |
| `predict_single_ok` / `predict_single_error` | 单文件检测成功/失败计数 |
| `predict_dir_ok` / `predict_dir_error` | 目录扫描成功/失败计数 |
| `scan_files_total` | 累计扫描文件数 |
| `scan_files_malicious` | 累计检出恶意文件数 |
| `scan_files_benign` | 累计检出正常文件数 |
| `last_predict_ms` | 最近一次预测耗时 (ms) |
| `unhandled_errors` | 未处理异常计数 |
| `uptime_sec` | 服务运行时长 (秒) |
| `models_loaded` | 模型是否已加载 |
| `model_count` | 集成子模型数量 |

---

### 3.4 健康检查

```bash
curl -s http://localhost:8000/health | python -m json.tool
```

预期输出：

```json
{
  "status": "ok",
  "model_path": "/app/ensemble.pkl",
  "model_count": 33,
  "view_count": 16,
  "malgraph": {
    "enabled": true,
    "status": "ready",
    "device": "cpu"
  }
}
```

---

### 3.5 单文件检测

```bash
curl -s -X POST http://localhost:8000/predict-single \
  -F "file=@/mnt/hgfs/VM_Share/testsamples/<样本文件名>" \
  -F "include_de=false" | python -m json.tool
```

> 将 `<样本文件名>` 替换为实际样本文件名。

预期输出（关键字段）：

```json
{
  "MD5": "...",
  "SHA256": "...",
  "FileName": "...",
  "FileSize": ...,
  "Status": 1,
  "StatusLabel": "Malicious",
  "IsMalicious": true,
  "Score": 0.8734,
  "ScanResult": {
    "VirusName": "...",
    "Packed": {
      "IsPacked": false,
      "PackerName": ""
    },
    "PerModelScores": { ... },
    "PerModelPreds": { ... },
    "DriftEvidence": {
      "surface": 0.0,
      "structure": 0.0,
      "global": 0.0
    },
    "SampleDriftType": "normal",
    "AnomalyScore": 0.0,
    "IsDriftCandidate": false,
    "ModelStats": {
      "TotalModels": 33,
      "OkModels": 30,
      "SkippedModels": 3,
      "ErrorModels": 0,
      "MaliciousVotes": 28,
      "BenignVotes": 2,
      "AgreementRate": 0.9333,
      "EnsembleScore": 0.8734,
      "EnsemblePred": 1
    }
  }
}
```

| 核心字段 | 含义 |
|----------|------|
| `Status` | 0=Benign, 1=Malicious, 2=DriftSuspect |
| `IsMalicious` | 集成判定结果 |
| `Score` | 集成模型恶意度得分 (0~1) |
| `ModelStats.AgreementRate` | 子模型投票一致率 |
| `DriftEvidence` | 三层次漂移证据 |

---

### 3.6 病毒库信息 `/model-info`

```bash
curl -s http://localhost:8000/model-info | python -m json.tool
```

预期输出包含 `file.md5`、`ensemble.model_count`、`models[]` 等字段。
`file.md5` 用于校验模型文件完整性。

---

### 3.7 模型热更新 `/reload-model`

```bash
# 原地重载
curl -s -X POST http://localhost:8000/reload-model | python -m json.tool

# 切换到新模型
curl -s -X POST "http://localhost:8000/reload-model?model_path=/app/ensemble_v2.pkl" | python -m json.tool
```

---

### 3.8 批量目录扫描

```bash
curl -s -X POST "http://localhost:8000/predict-directory?directory=/data/samples&recursive=true&include_de=false" | python -m json.tool
```

预期输出：

```json
{
  "Summary": {
    "TotalFiles": 100,
    "ScannedFiles": 98,
    "MaliciousCount": 73,
    "BenignCount": 25,
    "ErrorCount": 2,
    "Directory": "/data/samples"
  },
  "Results": [
    {
      "MD5": "...",
      "SHA256": "...",
      "FileName": "sample1.exe",
      "FilePath": "/data/samples/sample1.exe",
      "Status": 1,
      "StatusLabel": "Malicious",
      "IsMalicious": true,
      "Score": 0.95,
      "ScanResult": { ... }
    }
  ]
}
```

---

## 4. 一键测试脚本

```bash
#!/bin/bash
# 保存为 test_all.sh，chmod +x test_all.sh 后运行

echo "====== 1/6: 存活探针 ======"
curl -s http://localhost:8000/livez | python -m json.tool

echo ""
echo "====== 2/6: 就绪探针 ======"
curl -s http://localhost:8000/readyz | python -m json.tool

echo ""
echo "====== 3/6: 运行指标 ======"
curl -s http://localhost:8000/metrics | python -m json.tool

echo ""
echo "====== 4/6: 健康检查 ======"
curl -s http://localhost:8000/health | python -m json.tool

echo ""
echo "====== 5/6: 单文件检测 ======"
curl -s -X POST http://localhost:8000/predict-single \
  -F "file=@/mnt/hgfs/VM_Share/testsamples/<样本文件名>" \
  -F "include_de=false" | python -m json.tool

echo ""
echo "====== 6/6: 批量目录扫描 ======"
curl -s -X POST "http://localhost:8000/predict-directory?directory=/data/samples&recursive=true&include_de=false" | python -m json.tool

echo ""
echo "====== 全部完成 ======"
```

---

## 5. 常用管理命令

```bash
# 查看日志
docker logs -f codefender

# 进入容器
docker exec -it codefender bash

# 停止容器
docker stop codefender

# 删除容器
docker rm codefender
```

---

## 6. 故障排查

| 症状 | 排查命令 |
|------|----------|
| 容器一直重启 | `docker logs codefender --tail 50` |
| `/health` 返回 error | 检查 `ensemble.pkl` 是否在容器 `/app/` 目录下 |
| `/readyz` 返回 503 | 模型仍在加载中，等待 30~60s 后重试 |
| 端口被占用 | `lsof -i :8000` 或 `netstat -tlnp \| grep 8000` |
| 挂载目录不可见 | `docker exec codefender ls /data/samples` |
| 并发打满/请求排队 | 查看 `/metrics` 中 `last_predict_ms` 和错误计数；调整 `CODEFENDER_MAX_CONCURRENT` |
| 模型文件损坏 | 比对 `/health` 返回的 `model_md5` 与本地文件的 MD5 |

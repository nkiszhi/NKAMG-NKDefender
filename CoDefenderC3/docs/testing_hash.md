# CoDefenderC3 哈希库预检版 部署与测试指南（testing_hash.md）

> 本文档在 [TESTING.md](TESTING.md)（模型检测流程）基础上，补充 **nkrepo SHA256 + MD5 双哈希库预检**的部署与测试流程。
> 检测链路：`sha256 库命中 → Malicious（毫秒级）` → `md5 库命中 → Malicious` → `双库未命中 → 33 模型集成检测`。

---

## 0. 版本与结果判别（先看这个）

### 检测链路

```
上传样本 → 计算 sha256
  ├─ sha256 库命中（6,258 万条） → Malicious，Source="sha256_db"
  ├─ md5 库命中（54 万条）      → Malicious，Source="md5_db"
  └─ 双库未命中                → 33 模型集成检测（无 Source 字段）
```

### 结果判别表

| 返回特征 | 含义 | 耗时量级 |
|----------|------|---------|
| `Source: "sha256_db"`，`Score: 1.0` | SHA256 签名库命中，模型被短路 | 毫秒级 |
| `Source: "md5_db"`，`Score: 1.0` | MD5 签名库命中（老样本补漏） | 毫秒级 |
| 无 `Source` 字段，`Score: 0~1` | 走 33 模型集成链路 | 秒~分钟级 |
| 日志 `[hashdb] ... 降级` | 哈希库不可用，纯模型模式（功能不回退） | - |

> **⚠️ v4 库结构（2026-09-02 起）**：sha256 分片库已按 nkrepo `migrate_drop_name.py` 瘦身为两列 `(sha256 BLOB PK, size)`，**不再存检出名称**（命中即已知恶意）。因此哈希命中的 `VirusName` 统一为通用标签 `KnownMalware`（不再返回 `Win.Trojan...` 类 ClamAV 名）；`md5.db.shards` 尚未迁移（仍含 name 列，可被 v4 正常读取，但 v4 查询不返回 name）。检测代码须与库同版（`scaner/scanner.py` v4）——旧 v3 代码查 `SELECT size,name` 会报 `no such column: name`。

---

## 1. 签名库准备（宿主机 VM，一次性）

签名库共约 4.05GB，**必须放在 VM 本地 ext4 磁盘**（如 `/data/signatures`）。
⚠️ 严禁直接挂 `/mnt/hgfs` 使用：`_meta.db` 为 SQLite 且需 RW + POSIX 文件锁，vmhgfs-fuse 锁支持不完整会导致 `database is locked`。

```bash
# 1. 建目录并交给 lm（sudo 建的目录属 root，后续 rsync 会 Permission denied）
sudo mkdir -p /data/signatures
sudo chown -R lm:lm /data/signatures

# 2. 从共享目录拷贝（--no-o --no-g 规避 hgfs 属组报错；排除 legacy/resharding 和运行时临时文件）
rsync -a --no-o --no-g \
  --exclude='*.db-shm' --exclude='*.db-wal' \
  --exclude='sha256.db.shards.legacy' --exclude='sha256.db.shards.resharding' \
  /mnt/hgfs/VM_Share/CoDefenderC3_v8/CoDefenderC3/signatures/ /data/signatures/

# 3. 验证（3 项全对才继续）
du -sh /data/signatures                       # 总量约 4.0G
ls /data/signatures/sha256.db.shards | wc -l  # 应为 257（256 分片 + _meta.db）
ls /data/signatures/md5.db.shards | wc -l     # 应为 257
```

| 目录 | 大小 | 内容 |
|------|------|------|
| `sha256.db.shards/` | 2.5GB | 256 分片 + `_meta.db`，6,258 万条签名（v4 两列瘦身，不含 name） |
| `sha256.db.bloom/` | 73MB | 分片 Bloom 位图（须与 shards 成对存在） |
| `md5.db.shards/` | 41MB | 256 分片 + `_meta.db`，54 万条签名 |
| `md5.db.bloom/` | 1.2MB | 分片 Bloom 位图 |

> ⚠️ 同步时 `sha256.db.shards` 与 `sha256.db.bloom` 必须**成对**拷贝：分片瘦身迁移只改 shards、不改 bloom（bloom 按哈希构建与 name 无关），漏拷 bloom 只影响性能（负样本短路失效，回退 SQLite 点查，功能正常）。

---

## 2. Docker 部署

### 2.1 已有镜像的增量更新（不重新 build）

代码改动共 3 处：`engine/app.py`（预检逻辑）、`config.py`（HASH_DB 配置段）、`scaner/`（新增 nkrepo 6 个模块）。

**方式 A：bind mount 覆盖（开发调试用，改代码后 `docker restart` 即生效）**

```bash
docker run -d --name codefender \
  -p 8000:8000 \
  -v /mnt/hgfs/VM_Share/CoDefenderC3_v8/CoDefenderC3/engine/app.py:/app/engine/app.py:ro \
  -v /mnt/hgfs/VM_Share/CoDefenderC3_v8/CoDefenderC3/config.py:/app/config.py:ro \
  -v /mnt/hgfs/VM_Share/CoDefenderC3_v8/CoDefenderC3/scaner:/app/scaner:ro \
  -v /data/signatures:/app/signatures:rw \
  -v /mnt/hgfs/VM_Share/test_samples:/test_samples:ro \
  <原镜像名>
```

> ⚠️ bind mount 的文件**不会**进入 `docker commit`。固化必须走方式 B。
> ⚠️ 单文件挂载的 inode 陷阱：vim 等"重命名替换"方式保存后容器仍读旧 inode，改完代码务必 `docker restart` 并确认宿主机文件已落盘。

**方式 B：docker commit 固化（测试通过后的正式方式）**

```bash
# 1. 用原镜像建临时容器（不启动）
docker create --name cd_solid <原镜像名>

# 2. 把 3 处代码拷进容器文件系统层
docker cp /mnt/hgfs/VM_Share/CoDefenderC3_v8/CoDefenderC3/scaner/.      cd_solid:/app/scaner/
docker cp /mnt/hgfs/VM_Share/CoDefenderC3_v8/CoDefenderC3/engine/app.py cd_solid:/app/engine/app.py
docker cp /mnt/hgfs/VM_Share/CoDefenderC3_v8/CoDefenderC3/config.py     cd_solid:/app/config.py

# 3. 提交新 tag（仅新增几百 KB 代码层，秒级完成）
docker commit -m "hashdb: sha256+md5 precheck" cd_solid codefender:hashdb_v1
docker rm cd_solid

# 4. 备份镜像（可选）
docker save codefender:hashdb_v1 | gzip > ~/codefender_hashdb_v1.tar.gz
```

### 2.2 正式启动（固化镜像，双挂载）

```bash
docker run -d --name codefender \
  -p 8000:8000 \
  -v /data/signatures:/app/signatures:rw \
  -v /mnt/hgfs/VM_Share/test_samples:/test_samples:ro \
  --restart unless-stopped \
  codefender:hashdb_v1
```

| 挂载 | 模式 | 说明 |
|------|------|------|
| `/data/signatures:/app/signatures` | **:rw 必须可写** | `_meta.db` 启动时建表需要写权限；VM 本地 ext4，不可用 hgfs 路径 |
| `/mnt/hgfs/VM_Share/test_samples:/test_samples` | :ro | 测试样本目录，Windows 侧投放立即可见，无需重启 |

### 2.3 环境变量（可选）

| 变量 | 默认 | 说明 |
|------|------|------|
| `CODEFENDER_HASH_DB_ENABLED` | `1` | 置 `0` 关闭哈希预检，回退纯模型模式 |
| `CODEFENDER_HASH_DB_DIR` | `/app/signatures` | 签名库目录 |
| `CODEFENDER_HASH_DB_LAYOUT` | `hex` | **勿改**，与建库参数一致，否则触发全量重分片（62M 条约 311s） |
| `CODEFENDER_HASH_DB_FP_RATE` | `0.01` | **勿改**，同上 |
| `CODEFENDER_HASH_DB_MAX_OPEN` | `16` | **勿改**，同上 |
| `CODEFENDER_HASH_DB_SHARDS` | `4` | hex 布局下内部强制 256 片，传值无害 |
| `CODEFENDER_MAX_BATCH_FILES` | `500` | `/predict-directory` 批量文件数上限 |
| `CODEFENDER_MAX_UPLOAD_BYTES` | `134217728` | `/predict-single` 单文件上传上限（128MB） |

---

## 3. 启动验证

```bash
# 1. 确认双哈希库加载（最关键的一步）
docker logs codefender 2>&1 | grep -i hashdb
# 预期两行：
#   [hashdb] SHA256 签名库就绪: 62,587,049 条
#   [hashdb] MD5 签名库就绪: 540,169 条
# 若出现 "降级为纯模型检测" → 检查挂载路径与 :rw 权限（见第 7 节排查表）

# 2. 哈希命中不依赖模型，进程起来即可测；模型加载仍需等待
while ! curl -s http://localhost:8000/readyz | grep -q '"ready":true'; do
  echo "waiting for model to load..."
  sleep 5
done
echo "Model ready!"

# 3. 健康检查
curl -s http://localhost:8000/health | python3 -m json.tool
```

> 探针/指标接口（`/livez`、`/readyz`、`/metrics`、`/model-info`、`/reload-model`）用法与 TESTING.md 第 3.1~3.7 节完全一致，此处不再重复。

---

## 4. 接口测试（哈希预检部分）

### 4.1 单文件检测 —— 哈希命中

```bash
curl -s -X POST http://localhost:8000/predict-single \
  -F "file=@/mnt/hgfs/VM_Share/test_samples/<已知恶意样本>.exe" | python3 -m json.tool
```

预期输出（哈希命中，**毫秒级返回，模型未加载也能出结果**）：

```json
{
  "MD5": "...",
  "SHA256": "...",
  "CRC32": "........",
  "FileName": "<已知恶意样本>.exe",
  "FileSize": 123456,
  "Status": 1,
  "StatusLabel": "Malicious",
  "IsMalicious": true,
  "Score": 1.0,
  "Source": "sha256_db",
  "ScanResult": {
    "VirusName": "KnownMalware",
    "Packed": {"IsPacked": false, "PackerName": "", "IsArchive": false, "ArchiveName": ""},
    "PerModelScores": {},
    "PerModelPreds": {},
    "DriftEvidence": {"surface": 0.0, "structure": 0.0, "global": 0.0},
    "SampleDriftType": "normal",
    "AnomalyScore": 0.0,
    "IsDriftCandidate": false,
    "HashMatch": {
      "engine": "...", "type": "...",
      "detail": "..."
    }
  }
}
```

| 字段 | 哈希命中时的值 | 说明 |
|------|---------------|------|
| `Status` / `StatusLabel` | `1` / `Malicious` | 与模型输出同构 |
| `Score` | `1.0` | 签名命中即满 |
| `Source` | `"sha256_db"` 或 `"md5_db"` | **哈希命中专属字段**，模型输出无此字段 |
| `VirusName` | `"KnownMalware"`（通用标签） | v4 库不再存检出名称，命中即已知恶意 |
| `ScanResult.HashMatch` | 签名记录 `{engine,type,detail}` | **哈希命中专属字段**（不含 name/size） |
| `PerModelScores` / `PerModelPreds` / `ModelStats` | 空 / 缺省 | 模型被短路，未参与 |

### 4.2 单文件检测 —— 未命中走模型

```bash
curl -s -X POST http://localhost:8000/predict-single \
  -F "file=@/mnt/hgfs/VM_Share/test_samples/<普通样本>.exe" | python3 -m json.tool
```

预期输出与 TESTING.md 第 3.5 节**完全一致**（含 `ModelStats`、`DriftEvidence`），且**无 `Source` 字段**。

### 4.3 批量目录扫描

```bash
curl -s -X POST "http://localhost:8000/predict-directory?directory=/test_samples&recursive=true&include_de=false" \
  | python3 -m json.tool
```

预期输出（结构与 TESTING.md 3.8 节一致，`MaliciousCount` 统计**已包含哈希命中**）：

```json
{
  "Summary": {
    "TotalFiles": 100,
    "ScannedFiles": 98,
    "MaliciousCount": 73,
    "BenignCount": 25,
    "ErrorCount": 2,
    "Directory": "/test_samples"
  },
  "Results": [
    {
      "MD5": "...", "SHA256": "...",
      "FileName": "known_malware.exe",
      "FilePath": "/test_samples/known_malware.exe",
      "Status": 1, "StatusLabel": "Malicious", "IsMalicious": true, "Score": 1.0,
      "Source": "sha256_db",
      "ScanResult": { "VirusName": "...", "HashMatch": { ... } }
    },
    {
      "MD5": "...", "SHA256": "...",
      "FileName": "unknown.exe",
      "FilePath": "/test_samples/unknown.exe",
      "Status": 0, "StatusLabel": "Benign", "IsMalicious": false, "Score": 0.12,
      "ScanResult": { "...模型完整输出..." : "" }
    }
  ]
}
```

> 注意：`directory` 参数是**容器内路径**（`/test_samples`），不是宿主机路径。
> 哈希命中项带 `FilePath`；全部命中时跳过模型加载，整目录毫秒级返回。

### 4.4 确定性命中闭环测试（可选，验证预检链路）

手头样本不确定是否在库内时，可先注入其 sha256 再上传，验证短路链路（在 VM 上执行）：

```bash
python3 - <<'EOF'
import sys, hashlib
sys.path.insert(0, "/mnt/hgfs/VM_Share/CoDefenderC3_v8/CoDefenderC3/scaner")
from scanner import HashSignatureDB
p = "/mnt/hgfs/VM_Share/test_samples/<测试样本>.exe"
data = open(p, "rb").read()
db = HashSignatureDB("/data/signatures/sha256.db", shard_count=4,
                     bloom_fp_rate=0.01, max_open_shards=16, layout="hex")
db.add_hash(hashlib.sha256(data).hexdigest(), len(data), "Test.Hashdb.Sample")
print("injected sha256:", hashlib.sha256(data).hexdigest())
EOF
```

随后按 4.1 上传同一文件，应看到 `VirusName: "KnownMalware"` / `Source: "sha256_db"` / 毫秒级返回。
（v4 起 `add_hash` 的 `name` 参数仅兼容旧调用方、**不再入库**，故命中不返回注入的名称。）

> ⚠️ `add_hash` **写入真实签名库**（分片 + Bloom 同步落盘），测试签名永久留存。建议仅注入一个并用 `Test.` 前缀命名；需要纯净库则验证后从备份恢复。

---

## 5. 一键测试脚本

```bash
#!/bin/bash
# 保存为 test_all_hash.sh，chmod +x 后运行
BASE=http://localhost:8000

echo "====== 1/7: 哈希库加载确认 ======"
docker logs codefender 2>&1 | grep -i hashdb

echo ""
echo "====== 2/7: 存活探针 ======"
curl -s $BASE/livez | python3 -m json.tool

echo ""
echo "====== 3/7: 就绪探针 ======"
curl -s $BASE/readyz | python3 -m json.tool

echo ""
echo "====== 4/7: 健康检查 ======"
curl -s $BASE/health | python3 -m json.tool

echo ""
echo "====== 5/7: 单文件检测（哈希命中，应含 Source 字段）======"
curl -s -X POST $BASE/predict-single \
  -F "file=@/mnt/hgfs/VM_Share/test_samples/<已知恶意样本>.exe" | python3 -m json.tool

echo ""
echo "====== 6/7: 单文件检测（未命中走模型，应无 Source 字段）======"
curl -s -X POST $BASE/predict-single \
  -F "file=@/mnt/hgfs/VM_Share/test_samples/<普通样本>.exe" | python3 -m json.tool

echo ""
echo "====== 7/7: 批量目录扫描 ======"
curl -s -X POST "$BASE/predict-directory?directory=/test_samples&recursive=true&include_de=false" \
  | python3 -m json.tool

echo ""
echo "====== 全部完成 ======"
```

---

## 6. 常用管理命令

```bash
# 查看日志（重点看 [hashdb] 前缀行）
docker logs -f codefender

# 修改代码后重启（bind mount 方式）
docker restart codefender

# 容器内进程级重启（不动容器）
docker exec codefender supervisorctl restart codefender

# 进入容器 / 停止 / 删除
docker exec -it codefender bash
docker stop codefender
docker rm codefender
```

---

## 7. 故障排查

| 症状 | 原因与排查 |
|------|-----------|
| 日志 `SHA256 库不可用，双库降级为纯模型检测` | 挂载路径错误或不可读；`docker exec codefender ls /app/signatures` 确认；检查是否漏了 `:rw` |
| 日志 `MD5 库不可用，降级（仅 SHA256 预检）` | md5.db.shards 缺失/损坏，仅影响 MD5 补漏能力 |
| 启动卡住数分钟 + 日志出现"分片重排" | 构造参数与建库不一致触发全量重分片：确认未改 `CODEFENDER_HASH_DB_LAYOUT/FP_RATE/MAX_OPEN`，确认库里没有混入 legacy 单文件库 |
| `database is locked` / `_meta.db` 报错 | signatures 挂到了 hgfs 路径（锁不完整），或卷只读；必须是本地 ext4 + `:rw` |
| rsync `Permission denied` | `sudo mkdir` 后目录属 root：`sudo chown -R lm:lm /data/signatures` 后重跑 |
| rsync `chgrp failed` | hgfs 属组映射问题：rsync 加 `--no-o --no-g` |
| 哈希命中的文件被模型判为 Benign | 属预期：预检命中即 Malicious 直出，不再走模型（签名库优先级最高） |
| `/predict-directory` 返回 400 too many files | 目录超过 500 个 PE：加 `-e CODEFENDER_MAX_BATCH_FILES=2000` |
| 上传报 413 | 单文件超过 128MB：走容器内路径（directory 方式）或调 `CODEFENDER_MAX_UPLOAD_BYTES` |
| 想临时关闭哈希预检对比模型效果 | 容器加 `-e CODEFENDER_HASH_DB_ENABLED=0` 重启 |
| 其余模型链路问题 | 见 TESTING.md 第 6 节 |

---

## 8. 上线前检查清单

- [ ] `/data/signatures` 在 VM 本地 ext4（非 hgfs），属主 `lm`，双库各 257 文件
- [ ] 容器挂载 `signatures:rw` + `test_samples:ro`，启动日志双库 count 正常（62,587,049 / 540,169）
- [ ] 已知恶意样本 → `Source: "sha256_db"`，毫秒级返回
- [ ] 普通样本 → 无 `Source`，模型完整输出正常（回归原有链路）
- [ ] 批量扫描 `MaliciousCount` = 哈希命中数 + 模型检出数
- [ ] 确认是否残留 `Test.Hashdb.Sample` 测试签名（按需清理）
- [ ] `docker save` 备份 `codefender:hashdb_v1` 镜像

# Codefender 多模型恶意文件检测

Codefender 是一个独立的 PE 样本恶意文件检测 Web 项目。系统由三部分组成：

- 前端：Vue3 + Element Plus，提供样本上传、基础信息展示、PE 结构展示和模型结果展示。
- 后端：FastAPI，负责文件上传、PE 结构校验、基础信息计算、模型服务适配。
- 模型服务：Docker 镜像 `peng233/codefender:v2`，提供 33 个检测模型的扫描能力。

用户上传待检样本后，后端会先判断文件是否为有效 PE 样本；只有通过 PE 结构校验的文件才会提交给模型服务检测。

## 检测流程

1. 用户在网页上传待检 PE 样本。
2. 后端计算 SHA256、MD5、文件大小、文件类型。
3. 后端校验 PE 结构：`MZ` 头、`PE\0\0` 签名、COFF 文件头、PE32/PE32+ 可选头。
4. 后端提取 PE 基础信息：架构、入口点、镜像基址、子系统、节数量、编译时间、节表等。
5. 后端把有效 PE 样本提交给 Codefender Docker 模型服务。
6. 前端展示基础信息、PE 结构信息、33 个模型结果和集成结果。

非 PE 文件会被后端拒绝，不会继续提交给模型检测。

## 支持模型

当前模型清单共 33 个：

1. MalConv
2. MalConv2
3. ByteTransformer
4. SaxeBerlinDNN
5. EmberDNN
6. NatarajKNN
7. DrebinSVM
8. DL4MD
9. PEMinerRF
10. DrebinImport
11. ALOHANet
12. EmberGBDT_Import
13. RaffFeatureNet
14. DrebinString
15. OpcodeLSTM
16. OpcodeTransformer
17. OpcodeStatNet
18. AsmEmbedNet
19. MalGraph
20. CFGGAT
21. CFGGCN
22. CFGDGCNN
23. CallGraphGNN
24. GraphStatNet
25. GrayscaleCNN
26. InceptionV3
27. IMCFN
28. ColorCNN
29. MarkovCNN
30. EntropyMapCNN
31. HashEmbedNet
32. MetadataNet
33. ResourceNet

前端页面会通过后端 `/models` 接口显示模型清单。

## 目录结构

```text
backend/   FastAPI 后端，提供上传检测、PE结构分析和模型服务适配
frontend/  Vue3 + Element Plus 单页前端
config.ini 后端、Docker 模型服务和跨容器路径配置
```

## 环境要求

- Docker Desktop 或 Docker Engine
- Python 3.10+
- Node.js 18+ 和 npm
- Windows PowerShell、Linux shell 或 macOS shell 均可运行

## 部署步骤

### 1. 获取代码

```powershell
cd C:\zjp\github
git clone https://github.com/nkiszhi/codefender.git
cd C:\zjp\github\codefender
```

### 2. 启动 Codefender 模型服务

拉取公开 Docker 镜像：

```powershell
docker pull peng233/codefender:v2
```

启动模型服务：

```powershell
docker run -d --name codefender-api -p 8001:8000 peng233/codefender:v2
```

验证模型服务：

```powershell
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:8001/openapi.json
```

如果容器名不是 `codefender-api`，需要修改根目录 `config.ini`：

```ini
[docker]
container_name = 实际容器名
container_upload_dir = /tmp/codefender_uploads
```

查看实际容器名：

```powershell
docker ps
```

### 3. 配置后端

根目录 `config.ini` 默认配置如下：

```ini
[server]
host = 0.0.0.0
port = 5005
cors_origins = *
max_upload_size = 104857600

[paths]
upload_dir = backend/uploads

[codefender]
api_base = http://127.0.0.1:8001
scan_endpoint = /scan/file
predict_endpoint = /predict
timeout = 300
threshold = 0.5
host_path_prefix =
container_path_prefix =

[docker]
container_name = codefender-api
container_upload_dir = /tmp/codefender_uploads
```

默认模式不需要给 Docker 挂载目录。后端会把上传样本复制到模型容器的 `container_upload_dir`，再调用 `/scan/file` 检测。

如果你希望使用目录挂载方式，也可以这样启动 Docker：

```powershell
docker run -d --name codefender-api -p 8001:8000 `
  -v "C:\zjp\github\codefender\backend\uploads:/samples/uploads" `
  peng233/codefender:v2
```

同时把 `config.ini` 改成：

```ini
[codefender]
host_path_prefix = C:\zjp\github\codefender\backend\uploads
container_path_prefix = /samples/uploads
```

### 4. 启动后端

```powershell
cd C:\zjp\github\codefender\backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

默认后端地址：

```text
http://127.0.0.1:5005
```

验证后端：

```powershell
curl http://127.0.0.1:5005/health
curl http://127.0.0.1:5005/models
```

### 5. 启动前端

```powershell
cd C:\zjp\github\codefender\frontend
npm install
npm run dev
```

默认前端地址：

```text
http://127.0.0.1:5173
```

打开浏览器访问该地址，上传 PE 样本即可检测。

## 生产构建

前端构建：

```powershell
cd C:\zjp\github\codefender\frontend
npm run build
```

构建产物在：

```text
frontend/dist
```

后端可使用 `uvicorn` 启动：

```powershell
cd C:\zjp\github\codefender\backend
uvicorn main:app --host 0.0.0.0 --port 5005
```

## 常用接口

- `GET /health`：后端健康检查
- `GET /models`：返回 33 个模型名称
- `POST /detect`：上传 PE 文件并返回基础信息、PE 结构信息和模型检测结果

## 排查问题

### 前端能上传，但模型检测没有结果

优先检查模型 Docker 是否正常：

```powershell
docker ps
curl http://127.0.0.1:8001/health
```

再确认 `config.ini` 的容器名是否正确：

```ini
[docker]
container_name = codefender-api
```

如果 `docker ps` 显示的容器名不是 `codefender-api`，把 `container_name` 改成实际名称，然后重启后端。

### 上传后提示不是 PE 样本

后端会校验 `MZ` 头、`PE\0\0` 签名和可选头结构。普通文本、压缩包、PDF、损坏样本会被拒绝。

### Docker 端口冲突

如果本机 8001 已被占用，可以换一个宿主机端口：

```powershell
docker run -d --name codefender-api -p 8011:8000 peng233/codefender:v2
```

同时修改：

```ini
[codefender]
api_base = http://127.0.0.1:8011
```

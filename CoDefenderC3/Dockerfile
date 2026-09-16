FROM multiscan_basic_v3:with_binja


# ── 工作目录 ──────────────────────────────────────────────
WORKDIR /app

# ── 安装 PyTorch CPU-only + torchvision ────────────────────
RUN pip3 install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch>=2.0 \
        torchvision

# ── 安装 PyTorch Geometric ────────────────────────────────
RUN pip3 install --no-cache-dir --timeout 300 \
        -i https://mirrors.cloud.tencent.com/pypi/simple \
        --trusted-host mirrors.cloud.tencent.com \
        torch_geometric

# ── 安装其余 Python 依赖 ──────────────────────────────────
COPY requirements.txt .
RUN pip3 install --no-cache-dir --timeout 300 \
        -i https://mirrors.cloud.tencent.com/pypi/simple \
        --trusted-host mirrors.cloud.tencent.com \
        -r requirements.txt

# ── BinaryNinja license + startup script ──────────────────
COPY license.txt /root/.binaryninja/license.dat
COPY start_codefender.sh /app/
RUN chmod +x /app/start_codefender.sh
# ── 复制项目代码 ──────────────────────────────────────────
COPY config.py data_loader.py model_registry.py ember_vectorizer.py ./
COPY engine/             ./engine/
COPY core/               ./core/
COPY feature_extraction/ ./feature_extraction/
COPY models/             ./models/

# ── 模型文件（~834 MB，与代码层分离以利用构建缓存）─────
COPY ensemble.pkl ./ensemble.pkl

# ── supervisor 配置 ───────────────────────────────────────
COPY supervisord.conf /etc/supervisord.conf
COPY codefender.ini /etc/supervisord.d/codefender.ini
COPY MultiScanPoint.ini /etc/supervisord.d/MultiScanPoint.ini
RUN mkdir -p /var/log/supervisor /etc/supervisord.d /home/SCAN

# ── 环境变量（限制 OpenBLAS/OMP 线程数，避免容器进程数限制报错）──
ENV OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    CODEFENDER_MODEL_PATH=/app/ensemble.pkl

# ── 端口 ──────────────────────────────────────────────────
EXPOSE 8000

# ── 健康检查 ──────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

# ── 启动 ──────────────────────────────────────────────────
CMD ["/usr/local/bin/supervisord", "-n", "-c", "/etc/supervisord.conf"]

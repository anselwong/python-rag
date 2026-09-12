# 构建独立生产镜像；不复制本地 .venv，也不依赖服务器全局 Python。
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
    PIP_DEFAULT_TIMEOUT=120 \
    RAG_DATA_DIR=/var/lib/rag \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken \
    HF_HOME=/var/lib/rag/models/huggingface \
    DOCLING_ARTIFACTS_PATH=/var/lib/rag/models/docling

WORKDIR /app

# Docling 的默认依赖解析会在 Linux 上选择带 CUDA 的 PyTorch 包，使纯 CPU 的 API
# 镜像平白膨胀数 GB。先从官方 CPU wheel 索引安装 torch/torchvision，后续解析
# Docling 依赖时复用已安装的满足版本；生产环境不携带 GPU 运行时。
# 本层不依赖应用源码或 pyproject，日常改业务代码不会导致重新下载 CPU 推理依赖。
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision

COPY pyproject.toml ./
COPY app ./app
# 运维回填脚本与应用代码一同进入镜像，便于在容器内对已有数据执行安全迁移。
COPY scripts ./scripts

# 基础镜像自带 pip 可能无法正确解析较新的 PEP 517 依赖元数据；先升级打包工具，
# 并优先使用预编译 wheel，避免线上构建 C 扩展和依赖解析失败。

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir --prefer-binary .

# tiktoken 的词表首次使用需要下载。构建期预热并固定缓存路径，保证运行容器
# 不依赖外网，也不会在第一次用户问答时因缺少词表失败。
RUN mkdir -p "$TIKTOKEN_CACHE_DIR" \
    && python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" \
    && chmod -R a+rX "$TIKTOKEN_CACHE_DIR"

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /var/lib/rag/uploads /var/lib/rag/models/huggingface /var/lib/rag/models/docling \
    && chown -R appuser:appuser /app /var/lib/rag

USER appuser
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips=*"]

# CI 或发布前验收使用相同的 Python 3.11 与解析依赖层，只额外安装 pytest 并复制
# 测试源码；默认 runtime 镜像不包含这些开发依赖，保持线上运行面最小。
FROM runtime AS test

USER root
COPY tests ./tests
RUN python -m pip install --no-cache-dir '.[dev]'
USER appuser

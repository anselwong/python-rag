# 构建独立生产镜像；不复制本地 .venv，也不依赖服务器全局 Python。
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
    PIP_DEFAULT_TIMEOUT=120 \
    RAG_DATA_DIR=/var/lib/rag

WORKDIR /app

COPY pyproject.toml ./
COPY app ./app

# 基础镜像自带 pip 可能无法正确解析较新的 PEP 517 依赖元数据；先升级打包工具，
# 并优先使用 PyPI 的预编译 wheel，避免线上构建 C 扩展和依赖解析失败。

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir --prefer-binary .

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /var/lib/rag/uploads \
    && chown -R appuser:appuser /app /var/lib/rag

USER appuser
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips=*"]

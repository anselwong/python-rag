"""Embedding 生成服务（Day 6）：默认调用阿里云百炼 DashScope。"""

import hashlib
import math
import os
from typing import List

from dashscope import TextEmbedding
import dashscope
from dotenv import load_dotenv

# Embedding 服务也可能被重建脚本、后台任务直接导入，不一定经过 FastAPI 的
# config 模块；在模块边界加载 .env，保证这两条调用路径拿到同一份配置。
load_dotenv(override=False)


class EmbeddingError(RuntimeError):
    """向量服务不可用或返回格式不符合数据库维度约束时抛出。"""


def embed_texts(texts: List[str]) -> List[List[float]]:
    """批量生成向量，保证文档和查询始终使用同一模型、同一维度。

    ``dashscope`` 是默认生产提供商；``hash`` 仅为自动化测试保留，不能用于真实
    语义检索。真实 API 按批调用，避免每个切片一次 HTTP 请求造成明显延迟和费用。
    """
    if not texts:
        return []
    provider = os.getenv("EMBEDDING_PROVIDER", "dashscope").lower()
    if provider == "hash":
        dimension = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
        return [_hash_embedding(text, dimension) for text in texts]
    if provider != "dashscope":
        raise EmbeddingError("不支持的 EMBEDDING_PROVIDER，仅支持 dashscope 或测试用 hash")
    return _dashscope_embeddings(texts)


def _dashscope_embeddings(texts: List[str]) -> List[List[float]]:
    """调用百炼 text-embedding 模型，并校验服务返回维度。"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("DASHSCOPE_BASE_URL")
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
    dimension = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
    if not api_key or not base_url:
        raise EmbeddingError("缺少 DASHSCOPE_API_KEY 或 DASHSCOPE_BASE_URL，请检查 .env")
    if dimension != 1536:
        # 当前 ORM 列定义为 Vector(1536)，提前失败比 PostgreSQL 插入时报错更可定位。
        raise EmbeddingError("当前 chunks.embedding 为 1536 维，请将 EMBEDDING_DIMENSION 设为 1536")

    dashscope.api_key = api_key
    dashscope.base_http_api_url = base_url
    vectors: List[List[float]] = []
    # 小批量可控制单次请求体积；切片很多时比逐条请求更快、更便宜，也便于失败定位。
    for start in range(0, len(texts), 10):
        response = TextEmbedding.call(model=model, input=texts[start:start + 10], dimension=dimension)
        if response.status_code != 200:
            raise EmbeddingError("百炼 Embedding 调用失败（HTTP %s）：%s" % (response.status_code, getattr(response, "message", "未知错误")))
        batch = [item["embedding"] for item in response.output["embeddings"]]
        if len(batch) != len(texts[start:start + 10]) or any(len(vector) != dimension for vector in batch):
            raise EmbeddingError("百炼返回的向量数量或维度与配置不一致")
        vectors.extend(batch)
    return vectors


def _hash_embedding(text: str, dimension: int) -> List[float]:
    """用多个 hash 桶构造稳定向量；仅作为无模型开发替身，不代表语义模型质量。"""
    values = [0.0] * dimension
    encoded = text.encode("utf-8")
    for index in range(0, max(1, len(encoded)), 4):
        digest = hashlib.blake2b(encoded[index:index + 64], digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimension
        sign = 1.0 if digest[4] % 2 else -1.0
        values[bucket] += sign
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


def cosine_similarity(left: List[float], right: List[float]) -> float:
    """计算归一化向量余弦相似度，结果范围约为 [-1, 1]。"""
    return sum(a * b for a, b in zip(left, right))

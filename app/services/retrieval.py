"""基于 pgvector 的切片检索服务。"""

import json
import math
from typing import Dict, List

from sqlalchemy import select

from app.core import database
from app.core.database import Chunk, Document, get_session
from app.services.embedding import cosine_similarity, embed_texts


def search(knowledge_base_id: str, query: str, top_k: int, score_threshold: float) -> List[Dict]:
    """召回知识库 Top K 切片，并过滤低于阈值的结果。"""
    query_vector = embed_texts([query])[0]
    with get_session() as session:
        statement = select(Chunk, Document).join(Document).where(
            Document.knowledge_base_id == knowledge_base_id,
            Chunk.embedding.is_not(None),
        )
        # PostgreSQL 由 pgvector 在数据库侧按余弦距离排序，避免把全量向量搬到 Python。
        # SQLite 没有向量运算扩展，测试环境才使用下面的 Python 计算兜底。
        if database.DATABASE_URL.startswith("postgresql"):
            statement = statement.order_by(Chunk.embedding.cosine_distance(query_vector)).limit(top_k)
        rows = list(session.execute(statement).all())
        scored = []
        for chunk, document in rows:
            # 测试替身存的是 JSON 文本；生产 PostgreSQL 分支由数据库完成向量距离排序。
            vector = chunk.embedding
            if isinstance(vector, str):
                vector = json.loads(vector)
            score = cosine_similarity(query_vector, list(vector)) if vector is not None else 0.0
            if score >= score_threshold:
                scored.append((score, chunk, document))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "id": chunk.id,
                "document_name": document.name,
                "page": chunk.page,
                "content": chunk.content,
                # pgvector/NumPy 可能返回 numpy.float32；它能参与比较，却不能被
                # json.dumps 序列化。API 与聊天记录边界统一转换为 Python float。
                "score": float(round(float(score), 6)),
                "rank": index,
                "keywords": _keywords(chunk.content, query),
            }
            for index, (score, chunk, document) in enumerate(scored[:top_k], start=1)
        ]


def _keywords(content: str, query: str) -> List[str]:
    """提取简单关键词用于调试展示；正式混合检索 Day9 再接 BM25。"""
    terms = [term.strip() for term in query.replace("？", " ").replace("?", " ").split() if term.strip()]
    return [term for term in terms if term in content][:5]

"""基于 pgvector 的切片检索服务。"""

import json
import math
import re
import math
from typing import Dict, List

from sqlalchemy import select

from app.core import database
from app.core.database import Chunk, Document, get_session
from app.services.embedding import cosine_similarity, embed_texts


def search(knowledge_base_id: str, query: str, top_k: int, score_threshold: float, mode: str = "vector") -> List[Dict]:
    """召回知识库 Top K 切片，并过滤低于阈值的结果。"""
    query_vector = embed_texts([query])[0]
    with get_session() as session:
        statement = select(Chunk, Document).join(Document).where(
            Document.knowledge_base_id == knowledge_base_id,
            Chunk.embedding.is_not(None),
        )
        # PostgreSQL 由 pgvector 在数据库侧按余弦距离排序，避免把全量向量搬到 Python。
        # SQLite 没有向量运算扩展，测试环境才使用下面的 Python 计算兜底。
        if database.DATABASE_URL.startswith("postgresql") and mode == "vector":
            statement = statement.order_by(Chunk.embedding.cosine_distance(query_vector)).limit(top_k)
        rows = list(session.execute(statement).all())
        corpus_tokens = [_tokenize(chunk.content) for chunk, _ in rows]
        document_frequency = {}
        average_length = sum(len(tokens) for tokens in corpus_tokens) / len(corpus_tokens) if corpus_tokens else 1.0
        for tokens in corpus_tokens:
            for term in set(tokens):
                document_frequency[term] = document_frequency.get(term, 0) + 1
        scored = []
        for row_index, (chunk, document) in enumerate(rows):
            # 测试替身存的是 JSON 文本；生产 PostgreSQL 分支由数据库完成向量距离排序。
            vector = chunk.embedding
            if isinstance(vector, str):
                vector = json.loads(vector)
            vector_score = cosine_similarity(query_vector, list(vector)) if vector is not None else 0.0
            lexical_score = _bm25_score(_tokenize(query), corpus_tokens[row_index], document_frequency, len(rows), average_length)
            # hybrid 把语义分数与关键词分数融合；实体名、编号等精确词更依赖后者。
            score = vector_score if mode == "vector" else vector_score * 0.7 + lexical_score * 0.3
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


def _lexical_score(content: str, query: str) -> float:
    """轻量关键词覆盖率，Day9 先提供可解释替代，后续可替换为 BM25。"""
    terms = [term for term in re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_-]+", query) if term]
    if not terms:
        return 0.0
    return sum(1 for term in terms if term in content) / len(terms)


def _tokenize(text: str) -> List[str]:
    """中文按单字、英文按词切分，保证“角色/MES”这类词都可参与 BM25。"""
    return re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_-]+", text.lower())


def _bm25_score(query_terms: List[str], document_terms: List[str], document_frequency: Dict[str, int], total_documents: int, average_length: float) -> float:
    """BM25 简化实现；只依赖内存统计，适合当前规模，生产可下推 PostgreSQL FTS。"""
    if not query_terms or not document_terms:
        return 0.0
    k1, b = 1.5, 0.75
    frequencies = {term: document_terms.count(term) for term in set(query_terms)}
    score = 0.0
    for term, frequency in frequencies.items():
        if not frequency:
            continue
        df = document_frequency.get(term, 0)
        idf = math.log(1 + (total_documents - df + 0.5) / (df + 0.5))
        score += idf * (frequency * (k1 + 1)) / (frequency + k1 * (1 - b + b * len(document_terms) / max(1.0, average_length)))
    # 融合时只需要相对排序，压缩到 0~1 避免 BM25 数值压过向量分数。
    return score / (score + 1.0)

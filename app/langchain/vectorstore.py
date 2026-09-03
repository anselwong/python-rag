"""LangChain VectorStore 工厂：生产 PGVector，测试使用 InMemoryVectorStore。"""
import os
from typing import List, Optional
from sqlalchemy import delete
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_postgres import PGVector
from app.core import database
from .embeddings import BailianEmbeddings

_memory = {}

def get_store(kb_id: str, embeddings: BailianEmbeddings):
    if database.DATABASE_URL.startswith("postgresql"):
        return PGVector(embeddings=embeddings, connection=database.DATABASE_URL, collection_name=f"langchain_kb_{kb_id.replace('-', '_')}", embedding_length=int(os.getenv("EMBEDDING_DIMENSION", "1536")), create_extension=True)
    return _memory.setdefault(kb_id, InMemoryVectorStore(embedding=embeddings))

def add_documents(kb_id: str, documents: list[Document], embeddings: BailianEmbeddings) -> list[str]:
    # 显式传入业务 chunk_id，删除文档时可以精准调用 VectorStore.delete，避免依赖正文匹配。
    ids = [document.metadata.get("chunk_id") for document in documents]
    return get_store(kb_id, embeddings).add_documents(documents, ids=ids)

def similarity_search(kb_id: str, query: str, k: int, embeddings: BailianEmbeddings):
    return get_store(kb_id, embeddings).similarity_search_with_score(query, k=k)

def delete_document_vectors(kb_id: str, document_id: str, embeddings: BailianEmbeddings, vector_ids: Optional[List[str]] = None) -> None:
    """删除指定文档的 LangChain 向量，兼容新旧两种写入 ID。"""
    store = get_store(kb_id, embeddings)
    chunk_ids = vector_ids or []
    if isinstance(store, InMemoryVectorStore):
        # InMemoryVectorStore 的 delete 对不存在的 ID 是幂等的；测试和开发环境无需访问数据库。
        if chunk_ids:
            store.delete(chunk_ids)
        return
    # 旧版本未显式传 ID，PGVector 会生成 UUID；用 metadata 兜底清理这些历史记录。
    with store._make_sync_session() as session:
        collection = store.get_collection(session)
        if collection is None:
            return
        stmt = delete(store.EmbeddingStore).where(store.EmbeddingStore.collection_id == collection.uuid)
        stmt = stmt.where(
            (store.EmbeddingStore.id.in_(chunk_ids) if chunk_ids else False)
            | (store.EmbeddingStore.cmetadata["document_id"].astext == document_id)
        )
        session.execute(stmt)
        session.commit()

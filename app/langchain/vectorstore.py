"""LangChain VectorStore 工厂：生产 PGVector，测试使用 InMemoryVectorStore。"""
import os
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
    return get_store(kb_id, embeddings).add_documents(documents)

def similarity_search(kb_id: str, query: str, k: int, embeddings: BailianEmbeddings):
    return get_store(kb_id, embeddings).similarity_search_with_score(query, k=k)

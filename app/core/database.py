"""PostgreSQL 数据库连接、ORM 模型与表初始化。

生产环境使用 PostgreSQL；安装 pgvector 扩展后，Chunk.embedding 可以直接保存向量。
测试可通过 DATABASE_URL 指向 SQLite，但这只是测试替身，不是生产部署方案。
"""

import os
from contextlib import contextmanager
from datetime import datetime
from typing import Iterator

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

try:
    from pgvector.sqlalchemy import Vector
except ImportError:  # 仅允许无 pgvector 的测试环境导入
    Vector = None

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://rag:rag@localhost:5432/rag")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def configure_database(url: str) -> None:
    """切换当前进程的数据库连接，测试用 SQLite，生产始终使用 PostgreSQL URL。"""
    global DATABASE_URL, engine, SessionLocal
    DATABASE_URL = url
    engine = create_engine(url, pool_pre_ping=True)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(80), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    color: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    documents: Mapped[list["Document"]] = relationship(back_populates="knowledge_base", cascade="all, delete-orphan")


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    stored_name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    file_type: Mapped[str] = mapped_column(String(16), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    pages_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    knowledge_base: Mapped[KnowledgeBase] = relationship(back_populates="documents")


class Chunk(Base):
    """切片表预留 pgvector 向量列，Day 6 只需填充 embedding 即可检索。"""
    __tablename__ = "chunks"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    page: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding = mapped_column(Vector(1536) if Vector else Text, nullable=True)


Index("idx_documents_kb", Document.knowledge_base_id)
Index("idx_chunks_document", Chunk.document_id)


def initialize_database() -> None:
    """创建表和 pgvector 扩展；正式项目应再用 Alembic 管理版本迁移。"""
    with engine.begin() as connection:
        if DATABASE_URL.startswith("postgresql"):
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)


@contextmanager
def get_session() -> Iterator:
    """提供事务边界；正常退出提交，异常自动回滚，避免半写入状态。"""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

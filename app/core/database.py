"""PostgreSQL 数据库连接、ORM 模型与表初始化。

生产环境使用 PostgreSQL；安装 pgvector 扩展后，Chunk.embedding 可以直接保存向量。
测试可通过 DATABASE_URL 指向 SQLite，但这只是测试替身，不是生产部署方案。
"""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Iterator

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

try:
    from pgvector.sqlalchemy import Vector
except ImportError:  # 仅允许无 pgvector 的测试环境导入
    Vector = None

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://rag:rag@localhost:5432/rag")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


@event.listens_for(Engine, "connect")
def _enable_sqlite_foreign_keys(dbapi_connection, connection_record):
    """SQLite 默认关闭外键约束，必须逐连接执行 PRAGMA 才能让 ON DELETE CASCADE 生效。

    PostgreSQL 默认启用外键，无需处理。这个 listener 让 SQLite 测试环境与
    生产行为一致——否则删除文档会留下孤儿 chunk（集成测试已抓到该问题）。
    """
    if isinstance(dbapi_connection, sqlite3.Connection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


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
    # relationship 让 SQLAlchemy 的 flush 按依赖排序（先 documents 后 chunks）。
    # 仅靠表级 ForeignKey 不参与 mapper 间排序，外键开启后会插入乱序报错；
    # cascade 同时提供 ORM 级联删除，数据库外键作为最后兜底。
    chunks: Mapped[list["Chunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    """切片表预留 pgvector 向量列，Day 6 只需填充 embedding 即可检索。"""
    __tablename__ = "chunks"
    # id 格式为 "{document_id}-{序号}"（36+1+N 字符），比纯 UUID 更可读、可追溯，
    # 同一文档重新切片时天然幂等。列宽取 64 以容纳后缀序号——
    # SQLite 不校验 VARCHAR 长度，PostgreSQL 会严格拒绝超长值，测试替身抓不到这类问题。
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    page: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding = mapped_column(Vector(1536) if Vector else Text, nullable=True)
    document: Mapped["Document"] = relationship(back_populates="chunks")


class ChatSession(Base):
    """一条可持续的对话线程，归属于知识库。"""
    __tablename__ = "chat_sessions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False)
    title: Mapped[str] = mapped_column(String(120), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    messages: Mapped[list["ChatMessage"]] = relationship(back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    """聊天消息；引用以 JSON 保存 ID、页码和分数，避免重复存储正文。"""
    __tablename__ = "chat_messages"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    citations_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    session: Mapped[ChatSession] = relationship(back_populates="messages")


class EvaluationCase(Base):
    """属于某个知识库的检索评测题；expected_* 是人工确认的金标准。"""
    __tablename__ = "evaluation_cases"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    knowledge_base_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    expected_document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    expected_page: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


Index("idx_documents_kb", Document.knowledge_base_id)
Index("idx_chunks_document", Chunk.document_id)
Index("idx_chat_sessions_kb", ChatSession.knowledge_base_id)
Index("idx_chat_messages_session", ChatMessage.session_id)
Index("idx_evaluation_cases_kb", EvaluationCase.knowledge_base_id)


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

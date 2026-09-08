"""知识库、文件、文档解析与切片业务服务。

本模块处理元数据、原文解析和切片落库，不在 Day 5 提前调用 Embedding。
Day 6 会复用 Chunk ORM 模型把切片向量写入 pgvector 的 embedding 列。
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import UploadFile

from app.core.config import settings
from app.core.database import Chunk, Document, KnowledgeBase, LangChainChunk, get_session
from app.services.chunker import split_pages_into_chunks
from app.services.embedding import embed_texts
from app.services.parser import parse_document

UPLOAD_DIR = settings.data_dir / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
COLORS = ("#2f8f78", "#3468a5", "#a5652e", "#6b7280")


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def list_knowledge_bases(owner_user_id: Optional[str] = None) -> List[Dict]:
    with get_session() as session:
        query = session.query(KnowledgeBase).order_by(KnowledgeBase.updated_at.desc())
        if owner_user_id is not None:
            query = query.filter(KnowledgeBase.owner_user_id == owner_user_id)
        items = query.all()
        return [_knowledge_row(item) for item in items]


def create_knowledge_base(name: str, description: str, owner_user_id: Optional[str] = None) -> Dict:
    timestamp = now_utc()
    with get_session() as session:
        count = session.query(KnowledgeBase).count()
        item = KnowledgeBase(id=str(uuid.uuid4()), owner_user_id=owner_user_id, name=name, description=description, color=COLORS[count % len(COLORS)], created_at=timestamp, updated_at=timestamp)
        session.add(item)
        session.flush()
        return _knowledge_row(item)


def list_documents(knowledge_base_id: str) -> List[Dict]:
    with get_session() as session:
        items = session.query(Document).filter(Document.knowledge_base_id == knowledge_base_id).order_by(Document.created_at.desc()).all()
        return [_document_row(item) for item in items]


def ensure_knowledge_base_owner(knowledge_base_id: str, owner_user_id: str) -> None:
    """在进入文档、聊天、检索和评测链路前校验知识库私有归属。"""
    with get_session() as session:
        exists = session.query(KnowledgeBase.id).filter(
            KnowledgeBase.id == knowledge_base_id,
            KnowledgeBase.owner_user_id == owner_user_id,
        ).one_or_none()
    if exists is None:
        raise KeyError("知识库不存在")


def get_document(document_id: str) -> Dict:
    with get_session() as session:
        item = session.get(Document, document_id)
        if item is None:
            raise KeyError("文档不存在")
        result = _document_row(item)
        result["pages"] = [{"document_id": document_id, "page": page["page"], "text": page["text"]} for page in json.loads(item.pages_json)]
        # 详情接口同时返回切片，方便前端验证“解析文本 -> 切片”的中间产物。
        # embedding 属于内部向量数据，不能为了预览把它序列化到 HTTP 响应。
        # ID 后缀是数字序号，必须按整数排序；字符串排序会把 10 排在 2 前面。
        ordered_chunks = sorted(item.chunks, key=lambda value: int(value.id.rsplit("-", 1)[-1]))
        result["chunks"] = [
            {"id": chunk.id, "document_id": document_id, "page": chunk.page, "content": chunk.content, "token_count": chunk.token_count}
            for chunk in ordered_chunks
        ]
        return result


async def ingest_document(knowledge_base_id: str, upload: UploadFile) -> Dict:
    """校验、保存、解析并原子写入文档元数据；失败会删除已保存文件。"""
    # LangChain 分支由 app.langchain.service 负责 Loader/Splitter/Embedding 全链路。
    from app.langchain.service import ingest
    return await ingest(knowledge_base_id, upload)


def delete_document(knowledge_base_id: str, document_id: str) -> None:
    """先清理 LangChain 向量，再删除元数据和文件，避免产生孤儿向量。"""
    with get_session() as session:
        item = session.query(Document).filter(Document.id == document_id, Document.knowledge_base_id == knowledge_base_id).one_or_none()
        if item is None:
            raise KeyError("文档不存在")
        stored_path = UPLOAD_DIR / item.stored_name
        vector_ids = [row.id for row in session.query(LangChainChunk).filter(LangChainChunk.document_id == document_id).all()]
    from app.langchain.vectorstore import delete_document_vectors
    from app.langchain.service import embeddings
    # 先删向量；向量清理失败时保留业务数据，避免出现“文档已删但索引残留”。
    delete_document_vectors(knowledge_base_id, document_id, embeddings, vector_ids)
    with get_session() as session:
        item = session.query(Document).filter(Document.id == document_id, Document.knowledge_base_id == knowledge_base_id).one_or_none()
        if item is None:
            raise KeyError("文档不存在")
        session.delete(item)
    stored_path.unlink(missing_ok=True)


def reindex_embeddings(knowledge_base_id: str) -> int:
    """用当前模型覆盖知识库全部切片向量，返回重建数量。

    更换模型、版本或维度后，旧向量处于不同语义空间，不能与新查询向量混检；
    因此必须对所有切片重建。先在事务外调用远程模型，避免网络等待长期占用数据库事务。
    """
    from app.langchain.service import embeddings, similarity_search
    # LangChain VectorStore 的重建由上传/管理任务负责；这里保留旧 API 名称，改为批量刷新共享兼容表。
    with get_session() as session:
        knowledge_base = session.get(KnowledgeBase, knowledge_base_id)
        if knowledge_base is None:
            raise KeyError("知识库不存在")
        chunks = session.query(Chunk).join(Document).filter(Document.knowledge_base_id == knowledge_base_id).order_by(Chunk.id).all()
        payload = [(chunk.id, chunk.content) for chunk in chunks]
    vectors = embed_texts([content for _, content in payload])
    with get_session() as session:
        for (chunk_id, _), vector in zip(payload, vectors):
            chunk = session.get(Chunk, chunk_id)
            if chunk is not None:
                chunk.embedding = vector
    return len(payload)


def _knowledge_row(item: KnowledgeBase) -> Dict:
    return {"id": item.id, "name": item.name, "description": item.description, "document_count": len(item.documents), "chunk_count": sum(document.chunk_count for document in item.documents), "updated_at": item.updated_at, "color": item.color}


def _document_row(item: Document) -> Dict:
    return {"id": item.id, "knowledge_base_id": item.knowledge_base_id, "name": item.name, "type": item.file_type, "size": _format_size(item.size_bytes), "chunk_count": item.chunk_count, "status": item.status, "created_at": item.created_at}


def _format_size(size: int) -> str:
    return f"{max(1, round(size / 1024))} KB" if size < 1024 * 1024 else f"{size / (1024 * 1024):.1f} MB"

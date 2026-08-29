"""知识库、文件和文档解析的业务服务。"""

import json
import mimetypes
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from fastapi import UploadFile

from app.core.config import settings
from app.core.database import get_connection
from app.services.parser import parse_document

UPLOAD_DIR = settings.data_dir / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
COLORS = ("#2f8f78", "#3468a5", "#a5652e", "#6b7280")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def list_knowledge_bases() -> List[Dict]:
    with get_connection() as connection:
        rows = connection.execute(
            """SELECT kb.*, COUNT(d.id) AS document_count,
            COALESCE(SUM(d.chunk_count), 0) AS chunk_count
            FROM knowledge_bases kb LEFT JOIN documents d ON d.knowledge_base_id = kb.id
            GROUP BY kb.id ORDER BY kb.updated_at DESC"""
        ).fetchall()
    return [_knowledge_row(row) for row in rows]


def create_knowledge_base(name: str, description: str) -> Dict:
    knowledge_base_id = str(uuid.uuid4())
    timestamp = now_iso()
    with get_connection() as connection:
        count = connection.execute("SELECT COUNT(*) FROM knowledge_bases").fetchone()[0]
        connection.execute(
            "INSERT INTO knowledge_bases (id, name, description, color, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (knowledge_base_id, name, description, COLORS[count % len(COLORS)], timestamp, timestamp),
        )
    return next(item for item in list_knowledge_bases() if item["id"] == knowledge_base_id)


def list_documents(knowledge_base_id: str) -> List[Dict]:
    with get_connection() as connection:
        rows = connection.execute("SELECT * FROM documents WHERE knowledge_base_id = ? ORDER BY created_at DESC", (knowledge_base_id,)).fetchall()
    return [_document_row(row) for row in rows]


def get_document(document_id: str) -> Dict:
    with get_connection() as connection:
        row = connection.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
    if row is None:
        raise KeyError("文档不存在")
    result = _document_row(row)
    result["pages"] = [
        {"document_id": document_id, "page": page["page"], "text": page["text"]}
        for page in json.loads(row["pages_json"])
    ]
    return result


async def ingest_document(knowledge_base_id: str, upload: UploadFile) -> Dict:
    """校验、保存并解析上传文件；失败时删除临时文件，避免脏数据残留。"""
    original_name = Path(upload.filename or "未命名").name
    extension = Path(original_name).suffix.lower()
    if extension not in settings.allowed_extensions:
        raise ValueError("仅支持 PDF、DOCX、MD、TXT 文件")

    knowledge_base_exists = _knowledge_base_exists(knowledge_base_id)
    if not knowledge_base_exists:
        raise KeyError("知识库不存在")

    document_id = str(uuid.uuid4())
    stored_name = f"{document_id}{extension}"
    destination = UPLOAD_DIR / stored_name
    size = 0
    try:
        with destination.open("wb") as target:
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                if size > settings.max_upload_size:
                    raise ValueError("文件大小不能超过 20 MB")
                target.write(chunk)
        pages = [{"page": page, "text": text} for page, text in parse_document(destination, extension)]
        timestamp = now_iso()
        with get_connection() as connection:
            connection.execute(
                """INSERT INTO documents
                (id, knowledge_base_id, name, stored_name, file_type, size_bytes, chunk_count, status, pages_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (document_id, knowledge_base_id, original_name, stored_name, extension[1:].upper(), size, 0, "ready", json.dumps(pages, ensure_ascii=False), timestamp),
            )
            connection.execute("UPDATE knowledge_bases SET updated_at = ? WHERE id = ?", (timestamp, knowledge_base_id))
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return get_document(document_id)


def delete_document(knowledge_base_id: str, document_id: str) -> None:
    with get_connection() as connection:
        row = connection.execute("SELECT stored_name FROM documents WHERE id = ? AND knowledge_base_id = ?", (document_id, knowledge_base_id)).fetchone()
        if row is None:
            raise KeyError("文档不存在")
        connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
    (UPLOAD_DIR / row["stored_name"]).unlink(missing_ok=True)


def _knowledge_base_exists(knowledge_base_id: str) -> bool:
    with get_connection() as connection:
        return connection.execute("SELECT 1 FROM knowledge_bases WHERE id = ?", (knowledge_base_id,)).fetchone() is not None


def _knowledge_row(row) -> Dict:
    return {"id": row["id"], "name": row["name"], "description": row["description"], "document_count": row["document_count"], "chunk_count": row["chunk_count"], "updated_at": row["updated_at"], "color": row["color"]}


def _document_row(row) -> Dict:
    return {"id": row["id"], "knowledge_base_id": row["knowledge_base_id"], "name": row["name"], "type": row["file_type"], "size": _format_size(row["size_bytes"]), "chunk_count": row["chunk_count"], "status": row["status"], "created_at": row["created_at"]}


def _format_size(size: int) -> str:
    if size < 1024 * 1024:
        return f"{max(1, round(size / 1024))} KB"
    return f"{size / (1024 * 1024):.1f} MB"


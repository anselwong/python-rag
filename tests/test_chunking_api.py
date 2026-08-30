"""Day 5 集成测试：上传 → 切片落库 → 级联删除的完整链路。"""

from pathlib import Path

from fastapi.testclient import TestClient

from app.core import database
from app.core.database import Chunk
from app.services import knowledge

from test_health import make_client


def test_upload_creates_chunks_with_metadata(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    knowledge_base_id = client.post("/api/v1/knowledge-bases", json={"name": "切片测试库"}).json()["id"]

    # 20 个段落、每段约 300 token：必然产生多个 chunk。
    content = "\n\n".join(f"第{index}段：{ '切片集成测试内容，用于验证落库。' * 20 }" for index in range(20))
    upload = client.post(
        f"/api/v1/knowledge-bases/{knowledge_base_id}/documents",
        files={"file": ("big.txt", content, "text/plain")},
    )
    assert upload.status_code == 201
    document = upload.json()
    assert document["chunk_count"] > 1

    # chunks 表的行数、归属和元数据必须与响应一致。
    with database.get_session() as session:
        rows = session.query(Chunk).filter(Chunk.document_id == document["id"]).all()
    assert len(rows) == document["chunk_count"]
    assert all(row.document_id == document["id"] for row in rows)
    assert all(row.token_count > 0 for row in rows)
    assert all(row.embedding is None for row in rows)  # Day 6 才写向量

    detail = client.get(f"/api/v1/knowledge-bases/{knowledge_base_id}/documents/{document['id']}")
    assert detail.status_code == 200
    assert len(detail.json()["chunks"]) == document["chunk_count"]
    assert {"id", "page", "content", "token_count"}.issubset(detail.json()["chunks"][0])


def test_delete_document_cascades_chunks(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    knowledge_base_id = client.post("/api/v1/knowledge-bases", json={"name": "级联测试库"}).json()["id"]
    content = "\n\n".join(f"段落{index}。" * 50 for index in range(10))
    document = client.post(
        f"/api/v1/knowledge-bases/{knowledge_base_id}/documents",
        files={"file": ("doc.txt", content, "text/plain")},
    ).json()

    assert client.delete(f"/api/v1/knowledge-bases/{knowledge_base_id}/documents/{document['id']}").status_code == 204
    # 外键 ON DELETE CASCADE 必须清掉切片，避免孤儿 chunk 占用存储并污染检索。
    with database.get_session() as session:
        assert session.query(Chunk).filter(Chunk.document_id == document["id"]).count() == 0


def test_small_document_single_chunk(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    knowledge_base_id = client.post("/api/v1/knowledge-bases", json={"name": "小文档库"}).json()["id"]
    upload = client.post(
        f"/api/v1/knowledge-bases/{knowledge_base_id}/documents",
        files={"file": ("tiny.txt", "第一段\n\n第二段", "text/plain")},
    )
    assert upload.status_code == 201
    assert upload.json()["chunk_count"] == 1

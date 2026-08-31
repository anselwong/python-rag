"""Day 6 检索接口测试：使用 SQLite 验证与 pgvector 等价的排序逻辑。"""

from pathlib import Path
import json

from app.core import database
from app.core.database import Chunk
from app.services import knowledge
from test_health import make_client


def test_vector_search_returns_ranked_chunks(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    kb_id = client.post("/api/v1/knowledge-bases", json={"name": "检索测试库"}).json()["id"]
    content = "用户登录需要验证码。\n\n退款申请需要订单号。"
    document = client.post(f"/api/v1/knowledge-bases/{kb_id}/documents", files={"file": ("guide.txt", content, "text/plain")}).json()

    response = client.post(f"/api/v1/knowledge-bases/{kb_id}/retrieval/search", json={"query": "用户登录验证码", "top_k": 3, "score_threshold": -1})
    assert response.status_code == 200
    results = response.json()
    assert results
    assert results[0]["rank"] == 1
    assert results[0]["document_name"] == "guide.txt"
    assert all(results[index]["score"] >= results[index + 1]["score"] for index in range(len(results) - 1))
    # 引用结果会写入聊天记录 JSON，分数必须是标准库可序列化的原生 float。
    json.dumps(results)


def test_search_validates_empty_query(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    kb_id = client.post("/api/v1/knowledge-bases", json={"name": "校验库"}).json()["id"]
    response = client.post(f"/api/v1/knowledge-bases/{kb_id}/retrieval/search", json={"query": ""})
    assert response.status_code == 422


def test_reindex_overwrites_all_chunk_embeddings(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    kb_id = client.post("/api/v1/knowledge-bases", json={"name": "重建库"}).json()["id"]
    document = client.post(f"/api/v1/knowledge-bases/{kb_id}/documents", files={"file": ("guide.txt", "重建向量测试", "text/plain")}).json()
    assert knowledge.reindex_embeddings(kb_id) == document["chunk_count"]
    with database.get_session() as session:
        assert session.query(Chunk).filter(Chunk.document_id == document["id"], Chunk.embedding.is_not(None)).count() == document["chunk_count"]

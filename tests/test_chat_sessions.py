"""会话标题与删除接口的集成测试。"""

from pathlib import Path

from app.core import database
from datetime import datetime, timezone

from app.core.database import ChatMessage, ChatSession
from test_health import make_client


def _create_session(client, knowledge_base_id: str) -> str:
    """直接写入测试会话，避免测试依赖付费 LLM 网络请求。"""
    import uuid

    session_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc)
    with database.get_session() as session:
        session.add(ChatSession(id=session_id, knowledge_base_id=knowledge_base_id, title="测试会话", created_at=timestamp, updated_at=timestamp))
        session.add(ChatMessage(id=str(uuid.uuid4()), session_id=session_id, role="user", content="问题", created_at=timestamp))
    return session_id


def test_rename_chat_session(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    kb_id = client.post("/api/v1/knowledge-bases", json={"name": "会话库"}).json()["id"]
    session_id = _create_session(client, kb_id)

    response = client.patch(
        f"/api/v1/knowledge-bases/{kb_id}/chat-sessions/{session_id}",
        json={"title": "新的会话标题"},
    )
    assert response.status_code == 200
    assert response.json()["title"] == "新的会话标题"
    assert client.get(f"/api/v1/knowledge-bases/{kb_id}/chat-sessions").json()[0]["title"] == "新的会话标题"


def test_chat_session_is_scoped_and_cascades_messages(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    kb_id = client.post("/api/v1/knowledge-bases", json={"name": "主库"}).json()["id"]
    other_kb_id = client.post("/api/v1/knowledge-bases", json={"name": "其他库"}).json()["id"]
    session_id = _create_session(client, kb_id)

    forbidden = client.patch(
        f"/api/v1/knowledge-bases/{other_kb_id}/chat-sessions/{session_id}",
        json={"title": "越权标题"},
    )
    assert forbidden.status_code == 404

    assert client.delete(f"/api/v1/knowledge-bases/{kb_id}/chat-sessions/{session_id}").status_code == 204
    with database.get_session() as session:
        assert session.query(ChatMessage).filter(ChatMessage.session_id == session_id).count() == 0
    assert client.delete(f"/api/v1/knowledge-bases/{kb_id}/chat-sessions/{session_id}").status_code == 404

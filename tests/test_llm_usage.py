"""模型真实 Token usage 的归一化、持久化和接口返回测试。"""

import json
import uuid
from datetime import datetime, timezone

from langchain_core.messages import AIMessage

from app.core import database
from app.core.database import ChatMessage, ChatSession
from app.langchain.history import SqlChatMessageHistory
from app.langchain.usage import get_message_usage, normalize_usage
from test_health import make_client


def test_normalize_provider_usage_without_local_estimate() -> None:
    """仅转换供应商真实字段；没有 usage 时必须保持 None，不能伪造。"""
    assert normalize_usage(None) is None
    assert normalize_usage({"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}) == {
        "prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20,
    }
    message = AIMessage(content="回答", usage_metadata={"input_tokens": 12, "output_tokens": 8, "total_tokens": 20})
    assert get_message_usage(message) == {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}


def test_history_persists_usage_and_session_api_returns_it(tmp_path) -> None:
    """usage 应绑定本轮助手消息，并随会话恢复接口返回给前端。"""
    client = make_client(tmp_path)
    kb_id = client.post("/api/v1/knowledge-bases", json={"name": "用量库"}).json()["id"]
    session_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc)
    with database.get_session() as session:
        session.add(ChatSession(id=session_id, knowledge_base_id=kb_id, title="Token 用量", created_at=timestamp, updated_at=timestamp))

    history = SqlChatMessageHistory(session_id)
    usage = {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150}
    history.add_messages([AIMessage(content="模型回答", usage_metadata={"input_tokens": 120, "output_tokens": 30, "total_tokens": 150})])
    history.attach_response_metadata([], usage)

    with database.get_session() as session:
        message = session.get(ChatMessage, history.last_assistant_message_id)
        assert json.loads(message.usage_json) == usage

    response = client.get(f"/api/v1/knowledge-bases/{kb_id}/chat-sessions")
    assert response.status_code == 200
    assert response.json()[0]["messages"][0]["usage"] == usage

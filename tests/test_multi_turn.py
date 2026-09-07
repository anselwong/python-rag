"""RunnableWithMessageHistory 的 PostgreSQL 适配与多轮问答测试。"""

import uuid
from datetime import datetime, timedelta, timezone

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from app.core import database
from app.core.database import ChatMessage, ChatSession
from app.langchain import chain, service
from app.langchain.history import SqlChatMessageHistory
from test_health import make_client


def _seed_messages(knowledge_base_id: str, contents) -> str:
    """写入已存在的多轮消息，构造不依赖真实模型的历史读取场景。"""
    session_id = str(uuid.uuid4())
    # 预置的是“历史”消息，时间必须早于本次 Runnable 新增的消息。
    base = datetime.now(timezone.utc) - timedelta(seconds=len(contents) + 1)
    with database.get_session() as session:
        session.add(ChatSession(id=session_id, knowledge_base_id=knowledge_base_id, title="多轮会话", created_at=base, updated_at=base))
        for index, (role, content) in enumerate(contents):
            session.add(ChatMessage(id=str(uuid.uuid4()), session_id=session_id, role=role, content=content, created_at=base + timedelta(seconds=index)))
    return session_id


def test_sql_history_converts_messages_and_applies_window(tmp_path) -> None:
    """数据库行应还原为结构化消息，且只向模型暴露最近六条。"""
    client = make_client(tmp_path)
    kb_id = client.post("/api/v1/knowledge-bases", json={"name": "多轮库"}).json()["id"]
    contents = []
    for index in range(5):
        contents.extend([("user", f"问题{index}"), ("assistant", f"回答{index}")])
    history = SqlChatMessageHistory(_seed_messages(kb_id, contents))

    messages = history.messages
    assert len(messages) == 6
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == "问题2"
    assert isinstance(messages[-1], AIMessage)
    assert messages[-1].content == "回答4"


def test_runnable_history_persists_question_answer_and_citations(tmp_path, monkeypatch) -> None:
    """LCEL 成功完成后应由 Runnable 自动写入用户、助手消息，不经由路由手写 SQL。"""
    client = make_client(tmp_path)
    kb_id = client.post("/api/v1/knowledge-bases", json={"name": "多轮库"}).json()["id"]
    monkeypatch.setattr(chain, "_build_model", lambda: FakeListChatModel(responses=["模拟回答"]))
    monkeypatch.setattr(service, "retrieve_for_answer", lambda *args, **kwargs: [{
        "id": "chunk-1", "document_name": "知识.md", "page": 1, "content": "证据", "score": 0.9,
    }])

    answer, citations, session_id, message_id = service.answer(kb_id, "请根据资料回答")
    assert answer == "模拟回答"
    assert citations[0]["id"] == "chunk-1"

    with database.get_session() as session:
        rows = session.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at).all()
        assert [row.role for row in rows] == ["user", "assistant"]
        assert rows[0].content == "请根据资料回答"
        assert rows[1].id == message_id
        assert rows[1].content == "模拟回答"
        assert "chunk-1" in rows[1].citations_json


def test_runnable_history_reads_previous_turn_into_prompt(tmp_path, monkeypatch) -> None:
    """同一 session 的第二次调用会经 MessagesPlaceholder 注入结构化历史。"""
    client = make_client(tmp_path)
    kb_id = client.post("/api/v1/knowledge-bases", json={"name": "多轮库"}).json()["id"]
    session_id = _seed_messages(kb_id, [("user", "上一轮问题"), ("assistant", "上一轮回答")])
    monkeypatch.setattr(chain, "_build_model", lambda: FakeListChatModel(responses=["下一轮回答"]))
    monkeypatch.setattr(service, "retrieve_for_answer", lambda *args, **kwargs: [])

    service.answer(kb_id, "继续说明", session_id=session_id)

    with database.get_session() as session:
        rows = session.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at).all()
        assert [row.content for row in rows] == ["上一轮问题", "上一轮回答", "继续说明", "下一轮回答"]


def test_streaming_history_persists_only_after_stream_finishes(tmp_path, monkeypatch) -> None:
    """SSE 消费完成后，Runnable 才把完整回答和引用写入数据库。"""
    client = make_client(tmp_path)
    kb_id = client.post("/api/v1/knowledge-bases", json={"name": "流式多轮库"}).json()["id"]
    monkeypatch.setattr(chain, "_build_model", lambda: FakeListChatModel(responses=["流式回答"]))
    monkeypatch.setattr(service, "retrieve_for_answer", lambda *args, **kwargs: [{
        "id": "chunk-stream", "document_name": "流式.md", "page": 2, "content": "流式证据", "score": 0.8,
    }])

    stream, _, session_id = service.answer_stream(kb_id, "流式问题")
    assert "".join(stream) == "流式回答"

    with database.get_session() as session:
        rows = session.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at).all()
        assert [row.role for row in rows] == ["user", "assistant"]
        assert rows[1].content == "流式回答"
        assert "chunk-stream" in rows[1].citations_json

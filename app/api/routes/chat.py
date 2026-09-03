"""知识库问答接口：先检索，再把证据交给大模型。"""

import json
import uuid
from datetime import datetime, timezone
from typing import Dict

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.schemas.chat import ChatRequest, ChatResponse, ChatSessionRenameRequest
from app.schemas.chat_stream import ChatStreamRequest
from app.schemas.retrieval import RetrievalRequest
from app.langchain.service import answer as lc_answer, answer_stream as lc_answer_stream
from app.core.database import ChatMessage, ChatSession, KnowledgeBase, get_session

router = APIRouter(prefix="/knowledge-bases/{knowledge_base_id}")


@router.get("/chat-sessions", summary="List persisted chat sessions")
def get_chat_sessions(knowledge_base_id: str) -> list[dict]:
    with get_session() as session:
        rows = session.query(ChatSession).filter(ChatSession.knowledge_base_id == knowledge_base_id).order_by(ChatSession.updated_at.desc()).all()
        return [{"id": row.id, "title": row.title, "updated_at": row.updated_at.isoformat(), "messages": [{"id": message.id, "role": message.role, "content": message.content, "created_at": message.created_at.isoformat(), "citations": json.loads(message.citations_json)} for message in sorted(row.messages, key=lambda value: value.created_at)]} for row in rows]


@router.patch("/chat-sessions/{session_id}", summary="Rename a chat session")
def rename_chat_session(knowledge_base_id: str, session_id: str, payload: ChatSessionRenameRequest) -> dict:
    """修改会话标题，并校验会话属于当前知识库，防止越权修改其他知识库数据。"""
    title = payload.title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="会话名称不能为空")
    with get_session() as session:
        item = session.query(ChatSession).filter(
            ChatSession.id == session_id,
            ChatSession.knowledge_base_id == knowledge_base_id,
        ).one_or_none()
        if item is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        item.title = title
        item.updated_at = datetime.now(timezone.utc)
        return {"id": item.id, "title": item.title, "updated_at": item.updated_at.isoformat()}


@router.delete("/chat-sessions/{session_id}", status_code=204, summary="Delete a chat session")
def delete_chat_session(knowledge_base_id: str, session_id: str) -> None:
    """删除会话；ORM cascade 与数据库外键会一并删除其消息记录。"""
    with get_session() as session:
        item = session.query(ChatSession).filter(
            ChatSession.id == session_id,
            ChatSession.knowledge_base_id == knowledge_base_id,
        ).one_or_none()
        if item is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        session.delete(item)


@router.post("/chat", response_model=ChatResponse, summary="Answer with retrieved context")
def post_chat(knowledge_base_id: str, payload: ChatRequest) -> Dict:
    try:
        answer, contexts = lc_answer(knowledge_base_id, payload.message.strip(), session_id=payload.session_id)
    except Exception as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    timestamp = datetime.now(timezone.utc)
    with get_session() as session:
        if payload.session_id:
            chat_session = session.query(ChatSession).filter(ChatSession.id == payload.session_id, ChatSession.knowledge_base_id == knowledge_base_id).one_or_none()
            if chat_session is None:
                raise HTTPException(status_code=404, detail="会话不存在")
        else:
            chat_session = ChatSession(id=str(uuid.uuid4()), knowledge_base_id=knowledge_base_id, title=payload.message.strip()[:40], created_at=timestamp, updated_at=timestamp)
            session.add(chat_session)
            session.flush()
        session.add(ChatMessage(id=str(uuid.uuid4()), session_id=chat_session.id, role="user", content=payload.message.strip(), created_at=timestamp))
        message_id = str(uuid.uuid4())
        session.add(ChatMessage(id=message_id, session_id=chat_session.id, role="assistant", content=answer, citations_json=json.dumps(contexts, ensure_ascii=False), created_at=timestamp))
        chat_session.updated_at = timestamp
        return {"id": message_id, "session_id": chat_session.id, "role": "assistant", "content": answer, "citations": contexts}


@router.post("/chat/stream", summary="Stream answer with Server-Sent Events")
def post_chat_stream(knowledge_base_id: str, payload: ChatStreamRequest) -> StreamingResponse:
    try:
        stream, contexts = lc_answer_stream(knowledge_base_id, payload.message.strip())
        timestamp = datetime.now(timezone.utc)
        with get_session() as session:
            if payload.session_id:
                chat_session = session.query(ChatSession).filter(ChatSession.id == payload.session_id, ChatSession.knowledge_base_id == knowledge_base_id).one_or_none()
                if chat_session is None:
                    raise HTTPException(status_code=404, detail="会话不存在")
            else:
                chat_session = ChatSession(id=str(uuid.uuid4()), knowledge_base_id=knowledge_base_id, title=payload.message.strip()[:40], created_at=timestamp, updated_at=timestamp)
                session.add(chat_session)
                session.flush()
            session.add(ChatMessage(id=str(uuid.uuid4()), session_id=chat_session.id, role="user", content=payload.message.strip(), created_at=timestamp))
            session_id = chat_session.id

        def events():
            # SSE 事件分为元数据、增量文本和结束信息，前端无需猜测字符串含义。
            yield f"event: meta\ndata: {json.dumps({'session_id': session_id}, ensure_ascii=False)}\n\n"
            answer_parts = []
            try:
                for chunk in stream:
                    answer_parts.append(chunk)
                    yield f"event: delta\ndata: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
                answer = "".join(answer_parts)
                with get_session() as session:
                    item = session.get(ChatSession, session_id)
                    if item:
                        session.add(ChatMessage(id=str(uuid.uuid4()), session_id=session_id, role="assistant", content=answer, citations_json=json.dumps(contexts, ensure_ascii=False), created_at=datetime.now(timezone.utc)))
                        item.updated_at = datetime.now(timezone.utc)
                yield f"event: done\ndata: {json.dumps({'citations': contexts}, ensure_ascii=False)}\n\n"
            except Exception as error:
                yield f"event: error\ndata: {json.dumps({'message': str(error)}, ensure_ascii=False)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    except Exception as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

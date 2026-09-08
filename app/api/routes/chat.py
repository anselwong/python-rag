"""知识库问答接口：先检索，再把证据交给大模型。"""

import json
from datetime import datetime, timezone
from typing import Dict

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.schemas.chat import ChatRequest, ChatResponse, ChatSessionRenameRequest
from app.schemas.chat_stream import ChatStreamRequest
from app.schemas.retrieval import RetrievalRequest
from app.langchain.service import ChatSessionNotFoundError, answer as lc_answer, answer_stream as lc_answer_stream
from app.core.database import ChatSession, get_session
from app.core.database import User
from app.api.dependencies import get_current_user
from app.services.knowledge import ensure_knowledge_base_owner

router = APIRouter(prefix="/knowledge-bases/{knowledge_base_id}")


@router.get("/chat-sessions", summary="List persisted chat sessions")
def get_chat_sessions(knowledge_base_id: str, current_user: User = Depends(get_current_user)) -> list[dict]:
    try:
        ensure_knowledge_base_owner(knowledge_base_id, current_user.id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    with get_session() as session:
        rows = session.query(ChatSession).filter(ChatSession.knowledge_base_id == knowledge_base_id).order_by(ChatSession.updated_at.desc()).all()
        return [{"id": row.id, "title": row.title, "updated_at": row.updated_at.isoformat(), "messages": [{"id": message.id, "role": message.role, "content": message.content, "created_at": message.created_at.isoformat(), "citations": json.loads(message.citations_json), "usage": json.loads(message.usage_json)} for message in sorted(row.messages, key=lambda value: value.created_at)]} for row in rows]


@router.patch("/chat-sessions/{session_id}", summary="Rename a chat session")
def rename_chat_session(knowledge_base_id: str, session_id: str, payload: ChatSessionRenameRequest, current_user: User = Depends(get_current_user)) -> dict:
    """修改会话标题，并校验会话属于当前知识库，防止越权修改其他知识库数据。"""
    title = payload.title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="会话名称不能为空")
    try:
        ensure_knowledge_base_owner(knowledge_base_id, current_user.id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
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
def delete_chat_session(knowledge_base_id: str, session_id: str, current_user: User = Depends(get_current_user)) -> None:
    """删除会话；ORM cascade 与数据库外键会一并删除其消息记录。"""
    try:
        ensure_knowledge_base_owner(knowledge_base_id, current_user.id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    with get_session() as session:
        item = session.query(ChatSession).filter(
            ChatSession.id == session_id,
            ChatSession.knowledge_base_id == knowledge_base_id,
        ).one_or_none()
        if item is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        session.delete(item)


@router.post("/chat", response_model=ChatResponse, summary="Answer with retrieved context")
def post_chat(knowledge_base_id: str, payload: ChatRequest, current_user: User = Depends(get_current_user)) -> Dict:
    try:
        ensure_knowledge_base_owner(knowledge_base_id, current_user.id)
        answer, contexts, session_id, message_id, usage = lc_answer(
            knowledge_base_id,
            payload.message.strip(),
            session_id=payload.session_id,
        )
    except ChatSessionNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return {"id": message_id, "session_id": session_id, "role": "assistant", "content": answer, "citations": contexts, "usage": usage}


@router.post("/chat/stream", summary="Stream answer with Server-Sent Events")
def post_chat_stream(knowledge_base_id: str, payload: ChatStreamRequest, current_user: User = Depends(get_current_user)) -> StreamingResponse:
    try:
        ensure_knowledge_base_owner(knowledge_base_id, current_user.id)
        stream, contexts, session_id, usage = lc_answer_stream(
            knowledge_base_id,
            payload.message.strip(),
            session_id=payload.session_id,
        )

        def events():
            # SSE 事件分为元数据、增量文本和结束信息，前端无需猜测字符串含义。
            yield f"event: meta\ndata: {json.dumps({'session_id': session_id}, ensure_ascii=False)}\n\n"
            try:
                for chunk in stream:
                    yield f"event: delta\ndata: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
                # 只有流完整结束后才能确认 usage；前端据此展示真实输入/输出消费。
                yield f"event: done\ndata: {json.dumps({'citations': contexts, 'usage': usage.value}, ensure_ascii=False)}\n\n"
            except Exception as error:
                yield f"event: error\ndata: {json.dumps({'message': str(error)}, ensure_ascii=False)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    except ChatSessionNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

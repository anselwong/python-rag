"""知识库问答接口：先检索，再把证据交给大模型。"""

import uuid
from typing import Dict

from fastapi import APIRouter, HTTPException

from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.retrieval import RetrievalRequest
from app.services.llm import LLMError, generate_answer
from app.services.retrieval import search

router = APIRouter(prefix="/knowledge-bases/{knowledge_base_id}")


@router.post("/chat", response_model=ChatResponse, summary="Answer with retrieved context")
def post_chat(knowledge_base_id: str, payload: ChatRequest) -> Dict:
    contexts = search(knowledge_base_id, payload.message.strip(), top_k=5, score_threshold=0.35)
    try:
        answer = generate_answer(payload.message.strip(), contexts)
    except LLMError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return {"id": str(uuid.uuid4()), "role": "assistant", "content": answer, "citations": contexts}

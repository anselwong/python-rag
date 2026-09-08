"""检索调试 API。"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException

from app.schemas.retrieval import RetrievalRequest, RetrievalResultResponse
from app.langchain.service import retrieve
from app.core.database import User
from app.api.dependencies import get_current_user
from app.services.knowledge import ensure_knowledge_base_owner

router = APIRouter(prefix="/knowledge-bases/{knowledge_base_id}/retrieval")


@router.post("/search", response_model=List[RetrievalResultResponse], summary="Vector search chunks")
def post_search(knowledge_base_id: str, payload: RetrievalRequest, current_user: User = Depends(get_current_user)) -> List[dict]:
    try:
        ensure_knowledge_base_owner(knowledge_base_id, current_user.id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return retrieve(
        knowledge_base_id,
        payload.query.strip(),
        payload.top_k,
        payload.score_threshold,
        payload.mode,
        vector_weight=payload.vector_weight,
        keyword_weight=payload.keyword_weight,
    )

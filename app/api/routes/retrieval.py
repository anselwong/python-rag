"""检索调试 API。"""

from typing import List

from fastapi import APIRouter

from app.schemas.retrieval import RetrievalRequest, RetrievalResultResponse
from app.langchain.service import retrieve

router = APIRouter(prefix="/knowledge-bases/{knowledge_base_id}/retrieval")


@router.post("/search", response_model=List[RetrievalResultResponse], summary="Vector search chunks")
def post_search(knowledge_base_id: str, payload: RetrievalRequest) -> List[dict]:
    return retrieve(
        knowledge_base_id,
        payload.query.strip(),
        payload.top_k,
        payload.score_threshold,
        payload.mode,
        vector_weight=payload.vector_weight,
        keyword_weight=payload.keyword_weight,
    )

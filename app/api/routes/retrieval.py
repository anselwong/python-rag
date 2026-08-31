"""检索调试 API。"""

from typing import List

from fastapi import APIRouter

from app.schemas.retrieval import RetrievalRequest, RetrievalResultResponse
from app.services.retrieval import search

router = APIRouter(prefix="/knowledge-bases/{knowledge_base_id}/retrieval")


@router.post("/search", response_model=List[RetrievalResultResponse], summary="Vector search chunks")
def post_search(knowledge_base_id: str, payload: RetrievalRequest) -> List[dict]:
    return search(knowledge_base_id, payload.query.strip(), payload.top_k, payload.score_threshold)

"""知识库问答 API 契约。"""

from typing import List

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class CitationResponse(BaseModel):
    id: str
    document_name: str
    page: int
    content: str
    score: float


class ChatResponse(BaseModel):
    id: str
    role: str
    content: str
    citations: List[CitationResponse]

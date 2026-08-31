"""知识库问答 API 契约。"""

from typing import List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: Optional[str] = None


class CitationResponse(BaseModel):
    id: str
    document_name: str
    page: int
    content: str
    score: float


class ChatResponse(BaseModel):
    session_id: str
    id: str
    role: str
    content: str
    citations: List[CitationResponse]


class ChatSessionResponse(BaseModel):
    id: str
    title: str
    updated_at: str
    messages: List[ChatResponse]

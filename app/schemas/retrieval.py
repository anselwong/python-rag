"""检索接口的请求与响应契约。"""

from typing import List, Literal

from pydantic import BaseModel, Field


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    score_threshold: float = Field(default=0.0, ge=-1.0, le=1.0)
    mode: Literal["vector", "hybrid"] = "vector"


class RetrievalResultResponse(BaseModel):
    id: str
    document_name: str
    page: int
    content: str
    score: float
    rank: int
    keywords: List[str]

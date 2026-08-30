from datetime import datetime
from typing import List

from pydantic import BaseModel, Field


class KnowledgeBaseCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=240)


class KnowledgeBaseResponse(BaseModel):
    id: str
    name: str
    description: str
    document_count: int
    chunk_count: int
    updated_at: datetime
    color: str


class DocumentResponse(BaseModel):
    id: str
    knowledge_base_id: str
    name: str
    type: str
    size: str
    chunk_count: int
    status: str
    created_at: datetime


class DocumentPageResponse(BaseModel):
    document_id: str
    page: int
    text: str


class DocumentChunkResponse(BaseModel):
    """切片预览所需的最小字段；embedding 不通过接口返回，避免泄露大数组。"""

    id: str
    document_id: str
    page: int
    content: str
    token_count: int


class DocumentDetailResponse(DocumentResponse):
    pages: List[DocumentPageResponse]
    chunks: List[DocumentChunkResponse]

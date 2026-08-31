"""流式问答请求契约。"""

from pydantic import BaseModel, Field


class ChatStreamRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)

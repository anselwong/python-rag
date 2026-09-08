"""PostgreSQL 会话记录与 LangChain 消息历史之间的适配层。"""

import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.core.database import ChatMessage, ChatSession, get_session
from .tokenizer import count_messages
from .usage import get_message_usage


class SqlChatMessageHistory(BaseChatMessageHistory):
    """把 ``chat_messages`` 作为 LangChain 的唯一消息历史存储。

    ``RunnableWithMessageHistory`` 只面向 ``BaseChatMessageHistory`` 协议，
    不关心消息落在 Redis、文件还是数据库。这个适配器将其自动读写的
    ``HumanMessage``、``AIMessage`` 映射到既有业务表，同时继续保留引用元数据。
    """

    def __init__(self, session_id: str, max_messages: int = 6, max_tokens: Optional[int] = None) -> None:
        self.session_id = session_id
        self.max_messages = max_messages
        self.max_tokens = max_tokens if max_tokens is not None else int(os.getenv("HISTORY_TOKEN_BUDGET", "1200"))
        # Runnable 在本轮完成时调用 add_messages；保留助手消息 ID，才能把
        # 检索得到的 citations 精确回填给本轮答案，而不是按“最新一条”猜测。
        self.last_assistant_message_id: Optional[str] = None

    @property
    def messages(self) -> List[BaseMessage]:
        """按时间正序读取窗口内消息，供 ``MessagesPlaceholder`` 注入 Prompt。"""
        with get_session() as session:
            rows = (
                session.query(ChatMessage)
                .filter(ChatMessage.session_id == self.session_id)
                .order_by(ChatMessage.created_at.desc())
                .limit(self.max_messages)
                .all()
            )
        messages = [
            HumanMessage(content=row.content)
            if row.role == "user"
            else AIMessage(content=row.content)
            for row in reversed(rows)
        ]
        # 从最新消息向前保留，保证超出预算时优先丢弃最早的轮次。与固定 6 条
        # 窗口叠加，既限制轮数，也限制单条超长回答造成的上下文膨胀。
        selected: List[BaseMessage] = []
        for message in reversed(messages):
            if selected and count_messages([message] + selected) > self.max_tokens:
                break
            selected.append(message)
        return list(reversed(selected))

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """由 LangChain 在一次链调用成功后批量写入用户问题和模型回答。"""
        timestamp = datetime.now(timezone.utc)
        with get_session() as session:
            chat_session = session.get(ChatSession, self.session_id)
            if chat_session is None:
                raise ValueError("会话不存在，无法保存消息历史")

            for offset, message in enumerate(messages):
                if isinstance(message, HumanMessage):
                    role = "user"
                elif isinstance(message, AIMessage):
                    role = "assistant"
                else:
                    # 当前 RAG Prompt 只允许用户和助手消息进入历史；忽略系统
                    # 消息可避免把每轮相同的系统约束重复持久化、重复消耗 token。
                    continue

                message_id = str(uuid.uuid4())
                session.add(
                    ChatMessage(
                        id=message_id,
                        session_id=self.session_id,
                        role=role,
                        content=str(message.content),
                        # Runnable 在普通/流式链结束时会传入完整 AIMessage；先把
                        # 其中可能带的 usage 保存下来，服务层随后会以最终采集值回填。
                        usage_json=json.dumps(get_message_usage(message), ensure_ascii=False),
                        # 同一轮消息用微秒顺序，避免数据库在时间相同的情况下
                        # 无法稳定还原“用户 -> 助手”的历史顺序。
                        created_at=timestamp + timedelta(microseconds=offset),
                    )
                )
                if role == "assistant":
                    self.last_assistant_message_id = message_id
            chat_session.updated_at = timestamp

    def clear(self) -> None:
        """清空当前会话消息；会话删除接口通常通过外键级联完成相同操作。"""
        with get_session() as session:
            session.query(ChatMessage).filter(ChatMessage.session_id == self.session_id).delete()

    def attach_response_metadata(self, citations: list[dict], usage: Optional[dict]) -> None:
        """为本轮助手消息回填引用和模型真实 usage。"""
        if not self.last_assistant_message_id:
            return
        with get_session() as session:
            message = session.get(ChatMessage, self.last_assistant_message_id)
            if message is not None:
                message.citations_json = json.dumps(citations, ensure_ascii=False)
                message.usage_json = json.dumps(usage, ensure_ascii=False)

"""LCEL 问答链：Prompt -> ChatModel，普通与流式调用共享同一 Runnable。"""
import os
from typing import Iterable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from .history import SqlChatMessageHistory
from .prompts import RAG_PROMPT


def _build_model() -> ChatOpenAI:
    return ChatOpenAI(model=os.getenv("LLM_MODEL", "deepseek-chat"), api_key=os.getenv("LLM_API_KEY"), base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"), temperature=0.2, streaming=True)


def build_chain(history: SqlChatMessageHistory) -> RunnableWithMessageHistory:
    """构建带数据库消息历史的 LCEL 链。

    ``input_messages_key`` 指明当前问题来自 ``question``，
    ``history_messages_key`` 指明 Prompt 中的 ``MessagesPlaceholder``。调用成功后
    Runnable 会自动执行 history.add_messages()，路由层无需再手动写 chat_messages。
    """
    base_chain = RAG_PROMPT | _build_model() | StrOutputParser()
    return RunnableWithMessageHistory(
        base_chain,
        lambda _session_id: history,
        input_messages_key="question",
        history_messages_key="history",
    )

def _format_context(contexts: list[dict]) -> str:
    return "\n\n".join(f"[{i}] {item['document_name']} 第{item['page']}页\n{item['content']}" for i, item in enumerate(contexts, 1)) or "（没有检索到足够资料）"


def run(question: str, contexts: list[dict], history: SqlChatMessageHistory) -> str:
    return build_chain(history).invoke(
        {"question": question, "context": _format_context(contexts)},
        config={"configurable": {"session_id": history.session_id}},
    )

def stream(question: str, contexts: list[dict], history: SqlChatMessageHistory) -> Iterable[str]:
    return build_chain(history).stream(
        {"question": question, "context": _format_context(contexts)},
        config={"configurable": {"session_id": history.session_id}},
    )

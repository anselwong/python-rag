"""LCEL 问答链：Prompt -> ChatModel，普通与流式调用共享同一 Runnable。"""
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Optional, Tuple
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from .history import SqlChatMessageHistory
from .prompts import RAG_PROMPT
from .usage import get_message_usage


def _build_model() -> ChatOpenAI:
    # stream_usage 会要求 OpenAI 兼容服务在流的最后一个 chunk 返回 usage。
    # DeepSeek 不支持时链仍能正常回答，只是 usage 保持 None，绝不使用估算值冒充。
    return ChatOpenAI(model=os.getenv("LLM_MODEL", "deepseek-chat"), api_key=os.getenv("LLM_API_KEY"), base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"), temperature=0.2, streaming=True, stream_usage=True)


def build_chain(history: SqlChatMessageHistory) -> RunnableWithMessageHistory:
    """构建带数据库消息历史的 LCEL 链。

    ``input_messages_key`` 指明当前问题来自 ``question``，
    ``history_messages_key`` 指明 Prompt 中的 ``MessagesPlaceholder``。调用成功后
    Runnable 会自动执行 history.add_messages()，路由层无需再手动写 chat_messages。
    """
    # 不接 StrOutputParser：它会把 AIMessage 的 usage_metadata 丢成纯字符串。
    # 保留消息对象后，普通 invoke 和流式最后一个 chunk 都能读取真实用量。
    base_chain = RAG_PROMPT | _build_model()
    return RunnableWithMessageHistory(
        base_chain,
        lambda _session_id: history,
        input_messages_key="question",
        history_messages_key="history",
    )

def _format_context(contexts: list[dict]) -> str:
    return "\n\n".join(f"[{i}] {item['document_name']} 第{item['page']}页\n{item['content']}" for i, item in enumerate(contexts, 1)) or "（没有检索到足够资料）"


def run(question: str, contexts: list[dict], history: SqlChatMessageHistory) -> Tuple[str, Optional[Dict[str, int]]]:
    message = build_chain(history).invoke(
        {"question": question, "context": _format_context(contexts)},
        config={"configurable": {"session_id": history.session_id}},
    )
    return str(message.content), get_message_usage(message)

@dataclass
class StreamUsage:
    """流式迭代完成后由调用方读取的真实用量容器。"""

    value: Optional[Dict[str, int]] = None


def stream(question: str, contexts: list[dict], history: SqlChatMessageHistory) -> Tuple[Iterable[str], StreamUsage]:
    chunks = build_chain(history).stream(
        {"question": question, "context": _format_context(contexts)},
        config={"configurable": {"session_id": history.session_id}},
    )
    usage = StreamUsage()

    def text_chunks() -> Iterator[str]:
        for chunk in chunks:
            chunk_usage = get_message_usage(chunk)
            if chunk_usage:
                usage.value = chunk_usage
            content = str(chunk.content or "")
            if content:
                yield content

    return text_chunks(), usage

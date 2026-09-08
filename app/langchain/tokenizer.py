"""本地 Token 计算与调用前上下文预算。

DeepSeek 的账单 Token 应以 API 响应 ``usage`` 为准；本模块使用 ``cl100k_base``
在请求发送前做确定性的本地估算，解决切片和 Prompt 不能超过预算的问题。
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List

import tiktoken
from langchain_core.messages import BaseMessage


ENCODING_NAME = os.getenv("TOKENIZER_ENCODING", "cl100k_base")
# tiktoken 默认缓存可能落在用户目录。统一到项目 data 目录后，本地首次加载的
# 词表可复用，且 data 已在 .gitignore 中，不会把二进制词表提交到仓库。Docker
# 显式设置的 /opt/tiktoken 优先，不会被这里覆盖。
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(Path(os.getenv("RAG_DATA_DIR", "data")) / "tiktoken"))


@lru_cache(maxsize=1)
def get_encoding():
    """缓存词表加载结果，避免每次切片或问答都重新初始化 tokenizer。"""
    return tiktoken.get_encoding(ENCODING_NAME)


def count_tokens(text: str) -> int:
    """返回文本按本地编码切分后的 Token 数，不等同于字符数。"""
    return len(get_encoding().encode(text, disallowed_special=()))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """按 Token 边界截断文本，避免按字符硬切破坏多字节编码或低估预算。"""
    if max_tokens <= 0:
        return ""
    token_ids = get_encoding().encode(text, disallowed_special=())
    if len(token_ids) <= max_tokens:
        return text
    # 单个中文字符偶尔会跨越多个 BPE token。直接 decode 半个字符会得到 ``�``，
    # 因此逐个回退到可完整解码的边界，宁可少一个 token 也不损坏上下文原文。
    for end in range(max_tokens, 0, -1):
        candidate = get_encoding().decode(token_ids[:end])
        if "\ufffd" not in candidate:
            return candidate
    return ""


def count_messages(messages: Iterable[BaseMessage]) -> int:
    """估算 Chat Completions 消息结构的 Token 开销。

    每条消息除正文外还包含 role、分隔符等协议字段。不同模型的包装规则略有
    差异，因此此值用于保守预算，不应用作供应商账单核对依据。
    """
    return 3 + sum(3 + count_tokens(str(message.content)) for message in messages)


def fit_contexts_to_budget(
    question: str,
    history: Iterable[BaseMessage],
    contexts: List[dict],
    system_prompt: str,
) -> List[dict]:
    """按统一预算选择可放入 Prompt 的检索片段。

    总窗口拆为系统约束、当前问题、历史、检索上下文与模型输出预留。历史已由
    ``RunnableWithMessageHistory`` 注入，这里只控制检索片段，优先保留 Rerank
    排名靠前的内容；首个片段过长时按 Token 边界截断，而非整条丢弃。
    """
    context_window = int(os.getenv("LLM_CONTEXT_WINDOW", "8192"))
    output_reserve = int(os.getenv("LLM_OUTPUT_RESERVE_TOKENS", "1024"))
    configured_context_limit = int(os.getenv("MAX_CONTEXT_TOKENS", "3000"))
    # 模板中的“问题/检索资料”标签、引用编号、Chat Completions 包装字段等不属于
    # 任一正文，但同样会占 Token；预留固定余量避免预算刚好贴边后仍触发上游超限。
    prompt_overhead = int(os.getenv("PROMPT_FORMAT_OVERHEAD_TOKENS", "64"))
    fixed_tokens = count_tokens(system_prompt) + count_tokens(question) + count_messages(history) + prompt_overhead
    available = min(configured_context_limit, context_window - output_reserve - fixed_tokens)
    if available <= 0:
        return []

    selected: List[dict] = []
    for index, context in enumerate(contexts, 1):
        prefix = f"[{index}] {context['document_name']} 第{context['page']}页\n"
        prefix_tokens = count_tokens(prefix)
        remaining = available - sum(count_tokens(f"[{position}] {item['document_name']} 第{item['page']}页\n{item['content']}") for position, item in enumerate(selected, 1))
        if remaining <= prefix_tokens:
            break
        content = context["content"]
        total_tokens = prefix_tokens + count_tokens(content)
        if total_tokens <= remaining:
            selected.append(context.copy())
            continue
        # 只允许最后一个片段使用剩余空间，避免空上下文时完全失去最高分证据。
        truncated = truncate_to_tokens(content, remaining - prefix_tokens)
        if truncated:
            selected.append({**context, "content": truncated})
        break
    return selected

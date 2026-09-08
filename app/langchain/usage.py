"""归一化模型服务返回的真实 Token 用量。"""

from typing import Any, Dict, Optional


def normalize_usage(raw_usage: Optional[Dict[str, Any]]) -> Optional[Dict[str, int]]:
    """将不同 OpenAI 兼容字段统一为前端可消费的 usage。

    LangChain 新版本倾向使用 input/output_tokens，部分供应商仍返回
    prompt/completion_tokens。这里不使用本地 tokenizer 兜底，因为“真实用量”
    必须只来自模型服务响应，拿不到就明确返回空值而不是伪造数据。
    """
    if not raw_usage:
        return None
    prompt_tokens = raw_usage.get("prompt_tokens", raw_usage.get("input_tokens"))
    completion_tokens = raw_usage.get("completion_tokens", raw_usage.get("output_tokens"))
    total_tokens = raw_usage.get("total_tokens")
    if prompt_tokens is None and completion_tokens is None and total_tokens is None:
        return None
    prompt = int(prompt_tokens or 0)
    completion = int(completion_tokens or 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": int(total_tokens if total_tokens is not None else prompt + completion),
    }


def get_message_usage(message: Any) -> Optional[Dict[str, int]]:
    """从 LangChain 消息或增量消息读取供应商返回的 usage。"""
    usage = normalize_usage(getattr(message, "usage_metadata", None))
    if usage:
        return usage
    response_metadata = getattr(message, "response_metadata", {}) or {}
    return normalize_usage(response_metadata.get("token_usage") or response_metadata.get("usage"))

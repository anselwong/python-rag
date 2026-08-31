"""大模型回答服务（Day 7）。兼容 DeepSeek 等 OpenAI Chat Completions 接口。"""

import os
import json
from typing import Dict, Iterable, List

import httpx
from dotenv import load_dotenv
from app.services.chunker import estimate_tokens

load_dotenv(override=False)


class LLMError(RuntimeError):
    """大模型配置缺失或远端调用失败。"""


MAX_CONTEXT_TOKENS = 3000


def generate_answer(question: str, contexts: List[Dict]) -> str:
    """只依据检索上下文生成回答；没有证据时明确拒答，降低幻觉。"""
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com").rstrip("/")
    model = os.getenv("LLM_MODEL", "deepseek-chat")
    if not api_key:
        raise LLMError("缺少 LLM_API_KEY，请检查 .env")
    evidence = _build_evidence(contexts)
    prompt = f"问题：{question}\n\n检索到的资料：\n{evidence or '（没有检索到足够资料）'}"
    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": "你是企业知识库助手。只能依据用户提供的检索资料回答；资料不足时回答‘根据当前知识库无法确认’，不要编造信息。回答简洁，并用[1]、[2]标记引用。"},
            {"role": "user", "content": prompt},
        ],
    }
    try:
        response = httpx.post(f"{base_url}/chat/completions", headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json=payload, timeout=60)
    except httpx.HTTPError as error:
        raise LLMError("无法连接聊天模型服务") from error
    if response.status_code >= 400:
        raise LLMError(f"聊天模型调用失败（HTTP {response.status_code}）")
    try:
        return response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise LLMError("聊天模型返回格式异常") from error


def _build_evidence(contexts: List[Dict]) -> str:
    """按相似度顺序装入上下文，超过预算的切片直接截断。

    上下文窗口不是无限的：盲目把所有召回结果塞给模型会增加费用，并可能让模型
    忽略真正相关的片段。这里用切片已有 token_count/启发式估算做保守预算。
    """
    selected: List[str] = []
    used = 0
    for index, item in enumerate(contexts):
        text = f"[{index + 1}] {item['document_name']} 第{item['page']}页\n{item['content']}"
        tokens = estimate_tokens(text)
        if selected and used + tokens > MAX_CONTEXT_TOKENS:
            break
        selected.append(text)
        used += tokens
    return "\n\n".join(selected) or "（没有检索到足够资料）"


def stream_answer(question: str, contexts: List[Dict]) -> Iterable[str]:
    """以 SSE 数据片段输出模型回答；每个 data 块都能被前端即时渲染。"""
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com").rstrip("/")
    model = os.getenv("LLM_MODEL", "deepseek-chat")
    if not api_key:
        raise LLMError("缺少 LLM_API_KEY，请检查 .env")
    payload = {"model": model, "temperature": 0.2, "stream": True, "messages": [
        {"role": "system", "content": "你是企业知识库助手。只能依据检索资料回答；资料不足时回答‘根据当前知识库无法确认’，不要编造信息。"},
        {"role": "user", "content": f"问题：{question}\n\n检索资料：\n{_build_evidence(contexts)}"},
    ]}
    try:
        with httpx.stream("POST", f"{base_url}/chat/completions", headers={"Authorization": f"Bearer {api_key}"}, json=payload, timeout=60) as response:
            if response.status_code >= 400:
                raise LLMError(f"聊天模型调用失败（HTTP {response.status_code}）")
            for line in response.iter_lines():
                if line.startswith("data: ") and line[6:] != "[DONE]":
                    try:
                        delta = json.loads(line[6:])["choices"][0].get("delta", {}).get("content", "")
                    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
                        continue
                    if delta:
                        yield delta
    except httpx.HTTPError as error:
        raise LLMError("无法连接聊天模型服务") from error

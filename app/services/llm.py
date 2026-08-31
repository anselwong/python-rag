"""大模型回答服务（Day 7）。兼容 DeepSeek 等 OpenAI Chat Completions 接口。"""

import os
from typing import Dict, List

import httpx
from dotenv import load_dotenv

load_dotenv(override=False)


class LLMError(RuntimeError):
    """大模型配置缺失或远端调用失败。"""


def generate_answer(question: str, contexts: List[Dict]) -> str:
    """只依据检索上下文生成回答；没有证据时明确拒答，降低幻觉。"""
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com").rstrip("/")
    model = os.getenv("LLM_MODEL", "deepseek-chat")
    if not api_key:
        raise LLMError("缺少 LLM_API_KEY，请检查 .env")
    evidence = "\n\n".join(f"[{index + 1}] {item['document_name']} 第{item['page']}页\n{item['content']}" for index, item in enumerate(contexts))
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

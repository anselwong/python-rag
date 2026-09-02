"""Rerank 精排服务：对初步召回的候选片段进行更精确的相关性排序。"""

import os
import logging
from typing import Dict, List

import httpx
from dotenv import load_dotenv

load_dotenv(override=False)

logger = logging.getLogger(__name__)
# 本地调试阶段使用 WARNING，兼容当前 Uvicorn 启动配置的日志过滤级别；
# 生产环境应改为 INFO 并交给统一日志平台采集。
logger.setLevel(logging.WARNING)


class RerankError(RuntimeError):
    """Rerank 服务配置或远端响应异常。"""


def rerank(query: str, contexts: List[Dict], top_n: int) -> List[Dict]:
    """调用 OpenAI 兼容 Rerank 接口，并保留原始引用元数据。"""
    logger.warning("[Rerank] 进入精排: query_chars=%d, candidates=%d", len(query), len(contexts))
    api_key = os.getenv("RERANK_API_KEY")
    base_url = os.getenv("RERANK_BASE_URL", "").rstrip("/")
    model = os.getenv("RERANK_MODEL", "").strip()
    if not api_key or not base_url or not model:
        logger.warning("[Rerank] 未配置完整，跳过精排: provider=%s, model=%s, base_url=%s", os.getenv("RERANK_PROVIDER", ""), model or "<empty>", bool(base_url))
        return contexts[:top_n]
    provider = os.getenv("RERANK_PROVIDER", "").lower()
    documents = [item["content"] for item in contexts]
    if provider == "dashscope" and model == "qwen3-rerank":
        # qwen3-rerank 使用百炼兼容 OpenAI 风格接口：路径是复数 reranks。
        # 兼容用户此前填写的 /api/v1，自动切换到官方要求的 /compatible-api/v1。
        compatible_base = base_url.replace("/api/v1", "/compatible-api/v1")
        endpoint = f"{compatible_base}/reranks"
        payload = {"model": model, "query": query, "documents": documents, "top_n": top_n}
    elif provider == "dashscope":
        # 其他百炼排序模型使用原生 services 路径和 input/parameters 两层请求体。
        endpoint = f"{base_url}/services/rerank/text-rerank/text-rerank"
        payload = {"model": model, "input": {"query": query, "documents": documents}, "parameters": {"top_n": top_n}}
    else:
        endpoint = f"{base_url}/rerank"
        payload = {"model": model, "query": query, "documents": documents, "top_n": top_n}
    logger.warning("[Rerank] 开始请求: provider=%s, endpoint=%s, model=%s, candidates=%d, top_n=%d", provider, endpoint, model, len(contexts), top_n)
    try:
        response = httpx.post(endpoint, headers={"Authorization": f"Bearer {api_key}"}, json=payload, timeout=30)
        logger.warning("[Rerank] 收到响应: status=%d", response.status_code)
        response.raise_for_status()
        data = response.json()
        # 百炼结果嵌套在 output.results；兼容其他服务的 results/data 返回格式。
        output = data.get("output") or {}
        if provider == "dashscope" and model != "qwen3-rerank":
            results = output.get("results", [])
        else:
            results = data.get("results", data.get("data", []))
        logger.warning("[Rerank] 解析结果: result_count=%d", len(results) if isinstance(results, list) else 0)
        ordered = []
        for result in results:
            index = int(result.get("index", 0))
            if 0 <= index < len(contexts):
                item = dict(contexts[index])
                item["score"] = float(result.get("relevance_score", result.get("score", item["score"])))
                ordered.append(item)
        return ordered[:top_n] or contexts[:top_n]
    except (httpx.HTTPError, ValueError, KeyError, TypeError, IndexError) as error:
        # 精排是召回后的增强步骤；失败时保留 hybrid 结果，保证主问答链路可用。
        logger.exception("[Rerank] 调用失败，将回退到 hybrid: %s", error)
        raise RerankError("Rerank 服务调用失败") from error

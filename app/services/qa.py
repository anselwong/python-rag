"""共享问答编排：检索和上下文准备由普通、流式接口共同复用。"""

from typing import Dict, List

from app.services.retrieval import search
from app.services.rerank import RerankError, rerank


def retrieve_contexts(knowledge_base_id: str, question: str, top_k: int = 5, threshold: float = 0.35, mode: str = "hybrid") -> List[Dict]:
    """统一问答检索入口，确保 `/chat` 与 `/chat/stream` 使用相同候选证据。"""
    contexts = search(knowledge_base_id, question, top_k=max(top_k, 10), score_threshold=threshold, mode=mode)
    try:
        # 先扩大候选池，再由 Rerank 从中挑选最终上下文；未配置时服务函数直接返回原排序。
        return rerank(question, contexts, top_n=top_k)
    except RerankError:
        return contexts[:top_k]

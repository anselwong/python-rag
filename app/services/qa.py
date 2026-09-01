"""共享问答编排：检索和上下文准备由普通、流式接口共同复用。"""

from typing import Dict, List

from app.services.retrieval import search


def retrieve_contexts(knowledge_base_id: str, question: str, top_k: int = 5, threshold: float = 0.35, mode: str = "hybrid") -> List[Dict]:
    """统一问答检索入口，确保 `/chat` 与 `/chat/stream` 使用相同候选证据。"""
    return search(knowledge_base_id, question, top_k=top_k, score_threshold=threshold, mode=mode)

"""检索评测指标（Day 10）：不调用模型即可验证召回质量。"""

from typing import Iterable, List, Sequence


def recall_at_k(retrieved_ids: Sequence[str], expected_id: str, k: int) -> float:
    """正确切片出现在前 K 条返回 1，否则返回 0。"""
    return float(expected_id in retrieved_ids[:k])


def reciprocal_rank(retrieved_ids: Sequence[str], expected_id: str) -> float:
    """MRR 的单题倒数排名；越靠前越接近 1。"""
    try:
        return 1.0 / (retrieved_ids.index(expected_id) + 1)
    except ValueError:
        return 0.0


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0

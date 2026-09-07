"""评测扫描（Day 13）：一次运行多组检索配置，对比召回质量。"""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class SweepConfig(BaseModel):
    """单组检索配置：模式 + 融合权重 + 过滤阈值。"""

    mode: Literal["vector", "hybrid"] = "hybrid"
    vector_weight: float = Field(default=0.7, ge=0.0, le=1.0, description="向量相似度权重（hybrid 模式生效）")
    keyword_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="BM25 关键词权重（hybrid 模式生效）")
    threshold: float = Field(default=0.0, ge=-1.0, le=1.0, description="分数过滤阈值，用于校准默认 0.35")
    top_k: int = Field(default=5, ge=1, le=20)


class SweepRequest(BaseModel):
    """扫描请求；configs 为空时使用内置默认扫描集。"""

    configs: Optional[List[SweepConfig]] = None


DEFAULT_SWEEP_CONFIGS: List[SweepConfig] = [
    SweepConfig(mode="vector", threshold=0.0),
    SweepConfig(mode="vector", threshold=0.35),
    SweepConfig(mode="hybrid", vector_weight=0.7, keyword_weight=0.3, threshold=0.0),
    SweepConfig(mode="hybrid", vector_weight=0.7, keyword_weight=0.3, threshold=0.35),
    SweepConfig(mode="hybrid", vector_weight=0.6, keyword_weight=0.4, threshold=0.0),
    SweepConfig(mode="hybrid", vector_weight=0.5, keyword_weight=0.5, threshold=0.0),
]

"""按知识库隔离的检索评测接口（Day 12/13）。"""

import json
import time
import uuid
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, Depends, HTTPException

from app.core.database import Document, EvaluationCase, get_session
from app.schemas.evaluation import DEFAULT_SWEEP_CONFIGS, SweepConfig, SweepRequest
from app.services.evaluation import mean, recall_at_k, reciprocal_rank
from app.langchain.service import retrieve
from app.core.database import User
from app.api.dependencies import get_current_user
from app.services.knowledge import ensure_knowledge_base_owner


def _check_owner(knowledge_base_id: str, user_id: str) -> None:
    try:
        ensure_knowledge_base_owner(knowledge_base_id, user_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

router = APIRouter(prefix="/knowledge-bases/{knowledge_base_id}/evaluations")


def _generate_cases(knowledge_base_id: str) -> List[dict]:
    """从当前知识库页面生成题目，并把正确文档/页码作为金标准落库。"""
    with get_session() as session:
        documents = session.query(Document).filter(Document.knowledge_base_id == knowledge_base_id).order_by(Document.created_at.asc()).all()
        cases = []
        for document in documents:
            for page in json.loads(document.pages_json or "[]")[:2]:
                text = " ".join(str(page.get("text", "")).split())
                if not text:
                    continue
                snippet = text[:36]
                cases.append(EvaluationCase(id=str(uuid.uuid4()), knowledge_base_id=knowledge_base_id, question=f"文档中关于“{snippet}”的内容是什么？", expected_document_id=document.id, expected_page=int(page.get("page", 1)), created_at=datetime.now(timezone.utc)))
        session.add_all(cases)
        return [{"id": item.id, "question": item.question, "expected_source": next((doc.name for doc in documents if doc.id == item.expected_document_id), ""), "expected_page": item.expected_page, "status": "pending", "faithfulness": None, "retrieval_score": None} for item in cases]


def _ensure_cases(knowledge_base_id: str) -> List[dict]:
    with get_session() as session:
        rows = session.query(EvaluationCase, Document).join(Document, EvaluationCase.expected_document_id == Document.id).filter(EvaluationCase.knowledge_base_id == knowledge_base_id).order_by(EvaluationCase.created_at.asc()).all()
        if rows:
            return [{"id": case.id, "question": case.question, "expected_source": document.name, "expected_page": case.expected_page, "status": "pending", "faithfulness": None, "retrieval_score": None} for case, document in rows]
    return _generate_cases(knowledge_base_id)


def _run(knowledge_base_id: str) -> List[dict]:
    cases = _ensure_cases(knowledge_base_id)
    with get_session() as session:
        expected_docs = {item.id: item.expected_document_id for item in session.query(EvaluationCase).filter(EvaluationCase.knowledge_base_id == knowledge_base_id).all()}
    results = []
    for item in cases:
        started = time.perf_counter()
        hits = retrieve(knowledge_base_id, item["question"], top_k=5, threshold=-1.0, mode="vector")
        elapsed = round((time.perf_counter() - started) * 1000, 2)
        expected = expected_docs[item["id"]]
        hit_ids = [hit["id"] for hit in hits]
        matching = [hit for hit in hits if hit["id"].startswith(expected + "-")]
        first_id = matching[0]["id"] if matching else ""
        results.append({**item, "status": "passed" if matching else "pending", "retrieval_score": matching[0]["score"] if matching else None, "recall_at_k": recall_at_k(hit_ids, first_id, 5) if matching else 0.0, "mrr": reciprocal_rank(hit_ids, first_id) if matching else 0.0, "latency_ms": elapsed})
    return results


@router.get("", summary="List knowledge-base evaluation cases")
def list_evaluations(knowledge_base_id: str, current_user: User = Depends(get_current_user)) -> List[dict]:
    _check_owner(knowledge_base_id, current_user.id)
    return _ensure_cases(knowledge_base_id)


@router.post("/generate", summary="Generate evaluation cases from documents")
def generate_evaluations(knowledge_base_id: str, current_user: User = Depends(get_current_user)) -> List[dict]:
    _check_owner(knowledge_base_id, current_user.id)
    with get_session() as session:
        session.query(EvaluationCase).filter(EvaluationCase.knowledge_base_id == knowledge_base_id).delete(synchronize_session=False)
    return _generate_cases(knowledge_base_id)


@router.post("/run", summary="Run knowledge-base retrieval evaluation")
def run_evaluations(knowledge_base_id: str, current_user: User = Depends(get_current_user)) -> List[dict]:
    _check_owner(knowledge_base_id, current_user.id)
    return _run(knowledge_base_id)


def _run_single(knowledge_base_id: str, question: str, expected_document_id: str, config: SweepConfig) -> dict:
    """按单组配置跑一道评测题，返回命中判定与指标明细。"""
    started = time.perf_counter()
    hits = retrieve(
        knowledge_base_id,
        question,
        top_k=config.top_k,
        threshold=config.threshold,
        mode=config.mode,
        vector_weight=config.vector_weight,
        keyword_weight=config.keyword_weight,
    )
    elapsed = round((time.perf_counter() - started) * 1000, 2)
    # 金标准匹配沿用 Day 12 约定：chunk id 以文档 id 为前缀（f"{did}-{i}"）。
    matching = [hit for hit in hits if hit["id"].startswith(expected_document_id + "-")]
    hit_ids = [hit["id"] for hit in hits]
    first_id = matching[0]["id"] if matching else ""
    return {
        "recall_at_k": recall_at_k(hit_ids, first_id, config.top_k),
        "mrr": reciprocal_rank(hit_ids, first_id),
        "retrieval_score": matching[0]["score"] if matching else None,
        "latency_ms": elapsed,
    }


@router.post("/sweep", summary="Sweep retrieval configs and compare Recall@K / MRR")
def sweep_evaluations(knowledge_base_id: str, payload: SweepRequest, current_user: User = Depends(get_current_user)) -> dict:
    """Day 13 评测深化：一次运行多组配置，横向比较召回质量并给出建议。

    每组配置独立跑全部评测题，聚合 Recall@K、MRR、平均延迟和命中率；
    结果按 Recall@K 降序排列，附带阈值校准建议（默认 0.35 是否合适）。
    """
    _check_owner(knowledge_base_id, current_user.id)
    cases = _ensure_cases(knowledge_base_id)
    with get_session() as session:
        expected_docs = {
            item.id: item.expected_document_id
            for item in session.query(EvaluationCase).filter(EvaluationCase.knowledge_base_id == knowledge_base_id).all()
        }
    configs = payload.configs or DEFAULT_SWEEP_CONFIGS
    results = []
    for config in configs:
        metrics = [
            _run_single(knowledge_base_id, case["question"], expected_docs[case["id"]], config)
            for case in cases
        ]
        scores = [m["retrieval_score"] for m in metrics if m["retrieval_score"] is not None]
        results.append({
            "mode": config.mode,
            "vector_weight": config.vector_weight,
            "keyword_weight": config.keyword_weight,
            "threshold": config.threshold,
            "top_k": config.top_k,
            "recall_at_k": round(mean(m["recall_at_k"] for m in metrics), 4),
            "mrr": round(mean(m["mrr"] for m in metrics), 4),
            "avg_latency_ms": round(mean(m["latency_ms"] for m in metrics), 2),
            "avg_score": round(mean(scores), 4) if scores else None,
            "min_score": min(scores) if scores else None,
        })
    results.sort(key=lambda item: (item["recall_at_k"], item["mrr"]), reverse=True)
    best = results[0] if results else None
    # 阈值校准建议：命中分数的最小值是"不丢正确结果"的安全上限。
    return {
        "knowledge_base_id": knowledge_base_id,
        "case_count": len(cases),
        "results": results,
        "best": best,
        "threshold_advice": {
            "current_default": 0.35,
            "safe_upper_bound": best["min_score"] if best and best["min_score"] is not None else None,
            "note": "safe_upper_bound 为最优配置下所有命中分数的最小值；默认阈值高于该值会开始丢正确结果，低于该值则安全。阈值应结合业务容错在 min_score 与 avg_score 之间选取。",
        },
    }

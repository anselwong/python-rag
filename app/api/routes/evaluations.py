"""按知识库隔离的检索评测接口（Day 12）。"""

import json
import time
import uuid
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter

from app.core.database import Document, EvaluationCase, get_session
from app.services.evaluation import recall_at_k, reciprocal_rank
from app.services.qa import retrieve_contexts

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
        hits = retrieve_contexts(knowledge_base_id, item["question"], top_k=5, threshold=-1.0, mode="hybrid")
        elapsed = round((time.perf_counter() - started) * 1000, 2)
        expected = expected_docs[item["id"]]
        hit_ids = [hit["id"] for hit in hits]
        matching = [hit for hit in hits if hit["id"].startswith(expected + "-")]
        first_id = matching[0]["id"] if matching else ""
        results.append({**item, "status": "passed" if matching else "pending", "retrieval_score": matching[0]["score"] if matching else None, "recall_at_k": recall_at_k(hit_ids, first_id, 5) if matching else 0.0, "mrr": reciprocal_rank(hit_ids, first_id) if matching else 0.0, "latency_ms": elapsed})
    return results


@router.get("", summary="List knowledge-base evaluation cases")
def list_evaluations(knowledge_base_id: str) -> List[dict]:
    return _ensure_cases(knowledge_base_id)


@router.post("/generate", summary="Generate evaluation cases from documents")
def generate_evaluations(knowledge_base_id: str) -> List[dict]:
    with get_session() as session:
        session.query(EvaluationCase).filter(EvaluationCase.knowledge_base_id == knowledge_base_id).delete(synchronize_session=False)
    return _generate_cases(knowledge_base_id)


@router.post("/run", summary="Run knowledge-base retrieval evaluation")
def run_evaluations(knowledge_base_id: str) -> List[dict]:
    return _run(knowledge_base_id)

"""向量回填：为迁移到 LangChain 主链路之前用旧引擎灌入的文档重建 PGVector 向量。

背景：Day 13 评测扫描发现 Recall@K 全零——评测题集的金标准文档在旧引擎时期
写入的是 chunks 表，LangChain 的 PGVector 集合里没有对应向量，检索永远命中不到。
本脚本从 uploads 原始文件重跑解析/切片/向量化，并补齐 LangChainChunk 元数据表。

用法：python scripts/backfill_vectors.py <knowledge_base_id>
幂等：按集合中已有的 cmetadata.document_id 跳过已回填文档，可重复执行。
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.core.database import Document as DbDocument, LangChainChunk, get_session  # noqa: E402
from app.langchain.embeddings import BailianEmbeddings  # noqa: E402
from app.langchain.vectorstore import add_documents, get_store  # noqa: E402
from app.langchain.tokenizer import count_tokens  # noqa: E402
from app.services.document_processing import build_chunk_documents, element_dicts, parse_document  # noqa: E402


def _existing_document_ids(kb_id: str, embeddings: BailianEmbeddings) -> set:
    """查询 PGVector 集合中已有向量的 document_id 集合（幂等跳过依据）。"""
    store = get_store(kb_id, embeddings)
    with store._make_sync_session() as session:
        collection = store.get_collection(session)
        if collection is None:
            return set()
        rows = session.execute(
            text("SELECT DISTINCT cmetadata->>'document_id' FROM langchain_pg_embedding WHERE collection_id = :cid"),
            {"cid": collection.uuid},
        ).all()
    return {row[0] for row in rows if row[0]}


def backfill(kb_id: str) -> None:
    embeddings = BailianEmbeddings()
    done = _existing_document_ids(kb_id, embeddings)
    print(f"集合中已有向量的文档数: {len(done)}")
    with get_session() as session:
        documents = session.query(DbDocument).filter(DbDocument.knowledge_base_id == kb_id).all()
    for doc in documents:
        if doc.status != "ready":
            print(f"跳过（尚未发布）: {doc.name}")
            continue
        if doc.id in done:
            print(f"跳过（已有向量）: {doc.name}")
            continue
        path = settings.data_dir / "uploads" / doc.stored_name
        if not path.exists():
            print(f"警告：原始文件缺失，跳过 {doc.name} -> {path}")
            continue
        parsed = parse_document(path, Path(doc.stored_name).suffix.lower())
        chunks = build_chunk_documents(parsed.elements, doc.name)
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({"knowledge_base_id": kb_id, "document_id": doc.id, "chunk_id": f"{doc.id}-{i}", "document_name": doc.name, "parser_name": parsed.parser_name})
        add_documents(kb_id, chunks, embeddings)
        # 审计字段与主上传链路同步，确保历史文档重建后也能查看解析器和结构元素。
        timestamp = datetime.now(timezone.utc)
        with get_session() as session:
            current = session.get(DbDocument, doc.id)
            if current is not None:
                import json
                current.pages_json = json.dumps(parsed.pages, ensure_ascii=False)
                current.elements_json = json.dumps(element_dicts(parsed.elements), ensure_ascii=False)
                current.parser_name = parsed.parser_name
                current.parser_version = parsed.parser_version
                current.quality_json = json.dumps(parsed.quality, ensure_ascii=False)
                current.chunk_count = len(chunks)
            existing = {row.id for row in session.query(LangChainChunk).filter(LangChainChunk.document_id == doc.id).all()}
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc.id}-{i}"
                if chunk_id in existing:
                    continue
                import json
                session.add(LangChainChunk(id=chunk_id, document_id=doc.id, knowledge_base_id=kb_id, page=int(chunk.metadata.get("page", 1)), content=chunk.page_content, token_count=count_tokens(chunk.page_content), metadata_json=json.dumps(chunk.metadata, ensure_ascii=False), created_at=timestamp))
        print(f"回填完成: {doc.name} -> {len(chunks)} 个切片")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python scripts/backfill_vectors.py <knowledge_base_id>")
        raise SystemExit(2)
    backfill(sys.argv[1])

"""LangChain RAG 服务：Document、Embeddings、Retriever 和 LCEL 链的业务编排。"""
import json, math, uuid
from datetime import datetime, timezone
from pathlib import Path
from fastapi import UploadFile
from sqlalchemy import select
from langchain_core.documents import Document
from .loaders import load_file
from .splitters import split_documents
from .embeddings import BailianEmbeddings
from .vectorstore import add_documents, similarity_search
from .chain import run as chain_run, stream as chain_stream
from app.core.config import settings
from app.core.database import Document as DbDocument, Chunk, LangChainChunk, KnowledgeBase, get_session

UPLOAD_DIR = settings.data_dir / "uploads"; UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
embeddings = BailianEmbeddings()

def _cos(a,b): return sum(x*y for x,y in zip(a,b))

async def ingest(knowledge_base_id: str, upload: UploadFile) -> dict:
    name=Path(upload.filename or "未命名").name; ext=Path(name).suffix.lower()
    if ext not in settings.allowed_extensions: raise ValueError("仅支持 PDF、DOCX、MD、TXT 文件")
    did=str(uuid.uuid4()); destination=UPLOAD_DIR/f"{did}{ext}"; size=0
    try:
        with destination.open("wb") as f:
            while data:=await upload.read(1024*1024): size+=len(data); f.write(data)
        pages=load_file(destination, ext); chunks=split_documents(pages)
        for i, doc in enumerate(chunks):
            doc.metadata.update({"knowledge_base_id": knowledge_base_id, "document_id": did, "chunk_id": f"{did}-{i}", "document_name": name})
        # 向量化与持久化由 LangChain VectorStore 负责，业务层不再计算余弦或拼接向量 SQL。
        add_documents(knowledge_base_id, chunks, embeddings)
        timestamp=datetime.now(timezone.utc)
        with get_session() as session:
            kb=session.get(KnowledgeBase, knowledge_base_id)
            if not kb: raise KeyError("知识库不存在")
            page_rows=[{"page":d.metadata.get("page",1),"text":d.page_content} for d in pages]
            session.add(DbDocument(id=did, knowledge_base_id=knowledge_base_id,name=name,stored_name=destination.name,file_type=ext[1:].upper(),size_bytes=size,chunk_count=len(chunks),status="ready",pages_json=json.dumps(page_rows,ensure_ascii=False),created_at=timestamp))
            # LangChainChunk 没有 ORM relationship，显式 flush 保证父文档先于子切片插入。
            session.flush()
            # 仅保存可审计的切片元数据；向量正文由 LangChain PGVector 管理。
            for i, doc in enumerate(chunks):
                session.add(LangChainChunk(id=f"{did}-{i}",document_id=did,knowledge_base_id=knowledge_base_id,page=int(doc.metadata.get("page",1)),content=doc.page_content,token_count=len(doc.page_content),created_at=timestamp))
                # 旧 chunks 仅作为共用文档管理/迁移数据，不参与 LangChain 检索。
                session.add(Chunk(id=f"{did}-{i}", document_id=did, page=int(doc.metadata.get("page",1)), content=doc.page_content, token_count=len(doc.page_content), embedding=embeddings.embed_query(doc.page_content)))
            kb.updated_at=timestamp
        return {"id":did,"knowledge_base_id":knowledge_base_id,"name":name,"type":ext[1:].upper(),"size":f"{max(1,round(size/1024))} KB","chunk_count":len(chunks),"status":"ready","created_at":timestamp}
    except Exception:
        destination.unlink(missing_ok=True); raise

def retrieve(kb_id:str, query:str, top_k:int=5, threshold:float=0.0, mode:str="vector") -> list[dict]:
    matches = similarity_search(kb_id, query, top_k, embeddings)
    return [{"id":doc.metadata.get("chunk_id", doc.id), "document_name":doc.metadata.get("document_name", doc.metadata.get("source", "未知文档")), "page":int(doc.metadata.get("page", 1)), "content":doc.page_content, "score":float(round(score, 6)), "rank":i, "keywords":[term for term in query.split() if term and term in doc.page_content][:5]} for i,(doc,score) in enumerate(matches,1) if score >= threshold]

def answer(kb_id, question, top_k=5, session_id=None):
    contexts=retrieve(kb_id,question,top_k); return chain_run(question,contexts),contexts

def answer_stream(kb_id, question, top_k=5):
    contexts=retrieve(kb_id,question,top_k); return chain_stream(question,contexts),contexts

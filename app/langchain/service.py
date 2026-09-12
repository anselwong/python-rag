"""LangChain RAG 服务：Document、Embeddings、Retriever 和 LCEL 链的业务编排。"""
import json, math, os, uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from fastapi import UploadFile
from sqlalchemy import select
from langchain_core.documents import Document
from .embeddings import BailianEmbeddings
from .vectorstore import add_documents, similarity_search
from .chain import run as chain_run, stream as chain_stream
from app.core.config import settings
from app.core.database import ChatSession, Document as DbDocument, Chunk, LangChainChunk, KnowledgeBase, get_session
from .history import SqlChatMessageHistory
from .prompts import SYSTEM_PROMPT
from .tokenizer import count_tokens, fit_contexts_to_budget
from app.services.document_processing import build_chunk_documents, element_dicts, parse_document

UPLOAD_DIR = settings.data_dir / "uploads"; UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
embeddings = BailianEmbeddings()


class ChatSessionNotFoundError(ValueError):
    """请求携带的会话不存在，或不属于当前知识库。"""


def _cos(a,b): return sum(x*y for x,y in zip(a,b))

async def ingest(knowledge_base_id: str, upload: UploadFile) -> dict:
    name=Path(upload.filename or "未命名").name; ext=Path(name).suffix.lower()
    if ext not in settings.allowed_extensions: raise ValueError("不支持该文件类型")
    did=str(uuid.uuid4()); destination=UPLOAD_DIR/f"{did}{ext}"; size=0
    try:
        with destination.open("wb") as f:
            while data:=await upload.read(1024*1024): size+=len(data); f.write(data)
        parsed = parse_document(destination, ext)
        chunks=build_chunk_documents(parsed.elements, name)
        for i, doc in enumerate(chunks):
            doc.metadata.update({"knowledge_base_id": knowledge_base_id, "document_id": did, "chunk_id": f"{did}-{i}", "document_name": name, "parser_name": parsed.parser_name})
        timestamp=datetime.now(timezone.utc)
        status = parsed.quality["recommended_status"]
        with get_session() as session:
            kb=session.get(KnowledgeBase, knowledge_base_id)
            if not kb: raise KeyError("知识库不存在")
            session.add(DbDocument(id=did, knowledge_base_id=knowledge_base_id,name=name,stored_name=destination.name,file_type=ext[1:].upper(),size_bytes=size,chunk_count=len(chunks),status=status,pages_json=json.dumps(parsed.pages,ensure_ascii=False),elements_json=json.dumps(element_dicts(parsed.elements),ensure_ascii=False),parser_name=parsed.parser_name,parser_version=parsed.parser_version,quality_json=json.dumps(parsed.quality,ensure_ascii=False),created_at=timestamp))
            # LangChainChunk 没有 ORM relationship，显式 flush 保证父文档先于子切片插入。
            session.flush()
            # 仅保存可审计的切片元数据；向量正文由 LangChain PGVector 管理。
            # token_count 使用本地 tokenizer，不再把 Python 字符数错误标成 Token。
            for i, doc in enumerate(chunks):
                token_count = count_tokens(doc.page_content)
                metadata_json = json.dumps(doc.metadata, ensure_ascii=False)
                session.add(LangChainChunk(id=f"{did}-{i}",document_id=did,knowledge_base_id=knowledge_base_id,page=int(doc.metadata.get("page",1)),content=doc.page_content,token_count=token_count,metadata_json=metadata_json,created_at=timestamp))
                # 旧 chunks 仅作为共用文档管理/迁移数据，不参与 LangChain 检索。
                session.add(Chunk(id=f"{did}-{i}", document_id=did, page=int(doc.metadata.get("page",1)), content=doc.page_content, token_count=token_count, metadata_json=metadata_json, embedding=embeddings.embed_query(doc.page_content) if status == "ready" else None))
            kb.updated_at=timestamp
        # 审核前仍保存解析产物供预览，但绝不能将低质量文件写入向量库污染召回。
        if status == "ready" and chunks:
            add_documents(knowledge_base_id, chunks, embeddings)
        return {"id":did,"knowledge_base_id":knowledge_base_id,"name":name,"type":ext[1:].upper(),"size":f"{max(1,round(size/1024))} KB","chunk_count":len(chunks),"status":status,"parser_name":parsed.parser_name,"parser_version":parsed.parser_version,"quality":parsed.quality,"review_note":"","created_at":timestamp}
    except Exception:
        destination.unlink(missing_ok=True); raise


def review_document(knowledge_base_id: str, document_id: str, action: str, note: str = "") -> dict:
    """人工审核低质量文档；通过后才把已保存的 Chunk 发布到向量库。"""
    with get_session() as session:
        item = session.query(DbDocument).filter(DbDocument.id == document_id, DbDocument.knowledge_base_id == knowledge_base_id).one_or_none()
        if item is None:
            raise KeyError("文档不存在")
        if item.status not in {"needs_review", "rejected"}:
            raise ValueError("只有待审核或已拒绝的文档可以进行人工审核")
        if action == "reject":
            item.status = "rejected"
            item.review_note = note
            return _document_result(item)
        if action != "approve":
            raise ValueError("审核动作必须为 approve 或 reject")
        chunk_rows = session.query(LangChainChunk).filter(LangChainChunk.document_id == document_id).order_by(LangChainChunk.id).all()
        vector_documents = [Document(page_content=row.content, metadata=json.loads(row.metadata_json)) for row in chunk_rows]
        item.status = "ready"
        item.review_note = note
        result = _document_result(item)
    # 向量写入是外部存储操作，事务提交后执行；失败时状态回滚为待审核，避免假发布。
    try:
        if vector_documents:
            add_documents(knowledge_base_id, vector_documents, embeddings)
    except Exception:
        with get_session() as session:
            item = session.get(DbDocument, document_id)
            if item is not None:
                item.status = "needs_review"
                item.review_note = "发布向量失败，请重试审核。"
        raise
    return result


def _document_result(item: DbDocument) -> dict:
    return {"id": item.id, "knowledge_base_id": item.knowledge_base_id, "name": item.name, "type": item.file_type, "size": f"{max(1, round(item.size_bytes / 1024))} KB", "chunk_count": item.chunk_count, "status": item.status, "parser_name": item.parser_name, "parser_version": item.parser_version, "quality": json.loads(item.quality_json), "review_note": item.review_note, "created_at": item.created_at}

def retrieve(kb_id:str, query:str, top_k:int=5, threshold:float=0.0, mode:str="vector", vector_weight:float=None, keyword_weight:float=None) -> list[dict]:
    """统一检索入口。mode=hybrid 时在向量召回上叠加 BM25 关键词重排（Day 13）。

    hybrid 流程：先按 top_k*3 过采样向量候选池，再用手写 _bm25_score 对候选
    重打分并按 vector_weight/keyword_weight 加权融合，最后截断回 top_k。
    权重缺省读环境变量 HYBRID_VECTOR_WEIGHT / HYBRID_KEYWORD_WEIGHT，
    评测扫描接口可逐组显式传参覆盖，用于比较不同权重对召回质量的影响。
    """
    vw = float(vector_weight if vector_weight is not None else os.getenv("HYBRID_VECTOR_WEIGHT", "0.7"))
    kw = float(keyword_weight if keyword_weight is not None else os.getenv("HYBRID_KEYWORD_WEIGHT", "0.3"))
    fetch_k = top_k * 3 if mode == "hybrid" else top_k
    matches = similarity_search(kb_id, query, fetch_k, embeddings)
    if mode == "hybrid" and matches:
        matches = _hybrid_rerank(kb_id, query, matches, vw, kw)[:top_k]
    else:
        # vector 模式也必须统一分数方向：PGVector 返回余弦距离（越小越好），
        # 不转换的话下方 threshold 过滤方向相反，会把最优匹配整体丢掉。
        matches = _to_similarity(kb_id, matches)
    return [{"id":doc.metadata.get("chunk_id", doc.id), "document_name":doc.metadata.get("document_name", doc.metadata.get("source", "未知文档")), "page":int(doc.metadata.get("page", 1)), "content":doc.page_content, "score":float(round(score, 6)), "rank":i, "keywords":[term for term in query.split() if term and term in doc.page_content][:5]} for i,(doc,score) in enumerate(matches,1) if score >= threshold]


def _to_similarity(kb_id: str, matches: list) -> list:
    """把检索分数统一为"越大越好"的相似度语义。

    InMemoryVectorStore 返回余弦相似度（越大越相似），PGVector 的
    similarity_search_with_score 返回余弦距离（越小越相似）。两套存储、
    vector/hybrid 两种模式共用同一个 threshold 过滤，必须先归一方向。
    """
    from langchain_core.vectorstores import InMemoryVectorStore
    from app.langchain.vectorstore import get_store
    if isinstance(get_store(kb_id, embeddings), InMemoryVectorStore):
        return matches
    return [(doc, 1.0 - score) for doc, score in matches]


def _hybrid_rerank(kb_id:str, query:str, matches:list, vector_weight:float, keyword_weight:float) -> list:
    """把向量分数与 BM25 分数加权融合并重排（复用手写引擎的 BM25 实现）。

    分数方向统一由 _to_similarity 完成（PGVector 距离 -> 相似度），
    否则 PGVector 路径的 hybrid 排序会整体反转——这是评测扫描前必须先修掉的语义坑。
    """
    normalized = _to_similarity(kb_id, matches)
    corpus_tokens = [_tokenize(doc.page_content) for doc, _ in normalized]
    document_frequency = {}
    for tokens in corpus_tokens:
        for term in set(tokens):
            document_frequency[term] = document_frequency.get(term, 0) + 1
    average_length = sum(len(tokens) for tokens in corpus_tokens) / len(corpus_tokens) or 1.0
    query_terms = _tokenize(query)
    rescored = []
    for index, (doc, similarity) in enumerate(normalized):
        bm25 = _bm25_score(query_terms, corpus_tokens[index], document_frequency, len(corpus_tokens), average_length)
        rescored.append((doc, similarity * vector_weight + bm25 * keyword_weight))
    rescored.sort(key=lambda pair: pair[1], reverse=True)
    return rescored


# BM25 分词与打分直接复用手写引擎的实现：两套引擎共享同一套关键词统计逻辑，
# 评测对比时差异只来自融合权重，不引入第二份 BM25 代码的维护负担。
from app.services.retrieval import _bm25_score, _tokenize  # noqa: E402
from app.services.rerank import RerankError, rerank  # noqa: E402


def retrieve_for_answer(kb_id: str, question: str, top_k: int = 5, mode: str = "hybrid") -> list[dict]:
    """问答链路统一检索：扩大候选池 -> Rerank 精排 -> 失败回退原排序。

    与手写引擎 app/services/qa.py 策略对齐：候选池至少 10 条交给精排挑选；
    Rerank 未配置时 rerank() 原样截断返回，远端调用失败（RerankError）时
    回退向量/hybrid 排序，保证主问答链路不因精排服务抖动而中断。
    """
    contexts = retrieve(kb_id, question, top_k=max(top_k, 10), mode=mode)
    try:
        return rerank(question, contexts, top_n=top_k)
    except RerankError:
        return contexts[:top_k]


def ensure_chat_session(knowledge_base_id: str, question: str, session_id: Optional[str] = None) -> str:
    """创建或校验会话，再交由 RunnableWithMessageHistory 管理具体消息。

    会话归属校验仍是业务层责任：历史适配器只知道 session_id，不应允许调用方
    通过一个跨知识库 ID 读取其他知识库的上下文。
    """
    with get_session() as session:
        if session_id:
            item = session.query(ChatSession).filter(
                ChatSession.id == session_id,
                ChatSession.knowledge_base_id == knowledge_base_id,
            ).one_or_none()
            if item is None:
                raise ChatSessionNotFoundError("会话不存在")
            return item.id
        timestamp = datetime.now(timezone.utc)
        item = ChatSession(
            id=str(uuid.uuid4()),
            knowledge_base_id=knowledge_base_id,
            title=question[:40],
            created_at=timestamp,
            updated_at=timestamp,
        )
        session.add(item)
        return item.id


def answer(kb_id: str, question: str, top_k: int = 5, session_id: Optional[str] = None) -> tuple[str, list[dict], str, str, Optional[dict]]:
    """普通问答：Runnable 自动读写消息，服务层仅回填本轮检索引用。"""
    resolved_session_id = ensure_chat_session(kb_id, question, session_id)
    history = SqlChatMessageHistory(resolved_session_id)
    contexts = retrieve_for_answer(kb_id, question, top_k)
    prompt_contexts = fit_contexts_to_budget(question, history.messages, contexts, SYSTEM_PROMPT)
    response, usage = chain_run(question, prompt_contexts, history)
    history.attach_response_metadata(contexts, usage)
    if not history.last_assistant_message_id:
        raise RuntimeError("LangChain 未保存助手消息")
    return response, contexts, resolved_session_id, history.last_assistant_message_id, usage


def answer_stream(kb_id: str, question: str, top_k: int = 5, session_id: Optional[str] = None):
    """流式问答：仅在上游流完整结束后由 Runnable 持久化完整助手消息。"""
    resolved_session_id = ensure_chat_session(kb_id, question, session_id)
    history = SqlChatMessageHistory(resolved_session_id)
    contexts = retrieve_for_answer(kb_id, question, top_k)
    prompt_contexts = fit_contexts_to_budget(question, history.messages, contexts, SYSTEM_PROMPT)
    stream, stream_usage = chain_stream(question, prompt_contexts, history)

    def complete_stream():
        # 只有消费者正常读完流时 Runnable 才会写入完整回答；异常或断流不会留下
        # 半截助手消息，引用也只会绑定到已经成功写入的那一条回答。
        for chunk in stream:
            yield chunk
        history.attach_response_metadata(contexts, stream_usage.value)

    return complete_stream(), contexts, resolved_session_id, stream_usage

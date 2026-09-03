"""LCEL 问答链：Prompt -> ChatModel，普通与流式调用共享同一 Runnable。"""
import os
from typing import Iterable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from .prompts import RAG_PROMPT

def build_chain():
    model = ChatOpenAI(model=os.getenv("LLM_MODEL", "deepseek-chat"), api_key=os.getenv("LLM_API_KEY"), base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"), temperature=0.2, streaming=True)
    # LCEL 的 | 操作符将提示词、模型和解析器组合成可 invoke/stream 的 Runnable。
    return RAG_PROMPT | model | StrOutputParser()

def run(question: str, contexts: list[dict], history: str = "") -> str:
    context = "\n\n".join(f"[{i}] {x['document_name']} 第{x['page']}页\n{x['content']}" for i, x in enumerate(contexts, 1)) or "（没有检索到足够资料）"
    return build_chain().invoke({"question": question, "context": context, "history": history})

def stream(question: str, contexts: list[dict], history: str = "") -> Iterable[str]:
    context = "\n\n".join(f"[{i}] {x['document_name']} 第{x['page']}页\n{x['content']}" for i, x in enumerate(contexts, 1)) or "（没有检索到足够资料）"
    yield from build_chain().stream({"question": question, "context": context, "history": history})

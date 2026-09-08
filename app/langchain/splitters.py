"""LangChain 文档切分器，按本地 Token 预算而不是字符数控制片段大小。"""
import os
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .tokenizer import count_tokens


def split_documents(documents: list[Document], chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None) -> list[Document]:
    """优先保留段落/句子边界，并按 Token 而非 Python ``len()`` 切分。

    ``length_function`` 是 LangChain splitter 的长度度量入口。接入 tokenizer 后，
    ``chunk_size=800`` 的含义终于是 800 Token，不再是 800 个字符。
    """
    token_size = chunk_size if chunk_size is not None else int(os.getenv("CHUNK_SIZE_TOKENS", "800"))
    token_overlap = chunk_overlap if chunk_overlap is not None else int(os.getenv("CHUNK_OVERLAP_TOKENS", "120"))
    return RecursiveCharacterTextSplitter(
        chunk_size=token_size,
        chunk_overlap=token_overlap,
        length_function=count_tokens,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    ).split_documents(documents)

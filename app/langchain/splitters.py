"""LangChain 文档切分器。优先使用官方 splitter，不可用时提供等价递归实现。"""
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents: list[Document], chunk_size: int = 800, chunk_overlap: int = 120) -> list[Document]:
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", "。", "！", "？", " ", ""]).split_documents(documents)

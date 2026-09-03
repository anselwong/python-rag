"""文档加载器：统一产出 LangChain Document，并保留页码 metadata。"""
import json
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader, Docx2txtLoader

def load_file(path: Path, extension: str) -> list[Document]:
    loader = PyMuPDFLoader(str(path)) if extension == ".pdf" else Docx2txtLoader(str(path)) if extension == ".docx" else TextLoader(str(path), encoding="utf-8")
    documents = loader.load()
    for index, document in enumerate(documents, 1):
        # 不同 Loader 的页码 metadata 命名不一致，统一成前端契约使用的 1-based page。
        document.metadata.update({"page": int(document.metadata.get("page", index)) + (1 if extension == ".pdf" else 0), "source": path.name})
    return documents

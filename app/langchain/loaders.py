"""兼容加载器：复用项目结构化解析层，杜绝维护第二套 PyMuPDF/纯文本路径。"""
from pathlib import Path

from langchain_core.documents import Document

from app.services.document_processing import build_chunk_documents, parse_document


def load_file(path: Path, extension: str) -> list[Document]:
    """返回可供历史脚本使用的结构化 Chunk 文档。

    新上传链路直接调用 ``document_processing``；保留本函数仅为旧脚本兼容，避免
    任一维护入口重新引入 AGPL 的 PyMuPDF 或丢掉表格字段关系。
    """
    parsed = parse_document(path, extension)
    return build_chunk_documents(parsed.elements, path.name)

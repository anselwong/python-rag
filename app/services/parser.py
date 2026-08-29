"""支持 Day 4 文件格式的文本解析器。"""

import re
from pathlib import Path
from typing import List, Tuple


def parse_document(path: Path, extension: str) -> List[Tuple[int, str]]:
    """解析文件为 (页码, 文本) 列表，并尽量保留页码这一引用元数据。

    PDF 使用真实页码；DOCX、Markdown 和 TXT 没有页概念，因此统一返回第 1 页，
    后续切片时仍可通过 document_id 和 page 追溯原文。
    """
    if extension == ".pdf":
        return _parse_pdf(path)
    if extension == ".docx":
        from docx import Document

        paragraphs = [paragraph.text.strip() for paragraph in Document(path).paragraphs]
        return [(1, _clean_text("\n".join(paragraphs)))]
    if extension in {".md", ".txt"}:
        return [(1, _clean_text(path.read_text(encoding="utf-8", errors="replace")))]
    raise ValueError(f"不支持的文件类型: {extension}")


def _parse_pdf(path: Path) -> List[Tuple[int, str]]:
    import fitz

    with fitz.open(path) as document:
        return [(index + 1, _clean_text(page.get_text("text"))) for index, page in enumerate(document)]


def _clean_text(text: str) -> str:
    """统一换行和空白，避免 PDF 排版产生的碎片影响后续切片。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


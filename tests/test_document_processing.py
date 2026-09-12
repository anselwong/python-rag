"""结构化解析、质量门控和人工发布的集成测试。"""

from pathlib import Path

from pypdf import PdfWriter

from app.core import database
from app.core.database import LangChainChunk
from test_health import make_client


def test_markdown_table_becomes_header_aware_row_chunks(tmp_path: Path) -> None:
    """表格行必须保留表头字段，不能退化成没有对应关系的一串单元格文字。"""
    client = make_client(tmp_path)
    knowledge_base_id = client.post("/api/v1/knowledge-bases", json={"name": "制度库"}).json()["id"]
    content = """# 员工管理制度

## 试用期与转正

| 岗位 | 试用期 | 转正条件 |
| --- | --- | --- |
| 开发工程师 | 3个月 | 完成项目考核 |
| 销售经理 | 6个月 | 完成销售目标 |
"""
    response = client.post(
        f"/api/v1/knowledge-bases/{knowledge_base_id}/documents",
        files={"file": ("employee.md", content, "text/markdown")},
    )

    assert response.status_code == 201
    document = response.json()
    assert document["status"] == "ready"
    assert document["quality"]["table_row_count"] == 2
    detail = client.get(f"/api/v1/knowledge-bases/{knowledge_base_id}/documents/{document['id']}").json()
    table_chunks = [chunk for chunk in detail["chunks"] if chunk["metadata"]["content_type"] == "table_row"]
    assert len(table_chunks) == 2
    assert "岗位：销售经理" in table_chunks[1]["content"]
    assert "试用期：6个月" in table_chunks[1]["content"]
    assert "转正条件：完成销售目标" in table_chunks[1]["content"]
    assert table_chunks[1]["metadata"]["section_path"] == "员工管理制度 > 试用期与转正"


def test_low_text_pdf_requires_review_before_vector_publish(tmp_path: Path) -> None:
    """扫描件/空白 PDF 不得自动入向量库；审核发布才允许进入 ready。"""
    client = make_client(tmp_path)
    knowledge_base_id = client.post("/api/v1/knowledge-bases", json={"name": "审核库"}).json()["id"]
    pdf_path = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with pdf_path.open("wb") as file:
        writer.write(file)

    response = client.post(
        f"/api/v1/knowledge-bases/{knowledge_base_id}/documents",
        files={"file": ("scan.pdf", pdf_path.read_bytes(), "application/pdf")},
    )
    assert response.status_code == 201
    document = response.json()
    assert document["status"] == "needs_review"
    assert "PDF 文本过少" in "；".join(document["quality"]["review_reasons"])
    with database.get_session() as session:
        assert session.query(LangChainChunk).filter(LangChainChunk.document_id == document["id"]).count() == 0

    approved = client.post(
        f"/api/v1/knowledge-bases/{knowledge_base_id}/documents/{document['id']}/review",
        json={"action": "approve", "note": "已人工确认可发布"},
    )
    assert approved.status_code == 200
    assert approved.json()["status"] == "ready"
    assert approved.json()["review_note"] == "已人工确认可发布"

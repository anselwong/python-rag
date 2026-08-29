from fastapi.testclient import TestClient
from pathlib import Path
import tempfile

from app.main import app
from app.core import database
from app.services import knowledge



def make_client(tmp_path: Path) -> TestClient:
    database.configure_database(f"sqlite:///{tmp_path / 'test.sqlite3'}")
    knowledge.UPLOAD_DIR = tmp_path / "uploads"
    knowledge.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    database.initialize_database()
    return TestClient(app)


def test_health_check() -> None:
    with tempfile.TemporaryDirectory() as directory:
        response = make_client(Path(directory)).get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "python-rag"
    assert body["version"] == "0.1.0"
    assert body["timestamp"]


def test_openapi_document_is_available() -> None:
    with tempfile.TemporaryDirectory() as directory:
        response = make_client(Path(directory)).get("/openapi.json")

    assert response.status_code == 200
    assert "/api/v1/health" in response.json()["paths"]


def test_create_upload_parse_and_delete_txt(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    knowledge_base = client.post("/api/v1/knowledge-bases", json={"name": "测试库", "description": "Day 4"}).json()
    knowledge_base_id = knowledge_base["id"]

    upload = client.post(
        f"/api/v1/knowledge-bases/{knowledge_base_id}/documents",
        files={"file": ("guide.txt", "第一段\n\n第二段", "text/plain")},
    )
    assert upload.status_code == 201
    document = upload.json()
    assert document["type"] == "TXT"
    assert document["status"] == "ready"

    detail = client.get(f"/api/v1/knowledge-bases/{knowledge_base_id}/documents/{document['id']}")
    assert detail.status_code == 200
    assert detail.json()["pages"][0]["text"] == "第一段\n\n第二段"

    assert client.delete(f"/api/v1/knowledge-bases/{knowledge_base_id}/documents/{document['id']}").status_code == 204
    assert client.get(f"/api/v1/knowledge-bases/{knowledge_base_id}/documents/{document['id']}").status_code == 404


def test_upload_rejects_unsupported_extension(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    knowledge_base_id = client.post("/api/v1/knowledge-bases", json={"name": "测试库"}).json()["id"]
    response = client.post(
        f"/api/v1/knowledge-bases/{knowledge_base_id}/documents",
        files={"file": ("script.exe", b"not a document", "application/octet-stream")},
    )
    assert response.status_code == 400

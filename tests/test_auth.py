"""单角色 JWT 登录和知识库私有化测试。"""

from fastapi.testclient import TestClient

from app.core import database
from test_health import make_client


def test_admin_is_seeded_and_can_login(tmp_path) -> None:
    client = make_client(tmp_path)
    response = client.post("/api/v1/auth/login", json={"username": "admin", "password": "admin"})
    assert response.status_code == 200
    assert response.json()["user"]["username"] == "admin"
    assert response.json()["token_type"] == "bearer"


def test_protected_endpoint_requires_token(tmp_path) -> None:
    client = make_client(tmp_path)
    client.headers.pop("Authorization")
    assert client.get("/api/v1/knowledge-bases").status_code == 401
    assert client.get("/api/v1/health").status_code == 200


def test_registered_users_only_see_their_own_knowledge_bases(tmp_path) -> None:
    client = make_client(tmp_path)
    registration = client.post("/api/v1/auth/register", json={"username": "alice", "password": "alice123"})
    assert registration.status_code == 201
    client.headers.update({"Authorization": f"Bearer {registration.json()['access_token']}"})
    own = client.post("/api/v1/knowledge-bases", json={"name": "Alice 的知识库"}).json()

    client.headers.update({"Authorization": f"Bearer {client.post('/api/v1/auth/login', json={'username': 'admin', 'password': 'admin'}).json()['access_token']}"})
    admin_kbs = client.get("/api/v1/knowledge-bases").json()
    assert own["id"] not in [item["id"] for item in admin_kbs]

    client.headers.update({"Authorization": f"Bearer {registration.json()['access_token']}"})
    assert client.get(f"/api/v1/{'knowledge-bases'}/{own['id']}/documents").status_code == 200
    assert client.get(f"/api/v1/knowledge-bases/not-owned/documents").status_code == 404

from app.services.evaluation import mean, recall_at_k, reciprocal_rank
from test_health import make_client


def test_retrieval_metrics() -> None:
    ids = ["chunk-2", "chunk-1", "chunk-3"]
    assert recall_at_k(ids, "chunk-1", 2) == 1.0
    assert reciprocal_rank(ids, "chunk-1") == 0.5
    assert mean([1.0, 0.5]) == 0.75


def test_evaluation_cases_are_generated_per_knowledge_base(tmp_path) -> None:
    client = make_client(tmp_path)
    first = client.post("/api/v1/knowledge-bases", json={"name": "产品库"}).json()["id"]
    second = client.post("/api/v1/knowledge-bases", json={"name": "研发库"}).json()["id"]
    client.post(f"/api/v1/knowledge-bases/{first}/documents", files={"file": ("product.txt", "产品角色权限说明", "text/plain")})
    client.post(f"/api/v1/knowledge-bases/{second}/documents", files={"file": ("dev.txt", "部署环境变量说明", "text/plain")})

    first_cases = client.get(f"/api/v1/knowledge-bases/{first}/evaluations").json()
    second_cases = client.get(f"/api/v1/knowledge-bases/{second}/evaluations").json()
    assert first_cases and second_cases
    assert "product.txt" in first_cases[0]["expected_source"]
    assert "dev.txt" in second_cases[0]["expected_source"]
    assert first_cases[0]["id"] != second_cases[0]["id"]

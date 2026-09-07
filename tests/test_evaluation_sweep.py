"""Day 13 评测深化测试：hybrid 融合、权重扫描与阈值校准接口。"""
from test_health import make_client


def _prepare_kb(client) -> str:
    """上传两个可区分文档，生成金标准评测题，返回知识库 id。"""
    kb_id = client.post("/api/v1/knowledge-bases", json={"name": "评测库"}).json()["id"]
    client.post(f"/api/v1/knowledge-bases/{kb_id}/documents", files={"file": ("product.txt", "产品角色权限说明，管理员可以分配角色", "text/plain")})
    client.post(f"/api/v1/knowledge-bases/{kb_id}/documents", files={"file": ("dev.txt", "部署环境变量说明，数据库连接串配置", "text/plain")})
    client.post(f"/api/v1/knowledge-bases/{kb_id}/evaluations/generate")
    return kb_id


def test_hybrid_with_full_vector_weight_matches_vector_ranking(tmp_path) -> None:
    """vector_weight=1.0、keyword_weight=0.0 时 hybrid 排序必须与纯 vector 一致。

    这是权重参数化的确定性回归断言：融合退化为纯向量分数，任何排序差异
    都说明 BM25 分数泄漏进了融合结果。
    """
    client = make_client(tmp_path)
    kb_id = _prepare_kb(client)
    question = "角色权限怎么分配"
    vector_hits = client.post(
        f"/api/v1/knowledge-bases/{kb_id}/retrieval/search",
        json={"query": question, "top_k": 5, "score_threshold": -1.0, "mode": "vector"},
    ).json()
    hybrid_hits = client.post(
        f"/api/v1/knowledge-bases/{kb_id}/retrieval/search",
        json={"query": question, "top_k": 5, "score_threshold": -1.0, "mode": "hybrid",
              "vector_weight": 1.0, "keyword_weight": 0.0},
    ).json()
    assert [hit["id"] for hit in vector_hits] == [hit["id"] for hit in hybrid_hits]


def test_hybrid_returns_ranked_results_with_scores(tmp_path) -> None:
    """hybrid 模式返回结构完整：rank 连续、score 降序、字段与 vector 模式一致。"""
    client = make_client(tmp_path)
    kb_id = _prepare_kb(client)
    response = client.post(
        f"/api/v1/knowledge-bases/{kb_id}/retrieval/search",
        json={"query": "数据库连接串", "top_k": 3, "score_threshold": -1.0, "mode": "hybrid"},
    )
    assert response.status_code == 200
    hits = response.json()
    assert hits
    assert [hit["rank"] for hit in hits] == list(range(1, len(hits) + 1))
    scores = [hit["score"] for hit in hits]
    assert scores == sorted(scores, reverse=True)
    assert set(hits[0].keys()) >= {"id", "document_name", "page", "content", "score", "rank", "keywords"}


def test_sweep_compares_configs_and_advises_threshold(tmp_path) -> None:
    """扫描接口返回每组配置的 Recall@K/MRR/延迟，并给出阈值校准建议。"""
    client = make_client(tmp_path)
    kb_id = _prepare_kb(client)
    response = client.post(f"/api/v1/knowledge-bases/{kb_id}/evaluations/sweep", json={})
    assert response.status_code == 200
    body = response.json()
    assert body["case_count"] > 0
    assert len(body["results"]) == 6  # 内置默认扫描集
    for row in body["results"]:
        assert 0.0 <= row["recall_at_k"] <= 1.0
        assert 0.0 <= row["mrr"] <= 1.0
        assert row["avg_latency_ms"] >= 0
    # 结果按 Recall@K 降序，best 即第一行
    recalls = [row["recall_at_k"] for row in body["results"]]
    assert recalls == sorted(recalls, reverse=True)
    assert body["best"]["recall_at_k"] == recalls[0]
    assert body["threshold_advice"]["current_default"] == 0.35


def test_sweep_accepts_custom_configs(tmp_path) -> None:
    """显式传入配置时按传入集合运行，不使用内置默认扫描集。"""
    client = make_client(tmp_path)
    kb_id = _prepare_kb(client)
    configs = [{"mode": "hybrid", "vector_weight": 0.5, "keyword_weight": 0.5, "threshold": 0.0}]
    response = client.post(f"/api/v1/knowledge-bases/{kb_id}/evaluations/sweep", json={"configs": configs})
    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 1
    assert body["results"][0]["vector_weight"] == 0.5
    assert body["results"][0]["keyword_weight"] == 0.5


def test_threshold_filtering_drops_low_score_hits(tmp_path) -> None:
    """阈值过滤生效：threshold=1.0 时 hash 向量几乎不可能达到，命中数应为 0。"""
    client = make_client(tmp_path)
    kb_id = _prepare_kb(client)
    hits = client.post(
        f"/api/v1/knowledge-bases/{kb_id}/retrieval/search",
        json={"query": "角色权限", "top_k": 5, "score_threshold": 1.0, "mode": "hybrid"},
    ).json()
    assert hits == []

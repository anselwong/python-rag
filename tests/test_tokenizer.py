"""本地 tokenizer、Token 切片与 Prompt 预算的确定性测试。"""

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from app.langchain.splitters import split_documents
from app.langchain.tokenizer import count_tokens, fit_contexts_to_budget, truncate_to_tokens


def test_token_count_is_not_python_character_length() -> None:
    """中文、英文和符号的 Token 数由词表决定，不能用 len(text) 冒充。"""
    text = "Vue.use(ELEMENT) 同时注册 elementUI。"
    assert count_tokens(text) > 0
    assert count_tokens(text) != len(text)


def test_truncate_to_token_boundary() -> None:
    text = "知识库检索需要根据上下文和用户问题组织提示词。"
    truncated = truncate_to_tokens(text, 8)
    assert 0 < count_tokens(truncated) <= 8
    assert text.startswith(truncated)


def test_recursive_splitter_uses_token_length() -> None:
    """较长文本应按传入的 Token 尺寸拆分，而不是按字符数拆分。"""
    document = Document(page_content="第一段内容。" * 80, metadata={"page": 1})
    chunks = split_documents([document], chunk_size=30, chunk_overlap=5)
    assert len(chunks) > 1
    # 分隔符可能让边界略有差异，仍必须接近明确的 Token 上限。
    assert max(count_tokens(chunk.page_content) for chunk in chunks) <= 35


def test_context_budget_keeps_high_ranked_contexts(monkeypatch) -> None:
    """预算不足时按 Rerank 顺序保留高分片段，并截断最后一个可放入的片段。"""
    monkeypatch.setenv("LLM_CONTEXT_WINDOW", "80")
    monkeypatch.setenv("LLM_OUTPUT_RESERVE_TOKENS", "20")
    monkeypatch.setenv("MAX_CONTEXT_TOKENS", "40")
    monkeypatch.setenv("PROMPT_FORMAT_OVERHEAD_TOKENS", "5")
    contexts = [
        {"document_name": "高分.md", "page": 1, "content": "高分证据。" * 20, "score": 0.9},
        {"document_name": "低分.md", "page": 1, "content": "低分证据。" * 20, "score": 0.5},
    ]
    selected = fit_contexts_to_budget(
        "请总结制度",
        [HumanMessage(content="上一轮问题"), AIMessage(content="上一轮回答")],
        contexts,
        "只能依据资料回答。",
    )
    assert selected
    assert selected[0]["document_name"] == "高分.md"
    assert len(selected) < len(contexts) or selected[0]["content"] != contexts[0]["content"]

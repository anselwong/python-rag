"""chunker 单元测试：纯函数、确定性，不依赖数据库和网络。"""

from app.services.chunker import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    estimate_tokens,
    split_pages_into_chunks,
)


def test_estimate_tokens() -> None:
    assert estimate_tokens("") == 0
    # 中文按 1 字 1 token
    assert estimate_tokens("你好世界") == 4
    # 英文按 4 字符 1 token，向上取整
    assert estimate_tokens("abcdefgh") == 2
    assert estimate_tokens("abcde") == 2
    # 混合文本：中文逐字 + 英文按字符
    assert estimate_tokens("你好world") == 2 + 2


def test_short_text_produces_single_chunk() -> None:
    chunks = split_pages_into_chunks([(1, "第一段内容。\n\n第二段内容。")])
    assert len(chunks) == 1
    assert chunks[0]["page"] == 1
    assert "第一段内容" in chunks[0]["content"]
    assert "第二段内容" in chunks[0]["content"]
    assert chunks[0]["token_count"] == estimate_tokens(chunks[0]["content"])


def test_empty_pages_are_skipped() -> None:
    assert split_pages_into_chunks([(1, ""), (2, "   \n  ")]) == []


def test_long_text_produces_multiple_chunks_within_limit() -> None:
    paragraph = "这是一个用于测试切片边界的段落。" * 200
    chunks = split_pages_into_chunks([(1, paragraph)])
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk["token_count"] <= CHUNK_SIZE


def test_adjacent_chunks_overlap() -> None:
    paragraph = "这是用于验证重叠切片的句子，包含足够多的语义单元。" * 150
    chunks = split_pages_into_chunks([(1, paragraph)])
    assert len(chunks) > 1
    # 第二个 chunk 的开头（overlap 部分）必须完整出现在第一个 chunk 中，
    # 证明边界话题在相邻切片里都有完整上下文。
    first_unit_of_second = chunks[1]["content"].split("\n")[0]
    assert first_unit_of_second in chunks[0]["content"]


def test_pages_keep_page_metadata() -> None:
    chunks = split_pages_into_chunks([(1, "第一页内容。" * 80), (2, "第二页内容。" * 80)])
    pages = {chunk["page"] for chunk in chunks}
    assert pages == {1, 2}
    # 按页独立切片：第一页的文本不会混进第二页的 chunk。
    for chunk in chunks:
        if chunk["page"] == 1:
            assert "第二页" not in chunk["content"]


def test_unbroken_long_text_is_hard_split() -> None:
    # 无任何标点和空白的超长文本：句子降级失效，必须走字符硬切兜底。
    chunks = split_pages_into_chunks([(1, "字" * 3000)])
    assert len(chunks) > 1
    assert all(chunk["token_count"] <= CHUNK_SIZE for chunk in chunks)
    assert "".join(chunk["content"] for chunk in chunks) == "字" * 3000


def test_overlap_is_bounded() -> None:
    paragraph = "重叠长度验证段落。" * 300
    chunks = split_pages_into_chunks([(1, paragraph)])
    for previous, current in zip(chunks, chunks[1:]):
        overlap_units = set(current["content"].split("\n")) & set(previous["content"].split("\n"))
        overlap_tokens = sum(estimate_tokens(unit) for unit in overlap_units)
        # 重叠不应超过一个完整 chunk 的规模（正常约为 CHUNK_OVERLAP 量级）。
        assert overlap_tokens < CHUNK_SIZE

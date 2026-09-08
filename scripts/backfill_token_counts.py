"""为已有切片回填本地 tokenizer 计算的 token_count。

执行：.venv/bin/python scripts/backfill_token_counts.py
该脚本只更新 token_count，不重建 Embedding、不改正文，也不会调用外部模型。
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.database import Chunk, LangChainChunk, get_session  # noqa: E402
from app.langchain.tokenizer import count_tokens  # noqa: E402


def backfill() -> tuple[int, int]:
    """更新手写表和 LangChain 审计表，返回两类实际更新行数。"""
    chunk_updates = 0
    langchain_updates = 0
    with get_session() as session:
        for row in session.query(Chunk).all():
            token_count = count_tokens(row.content)
            if row.token_count != token_count:
                row.token_count = token_count
                chunk_updates += 1
        for row in session.query(LangChainChunk).all():
            token_count = count_tokens(row.content)
            if row.token_count != token_count:
                row.token_count = token_count
                langchain_updates += 1
    return chunk_updates, langchain_updates


if __name__ == "__main__":
    chunks, langchain_chunks = backfill()
    print(f"已更新 chunks: {chunks}，langchain_chunks: {langchain_chunks}")

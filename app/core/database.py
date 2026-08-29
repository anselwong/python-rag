"""SQLite 元数据存储。

文件内容单独存放在 data/uploads，SQLite 只保存知识库、文档和解析结果元数据。
这样删除文档或未来迁移向量库时，不会把二进制文件和检索数据混在同一层。
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from app.core.config import settings

DATABASE_PATH = settings.data_dir / "rag.sqlite3"


def initialize_database() -> None:
    """创建 Day 4 所需的数据表；IF NOT EXISTS 使启动过程可重复执行。"""
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS knowledge_bases (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                color TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                knowledge_base_id TEXT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                stored_name TEXT NOT NULL UNIQUE,
                file_type TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL,
                pages_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_documents_kb ON documents(knowledge_base_id);
            """
        )


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """提供带事务边界的连接；with 退出时自动提交，异常时自动回滚。"""
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


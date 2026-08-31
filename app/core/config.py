import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# 本地开发从 .env 读取密钥；生产环境通常由容器/部署平台注入环境变量。
# override=False 保证生产环境显式注入的变量优先级高于本地文件。
load_dotenv(override=False)


def _cors_origins() -> List[str]:
    raw_origins = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]


@dataclass(frozen=True)
class Settings:
    app_name: str = "python-rag"
    app_version: str = "0.1.0"
    app_env: str = os.getenv("APP_ENV", "development")
    api_prefix: str = "/api/v1"
    cors_origins: List[str] = None  # type: ignore[assignment]
    data_dir: Path = Path(os.getenv("RAG_DATA_DIR", "data"))
    max_upload_size: int = 20 * 1024 * 1024
    allowed_extensions: tuple = (".pdf", ".docx", ".md", ".txt")

    def __post_init__(self) -> None:
        object.__setattr__(self, "cors_origins", _cors_origins())


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)

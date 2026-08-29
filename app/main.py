from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import settings
from app.core.database import initialize_database


def create_app() -> FastAPI:
    application = FastAPI(
        title="RAG Knowledge Base API",
        version=settings.app_version,
        description="Backend API for document ingestion, retrieval, and RAG chat.",
        docs_url="/docs",
        redoc_url=None,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(api_router, prefix=settings.api_prefix)
    @application.on_event("startup")
    def startup() -> None:
        # 启动时初始化表结构，保证本地开发无需手动执行迁移脚本。
        initialize_database()
    return application


app = create_app()

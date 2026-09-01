from fastapi import APIRouter

from app.api.routes.health import router as health_router
from app.api.routes.knowledge import router as knowledge_router
from app.api.routes.retrieval import router as retrieval_router
from app.api.routes.chat import router as chat_router
from app.api.routes.evaluations import router as evaluations_router

api_router = APIRouter()
api_router.include_router(health_router, tags=["system"])
api_router.include_router(knowledge_router, tags=["knowledge-base"])
api_router.include_router(retrieval_router, tags=["retrieval"])
api_router.include_router(chat_router, tags=["chat"])
api_router.include_router(evaluations_router, tags=["evaluation"])

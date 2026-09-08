from fastapi import APIRouter

from app.api.routes.health import router as health_router
from app.api.routes.knowledge import router as knowledge_router
from app.api.routes.retrieval import router as retrieval_router
from app.api.routes.chat import router as chat_router
from app.api.routes.evaluations import router as evaluations_router
from app.api.routes.auth import router as auth_router
from fastapi import Depends
from app.api.dependencies import get_current_user

api_router = APIRouter()
api_router.include_router(health_router, tags=["system"])
api_router.include_router(auth_router, tags=["auth"])
api_router.include_router(knowledge_router, tags=["knowledge-base"], dependencies=[Depends(get_current_user)])
api_router.include_router(retrieval_router, tags=["retrieval"], dependencies=[Depends(get_current_user)])
api_router.include_router(chat_router, tags=["chat"], dependencies=[Depends(get_current_user)])
api_router.include_router(evaluations_router, tags=["evaluation"], dependencies=[Depends(get_current_user)])

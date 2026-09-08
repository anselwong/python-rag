"""FastAPI 认证和知识库私有化依赖。"""

from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.database import User, get_session
from app.core.security import decode_access_token

bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)) -> User:
    """从 Authorization Bearer JWT 读取当前用户。"""
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="请先登录", headers={"WWW-Authenticate": "Bearer"})
    try:
        subject = decode_access_token(credentials.credentials)["sub"]
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="登录已失效", headers={"WWW-Authenticate": "Bearer"}) from error
    with get_session() as session:
        user = session.get(User, subject)
        if user is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户不存在", headers={"WWW-Authenticate": "Bearer"})
        session.expunge(user)
        return user


def require_knowledge_base_owner(knowledge_base_id: str, current_user: User = Depends(get_current_user)) -> User:
    """校验当前用户拥有路径中的知识库，避免只凭 UUID 访问他人数据。"""
    from app.core.database import KnowledgeBase

    with get_session() as session:
        item = session.query(KnowledgeBase.id).filter(
            KnowledgeBase.id == knowledge_base_id,
            KnowledgeBase.owner_user_id == current_user.id,
        ).one_or_none()
    if item is None:
        raise HTTPException(status_code=404, detail="知识库不存在")
    return current_user

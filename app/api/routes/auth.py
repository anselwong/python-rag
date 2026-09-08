"""注册、登录和当前用户接口。"""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status
from fastapi import Depends

from app.core.database import User, get_session
from app.core.security import create_access_token, hash_password, verify_password
from app.api.dependencies import get_current_user
from app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse, UserResponse

router = APIRouter(prefix="/auth")


def _user_response(user: User) -> dict:
    return {"id": user.id, "username": user.username, "role": user.role, "created_at": user.created_at}


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED, summary="Register a user")
def register(payload: RegisterRequest) -> dict:
    username = payload.username.strip()
    with get_session() as session:
        if session.query(User).filter(User.username == username).one_or_none() is not None:
            raise HTTPException(status_code=409, detail="用户名已存在")
        user = User(id=str(uuid.uuid4()), username=username, password_hash=hash_password(payload.password), role="user", created_at=datetime.now(timezone.utc))
        session.add(user)
        session.flush()
        return {"access_token": create_access_token(user.id), "token_type": "bearer", "user": _user_response(user)}


@router.post("/login", response_model=TokenResponse, summary="Login")
def login(payload: LoginRequest) -> dict:
    with get_session() as session:
        user = session.query(User).filter(User.username == payload.username.strip()).one_or_none()
        if user is None or not verify_password(payload.password, user.password_hash):
            raise HTTPException(status_code=401, detail="用户名或密码错误", headers={"WWW-Authenticate": "Bearer"})
        return {"access_token": create_access_token(user.id), "token_type": "bearer", "user": _user_response(user)}


@router.get("/me", response_model=UserResponse, summary="Get current user")
def me(current_user: User = Depends(get_current_user)):
    return _user_response(current_user)

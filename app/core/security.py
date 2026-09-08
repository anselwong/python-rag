"""认证基础能力：PBKDF2 密码哈希和最小 JWT 实现。

当前项目只有一个角色，认证模块只负责确认用户身份；资源权限由知识库 owner
字段和路由层的归属校验负责。使用标准库可以让演示环境不依赖额外密码/JWT 包。
"""

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Any, Dict, Optional


PASSWORD_ITERATIONS = 310_000


def hash_password(password: str) -> str:
    """使用随机 salt 和 PBKDF2-SHA256 保存密码，绝不保存明文。"""
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PASSWORD_ITERATIONS)
    return f"pbkdf2_sha256${PASSWORD_ITERATIONS}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, encoded: str) -> bool:
    """校验密码并使用 constant-time compare，避免泄露哈希信息。"""
    try:
        algorithm, iterations, salt, expected = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), _unb64(salt), int(iterations))
        return hmac.compare_digest(_b64(digest), expected)
    except (TypeError, ValueError):
        return False


def create_access_token(subject: str, expires_in: Optional[int] = None) -> str:
    """生成带 subject/exp 的 HS256 JWT。"""
    now = int(time.time())
    lifetime = expires_in or int(os.getenv("JWT_EXPIRE_SECONDS", "86400"))
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {"sub": subject, "iat": now, "exp": now + lifetime}
    encoded_header = _b64(json.dumps(header, separators=(",", ":")).encode())
    encoded_payload = _b64(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{encoded_header}.{encoded_payload}".encode()
    signature = hmac.new(_secret(), signing_input, hashlib.sha256).digest()
    return f"{encoded_header}.{encoded_payload}.{_b64(signature)}"


def decode_access_token(token: str) -> Dict[str, Any]:
    """校验 JWT 签名和过期时间，失败时抛出 ValueError。"""
    try:
        encoded_header, encoded_payload, encoded_signature = token.split(".", 2)
        header = json.loads(_unb64(encoded_header))
        payload = json.loads(_unb64(encoded_payload))
        if header.get("alg") != "HS256":
            raise ValueError("不支持的 Token 算法")
        expected = hmac.new(_secret(), f"{encoded_header}.{encoded_payload}".encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(_b64(expected), encoded_signature):
            raise ValueError("Token 签名无效")
        if int(payload.get("exp", 0)) <= int(time.time()):
            raise ValueError("Token 已过期")
        if not payload.get("sub"):
            raise ValueError("Token 缺少用户标识")
        return payload
    except (TypeError, ValueError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("Token 无效") from error


def _secret() -> bytes:
    return os.getenv("JWT_SECRET", "development-only-change-this-secret").encode()


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()


def _unb64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))

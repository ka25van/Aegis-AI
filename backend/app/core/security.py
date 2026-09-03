import uuid
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

from jose import jwt
from passlib.context import CryptContext
from passlib.hash import argon2

from app.core.config import settings

# Use argon2 (more modern, no bcrypt compatibility issues)
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "type": "access", "jti": str(uuid.uuid4())})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def create_refresh_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh", "jti": str(uuid.uuid4())})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_token(token: str) -> dict:
    """Strict decode — raises on invalid/expired instead of swallowing (security.py:43 fix)."""
    payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    return payload


# In-memory fallback when Redis is down (single-process)
_REVOKED_FALLBACK: dict[str, float] = {}


async def is_token_revoked(jti: str, redis_client) -> bool:
    try:
        return bool(await redis_client.get(f"revoked_jti:{jti}"))
    except Exception:
        exp = _REVOKED_FALLBACK.get(jti)
        if exp and exp > datetime.utcnow().timestamp():
            return True
        return False


async def revoke_token(jti: str, exp_epoch: int, redis_client) -> None:
    try:
        ttl = max(1, exp_epoch - int(datetime.utcnow().timestamp()))
        await redis_client.setex(f"revoked_jti:{jti}", ttl, "1")
    except Exception:
        _REVOKED_FALLBACK[jti] = float(exp_epoch)
        # prune expired
        now = datetime.utcnow().timestamp()
        for k, v in list(_REVOKED_FALLBACK.items()):
            if v < now:
                _REVOKED_FALLBACK.pop(k, None)


def create_token_pair(user_id: UUID, email: str) -> dict:
    access_token = create_access_token({"sub": str(user_id), "email": email})
    refresh_token = create_refresh_token({"sub": str(user_id), "email": email})
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
    }
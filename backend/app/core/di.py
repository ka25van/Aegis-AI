from typing import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import async_session_maker, engine
from app.core.redis import get_redis, init_redis, close_redis
from app.core.config import settings
from redis.asyncio import Redis


class Container:
    """Simple DI container for explicit dependency management."""

    def __init__(self):
        self._db_session_maker = async_session_maker
        self._redis_client: Redis | None = None
        self._engine = engine

    async def initialize(self) -> None:
        """Initialize async resources — Redis is optional, fail open."""
        try:
            await init_redis()
            self._redis_client = get_redis()
        except Exception as e:
            # Redis unavailable — continue with in-memory fallbacks (security.revoked, idempotency)
            import logging
            logging.getLogger(__name__).warning("redis_init_failed_fallback_to_memory: %s", e)
            self._redis_client = None  # type: ignore

    def is_redis_available(self) -> bool:
        return self._redis_client is not None

    async def shutdown(self) -> None:
        """Cleanup async resources."""
        await close_redis()
        await self._engine.dispose()

    @asynccontextmanager
    async def db_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Provide a database session."""
        async with self._db_session_maker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    def get_redis(self) -> Redis:
        """Get Redis client."""
        if self._redis_client is None:
            raise RuntimeError("Container not initialized. Call initialize() first.")
        return self._redis_client

    @property
    def db_engine(self):
        """Get database engine."""
        return self._engine


# Global container instance
container = Container()


# FastAPI dependency providers
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database session."""
    async with container.db_session() as session:
        yield session


def get_redis_client() -> Redis:
    """FastAPI dependency for Redis client."""
    return container.get_redis()


def get_settings():
    """FastAPI dependency for settings."""
    return settings
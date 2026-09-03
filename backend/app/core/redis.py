import redis.asyncio as redis
from app.core.config import settings

redis_client: redis.Redis | None = None


async def init_redis() -> None:
    global redis_client
    redis_client = redis.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=True,
    )


async def close_redis() -> None:
    global redis_client
    if redis_client:
        await redis_client.close()
        redis_client = None


def get_redis() -> redis.Redis:
    if redis_client is None:
        raise RuntimeError("Redis not initialized. Call init_redis() first.")
    return redis_client


def is_redis_available() -> bool:
    return redis_client is not None
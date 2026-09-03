from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
import time

from app.core.config import settings
from app.core.logging import setup_logging, get_logger
from app.core.di import container
from app.api.v1.router import api_router

# Register MCP tool adapters (import triggers self-registration)
import app.mcp.github  # noqa: F401
import app.mcp.filesystem  # noqa: F401
import app.mcp.docker  # noqa: F401
import app.mcp.aws  # noqa: F401
import app.mcp.kubernetes  # noqa: F401
import app.mcp.prometheus  # noqa: F401


# Setup logging
setup_logging(settings.LOG_LEVEL)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("application_starting", env=getattr(settings, "ENV", "dev"))
    await container.initialize()
    if not container.is_redis_available():
        logger.warning("application_starting_without_redis_fallback_active")
    # Optional: Langfuse OTEL instrumentation (no-op if not configured)
    try:
        from app.services.tracing import get_langfuse

        get_langfuse()
    except Exception:
        pass
    yield
    logger.info("application_shutting_down_graceful")
    try:
        from app.services.tracing import flush as flush_langfuse

        flush_langfuse()
    except Exception:
        pass
    await container.shutdown()
    logger.info("application_shutting_down")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        routes=app.routes,
    )
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
    }
    for path in openapi_schema["paths"].values():
        for method in path.values():
            method.setdefault("security", [{"BearerAuth": []}])
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    if request.url.scheme == "https":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(
        "http_request",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=round(process_time * 1000, 2),
    )
    return response


app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/health")
async def health_check():
    """Liveness — always 200 if process is up."""
    return {"status": "ok", "version": settings.VERSION}


@app.get("/ready")
async def readiness_check():
    """Readiness — checks DB + Redis (K8s readinessProbe)."""
    checks: dict = {}
    ok = True
    # DB
    try:
        from app.db.session import engine
        from sqlalchemy import text
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        checks["db"] = "ok"
    except Exception as e:
        checks["db"] = f"fail: {e}"
        ok = False
    # Redis
    try:
        from app.core.redis import get_redis
        r = get_redis()
        await r.ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"fail: {e}"
        ok = False
    # Ollama (optional — not required for readiness, but report)
    try:
        import httpx
        async with httpx.AsyncClient(timeout=2) as c:
            resp = await c.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
            checks["ollama"] = "ok" if resp.status_code == 200 else f"fail:{resp.status_code}"
    except Exception as e:
        checks["ollama"] = f"unavailable: {e}"

    status_code = 200 if ok else 503
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=status_code, content={"status": "ready" if ok else "not_ready", "checks": checks})


@app.get("/live")
async def liveness_check():
    return {"status": "alive"}
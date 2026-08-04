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
    logger.info("application_starting")
    await container.initialize()
    yield
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
    return {"status": "ok"}
import os
from typing import List, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "Aegis AI"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"

    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]

    # Default: Docker Compose (localhost); override with env vars for K8s
    # kubectl creates aegis-secrets with DATABASE_URL = postgresql+asyncpg://postgres:<pw>@postgres:5432/aegis
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/aegis"
    REDIS_URL: str = "redis://localhost:6379/0"

    SECRET_KEY: str = "dev-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    LOG_LEVEL: str = "INFO"
    # Security: webhook shared secret for Prometheus Alertmanager
    ALERT_WEBHOOK_SECRET: str = ""  # set via env; if empty webhook is open (dev only)

    LLM_PROVIDER: str = "ollama"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_BASE_URL: str = ""  # e.g. https://openrouter.ai/api/v1 for OpenRouter; empty = default OpenAI
    MODEL: str = ""  # alias for OPENAI_MODEL (you set nvidia/nemotron-3-ultra-550b-a55b:free)
    EMBEDDING_MODEL: str = "nomic-embed-text"  # Ollama only — embeddings always via Ollama
    MAX_TOKENS: int = 4096
    TEMPERATURE: float = 0.7

    # Hardcoded limit replacements (Milestone 7)
    REPOSITORY_FILE_LIMIT: int = 200
    BATCH_FILE_LIMIT: int = 500
    FILE_PREVIEW_LIMIT: int = 50
    FILE_PREVIEW_CHARS: int = 300
    CONTEXT_TRUNCATION_LIMIT: int = 30
    SIMILARITY_THRESHOLD: float = 0.3

    # Langfuse (open-source) — optional; no-op if keys empty
    # Official env names per https://langfuse.com/docs/observability/get-started
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_BASE_URL: str = "https://us.cloud.langfuse.com"  # EU: https://cloud.langfuse.com
    LANGFUSE_HOST: str = ""  # alias for BASE_URL (skill says export LANGFUSE_HOST="$LANGFUSE_BASE_URL")
    LANGFUSE_ENABLED: bool = False  # legacy flag; auto-enabled when keys present

    @property
    def langfuse_host(self) -> str:
        return self.LANGFUSE_HOST or self.LANGFUSE_BASE_URL

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.LANGFUSE_SECRET_KEY and self.LANGFUSE_PUBLIC_KEY and (self.LANGFUSE_ENABLED or self.LANGFUSE_SECRET_KEY.startswith("sk-lf-")))

    @field_validator("SECRET_KEY")
    @classmethod
    def _validate_secret_key(cls, v: str) -> str:
        # Fail fast in production; allow short dev key only when explicitly dev
        env = os.getenv("ENV", "dev").lower()
        if env in ("prod", "production") and len(v) < 32:
            raise ValueError("SECRET_KEY must be >=32 chars in production (ENV=prod)")
        if len(v) < 16:
            raise ValueError("SECRET_KEY must be >=16 chars")
        return v

    @field_validator("BACKEND_CORS_ORIGINS")
    @classmethod
    def _validate_cors(cls, v: List[str]) -> List[str]:
        # Reject wildcard in production
        env = os.getenv("ENV", "dev").lower()
        if env in ("prod", "production") and "*" in v:
            raise ValueError("CORS wildcard '*' not allowed in production")
        return v

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
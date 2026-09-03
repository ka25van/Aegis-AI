"""Langfuse tracing — open-source LLM observability (backend/app/services/tracing.py)

Follows https://github.com/langfuse/skills skill best-practices:
- LangChain CallbackHandler integration (preferred over manual)
- @observe decorator for agent/workflow spans
- get_client() + start_as_current_observation for manual generations
- Optional/no-op when keys unset; env vars loaded before import.

Official env: LANGFUSE_SECRET_KEY / LANGFUSE_PUBLIC_KEY / LANGFUSE_BASE_URL
(US: https://us.cloud.langfuse.com, EU: https://cloud.langfuse.com)
LANGFUSE_HOST is accepted as alias (export LANGFUSE_HOST="$LANGFUSE_BASE_URL").
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Optional

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_client: Any = None
_callback_handler: Any = None


def _enabled() -> bool:
    return bool(settings.langfuse_enabled)


def _resolve_host() -> str:
    return settings.langfuse_host


def get_langfuse() -> Optional[Any]:
    """Lazy singleton via langfuse.get_client() (SDK v3) or Langfuse() fallback."""
    global _client
    if not _enabled():
        return None
    if _client is not None:
        return _client
    try:
        # SDK v3 preferred
        try:
            from langfuse import get_client

            # Ensure env vars are set for get_client() auto-config
            import os

            os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.LANGFUSE_SECRET_KEY)
            os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.LANGFUSE_PUBLIC_KEY)
            os.environ.setdefault("LANGFUSE_HOST", _resolve_host())
            os.environ.setdefault("LANGFUSE_BASE_URL", _resolve_host())
            _client = get_client()
            logger.info("langfuse_enabled", host=_resolve_host(), via="get_client")
            return _client
        except ImportError:
            from langfuse import Langfuse

            _client = Langfuse(
                secret_key=settings.LANGFUSE_SECRET_KEY,
                public_key=settings.LANGFUSE_PUBLIC_KEY,
                host=_resolve_host(),
            )
            logger.info("langfuse_enabled", host=_resolve_host(), via="Langfuse")
            return _client
    except Exception as e:  # pragma: no cover
        logger.warning("langfuse_init_failed", error=str(e))
        return None


def get_callback_handler() -> Optional[Any]:
    """LangChain CallbackHandler — captures model, tokens, cost auto. No-op if disabled."""
    global _callback_handler
    if not _enabled():
        return None
    if _callback_handler is not None:
        return _callback_handler
    try:
        from langfuse.langchain import CallbackHandler

        # CallbackHandler reads env vars LANGFUSE_* automatically
        import os

        os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.LANGFUSE_SECRET_KEY)
        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.LANGFUSE_PUBLIC_KEY)
        os.environ.setdefault("LANGFUSE_HOST", _resolve_host())
        os.environ.setdefault("LANGFUSE_BASE_URL", _resolve_host())
        _callback_handler = CallbackHandler()
        return _callback_handler
    except Exception as e:
        logger.warning("langfuse_callback_failed", error=str(e))
        return None


def flush() -> None:
    if _client is not None:
        try:
            _client.flush()
        except Exception:
            pass
    if _callback_handler is not None:
        try:
            _callback_handler.flush()  # type: ignore
        except Exception:
            pass


# ---- manual generation helper (fallback when not using CallbackHandler) ----

@contextmanager
def trace_generation(
    *,
    name: str = "llm.generate",
    model: Optional[str] = None,
    input_text: Optional[str] = None,
    metadata: Optional[dict] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
):
    """Manual generation span via get_client().start_as_current_observation."""
    lf = get_langfuse()
    if lf is None:
        yield _NoopSpan()
        return

    # Prefer start_as_current_observation (SDK v3)
    gen = None
    try:
        if hasattr(lf, "start_as_current_observation"):
            ctx = lf.start_as_current_observation(as_type="generation", name=name, model=model or settings.OLLAMA_MODEL, input=input_text, metadata=metadata or {})
            with ctx as gen:
                start = time.monotonic()
                try:
                    yield gen
                finally:
                    pass
            try:
                lf.flush()
            except Exception:
                pass
            return
    except Exception:
        pass

    # Fallback: trace().generation()
    try:
        trace = lf.trace(name=name, session_id=session_id, user_id=user_id, metadata=metadata or {})
        gen = trace.generation(name=name, model=model or settings.OLLAMA_MODEL, input=input_text, metadata=metadata or {})
        try:
            yield gen
        finally:
            try:
                lf.flush()
            except Exception:
                pass
    except Exception:
        yield _NoopSpan()


class _NoopSpan:
    def update(self, **kwargs: Any) -> None:
        pass

    def end(self, **kwargs: Any) -> None:
        pass

    def score(self, **kwargs: Any) -> None:
        pass

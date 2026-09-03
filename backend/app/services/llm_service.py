from typing import Optional
from fastapi import Depends
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from app.core.config import settings
from app.core.reliability import CircuitBreaker

# Shared circuit breaker for Ollama (shared across requests)
_ollama_breaker = CircuitBreaker(failure_threshold=5, window_s=60, recovery_timeout=30)


class LLMService:
    def __init__(self):
        self._model: Optional[BaseChatModel] = None

    def _get_model(self) -> BaseChatModel:
        if self._model is not None:
            return self._model
        provider = settings.LLM_PROVIDER
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            # Support MODEL alias you set (nvidia/nemotron-3-ultra-550b-a55b:free)
            model_name = getattr(settings, "MODEL", "") or settings.OPENAI_MODEL
            kwargs: dict = dict(
                model=model_name,
                api_key=settings.OPENAI_API_KEY,
                temperature=settings.TEMPERATURE,
                max_tokens=settings.MAX_TOKENS,
            )
            if getattr(settings, "OPENAI_BASE_URL", ""):
                kwargs["base_url"] = settings.OPENAI_BASE_URL
            self._model = ChatOpenAI(**kwargs)
        else:
            from langchain_ollama import ChatOllama
            self._model = ChatOllama(
                model=settings.OLLAMA_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=settings.TEMPERATURE,
                num_predict=settings.MAX_TOKENS,
            )
        return self._model

    def get_model(self) -> BaseChatModel:
        return self._get_model()

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        """LangChain CallbackHandler auto-traces model, tokens, cost (skill best-practice).

        Wrapped in CircuitBreaker — fast-fail when Ollama is down after 5 failures.
        """
        from app.services.tracing import get_callback_handler

        if not _ollama_breaker.allow_request():
            raise ConnectionError(f"Ollama circuit breaker open ({_ollama_breaker.state}), retry after {int(_ollama_breaker.remaining())}s")

        model = self._get_model()
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        chain = model | StrOutputParser()
        handler = get_callback_handler()
        callbacks = [handler] if handler else []
        try:
            if callbacks:
                result = await chain.ainvoke(messages, config={"callbacks": callbacks})
            else:
                from app.services.tracing import trace_generation

                model_name = getattr(self._model, "model", None) or getattr(self._model, "model_name", None) or settings.OLLAMA_MODEL
                input_text = f"SYSTEM: {system_prompt}\nUSER: {user_prompt}"
                with trace_generation(name="llm.generate", model=model_name, input_text=input_text) as gen:
                    import time as _t

                    start = _t.monotonic()
                    result = await chain.ainvoke(messages)
                    latency = _t.monotonic() - start
                    try:
                        gen.update(output=result, metadata={"latency_s": round(latency, 3)})
                    except Exception:
                        pass
            _ollama_breaker.record_success()
            return result
        except Exception as e:
            msg = str(e).lower()
            if any(m in msg for m in ("connect", "timeout", "connection", "unavailable", "refused")):
                _ollama_breaker.record_failure()
            raise

    async def generate_with_context(self, system_prompt: str, context: str, user_query: str) -> str:
        from app.services.tracing import get_callback_handler

        if not _ollama_breaker.allow_request():
            raise ConnectionError(f"Ollama circuit breaker open ({_ollama_breaker.state})")

        model = self._get_model()
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Context:\n{context}\n\nQuery: {user_query}"),
        ]
        chain = model | StrOutputParser()
        handler = get_callback_handler()
        callbacks = [handler] if handler else []
        try:
            if callbacks:
                result = await chain.ainvoke(messages, config={"callbacks": callbacks})
            else:
                from app.services.tracing import trace_generation

                model_name = getattr(self._model, "model", None) or getattr(self._model, "model_name", None) or settings.OLLAMA_MODEL
                input_text = f"SYSTEM: {system_prompt}\nCONTEXT: {context[:2000]}\nQUERY: {user_query}"
                with trace_generation(name="llm.generate_with_context", model=model_name, input_text=input_text) as gen:
                    import time as _t

                    start = _t.monotonic()
                    result = await chain.ainvoke(messages)
                    latency = _t.monotonic() - start
                    try:
                        gen.update(output=result, metadata={"latency_s": round(latency, 3)})
                    except Exception:
                        pass
                    _ollama_breaker.record_success()
                    return result
            _ollama_breaker.record_success()
            return result
        except Exception as e:
            msg = str(e).lower()
            if any(m in msg for m in ("connect", "timeout", "connection", "unavailable", "refused")):
                _ollama_breaker.record_failure()
            raise


llm_service = LLMService()


async def get_llm_service() -> LLMService:
    return llm_service

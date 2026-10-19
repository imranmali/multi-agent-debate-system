"""Backend for Anthropic Claude models (optional ``anthropic`` extra).

The SDK is imported lazily when a client is created, so the core package has no
runtime dependencies. Only ``system`` and ``prompt`` of a request are sent.
"""

from __future__ import annotations

from typing import Any

from debate_system.llm.base import BackendError, LLMRequest

DEFAULT_MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
INSTALL_HINT = (
    "The Anthropic backend requires the 'anthropic' package: "
    "pip install 'multi-agent-debate[anthropic]'"
)


class AnthropicBackend:
    """Calls the Messages API through ``anthropic.AsyncAnthropic``.

    ``effort`` sets ``output_config.effort`` (``None`` leaves the API default);
    ``use_fallbacks`` enables server-side refusal fallback. Pass ``client`` to
    inject a pre-built (or fake) async client; otherwise ``client_kwargs`` are
    forwarded to ``AsyncAnthropic`` (credentials default to the environment).
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 16000,
        effort: str | None = "medium",
        use_fallbacks: bool = True,
        client: Any | None = None,
        **client_kwargs: Any,
    ) -> None:
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        if client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise BackendError(INSTALL_HINT) from exc
            client = anthropic.AsyncAnthropic(**client_kwargs)
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.use_fallbacks = use_fallbacks
        self._client = client

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r})"

    def _extra_params(self) -> dict[str, Any]:
        extra: dict[str, Any] = {}
        if self.effort is not None:
            extra["output_config"] = {"effort": self.effort}
        if self.use_fallbacks:
            extra["betas"] = [FALLBACK_BETA]
            extra["fallbacks"] = "default"
        return extra

    async def complete(self, request: LLMRequest) -> str:
        try:
            response = await self._client.beta.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=request.system,
                messages=[{"role": "user", "content": request.prompt}],
                **self._extra_params(),
            )
        except Exception as exc:
            raise BackendError(f"Anthropic API call failed ({type(exc).__name__}): {exc}") from exc

        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details is not None else None
            suffix = f" (category: {category})" if category else ""
            raise BackendError(f"The model refused the {request.role} request{suffix}")

        parts: list[str] = [
            str(block.text)
            for block in getattr(response, "content", None) or ()
            if getattr(block, "type", None) == "text"
        ]
        text = "".join(parts)
        if not text.strip():
            raise BackendError(f"The model returned no text for the {request.role} request")
        return text

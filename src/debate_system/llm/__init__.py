"""Model backends and the :func:`create_backend` factory."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from debate_system.llm.anthropic_backend import AnthropicBackend
from debate_system.llm.base import ROLES, BackendError, LLMBackend, LLMRequest, Role
from debate_system.llm.mock import MockBackend
from debate_system.llm.scripted import ScriptedBackend

#: Names accepted by :func:`create_backend`.
BACKENDS: tuple[str, ...] = ("mock", "anthropic")

_FACTORIES: dict[str, Callable[..., LLMBackend]] = {
    "mock": MockBackend,
    "anthropic": AnthropicBackend,
}


def create_backend(name: str, **kwargs: Any) -> LLMBackend:
    """Build the backend called ``name`` (case-insensitive) with ``kwargs``."""
    factory = _FACTORIES.get(name.strip().lower())
    if factory is None:
        raise BackendError(f"Unknown backend {name!r}; choose one of: {', '.join(BACKENDS)}")
    return factory(**kwargs)


__all__ = [
    "BACKENDS",
    "ROLES",
    "AnthropicBackend",
    "BackendError",
    "LLMBackend",
    "LLMRequest",
    "MockBackend",
    "Role",
    "ScriptedBackend",
    "create_backend",
]

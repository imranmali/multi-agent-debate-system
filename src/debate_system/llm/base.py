"""Provider-agnostic backend contract.

A backend turns an :class:`LLMRequest` into reply text. Real providers use only
``system`` and ``prompt``; the offline :class:`~debate_system.llm.mock.MockBackend`
reads ``metadata`` (keys documented in design.md - Backend contract) so it can
simulate a swarm without parsing prose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

Role = Literal["proposer", "reviser", "critic", "voter", "aggregator"]
ROLES: tuple[Role, ...] = ("proposer", "reviser", "critic", "voter", "aggregator")


class BackendError(RuntimeError):
    """Raised when a backend cannot produce a reply (missing SDK, refusal, API error)."""


@dataclass(frozen=True)
class LLMRequest:
    """A single model call made by an agent."""

    role: Role
    system: str
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class LLMBackend(Protocol):
    """Anything with an async ``complete`` method can drive a debate."""

    async def complete(self, request: LLMRequest) -> str: ...

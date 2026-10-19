"""Backend that replays canned replies, for tests.

Each role has a queue of replies consumed in call order; the last entry repeats
once the queue is exhausted. An entry is returned (``str``), raised
(``BaseException`` instance) or called with the request (callable).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from typing import TypeAlias

from debate_system.llm.base import BackendError, LLMRequest

Reply: TypeAlias = "str | BaseException | Callable[[LLMRequest], str]"
Script: TypeAlias = "Reply | Sequence[Reply]"


class ScriptedBackend:
    """Replies from a per-role script and records every request in :attr:`calls`.

    A single reply instead of a sequence (e.g. a bare callable) is used for every
    request. Roles without a script get ``default`` when it is set, otherwise
    :class:`BackendError`. ``delay`` seconds are awaited before each reply so tests
    can exercise concurrency.
    """

    def __init__(
        self,
        replies: Mapping[str, Script] | None = None,
        default: str | None = None,
        delay: float = 0.0,
    ) -> None:
        if delay < 0:
            raise ValueError(f"delay must be >= 0, got {delay}")
        self._queues: dict[str, list[Reply]] = {}
        for role, script in (replies or {}).items():
            if isinstance(script, (str, BaseException)) or callable(script):
                # A single reply (or bare callable) repeats forever.
                self._queues[role] = [script]
            elif not script:
                raise ValueError(f"script for role {role!r} is empty")
            else:
                self._queues[role] = list(script)
        self._positions: dict[str, int] = dict.fromkeys(self._queues, 0)
        self.default = default
        self.delay = delay
        self.calls: list[LLMRequest] = []

    def calls_for(self, role: str) -> list[LLMRequest]:
        """The recorded requests made with ``role``."""
        return [call for call in self.calls if call.role == role]

    async def complete(self, request: LLMRequest) -> str:
        # Record and pick the reply before awaiting so ordering follows call order.
        self.calls.append(request)
        reply = self._next_reply(request.role)
        if self.delay:
            await asyncio.sleep(self.delay)
        if isinstance(reply, BaseException):
            raise reply
        if callable(reply):
            return reply(request)
        return reply

    def _next_reply(self, role: str) -> Reply:
        queue = self._queues.get(role)
        if queue is None:
            if self.default is None:
                raise BackendError(f"ScriptedBackend: no reply scripted for role {role!r}")
            return self.default
        position = self._positions[role]
        self._positions[role] = position + 1
        return queue[min(position, len(queue) - 1)]

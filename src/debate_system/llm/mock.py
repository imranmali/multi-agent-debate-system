"""Deterministic offline backend that simulates a swarm of debating agents.

:class:`MockBackend` never parses prose: it reads the structured ``metadata`` of
each :class:`~debate_system.llm.base.LLMRequest` (keys documented in design.md -
Backend contract) and replies with the JSON the agents expect. All randomness is
derived from SHA-256 digests of the seed and the call's inputs, so results are
identical across processes for the same seed.
"""

from __future__ import annotations

import hashlib
import json
import re
import string
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from debate_system.llm.base import BackendError, LLMRequest
from debate_system.models import CRITERIA

#: Canned stances a simulated proposer can take. ``num_stances`` selects a prefix.
STANCES: tuple[str, ...] = (
    "Start with the simplest solution that works and iterate",
    "Optimize for long-term scalability from the start",
    "Prioritize correctness and safety over speed of delivery",
    "Choose the most widely adopted, well-documented option",
    "Run a small experiment first and decide from measured data",
    "Minimize cost and operational overhead",
    "Keep the decision reversible so it can be changed cheaply",
    "Optimize for the end-user experience above internal concerns",
)

MIN_SCORE = 4
MAX_SCORE = 10
JITTER = 0.05

_WHITESPACE = re.compile(r"\s+")


def _digest(seed: int, *parts: object) -> int:
    """Map ``(seed, *parts)`` to a stable 64-bit integer."""
    h = hashlib.sha256(str(seed).encode())
    for part in parts:
        h.update(b"\x1f")
        h.update(str(part).encode())
    return int.from_bytes(h.digest()[:8], "big")


def _unit(seed: int, *parts: object) -> float:
    """Map ``(seed, *parts)`` to a stable float in ``[0, 1)``."""
    return _digest(seed, *parts) / 2**64


def normalize_answer(text: str) -> str:
    """Casefold, collapse whitespace and strip trailing punctuation."""
    collapsed = _WHITESPACE.sub(" ", text.casefold()).strip()
    return collapsed.rstrip(string.punctuation + " ")


def _require(metadata: Mapping[str, Any], role: str, *keys: str) -> list[Any]:
    missing = [key for key in keys if metadata.get(key) is None]
    if missing:
        raise BackendError(f"MockBackend: {role} request is missing metadata {', '.join(missing)}")
    return [metadata[key] for key in keys]


def _score(entry: Mapping[str, Any]) -> float:
    return float(entry.get("score") or 0.0)


class MockBackend:
    """Offline swarm simulator.

    ``seed`` drives every pseudo-random choice, ``convergence`` is the probability
    that a reviser adopts the best-scored peer answer, and ``num_stances`` is how
    many of :data:`STANCES` proposers choose from.
    """

    def __init__(self, seed: int = 0, convergence: float = 0.7, num_stances: int = 4) -> None:
        if not 0.0 <= convergence <= 1.0:
            raise ValueError(f"convergence must be in [0, 1], got {convergence}")
        if not 1 <= num_stances <= len(STANCES):
            raise ValueError(f"num_stances must be in [1, {len(STANCES)}], got {num_stances}")
        self.seed = seed
        self.convergence = convergence
        self.num_stances = num_stances
        self._handlers: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
            "proposer": self._propose,
            "reviser": self._revise,
            "critic": self._critique,
            "voter": self._vote,
            "aggregator": self._aggregate,
        }

    async def complete(self, request: LLMRequest) -> str:
        handler = self._handlers.get(request.role)
        if handler is None:
            raise BackendError(f"MockBackend: unknown role {request.role!r}")
        return json.dumps(handler(request.metadata))

    def stance_for(self, agent_id: str, persona: str) -> str:
        """The stance a proposer with this identity takes under this seed."""
        return STANCES[_digest(self.seed, "stance", agent_id, persona) % self.num_stances]

    def _propose(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        agent_id, persona = _require(metadata, "proposer", "agent_id", "persona")
        stance = self.stance_for(str(agent_id), str(persona))
        return {
            "answer": f"{stance}.",
            "rationale": f"As the {persona}, I weigh the trade-offs and favor this approach.",
        }

    def _revise(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        agent_id, round_, own_answer, peers = _require(
            metadata, "reviser", "agent_id", "round", "own_answer", "peers"
        )
        adopt = _unit(self.seed, "revise", agent_id, round_) < self.convergence
        best = max(peers, key=_score, default=None) if adopt else None
        if best is None or best.get("answer") is None:
            return {"answer": own_answer, "rationale": "I keep my answer after reviewing peers."}
        return {
            "answer": best["answer"],
            "rationale": f"I adopt {best.get('agent_id', 'a peer')}'s answer, "
            f"the best-scored ({_score(best):.2f}).",
        }

    def _critique(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        (proposal,) = _require(metadata, "critic", "proposal")
        answer = proposal.get("answer")
        if answer is None:
            raise BackendError("MockBackend: critic request proposal has no answer")
        key = normalize_answer(str(answer))
        span = MAX_SCORE - MIN_SCORE + 1
        scores = {c: MIN_SCORE + _digest(self.seed, "critic", key, c) % span for c in CRITERIA}
        weakest = min(CRITERIA, key=scores.__getitem__)
        strongest = max(CRITERIA, key=scores.__getitem__)
        feedback = (
            f"Strongest on {strongest}; weakest on {weakest} ({scores[weakest]}/10) - address it."
        )
        return {"scores": scores, "feedback": feedback}

    def _vote(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        agent_id, candidates = _require(metadata, "voter", "agent_id", "candidates")

        def key(candidate: Mapping[str, Any]) -> float:
            jitter = (2 * _unit(self.seed, "vote", agent_id, candidate["id"]) - 1) * JITTER
            return _score(candidate) + jitter

        ranked: Sequence[Mapping[str, Any]] = sorted(candidates, key=key, reverse=True)
        return {"ranking": [candidate["id"] for candidate in ranked]}

    def _aggregate(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        (candidates,) = _require(metadata, "aggregator", "candidates")
        if not candidates:
            raise BackendError("MockBackend: aggregator request has no candidates")
        winner_id = metadata.get("winner_id")
        winner = next((c for c in candidates if c.get("id") == winner_id), candidates[0])
        key = normalize_answer(str(winner["answer"]))
        supporters = [c for c in candidates if normalize_answer(str(c.get("answer", ""))) == key]
        agents = {c.get("agent_id") for c in supporters}
        return {
            "final_answer": winner["answer"],
            "summary": f"{len(supporters)} of {len(candidates)} candidates "
            f"from {len(agents)} agent(s) supported this answer.",
        }

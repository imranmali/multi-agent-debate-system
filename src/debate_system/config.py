"""Debate configuration and its validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

#: Voting strategies understood by :mod:`debate_system.consensus`.
VOTING_STRATEGIES: tuple[str, ...] = ("plurality", "borda", "weighted")

#: Personas assigned to proposer agents in order; reused cyclically.
DEFAULT_PERSONAS: tuple[str, ...] = (
    "pragmatist",
    "skeptic",
    "innovator",
    "analyst",
    "contrarian",
    "systems-thinker",
    "user-advocate",
    "risk-manager",
)

#: Default confidence weights (see design.md - Confidence).
DEFAULT_CONFIDENCE_WEIGHTS: dict[str, float] = {
    "support": 0.4,
    "quality": 0.3,
    "agreement": 0.2,
    "stability": 0.1,
}


class ConfigError(ValueError):
    """Raised when a :class:`DebateConfig` is invalid. ``field`` names the culprit."""

    def __init__(self, field_name: str, message: str) -> None:
        super().__init__(f"{field_name}: {message}")
        self.field = field_name


@dataclass
class DebateConfig:
    """Tunable parameters for a debate. Call :meth:`validate` before use."""

    num_agents: int = 3
    max_rounds: int = 3
    consensus_threshold: float = 0.6
    voting_strategy: str = "weighted"
    allow_self_vote: bool = False
    timeout_seconds: float = 120.0
    personas: tuple[str, ...] = DEFAULT_PERSONAS
    confidence_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_CONFIDENCE_WEIGHTS)
    )

    def validate(self) -> DebateConfig:
        """Raise :class:`ConfigError` if invalid; return ``self`` for chaining."""
        if self.num_agents < 2:
            raise ConfigError("num_agents", "a debate needs at least 2 agents")
        if self.max_rounds < 1:
            raise ConfigError("max_rounds", "must be at least 1")
        if not 0 < self.consensus_threshold <= 1:
            raise ConfigError("consensus_threshold", "must be in (0, 1]")
        if self.voting_strategy not in VOTING_STRATEGIES:
            raise ConfigError(
                "voting_strategy",
                f"unknown strategy {self.voting_strategy!r}; "
                f"choose from {', '.join(VOTING_STRATEGIES)}",
            )
        if self.timeout_seconds <= 0:
            raise ConfigError("timeout_seconds", "must be positive")
        if not self.personas:
            raise ConfigError("personas", "at least one persona is required")
        unknown = set(self.confidence_weights) - set(DEFAULT_CONFIDENCE_WEIGHTS)
        if unknown:
            raise ConfigError("confidence_weights", f"unknown components {sorted(unknown)}")
        if any(w < 0 for w in self.confidence_weights.values()):
            raise ConfigError("confidence_weights", "weights must be non-negative")
        if sum(self.confidence_weights.values()) <= 0:
            raise ConfigError("confidence_weights", "at least one weight must be positive")
        return self

    def persona_for(self, index: int) -> str:
        """Persona of the agent at ``index`` (cycles through :attr:`personas`)."""
        return self.personas[index % len(self.personas)]

    def agent_ids(self) -> list[str]:
        """Stable agent ids ``agent-1`` .. ``agent-N``."""
        return [f"agent-{i + 1}" for i in range(self.num_agents)]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["personas"] = list(self.personas)
        return data

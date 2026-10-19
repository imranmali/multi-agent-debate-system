"""Shared data model for the debate system.

Every module builds against these dataclasses. All of them are plain data and
serialize to JSON-compatible dictionaries through :func:`to_dict`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

#: Rubric criteria the critic scores on a 0-10 scale.
CRITERIA: tuple[str, ...] = ("correctness", "completeness", "clarity", "feasibility")


@dataclass(frozen=True)
class Proposal:
    """One agent's answer in one round.

    ``id`` is unique within a debate and has the form ``r{round}-{agent_id}``.
    """

    id: str
    agent_id: str
    persona: str
    round: int
    answer: str
    rationale: str = ""


@dataclass(frozen=True)
class Critique:
    """The critic's evaluation of one proposal.

    ``scores`` maps each criterion in :data:`CRITERIA` to a value in [0, 10];
    ``overall`` is their mean divided by 10, in [0, 1]. ``parsed`` is False when
    the critic's reply could not be parsed and neutral scores were substituted.
    """

    proposal_id: str
    scores: dict[str, float]
    overall: float
    feedback: str = ""
    parsed: bool = True


@dataclass(frozen=True)
class Ballot:
    """A voter's ranking of proposal ids, best first."""

    voter_id: str
    ranking: list[str]
    weight: float = 1.0


@dataclass(frozen=True)
class Position:
    """A group of proposals whose answers are equivalent after normalization."""

    key: str
    answer: str
    proposal_ids: list[str]
    agent_ids: list[str]
    mean_score: float


@dataclass(frozen=True)
class VoteResult:
    """Outcome of tallying the ballots of one round."""

    strategy: str
    positions: list[Position]
    points: dict[str, float]
    winner_key: str
    winner_proposal_id: str
    support: float
    consensus: bool


@dataclass(frozen=True)
class AgentError:
    """A failed agent call recorded in the transcript."""

    agent_id: str
    role: str
    message: str


@dataclass
class RoundRecord:
    """Everything that happened in one debate round."""

    round: int
    proposals: list[Proposal] = field(default_factory=list)
    critiques: list[Critique] = field(default_factory=list)
    ballots: list[Ballot] = field(default_factory=list)
    vote: VoteResult | None = None
    errors: list[AgentError] = field(default_factory=list)


@dataclass(frozen=True)
class ConfidenceComponent:
    """One term of the confidence score: ``value`` in [0, 1] times ``weight``."""

    name: str
    value: float
    weight: float


@dataclass(frozen=True)
class Confidence:
    """Final confidence score with its audit trail."""

    score: float
    components: list[ConfidenceComponent]


@dataclass
class DebateResult:
    """The full outcome and transcript of a debate."""

    question: str
    final_answer: str
    summary: str
    winner_proposal_id: str
    confidence: Confidence
    consensus_reached: bool
    rounds_run: int
    rounds: list[RoundRecord]
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_dict(self)


def to_dict(obj: Any) -> dict[str, Any]:
    """Convert any dataclass in this module to a JSON-compatible dict."""
    return asdict(obj)

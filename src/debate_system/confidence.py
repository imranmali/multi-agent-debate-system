"""Explainable confidence score for a finished debate.

``confidence = sum(weight_i * value_i)`` over :data:`COMPONENTS`, clamped to
[0, 1]; see design.md - Confidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from debate_system.models import Confidence, ConfidenceComponent, RoundRecord, VoteResult

#: Confidence components in reporting order.
COMPONENTS: tuple[str, ...] = ("support", "quality", "agreement", "stability")


def normalize_weights(weights: Mapping[str, float]) -> dict[str, float]:
    """Scale ``weights`` to sum to 1 over all :data:`COMPONENTS` (missing ones are 0).

    Raises :class:`ValueError` on unknown components, negative weights or a zero total.
    """
    unknown = set(weights) - set(COMPONENTS)
    if unknown:
        raise ValueError(f"unknown confidence components {sorted(unknown)}")
    if any(w < 0 for w in weights.values()):
        raise ValueError("confidence weights must be non-negative")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("at least one confidence weight must be positive")
    return {name: weights.get(name, 0.0) / total for name in COMPONENTS}


def compute_confidence(rounds: Sequence[RoundRecord], weights: Mapping[str, float]) -> Confidence:
    """Confidence in the winner of the last voted round of ``rounds``.

    - ``support``: that vote's support.
    - ``quality``: mean critic score of the winning position.
    - ``agreement``: winning position's share of the proposals tallied in that round.
    - ``stability``: fraction of voted rounds won by the same position.

    Raises :class:`ValueError` if no round has a vote, or on invalid ``weights``.
    """
    votes: list[VoteResult] = [r.vote for r in rounds if r.vote is not None]
    if not votes:
        raise ValueError("cannot compute confidence: no round has a vote")
    final = votes[-1]
    winner = next(p for p in final.positions if p.key == final.winner_key)
    tallied = sum(len(p.proposal_ids) for p in final.positions)

    values = {
        "support": final.support,
        "quality": winner.mean_score,
        "agreement": len(winner.proposal_ids) / tallied,
        "stability": sum(v.winner_key == final.winner_key for v in votes) / len(votes),
    }
    normalized = normalize_weights(weights)
    score = sum(normalized[name] * values[name] for name in COMPONENTS)
    return Confidence(
        score=min(1.0, max(0.0, score)),
        components=[
            ConfidenceComponent(name, values[name], normalized[name]) for name in COMPONENTS
        ],
    )

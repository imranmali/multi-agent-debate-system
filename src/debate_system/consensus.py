"""Answer positions, ballot tallying and the consensus rule.

Pure functions over :mod:`debate_system.models`; see design.md - Positions,
voting and consensus. Critiques are looked up by proposal id; a proposal
without a critique counts as the neutral score :data:`NEUTRAL_SCORE`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from debate_system.config import VOTING_STRATEGIES
from debate_system.models import Ballot, Critique, Position, Proposal, VoteResult

#: Critic score assumed for a proposal that has no critique.
NEUTRAL_SCORE = 0.5

#: Lowest weight a voter can have under the ``weighted`` strategy.
MIN_VOTER_WEIGHT = 0.1

#: Characters stripped from the end of an answer by :func:`normalize_answer`.
#: Includes the ellipsis and the CJK full stop/fullwidth forms of ``!?;:,``.
TRAILING_PUNCTUATION = ".!?;:,\u2026\u3002\uff01\uff1f\uff1b\uff1a\uff0c"

#: Slack for float comparisons so that e.g. 3/5 support meets a 0.6 threshold.
_EPSILON = 1e-9


def normalize_answer(text: str) -> str:
    """Canonical form of an answer: casefolded, whitespace collapsed, no trailing punctuation."""
    collapsed = " ".join(text.casefold().split())
    return collapsed.rstrip(TRAILING_PUNCTUATION + " ")


def _score(critiques: Mapping[str, Critique], proposal_id: str) -> float:
    critique = critiques.get(proposal_id)
    return critique.overall if critique is not None else NEUTRAL_SCORE


def _best_proposal(group: Sequence[Proposal], critiques: Mapping[str, Critique]) -> Proposal:
    """Highest critic-scored proposal of ``group``; ties go to the earliest."""
    return max(group, key=lambda p: _score(critiques, p.id))  # max keeps the first maximum


def build_positions(
    proposals: Sequence[Proposal], critiques: Mapping[str, Critique]
) -> list[Position]:
    """Group proposals by normalized answer, in order of first appearance."""
    groups: dict[str, list[Proposal]] = {}
    for proposal in proposals:
        groups.setdefault(normalize_answer(proposal.answer), []).append(proposal)
    return [
        Position(
            key=key,
            answer=_best_proposal(group, critiques).answer,
            proposal_ids=[p.id for p in group],
            agent_ids=[p.agent_id for p in group],
            mean_score=sum(_score(critiques, p.id) for p in group) / len(group),
        )
        for key, group in groups.items()
    ]


def voter_weights(
    proposals: Sequence[Proposal], critiques: Mapping[str, Critique]
) -> dict[str, float]:
    """Agent id -> critic score of its latest proposal, floored at :data:`MIN_VOTER_WEIGHT`."""
    return {p.agent_id: max(MIN_VOTER_WEIGHT, _score(critiques, p.id)) for p in proposals}


def _ballot_positions(ballot: Ballot, position_of: Mapping[str, str]) -> list[str]:
    """Position keys of a ballot, best first; unknown ids dropped, duplicates collapsed."""
    keys = (position_of[pid] for pid in ballot.ranking if pid in position_of)
    return list(dict.fromkeys(keys))


def tally(
    proposals: Sequence[Proposal],
    critiques: Mapping[str, Critique],
    ballots: Sequence[Ballot],
    strategy: str,
    threshold: float,
) -> VoteResult:
    """Tally ``ballots`` over the positions of ``proposals`` with ``strategy``.

    Ballots rank proposal ids; each is mapped to positions (first appearance
    wins) and skipped if nothing known remains. ``plurality`` gives one point
    to the first position, ``borda`` gives ``n-1-rank`` points over the ``n``
    positions on the ballot, and ``weighted`` multiplies Borda points by
    :attr:`Ballot.weight`.

    The winner has the most points; ties break on higher mean critic score,
    then on the position whose first proposal appears earliest. Under Borda
    every single-position ballot scores zero, so when no position scored at all
    the (weighted) first-place tallies stand in for points when ranking; the
    reported ``points`` are unchanged.

    Support is the (weighted, for ``weighted``) fraction of non-empty ballots
    whose first position is the winner, or 0.0 without any; consensus is
    ``support >= threshold``.

    Raises :class:`ValueError` on empty ``proposals`` or an unknown strategy.
    """
    if not proposals:
        raise ValueError("cannot tally a round without proposals")
    if strategy not in VOTING_STRATEGIES:
        raise ValueError(
            f"unknown voting strategy {strategy!r}; choose from {', '.join(VOTING_STRATEGIES)}"
        )

    positions = build_positions(proposals, critiques)
    position_of = {p.id: normalize_answer(p.answer) for p in proposals}
    points = dict.fromkeys(position_of.values(), 0.0)
    first_places = dict.fromkeys(points, 0.0)
    total_weight = 0.0

    for ballot in ballots:
        keys = _ballot_positions(ballot, position_of)
        if not keys:
            continue
        weight = ballot.weight if strategy == "weighted" else 1.0
        total_weight += weight
        first_places[keys[0]] += weight
        if strategy == "plurality":
            points[keys[0]] += 1.0
        else:
            for rank, key in enumerate(keys):
                points[key] += (len(keys) - 1 - rank) * weight

    ranking_points = points if any(points.values()) else first_places
    # ``positions`` is in order of first appearance, so ``max`` keeps the earliest on a full tie.
    winner = max(positions, key=lambda pos: (ranking_points[pos.key], pos.mean_score))
    winner_proposal = _best_proposal(
        [p for p in proposals if position_of[p.id] == winner.key], critiques
    )
    support = first_places[winner.key] / total_weight if total_weight > 0 else 0.0

    return VoteResult(
        strategy=strategy,
        positions=positions,
        points=points,
        winner_key=winner.key,
        winner_proposal_id=winner_proposal.id,
        support=support,
        consensus=support + _EPSILON >= threshold,
    )

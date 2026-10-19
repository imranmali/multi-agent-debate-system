from collections.abc import Sequence

import pytest

from debate_system.consensus import (
    build_positions,
    normalize_answer,
    tally,
    voter_weights,
)
from debate_system.models import Ballot, Critique, Proposal


def proposal(agent: str, answer: str, round_: int = 1) -> Proposal:
    return Proposal(
        id=f"r{round_}-{agent}", agent_id=agent, persona="analyst", round=round_, answer=answer
    )


def critiques(**overall: float) -> dict[str, Critique]:
    """Critiques keyed by proposal id; keyword ``a=0.8`` scores proposal ``r1-a``."""
    return {
        f"r1-{agent}": Critique(proposal_id=f"r1-{agent}", scores={}, overall=score)
        for agent, score in overall.items()
    }


def ballots(*rankings: Sequence[str], weights: Sequence[float] | None = None) -> list[Ballot]:
    """Ballots ranking agents' round-1 proposals, e.g. ``ballots("ab", "ba")``."""
    weights = weights or [1.0] * len(rankings)
    return [
        Ballot(voter_id=f"voter-{i}", ranking=[f"r1-{a}" for a in ranking], weight=w)
        for i, (ranking, w) in enumerate(zip(rankings, weights, strict=True))
    ]


# --- normalization and positions -------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Use Postgres.", "use postgres"),
        ("use   postgres", "use postgres"),
        ("  USE\tPostgres !?\n", "use postgres"),
        ("Yes; no.", "yes; no"),
        ("...", ""),
    ],
)
def test_normalize_answer(raw: str, expected: str) -> None:
    assert normalize_answer(raw) == expected


def test_equivalent_answers_merge() -> None:
    proposals = [
        proposal("a", "Use Postgres."),
        proposal("b", "SQLite"),
        proposal("c", "use postgres"),
    ]
    positions = build_positions(proposals, critiques(a=0.6, b=0.9, c=0.8))

    assert [p.key for p in positions] == ["use postgres", "sqlite"]
    postgres = positions[0]
    assert postgres.proposal_ids == ["r1-a", "r1-c"]
    assert postgres.agent_ids == ["a", "c"]
    assert postgres.answer == "use postgres"  # best-scored member's original text
    assert postgres.mean_score == pytest.approx(0.7)


def test_position_answer_ties_go_to_earliest_and_missing_critique_is_neutral() -> None:
    proposals = [proposal("a", "Use Postgres."), proposal("b", "use postgres")]
    (position,) = build_positions(proposals, {})

    assert position.answer == "Use Postgres."
    assert position.mean_score == 0.5


def test_voter_weights_use_latest_proposal_floor_and_neutral_default() -> None:
    proposals = [
        proposal("a", "X", 1),
        proposal("a", "Y", 2),
        proposal("b", "X"),
        proposal("c", "Z"),
    ]
    scores = {
        "r1-a": Critique("r1-a", {}, 0.9),
        "r2-a": Critique("r2-a", {}, 0.3),
        "r1-b": Critique("r1-b", {}, 0.0),
    }

    assert voter_weights(proposals, scores) == {"a": 0.3, "b": 0.1, "c": 0.5}


# --- strategies ---------------------------------------------------------------


def test_plurality_winner_three_of_five() -> None:
    proposals = [proposal(a, a.upper()) for a in "xyz"]
    result = tally(proposals, {}, ballots("xyz", "xzy", "xyz", "yxz", "zyx"), "plurality", 0.6)

    assert result.winner_key == "x"
    assert result.winner_proposal_id == "r1-x"
    assert result.points == {"x": 3.0, "y": 1.0, "z": 1.0}
    assert result.support == pytest.approx(0.6)
    assert result.consensus  # threshold met at exactly 0.6
    assert result.strategy == "plurality"
    assert [p.key for p in result.positions] == ["x", "y", "z"]


def test_borda_rewards_broad_support() -> None:
    # Y is second on every ballot; X is first on two and last on three.
    proposals = [proposal(a, a.upper()) for a in "xyab"]
    votes = ballots("xyab", "xyab", "aybx", "byax", "aybx")

    borda = tally(proposals, {}, votes, "borda", 0.6)
    assert borda.points == {"x": 6.0, "y": 10.0, "a": 9.0, "b": 5.0}
    assert borda.winner_key == "y"
    assert borda.support == 0.0  # nobody ranked Y first
    assert not borda.consensus

    plurality = tally(proposals, {}, votes, "plurality", 0.6)
    assert plurality.points["y"] == 0.0
    assert plurality.winner_key == "x"  # tied with A on 2, X appears first


def test_borda_ignores_positions_missing_from_a_ballot() -> None:
    proposals = [proposal(a, a.upper()) for a in "xyz"]
    result = tally(proposals, {}, ballots("xy", "zx"), "borda", 0.5)

    assert result.points == {"x": 1.0, "y": 0.0, "z": 1.0}


def test_weighted_scales_borda_points_and_support_by_ballot_weight() -> None:
    proposals = [proposal(a, a.upper()) for a in "xy"]
    votes = ballots("xy", "yx", "yx", weights=[0.9, 0.2, 0.1])
    result = tally(proposals, {}, votes, "weighted", 0.6)

    assert result.points == pytest.approx({"x": 0.9, "y": 0.3})
    assert result.winner_key == "x"
    assert result.support == pytest.approx(0.75)
    assert result.consensus

    unweighted = tally(proposals, {}, votes, "borda", 0.6)
    assert unweighted.winner_key == "y"
    assert unweighted.support == pytest.approx(2 / 3)


def test_single_position_ballots_fall_back_to_first_place_counts() -> None:
    # Every ballot names one position, so all Borda points are zero.
    proposals = [proposal(a, a.upper()) for a in "xy"]
    scores = critiques(x=0.2, y=0.9)
    result = tally(proposals, scores, ballots("x", "x", "y"), "borda", 0.6)

    assert result.points == {"x": 0.0, "y": 0.0}
    assert result.winner_key == "x"
    assert result.support == pytest.approx(2 / 3)
    assert result.consensus


def test_unanimous_single_candidate_has_full_support() -> None:
    proposals = [proposal(a, "Use Postgres") for a in "abc"]
    result = tally(
        proposals, critiques(a=0.4, b=0.8, c=0.8), ballots("bc", "ac", "ab"), "weighted", 0.6
    )

    assert result.winner_key == "use postgres"
    assert result.winner_proposal_id == "r1-b"  # best-scored member, earliest on a tie
    assert result.support == 1.0
    assert result.consensus


# --- tie-breaks and consensus rule -------------------------------------------


def test_tie_on_points_breaks_on_critic_score() -> None:
    proposals = [proposal(a, a.upper()) for a in "xy"]
    result = tally(proposals, critiques(x=0.4, y=0.8), ballots("xy", "yx"), "borda", 0.6)

    assert result.points == {"x": 1.0, "y": 1.0}
    assert result.winner_key == "y"
    assert result.support == 0.5


def test_full_tie_breaks_on_earliest_proposal() -> None:
    proposals = [proposal(a, a.upper()) for a in "yx"]
    result = tally(proposals, critiques(x=0.7, y=0.7), ballots("xy", "yx"), "plurality", 0.6)

    assert result.winner_key == "y"


def test_threshold_missed_at_half_support() -> None:
    proposals = [proposal(a, a.upper()) for a in "xy"]
    result = tally(proposals, {}, ballots("xy", "xy", "yx", "yx"), "plurality", 0.6)

    assert result.support == 0.5
    assert not result.consensus


# --- edge cases ---------------------------------------------------------------


def test_unknown_ids_are_ignored_and_duplicate_positions_collapse() -> None:
    proposals = [proposal("a", "X"), proposal("b", "x."), proposal("c", "Y")]
    votes = [
        Ballot("v1", ["ghost", "r1-a", "r1-b", "r1-c"]),  # -> [x, y]
        Ballot("v2", ["r1-c", "r1-b", "r1-a"]),  # -> [y, x]
        Ballot("v3", ["ghost"]),  # empty after mapping: skipped
        Ballot("v4", []),  # empty: skipped
    ]
    result = tally(proposals, {}, votes, "borda", 0.5)

    assert result.points == {"x": 1.0, "y": 1.0}
    assert result.support == 0.5
    assert result.consensus


def test_no_ballots_still_picks_a_winner_without_support() -> None:
    proposals = [proposal(a, a.upper()) for a in "xy"]
    result = tally(proposals, critiques(x=0.3, y=0.6), [], "weighted", 0.6)

    assert result.points == {"x": 0.0, "y": 0.0}
    assert result.winner_key == "y"
    assert result.winner_proposal_id == "r1-y"
    assert result.support == 0.0
    assert not result.consensus


def test_tally_rejects_empty_proposals() -> None:
    with pytest.raises(ValueError, match="without proposals"):
        tally([], {}, [], "borda", 0.6)


def test_tally_rejects_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="unknown voting strategy 'approval'"):
        tally([proposal("a", "X")], {}, [], "approval", 0.6)

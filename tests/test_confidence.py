import pytest

from debate_system.confidence import COMPONENTS, compute_confidence, normalize_weights
from debate_system.config import DEFAULT_CONFIDENCE_WEIGHTS
from debate_system.consensus import tally
from debate_system.models import Ballot, Critique, Proposal, RoundRecord


def debate_round(
    round_: int, answers: dict[str, str], scores: dict[str, float], rankings: list[str]
) -> RoundRecord:
    """A tallied round; ``rankings`` are strings of agent ids, best first."""
    proposals = [
        Proposal(id=f"r{round_}-{a}", agent_id=a, persona="analyst", round=round_, answer=text)
        for a, text in answers.items()
    ]
    critiques = [Critique(f"r{round_}-{a}", {}, s) for a, s in scores.items()]
    ballots = [
        Ballot(voter_id=f"v{i}", ranking=[f"r{round_}-{a}" for a in ranking])
        for i, ranking in enumerate(rankings)
    ]
    vote = tally(proposals, {c.proposal_id: c for c in critiques}, ballots, "borda", 0.6)
    return RoundRecord(round_, proposals, critiques, ballots, vote)


def test_custom_weights_normalized() -> None:
    weights = normalize_weights({"support": 2, "quality": 1, "agreement": 1, "stability": 0})
    assert weights == {"support": 0.5, "quality": 0.25, "agreement": 0.25, "stability": 0.0}


def test_missing_weights_default_to_zero() -> None:
    assert normalize_weights({"quality": 3}) == {
        "support": 0.0,
        "quality": 1.0,
        "agreement": 0.0,
        "stability": 0.0,
    }


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ({"luck": 1.0}, "unknown"),
        ({"support": -1.0, "quality": 2.0}, "non-negative"),
        ({"support": 0.0}, "positive"),
        ({}, "positive"),
    ],
)
def test_invalid_weights_rejected(weights: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_weights(weights)


def test_unanimous_high_quality_debate_has_full_confidence() -> None:
    answers = {"a": "Use Postgres", "b": "use postgres.", "c": "USE POSTGRES"}
    perfect = {"a": 1.0, "b": 1.0, "c": 1.0}
    rounds = [debate_round(r, answers, perfect, ["bc", "ac", "ab"]) for r in (1, 2)]

    confidence = compute_confidence(rounds, DEFAULT_CONFIDENCE_WEIGHTS)

    assert confidence.score == pytest.approx(1.0)
    assert [c.name for c in confidence.components] == list(COMPONENTS)
    assert all(c.value == pytest.approx(1.0) for c in confidence.components)
    assert [c.weight for c in confidence.components] == pytest.approx([0.4, 0.3, 0.2, 0.1])


def test_split_debate_with_winner_change_is_less_confident() -> None:
    scores = {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0}
    round1 = debate_round(
        1, {"a": "X", "b": "X", "c": "Y", "d": "Y"}, scores, ["xy", "xy", "yx", "yx"]
    )
    round2 = debate_round(
        2, {"a": "Y", "b": "X", "c": "Y", "d": "X"}, scores, ["ab", "ab", "ba", "ba"]
    )
    # Round 1 ties on points and score, so X (first) wins; round 2 is won by Y.
    assert round1.vote is not None and round1.vote.winner_key == "x"
    assert round2.vote is not None and round2.vote.winner_key == "y"

    unanimous = compute_confidence(
        [debate_round(1, {"a": "X", "b": "X"}, {"a": 1.0, "b": 1.0}, ["b", "a"])],
        DEFAULT_CONFIDENCE_WEIGHTS,
    )
    split = compute_confidence([round1, round2], DEFAULT_CONFIDENCE_WEIGHTS)

    values = {c.name: c.value for c in split.components}
    assert values == pytest.approx(
        {"support": 0.5, "quality": 1.0, "agreement": 0.5, "stability": 0.5}
    )
    assert split.score == pytest.approx(0.4 * 0.5 + 0.3 + 0.2 * 0.5 + 0.1 * 0.5)
    assert split.score < unanimous.score == pytest.approx(1.0)


def test_uses_last_voted_round_and_ignores_rounds_without_a_vote() -> None:
    voted = debate_round(1, {"a": "X", "b": "Y"}, {"a": 0.8, "b": 0.4}, ["ab", "ab"])
    unvoted = RoundRecord(round=2)

    confidence = compute_confidence([voted, unvoted], {"quality": 1.0, "stability": 1.0})

    values = {c.name: c.value for c in confidence.components}
    assert values["quality"] == pytest.approx(0.8)
    assert values["stability"] == 1.0
    assert confidence.score == pytest.approx(0.9)


def test_requires_a_vote() -> None:
    with pytest.raises(ValueError, match="no round has a vote"):
        compute_confidence([RoundRecord(round=1)], DEFAULT_CONFIDENCE_WEIGHTS)

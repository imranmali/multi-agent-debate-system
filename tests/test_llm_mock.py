import json
from typing import Any

import pytest

from debate_system.llm import BackendError, LLMRequest, MockBackend
from debate_system.llm.base import Role
from debate_system.llm.mock import MAX_SCORE, MIN_SCORE, STANCES, normalize_answer
from debate_system.models import CRITERIA

QUESTION = "How should we design the cache layer?"


async def call(backend: MockBackend, role: Role, **metadata: Any) -> dict[str, Any]:
    reply = await backend.complete(LLMRequest(role=role, system="", prompt="", metadata=metadata))
    data: dict[str, Any] = json.loads(reply)
    return data


async def propose(backend: MockBackend, agent_id: str, persona: str = "pragmatist") -> str:
    data = await call(
        backend, "proposer", agent_id=agent_id, persona=persona, round=1, question=QUESTION
    )
    answer: str = data["answer"]
    return answer


def peers(*scores: float) -> list[dict[str, Any]]:
    return [
        {"id": f"r1-a{i}", "agent_id": f"a{i}", "answer": f"Answer {i}.", "score": score}
        for i, score in enumerate(scores)
    ]


async def run_all_roles(backend: MockBackend) -> list[dict[str, Any]]:
    candidates = peers(0.6, 0.8, 0.7)
    return [
        await call(backend, "proposer", agent_id="a0", persona="skeptic", round=1, question="q"),
        await call(
            backend,
            "reviser",
            agent_id="a0",
            persona="skeptic",
            round=2,
            question="q",
            own_answer="Mine.",
            peers=candidates,
        ),
        await call(backend, "critic", question="q", proposal=candidates[0]),
        await call(backend, "voter", agent_id="a0", question="q", candidates=candidates),
        await call(backend, "aggregator", question="q", candidates=candidates, winner_id="r1-a1"),
    ]


async def test_same_seed_gives_identical_outputs() -> None:
    assert await run_all_roles(MockBackend(seed=7)) == await run_all_roles(MockBackend(seed=7))


async def test_different_seeds_can_differ() -> None:
    outcomes = set()
    for seed in range(8):
        backend = MockBackend(seed=seed)
        outcomes.add(tuple([await propose(backend, f"agent-{i}") for i in range(3)]))
    assert len(outcomes) >= 2


async def test_proposer_returns_answer_and_rationale_from_stances() -> None:
    data = await call(
        MockBackend(), "proposer", agent_id="a1", persona="skeptic", round=1, question=QUESTION
    )
    assert data["answer"].rstrip(".") in STANCES
    assert "skeptic" in data["rationale"]


async def test_same_stance_gives_same_answer_text_across_agents() -> None:
    backend = MockBackend(seed=3, num_stances=2)
    by_stance: dict[str, set[str]] = {}
    for i in range(12):
        agent_id = f"agent-{i}"
        by_stance.setdefault(backend.stance_for(agent_id, "pragmatist"), set()).add(
            await propose(backend, agent_id)
        )
    assert len(by_stance) == 2
    assert all(len(answers) == 1 for answers in by_stance.values())


async def test_single_stance_means_unanimous_proposals() -> None:
    backend = MockBackend(num_stances=1)
    answers = {await propose(backend, f"a{i}", persona=f"p{i}") for i in range(5)}
    assert answers == {f"{STANCES[0]}."}


@pytest.mark.parametrize("round_", [2, 3, 4, 5])
async def test_full_convergence_adopts_best_peer(round_: int) -> None:
    data = await call(
        MockBackend(convergence=1.0),
        "reviser",
        agent_id="a0",
        persona="p",
        round=round_,
        question=QUESTION,
        own_answer="Answer 0.",
        own_feedback="be clearer",
        peers=peers(0.5, 0.9, 0.9, 0.1),
    )
    assert data["answer"] == "Answer 1."  # ties break on list order
    assert "a1" in data["rationale"]


@pytest.mark.parametrize("round_", [2, 3, 4, 5])
async def test_zero_convergence_keeps_own_answer(round_: int) -> None:
    data = await call(
        MockBackend(convergence=0.0),
        "reviser",
        agent_id="a0",
        persona="p",
        round=round_,
        question=QUESTION,
        own_answer="My own answer.",
        peers=peers(0.5, 0.9),
    )
    assert data["answer"] == "My own answer."


async def test_reviser_without_peers_keeps_own_answer() -> None:
    data = await call(
        MockBackend(convergence=1.0),
        "reviser",
        agent_id="a0",
        persona="p",
        round=2,
        question=QUESTION,
        own_answer="Mine.",
        peers=[],
    )
    assert data["answer"] == "Mine."


async def test_partial_convergence_mixes_outcomes() -> None:
    backend = MockBackend(seed=1, convergence=0.5)
    answers = set()
    for i in range(20):
        data = await call(
            backend,
            "reviser",
            agent_id=f"a{i}",
            persona="p",
            round=2,
            question=QUESTION,
            own_answer="Mine.",
            peers=peers(0.1, 0.9),
        )
        answers.add(data["answer"])
    assert answers == {"Mine.", "Answer 1."}


async def test_critic_scores_in_range_with_feedback_naming_weakest() -> None:
    backend = MockBackend()
    for text in ("Use Redis.", "Use Memcached.", "Write a custom cache.", "Skip caching."):
        data = await call(
            backend,
            "critic",
            question=QUESTION,
            proposal={"id": "x", "agent_id": "a", "answer": text},
        )
        scores = data["scores"]
        assert set(scores) == set(CRITERIA)
        assert all(isinstance(v, int) and MIN_SCORE <= v <= MAX_SCORE for v in scores.values())
        weakest = min(CRITERIA, key=scores.__getitem__)
        assert f"weakest on {weakest}" in data["feedback"]


async def test_identical_answers_get_identical_critic_scores() -> None:
    backend = MockBackend(seed=5)
    variants = ["Use Redis.", "  use   REDIS ", "Use redis!!", "use\nredis"]
    results = [
        await call(
            backend,
            "critic",
            question=QUESTION,
            proposal={"id": f"p{i}", "agent_id": f"a{i}", "answer": v},
        )
        for i, v in enumerate(variants)
    ]
    assert all(r == results[0] for r in results)


async def test_critic_scores_vary_with_answer() -> None:
    backend = MockBackend()
    distinct = set()
    for stance in STANCES:
        data = await call(
            backend,
            "critic",
            question=QUESTION,
            proposal={"id": "p", "agent_id": "a", "answer": stance},
        )
        distinct.add(tuple(data["scores"].values()))
    assert len(distinct) > 1


async def test_critic_requires_proposal_answer() -> None:
    with pytest.raises(BackendError, match="no answer"):
        await call(MockBackend(), "critic", question=QUESTION, proposal={"id": "p"})


async def test_voter_ranks_all_candidates_by_score() -> None:
    candidates = peers(0.2, 0.9, 0.5, 0.7)
    data = await call(
        MockBackend(), "voter", agent_id="a0", question=QUESTION, candidates=candidates
    )
    assert data["ranking"] == ["r1-a1", "r1-a3", "r1-a2", "r1-a0"]


async def test_voter_jitter_breaks_ties_differently_per_voter() -> None:
    backend = MockBackend()
    candidates = peers(*[0.5] * 6)
    rankings = set()
    for i in range(6):
        data = await call(
            backend, "voter", agent_id=f"v{i}", question=QUESTION, candidates=candidates
        )
        assert sorted(data["ranking"]) == sorted(c["id"] for c in candidates)
        rankings.add(tuple(data["ranking"]))
    assert len(rankings) > 1


async def test_aggregator_returns_winner_and_support_summary() -> None:
    candidates = [
        {"id": "r2-a0", "agent_id": "a0", "answer": "Use Redis.", "score": 0.8},
        {"id": "r2-a1", "agent_id": "a1", "answer": "use redis", "score": 0.8},
        {"id": "r2-a2", "agent_id": "a2", "answer": "Skip it.", "score": 0.4},
    ]
    data = await call(
        MockBackend(), "aggregator", question=QUESTION, candidates=candidates, winner_id="r2-a1"
    )
    assert data["final_answer"] == "use redis"
    assert data["summary"].startswith("2 of 3 candidates from 2 agent(s)")


async def test_aggregator_falls_back_to_first_candidate() -> None:
    data = await call(
        MockBackend(), "aggregator", question=QUESTION, candidates=peers(0.1, 0.9), winner_id="nope"
    )
    assert data["final_answer"] == "Answer 0."


async def test_aggregator_requires_candidates() -> None:
    with pytest.raises(BackendError, match="no candidates"):
        await call(MockBackend(), "aggregator", question=QUESTION, candidates=[])


async def test_unknown_role_raises() -> None:
    request = LLMRequest(role="judge", system="", prompt="")  # type: ignore[arg-type]
    with pytest.raises(BackendError, match="unknown role 'judge'"):
        await MockBackend().complete(request)


@pytest.mark.parametrize(
    ("role", "metadata", "missing"),
    [
        ("proposer", {"agent_id": "a"}, "persona"),
        ("reviser", {"agent_id": "a", "round": 2, "peers": []}, "own_answer"),
        ("critic", {"question": "q"}, "proposal"),
        ("voter", {"candidates": []}, "agent_id"),
        ("aggregator", {"question": "q"}, "candidates"),
    ],
)
async def test_missing_metadata_raises(role: Role, metadata: dict[str, Any], missing: str) -> None:
    with pytest.raises(BackendError, match=f"{role} request is missing metadata {missing}"):
        await call(MockBackend(), role, **metadata)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"convergence": -0.1},
        {"convergence": 1.5},
        {"num_stances": 0},
        {"num_stances": len(STANCES) + 1},
    ],
)
def test_invalid_constructor_args(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        MockBackend(**kwargs)


def test_normalize_answer() -> None:
    assert normalize_answer("  Use\t REDIS!?. ") == "use redis"
    assert normalize_answer("...") == ""

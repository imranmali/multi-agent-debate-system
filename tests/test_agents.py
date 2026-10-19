"""Tests for the four debate agents, driven by a local fake backend."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

import pytest

from debate_system.agents import (
    AggregatorAgent,
    CriticAgent,
    ProposerAgent,
    VoterAgent,
    proposal_id,
)
from debate_system.llm.base import LLMBackend, LLMRequest
from debate_system.models import CRITERIA, Critique, Proposal

Reply = str | Callable[[LLMRequest], str]


class FakeBackend:
    """Records every request and answers from a queue of replies (or a callable)."""

    def __init__(self, *replies: Reply) -> None:
        self.replies = list(replies)
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> str:
        self.requests.append(request)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return reply(request) if callable(reply) else reply

    @property
    def last(self) -> LLMRequest:
        return self.requests[-1]


def make_proposal(agent_id: str, answer: str, round_no: int = 1) -> Proposal:
    return Proposal(
        id=proposal_id(round_no, agent_id),
        agent_id=agent_id,
        persona="analyst",
        round=round_no,
        answer=answer,
    )


def make_critique(proposal: Proposal, overall: float, feedback: str = "") -> Critique:
    scores = dict.fromkeys(CRITERIA, overall * 10)
    return Critique(proposal_id=proposal.id, scores=scores, overall=overall, feedback=feedback)


@pytest.fixture
def proposals() -> list[Proposal]:
    return [
        make_proposal("agent-1", "Use Postgres."),
        make_proposal("agent-2", "Use SQLite."),
        make_proposal("agent-3", "Use MongoDB."),
    ]


@pytest.fixture
def critiques(proposals: list[Proposal]) -> dict[str, Critique]:
    a, b, c = proposals
    return {
        a.id: make_critique(a, 0.6, "Mention backups."),
        b.id: make_critique(b, 0.9, "Solid."),
        c.id: make_critique(c, 0.3, "Weak on transactions."),
    }


def test_fake_backend_satisfies_protocol() -> None:
    assert isinstance(FakeBackend("x"), LLMBackend)


# --- proposer ------------------------------------------------------------------


async def test_propose_parses_json_and_sets_metadata() -> None:
    backend = FakeBackend('{"answer": " Use Postgres. ", "rationale": " ACID. "}')
    agent = ProposerAgent("agent-1", "skeptic", backend)

    proposal = await agent.propose("Which DB?", 1)

    assert proposal == Proposal(
        id="r1-agent-1",
        agent_id="agent-1",
        persona="skeptic",
        round=1,
        answer="Use Postgres.",
        rationale="ACID.",
    )
    request = backend.last
    assert request.role == "proposer"
    assert request.metadata == {
        "agent_id": "agent-1",
        "persona": "skeptic",
        "round": 1,
        "question": "Which DB?",
    }
    assert "Which DB?" in request.prompt
    assert "skeptic" in request.system
    assert '{"answer": "<your answer>", "rationale": "<why this answer is right>"}' in (
        request.system
    )


async def test_propose_accepts_fenced_json_in_prose() -> None:
    backend = FakeBackend('Sure:\n```json\n{"answer": "42", "rationale": 7}\n```')
    proposal = await ProposerAgent("a", "p", backend).propose("q", 1)
    assert (proposal.answer, proposal.rationale) == ("42", "")


@pytest.mark.parametrize(
    ("reply", "answer"),
    [
        ("  Just use Postgres.  ", "Just use Postgres."),
        ('{"answer": "", "rationale": "x"}', '{"answer": "", "rationale": "x"}'),
        ('{"rationale": "no answer"}', '{"rationale": "no answer"}'),
        ('{"answer": 42}', '{"answer": 42}'),
    ],
)
async def test_propose_falls_back_to_raw_text(reply: str, answer: str) -> None:
    proposal = await ProposerAgent("a", "p", FakeBackend(reply)).propose("q", 2)
    assert proposal.answer == answer
    assert proposal.rationale == ""
    assert proposal.id == "r2-a"


@pytest.mark.parametrize("reply", ["", "   \n "])
async def test_propose_empty_reply_raises(reply: str) -> None:
    with pytest.raises(ValueError, match="empty"):
        await ProposerAgent("a", "p", FakeBackend(reply)).propose("q", 1)


async def test_propose_propagates_backend_errors() -> None:
    def boom(request: LLMRequest) -> str:
        raise RuntimeError("down")

    with pytest.raises(RuntimeError, match="down"):
        await ProposerAgent("a", "p", FakeBackend(boom)).propose("q", 1)


# --- reviser -------------------------------------------------------------------


async def test_revise_includes_feedback_and_peer_scores(
    proposals: list[Proposal], critiques: dict[str, Critique]
) -> None:
    backend = FakeBackend('{"answer": "Use Postgres with backups.", "rationale": "fixed"}')
    agent = ProposerAgent("agent-1", "analyst", backend)
    own = proposals[0]

    revised = await agent.revise("Which DB?", 2, own, proposals, critiques)

    assert revised.id == "r2-agent-1"
    assert revised.round == 2
    assert revised.answer == "Use Postgres with backups."
    request = backend.last
    assert request.role == "reviser"
    assert request.metadata == {
        "agent_id": "agent-1",
        "persona": "analyst",
        "round": 2,
        "question": "Which DB?",
        "own_answer": "Use Postgres.",
        "own_feedback": "Mention backups.",
        "peers": [
            {"id": "r1-agent-1", "agent_id": "agent-1", "answer": "Use Postgres.", "score": 0.6},
            {"id": "r1-agent-2", "agent_id": "agent-2", "answer": "Use SQLite.", "score": 0.9},
            {"id": "r1-agent-3", "agent_id": "agent-3", "answer": "Use MongoDB.", "score": 0.3},
        ],
    }
    assert "Mention backups." in request.prompt
    for peer, score in zip(proposals, ("0.60", "0.90", "0.30"), strict=True):
        assert peer.answer in request.prompt
        assert f"[{peer.id}] by {peer.agent_id} (critic score {score})" in request.prompt


async def test_revise_without_critiques_uses_defaults(proposals: list[Proposal]) -> None:
    backend = FakeBackend('{"answer": "Use SQLite."}')
    own = proposals[0]

    await ProposerAgent("agent-1", "analyst", backend).revise("q", 2, own, proposals[1:], {})

    meta = backend.last.metadata
    assert meta["own_feedback"] == ""
    assert [peer["id"] for peer in meta["peers"]] == ["r1-agent-1", "r1-agent-2", "r1-agent-3"]
    assert {peer["score"] for peer in meta["peers"]} == {0.5}
    assert "(no feedback)" in backend.last.prompt


async def test_revise_raw_text_fallback(proposals: list[Proposal]) -> None:
    backend = FakeBackend("I now prefer SQLite.")
    revised = await ProposerAgent("agent-1", "a", backend).revise(
        "q", 3, proposals[0], proposals, {}
    )
    assert (revised.id, revised.answer) == ("r3-agent-1", "I now prefer SQLite.")


# --- critic --------------------------------------------------------------------


def critic_reply(scores: Any, feedback: Any = "Good.") -> str:
    return json.dumps({"scores": scores, "feedback": feedback})


async def test_critique_overall_is_mean_over_ten(proposals: list[Proposal]) -> None:
    scores = {"correctness": 8, "completeness": 6, "clarity": 10, "feasibility": 4}
    backend = FakeBackend(critic_reply(scores, " Add detail. "))

    critique = await CriticAgent(backend).critique("Which DB?", proposals[0])

    assert critique.overall == pytest.approx(0.7)
    assert critique.scores == {k: float(v) for k, v in scores.items()}
    assert critique.feedback == "Add detail."
    assert critique.parsed is True
    assert critique.proposal_id == "r1-agent-1"
    request = backend.last
    assert request.role == "critic"
    assert request.metadata == {
        "question": "Which DB?",
        "proposal": {"id": "r1-agent-1", "agent_id": "agent-1", "answer": "Use Postgres."},
    }
    assert "Use Postgres." in request.prompt
    assert all(criterion in request.system for criterion in CRITERIA)


async def test_critique_clamps_out_of_range_scores(proposals: list[Proposal]) -> None:
    scores = {"correctness": 14, "completeness": -2, "clarity": "7.5", "feasibility": 5}
    critique = await CriticAgent(FakeBackend(critic_reply(scores))).critique("q", proposals[0])
    assert critique.scores == {
        "correctness": 10.0,
        "completeness": 0.0,
        "clarity": 7.5,
        "feasibility": 5.0,
    }
    assert critique.overall == pytest.approx(0.5625)


async def test_critique_missing_or_bad_criterion_is_neutral(proposals: list[Proposal]) -> None:
    scores = {"correctness": 9, "clarity": "great", "feasibility": True}
    critique = await CriticAgent(FakeBackend(critic_reply(scores, 3))).critique("q", proposals[0])
    assert critique.scores == {
        "correctness": 9.0,
        "completeness": 5.0,
        "clarity": 5.0,
        "feasibility": 5.0,
    }
    assert critique.parsed is True
    assert critique.feedback == ""


@pytest.mark.parametrize(
    "reply",
    [
        "This answer is decent but lacks detail.",
        critic_reply("eight"),
        critic_reply({"correctness": "n/a", "clarity": None}),
        '{"feedback": "no scores at all"}',
    ],
)
async def test_critique_unparseable_is_neutral_and_flagged(
    reply: str, proposals: list[Proposal]
) -> None:
    critique = await CriticAgent(FakeBackend(reply)).critique("q", proposals[0])
    assert critique.overall == 0.5
    assert critique.scores == dict.fromkeys(CRITERIA, 5.0)
    assert critique.parsed is False
    assert critique.feedback == reply.strip()


async def test_critique_unparseable_feedback_is_truncated(proposals: list[Proposal]) -> None:
    critique = await CriticAgent(FakeBackend("x" * 900)).critique("q", proposals[0])
    assert critique.feedback == "x" * 500


# --- voter ---------------------------------------------------------------------


async def test_vote_excludes_own_proposal(
    proposals: list[Proposal], critiques: dict[str, Critique]
) -> None:
    backend = FakeBackend('{"ranking": ["r1-agent-1", "r1-agent-3", "r1-agent-2"]}')
    ballot = await VoterAgent("agent-1", backend).vote("Which DB?", proposals, critiques)

    assert ballot.voter_id == "agent-1"
    assert ballot.ranking == ["r1-agent-3", "r1-agent-2"]
    assert ballot.weight == 1.0
    request = backend.last
    assert request.role == "voter"
    assert request.metadata == {
        "agent_id": "agent-1",
        "question": "Which DB?",
        "round": 1,
        "candidates": [
            {"id": "r1-agent-2", "agent_id": "agent-2", "answer": "Use SQLite.", "score": 0.9},
            {"id": "r1-agent-3", "agent_id": "agent-3", "answer": "Use MongoDB.", "score": 0.3},
        ],
    }
    assert "r1-agent-1" not in request.prompt
    assert "Use SQLite." in request.prompt


async def test_vote_allows_self_when_enabled(
    proposals: list[Proposal], critiques: dict[str, Critique]
) -> None:
    backend = FakeBackend('{"ranking": ["r1-agent-1"]}')
    ballot = await VoterAgent("agent-1", backend, allow_self_vote=True).vote(
        "q", proposals, critiques
    )
    assert ballot.ranking == ["r1-agent-1", "r1-agent-2", "r1-agent-3"]
    assert len(backend.last.metadata["candidates"]) == 3


async def test_vote_filters_dedupes_and_appends_by_score(
    proposals: list[Proposal], critiques: dict[str, Critique]
) -> None:
    extra = make_proposal("agent-4", "Use Redis.")
    pool = [*proposals, extra]
    reply = '{"ranking": ["bogus", "r1-agent-3", 7, "r1-agent-3", "r1-agent-4"]}'

    ballot = await VoterAgent("voter", FakeBackend(reply)).vote("q", pool, critiques)

    # agent-1 (0.6) and agent-2 (0.9) were unranked: appended by critic score.
    assert ballot.ranking == ["r1-agent-3", "r1-agent-4", "r1-agent-2", "r1-agent-1"]


@pytest.mark.parametrize("reply", ["I like SQLite best.", '{"ranking": "r1-agent-2"}'])
async def test_vote_unparseable_ranks_by_score(
    reply: str, proposals: list[Proposal], critiques: dict[str, Critique]
) -> None:
    ballot = await VoterAgent("voter", FakeBackend(reply)).vote("q", proposals, critiques)
    assert ballot.ranking == ["r1-agent-2", "r1-agent-1", "r1-agent-3"]


async def test_vote_score_ties_keep_original_order(proposals: list[Proposal]) -> None:
    ballot = await VoterAgent("voter", FakeBackend("?")).vote("q", proposals, {})
    assert ballot.ranking == [p.id for p in proposals]


@pytest.mark.parametrize("pool", [[], [make_proposal("agent-1", "mine")]])
async def test_vote_without_candidates_skips_backend(pool: Sequence[Proposal]) -> None:
    backend = FakeBackend("unused")
    ballot = await VoterAgent("agent-1", backend).vote("q", pool, {})
    assert ballot.ranking == []
    assert ballot.voter_id == "agent-1"
    assert backend.requests == []


# --- aggregator ----------------------------------------------------------------


async def test_aggregate_uses_model_answer(
    proposals: list[Proposal], critiques: dict[str, Critique]
) -> None:
    backend = FakeBackend('{"final_answer": " Use SQLite, add backups. ", "summary": " Agreed. "}')

    result = await AggregatorAgent(backend).aggregate(
        "Which DB?", proposals, critiques, "r1-agent-2"
    )

    assert result == ("Use SQLite, add backups.", "Agreed.")
    request = backend.last
    assert request.role == "aggregator"
    assert request.metadata == {
        "question": "Which DB?",
        "candidates": [
            {"id": "r1-agent-1", "agent_id": "agent-1", "answer": "Use Postgres.", "score": 0.6},
            {"id": "r1-agent-2", "agent_id": "agent-2", "answer": "Use SQLite.", "score": 0.9},
            {"id": "r1-agent-3", "agent_id": "agent-3", "answer": "Use MongoDB.", "score": 0.3},
        ],
        "winner_id": "r1-agent-2",
    }
    assert "Winning candidate: r1-agent-2" in request.prompt
    assert "Weak on transactions." in request.prompt
    assert all(p.answer in request.prompt for p in proposals)


async def test_aggregate_missing_summary_is_empty(proposals: list[Proposal]) -> None:
    backend = FakeBackend('{"final_answer": "Use SQLite."}')
    result = await AggregatorAgent(backend).aggregate("q", proposals, {}, "r1-agent-2")
    assert result == ("Use SQLite.", "")


async def test_aggregate_plain_text_becomes_answer(proposals: list[Proposal]) -> None:
    backend = FakeBackend("  Use SQLite for now.  ")
    result = await AggregatorAgent(backend).aggregate("q", proposals, {}, "r1-agent-2")
    assert result == ("Use SQLite for now.", "")


@pytest.mark.parametrize(
    "reply", ["", "   ", '{"final_answer": ""}', '{"summary": "only"}', '{"final_answer": 1}']
)
async def test_aggregate_falls_back_to_winner(reply: str, proposals: list[Proposal]) -> None:
    backend = FakeBackend(reply)
    result = await AggregatorAgent(backend).aggregate("q", proposals, {}, "r1-agent-3")
    assert result == ("Use MongoDB.", "")


async def test_aggregate_unknown_winner_raises(proposals: list[Proposal]) -> None:
    backend = FakeBackend("unused")
    with pytest.raises(ValueError, match="r9-nobody"):
        await AggregatorAgent(backend).aggregate("q", proposals, {}, "r9-nobody")
    assert backend.requests == []


# --- prompts -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("make_call", "shape"),
    [
        (lambda b: ProposerAgent("a", "p", b).propose("q", 1), '"answer"'),
        (lambda b: CriticAgent(b).critique("q", make_proposal("a", "x")), '"scores"'),
        (lambda b: VoterAgent("v", b).vote("q", [make_proposal("a", "x")], {}), '"ranking"'),
        (
            lambda b: AggregatorAgent(b).aggregate("q", [make_proposal("a", "x")], {}, "r1-a"),
            '"final_answer"',
        ),
    ],
)
async def test_system_prompts_demand_json_only(make_call: Any, shape: str) -> None:
    backend = FakeBackend('{"answer": "x"}')
    await make_call(backend)
    system = backend.last.system
    assert "ONLY a single JSON object" in system
    assert shape in system
    assert "{{" not in system

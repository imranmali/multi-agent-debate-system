import asyncio
import json
import time
from typing import Any

import pytest

from debate_system.config import DebateConfig
from debate_system.llm import LLMRequest, MockBackend, ScriptedBackend
from debate_system.orchestrator import DebateError, DebateOrchestrator

SCORES = json.dumps(
    {
        "scores": {"correctness": 8, "completeness": 8, "clarity": 8, "feasibility": 8},
        "feedback": "Add a migration plan.",
    }
)


def rank_in_order(request: LLMRequest) -> str:
    return json.dumps({"ranking": [c["id"] for c in request.metadata["candidates"]]})


def rank_next_agent(request: LLMRequest) -> str:
    """Each voter puts the 'next' agent first, so votes never concentrate."""
    me = int(request.metadata["agent_id"].split("-")[1])
    candidates = request.metadata["candidates"]
    n = len(candidates) + 1

    def distance(c: dict[str, Any]) -> int:
        return (int(c["agent_id"].split("-")[1]) - me) % n

    return json.dumps({"ranking": [c["id"] for c in sorted(candidates, key=distance)]})


def unanimous_backend(**kwargs: Any) -> ScriptedBackend:
    return ScriptedBackend(
        {
            "proposer": [json.dumps({"answer": "Use Postgres.", "rationale": "boring tech"})],
            "reviser": [json.dumps({"answer": "Use Postgres.", "rationale": "still"})],
            "critic": [SCORES],
            "voter": rank_in_order,
            "aggregator": [json.dumps({"final_answer": "Use Postgres.", "summary": "all agree"})],
        },
        **kwargs,
    )


def split_backend() -> ScriptedBackend:
    def propose(request: LLMRequest) -> str:
        return json.dumps({"answer": f"Option {request.metadata['agent_id']}"})

    def keep(request: LLMRequest) -> str:
        return json.dumps({"answer": request.metadata["own_answer"]})

    return ScriptedBackend(
        {
            "proposer": propose,
            "reviser": keep,
            "critic": [SCORES],
            "voter": rank_next_agent,
            "aggregator": [json.dumps({"final_answer": "Option agent-1", "summary": ""})],
        }
    )


async def test_every_agent_proposes_with_a_distinct_persona() -> None:
    result = await DebateOrchestrator(MockBackend(seed=1), DebateConfig(num_agents=4)).run("Q?")
    first = result.rounds[0]
    assert [p.agent_id for p in first.proposals] == ["agent-1", "agent-2", "agent-3", "agent-4"]
    assert len({p.persona for p in first.proposals}) == 4
    assert len(first.critiques) == 4


async def test_agent_calls_run_concurrently() -> None:
    backend = unanimous_backend(delay=0.1)
    config = DebateConfig(num_agents=4, max_rounds=1)
    start = time.perf_counter()
    await DebateOrchestrator(backend, config).run("Q?")
    elapsed = time.perf_counter() - start
    # 4 phases (propose, critique, vote, aggregate) of 0.1 s each; sequential would be 1.3 s.
    assert elapsed < 0.8
    assert len(backend.calls) == 4 + 4 + 4 + 1


async def test_consensus_in_round_one_stops_early() -> None:
    result = await DebateOrchestrator(unanimous_backend(), DebateConfig(max_rounds=3)).run("Q?")
    assert result.rounds_run == 1
    assert result.consensus_reached
    assert result.final_answer == "Use Postgres."
    assert result.summary == "all agree"
    assert result.confidence.score == pytest.approx(0.4 + 0.3 * 0.8 + 0.2 + 0.1)


async def test_no_consensus_runs_all_rounds() -> None:
    result = await DebateOrchestrator(split_backend(), DebateConfig(max_rounds=3)).run("Q?")
    assert result.rounds_run == 3
    assert not result.consensus_reached
    assert all(r.vote is not None and r.vote.support < 0.6 for r in result.rounds)


async def test_revisers_see_peers_scores_and_feedback() -> None:
    backend = split_backend()
    await DebateOrchestrator(backend, DebateConfig(max_rounds=2)).run("Q?")
    revisions = [c for c in backend.calls if c.role == "reviser"]
    assert len(revisions) == 3
    for request in revisions:
        assert {p["agent_id"] for p in request.metadata["peers"]} == {
            "agent-1",
            "agent-2",
            "agent-3",
        }
        assert all(p["score"] == pytest.approx(0.8) for p in request.metadata["peers"])
        assert "Add a migration plan." in request.prompt


async def test_one_failing_proposer_is_recorded_and_debate_continues() -> None:
    def propose(request: LLMRequest) -> str:
        if request.metadata["agent_id"] == "agent-2":
            raise RuntimeError("provider exploded")
        return json.dumps({"answer": "Use Postgres."})

    backend = ScriptedBackend(
        {
            "proposer": propose,
            "critic": [SCORES],
            "voter": rank_in_order,
            "aggregator": [json.dumps({"final_answer": "Use Postgres."})],
        }
    )
    result = await DebateOrchestrator(backend, DebateConfig(max_rounds=1)).run("Q?")
    first = result.rounds[0]
    assert len(first.proposals) == 2
    assert [(e.agent_id, e.role) for e in first.errors] == [("agent-2", "proposer")]
    assert "provider exploded" in first.errors[0].message


async def test_all_proposers_failing_raises() -> None:
    backend = ScriptedBackend({"proposer": [RuntimeError("down")]})
    with pytest.raises(DebateError, match="round 1"):
        await DebateOrchestrator(backend, DebateConfig()).run("Q?")


class SlowAgentBackend(MockBackend):
    """Mock swarm in which one agent's proposals never arrive in time."""

    def __init__(self, slow_agent: str) -> None:
        super().__init__(seed=3)
        self.slow_agent = slow_agent

    async def complete(self, request: LLMRequest) -> str:
        if request.role == "proposer" and request.metadata["agent_id"] == self.slow_agent:
            await asyncio.sleep(5)
        return await super().complete(request)


async def test_timeouts_are_recorded_as_errors() -> None:
    config = DebateConfig(num_agents=3, max_rounds=1, timeout_seconds=0.05)
    result = await DebateOrchestrator(SlowAgentBackend("agent-3"), config).run("Q?")
    errors = result.rounds[0].errors
    assert [(e.agent_id, e.message) for e in errors] == [("agent-3", "timed out")]
    assert len(result.rounds[0].proposals) == 2


async def test_agent_that_failed_proposes_fresh_next_round() -> None:
    def propose(request: LLMRequest) -> str:
        if request.metadata["agent_id"] == "agent-3" and request.metadata["round"] == 1:
            raise RuntimeError("flaky")
        return json.dumps({"answer": f"Option {request.metadata['agent_id']}"})

    backend = ScriptedBackend(
        {
            "proposer": propose,
            "reviser": lambda r: json.dumps({"answer": r.metadata["own_answer"]}),
            "critic": [SCORES],
            "voter": rank_next_agent,
            "aggregator": [json.dumps({"final_answer": "x"})],
        }
    )
    result = await DebateOrchestrator(backend, DebateConfig(max_rounds=2)).run("Q?")
    assert [p.agent_id for p in result.rounds[1].proposals] == ["agent-1", "agent-2", "agent-3"]
    assert result.rounds[1].proposals[2].id == "r2-agent-3"


async def test_aggregator_failure_falls_back_to_winner() -> None:
    backend = ScriptedBackend(
        {
            "proposer": [json.dumps({"answer": "Use Postgres."})],
            "critic": [SCORES],
            "voter": rank_in_order,
            "aggregator": [RuntimeError("aggregator down")],
        }
    )
    result = await DebateOrchestrator(backend, DebateConfig(max_rounds=1)).run("Q?")
    assert result.final_answer == "Use Postgres."
    assert result.summary == ""
    assert result.rounds[-1].errors[-1].role == "aggregator"


async def test_ballots_carry_voter_weights() -> None:
    result = await DebateOrchestrator(unanimous_backend(), DebateConfig()).run("Q?")
    assert all(b.weight == pytest.approx(0.8) for b in result.rounds[0].ballots)


async def test_events_are_emitted() -> None:
    events: list[str] = []
    orchestrator = DebateOrchestrator(
        unanimous_backend(), DebateConfig(), on_event=lambda name, _: events.append(name)
    )
    await orchestrator.run("Q?")
    assert events == ["round_started", "round_completed", "debate_completed"]


async def test_empty_question_rejected() -> None:
    with pytest.raises(ValueError):
        await DebateOrchestrator(MockBackend(), DebateConfig()).run("   ")


async def test_mock_debates_are_deterministic_per_seed() -> None:
    config = DebateConfig(num_agents=4, max_rounds=3)
    a = await DebateOrchestrator(MockBackend(seed=7), config).run("Pick a cache strategy")
    b = await DebateOrchestrator(MockBackend(seed=7), config).run("Pick a cache strategy")
    assert a.to_dict() == b.to_dict()
    transcripts = set()
    for seed in range(6):
        result = await DebateOrchestrator(MockBackend(seed=seed), config).run("Pick a cache")
        transcripts.add(json.dumps(result.to_dict(), sort_keys=True))
    assert len(transcripts) >= 2


def test_result_is_json_serializable_and_run_sync_works() -> None:
    result = DebateOrchestrator(MockBackend(seed=2), DebateConfig()).run_sync("Q?")
    data = json.loads(json.dumps(result.to_dict()))
    for key in (
        "question",
        "final_answer",
        "confidence",
        "consensus_reached",
        "rounds_run",
        "rounds",
    ):
        assert key in data
    assert 0.0 <= data["confidence"]["score"] <= 1.0
    assert data["rounds_run"] == len(data["rounds"])

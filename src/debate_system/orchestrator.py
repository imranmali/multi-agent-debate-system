"""The debate round loop: propose -> critique -> vote -> (revise ...) -> aggregate."""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, TypeVar

from debate_system.agents import AggregatorAgent, CriticAgent, ProposerAgent, VoterAgent
from debate_system.confidence import compute_confidence
from debate_system.config import DebateConfig
from debate_system.consensus import tally, voter_weights
from debate_system.llm.base import LLMBackend
from debate_system.models import AgentError, DebateResult, Proposal, RoundRecord, VoteResult

T = TypeVar("T")

#: Optional progress callback: ``on_event(name, payload)``.
EventHandler = Callable[[str, dict[str, Any]], None]


class DebateError(RuntimeError):
    """Raised when a debate cannot continue (e.g. no proposal survived a round)."""


class DebateOrchestrator:
    """Runs a swarm of proposer agents, a critic and an aggregator over one backend."""

    def __init__(
        self,
        backend: LLMBackend,
        config: DebateConfig | None = None,
        *,
        on_event: EventHandler | None = None,
    ) -> None:
        self.backend = backend
        self.config = (config or DebateConfig()).validate()
        self._on_event = on_event
        ids = self.config.agent_ids()
        self.proposers = [
            ProposerAgent(agent_id, self.config.persona_for(i), backend)
            for i, agent_id in enumerate(ids)
        ]
        self.voters = {
            agent_id: VoterAgent(agent_id, backend, allow_self_vote=self.config.allow_self_vote)
            for agent_id in ids
        }
        self.critic = CriticAgent(backend)
        self.aggregator = AggregatorAgent(backend)

    def run_sync(self, question: str) -> DebateResult:
        """Blocking wrapper around :meth:`run`."""
        return asyncio.run(self.run(question))

    async def run(self, question: str) -> DebateResult:
        """Debate ``question`` until consensus or ``max_rounds`` and return the result."""
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")

        rounds: list[RoundRecord] = []
        previous: RoundRecord | None = None
        for round_no in range(1, self.config.max_rounds + 1):
            record = await self._run_round(question, round_no, previous)
            rounds.append(record)
            self._emit("round_completed", record=record)
            assert record.vote is not None
            if record.vote.consensus:
                break
            previous = record

        final = rounds[-1]
        vote = final.vote
        assert vote is not None
        final_answer, summary = await self._aggregate(question, final, vote)
        confidence = compute_confidence(rounds, self.config.confidence_weights)
        result = DebateResult(
            question=question,
            final_answer=final_answer,
            summary=summary,
            winner_proposal_id=vote.winner_proposal_id,
            confidence=confidence,
            consensus_reached=vote.consensus,
            rounds_run=len(rounds),
            rounds=rounds,
            config=self.config.to_dict(),
        )
        self._emit("debate_completed", result=result)
        return result

    async def _run_round(
        self, question: str, round_no: int, previous: RoundRecord | None
    ) -> RoundRecord:
        record = RoundRecord(round=round_no)
        self._emit("round_started", round=round_no)

        record.proposals = await self._gather(
            record,
            "proposer" if previous is None else "reviser",
            [
                (agent.agent_id, self._propose_or_revise(agent, question, round_no, previous))
                for agent in self.proposers
            ],
        )
        if not record.proposals:
            raise DebateError(f"no proposal survived round {round_no}")

        critiques = await self._gather(
            record,
            "critic",
            [
                (self.critic.agent_id, self.critic.critique(question, proposal))
                for proposal in record.proposals
            ],
        )
        record.critiques = critiques
        by_id = {c.proposal_id: c for c in critiques}

        weights = voter_weights(record.proposals, by_id)
        ballots = await self._gather(
            record,
            "voter",
            [
                (p.agent_id, self.voters[p.agent_id].vote(question, record.proposals, by_id))
                for p in record.proposals
            ],
        )
        record.ballots = [
            dataclasses.replace(b, weight=weights.get(b.voter_id, 1.0)) for b in ballots
        ]
        record.vote = tally(
            record.proposals,
            by_id,
            record.ballots,
            self.config.voting_strategy,
            self.config.consensus_threshold,
        )
        return record

    def _propose_or_revise(
        self,
        agent: ProposerAgent,
        question: str,
        round_no: int,
        previous: RoundRecord | None,
    ) -> Awaitable[Proposal]:
        own = None if previous is None else _latest(previous.proposals, agent.agent_id)
        if previous is None or own is None:
            # First round, or the agent failed last round: start from a fresh proposal.
            return agent.propose(question, round_no)
        critiques = {c.proposal_id: c for c in previous.critiques}
        return agent.revise(question, round_no, own, previous.proposals, critiques)

    async def _aggregate(
        self, question: str, record: RoundRecord, vote: VoteResult
    ) -> tuple[str, str]:
        critiques = {c.proposal_id: c for c in record.critiques}
        try:
            return await asyncio.wait_for(
                self.aggregator.aggregate(
                    question, record.proposals, critiques, vote.winner_proposal_id
                ),
                timeout=self.config.timeout_seconds,
            )
        except Exception as exc:  # fall back to the winning proposal
            record.errors.append(_error(self.aggregator.agent_id, "aggregator", exc))
            winner = next(p for p in record.proposals if p.id == vote.winner_proposal_id)
            return winner.answer, ""

    async def _gather(
        self,
        record: RoundRecord,
        role: str,
        calls: Sequence[tuple[str, Awaitable[T]]],
    ) -> list[T]:
        """Run ``calls`` concurrently with a timeout; record failures, keep order."""
        results = await asyncio.gather(
            *(asyncio.wait_for(call, timeout=self.config.timeout_seconds) for _, call in calls),
            return_exceptions=True,
        )
        values: list[T] = []
        for (agent_id, _), outcome in zip(calls, results, strict=True):
            if isinstance(outcome, BaseException):
                if not isinstance(outcome, Exception):
                    raise outcome  # pragma: no cover - never swallow cancellation
                error = _error(agent_id, role, outcome)
                record.errors.append(error)
                self._emit("agent_failed", error=error)
            else:
                values.append(outcome)
        return values

    def _emit(self, name: str, **payload: Any) -> None:
        if self._on_event is not None:
            self._on_event(name, payload)


def _latest(proposals: Sequence[Proposal], agent_id: str) -> Proposal | None:
    for proposal in reversed(proposals):
        if proposal.agent_id == agent_id:
            return proposal
    return None


def _error(agent_id: str, role: str, exc: Exception) -> AgentError:
    if isinstance(exc, asyncio.TimeoutError):
        return AgentError(agent_id=agent_id, role=role, message="timed out")
    return AgentError(agent_id=agent_id, role=role, message=f"{type(exc).__name__}: {exc}")

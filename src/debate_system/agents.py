"""The four debate roles: proposer, critic, voter and aggregator.

Each agent turns debate state into an :class:`~debate_system.llm.base.LLMRequest`
(prompt text for real models, ``metadata`` for the offline backend - keys per
design.md - Backend contract) and parses the reply tolerantly, falling back to
safe defaults when the model does not return the requested JSON. Backend
exceptions propagate; the orchestrator records them as agent errors.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from debate_system.llm.base import LLMBackend, LLMRequest
from debate_system.models import CRITERIA, Ballot, Critique, Proposal
from debate_system.parsing import clamp, extract_json, parse_float

#: Score assumed for a proposal the critic has not (successfully) scored, in [0, 1].
DEFAULT_SCORE = 0.5
#: Criterion score substituted when the critic omits or garbles one, in [0, 10].
NEUTRAL_CRITERION_SCORE = 5.0
#: Longest raw critic reply kept as feedback when the reply cannot be parsed.
MAX_RAW_FEEDBACK = 500

_JSON_ONLY = (
    "Reply with ONLY a single JSON object, no Markdown fences and no text before or after it."
)

PROPOSER_SYSTEM = (
    "You are one member of a panel of expert agents debating a question to reach the best "
    "possible answer. Your perspective is that of a {persona}: let it shape what you notice "
    "and emphasize, but never at the expense of being correct.\n"
    "Give a direct, self-contained answer and a brief rationale explaining why it is right.\n"
    f"{_JSON_ONLY} Use exactly this shape:\n"
    '{{"answer": "<your answer>", "rationale": "<why this answer is right>"}}'
)

REVISER_SYSTEM = (
    "You are one member of a panel of expert agents debating a question, now in a revision "
    "round. Your perspective is that of a {persona}.\n"
    "You will see your previous answer, the critic's feedback on it, and every panel answer "
    "from the last round with its critic score (0-1). Weigh them on their merits: keep your "
    "answer if it is still the best, fix the weaknesses the critic raised, or adopt a better "
    "answer from a peer if one convinces you. Do not change your answer merely to conform.\n"
    f"{_JSON_ONLY} Use exactly this shape:\n"
    '{{"answer": "<your revised answer>", "rationale": "<what you kept or changed, and why>"}}'
)

CRITIC_SYSTEM = (
    "You are a rigorous, impartial critic evaluating one proposed answer to a question.\n"
    "Score the answer from 0 (worst) to 10 (best) on each criterion:\n"
    "- correctness: is it factually and logically right?\n"
    "- completeness: does it address every part of the question?\n"
    "- clarity: is it clear, precise and unambiguous?\n"
    "- feasibility: is it practical and actionable as stated?\n"
    "Then give concise, specific feedback the author can act on to improve the answer.\n"
    f"{_JSON_ONLY} Use exactly this shape, with numeric scores:\n"
    '{"scores": {"correctness": 0, "completeness": 0, "clarity": 0, "feasibility": 0}, '
    '"feedback": "<actionable feedback>"}'
)

VOTER_SYSTEM = (
    "You are a member of a panel of expert agents voting on candidate answers to a question.\n"
    "Rank every candidate from best to worst by how well it answers the question. Critic "
    "scores (0-1) are provided as a guide, but use your own judgement.\n"
    f"{_JSON_ONLY} Use exactly this shape, listing candidate ids best first:\n"
    '{"ranking": ["<best candidate id>", "<next candidate id>", "..."]}'
)

AGGREGATOR_SYSTEM = (
    "You are the moderator of a panel of expert agents that has finished debating a question.\n"
    "Write the single strongest final answer. Anchor it on the winning candidate chosen by "
    "the panel's vote, and incorporate any correct, valuable points from the other candidates "
    "and the critic's feedback; drop anything incorrect or contradictory.\n"
    "Also write a short summary of how the panel reached this answer and where it disagreed.\n"
    f"{_JSON_ONLY} Use exactly this shape:\n"
    '{"final_answer": "<the final answer>", "summary": "<how the debate reached it>"}'
)


def proposal_id(round_no: int, agent_id: str) -> str:
    """Id of ``agent_id``'s proposal in round ``round_no`` (``r{round}-{agent_id}``)."""
    return f"r{round_no}-{agent_id}"


def _score(proposal: Proposal, critiques: Mapping[str, Critique]) -> float:
    critique = critiques.get(proposal.id)
    return critique.overall if critique is not None else DEFAULT_SCORE


def _candidate(proposal: Proposal, critiques: Mapping[str, Critique]) -> dict[str, Any]:
    return {
        "id": proposal.id,
        "agent_id": proposal.agent_id,
        "answer": proposal.answer,
        "score": _score(proposal, critiques),
    }


def _format_candidates(
    proposals: Sequence[Proposal],
    critiques: Mapping[str, Critique],
    *,
    with_feedback: bool = False,
) -> str:
    blocks = []
    for proposal in proposals:
        lines = [
            f"[{proposal.id}] by {proposal.agent_id} "
            f"(critic score {_score(proposal, critiques):.2f})",
            f"Answer: {proposal.answer}",
        ]
        critique = critiques.get(proposal.id)
        if with_feedback and critique is not None and critique.feedback:
            lines.append(f"Critic feedback: {critique.feedback}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _parse_answer(raw: str) -> tuple[str, str]:
    """``(answer, rationale)`` from a proposer/reviser reply; raw text if not JSON."""
    data = extract_json(raw)
    if data is not None:
        answer = data.get("answer")
        if isinstance(answer, str) and answer.strip():
            rationale = data.get("rationale")
            return answer.strip(), rationale.strip() if isinstance(rationale, str) else ""
    answer = raw.strip()
    if not answer:
        raise ValueError("model returned an empty answer")
    return answer, ""


class ProposerAgent:
    """Answers the question in round 1 and revises its answer in later rounds."""

    def __init__(self, agent_id: str, persona: str, backend: LLMBackend) -> None:
        self.agent_id = agent_id
        self.persona = persona
        self.backend = backend

    def _metadata(self, question: str, round_no: int) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "persona": self.persona,
            "round": round_no,
            "question": question,
        }

    async def _ask(self, request: LLMRequest, round_no: int) -> Proposal:
        answer, rationale = _parse_answer(await self.backend.complete(request))
        return Proposal(
            id=proposal_id(round_no, self.agent_id),
            agent_id=self.agent_id,
            persona=self.persona,
            round=round_no,
            answer=answer,
            rationale=rationale,
        )

    async def propose(self, question: str, round_no: int) -> Proposal:
        """Produce an initial answer. Raises ValueError if the reply is empty."""
        prompt = f"Question:\n{question}\n\nGive your answer as JSON."
        request = LLMRequest(
            role="proposer",
            system=PROPOSER_SYSTEM.format(persona=self.persona),
            prompt=prompt,
            metadata=self._metadata(question, round_no),
        )
        return await self._ask(request, round_no)

    async def revise(
        self,
        question: str,
        round_no: int,
        own: Proposal,
        peers: Sequence[Proposal],
        critiques: Mapping[str, Critique],
    ) -> Proposal:
        """Revise ``own`` after seeing every previous-round answer and its critique.

        ``peers`` is the previous round's proposals; ``own`` is included in the
        request even if it is missing from ``peers``. Raises ValueError if the
        reply is empty.
        """
        everyone = list(peers)
        if all(peer.id != own.id for peer in everyone):
            everyone.insert(0, own)
        own_critique = critiques.get(own.id)
        own_feedback = own_critique.feedback if own_critique is not None else ""
        meta = self._metadata(question, round_no)
        meta["own_answer"] = own.answer
        meta["own_feedback"] = own_feedback
        meta["peers"] = [_candidate(peer, critiques) for peer in everyone]
        prompt = (
            f"Question:\n{question}\n\n"
            f"Your previous answer ({own.id}, critic score {_score(own, critiques):.2f}):\n"
            f"{own.answer}\n\n"
            f"Critic feedback on your answer:\n{own_feedback or '(no feedback)'}\n\n"
            f"All answers from the previous round:\n\n{_format_candidates(everyone, critiques)}"
            "\n\nGive your revised answer as JSON."
        )
        request = LLMRequest(
            role="reviser",
            system=REVISER_SYSTEM.format(persona=self.persona),
            prompt=prompt,
            metadata=meta,
        )
        return await self._ask(request, round_no)


class CriticAgent:
    """Scores a proposal on the rubric in :data:`~debate_system.models.CRITERIA`."""

    def __init__(self, backend: LLMBackend, agent_id: str = "critic") -> None:
        self.agent_id = agent_id
        self.backend = backend

    async def critique(self, question: str, proposal: Proposal) -> Critique:
        """Score ``proposal``; neutral scores and ``parsed=False`` if the reply has none."""
        prompt = (
            f"Question:\n{question}\n\n"
            f"Proposed answer ({proposal.id}):\n{proposal.answer}\n\n"
            "Score this answer and give feedback as JSON."
        )
        request = LLMRequest(
            role="critic",
            system=CRITIC_SYSTEM,
            prompt=prompt,
            metadata={
                "question": question,
                "proposal": {
                    "id": proposal.id,
                    "agent_id": proposal.agent_id,
                    "answer": proposal.answer,
                },
            },
        )
        raw = await self.backend.complete(request)
        data = extract_json(raw) or {}
        raw_scores = data.get("scores")
        parsed_scores: dict[str, float | None] = {
            criterion: parse_float(raw_scores.get(criterion))
            if isinstance(raw_scores, dict)
            else None
            for criterion in CRITERIA
        }
        parsed = any(value is not None for value in parsed_scores.values())
        scores = {
            criterion: NEUTRAL_CRITERION_SCORE if value is None else clamp(value, 0.0, 10.0)
            for criterion, value in parsed_scores.items()
        }
        if parsed:
            feedback = data.get("feedback")
            feedback = feedback.strip() if isinstance(feedback, str) else ""
        else:
            feedback = raw.strip()[:MAX_RAW_FEEDBACK]
        overall = sum(scores.values()) / len(scores) / 10.0
        return Critique(
            proposal_id=proposal.id,
            scores=scores,
            overall=overall,
            feedback=feedback,
            parsed=parsed,
        )


class VoterAgent:
    """Ranks a round's proposals, best first, excluding its own unless allowed."""

    def __init__(self, agent_id: str, backend: LLMBackend, allow_self_vote: bool = False) -> None:
        self.agent_id = agent_id
        self.backend = backend
        self.allow_self_vote = allow_self_vote

    async def vote(
        self,
        question: str,
        proposals: Sequence[Proposal],
        critiques: Mapping[str, Critique],
    ) -> Ballot:
        """Return a complete ranking of the eligible candidates (weight 1.0).

        Unknown and duplicate ids are dropped and unranked candidates appended by
        critic score. With no eligible candidates the backend is not called.
        """
        candidates = [p for p in proposals if self.allow_self_vote or p.agent_id != self.agent_id]
        if not candidates:
            return Ballot(voter_id=self.agent_id, ranking=[])
        prompt = (
            f"Question:\n{question}\n\n"
            f"Candidates:\n\n{_format_candidates(candidates, critiques)}\n\n"
            "Rank all candidates by id, best first, as JSON."
        )
        request = LLMRequest(
            role="voter",
            system=VOTER_SYSTEM,
            prompt=prompt,
            metadata={
                "agent_id": self.agent_id,
                "question": question,
                "round": proposals[0].round,
                "candidates": [_candidate(p, critiques) for p in candidates],
            },
        )
        data = extract_json(await self.backend.complete(request)) or {}
        raw_ranking = data.get("ranking")
        known = {p.id for p in candidates}
        ranking: list[str] = []
        if isinstance(raw_ranking, list):
            for item in raw_ranking:
                if isinstance(item, str) and item in known and item not in ranking:
                    ranking.append(item)
        by_score = sorted(candidates, key=lambda p: _score(p, critiques), reverse=True)
        ranking.extend(p.id for p in by_score if p.id not in ranking)
        return Ballot(voter_id=self.agent_id, ranking=ranking)


class AggregatorAgent:
    """Synthesizes the final answer, anchored on the winning proposal."""

    def __init__(self, backend: LLMBackend, agent_id: str = "aggregator") -> None:
        self.agent_id = agent_id
        self.backend = backend

    async def aggregate(
        self,
        question: str,
        proposals: Sequence[Proposal],
        critiques: Mapping[str, Critique],
        winner_proposal_id: str,
    ) -> tuple[str, str]:
        """Return ``(final_answer, summary)``.

        Falls back to the reply text if it is not JSON, else to the winner's
        answer with an empty summary. Raises ValueError if the winner is unknown.
        """
        winner = next((p for p in proposals if p.id == winner_proposal_id), None)
        if winner is None:
            raise ValueError(f"winner {winner_proposal_id!r} is not among the proposals")
        prompt = (
            f"Question:\n{question}\n\n"
            f"Winning candidate: {winner.id}\n\n"
            f"Candidates:\n\n{_format_candidates(proposals, critiques, with_feedback=True)}"
            "\n\nWrite the final answer and summary as JSON."
        )
        request = LLMRequest(
            role="aggregator",
            system=AGGREGATOR_SYSTEM,
            prompt=prompt,
            metadata={
                "question": question,
                "candidates": [_candidate(p, critiques) for p in proposals],
                "winner_id": winner.id,
            },
        )
        raw = await self.backend.complete(request)
        data = extract_json(raw)
        if data is None:
            text = raw.strip()
            return (text, "") if text else (winner.answer, "")
        final = data.get("final_answer")
        if isinstance(final, str) and final.strip():
            summary = data.get("summary")
            return final.strip(), summary.strip() if isinstance(summary, str) else ""
        return winner.answer, ""

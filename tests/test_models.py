import json

from debate_system.models import (
    Confidence,
    ConfidenceComponent,
    Critique,
    DebateResult,
    Proposal,
    RoundRecord,
)


def test_debate_result_serializes_to_json() -> None:
    proposal = Proposal(id="r1-agent-1", agent_id="agent-1", persona="analyst", round=1, answer="A")
    critique = Critique(proposal_id=proposal.id, scores={"correctness": 8.0}, overall=0.8)
    result = DebateResult(
        question="Q?",
        final_answer="A",
        summary="",
        winner_proposal_id=proposal.id,
        confidence=Confidence(score=0.9, components=[ConfidenceComponent("support", 1.0, 0.4)]),
        consensus_reached=True,
        rounds_run=1,
        rounds=[RoundRecord(round=1, proposals=[proposal], critiques=[critique])],
    )
    data = json.loads(json.dumps(result.to_dict()))
    assert data["rounds"][0]["proposals"][0]["answer"] == "A"
    assert data["confidence"]["components"][0]["name"] == "support"
    assert data["rounds"][0]["vote"] is None

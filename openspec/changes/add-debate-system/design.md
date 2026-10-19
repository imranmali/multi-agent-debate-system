# Design

## Context

The repository is empty (see proposal.md - Why). The system must run fully
offline so tests and CI never need network access or API keys, while still
being able to drive a real model. Python 3.10+ is the target runtime.

## Goals / Non-Goals

**Goals:**
- One debate engine that is independent of the model provider.
- Concurrency: every agent call within a phase runs in parallel (`asyncio`).
- Deterministic results for a given seed with the offline backend.
- Confidence that is explainable: every component of the score is reported.
- Fault tolerance: one failing agent does not abort the debate.

**Non-Goals:**
- A web UI or HTTP server.
- Persistent storage of debates (output is returned / printed as JSON).
- Tool use or retrieval by the agents; agents only exchange text.

## Decisions

### Package layout

```
src/debate_system/
  models.py        dataclasses shared by every module (contract)
  config.py        DebateConfig + validation
  llm/base.py      LLMRequest, LLMBackend protocol, BackendError
  llm/mock.py      MockBackend (deterministic offline swarm)
  llm/scripted.py  ScriptedBackend (canned replies for tests)
  llm/anthropic_backend.py  AnthropicBackend (optional extra)
  llm/__init__.py  create_backend(name, **kwargs) factory
  parsing.py       tolerant JSON extraction from model text
  agents.py        ProposerAgent, CriticAgent, VoterAgent, AggregatorAgent
  consensus.py     answer clustering, voting strategies, consensus rule
  confidence.py    confidence scoring and breakdown
  orchestrator.py  DebateOrchestrator (round loop)
  cli.py           `debate` command; __main__.py for `python -m`
```

Rationale: pure logic (`consensus`, `confidence`, `parsing`) is separated
from anything that awaits a model, so it is unit-testable without mocks, and
modules can be built in parallel against the `models.py` / `llm/base.py`
contract.

### Backend contract

`LLMBackend` is a `typing.Protocol` with one coroutine:
`async def complete(self, request: LLMRequest) -> str`.
`LLMRequest` carries `role` (`proposer` | `reviser` | `critic` | `voter` |
`aggregator`), `system`, `prompt`, and `metadata: dict`. Real backends use only
`system` and `prompt`; the offline backend uses `metadata` so it can simulate a
swarm without parsing prose. Alternative considered: role-specific backend
methods - rejected because it couples every provider to the debate protocol.

Metadata keys the orchestrator/agents MUST populate:

| role | keys |
|---|---|
| proposer | `agent_id`, `persona`, `round`, `question` |
| reviser | as proposer + `own_answer`, `peers`: list of `{id, agent_id, answer, score}` (includes own proposal) |
| critic | `question`, `proposal`: `{id, agent_id, answer}` |
| voter | `agent_id`, `question`, `candidates`: list of `{id, agent_id, answer, score}` |
| aggregator | `question`, `candidates` (as voter), `winner_id` |

Every model reply is JSON (agents ask for it in the prompt and parse
tolerantly):

| role | reply |
|---|---|
| proposer / reviser | `{"answer": str, "rationale": str}` |
| critic | `{"scores": {"correctness","completeness","clarity","feasibility": 0-10}, "feedback": str}` |
| voter | `{"ranking": [proposal_id, ...]}` best first |
| aggregator | `{"final_answer": str, "summary": str}` |

Parsing falls back gracefully: a proposer reply that is not JSON becomes the
answer verbatim; a critic reply without scores yields neutral scores (5/10)
and `parsed=False`; a voter ranking is filtered to known ids and missing ids
are appended by critic score; an aggregator failure falls back to the winning
proposal's answer.

### Offline MockBackend

Seeded with `hashlib` (never `hash()`, which is salted per process). Proposers
pick one of K canned stances from the persona and seed; revisers adopt the
best-scored peer answer with probability `convergence` (default 0.7); critic
scores are a deterministic function of the answer text so identical answers
always score the same; voters rank by score with small seeded jitter. This
produces realistic convergence over rounds with no network.

### Anthropic backend

Uses the official `anthropic` SDK (`AsyncAnthropic`), default model
`claude-opus-5-5`, `max_tokens=16000`, `output_config={"effort": "medium"}`,
server-side refusal fallback (`fallbacks="default"` with beta
`server-side-fallback-2026-07-01`). Text blocks are concatenated; a
`stop_reason == "refusal"` raises `BackendError`. The SDK is imported lazily
so the core package has zero runtime dependencies.

### Positions, voting and consensus

Proposals whose normalized answers (casefold, collapse whitespace, strip
trailing punctuation) are equal form one *position*. Voters rank proposals;
ballots are mapped to positions (first appearance wins), so converged agents do
not split the vote. Strategies:
- `plurality`: one point per first-place ballot.
- `borda`: `n-1-rank` points per ballot.
- `weighted` (default): Borda points multiplied by the voter's weight, where
  weight is the critic score of the voter's own latest proposal (floor 0.1).

The winner is the highest-scoring position; ties break on mean critic score,
then on earliest proposal id. *Support* is the (weighted) fraction of ballots
whose first choice is the winner. Consensus is reached when support is at or
above `consensus_threshold` (default 0.6). Self-votes are excluded unless
`allow_self_vote` is set (a voter's own proposal is dropped from its ballot).

### Confidence

`confidence = Σ wᵢ·cᵢ` with default weights: vote support 0.4, winner critic
quality 0.3, answer agreement (largest position size / proposals) 0.2,
stability (fraction of rounds whose winning position equals the final one)
0.1. Weights are configurable and normalized to sum to 1; the result is
clamped to `[0, 1]` and reported with each component.

### Round loop

Round 1: all proposers answer concurrently. Rounds ≥ 2: all agents revise
concurrently, seeing peers' answers and critic scores. Each round then runs
all critiques concurrently and all votes concurrently, tallies, and stops early
on consensus. Each call is wrapped in `asyncio.wait_for(timeout)`; failures are
recorded as `AgentError` entries in the round. If no proposal survives a
round, `DebateError` is raised. The aggregator runs once after the loop.

### CLI

`argparse`, no extra dependency. `debate QUESTION [--agents N] [--rounds R]
[--threshold T] [--strategy S] [--backend mock|anthropic] [--model M]
[--seed N] [--json] [--verbose]`. Exit 0 on success, 1 on debate/backend
error, 2 on invalid arguments.

### CI/CD

- `ci.yml` on push/PR: ruff lint + format check, mypy strict on `src`,
  pytest with coverage gate (≥ 90%) on Python 3.10-3.13, `openspec validate`,
  build sdist/wheel, Docker build and offline smoke run.
- `release.yml`: on push to `main` and `v*` tags, build and push the image
  to GHCR; on `v*` tags also build the package and publish a GitHub Release
  with the artifacts.

## Risks / Trade-offs

- [Real models may return non-JSON] → tolerant parser with explicit fallbacks
  and a `parsed` flag in critiques so callers can see degraded data.
- [Weighted voting lets a strong early agent dominate] → weight floor 0.1 and
  self-votes excluded by default.
- [Confidence is a heuristic, not a calibrated probability] → documented as
  such; every component is reported so users can re-weight.
- [Cost: agents × rounds × (1 + agents) calls on a paid backend] → small
  defaults (3 agents, 3 rounds) and early stop on consensus.

## Migration Plan

Greenfield; nothing to migrate. Rollback is reverting the merge commit.

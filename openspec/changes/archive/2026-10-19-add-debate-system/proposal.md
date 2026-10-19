# Proposal

## Why

A single LLM agent answers once and has no way to notice its own blind spots. A
swarm of agents that propose independently, critique each other, vote and then
synthesize a final answer is a well-known way to raise answer quality and to put
a calibrated confidence number on the result. This repository is empty, so this
change builds that system from scratch and shows how to orchestrate a swarm
rather than a single agent.

## What Changes

- Add a Python package (`debate_system`) that runs a multi-round debate:
  several proposer agents answer a question concurrently, a critic agent scores
  every proposal, proposers revise their answers using the critique, and the
  loop stops early when the swarm reaches consensus.
- Add pluggable voting strategies (plurality, Borda count, critic-score
  weighted) and a consensus rule with a configurable threshold.
- Add an aggregator agent that synthesizes one final answer and reports a
  confidence score in `[0, 1]` with a breakdown of where it came from.
- Add a pluggable LLM backend layer with a deterministic offline backend (the
  default, used by tests and CI) and an optional Anthropic Claude backend.
- Add a `debate` command-line interface that prints a human-readable report or
  JSON.
- Add project tooling: tests, lint, type checks, GitHub Actions CI, a release
  and container deployment workflow, a Dockerfile, README and MIT license.

## Capabilities

### New Capabilities
- `debate-orchestration`: running proposer agents concurrently over multiple rounds, revision, early stop and the debate transcript.
- `solution-critique`: the critic agent's scoring of each proposal against a rubric with written feedback.
- `consensus-voting`: voting strategies that rank proposals and the rule that decides whether consensus was reached.
- `answer-aggregation`: synthesis of the final answer and its confidence score.
- `llm-backends`: the model-provider abstraction, its deterministic offline backend and the Anthropic backend.
- `debate-cli`: the command-line entry point and its output formats.

### Modified Capabilities
<!-- None: this is the first change in the repository. -->

## Impact

- New code under `src/debate_system/`, tests under `tests/`.
- New runtime dependency: none required; `anthropic` is an optional extra.
- New dev dependencies: `pytest`, `pytest-asyncio`, `pytest-cov`, `ruff`, `mypy`.
- New CI/CD: `.github/workflows/ci.yml` (lint, type check, tests, build) and
  `.github/workflows/release.yml` (package release and container image).

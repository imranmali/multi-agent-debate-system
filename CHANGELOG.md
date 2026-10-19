# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-19

### Added

- Debate orchestration: concurrent proposer agents with distinct personas, multi-round
  revision using peer answers and critic feedback, early stop on consensus, per-call timeouts
  and fault tolerance for failing agents.
- Critic agent scoring each proposal on correctness, completeness, clarity and feasibility,
  with written feedback and flagged fallback scores for unparseable replies.
- Voting strategies `plurality`, `borda` and `weighted` (default), answer positions that merge
  equivalent answers, deterministic tie-breaking and a configurable consensus threshold.
- Aggregator agent that synthesizes the final answer, with a fallback to the winning proposal.
- Explainable confidence score combining vote support, critic quality, answer agreement and
  stability, with configurable, normalized weights.
- Pluggable LLM backends: a deterministic offline `mock` backend (default), a `scripted`
  backend for tests and an optional Anthropic Claude backend (`claude-opus-5-5` by default).
- `debate` command-line interface (also `python -m debate_system`) with human-readable and
  JSON output and documented exit codes.
- Python API: `DebateConfig`, `DebateOrchestrator`, `create_backend` and a JSON-serializable
  `DebateResult` transcript.
- GitHub Actions CI (lint, type check, tests on Python 3.10-3.13 with a 90% coverage gate,
  OpenSpec validation, package build and Docker smoke test) and a release workflow that
  publishes a container image to GHCR and GitHub Releases for `v*` tags.
- Dockerfile, README, contributing guide and MIT license.

[Unreleased]: https://github.com/imranmali/multi-agent-debate-system/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/imranmali/multi-agent-debate-system/releases/tag/v0.1.0

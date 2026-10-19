# Tasks

## 1. Scaffolding and shared contract

- [x] 1.1 Add `pyproject.toml` (hatchling, `debate` script, `anthropic` extra, dev extra, ruff/mypy/pytest config) and verify `pip install -e .[dev]` succeeds
- [x] 1.2 Add `models.py`, `config.py` and `llm/base.py` with the shared dataclasses, `DebateConfig` validation and the backend protocol; verify with `tests/test_config.py` and `tests/test_models.py`

## 2. LLM backends (workstream A)

- [x] 2.1 Implement `MockBackend` (seeded stances, convergence, deterministic critic scores, jittered voting) and verify determinism tests in `tests/test_llm_mock.py`
- [x] 2.2 Implement `ScriptedBackend` for tests and verify it in `tests/test_llm_scripted.py`
- [x] 2.3 Implement `AnthropicBackend` with lazy import, refusal handling and fallbacks, and verify with a fake client in `tests/test_llm_anthropic.py`
- [x] 2.4 Implement `create_backend` factory and verify unknown-name error in `tests/test_llm_factory.py`

## 3. Agents and parsing (workstream B)

- [x] 3.1 Implement tolerant JSON extraction in `parsing.py` and verify with `tests/test_parsing.py`
- [x] 3.2 Implement proposer/reviser, critic, voter and aggregator agents with prompts, metadata and fallbacks; verify with `tests/test_agents.py`

## 4. Consensus and confidence (workstream C)

- [x] 4.1 Implement answer normalization, positions, plurality/borda/weighted voting, tie-break and consensus in `consensus.py`; verify with `tests/test_consensus.py`
- [x] 4.2 Implement confidence scoring and weight normalization in `confidence.py`; verify with `tests/test_confidence.py`

## 5. Orchestration and CLI

- [x] 5.1 Implement `DebateOrchestrator` round loop with concurrency, timeouts, error capture and early stop; verify with `tests/test_orchestrator.py`
- [x] 5.2 Implement the `debate` CLI and `python -m debate_system`; verify exit codes and JSON output with `tests/test_cli.py`

## 6. Delivery (workstream D)

- [ ] 6.1 Add `.github/workflows/ci.yml` (lint, types, tests 3.10-3.13 with coverage gate, OpenSpec validation, build, Docker smoke) and verify the workflow passes on the PR
- [x] 6.2 Add `.github/workflows/release.yml` (GHCR image on main/tags, GitHub Release on tags) and verify both workflows with `actionlint` (no findings)
- [x] 6.3 Add `Dockerfile` and `.dockerignore`; verify `docker build` and an offline smoke run in CI
- [x] 6.4 Add `README.md`, `LICENSE` (MIT), `CONTRIBUTING.md` and `CHANGELOG.md`; verify the README quickstart commands run as written

## 7. Integration

- [x] 7.1 Run the full suite (`ruff check`, `ruff format --check`, `mypy`, `pytest --cov`) and verify coverage is at least 90%

## Workflow follow-up

- Archive the change with `openspec archive add-debate-system` once implementation is complete and verify `openspec/specs/` holds the six capabilities.
- Open a pull request to `main`, wait for green CI, then merge.

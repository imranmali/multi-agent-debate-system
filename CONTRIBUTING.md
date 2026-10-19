# Contributing

Thanks for helping improve the Multi-Agent Debate System. This guide covers setup, the branch
and commit conventions, the OpenSpec change workflow and the checks every pull request must pass.

## Setup

Requires Python 3.10 or newer. Node.js 22 is only needed for the OpenSpec CLI.

```bash
git clone https://github.com/imranmali/multi-agent-dDebate-system.git
cd multi-agent-dDebate-system
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
npm install -g @fission-ai/openspec   # or use: npx @fission-ai/openspec ...
```

Tests and CI use the deterministic offline `mock` backend, so no API key is needed. To try the
Claude backend locally, install the extra (`pip install -e '.[dev,anthropic]'`) and set
`ANTHROPIC_API_KEY`.

## Branches

- `main` is always releasable and only changes through pull requests that pass CI.
- Create a branch per change: `feature/<short-description>` (for example
  `feature/borda-tiebreak`). Fixes and chores may use `fix/*` or `chore/*`.
- Branches are merged into `main` with `--no-ff` so each feature stays visible in history.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/):
`<type>(<optional scope>): <description>`.

Common types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `build`, `ci`, `chore`.

```
feat(consensus): add borda voting strategy
fix(orchestrator): record timed-out critic calls as agent errors
docs(readme): document confidence weights
test(cli): cover exit code 2 for invalid --agents
ci: run tests on Python 3.13
feat(cli)!: rename --threshold to --consensus-threshold
```

Mark breaking changes with `!` after the type/scope or a `BREAKING CHANGE:` footer.

## OpenSpec change workflow

Behaviour changes are specified before they are implemented, using
[OpenSpec](https://github.com/Fission-AI/OpenSpec). The source-of-truth specs live in
`openspec/specs/`; proposed changes live in `openspec/changes/<change-id>/`.

1. **Proposal** - `proposal.md`: why the change is needed and what changes.
2. **Specs** - `specs/<capability>/spec.md`: requirement deltas (`## ADDED`, `## MODIFIED`,
   `## REMOVED Requirements`), each with at least one `#### Scenario:`.
3. **Design** - `design.md`: technical decisions and trade-offs (when non-trivial).
4. **Tasks** - `tasks.md`: an implementation checklist.
5. **Apply** - implement the tasks with tests, ticking them off in `tasks.md`.
6. **Archive** - after merge, `openspec archive <change-id>` folds the deltas into
   `openspec/specs/` and moves the change to `openspec/changes/archive/`.

Validate at any point with:

```bash
openspec validate --all --strict
```

Pure refactors, documentation and CI changes do not need an OpenSpec change.

## Before opening a pull request

Run the same checks as CI:

```bash
ruff check .
ruff format --check .
mypy
pytest --cov=debate_system --cov-report=term-missing   # must stay >= 90% coverage
openspec validate --all --strict
```

Then open a pull request against `main`, fill in the template (summary, OpenSpec change,
checklist), and update `README.md` and `CHANGELOG.md` when user-facing behaviour changes.

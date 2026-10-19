"""``debate`` command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from collections.abc import Sequence
from typing import Any, TextIO

from debate_system import __version__
from debate_system.config import VOTING_STRATEGIES, ConfigError, DebateConfig
from debate_system.llm import BACKENDS, BackendError, LLMBackend, create_backend
from debate_system.models import DebateResult, RoundRecord
from debate_system.orchestrator import DebateError, DebateOrchestrator, EventHandler

EXIT_OK, EXIT_FAILURE, EXIT_USAGE = 0, 1, 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="debate",
        description="Run a multi-agent debate: agents propose, a critic scores, the swarm "
        "votes, and an aggregator synthesizes the answer with a confidence score.",
    )
    parser.add_argument("question", help="the question or problem to debate")
    parser.add_argument("--agents", type=int, default=3, help="number of proposer agents")
    parser.add_argument("--rounds", type=int, default=3, help="maximum debate rounds")
    parser.add_argument(
        "--threshold", type=float, default=0.6, help="vote support needed for consensus"
    )
    parser.add_argument("--strategy", choices=VOTING_STRATEGIES, default="weighted")
    parser.add_argument("--backend", choices=BACKENDS, default="mock")
    parser.add_argument(
        "--model", default=None, help="model id for the anthropic backend (claude-opus-5-5)"
    )
    parser.add_argument("--seed", type=int, default=0, help="seed for the mock backend")
    parser.add_argument("--timeout", type=float, default=120.0, help="per-call timeout (s)")
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    parser.add_argument("--verbose", action="store_true", help="print progress to stderr")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def make_backend(args: argparse.Namespace) -> LLMBackend:
    if args.backend == "mock":
        return create_backend("mock", seed=args.seed)
    kwargs: dict[str, Any] = {}
    if args.model:
        kwargs["model"] = args.model
    return create_backend(args.backend, **kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse exits 0 for --help/--version, 2 on errors
        return int(exc.code or 0)

    config = DebateConfig(
        num_agents=args.agents,
        max_rounds=args.rounds,
        consensus_threshold=args.threshold,
        voting_strategy=args.strategy,
        timeout_seconds=args.timeout,
    )
    try:
        config.validate()
    except ConfigError as exc:
        print(f"debate: invalid configuration: {exc}", file=sys.stderr)
        return EXIT_USAGE

    try:
        backend = make_backend(args)
        on_event = _progress_printer(sys.stderr) if args.verbose else None
        result = DebateOrchestrator(backend, config, on_event=on_event).run_sync(args.question)
    except ValueError as exc:
        print(f"debate: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except (BackendError, DebateError) as exc:
        print(f"debate: {exc}", file=sys.stderr)
        return EXIT_FAILURE

    if args.json:
        json.dump(result.to_dict(), sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(format_report(result, backend=args.backend))
    return EXIT_OK


def format_report(result: DebateResult, *, backend: str = "") -> str:
    """Human-readable summary of a debate."""
    cfg = result.config
    lines = [f"Question: {result.question}"]
    if cfg:
        lines.append(
            f"Backend: {backend or '?'} | agents: {cfg['num_agents']} | "
            f"strategy: {cfg['voting_strategy']} | threshold: {cfg['consensus_threshold']:.2f}"
        )
    for record in result.rounds:
        lines += ["", *_format_round(record)]
    outcome = "consensus reached" if result.consensus_reached else "no consensus"
    plural = "s" if result.rounds_run != 1 else ""
    lines += [
        "",
        f"Final answer ({outcome} after {result.rounds_run} round{plural}, "
        f"winner {result.winner_proposal_id}):",
        textwrap.indent(textwrap.fill(result.final_answer, width=88), "  "),
    ]
    if result.summary:
        lines.append(textwrap.indent(textwrap.fill(result.summary, width=88), "  "))
    lines += ["", f"Confidence: {result.confidence.score:.2f}"]
    lines += [
        f"  {c.name:<10} {c.value:.2f} x {c.weight:.2f}" for c in result.confidence.components
    ]
    return "\n".join(lines) + "\n"


def _format_round(record: RoundRecord) -> list[str]:
    scores = {c.proposal_id: c.overall for c in record.critiques}
    lines = [f"Round {record.round}"]
    for p in record.proposals:
        score = scores.get(p.id)
        shown = f"{score:.2f}" if score is not None else " n/a"
        lines.append(f"  {p.agent_id:<9} {p.persona:<16} score {shown}  {_shorten(p.answer)}")
    for err in record.errors:
        lines.append(f"  ! {err.agent_id} ({err.role}) failed: {err.message}")
    if record.vote is not None:
        v = record.vote
        lines.append(
            f"  -> winner {v.winner_proposal_id} | support {v.support:.2f} | "
            f"consensus {'yes' if v.consensus else 'no'}"
        )
    return lines


def _shorten(text: str, width: int = 60) -> str:
    return textwrap.shorten(" ".join(text.split()), width=width, placeholder="...")


def _progress_printer(stream: TextIO) -> EventHandler:
    def on_event(name: str, payload: dict[str, Any]) -> None:
        if name == "round_started":
            print(f"[debate] round {payload['round']} started", file=stream)
        elif name == "agent_failed":
            err = payload["error"]
            print(f"[debate] {err.agent_id} ({err.role}) failed: {err.message}", file=stream)
        elif name == "round_completed":
            vote = payload["record"].vote
            print(
                f"[debate] round {payload['record'].round} done: support {vote.support:.2f}, "
                f"consensus {'yes' if vote.consensus else 'no'}",
                file=stream,
            )
        elif name == "debate_completed":
            print("[debate] aggregation done", file=stream)

    return on_event


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

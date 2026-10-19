import json
import subprocess
import sys

import pytest

from debate_system import cli
from debate_system.llm import BackendError


def test_default_run_prints_report(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["Which database should we use?"]) == 0
    out = capsys.readouterr().out
    assert "Question: Which database should we use?" in out
    assert "Round 1" in out
    assert "Final answer" in out
    assert "Confidence:" in out


def test_json_output_is_one_document(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["Q?", "--json", "--agents", "4", "--seed", "3", "--strategy", "borda"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["final_answer"]
    assert 0 <= data["confidence"]["score"] <= 1
    assert data["config"]["num_agents"] == 4
    assert data["config"]["voting_strategy"] == "borda"


def test_invalid_agent_count_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["Q?", "--agents", "1"]) == 2
    assert "num_agents" in capsys.readouterr().err


def test_unknown_strategy_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["Q?", "--strategy", "dictator"]) == 2


def test_empty_question_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["   "]) == 2
    assert "empty" in capsys.readouterr().err


def test_backend_failure_exits_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom(name: str, **kwargs: object) -> None:
        raise BackendError(f"cannot create {name}")

    monkeypatch.setattr(cli, "create_backend", boom)
    assert cli.main(["Q?", "--backend", "anthropic", "--model", "claude-opus-5-5"]) == 1
    assert "cannot create anthropic" in capsys.readouterr().err


def test_anthropic_backend_receives_model(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake(name: str, **kwargs: object) -> None:
        seen.update(name=name, **kwargs)
        raise BackendError("stop here")

    monkeypatch.setattr(cli, "create_backend", fake)
    cli.main(["Q?", "--backend", "anthropic", "--model", "claude-sonnet-5-5"])
    assert seen == {"name": "anthropic", "model": "claude-sonnet-5-5"}


def test_verbose_reports_progress(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["Q?", "--verbose", "--rounds", "2"]) == 0
    err = capsys.readouterr().err
    assert "[debate] round 1 started" in err
    assert "aggregation done" in err


def test_help_exits_0(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--help"]) == 0
    assert "--strategy" in capsys.readouterr().out


def test_python_dash_m_entry_point() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "debate_system", "Q?", "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["rounds_run"] >= 1

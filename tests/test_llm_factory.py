import subprocess
import sys
from types import SimpleNamespace

import pytest

from debate_system.llm import (
    BACKENDS,
    AnthropicBackend,
    BackendError,
    LLMBackend,
    MockBackend,
    ScriptedBackend,
    create_backend,
)


def test_backend_names() -> None:
    assert BACKENDS == ("mock", "anthropic")


@pytest.mark.parametrize("name", ["mock", "MOCK", " Mock "])
def test_create_mock_is_case_insensitive_and_forwards_kwargs(name: str) -> None:
    backend = create_backend(name, seed=3, convergence=0.5)
    assert isinstance(backend, MockBackend)
    assert (backend.seed, backend.convergence) == (3, 0.5)


def test_create_anthropic_forwards_kwargs() -> None:
    client = SimpleNamespace()
    backend = create_backend("Anthropic", model="claude-haiku-5-5", client=client)
    assert isinstance(backend, AnthropicBackend)
    assert backend.model == "claude-haiku-5-5"


def test_unknown_backend_lists_valid_names() -> None:
    with pytest.raises(BackendError) as exc:
        create_backend("gpt")
    message = str(exc.value)
    assert "'gpt'" in message
    assert "mock" in message
    assert "anthropic" in message


def test_backends_satisfy_protocol() -> None:
    assert isinstance(MockBackend(), LLMBackend)
    assert isinstance(ScriptedBackend(), LLMBackend)
    assert isinstance(AnthropicBackend(client=SimpleNamespace()), LLMBackend)


def test_importing_llm_does_not_import_anthropic() -> None:
    code = "import sys, debate_system.llm; assert 'anthropic' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from debate_system.llm import AnthropicBackend, BackendError, LLMRequest
from debate_system.llm.anthropic_backend import FALLBACK_BETA, INSTALL_HINT

REQUEST = LLMRequest(role="proposer", system="You are terse.", prompt="Answer in JSON.")


def text(value: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=value)


def response(*blocks: Any, stop_reason: str = "end_turn", stop_details: Any = None) -> Any:
    return SimpleNamespace(content=list(blocks), stop_reason=stop_reason, stop_details=stop_details)


class FakeMessages:
    def __init__(self, reply: Any) -> None:
        self.reply = reply
        self.kwargs: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.kwargs.append(kwargs)
        if isinstance(self.reply, BaseException):
            raise self.reply
        return self.reply


class FakeClient:
    def __init__(self, reply: Any) -> None:
        self.messages = FakeMessages(reply)
        self.beta = SimpleNamespace(messages=self.messages)


async def test_success_concatenates_text_blocks_and_sends_defaults() -> None:
    thinking = SimpleNamespace(type="thinking", thinking="")
    client = FakeClient(response(thinking, text('{"answer": '), text('"x"}')))
    backend = AnthropicBackend(client=client)

    assert await backend.complete(REQUEST) == '{"answer": "x"}'
    assert client.messages.kwargs == [
        {
            "model": "claude-opus-5-5",
            "max_tokens": 16000,
            "system": "You are terse.",
            "messages": [{"role": "user", "content": "Answer in JSON."}],
            "output_config": {"effort": "medium"},
            "betas": [FALLBACK_BETA],
            "fallbacks": "default",
        }
    ]


async def test_overrides_and_optional_params_omitted() -> None:
    client = FakeClient(response(text("ok")))
    backend = AnthropicBackend(
        model="claude-sonnet-5-5", max_tokens=512, effort=None, use_fallbacks=False, client=client
    )
    assert await backend.complete(REQUEST) == "ok"
    (kwargs,) = client.messages.kwargs
    assert kwargs["model"] == "claude-sonnet-5-5"
    assert kwargs["max_tokens"] == 512
    assert not {"output_config", "betas", "fallbacks"} & kwargs.keys()


@pytest.mark.parametrize(
    ("stop_details", "expected"),
    [
        (
            SimpleNamespace(type="refusal", category="cyber", explanation="no"),
            r" \(category: cyber\)$",
        ),
        (SimpleNamespace(type="refusal", category=None), r"$"),
        (None, r"$"),
    ],
)
async def test_refusal_raises_before_reading_content(stop_details: Any, expected: str) -> None:
    client = FakeClient(response(text("partial"), stop_reason="refusal", stop_details=stop_details))
    with pytest.raises(BackendError, match=f"refused the proposer request{expected}"):
        await AnthropicBackend(client=client).complete(REQUEST)


@pytest.mark.parametrize("blocks", [(), (text("   "),), (SimpleNamespace(type="tool_use"),)])
async def test_empty_text_raises(blocks: tuple[Any, ...]) -> None:
    client = FakeClient(response(*blocks))
    with pytest.raises(BackendError, match="no text"):
        await AnthropicBackend(client=client).complete(REQUEST)


async def test_sdk_exceptions_are_wrapped() -> None:
    error = ConnectionError("network down")
    client = FakeClient(error)
    with pytest.raises(BackendError, match=r"ConnectionError\): network down") as exc:
        await AnthropicBackend(client=client).complete(REQUEST)
    assert exc.value.__cause__ is error


def test_missing_sdk_raises_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(BackendError) as exc:
        AnthropicBackend()
    assert str(exc.value) == INSTALL_HINT
    assert "pip install 'multi-agent-debate[anthropic]'" in INSTALL_HINT


def test_creates_async_client_from_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[dict[str, Any]] = []

    class AsyncAnthropic:
        def __init__(self, **kwargs: Any) -> None:
            created.append(kwargs)

    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(AsyncAnthropic=AsyncAnthropic))
    backend = AnthropicBackend(api_key="sk-secret", timeout=30.0)
    assert created == [{"api_key": "sk-secret", "timeout": 30.0}]
    assert "sk-secret" not in repr(backend)
    assert repr(backend) == "AnthropicBackend(model='claude-opus-5-5')"


def test_invalid_max_tokens() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        AnthropicBackend(max_tokens=0, client=FakeClient(None))

import asyncio
import time

import pytest

from debate_system.llm import BackendError, LLMRequest, ScriptedBackend
from debate_system.llm.base import Role


def req(role: Role = "proposer", prompt: str = "") -> LLMRequest:
    return LLMRequest(role=role, system="sys", prompt=prompt)


async def test_queue_is_consumed_in_order_and_last_entry_repeats() -> None:
    backend = ScriptedBackend({"proposer": ["one", "two"]})
    assert [await backend.complete(req()) for _ in range(4)] == ["one", "two", "two", "two"]


async def test_queues_are_per_role() -> None:
    backend = ScriptedBackend({"proposer": ["p1", "p2"], "critic": ["c1"]})
    assert await backend.complete(req("proposer")) == "p1"
    assert await backend.complete(req("critic")) == "c1"
    assert await backend.complete(req("proposer")) == "p2"


async def test_exception_entries_are_raised_then_queue_continues() -> None:
    backend = ScriptedBackend({"voter": [BackendError("boom"), "ok"]})
    with pytest.raises(BackendError, match="boom"):
        await backend.complete(req("voter"))
    assert await backend.complete(req("voter")) == "ok"


async def test_callable_entries_receive_the_request() -> None:
    backend = ScriptedBackend({"critic": [lambda r: f"echo {r.prompt}", "done"]})
    assert await backend.complete(req("critic", "hi")) == "echo hi"
    assert await backend.complete(req("critic", "hi")) == "done"


async def test_bare_callable_is_called_every_time() -> None:
    backend = ScriptedBackend({"reviser": lambda r: r.prompt.upper()})
    assert await backend.complete(req("reviser", "a")) == "A"
    assert await backend.complete(req("reviser", "b")) == "B"


async def test_bare_string_and_exception_repeat() -> None:
    backend = ScriptedBackend({"proposer": "same", "critic": TimeoutError("slow")})
    assert await backend.complete(req()) == await backend.complete(req()) == "same"
    with pytest.raises(TimeoutError):
        await backend.complete(req("critic"))


async def test_default_reply_for_unscripted_roles() -> None:
    backend = ScriptedBackend({"proposer": ["p"]}, default="{}")
    assert await backend.complete(req("aggregator")) == "{}"
    assert await backend.complete(req("proposer")) == "p"


async def test_unscripted_role_without_default_raises() -> None:
    backend = ScriptedBackend()
    with pytest.raises(BackendError, match="no reply scripted for role 'voter'"):
        await backend.complete(req("voter"))
    assert len(backend.calls) == 1


async def test_records_every_call() -> None:
    backend = ScriptedBackend(default="x")
    requests = [req("proposer", "1"), req("critic", "2"), req("proposer", "3")]
    for request in requests:
        await backend.complete(request)
    assert backend.calls == requests
    assert backend.calls_for("proposer") == [requests[0], requests[2]]


async def test_concurrent_calls_run_in_parallel_and_follow_call_order() -> None:
    backend = ScriptedBackend({"proposer": [str(i) for i in range(10)]}, delay=0.05)
    start = time.perf_counter()
    replies = await asyncio.gather(*(backend.complete(req(prompt=str(i))) for i in range(10)))
    elapsed = time.perf_counter() - start
    assert replies == [str(i) for i in range(10)]
    assert len(backend.calls) == 10
    assert elapsed < 0.4


def test_empty_script_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        ScriptedBackend({"proposer": []})


def test_negative_delay_is_rejected() -> None:
    with pytest.raises(ValueError, match="delay"):
        ScriptedBackend(delay=-1)

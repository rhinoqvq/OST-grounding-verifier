import pytest

from verifier.harness.errors import ReplayMiss
from verifier.models.cache import ModelCallCache
from verifier.rag.store import connect, init_db


@pytest.fixture
def conn():
    conn = connect(":memory:")
    init_db(conn)
    return conn


async def test_record_then_replay(conn):
    calls = {"n": 0}

    async def fetch():
        calls["n"] += 1
        return {"response": {"answer": 42}, "usage": {"input_tokens": 10}}

    record = ModelCallCache(conn, mode="record")
    first = await record.call(provider="jev", model="m", request={"q": 1}, fetch=fetch)
    second = await record.call(provider="jev", model="m", request={"q": 1}, fetch=fetch)
    assert calls["n"] == 1, "second call must come from cache"
    assert second["response"] == first["response"]

    replay = ModelCallCache(conn, mode="replay")
    replayed = await replay.call(provider="jev", model="m", request={"q": 1}, fetch=fetch)
    assert replayed["response"] == {"answer": 42}
    assert calls["n"] == 1, "replay must not hit the network"


async def test_replay_miss_is_an_error(conn):
    async def fetch():
        return {"response": 1}

    replay = ModelCallCache(conn, mode="replay")
    with pytest.raises(ReplayMiss):
        await replay.call(provider="jev", model="m", request={"q": "never-recorded"}, fetch=fetch)


async def test_off_mode_never_stores(conn):
    async def fetch():
        return {"response": 1}

    off = ModelCallCache(conn, mode="off")
    await off.call(provider="jev", model="m", request={"q": 2}, fetch=fetch)
    assert conn.execute("SELECT COUNT(*) FROM model_calls").fetchone()[0] == 0


def test_key_is_order_insensitive(conn):
    k1 = ModelCallCache.key("p", "m", {"a": 1, "b": 2})
    k2 = ModelCallCache.key("p", "m", {"b": 2, "a": 1})
    assert k1 == k2

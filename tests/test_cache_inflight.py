"""Record-mode cache under concurrency: identical requests in flight share one
fetch, a stored row is never rewritten, a failure stores nothing (2026-10-23:
answers sharing a source raced on identical source-tree requests, and the
last write replaced the response the first answer had used)."""

import asyncio

import pytest

from verifier.models.cache import ModelCallCache
from verifier.rag.store import connect, init_db


@pytest.fixture
def conn():
    conn = connect(":memory:")
    init_db(conn)
    return conn


def rows(conn):
    return {r["cache_key"]: r["response_json"] for r in conn.execute("SELECT cache_key, response_json FROM model_calls")}


class Upstream:
    """A fake model: counts calls, answers once released, numbers its answers."""

    def __init__(self):
        self.calls = 0
        self.release = asyncio.Event()

    def fetch(self, fail: Exception | None = None):
        async def go():
            self.calls += 1
            n = self.calls
            await self.release.wait()
            if fail is not None:
                raise fail
            return {"response": {"sample": n}, "usage": {"input_tokens": 5}}

        return go


async def test_concurrent_identical_requests_fetch_once(conn):
    cache, up = ModelCallCache(conn, mode="record"), Upstream()
    tasks = [asyncio.create_task(cache.call(provider="jev", model="m", request={"q": 1}, fetch=up.fetch()))
             for _ in range(8)]
    await asyncio.sleep(0)
    up.release.set()
    results = await asyncio.gather(*tasks)
    assert up.calls == 1
    assert all(r == results[0] for r in results)
    assert results[0]["response"] == {"sample": 1}
    assert len(rows(conn)) == 1


async def test_different_keys_do_not_wait_for_each_other(conn):
    cache = ModelCallCache(conn, mode="record")
    slow, fast = Upstream(), Upstream()
    fast.release.set()
    a = asyncio.create_task(cache.call(provider="jev", model="m", request={"q": "slow"}, fetch=slow.fetch()))
    b = await cache.call(provider="jev", model="m", request={"q": "fast"}, fetch=fast.fetch())
    assert b["response"] == {"sample": 1} and not a.done()
    slow.release.set()
    await a
    assert (slow.calls, fast.calls) == (1, 1)
    assert len(rows(conn)) == 2


async def test_failure_reaches_every_waiter_and_stores_nothing(conn):
    cache, up = ModelCallCache(conn, mode="record"), Upstream()
    boom = RuntimeError("upstream down")
    tasks = [asyncio.create_task(cache.call(provider="jev", model="m", request={"q": 1}, fetch=up.fetch(boom)))
             for _ in range(4)]
    await asyncio.sleep(0)
    up.release.set()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    assert up.calls == 1
    assert all(r is boom for r in results)
    assert rows(conn) == {}
    # the next call retries
    again = await cache.call(provider="jev", model="m", request={"q": 1}, fetch=up.fetch())
    assert up.calls == 2 and again["response"] == {"sample": 2}


async def test_stored_row_is_never_rewritten(conn):
    cache = ModelCallCache(conn, mode="record")
    assert cache.put("k", provider="jev", model="m", request={}, response={"response": "first"})
    assert not cache.put("k", provider="jev", model="m", request={}, response={"response": "second"})
    assert cache.get("k") == {"response": "first"}


async def test_losing_a_race_to_another_process_returns_the_stored_row(conn):
    cache = ModelCallCache(conn, mode="record")
    other = ModelCallCache(conn, mode="record")  # another process writing the same db
    request = {"q": 1}

    async def fetch():
        other.put(cache.key("jev", "m", request), provider="jev", model="m", request=request,
                  response={"response": "theirs"})
        return {"response": "mine"}

    result = await cache.call(provider="jev", model="m", request=request, fetch=fetch)
    assert result == {"response": "theirs"}
    assert cache.get(cache.key("jev", "m", request)) == {"response": "theirs"}


async def test_cancelled_leader_leaves_waiters_to_fetch(conn):
    cache, up = ModelCallCache(conn, mode="record"), Upstream()
    leader = asyncio.create_task(cache.call(provider="jev", model="m", request={"q": 1}, fetch=up.fetch()))
    await asyncio.sleep(0)
    waiter = asyncio.create_task(cache.call(provider="jev", model="m", request={"q": 1}, fetch=up.fetch()))
    await asyncio.sleep(0)
    leader.cancel()
    with pytest.raises(asyncio.CancelledError):
        await leader
    await asyncio.sleep(0)
    up.release.set()
    result = await waiter
    assert up.calls == 2 and result["response"] == {"sample": 2}
    assert len(rows(conn)) == 1


async def test_cancelled_waiter_does_not_cancel_the_fetch(conn):
    cache, up = ModelCallCache(conn, mode="record"), Upstream()
    leader = asyncio.create_task(cache.call(provider="jev", model="m", request={"q": 1}, fetch=up.fetch()))
    await asyncio.sleep(0)
    waiter = asyncio.create_task(cache.call(provider="jev", model="m", request={"q": 1}, fetch=up.fetch()))
    await asyncio.sleep(0)
    waiter.cancel()
    up.release.set()
    result = await leader
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert up.calls == 1 and result["response"] == {"sample": 1}
    assert len(rows(conn)) == 1


async def test_off_mode_still_fetches_every_call(conn):
    cache, up = ModelCallCache(conn, mode="off"), Upstream()
    up.release.set()
    await asyncio.gather(*(cache.call(provider="jev", model="m", request={"q": 1}, fetch=up.fetch()) for _ in range(3)))
    assert up.calls == 3 and rows(conn) == {}

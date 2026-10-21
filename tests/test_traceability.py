"""Run-id attribution and served-model drift sentinel (2026-10-22 audit)."""

from __future__ import annotations

import sqlite3

from verifier.harness.runctx import CURRENT_RUN_ID
from verifier.models.cache import ModelCallCache
from verifier.models.jev import JevDecider
from verifier.rag import store


_OLD_MODEL_CALLS = """
CREATE TABLE model_calls (
    cache_key TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    request_json TEXT NOT NULL,
    response_json TEXT NOT NULL,
    usage_json TEXT,
    latency_ms REAL,
    created REAL NOT NULL
);
"""


def test_init_db_migrates_old_model_calls_schema():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(_OLD_MODEL_CALLS)
    conn.execute(
        "INSERT INTO model_calls VALUES ('k', 'jev', 'm', '{}', '{}', NULL, 1.0, 0.0)"
    )
    store.init_db(conn)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(model_calls)")}
    assert "run_id" in columns
    # the pre-migration row survives with a NULL run_id
    row = conn.execute("SELECT run_id FROM model_calls WHERE cache_key='k'").fetchone()
    assert row["run_id"] is None


def test_cache_put_stamps_current_run_id():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    store.init_db(conn)
    cache = ModelCallCache(conn, mode="record")
    token = CURRENT_RUN_ID.set("run-abc")
    try:
        cache.put("key1", provider="jev", model="m", request={}, response={})
    finally:
        CURRENT_RUN_ID.reset(token)
    cache.put("key2", provider="jev", model="m", request={}, response={})
    rows = {r["cache_key"]: r["run_id"] for r in conn.execute("SELECT cache_key, run_id FROM model_calls")}
    assert rows == {"key1": "run-abc", "key2": None}


def test_save_run_reuses_current_run_id():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    store.init_db(conn)
    token = CURRENT_RUN_ID.set("run-xyz")
    try:
        run_id = store.save_run(
            conn, request={}, response=None, status="ok", tree=None, judgments=[], trace={}
        )
    finally:
        CURRENT_RUN_ID.reset(token)
    assert run_id == "run-xyz"
    assert conn.execute("SELECT run_id FROM runs").fetchone()["run_id"] == "run-xyz"


def test_served_model_drift_sentinel():
    decider = JevDecider(api_key=None, model="jev-latest", cache=None, expected_served="jev-1.13.0")
    decider._check_served("jev-1.13.0")
    assert decider.served_drift is None
    decider._check_served(None)
    assert decider.served_drift is None
    decider._check_served("jev-1.14.0")
    assert decider.served_drift == "jev-1.14.0"
    # no sentinel configured -> never flags
    loose = JevDecider(api_key=None, model="jev-latest", cache=None, expected_served=None)
    loose._check_served("anything")
    assert loose.served_drift is None

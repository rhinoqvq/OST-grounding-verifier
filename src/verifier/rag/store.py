"""SQLite persistence: finished runs, their trees, judgments and traces.

During a request the tree lives in memory (it is tiny and short-lived).
Everything is saved afterwards so a run can be inspected and replayed:
`SELECT ... FROM nodes WHERE run_id=?` beats re-running the pipeline when
debugging a wrong verdict.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from verifier.rag.tree import SemanticTree

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created REAL NOT NULL,
    request_json TEXT NOT NULL,
    response_json TEXT,
    status TEXT NOT NULL,
    trace_json TEXT
);
CREATE TABLE IF NOT EXISTS nodes (
    run_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    parent_id TEXT,
    json TEXT NOT NULL,
    PRIMARY KEY (run_id, node_id)
);
CREATE TABLE IF NOT EXISTS edges (
    run_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    from_id TEXT NOT NULL,
    to_id TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS judgments (
    run_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    node_version INTEGER NOT NULL,
    json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS model_calls (
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


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    conn.commit()


def save_run(
    conn: sqlite3.Connection,
    *,
    request: dict[str, Any],
    response: dict[str, Any] | None,
    status: str,
    tree: SemanticTree | None,
    judgments: list[dict[str, Any]],
    trace: dict[str, Any],
) -> str:
    run_id = uuid.uuid4().hex
    conn.execute(
        "INSERT INTO runs VALUES (?,?,?,?,?,?)",
        (
            run_id,
            time.time(),
            json.dumps(request, ensure_ascii=False),
            json.dumps(response, ensure_ascii=False) if response is not None else None,
            status,
            json.dumps(trace, ensure_ascii=False),
        ),
    )
    if tree is not None:
        conn.executemany(
            "INSERT INTO nodes VALUES (?,?,?,?)",
            [
                (run_id, n.node_id, n.parent_id, json.dumps(asdict(n), ensure_ascii=False, default=str))
                for n in tree.nodes.values()
            ],
        )
        conn.executemany(
            "INSERT INTO edges VALUES (?,?,?,?)",
            [(run_id, e.kind, e.from_id, e.to_id) for e in tree.edges],
        )
    conn.executemany(
        "INSERT INTO judgments VALUES (?,?,?,?)",
        [
            (run_id, j["node_id"], j.get("node_version", 1), json.dumps(j, ensure_ascii=False))
            for j in judgments
        ],
    )
    conn.commit()
    return run_id


def load_run(conn: sqlite3.Connection, run_id: str) -> dict[str, Any]:
    run = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
    if run is None:
        raise KeyError(f"unknown run {run_id}")
    nodes = conn.execute("SELECT * FROM nodes WHERE run_id=?", (run_id,)).fetchall()
    judgments = conn.execute("SELECT * FROM judgments WHERE run_id=?", (run_id,)).fetchall()
    return {
        "run": dict(run),
        "nodes": [json.loads(r["json"]) for r in nodes],
        "judgments": [json.loads(r["json"]) for r in judgments],
    }


def query_subtree(conn: sqlite3.Connection, run_id: str, node_id: str) -> list[dict[str, Any]]:
    """Fetch a node and all descendants with one recursive query (debug helper)."""
    rows = conn.execute(
        """
        WITH RECURSIVE sub(node_id) AS (
            SELECT node_id FROM nodes WHERE run_id=:run AND node_id=:node
            UNION ALL
            SELECT n.node_id FROM nodes n JOIN sub ON n.parent_id = sub.node_id
            WHERE n.run_id = :run
        )
        SELECT n.json FROM nodes n JOIN sub USING (node_id) WHERE n.run_id = :run
        """,
        {"run": run_id, "node": node_id},
    ).fetchall()
    return [json.loads(r["json"]) for r in rows]

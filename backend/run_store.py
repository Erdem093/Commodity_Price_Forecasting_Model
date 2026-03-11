from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


class RunStore:
    def __init__(self, db_path: str | Path = "artifacts/run_store.sqlite") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                  run_id TEXT PRIMARY KEY,
                  mode TEXT NOT NULL,
                  status TEXT NOT NULL,
                  progress_pct REAL NOT NULL,
                  current_stage TEXT,
                  created_at TEXT NOT NULL,
                  started_at TEXT,
                  finished_at TEXT,
                  runtime_sec REAL,
                  trace_json TEXT,
                  error_json TEXT,
                  results_json TEXT,
                  provenance_json TEXT,
                  timings_json TEXT
                )
                """
            )

    def create_run(self, run_id: str, mode: str, created_at: str) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                """
                INSERT INTO runs(
                    run_id, mode, status, progress_pct, current_stage, created_at, trace_json
                ) VALUES (?, ?, 'queued', 0.0, 'queued', ?, '[]')
                """,
                (run_id, mode, created_at),
            )

    def update(self, run_id: str, **fields: Any) -> None:
        if not fields:
            return
        keys = []
        vals = []
        for k, v in fields.items():
            keys.append(f"{k} = ?")
            if isinstance(v, (dict, list)):
                vals.append(json.dumps(v))
            else:
                vals.append(v)
        vals.append(run_id)
        sql = f"UPDATE runs SET {', '.join(keys)} WHERE run_id = ?"
        with self._lock, self._conn() as conn:
            conn.execute(sql, vals)

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self._lock, self._conn() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if row is None:
                return None

        out = dict(row)
        for key in ["trace_json", "error_json", "results_json", "provenance_json", "timings_json"]:
            if out.get(key):
                out[key] = json.loads(out[key])
            else:
                out[key] = None
        return out

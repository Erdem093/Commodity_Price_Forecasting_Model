from __future__ import annotations

from datetime import datetime, timezone
from queue import Queue
from threading import Thread
from typing import Any
from uuid import uuid4

from flask import Flask, jsonify, request

from backend.run_store import RunStore
from backend.runner import run_demo_quick
from commodity_forecasting.utils import make_run_id


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_app(start_worker: bool = True) -> Flask:
    app = Flask(__name__)
    store = RunStore()
    job_queue: Queue[tuple[str, str]] = Queue()
    app.config["RUN_STORE"] = store
    app.config["RUN_QUEUE"] = job_queue

    def set_stage(run_id: str, stage: str, progress: float, stats: dict[str, Any]) -> None:
        run = store.get(run_id)
        trace = run["trace_json"] or []
        step = {
            "stage": stage,
            "timestamp": _now_iso(),
            "progress_pct": float(progress),
            "stats": stats,
        }
        trace.append(step)
        store.update(run_id, current_stage=stage, progress_pct=float(progress), trace_json=trace)

    def worker_loop() -> None:
        while True:
            run_id, mode = job_queue.get()
            start_ts = datetime.now(timezone.utc)
            store.update(run_id, status="running", started_at=start_ts.isoformat())
            try:
                if mode != "demo_quick":
                    raise ValueError(f"Unsupported mode: {mode}")

                payload = run_demo_quick(run_id, lambda stage, p, s: set_stage(run_id, stage, p, s))
                finish_ts = datetime.now(timezone.utc)
                runtime = (finish_ts - start_ts).total_seconds()
                store.update(
                    run_id,
                    status="completed",
                    progress_pct=100.0,
                    current_stage="completed",
                    finished_at=finish_ts.isoformat(),
                    runtime_sec=runtime,
                    results_json=payload["results"],
                    provenance_json=payload["provenance"],
                    timings_json={"started_at": start_ts.isoformat(), "finished_at": finish_ts.isoformat(), "runtime_sec": runtime},
                )
            except Exception as exc:  # pragma: no cover - runtime/network dependent
                finish_ts = datetime.now(timezone.utc)
                runtime = (finish_ts - start_ts).total_seconds()
                store.update(
                    run_id,
                    status="failed",
                    current_stage="failed",
                    finished_at=finish_ts.isoformat(),
                    runtime_sec=runtime,
                    error_json={"message": str(exc)},
                )
            finally:
                job_queue.task_done()

    if start_worker:
        t = Thread(target=worker_loop, daemon=True)
        t.start()

    @app.post("/api/runs")
    def create_run() -> Any:
        payload = request.get_json(silent=True) or {}
        mode = payload.get("mode", "demo_quick")
        run_id = f"{make_run_id('live')}_{uuid4().hex[:6]}"
        store.create_run(run_id=run_id, mode=mode, created_at=_now_iso())
        job_queue.put((run_id, mode))
        return jsonify({"run_id": run_id})

    @app.get("/api/runs/<run_id>")
    def get_run(run_id: str) -> Any:
        run = store.get(run_id)
        if run is None:
            return jsonify({"error": "run_not_found"}), 404

        return jsonify(
            {
                "run_id": run_id,
                "status": run["status"],
                "progress_pct": run["progress_pct"],
                "current_stage": run["current_stage"],
                "created_at": run["created_at"],
                "started_at": run["started_at"],
                "finished_at": run["finished_at"],
                "runtime_sec": run["runtime_sec"],
                "trace": run["trace_json"] or [],
                "error": run["error_json"],
            }
        )

    @app.get("/api/runs/<run_id>/results")
    def get_results(run_id: str) -> Any:
        run = store.get(run_id)
        if run is None:
            return jsonify({"error": "run_not_found"}), 404
        if run["status"] != "completed":
            return jsonify({"error": "run_not_completed", "status": run["status"]}), 409

        return jsonify(
            {
                "run_id": run_id,
                "status": run["status"],
                "started_at": run["started_at"],
                "finished_at": run["finished_at"],
                "runtime_sec": run["runtime_sec"],
                "results": run["results_json"],
                "provenance": run["provenance_json"],
                "timings": run["timings_json"],
            }
        )

    return app


app = create_app(start_worker=True)

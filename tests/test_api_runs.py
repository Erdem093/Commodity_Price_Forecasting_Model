from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from api.app import create_app


def test_create_run_and_get_status() -> None:
    app = create_app(start_worker=False)
    client = app.test_client()

    r = client.post("/api/runs", json={"mode": "demo_quick"})
    assert r.status_code == 200
    payload = r.get_json()
    assert "run_id" in payload
    run_id = payload["run_id"]

    s = client.get(f"/api/runs/{run_id}")
    assert s.status_code == 200
    status = s.get_json()
    assert status["status"] == "queued"
    assert status["run_id"] == run_id


def test_results_not_ready_when_queued() -> None:
    app = create_app(start_worker=False)
    client = app.test_client()

    r = client.post("/api/runs", json={"mode": "demo_quick"})
    run_id = r.get_json()["run_id"]

    res = client.get(f"/api/runs/{run_id}/results")
    assert res.status_code == 409
    body = res.get_json()
    assert body["error"] == "run_not_completed"

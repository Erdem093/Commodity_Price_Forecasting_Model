# Commodity Price Forecasting Model

End-to-end WTI next-day direction forecasting pipeline using weather + yfinance data.

## Live Demo

Frontend is live at: **https://commoditypriceforecasting.vercel.app**

Run Demo now triggers a **live computed mini-run** through `/api/runs`, streams stage status in a Run Trace panel, and renders computed outputs with provenance.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]

pipeline ingest --start 2021-01-01 --end 2024-12-31
pipeline train --asof 2024-12-31
pipeline backtest --start 2023-01-01 --end 2024-12-31
pipeline report --run-id <run_id>
```

All outputs are written under `artifacts/<run_id>/`.

## Frontend Dashboard

```bash
source .venv/bin/activate
pip install -e '.[dev]'
FLASK_APP=api.app flask run -p 8080
```

Then open `http://localhost:8080`.

- `Run Live Computation` starts a backend run via `POST /api/runs`, polls live status, then renders computed results.
- `Upload Precomputed Run` accepts a JSON payload and renders with schema validation + fallback handling.
- `Run Trace` panel shows stages/progress/error.
- `Data Provenance` panel shows date range, row counts, selected model, and artifact references.
- `Replay Tour` reruns the walkthrough after results are rendered.

Expected JSON keys for upload:

- `run_id`
- `summary`
- `model_comparison`
- `equity_curve`
- `drawdown`
- `feature_importance`
- `failure_periods`

## API Endpoints

- `POST /api/runs` with `{ \"mode\": \"demo_quick\" }` -> `{ \"run_id\": ... }`
- `GET /api/runs/<run_id>` -> status, progress, stage, trace, error
- `GET /api/runs/<run_id>/results` -> computed results + provenance + timings

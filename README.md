# Commodity Price Forecasting Model

End-to-end WTI next-day direction forecasting pipeline using weather + yfinance data.

## Live Demo

Frontend is live at: **https://commoditypriceforecasting.vercel.app**

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
cd frontend
python3 -m http.server 8080
```

Then open `http://localhost:8080`.

- `Run Live Demo` scrolls to the dashboard and starts a guided tour.
- `Upload Data` accepts a JSON payload and renders with schema validation + fallback handling.
- `Replay Tour` reruns the live walkthrough.

Expected JSON keys for upload:

- `run_id`
- `summary`
- `model_comparison`
- `equity_curve`
- `drawdown`
- `feature_importance`
- `failure_periods`

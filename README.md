# Commodity Price Forecasting Model

Live demo: https://commoditypriceforecasting.vercel.app/

End-to-end WTI crude oil next-day direction forecasting project with:

- A Python ML pipeline for ingestion, feature engineering, walk-forward training, backtesting, and reporting
- A Flask API that exposes live run creation, status polling, and results retrieval
- A Vercel-hosted frontend dashboard that lets users trigger a live demo run and inspect trace, provenance, metrics, charts, and failure windows

## What This Project Does

This project forecasts the next trading day's WTI direction using a mix of:

- Market data from `yfinance`
- Weather-derived demand proxies from Open-Meteo
- Engineered lag, volatility, momentum, calendar, and temperature features

The modeling workflow compares:

- Logistic Regression
- Random Forest
- LightGBM

Models are evaluated with walk-forward validation, then ranked using trading-oriented metrics instead of pure classification accuracy alone. The downstream backtest applies a long/flat policy using probability thresholds and transaction costs.

## Live Demo

The public app is available at:

- https://commoditypriceforecasting.vercel.app/

What users can do in the demo:

- Trigger a live run from the browser
- Watch stage-by-stage execution in the Run Trace panel
- Review provenance such as date range, row counts, selected model, and artifact references
- Compare models on AUC, balanced accuracy, information ratio, and Sharpe
- Inspect equity curve, drawdown, feature importance, and failure windows
- Upload a precomputed JSON run and render it through the same dashboard

Note: the hosted demo uses a lightweight serverless-safe runner backed by a cached demo payload so the UX stays responsive on Vercel. The local Python pipeline is the full implementation for ingestion, training, backtesting, and report generation.

## Project Architecture

```text
frontend/        Static dashboard UI
api/             Flask app with /api/runs endpoints
backend/         Demo run orchestration + SQLite-backed run state
src/             Core forecasting package
config/          YAML configuration
tests/           Unit and behavior tests
vercel.json      Vercel routing/build config
```

High-level flow:

1. The frontend calls `POST /api/runs`.
2. The Flask app creates a run record and queues work.
3. The backend runner simulates a live computation flow and records stage/provenance data.
4. The frontend polls run status and results.
5. The dashboard renders metrics, charts, trace logs, and artifacts.

Separately, the local CLI runs the full data science pipeline against real data providers and writes outputs under `artifacts/<run_id>/`.

## ML Pipeline

### 1. Ingestion

Source data is fetched from:

- `CL=F` for primary WTI futures pricing
- Context tickers `DX-Y.NYB` and `^VIX`
- Open-Meteo archive API for weighted temperature signals across New York, Chicago, Houston, and Los Angeles

Generated datasets include:

- `prices_daily.parquet`
- `weather_daily.parquet`

### 2. Feature Engineering

The feature builder creates:

- Lagged returns
- Rolling volatility features
- Moving-average momentum features
- Calendar features such as weekday and month
- Heating degree day, cooling degree day, anomaly, and one-day delta weather features

The prediction target is `target_dir_t1`, the sign of the next day's forward return.

### 3. Training and Model Selection

The training module:

- Uses walk-forward splits
- Fits all three candidate models on the same feature set
- Scores both classification and trading outcomes
- Selects the best model by `information_ratio`
- Saves the fitted winner as a `.joblib` artifact

### 4. Backtesting

The backtest:

- Converts predicted probabilities into signals using `p_threshold`
- Shifts positions to avoid lookahead
- Applies transaction costs only when positions change
- Produces return, exposure, turnover, drawdown, CAGR, Sharpe, and information ratio outputs

### 5. Reporting

The reporting step generates:

- Equity curve plot
- Drawdown plot
- Summary metrics JSON
- Feature drift JSON
- SHAP outputs for supported tree models
- A notebook scaffold under `reports/report.ipynb`

## Configuration

Main config file:

- `config/default.yaml`

Key configurable areas:

- Tickers and weather cities
- Feature windows
- Training window sizes
- Model hyperparameters
- Strategy probability threshold and transaction costs
- Artifact output directory

## Quickstart

### Prerequisites

- Python 3.9+
- `pip`

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

### Run The Full Pipeline

```bash
pipeline ingest --start 2021-01-01 --end 2024-12-31
pipeline train
pipeline backtest --start 2023-01-01 --end 2024-12-31
pipeline report --run-id <run_id>
```

Notes:

- `ingest` writes a new run under `artifacts/<run_id>/`
- `train` and `backtest` default to the latest run if `--run-id` is omitted
- `report` requires an explicit `--run-id`

## Run The API And Frontend Locally

The frontend is served as static files and the API is a Flask app.

```bash
source .venv/bin/activate
pip install -e '.[dev]'
flask --app api.app run --port 8080
```

Then open:

- `http://127.0.0.1:8080`

Useful API routes:

- `POST /api/runs` with `{"mode":"demo_quick"}`
- `GET /api/runs/<run_id>`
- `GET /api/runs/<run_id>/results`

## Example Output Layout

Typical run artifacts:

```text
artifacts/<run_id>/
  datasets/
    prices_daily.parquet
    weather_daily.parquet
    model_table.parquet
    predictions.parquet
    trades.parquet
  metrics/
    model_metrics.parquet
    model_metrics.json
    backtest_metrics.parquet
    metrics.json
  models/
    best_model.json
    <best_model>.joblib
  plots/
    equity_curve.png
    drawdown.png
  reports/
    summary_metrics.json
    feature_drift.json
    report.ipynb
  shap/
    shap_values.npy
    shap_features.npy
    shap_summary.png
```

## Testing

Run the test suite with:

```bash
pytest
```

Current tests cover:

- API run creation and incomplete-result behavior
- Walk-forward split ordering
- Backtest cost application
- Leakage and label correctness
- Validation checks for duplicates, nulls, and missing columns
- Deterministic metrics across repeated model runs

## Deployment

The repository is configured for Vercel:

- `api/app.py` is deployed as a Python serverless function
- `frontend/` is deployed as static assets
- `vercel.json` routes `/api/*` to Flask and everything else to the frontend

The live site:

- https://commoditypriceforecasting.vercel.app/

## Tech Stack

- Python
- Polars
- Pandas
- scikit-learn
- LightGBM
- SHAP
- Flask
- SQLite
- HTML/CSS/JavaScript
- Vercel

## Current Limitations

- The public live demo uses a lightweight cached payload rather than full heavy model training in the serverless path
- Data coverage is intentionally narrow and focused on WTI, macro context tickers, and weather proxies
- Current strategy implementation is long/flat only
- More production-grade queueing and worker infrastructure would be needed for large-scale real-time execution

## Repository Goal

This codebase is structured as both:

- A working ML forecasting pipeline
- A portfolio-ready product demo that shows not just model output, but also execution traceability, provenance, and deployment awareness

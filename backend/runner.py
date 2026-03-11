from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl

from commodity_forecasting.backtest import performance_metrics, run_backtest
from commodity_forecasting.config import load_config
from commodity_forecasting.features import build_model_table
from commodity_forecasting.ingest import run_ingestion
from commodity_forecasting.modeling import train_and_select_model
from commodity_forecasting.utils import ensure_run_dirs


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ts(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.isoformat()


def _compute_drawdown_curve(net_ret: np.ndarray) -> np.ndarray:
    eq = np.cumprod(1.0 + net_ret)
    peak = np.maximum.accumulate(eq)
    return eq / peak - 1.0


def _failure_windows(trades: pl.DataFrame, top_n: int = 2, window: int = 5) -> list[dict[str, str]]:
    if trades.height < window:
        return []
    t = trades.sort("date")
    net = t["net_ret"].to_numpy()
    dates = t["date"].to_list()
    sums = []
    for i in range(0, len(net) - window + 1):
        sums.append((i, float(net[i : i + window].sum())))
    worst = sorted(sums, key=lambda x: x[1])[:top_n]

    rows = []
    for idx, _ in worst:
        start = dates[idx]
        end = dates[idx + window - 1]
        rows.append({"window": f"{start} to {end}", "note": "Underperformance cluster during local regime shift."})
    return rows


def _feature_importance_from_model(model_path: Path) -> list[list[Any]]:
    bundle = joblib.load(model_path)
    pipe = bundle["model"]
    feat = bundle["features"]
    model = pipe.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
    elif hasattr(model, "coef_"):
        vals = np.abs(model.coef_[0])
    else:
        vals = np.zeros(len(feat))

    pairs = sorted(zip(feat, vals), key=lambda x: x[1], reverse=True)
    return [[str(k), float(v)] for k, v in pairs[:12]]


def run_demo_quick(run_id: str, update_stage) -> dict[str, Any]:
    cfg = load_config()
    # Override for stable quick demo runtime.
    cfg["training"]["train_window_days"] = 140
    cfg["training"]["test_window_days"] = 14
    cfg["training"]["min_train_rows"] = 100
    cfg["models"]["random_forest"]["n_estimators"] = 120
    cfg["models"]["lightgbm"]["n_estimators"] = 120

    end_dt = _utc_now().date() - timedelta(days=1)
    start_dt = end_dt - timedelta(days=220)
    start = start_dt.isoformat()
    end = end_dt.isoformat()

    dirs = ensure_run_dirs(cfg["artifacts"]["base_dir"], run_id)

    update_stage("ingest", 15.0, {"window_start": start, "window_end": end})
    ingest = run_ingestion(cfg, start=start, end=end)
    ingest.prices_daily.write_parquet(dirs["datasets"] / "prices_daily.parquet")
    ingest.weather_daily.write_parquet(dirs["datasets"] / "weather_daily.parquet")

    update_stage("feature_engineering", 35.0, {"price_rows": ingest.prices_daily.height, "weather_rows": ingest.weather_daily.height})
    model_table = build_model_table(ingest.prices_daily, ingest.weather_daily, cfg)
    model_table.write_parquet(dirs["datasets"] / "model_table.parquet")

    update_stage("train_compare", 65.0, {"model_rows": model_table.height})
    outputs = train_and_select_model(model_table, cfg, dirs["models"], dirs["metrics"])
    outputs.predictions.write_parquet(dirs["datasets"] / "predictions.parquet")
    outputs.metrics_table.write_parquet(dirs["metrics"] / "model_metrics.parquet")

    best_model = outputs.best_model_name
    best_preds = outputs.predictions.filter(pl.col("model_name") == best_model).sort("date")

    update_stage("backtest", 85.0, {"best_model": best_model, "prediction_rows": best_preds.height})
    tc = float(cfg["strategy"]["transaction_cost_bps"])
    p_thr = float(cfg["strategy"]["p_threshold"])
    trades = run_backtest(best_preds, transaction_cost_bps=tc, p_threshold=p_thr)
    bt = performance_metrics(trades)
    trades.write_parquet(dirs["datasets"] / "trades.parquet")

    update_stage("summarize", 97.0, {"trade_rows": trades.height})
    best_row = outputs.metrics_table.filter(pl.col("model_name") == best_model).to_dicts()[0]

    eq = np.cumprod(1.0 + trades["net_ret"].to_numpy())
    dd = _compute_drawdown_curve(trades["net_ret"].to_numpy())

    equity_curve = [[str(d), float(v)] for d, v in zip(trades["date"].to_list(), eq)]
    drawdown = [[str(d), float(v)] for d, v in zip(trades["date"].to_list(), dd)]

    summary = {
        "best_model": best_model,
        "auc": float(best_row["auc"]),
        "balanced_accuracy": float(best_row["balanced_accuracy"]),
        "information_ratio": float(best_row["information_ratio"]),
        "cagr": float(bt["cagr"]),
        "sharpe": float(bt["sharpe"]),
        "max_drawdown": float(bt["max_drawdown"]),
        "exposure": float(bt["exposure"]),
    }

    comparison = outputs.metrics_table.select(["model_name", "auc", "balanced_accuracy", "information_ratio", "sharpe"]).to_dicts()
    feature_importance = _feature_importance_from_model(outputs.best_model_path)
    failures = _failure_windows(trades)

    results = {
        "run_id": run_id,
        "summary": summary,
        "model_comparison": comparison,
        "equity_curve": equity_curve,
        "drawdown": drawdown,
        "feature_importance": feature_importance,
        "failure_periods": failures,
    }

    provenance = {
        "date_range": {"start": start, "end": end},
        "row_counts": {
            "prices": ingest.prices_daily.height,
            "weather": ingest.weather_daily.height,
            "model_table": model_table.height,
            "predictions_best_model": best_preds.height,
            "trades": trades.height,
        },
        "selected_model": best_model,
        "artifacts": [
            str(dirs["datasets"] / "prices_daily.parquet"),
            str(dirs["datasets"] / "weather_daily.parquet"),
            str(dirs["datasets"] / "model_table.parquet"),
            str(dirs["datasets"] / "trades.parquet"),
            str(dirs["models"] / f"{best_model}.joblib"),
        ],
        "model_params": {
            "probability_threshold": p_thr,
            "transaction_cost_bps": tc,
            "train_window_days": cfg["training"]["train_window_days"],
            "test_window_days": cfg["training"]["test_window_days"],
        },
    }

    return {
        "results": results,
        "provenance": provenance,
    }

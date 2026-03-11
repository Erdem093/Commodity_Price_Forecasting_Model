from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import shap



def _save_equity_plot(trades: pl.DataFrame, output_path: Path) -> None:
    t = trades.sort("date")
    eq = (1.0 + t["net_ret"].to_numpy()).cumprod()
    dates = t["date"].to_list()
    plt.figure(figsize=(10, 4))
    plt.plot(dates, eq, label="Strategy Equity")
    plt.title("Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _save_drawdown_plot(trades: pl.DataFrame, output_path: Path) -> None:
    t = trades.sort("date")
    eq = (1.0 + t["net_ret"].to_numpy()).cumprod()
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    dates = t["date"].to_list()

    plt.figure(figsize=(10, 4))
    plt.plot(dates, dd, color="tomato", label="Drawdown")
    plt.title("Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _feature_drift(train_df: pl.DataFrame, test_df: pl.DataFrame, features: list[str]) -> list[dict[str, float]]:
    rows = []
    for f in features:
        tr = train_df[f].to_numpy()
        te = test_df[f].to_numpy()
        tr_mean = float(np.nanmean(tr))
        te_mean = float(np.nanmean(te))
        tr_std = float(np.nanstd(tr))
        if tr_std <= 1e-12:
            z_shift = 0.0
        else:
            z_shift = (te_mean - tr_mean) / tr_std
        rows.append({"feature": f, "train_mean": tr_mean, "test_mean": te_mean, "z_shift": float(z_shift)})
    rows.sort(key=lambda x: abs(x["z_shift"]), reverse=True)
    return rows


def _build_notebook(metrics: list[dict[str, Any]], report_path: Path) -> None:
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# WTI Forecasting Report\\n",
                    "Auto-generated summary notebook for model/backtest outputs."
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import json\\n",
                    f"with open('{str(report_path.parent / 'summary_metrics.json')}', 'r') as f:\\n",
                    "    metrics = json.load(f)\\n",
                    "metrics"
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)


def generate_report(
    run_root: Path,
    model_table: pl.DataFrame,
    predictions: pl.DataFrame,
    trades: pl.DataFrame,
    metrics_table: pl.DataFrame,
    best_model_path: Path,
) -> None:
    plots_dir = run_root / "plots"
    shap_dir = run_root / "shap"
    reports_dir = run_root / "reports"
    for d in [plots_dir, shap_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    _save_equity_plot(trades, plots_dir / "equity_curve.png")
    _save_drawdown_plot(trades, plots_dir / "drawdown.png")

    metrics = metrics_table.to_dicts()
    with (reports_dir / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Feature drift: first 70% as train-like segment, last 30% as test-like segment.
    cutoff = int(model_table.height * 0.7)
    train_df = model_table.slice(0, cutoff)
    test_df = model_table.slice(cutoff, model_table.height - cutoff)
    feat_cols = [c for c in model_table.columns if c not in {"date", "target_dir_t1", "ret_fwd_1d"}]
    drift = _feature_drift(train_df, test_df, feat_cols)
    with (reports_dir / "feature_drift.json").open("w", encoding="utf-8") as f:
        json.dump(drift[:20], f, indent=2)

    # SHAP for tree models when available.
    best = joblib.load(best_model_path)
    pipe = best["model"]
    features = best["features"]

    best_model_name = metrics_table.sort("information_ratio", descending=True)["model_name"][0]
    if best_model_name in {"lightgbm", "random_forest"}:
        sample = model_table.select(features).to_pandas().tail(min(500, model_table.height))
        X_imp = pipe.named_steps["imputer"].transform(sample.values)
        tree_model = pipe.named_steps["model"]
        explainer = shap.TreeExplainer(tree_model)
        shap_values = explainer.shap_values(X_imp)

        if isinstance(shap_values, list):
            shap_arr = np.array(shap_values[1])
        else:
            shap_arr = np.array(shap_values)

        np.save(shap_dir / "shap_values.npy", shap_arr)
        np.save(shap_dir / "shap_features.npy", np.array(features, dtype=object))

        plt.figure(figsize=(10, 5))
        mean_abs = np.abs(shap_arr).mean(axis=0)
        order = np.argsort(mean_abs)[::-1][:15]
        names = np.array(features)[order]
        vals = mean_abs[order]
        plt.barh(names[::-1], vals[::-1])
        plt.title("SHAP Mean |Value| (Top 15)")
        plt.tight_layout()
        plt.savefig(shap_dir / "shap_summary.png")
        plt.close()

    report_nb = reports_dir / "report.ipynb"
    _build_notebook(metrics, report_nb)

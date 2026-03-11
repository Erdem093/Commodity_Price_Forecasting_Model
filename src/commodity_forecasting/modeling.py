from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from commodity_forecasting.backtest import performance_metrics, run_backtest
from commodity_forecasting.evaluation import classification_metrics, merge_metrics
from commodity_forecasting.features import feature_columns


@dataclass
class ModelOutputs:
    predictions: pl.DataFrame
    metrics_table: pl.DataFrame
    best_model_name: str
    best_model_path: Path


def _build_models(cfg: dict[str, Any]) -> dict[str, Pipeline]:
    seed = int(cfg["project"]["random_seed"])
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=float(cfg["models"]["logistic_regression"].get("C", 1.0)),
                        max_iter=int(cfg["models"]["logistic_regression"].get("max_iter", 1000)),
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=int(cfg["models"]["random_forest"].get("n_estimators", 300)),
                        max_depth=int(cfg["models"]["random_forest"].get("max_depth", 6)),
                        min_samples_leaf=int(cfg["models"]["random_forest"].get("min_samples_leaf", 5)),
                        random_state=seed,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "lightgbm": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=int(cfg["models"]["lightgbm"].get("n_estimators", 500)),
                        learning_rate=float(cfg["models"]["lightgbm"].get("learning_rate", 0.03)),
                        num_leaves=int(cfg["models"]["lightgbm"].get("num_leaves", 31)),
                        subsample=float(cfg["models"]["lightgbm"].get("subsample", 0.8)),
                        colsample_bytree=float(cfg["models"]["lightgbm"].get("colsample_bytree", 0.8)),
                        random_state=seed,
                        n_jobs=1,
                        verbosity=-1,
                    ),
                ),
            ]
        ),
    }


def _walk_forward_indices(n_rows: int, train_window: int, test_window: int, min_train_rows: int) -> list[tuple[slice, slice]]:
    if n_rows < min_train_rows + test_window:
        return []

    splits: list[tuple[slice, slice]] = []
    train_start = 0
    while True:
        train_end = min(train_start + train_window, n_rows - test_window)
        if train_end - train_start < min_train_rows:
            break
        test_end = min(train_end + test_window, n_rows)
        train_slice = slice(train_start, train_end)
        test_slice = slice(train_end, test_end)
        if test_slice.start >= test_slice.stop:
            break
        splits.append((train_slice, test_slice))
        if test_end >= n_rows:
            break
        train_start += test_window
    return splits


def _train_predict_single_model(model: Pipeline, model_name: str, table: pl.DataFrame, features: list[str], cfg: dict[str, Any]) -> pl.DataFrame:
    pdf = table.to_pandas()
    X = pdf[features].values
    y = pdf["target_dir_t1"].values.astype(int)

    splits = _walk_forward_indices(
        n_rows=len(pdf),
        train_window=int(cfg["training"].get("train_window_days", 504)),
        test_window=int(cfg["training"].get("test_window_days", 21)),
        min_train_rows=int(cfg["training"].get("min_train_rows", 252)),
    )
    if not splits:
        raise ValueError("Insufficient rows for walk-forward training. Increase history or reduce window sizes.")

    pred_parts: list[pl.DataFrame] = []
    for train_slice, test_slice in splits:
        X_train, y_train = X[train_slice], y[train_slice]
        X_test, y_test = X[test_slice], y[test_slice]

        model.fit(X_train, y_train)
        p_up = model.predict_proba(X_test)[:, 1]
        y_pred = (p_up >= 0.5).astype(int)

        part = pl.DataFrame(
            {
                "date": pdf.iloc[test_slice]["date"].values,
                "ret_fwd_1d": pdf.iloc[test_slice]["ret_fwd_1d"].values,
                "y_true": y_test,
                "p_up": p_up,
                "y_pred": y_pred,
                "model_name": [model_name] * len(y_test),
            }
        )
        pred_parts.append(part)

    return pl.concat(pred_parts).sort("date")


def train_and_select_model(model_table: pl.DataFrame, cfg: dict[str, Any], model_dir: Path, metrics_dir: Path) -> ModelOutputs:
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    feats = feature_columns(model_table)
    models = _build_models(cfg)

    threshold = float(cfg["strategy"].get("p_threshold", 0.55))
    tc_bps = float(cfg["strategy"].get("transaction_cost_bps", 3.0))

    pred_frames = []
    rows = []

    for name, model in models.items():
        preds = _train_predict_single_model(model, name, model_table, feats, cfg)
        trades = run_backtest(preds, transaction_cost_bps=tc_bps, p_threshold=threshold)
        cls = classification_metrics(preds)
        bt = performance_metrics(trades)
        rows.append(merge_metrics(name, cls, bt))
        pred_frames.append(preds)

    metrics_table = pl.DataFrame(rows).sort("information_ratio", descending=True)
    best_model_name = metrics_table["model_name"][0]

    # Fit best model on all available data for downstream inference/reporting.
    best_model = models[best_model_name]
    pdf = model_table.to_pandas()
    best_model.fit(pdf[feats].values, pdf["target_dir_t1"].values.astype(int))
    best_model_path = model_dir / f"{best_model_name}.joblib"
    joblib.dump({"model": best_model, "features": feats}, best_model_path)

    predictions = pl.concat(pred_frames).sort(["date", "model_name"])

    metrics_json_path = metrics_dir / "model_metrics.json"
    with metrics_json_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_table.to_dicts(), f, indent=2)

    return ModelOutputs(
        predictions=predictions,
        metrics_table=metrics_table,
        best_model_name=str(best_model_name),
        best_model_path=best_model_path,
    )

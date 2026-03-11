from __future__ import annotations

from datetime import date

import polars as pl

from commodity_forecasting.modeling import train_and_select_model


def _synthetic_table() -> pl.DataFrame:
    n = 700
    dates = pl.date_range(date(2020, 1, 1), date(2021, 12, 31), interval="1d", eager=True)[:n]
    x1 = [((i % 10) - 5) / 10 for i in range(n)]
    x2 = [((i % 7) - 3) / 10 for i in range(n)]
    ret = [0.002 * (1 if (a + b) > 0 else -1) for a, b in zip(x1, x2)]
    y = [1 if r > 0 else 0 for r in ret]
    return pl.DataFrame(
        {
            "date": dates,
            "open": [50 + i * 0.01 for i in range(n)],
            "high": [50 + i * 0.01 + 0.2 for i in range(n)],
            "low": [50 + i * 0.01 - 0.2 for i in range(n)],
            "close": [50 + i * 0.01 for i in range(n)],
            "volume": [1000 + i for i in range(n)],
            "ret_1d": ret,
            "ret_fwd_1d": ret,
            "target_dir_t1": y,
            "ret_lag_1": x1,
            "ret_lag_2": x2,
            "vol_5": [abs(v) for v in x1],
            "mom_ma_5": x2,
            "dow": [i % 5 for i in range(n)],
            "month": [(i % 12) + 1 for i in range(n)],
            "hdd_us_l1": [1.0] * n,
            "cdd_us_l1": [2.0] * n,
            "temp_anom_us_l1": [0.1] * n,
            "hdd_delta_1d": [0.0] * n,
            "cdd_delta_1d": [0.0] * n,
            "temp_anom_delta_1d": [0.0] * n,
        }
    )


def test_reproducible_metrics(tmp_path) -> None:
    cfg = {
        "project": {"random_seed": 42},
        "training": {"train_window_days": 252, "test_window_days": 21, "min_train_rows": 200},
        "strategy": {"p_threshold": 0.55, "transaction_cost_bps": 3.0},
        "models": {
            "logistic_regression": {"C": 1.0, "max_iter": 200},
            "random_forest": {"n_estimators": 50, "max_depth": 4, "min_samples_leaf": 2},
            "lightgbm": {"n_estimators": 50, "learning_rate": 0.05, "num_leaves": 15, "subsample": 0.9, "colsample_bytree": 0.9},
        },
    }

    table = _synthetic_table()

    out1 = train_and_select_model(table, cfg, tmp_path / "m1", tmp_path / "k1")
    out2 = train_and_select_model(table, cfg, tmp_path / "m2", tmp_path / "k2")

    m1 = out1.metrics_table.sort("model_name")
    m2 = out2.metrics_table.sort("model_name")

    assert m1["information_ratio"].to_list() == m2["information_ratio"].to_list()
    assert m1["auc"].to_list() == m2["auc"].to_list()

from __future__ import annotations

from typing import Any

import polars as pl


def _add_price_features(df: pl.DataFrame, cfg: dict[str, Any]) -> pl.DataFrame:
    out = df.sort("date")

    for lag in cfg["features"]["return_lags"]:
        out = out.with_columns(pl.col("ret_1d").shift(lag).alias(f"ret_lag_{lag}"))

    for w in cfg["features"]["vol_windows"]:
        out = out.with_columns(pl.col("ret_1d").rolling_std(window_size=w).shift(1).alias(f"vol_{w}"))

    for w in cfg["features"]["ma_windows"]:
        out = out.with_columns(
            (pl.col("close") / pl.col("close").rolling_mean(window_size=w).shift(1) - 1.0).alias(f"mom_ma_{w}")
        )

    out = out.with_columns(
        [
            pl.col("date").dt.weekday().alias("dow"),
            pl.col("date").dt.month().alias("month"),
        ]
    )
    return out


def _add_weather_features(df: pl.DataFrame) -> pl.DataFrame:
    out = df.sort("date")
    out = out.with_columns(
        [
            pl.col("hdd_us").shift(1).alias("hdd_us_l1"),
            pl.col("cdd_us").shift(1).alias("cdd_us_l1"),
            pl.col("temp_anom_us").shift(1).alias("temp_anom_us_l1"),
            (pl.col("hdd_us").shift(1) - pl.col("hdd_us").shift(2)).alias("hdd_delta_1d"),
            (pl.col("cdd_us").shift(1) - pl.col("cdd_us").shift(2)).alias("cdd_delta_1d"),
            (pl.col("temp_anom_us").shift(1) - pl.col("temp_anom_us").shift(2)).alias("temp_anom_delta_1d"),
        ]
    )
    return out


def build_model_table(prices_daily: pl.DataFrame, weather_daily: pl.DataFrame, cfg: dict[str, Any]) -> pl.DataFrame:
    p = _add_price_features(prices_daily, cfg)
    w = _add_weather_features(weather_daily)

    df = p.join(w, on="date", how="left")
    df = df.with_columns(
        [
            pl.col("ret_1d").shift(-1).alias("ret_fwd_1d"),
            (pl.col("ret_1d").shift(-1) > 0).cast(pl.Int8).alias("target_dir_t1"),
        ]
    )

    keep_cols = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ret_1d",
        "ret_fwd_1d",
        "target_dir_t1",
    ] + [
        c for c in df.columns if c not in {"ticker", "date", "open", "high", "low", "close", "volume", "ret_1d", "ret_fwd_1d", "target_dir_t1"}
    ]

    df = df.select(keep_cols)
    df = df.sort("date")
    # Drop rows where feature/label availability is incomplete.
    df = df.drop_nulls()
    return df


def feature_columns(model_table: pl.DataFrame) -> list[str]:
    excluded = {"date", "target_dir_t1", "ret_fwd_1d", "ticker"}
    return [c for c in model_table.columns if c not in excluded]

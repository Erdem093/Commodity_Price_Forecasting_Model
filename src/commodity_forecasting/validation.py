from __future__ import annotations

import polars as pl


def validate_time_series(df: pl.DataFrame, date_col: str = "date") -> None:
    if df.height == 0:
        raise ValueError("DataFrame is empty")
    if df.select(pl.col(date_col).is_duplicated().any()).item():
        raise ValueError(f"Duplicate dates found in {date_col}")
    sorted_df = df.sort(date_col)
    if not df[date_col].to_list() == sorted_df[date_col].to_list():
        raise ValueError(f"{date_col} is not monotonic increasing")


def validate_model_table(df: pl.DataFrame) -> None:
    required = {"date", "ret_fwd_1d", "target_dir_t1"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.null_count().sum_horizontal().sum() > 0:
        raise ValueError("Model table contains null values")

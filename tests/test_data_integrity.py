from __future__ import annotations

import polars as pl
import pytest

from commodity_forecasting.validation import validate_model_table, validate_time_series


def test_validate_time_series_rejects_duplicates() -> None:
    df = pl.DataFrame({"date": ["2024-01-01", "2024-01-01"], "x": [1, 2]})
    with pytest.raises(ValueError, match="Duplicate"):
        validate_time_series(df)


def test_validate_time_series_rejects_non_monotonic() -> None:
    df = pl.DataFrame({"date": ["2024-01-02", "2024-01-01"], "x": [1, 2]})
    with pytest.raises(ValueError, match="monotonic"):
        validate_time_series(df)


def test_validate_model_table_requires_core_columns() -> None:
    df = pl.DataFrame({"date": ["2024-01-01"], "target_dir_t1": [1]})
    with pytest.raises(ValueError, match="Missing"):
        validate_model_table(df)


def test_validate_model_table_rejects_nulls() -> None:
    df = pl.DataFrame(
        {
            "date": ["2024-01-01"],
            "ret_fwd_1d": [None],
            "target_dir_t1": [1],
        }
    )
    with pytest.raises(ValueError, match="null"):
        validate_model_table(df)

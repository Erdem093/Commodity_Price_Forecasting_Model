from __future__ import annotations

from datetime import date

import polars as pl

from commodity_forecasting.features import build_model_table


def test_target_dir_t1_is_next_day_return_sign() -> None:
    cfg = {
        "features": {"return_lags": [1], "vol_windows": [2], "ma_windows": [2], "weather_anom_window": 30},
    }
    prices = pl.DataFrame(
        {
            "date": pl.date_range(date(2024, 1, 1), date(2024, 1, 10), eager=True),
            "open": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "high": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "low": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "close": [10, 11, 10, 12, 11, 13, 12, 14, 13, 15],
            "volume": [100] * 10,
            "ret_1d": [None, 0.1, -0.0909, 0.2, -0.0833, 0.1818, -0.0769, 0.1667, -0.0714, 0.1538],
        }
    ).drop_nulls()

    weather = pl.DataFrame(
        {
            "date": pl.date_range(date(2024, 1, 1), date(2024, 1, 10), eager=True),
            "hdd_us": [1.0] * 10,
            "cdd_us": [2.0] * 10,
            "temp_anom_us": [0.1] * 10,
        }
    )

    table = build_model_table(prices, weather, cfg)

    # ret_fwd_1d and target_dir_t1 should have matching sign logic.
    mismatch = table.filter(((pl.col("ret_fwd_1d") > 0).cast(pl.Int8)) != pl.col("target_dir_t1"))
    assert mismatch.height == 0

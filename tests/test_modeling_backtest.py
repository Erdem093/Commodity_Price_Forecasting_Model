from __future__ import annotations

from datetime import date

import polars as pl

from commodity_forecasting.backtest import run_backtest
from commodity_forecasting.modeling import _walk_forward_indices


def test_walk_forward_indices_respect_temporal_order() -> None:
    splits = _walk_forward_indices(n_rows=300, train_window=120, test_window=20, min_train_rows=100)
    assert len(splits) > 0
    for tr, te in splits:
        assert tr.stop <= te.start


def test_backtest_applies_cost_on_position_changes_only() -> None:
    preds = pl.DataFrame(
        {
            "date": pl.date_range(date(2024, 1, 1), date(2024, 1, 6), eager=True),
            "ret_fwd_1d": [0.01, -0.01, 0.02, -0.01, 0.0, 0.01],
            "y_true": [1, 0, 1, 0, 0, 1],
            "p_up": [0.8, 0.3, 0.7, 0.4, 0.2, 0.9],
            "y_pred": [1, 0, 1, 0, 0, 1],
            "model_name": ["m"] * 6,
        }
    )

    trades = run_backtest(preds, transaction_cost_bps=10.0, p_threshold=0.55)
    # position is shifted signal => [0,1,0,1,0,0], changes on days 2,3,4,5 (4 turns)
    total_cost = float(trades["cost"].sum())
    assert abs(total_cost - 0.004) < 1e-9

from __future__ import annotations

import math

import polars as pl


def apply_signal_policy(predictions: pl.DataFrame, p_threshold: float) -> pl.DataFrame:
    df = predictions.sort("date")
    df = df.with_columns((pl.col("p_up") >= p_threshold).cast(pl.Int8).alias("signal"))
    # Position on t+1 is yesterday's signal because signal is generated after close on t.
    df = df.with_columns(pl.col("signal").shift(1).fill_null(0).cast(pl.Int8).alias("position"))
    return df


def run_backtest(predictions: pl.DataFrame, transaction_cost_bps: float, p_threshold: float) -> pl.DataFrame:
    df = apply_signal_policy(predictions, p_threshold=p_threshold)
    cost_per_turn = transaction_cost_bps / 10_000.0

    df = df.with_columns(
        [
            pl.col("position").diff().abs().fill_null(pl.col("position")).alias("turnover_units"),
            (pl.col("position") * pl.col("ret_fwd_1d")).alias("gross_ret"),
        ]
    )
    df = df.with_columns((pl.col("turnover_units") * cost_per_turn).alias("cost"))
    df = df.with_columns((pl.col("gross_ret") - pl.col("cost")).alias("net_ret"))

    return df.select(
        [
            "date",
            "model_name",
            "p_up",
            "y_true",
            "y_pred",
            "signal",
            "position",
            "ret_fwd_1d",
            "gross_ret",
            "cost",
            "net_ret",
            "turnover_units",
        ]
    )


def _safe_div(a: float, b: float) -> float:
    return 0.0 if abs(b) < 1e-12 else a / b


def performance_metrics(trades: pl.DataFrame) -> dict[str, float]:
    if trades.height == 0:
        return {
            "n_days": 0,
            "cagr": 0.0,
            "ann_vol": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "hit_rate": 0.0,
            "turnover": 0.0,
            "exposure": 0.0,
            "avg_gross_ret": 0.0,
            "avg_net_ret": 0.0,
            "information_ratio": 0.0,
        }

    net = trades["net_ret"].to_numpy()
    gross = trades["gross_ret"].to_numpy()
    pos = trades["position"].to_numpy()
    n = len(net)

    mean_d = float(net.mean())
    std_d = float(net.std(ddof=1)) if n > 1 else 0.0
    ann_factor = 252.0

    ann_vol = std_d * math.sqrt(ann_factor)
    sharpe = _safe_div(mean_d, std_d) * math.sqrt(ann_factor) if std_d > 0 else 0.0
    cagr = float((1.0 + net).prod() ** (ann_factor / n) - 1.0)

    equity = (1.0 + net).cumprod()
    peak = equity.copy()
    for i in range(1, len(peak)):
        peak[i] = max(peak[i], peak[i - 1])
    drawdowns = equity / peak - 1.0
    mdd = float(drawdowns.min()) if len(drawdowns) else 0.0

    active_idx = pos > 0
    hit_rate = float((gross[active_idx] > 0).mean()) if active_idx.any() else 0.0

    benchmark = trades["ret_fwd_1d"].to_numpy()
    active = net - benchmark
    ir = 0.0
    if len(active) > 1:
        act_std = float(active.std(ddof=1))
        if act_std > 0:
            ir = float(active.mean()) / act_std * math.sqrt(ann_factor)

    return {
        "n_days": int(n),
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "hit_rate": hit_rate,
        "turnover": float(trades["turnover_units"].mean()),
        "exposure": float(pos.mean()),
        "avg_gross_ret": float(gross.mean()),
        "avg_net_ret": float(net.mean()),
        "information_ratio": ir,
    }

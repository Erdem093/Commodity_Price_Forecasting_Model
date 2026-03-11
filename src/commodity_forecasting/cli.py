from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import polars as pl

from commodity_forecasting.backtest import performance_metrics, run_backtest
from commodity_forecasting.config import load_config
from commodity_forecasting.features import build_model_table
from commodity_forecasting.ingest import run_ingestion
from commodity_forecasting.modeling import train_and_select_model
from commodity_forecasting.reporting import generate_report
from commodity_forecasting.utils import ensure_run_dirs, make_run_id
from commodity_forecasting.validation import validate_model_table, validate_time_series


def _base_dir(cfg: dict[str, Any]) -> Path:
    return Path(cfg["artifacts"].get("base_dir", "artifacts"))


def _find_latest_run(base_dir: Path) -> str:
    runs = sorted([p.name for p in base_dir.glob("run_*") if p.is_dir()])
    if not runs:
        raise ValueError("No run directories found. Execute ingest first or pass --run-id.")
    return runs[-1]


def cmd_ingest(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    run_id = args.run_id or make_run_id("run")
    dirs = ensure_run_dirs(_base_dir(cfg), run_id)

    result = run_ingestion(cfg, start=args.start, end=args.end)
    validate_time_series(result.prices_daily)
    validate_time_series(result.weather_daily)

    result.prices_daily.write_parquet(dirs["datasets"] / "prices_daily.parquet")
    result.weather_daily.write_parquet(dirs["datasets"] / "weather_daily.parquet")

    metadata = {"run_id": run_id, "start": args.start, "end": args.end}
    with (dirs["root"] / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Ingestion complete. run_id={run_id}")


def cmd_train(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    base_dir = _base_dir(cfg)
    run_id = args.run_id or _find_latest_run(base_dir)
    dirs = ensure_run_dirs(base_dir, run_id)

    prices = pl.read_parquet(dirs["datasets"] / "prices_daily.parquet")
    weather = pl.read_parquet(dirs["datasets"] / "weather_daily.parquet")

    model_table = build_model_table(prices, weather, cfg)
    validate_model_table(model_table)
    model_table.write_parquet(dirs["datasets"] / "model_table.parquet")

    outputs = train_and_select_model(model_table, cfg, dirs["models"], dirs["metrics"])
    outputs.predictions.write_parquet(dirs["datasets"] / "predictions.parquet")
    outputs.metrics_table.write_parquet(dirs["metrics"] / "model_metrics.parquet")

    with (dirs["models"] / "best_model.json").open("w", encoding="utf-8") as f:
        json.dump({"best_model_name": outputs.best_model_name, "model_path": str(outputs.best_model_path)}, f, indent=2)

    print(f"Training complete. run_id={run_id}, best_model={outputs.best_model_name}")


def cmd_backtest(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    base_dir = _base_dir(cfg)
    run_id = args.run_id or _find_latest_run(base_dir)
    dirs = ensure_run_dirs(base_dir, run_id)

    preds = pl.read_parquet(dirs["datasets"] / "predictions.parquet")

    if args.start:
        preds = preds.filter(pl.col("date") >= pl.lit(args.start))
    if args.end:
        preds = preds.filter(pl.col("date") <= pl.lit(args.end))

    threshold = float(cfg["strategy"].get("p_threshold", 0.55))
    tc_bps = float(cfg["strategy"].get("transaction_cost_bps", 3.0))

    all_trades = []
    summary = []
    for model_name in preds["model_name"].unique().to_list():
        p = preds.filter(pl.col("model_name") == model_name).sort("date")
        t = run_backtest(p, transaction_cost_bps=tc_bps, p_threshold=threshold)
        all_trades.append(t)
        met = performance_metrics(t)
        summary.append({"model_name": model_name, **met})

    trades = pl.concat(all_trades).sort(["date", "model_name"])
    metrics = pl.DataFrame(summary).sort("information_ratio", descending=True)

    trades.write_parquet(dirs["datasets"] / "trades.parquet")
    metrics.write_parquet(dirs["metrics"] / "backtest_metrics.parquet")

    with (dirs["metrics"] / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics.to_dicts(), f, indent=2)

    print(f"Backtest complete. run_id={run_id}")


def cmd_report(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    base_dir = _base_dir(cfg)
    run_id = args.run_id or _find_latest_run(base_dir)
    dirs = ensure_run_dirs(base_dir, run_id)

    model_table = pl.read_parquet(dirs["datasets"] / "model_table.parquet")
    preds = pl.read_parquet(dirs["datasets"] / "predictions.parquet")
    trades = pl.read_parquet(dirs["datasets"] / "trades.parquet")
    metrics = pl.read_parquet(dirs["metrics"] / "model_metrics.parquet")

    with (dirs["models"] / "best_model.json").open("r", encoding="utf-8") as f:
        best = json.load(f)

    generate_report(
        run_root=dirs["root"],
        model_table=model_table,
        predictions=preds,
        trades=trades,
        metrics_table=metrics,
        best_model_path=Path(best["model_path"]),
    )

    print(f"Report generated. run_id={run_id}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WTI Forecasting Pipeline")
    parser.add_argument("--config", default="config/default.yaml")

    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Fetch source data")
    p_ingest.add_argument("--start", required=True)
    p_ingest.add_argument("--end", required=True)
    p_ingest.add_argument("--run-id", default=None)
    p_ingest.set_defaults(func=cmd_ingest)

    p_train = sub.add_parser("train", help="Train models and select winner")
    p_train.add_argument("--asof", required=False, default=None)
    p_train.add_argument("--run-id", default=None)
    p_train.set_defaults(func=cmd_train)

    p_backtest = sub.add_parser("backtest", help="Run strategy backtest")
    p_backtest.add_argument("--start", required=False, default=None)
    p_backtest.add_argument("--end", required=False, default=None)
    p_backtest.add_argument("--run-id", default=None)
    p_backtest.set_defaults(func=cmd_backtest)

    p_report = sub.add_parser("report", help="Generate plots/report artifacts")
    p_report.add_argument("--run-id", required=True)
    p_report.set_defaults(func=cmd_report)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

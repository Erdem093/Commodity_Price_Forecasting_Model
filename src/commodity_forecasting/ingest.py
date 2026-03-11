from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import requests
import yfinance as yf


OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"


@dataclass
class IngestResult:
    prices_daily: pl.DataFrame
    weather_daily: pl.DataFrame


def _download_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    last_err: Exception | None = None
    df = pd.DataFrame()
    for attempt in range(4):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if not df.empty:
                break
        except Exception as err:  # pragma: no cover - network/provider dependent
            last_err = err
        time.sleep(1.5 * (attempt + 1))

    if df.empty:
        if last_err is not None:
            raise ValueError(
                f"Failed to fetch {ticker} from yfinance ({start} to {end}). Last error: {last_err}"
            ) from last_err
        raise ValueError(
            f"No yfinance data returned for {ticker} between {start} and {end}. "
            "This may be a temporary rate-limit; retry in a few minutes."
        )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.reset_index().rename(columns=str.lower)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df[["date", "open", "high", "low", "close", "volume"]]


def fetch_prices(
    primary_ticker: str,
    context_tickers: list[str],
    start: str,
    end: str,
) -> pl.DataFrame:
    primary = _download_ticker(primary_ticker, start, end)
    primary["ret_1d"] = primary["close"].pct_change()
    primary["ticker"] = primary_ticker

    out = pl.from_pandas(primary)

    for ticker in context_tickers:
        context = _download_ticker(ticker, start, end)
        context = context[["date", "close"]].rename(columns={"close": f"close_{ticker}"})
        out = out.join(pl.from_pandas(context), on="date", how="left")

    return out.sort("date")


def _fetch_city_temperatures(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    params: dict[str, Any] = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": "temperature_2m_mean",
        "timezone": "UTC",
    }
    resp = requests.get(OPEN_METEO_ARCHIVE, params=params, timeout=60)
    resp.raise_for_status()
    payload = resp.json()

    daily = payload.get("daily", {})
    dates = daily.get("time", [])
    temps = daily.get("temperature_2m_mean", [])
    if not dates or not temps:
        raise ValueError(f"No weather data for lat={lat}, lon={lon}")

    df = pd.DataFrame({"date": pd.to_datetime(dates).date, "temp_c": temps})
    return df


def fetch_weather(cities: list[dict[str, float]], start: str, end: str, base_temp_c: float) -> pl.DataFrame:
    city_frames = []
    weights = []
    for city in cities:
        city_df = _fetch_city_temperatures(city["lat"], city["lon"], start, end)
        city_df = city_df.rename(columns={"temp_c": f"temp_{city['name'].replace(' ', '_').lower()}"})
        city_frames.append(city_df)
        weights.append(city.get("weight", 0.0))

    merged = city_frames[0]
    for frame in city_frames[1:]:
        merged = merged.merge(frame, on="date", how="inner")

    temp_cols = [c for c in merged.columns if c.startswith("temp_")]
    w = np.array(weights, dtype=float)
    if w.sum() == 0:
        w = np.ones(len(temp_cols), dtype=float) / len(temp_cols)
    else:
        w = w / w.sum()

    merged["temp_c_us"] = merged[temp_cols].to_numpy().dot(w)
    merged["hdd_us"] = np.maximum(0.0, base_temp_c - merged["temp_c_us"])
    merged["cdd_us"] = np.maximum(0.0, merged["temp_c_us"] - base_temp_c)
    merged["temp_anom_us"] = merged["temp_c_us"] - merged["temp_c_us"].rolling(30, min_periods=7).mean()

    out = merged[["date", "hdd_us", "cdd_us", "temp_anom_us"]].copy()
    return pl.from_pandas(out).sort("date")


def run_ingestion(config: dict[str, Any], start: str, end: str) -> IngestResult:
    prices = fetch_prices(
        primary_ticker=config["data"]["primary_ticker"],
        context_tickers=config["data"].get("context_tickers", []),
        start=start,
        end=end,
    )
    weather = fetch_weather(
        cities=config["data"]["weather"]["cities"],
        start=start,
        end=end,
        base_temp_c=float(config["data"]["weather"].get("base_temp_c", 18.0)),
    )
    return IngestResult(prices_daily=prices, weather_daily=weather)

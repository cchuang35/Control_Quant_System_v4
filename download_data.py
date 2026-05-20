"""Download OHLCV data for the backtester.

Default: BTCUSDT 1h klines from Binance public API for the last 90 days.
Output columns: timestamp,open,high,low,close,volume
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import sleep
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
INTERVAL_MS = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "6h": 6 * 60 * 60_000,
    "8h": 8 * 60 * 60_000,
    "12h": 12 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}
OUTPUT_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def download_binance_ohlcv(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    start: str | None = None,
    end: str | None = None,
    limit: int = 1000,
) -> list[dict[str, str | float]]:
    """Download Binance spot klines and convert them to OHLCV rows."""

    if interval not in INTERVAL_MS:
        raise ValueError(f"unsupported interval: {interval}. Supported: {sorted(INTERVAL_MS)}")
    if not 1 <= limit <= 1000:
        raise ValueError("limit must be between 1 and 1000")

    end_dt = parse_datetime(end) if end else datetime.now(timezone.utc)
    start_dt = parse_datetime(start) if start else end_dt - timedelta(days=90)
    if start_dt >= end_dt:
        raise ValueError("start must be earlier than end")

    start_ms = datetime_to_ms(start_dt)
    end_ms = datetime_to_ms(end_dt)
    rows: list[dict[str, str | float]] = []

    while start_ms < end_ms:
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": limit,
        }
        payload = fetch_json(BINANCE_KLINES_URL, params)
        if not payload:
            break

        batch = [kline_to_row(item) for item in payload]
        rows.extend(batch)
        last_open_time = int(payload[-1][0])
        next_start = last_open_time + INTERVAL_MS[interval]
        if next_start <= start_ms:
            break
        start_ms = next_start
        sleep(0.15)

    return dedupe_rows(rows)


def fetch_json(url: str, params: dict[str, Any]) -> Any:
    request_url = f"{url}?{urlencode(params)}"
    request = Request(request_url, headers={"User-Agent": "control-quant-system-v1/1.0"})
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def kline_to_row(kline: list[Any]) -> dict[str, str | float]:
    """Convert one Binance kline item to the local OHLCV schema."""

    return {
        "timestamp": ms_to_iso(int(kline[0])),
        "open": float(kline[1]),
        "high": float(kline[2]),
        "low": float(kline[3]),
        "close": float(kline[4]),
        "volume": float(kline[5]),
    }


def write_ohlcv_csv(rows: list[dict[str, str | float]], output_path: str | Path) -> Path:
    if not rows:
        raise ValueError("no rows to write")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def default_output_path(symbol: str, interval: str) -> Path:
    return Path("data") / f"{symbol.lower()}_{interval}.csv"


def parse_datetime(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    if len(normalized) == 10:
        normalized = f"{normalized}T00:00:00+00:00"
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def datetime_to_ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def ms_to_iso(value: int) -> str:
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def dedupe_rows(rows: list[dict[str, str | float]]) -> list[dict[str, str | float]]:
    by_timestamp = {str(row["timestamp"]): row for row in rows}
    return [by_timestamp[key] for key in sorted(by_timestamp)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download OHLCV CSV data for the v1 backtester.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Binance spot symbol. Default: BTCUSDT")
    parser.add_argument("--interval", default="1h", help="Kline interval. Default: 1h")
    parser.add_argument("--start", default=None, help="UTC start time, e.g. 2025-01-01 or 2025-01-01T00:00:00Z. Default: last 90 days.")
    parser.add_argument("--end", default=None, help="UTC end time. Default: now.")
    parser.add_argument("--output", type=Path, default=None, help="Output CSV path. Default: data/{symbol}_{interval}.csv")
    args = parser.parse_args()

    rows = download_binance_ohlcv(args.symbol, args.interval, args.start, args.end)
    output = args.output or default_output_path(args.symbol, args.interval)
    path = write_ohlcv_csv(rows, output)
    print(f"Downloaded {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()


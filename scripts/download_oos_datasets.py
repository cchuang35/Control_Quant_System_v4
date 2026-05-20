from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from download_data import download_binance_ohlcv, write_ohlcv_csv


OUTPUT_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
DEFAULT_DATASETS = (
    ("BTCUSDT", "1h", 730, "btcusdt_1h_2y.csv"),
    ("BTCUSDT", "1h", 1095, "btcusdt_1h_3y.csv"),
    ("BTCUSDT", "1h", 1825, "btcusdt_1h_5y.csv"),
    ("ETHUSDT", "1h", 365, "ethusdt_1h_365d.csv"),
    ("ETHUSDT", "1h", 730, "ethusdt_1h_2y.csv"),
    ("ETHUSDT", "1h", 1095, "ethusdt_1h_3y.csv"),
    ("ETHUSDT", "1h", 1825, "ethusdt_1h_5y.csv"),
)


@dataclass(frozen=True)
class DatasetSpec:
    symbol: str
    interval: str
    days: int
    filename: str


def iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def validate_ohlcv_csv(path: Path, expected_interval: str = "1h") -> dict[str, int | float | str | bool]:
    frame = pd.read_csv(path)
    missing_columns = [column for column in OUTPUT_COLUMNS if column not in frame.columns]
    if missing_columns:
        return {
            "path": str(path),
            "ok": False,
            "error": f"missing columns: {missing_columns}",
            "rows": len(frame),
        }

    timestamp = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    duplicate_count = int(timestamp.duplicated().sum())
    sorted_ascending = bool(timestamp.is_monotonic_increasing)
    invalid_timestamp_count = int(timestamp.isna().sum())
    close = pd.to_numeric(frame["close"], errors="coerce")
    non_positive_close_count = int((close <= 0.0).sum())
    invalid_close_count = int(close.isna().sum())

    expected_delta = pd.Timedelta(hours=1) if expected_interval == "1h" else None
    spacing_issue_count = 0
    missing_bar_count = 0
    if expected_delta is not None and len(timestamp) > 1:
        deltas = timestamp.sort_values().drop_duplicates().diff().dropna()
        spacing_issue_count = int((deltas != expected_delta).sum())
        missing_bar_count = int(sum(max(int(delta / expected_delta) - 1, 0) for delta in deltas if delta > expected_delta))

    ok = (
        invalid_timestamp_count == 0
        and duplicate_count == 0
        and sorted_ascending
        and spacing_issue_count == 0
        and missing_bar_count == 0
        and invalid_close_count == 0
        and non_positive_close_count == 0
    )
    return {
        "path": str(path),
        "ok": ok,
        "rows": len(frame),
        "start": frame["timestamp"].iloc[0] if len(frame) else "",
        "end": frame["timestamp"].iloc[-1] if len(frame) else "",
        "duplicate_timestamps": duplicate_count,
        "timestamps_sorted_ascending": sorted_ascending,
        "invalid_timestamps": invalid_timestamp_count,
        "spacing_issue_count": spacing_issue_count,
        "missing_bar_count": missing_bar_count,
        "invalid_close_count": invalid_close_count,
        "non_positive_close_count": non_positive_close_count,
    }


def parse_specs(names: list[str] | None) -> list[DatasetSpec]:
    specs = [DatasetSpec(*item) for item in DEFAULT_DATASETS]
    if not names:
        return specs
    wanted = {name.lower() for name in names}
    return [spec for spec in specs if spec.filename.lower() in wanted or spec.symbol.lower() in wanted]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download OOS Binance 1h datasets and validate OHLCV quality.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory to write CSVs.")
    parser.add_argument("--report", type=Path, default=Path("reports") / "data_quality" / "oos_data_quality.csv")
    parser.add_argument("--end", default=None, help="UTC end time. Default: current UTC time.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--dataset", action="append", help="Optional filename or symbol filter. Can be repeated.")
    args = parser.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    end_dt = datetime.fromisoformat(args.end.replace("Z", "+00:00")) if args.end else datetime.now(timezone.utc)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)
    end_dt = end_dt.astimezone(timezone.utc)

    quality_rows: list[dict[str, int | float | str | bool]] = []
    for spec in parse_specs(args.dataset):
        output_path = args.data_dir / spec.filename
        start_dt = end_dt - timedelta(days=spec.days)
        if output_path.exists() and not args.overwrite:
            print(f"skip existing {output_path}")
        else:
            print(f"download {spec.symbol} {spec.interval} {spec.days}d -> {output_path}")
            rows = download_binance_ohlcv(
                symbol=spec.symbol,
                interval=spec.interval,
                start=iso_utc(start_dt),
                end=iso_utc(end_dt),
            )
            write_ohlcv_csv(rows, output_path)
            print(f"wrote {len(rows)} rows to {output_path}")
        quality = validate_ohlcv_csv(output_path, expected_interval=spec.interval)
        quality.update(
            {
                "symbol": spec.symbol,
                "interval": spec.interval,
                "requested_days": spec.days,
                "filename": spec.filename,
            }
        )
        quality_rows.append(quality)

    quality_frame = pd.DataFrame(quality_rows)
    quality_frame.to_csv(args.report, index=False)
    print(f"wrote quality report to {args.report}")
    print(quality_frame.to_string(index=False))


if __name__ == "__main__":
    main()

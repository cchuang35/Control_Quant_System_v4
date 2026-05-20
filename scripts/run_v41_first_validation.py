from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v4 import (  # noqa: E402
    BacktestConfig,
    create_v41_default_config,
    evaluate_metrics,
    run_controller_buy_and_hold_benchmark,
    run_fixed_exposure_benchmark,
    run_v41_backtest,
    run_zero_position_benchmark,
)


FEE_RATE = 0.001
PERIODS_PER_YEAR = 365
OUT_DIR = Path("reports") / "v41_first_validation"
DATASETS = (
    ("BTC", "365d", Path("data") / "btcusdt_1h_365d.csv"),
    ("BTC", "2y", Path("data") / "btcusdt_1h_2y.csv"),
    ("BTC", "3y", Path("data") / "btcusdt_1h_3y.csv"),
    ("BTC", "5y", Path("data") / "btcusdt_1h_5y.csv"),
    ("ETH", "365d", Path("data") / "ethusdt_1h_365d.csv"),
    ("ETH", "2y", Path("data") / "ethusdt_1h_2y.csv"),
    ("ETH", "3y", Path("data") / "ethusdt_1h_3y.csv"),
    ("ETH", "5y", Path("data") / "ethusdt_1h_5y.csv"),
)
METRIC_COLUMNS = [
    "asset",
    "window",
    "strategy_name",
    "total_return",
    "annualized_return",
    "max_drawdown",
    "sharpe_ratio",
    "total_turnover",
    "average_turnover",
    "average_exposure",
    "total_fee_cost",
    "trade_count",
    "final_equity",
    "min_position",
    "max_position",
    "position_std",
    "average_drawdown",
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for asset, window, path in DATASETS:
        daily_close = load_daily_close(path)
        rows.extend(validate_dataset(asset=asset, window=window, prices=daily_close))

    comparison = pd.DataFrame(rows, columns=METRIC_COLUMNS)
    csv_path = OUT_DIR / "v41_first_validation_comparison.csv"
    md_path = OUT_DIR / "v41_first_validation_comparison.md"
    comparison.to_csv(csv_path, index=False)
    md_path.write_text(frame_to_markdown(comparison), encoding="utf-8")
    print(f"comparison_csv: {csv_path}")
    print(f"comparison_markdown: {md_path}")
    print(comparison.to_string(index=False))


def frame_to_markdown(frame: pd.DataFrame) -> str:
    rendered = frame.copy()
    for column in rendered.columns:
        if pd.api.types.is_float_dtype(rendered[column]):
            rendered[column] = rendered[column].map(lambda value: "nan" if pd.isna(value) else f"{value:.6f}")
        else:
            rendered[column] = rendered[column].astype(str)
    headers = list(rendered.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rendered.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


def validate_dataset(*, asset: str, window: str, prices: pd.Series) -> list[dict[str, Any]]:
    config = create_v41_default_config()
    backtest_config = BacktestConfig(
        fee_rate=FEE_RATE,
        initial_equity=config.backtest.initial_equity,
        initial_high_watermark=config.backtest.initial_high_watermark,
        initial_position=config.backtest.initial_position,
    )
    benchmark_config = BacktestConfig(fee_rate=FEE_RATE)
    runs = [
        (
            "v4.1-minimal-control-strategy",
            run_v41_backtest(
                prices,
                config=create_v41_default_config(),
            ),
        ),
        (
            "zero_position",
            run_zero_position_benchmark(prices, config=benchmark_config),
        ),
        (
            "controller_buy_and_hold",
            run_controller_buy_and_hold_benchmark(prices, config=benchmark_config),
        ),
        (
            "fixed_0_5_exposure",
            run_fixed_exposure_benchmark(prices, exposure=0.5, config=benchmark_config),
        ),
    ]
    rows = [
        summarize_backtest_result(asset=asset, window=window, strategy_name=name, result=result)
        for name, result in runs
    ]
    rows.append(summarize_true_buy_and_hold(asset=asset, window=window, prices=prices, fee_rate=FEE_RATE))
    return rows


def load_daily_close(path: Path) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if "timestamp" not in frame.columns or "close" not in frame.columns:
        raise ValueError(f"{path} must contain timestamp and close columns")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("timestamp").set_index("timestamp")
    daily = frame["close"].astype(float).resample("1D").last().dropna()
    if len(daily) < 2:
        raise ValueError(f"{path} produced fewer than two daily closes")
    if (daily <= 0.0).any():
        raise ValueError(f"{path} contains non-positive daily closes")
    return daily


def summarize_backtest_result(
    *,
    asset: str,
    window: str,
    strategy_name: str,
    result: pd.DataFrame,
) -> dict[str, Any]:
    metrics = evaluate_metrics(result, periods_per_year=PERIODS_PER_YEAR)
    return {
        "asset": asset,
        "window": window,
        "strategy_name": strategy_name,
        **metrics,
        "final_equity": float(result["equity"].iloc[-1]),
        "min_position": float(result["position"].min()),
        "max_position": float(result["position"].max()),
        "position_std": float(result["position"].std(ddof=0)),
        "average_drawdown": float(result["drawdown"].mean()),
    }


def summarize_true_buy_and_hold(
    *,
    asset: str,
    window: str,
    prices: pd.Series,
    fee_rate: float,
) -> dict[str, Any]:
    price_values = prices.astype(float).to_numpy()
    period_count = len(price_values) - 1
    equity = (1.0 - fee_rate) * price_values[1:] / price_values[0]
    equity_with_initial = np.concatenate([[1.0], equity])
    returns = equity_with_initial[1:] / equity_with_initial[:-1] - 1.0
    high_watermark = np.maximum.accumulate(equity_with_initial)
    drawdown = 1.0 - equity_with_initial[1:] / high_watermark[1:]
    std_return = float(np.std(returns))
    sharpe = math.nan if std_return == 0.0 else float(np.mean(returns)) / std_return * math.sqrt(PERIODS_PER_YEAR)
    final_equity = float(equity[-1])
    return {
        "asset": asset,
        "window": window,
        "strategy_name": "true_buy_and_hold",
        "total_return": final_equity - 1.0,
        "annualized_return": final_equity ** (PERIODS_PER_YEAR / period_count) - 1.0,
        "max_drawdown": float(np.max(drawdown)),
        "sharpe_ratio": sharpe,
        "total_turnover": 1.0,
        "average_turnover": 1.0 / period_count,
        "average_exposure": 1.0,
        "total_fee_cost": fee_rate,
        "trade_count": 1,
        "final_equity": final_equity,
        "min_position": 1.0,
        "max_position": 1.0,
        "position_std": 0.0,
        "average_drawdown": float(np.mean(drawdown)),
    }


if __name__ == "__main__":
    main()

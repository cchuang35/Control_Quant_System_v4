"""Benchmark helpers for v4.1 minimal validation."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from .backtest import run_backtest
from .controllers import BuyAndHoldController, FixedExposureController, ZeroController
from .data_types import BacktestConfig
from .metrics import evaluate_metrics


def run_zero_position_benchmark(
    prices: Sequence[float] | pd.Series | pd.DataFrame,
    *,
    config: BacktestConfig | None = None,
    price_column: str = "close",
) -> pd.DataFrame:
    return run_backtest(prices, controller=ZeroController(), config=config or BacktestConfig(), price_column=price_column)


def run_controller_buy_and_hold_benchmark(
    prices: Sequence[float] | pd.Series | pd.DataFrame,
    *,
    config: BacktestConfig | None = None,
    initial_position: float = 0.0,
    price_column: str = "close",
) -> pd.DataFrame:
    """Run the controller-based buy-and-hold benchmark.

    With initial_position=0.0, entry happens after the first observation. Use
    initial_position=1.0 to start invested from the first return interval.
    """

    base_config = config or BacktestConfig()
    benchmark_config = BacktestConfig(
        fee_rate=base_config.fee_rate,
        initial_equity=base_config.initial_equity,
        initial_high_watermark=base_config.initial_high_watermark,
        initial_position=initial_position,
    )
    return run_backtest(
        prices,
        controller=BuyAndHoldController(),
        config=benchmark_config,
        price_column=price_column,
    )


def run_fixed_exposure_benchmark(
    prices: Sequence[float] | pd.Series | pd.DataFrame,
    *,
    exposure: float = 0.5,
    config: BacktestConfig | None = None,
    price_column: str = "close",
) -> pd.DataFrame:
    return run_backtest(
        prices,
        controller=FixedExposureController(exposure),
        config=config or BacktestConfig(),
        price_column=price_column,
    )


def compute_true_buy_and_hold(
    prices: Sequence[float] | pd.Series | pd.DataFrame,
    *,
    price_column: str = "close",
) -> pd.DataFrame:
    """Compute traditional buy-and-hold from P_0 without controller timing."""

    price_series = _coerce_prices(prices, price_column=price_column)
    initial_price = float(price_series.iloc[0])
    rows = []
    high_watermark = 1.0
    for step in range(1, len(price_series)):
        price = float(price_series.iloc[step])
        equity = price / initial_price
        high_watermark = max(high_watermark, equity)
        rows.append(
            {
                "timestamp": price_series.index[step],
                "price": price,
                "equity": equity,
                "high_watermark": high_watermark,
                "drawdown": 1.0 - equity / high_watermark,
            }
        )
    result = pd.DataFrame(rows)
    result.attrs["benchmark_name"] = "true_buy_and_hold"
    result.attrs["initial_equity"] = 1.0
    return result


def evaluate_standard_benchmarks(
    prices: Sequence[float] | pd.Series | pd.DataFrame,
    *,
    periods_per_year: int | float = 365,
    config: BacktestConfig | None = None,
    price_column: str = "close",
) -> dict[str, dict[str, float]]:
    """Evaluate the standard v4.1 controller-based benchmarks."""

    resolved = config or BacktestConfig()
    return {
        "zero_position": evaluate_metrics(
            run_zero_position_benchmark(prices, config=resolved, price_column=price_column),
            periods_per_year=periods_per_year,
        ),
        "controller_buy_and_hold": evaluate_metrics(
            run_controller_buy_and_hold_benchmark(prices, config=resolved, price_column=price_column),
            periods_per_year=periods_per_year,
        ),
        "fixed_0_5_exposure": evaluate_metrics(
            run_fixed_exposure_benchmark(prices, exposure=0.5, config=resolved, price_column=price_column),
            periods_per_year=periods_per_year,
        ),
    }


def _coerce_prices(
    prices: Sequence[float] | pd.Series | pd.DataFrame,
    *,
    price_column: str,
) -> pd.Series:
    if isinstance(prices, pd.DataFrame):
        if price_column not in prices.columns:
            raise ValueError(f"price_column '{price_column}' not found")
        series = prices[price_column]
    elif isinstance(prices, pd.Series):
        series = prices
    else:
        series = pd.Series(list(prices))
    numeric = pd.to_numeric(series, errors="coerce")
    if len(numeric) < 2:
        raise ValueError("at least two prices are required")
    if numeric.isna().any():
        raise ValueError("prices must be numeric and non-missing")
    if (numeric <= 0.0).any():
        raise ValueError("prices must be positive")
    return numeric.astype(float)

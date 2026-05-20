"""Run v4.1-minimal-control-strategy on a tiny daily price series."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.v4 import (
    BacktestConfig,
    ControllerConfig,
    FilterConfig,
    MetricsEvaluator,
    MinimalContinuousController,
    MinimalFilterLayer,
    MinimalStateEstimator,
    StateEstimatorConfig,
    run_backtest,
)


def main() -> None:
    prices = pd.Series(
        [100.0, 101.0, 102.0, 101.0, 103.0, 104.0, 102.0, 105.0, 106.0, 107.0],
        index=pd.date_range("2024-01-01", periods=10, freq="D"),
    )

    config = BacktestConfig(fee_rate=0.001)
    filter_config = FilterConfig(short_window=10, vol_window=30, long_window=60)
    estimator_config = StateEstimatorConfig(
        k_tau=1.0,
        k_epsilon=1.0,
        vol_ref=0.03,
        drawdown_ref=0.20,
        epsilon=1e-8,
    )
    controller_config = ControllerConfig(
        w_epsilon=0.25,
        w_volatility=0.50,
        w_portfolio_risk=0.75,
        max_position_change=0.20,
    )

    estimator = MinimalStateEstimator(
        filter_layer=MinimalFilterLayer(filter_config),
        config=estimator_config,
    )
    controller = MinimalContinuousController(controller_config)
    result = run_backtest(prices, controller=controller, state_estimator=estimator, config=config)

    # Daily crypto data uses 365 periods/year. Common alternatives:
    # 4h crypto data: 2190, 1h crypto data: 8760.
    metrics = MetricsEvaluator().evaluate(result, periods_per_year=365)
    print(result.tail())
    print(metrics)


if __name__ == "__main__":
    main()

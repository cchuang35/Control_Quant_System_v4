import math

import numpy as np
import pandas as pd
import pytest

from src.v4 import (
    DAILY_CRYPTO_PERIODS_PER_YEAR,
    BacktestConfig,
    V41_VERSION_NAME,
    compute_true_buy_and_hold,
    create_v41_default_config,
    evaluate_metrics,
    evaluate_metrics_after_warmup,
    evaluate_standard_benchmarks,
    run_controller_buy_and_hold_benchmark,
    run_fixed_exposure_benchmark,
    run_v41_backtest,
    run_zero_position_benchmark,
)


def synthetic_daily_prices(count: int = 90) -> pd.Series:
    prices = []
    price = 100.0
    for idx in range(count):
        drift = 0.002 if idx < count // 2 else 0.001
        wobble = -0.01 if idx % 17 == 0 and idx > 0 else 0.003 if idx % 11 == 0 else 0.0
        price *= 1.0 + drift + wobble
        prices.append(price)
    return pd.Series(prices, index=pd.date_range("2024-01-01", periods=count, freq="D"))


def test_v41_default_config_exposes_official_defaults() -> None:
    config = create_v41_default_config()

    assert config.version_name == V41_VERSION_NAME == "v4.1-minimal-control-strategy"
    assert config.periods_per_year == DAILY_CRYPTO_PERIODS_PER_YEAR == 365
    assert config.backtest.initial_equity == pytest.approx(1.0)
    assert config.backtest.initial_high_watermark == pytest.approx(1.0)
    assert config.backtest.initial_position == pytest.approx(0.0)
    assert config.backtest.fee_rate == pytest.approx(0.001)
    assert config.position_range == (0.0, 1.0)
    assert (config.filter.short_window, config.filter.vol_window, config.filter.long_window) == (10, 30, 60)
    assert config.warmup_period == 60
    assert config.state_estimator.vol_ref == pytest.approx(0.03)
    assert config.state_estimator.drawdown_ref == pytest.approx(0.20)
    assert config.controller.max_position_change == pytest.approx(0.20)


def test_v41_full_system_runs_end_to_end_and_outputs_valid_records() -> None:
    result = run_v41_backtest(synthetic_daily_prices())
    essential_numeric_columns = [
        "price",
        "simple_return",
        "log_return",
        "pre_trade_equity",
        "pre_trade_high_watermark",
        "pre_trade_drawdown",
        "previous_position",
        "raw_target_position",
        "position",
        "turnover",
        "transaction_cost",
        "equity",
        "high_watermark",
        "drawdown",
    ]
    required_columns = set(essential_numeric_columns) | {"timestamp", "observation", "state"}

    assert not result.empty
    assert required_columns.issubset(result.columns)
    assert result["position"].between(0.0, 1.0).all()
    assert result["previous_position"].between(0.0, 1.0).all()
    assert (result["equity"] > 0.0).all()
    assert (result["high_watermark"] > 0.0).all()
    assert result["drawdown"].between(0.0, 1.0).all()
    assert (result["turnover"] >= 0.0).all()
    assert (result["transaction_cost"] >= 0.0).all()
    assert not result[essential_numeric_columns].isna().any().any()

    metrics = evaluate_metrics(result, periods_per_year=365)

    assert "total_return" in metrics
    assert "max_drawdown" in metrics
    assert "sharpe_ratio" in metrics
    assert 0.0 <= metrics["average_exposure"] <= 1.0


def test_v41_metrics_after_warmup_are_supported_without_changing_full_metrics() -> None:
    prices = synthetic_daily_prices(120)
    result = run_v41_backtest(prices)
    config = create_v41_default_config()

    full_metrics = evaluate_metrics(result, periods_per_year=config.periods_per_year)
    warmup_metrics = evaluate_metrics_after_warmup(
        result,
        periods_per_year=config.periods_per_year,
        warmup_period=config.warmup_period,
    )
    direct_warmup_metrics = evaluate_metrics(
        result,
        periods_per_year=config.periods_per_year,
        start_index=config.warmup_period,
    )

    assert set(full_metrics) == set(warmup_metrics)
    assert warmup_metrics == direct_warmup_metrics
    assert "total_return" in warmup_metrics
    assert 0.0 <= warmup_metrics["average_exposure"] <= 1.0


def test_v41_standard_benchmark_helpers_are_clear_and_valid() -> None:
    prices = synthetic_daily_prices()
    config = BacktestConfig(fee_rate=0.001)

    zero = run_zero_position_benchmark(prices, config=config)
    controller_buy_hold = run_controller_buy_and_hold_benchmark(prices, config=config)
    controller_start_invested = run_controller_buy_and_hold_benchmark(prices, config=config, initial_position=1.0)
    fixed_half = run_fixed_exposure_benchmark(prices, config=config)
    true_buy_hold = compute_true_buy_and_hold(prices)
    benchmark_metrics = evaluate_standard_benchmarks(prices, periods_per_year=365, config=config)

    assert np.allclose(zero["equity"], 1.0)
    assert np.allclose(zero["drawdown"], 0.0)
    assert np.allclose(zero["transaction_cost"], 0.0)
    assert controller_buy_hold["previous_position"].iloc[0] == pytest.approx(0.0)
    assert controller_start_invested["previous_position"].iloc[0] == pytest.approx(1.0)
    assert fixed_half["position"].between(0.0, 1.0).all()
    assert true_buy_hold.attrs["benchmark_name"] == "true_buy_and_hold"
    assert true_buy_hold["equity"].iloc[-1] == pytest.approx(float(prices.iloc[-1] / prices.iloc[0]))
    assert {"zero_position", "controller_buy_and_hold", "fixed_0_5_exposure"} == set(benchmark_metrics)
    assert all("total_return" in metrics for metrics in benchmark_metrics.values())


def test_warmup_metrics_reject_invalid_start_points() -> None:
    result = run_v41_backtest(synthetic_daily_prices(10))

    with pytest.raises(ValueError, match="warmup_period must be non-negative"):
        evaluate_metrics_after_warmup(result, periods_per_year=365, warmup_period=-1)
    with pytest.raises(ValueError, match="start_index must be less"):
        evaluate_metrics_after_warmup(result, periods_per_year=365, warmup_period=len(result))


def test_v41_sharpe_key_can_be_nan_for_flat_zero_benchmark() -> None:
    result = run_zero_position_benchmark(synthetic_daily_prices())
    metrics = evaluate_metrics(result, periods_per_year=365)

    assert "sharpe_ratio" in metrics
    assert math.isnan(metrics["sharpe_ratio"])

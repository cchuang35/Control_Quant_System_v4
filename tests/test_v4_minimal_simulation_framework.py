import math

import numpy as np
import pandas as pd
import pytest

from src.v4 import (
    BacktestConfig,
    BacktestEngine,
    BuyAndHoldController,
    FixedExposureController,
    MetricsEvaluator,
    ZeroController,
)


def sample_prices() -> pd.Series:
    return pd.Series(
        [100.0, 110.0, 105.0, 120.0, 118.0],
        index=pd.date_range("2024-01-01", periods=5, freq="D"),
    )


def test_zero_controller_keeps_equity_flat_without_fees_or_drawdown() -> None:
    result = BacktestEngine(
        controller=ZeroController(),
        config=BacktestConfig(fee_rate=0.001),
    ).run(sample_prices())
    metrics = MetricsEvaluator().evaluate(result, periods_per_year=365)

    assert np.allclose(result["equity"], 1.0)
    assert np.allclose(result["turnover"], 0.0)
    assert np.allclose(result["transaction_cost"], 0.0)
    assert np.allclose(result["drawdown"], 0.0)
    assert metrics["total_return"] == pytest.approx(0.0)
    assert metrics["max_drawdown"] == pytest.approx(0.0)
    assert math.isnan(metrics["sharpe_ratio"])


def test_buy_and_hold_uses_initial_entry_fee_and_lagged_position() -> None:
    fee_rate = 0.001
    result = BacktestEngine(
        controller=BuyAndHoldController(),
        config=BacktestConfig(fee_rate=fee_rate),
    ).run(sample_prices())

    assert result["previous_position"].tolist() == [0.0, 1.0, 1.0, 1.0]
    assert result["turnover"].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert result["transaction_cost"].iloc[0] == pytest.approx(fee_rate * result["pre_trade_equity"].iloc[0])

    expected_equity = 1.0 * (1.0 - fee_rate)
    prices = sample_prices().to_numpy()
    for idx in range(2, len(prices)):
        expected_equity *= prices[idx] / prices[idx - 1]
    assert result["equity"].iloc[-1] == pytest.approx(expected_equity)


def test_fixed_exposure_has_lagged_average_exposure_near_fixed_value() -> None:
    prices = pd.Series([100.0] + [101.0 + idx for idx in range(20)])
    result = BacktestEngine(
        controller=FixedExposureController(0.5),
        config=BacktestConfig(fee_rate=0.0),
    ).run(prices)
    metrics = MetricsEvaluator().evaluate(result, periods_per_year=365)

    assert result["position"].between(0.0, 1.0).all()
    assert result["previous_position"].iloc[0] == pytest.approx(0.0)
    assert metrics["average_exposure"] == pytest.approx(0.5 * (len(result) - 1) / len(result))
    assert abs(metrics["average_exposure"] - 0.5) < 0.03


def test_raw_positions_are_clipped_to_long_only_no_leverage_range() -> None:
    result_high = BacktestEngine(
        controller=FixedExposureController(1.5),
        config=BacktestConfig(fee_rate=0.0),
    ).run(sample_prices())
    result_low = BacktestEngine(
        controller=FixedExposureController(-0.5),
        config=BacktestConfig(fee_rate=0.0),
    ).run(sample_prices())

    assert np.allclose(result_high["position"], 1.0)
    assert np.allclose(result_low["position"], 0.0)


def test_transaction_cost_uses_pre_trade_equity() -> None:
    result = BacktestEngine(
        controller=FixedExposureController(0.5),
        config=BacktestConfig(fee_rate=0.01),
    ).run([100.0, 200.0])

    assert result["pre_trade_equity"].iloc[0] == pytest.approx(1.0)
    assert result["turnover"].iloc[0] == pytest.approx(0.5)
    assert result["transaction_cost"].iloc[0] == pytest.approx(0.005)


def test_rejects_empty_missing_or_non_positive_prices() -> None:
    engine = BacktestEngine(controller=ZeroController())

    with pytest.raises(ValueError, match="at least two prices"):
        engine.run([])
    with pytest.raises(ValueError, match="missing"):
        engine.run([100.0, np.nan])
    with pytest.raises(ValueError, match="positive"):
        engine.run([100.0, 0.0])

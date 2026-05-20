import math

import pandas as pd
import pytest

from src.v4 import (
    BacktestConfig,
    BacktestEngine,
    FilteredSignals,
    FixedExposureController,
    MinimalStateEstimator,
    Observation,
    StateEstimatorConfig,
)


def filtered(
    long_trend: float = 0.0,
    volatility: float = 0.0,
    short_timing: float = 0.0,
    drawdown: float = 0.0,
    previous_position: float = 0.0,
) -> FilteredSignals:
    return FilteredSignals(
        long_trend=long_trend,
        volatility=volatility,
        short_timing=short_timing,
        drawdown=drawdown,
        previous_position=previous_position,
    )


def test_estimator_bounds_extreme_filtered_signals() -> None:
    estimator = MinimalStateEstimator()
    state = estimator.estimate_from_filtered(
        filtered(
            long_trend=100.0,
            volatility=100.0,
            short_timing=-100.0,
            drawdown=100.0,
            previous_position=10.0,
        )
    )

    assert -1.0 <= state.tau <= 1.0
    assert -1.0 <= state.epsilon <= 1.0
    assert 0.0 <= state.nu <= 1.0
    assert 0.0 <= state.rho <= 1.0
    assert 0.0 <= state.previous_position <= 1.0


def test_zero_filtered_signal_maps_to_zero_state() -> None:
    state = MinimalStateEstimator().estimate_from_filtered(filtered())

    assert state.tau == pytest.approx(0.0)
    assert state.nu == pytest.approx(0.0)
    assert state.epsilon == pytest.approx(0.0)
    assert state.rho == pytest.approx(0.0)
    assert state.previous_position == pytest.approx(0.0)


def test_positive_and_negative_trend_map_to_signed_tau() -> None:
    estimator = MinimalStateEstimator()

    positive = estimator.estimate_from_filtered(filtered(long_trend=0.01, volatility=0.02))
    negative = estimator.estimate_from_filtered(filtered(long_trend=-0.01, volatility=0.02))

    assert positive.tau > 0.0
    assert negative.tau < 0.0


def test_volatility_normalization_clips_at_reference() -> None:
    config = StateEstimatorConfig(vol_ref=0.04)
    estimator = MinimalStateEstimator(config=config)

    half = estimator.estimate_from_filtered(filtered(volatility=0.02))
    capped = estimator.estimate_from_filtered(filtered(volatility=0.04))
    above = estimator.estimate_from_filtered(filtered(volatility=0.08))

    assert half.nu == pytest.approx(0.5)
    assert capped.nu == pytest.approx(1.0)
    assert above.nu == pytest.approx(1.0)


def test_drawdown_normalization_clips_at_reference() -> None:
    config = StateEstimatorConfig(drawdown_ref=0.30)
    estimator = MinimalStateEstimator(config=config)

    half = estimator.estimate_from_filtered(filtered(drawdown=0.15))
    capped = estimator.estimate_from_filtered(filtered(drawdown=0.30))
    above = estimator.estimate_from_filtered(filtered(drawdown=0.60))

    assert half.rho == pytest.approx(0.5)
    assert capped.rho == pytest.approx(1.0)
    assert above.rho == pytest.approx(1.0)


def test_short_timing_maps_to_signed_epsilon() -> None:
    estimator = MinimalStateEstimator()

    positive = estimator.estimate_from_filtered(filtered(volatility=0.02, short_timing=0.01))
    negative = estimator.estimate_from_filtered(filtered(volatility=0.02, short_timing=-0.01))

    assert positive.epsilon > 0.0
    assert negative.epsilon < 0.0


def test_previous_position_is_clipped() -> None:
    estimator = MinimalStateEstimator()

    low = estimator.estimate_from_filtered(filtered(previous_position=-0.5))
    high = estimator.estimate_from_filtered(filtered(previous_position=1.5))

    assert low.previous_position == pytest.approx(0.0)
    assert high.previous_position == pytest.approx(1.0)


def test_update_consumes_observations_and_backtest_engine_accepts_estimator() -> None:
    estimator = MinimalStateEstimator()
    observations = [
        Observation(log_return=0.01, pre_trade_drawdown=0.0, previous_position=0.0),
        Observation(log_return=-0.02, pre_trade_drawdown=0.1, previous_position=0.5),
    ]

    states = [estimator.update(observation) for observation in observations]

    assert all(state.as_tuple() for state in states)
    assert estimator.filter_layer.step_count == 2

    result = BacktestEngine(
        controller=FixedExposureController(0.5),
        state_estimator=MinimalStateEstimator(),
        config=BacktestConfig(fee_rate=0.0),
    ).run(pd.Series([100.0, 101.0, 100.0, 102.0]))

    assert len(result) == 3
    assert result["position"].tolist() == [0.5, 0.5, 0.5]


def test_reset_restores_internal_filter_state() -> None:
    estimator = MinimalStateEstimator()
    estimator.update(Observation(log_return=0.02, pre_trade_drawdown=0.0, previous_position=0.0))

    estimator.reset()

    assert estimator.filter_layer.long_trend == pytest.approx(0.0)
    assert estimator.filter_layer.short_momentum == pytest.approx(0.0)
    assert estimator.filter_layer.variance == pytest.approx(0.0)
    assert estimator.filter_layer.step_count == 0


def test_state_estimator_config_validates_parameters() -> None:
    with pytest.raises(ValueError, match="k_tau must be finite"):
        StateEstimatorConfig(k_tau=math.inf)
    with pytest.raises(ValueError, match="k_epsilon must be finite"):
        StateEstimatorConfig(k_epsilon=math.nan)
    with pytest.raises(ValueError, match="vol_ref must be positive"):
        StateEstimatorConfig(vol_ref=0.0)
    with pytest.raises(ValueError, match="drawdown_ref must be positive"):
        StateEstimatorConfig(drawdown_ref=0.0)
    with pytest.raises(ValueError, match="epsilon must be positive"):
        StateEstimatorConfig(epsilon=0.0)

import math

import pandas as pd
import pytest

from src.v4 import (
    BacktestConfig,
    BacktestEngine,
    ControllerConfig,
    MinimalContinuousController,
    MinimalStateEstimator,
    StateVector,
)


def state(
    tau: float = 0.0,
    nu: float = 0.0,
    epsilon: float = 0.0,
    rho: float = 0.0,
    previous_position: float = 0.0,
) -> StateVector:
    return StateVector(
        tau=tau,
        nu=nu,
        epsilon=epsilon,
        rho=rho,
        previous_position=previous_position,
    )


def test_negative_trend_gives_zero_target_direction() -> None:
    controller = MinimalContinuousController()

    target = controller.decide(state(tau=-1.0, nu=0.0, epsilon=1.0, rho=0.0, previous_position=0.0))

    assert target == pytest.approx(0.0)


def test_zero_trend_gives_zero_target_when_flat() -> None:
    controller = MinimalContinuousController()

    target = controller.decide(state(tau=0.0, nu=0.0, epsilon=1.0, rho=0.0, previous_position=0.0))

    assert target == pytest.approx(0.0)


def test_positive_trend_creates_exposure_limited_by_smoothing() -> None:
    controller = MinimalContinuousController(ControllerConfig(max_position_change=0.2))

    target = controller.decide(state(tau=1.0, nu=0.0, epsilon=0.0, rho=0.0, previous_position=0.0))
    trace = controller.explain(state(tau=1.0, nu=0.0, epsilon=0.0, rho=0.0, previous_position=0.0))

    assert target == pytest.approx(0.2)
    assert trace["unsmoothed_target"] == pytest.approx(1.0)
    assert trace["raw_target_position"] == pytest.approx(target)
    assert trace["final_position"] == pytest.approx(target)


def test_position_smoothing_limits_upward_movement() -> None:
    controller = MinimalContinuousController(ControllerConfig(max_position_change=0.2))

    target = controller.decide(state(tau=1.0, nu=0.0, epsilon=0.0, rho=0.0, previous_position=0.1))

    assert target == pytest.approx(0.3)


def test_position_smoothing_limits_downward_movement() -> None:
    controller = MinimalContinuousController(ControllerConfig(max_position_change=0.2))

    target = controller.decide(state(tau=-1.0, nu=0.0, epsilon=0.0, rho=0.0, previous_position=0.8))

    assert target >= 0.6
    assert target == pytest.approx(0.6)


def test_market_risk_reduces_exposure() -> None:
    controller = MinimalContinuousController(ControllerConfig(max_position_change=1.0))

    low_risk = controller.decide(state(tau=1.0, nu=0.0, epsilon=0.0, rho=0.0))
    high_risk = controller.decide(state(tau=1.0, nu=1.0, epsilon=0.0, rho=0.0))

    assert high_risk <= low_risk
    assert high_risk == pytest.approx(0.5)


def test_portfolio_risk_reduces_exposure() -> None:
    controller = MinimalContinuousController(ControllerConfig(max_position_change=1.0))

    low_risk = controller.decide(state(tau=1.0, nu=0.0, epsilon=0.0, rho=0.0))
    high_risk = controller.decide(state(tau=1.0, nu=0.0, epsilon=0.0, rho=1.0))

    assert high_risk <= low_risk
    assert high_risk == pytest.approx(0.25)


def test_short_term_timing_is_auxiliary_to_positive_base_exposure() -> None:
    controller = MinimalContinuousController(ControllerConfig(max_position_change=1.0))

    negative_timing = controller.decide(state(tau=0.6, nu=0.0, epsilon=-1.0, rho=0.0))
    positive_timing = controller.decide(state(tau=0.6, nu=0.0, epsilon=1.0, rho=0.0))
    no_trend_positive_timing = controller.decide(state(tau=0.0, nu=0.0, epsilon=1.0, rho=0.0))

    assert positive_timing >= negative_timing
    assert negative_timing == pytest.approx(0.6 * 0.75)
    assert positive_timing == pytest.approx(0.6 * 1.25)
    assert no_trend_positive_timing == pytest.approx(0.0)


def test_output_is_bounded_for_extreme_inputs() -> None:
    controller = MinimalContinuousController(ControllerConfig(max_position_change=1.0))

    target = controller.decide(state(tau=10.0, nu=-10.0, epsilon=10.0, rho=-10.0, previous_position=10.0))

    assert 0.0 <= target <= 1.0


def test_controller_config_validates_parameters() -> None:
    with pytest.raises(ValueError, match="w_epsilon must be in"):
        ControllerConfig(w_epsilon=-0.1)
    with pytest.raises(ValueError, match="w_epsilon must be in"):
        ControllerConfig(w_epsilon=1.1)
    with pytest.raises(ValueError, match="w_volatility must be in"):
        ControllerConfig(w_volatility=-0.1)
    with pytest.raises(ValueError, match="w_volatility must be in"):
        ControllerConfig(w_volatility=1.1)
    with pytest.raises(ValueError, match="w_portfolio_risk must be in"):
        ControllerConfig(w_portfolio_risk=-0.1)
    with pytest.raises(ValueError, match="w_portfolio_risk must be in"):
        ControllerConfig(w_portfolio_risk=1.1)
    with pytest.raises(ValueError, match="max_position_change must be in"):
        ControllerConfig(max_position_change=0.0)
    with pytest.raises(ValueError, match="max_position_change must be in"):
        ControllerConfig(max_position_change=1.1)
    with pytest.raises(ValueError, match="max_position_change must be in"):
        ControllerConfig(max_position_change=math.inf)
    with pytest.raises(ValueError, match="tau_floor must be in"):
        ControllerConfig(tau_floor=-0.1)
    with pytest.raises(ValueError, match="tau_floor must be in"):
        ControllerConfig(tau_floor=1.0)
    with pytest.raises(ValueError, match="tau_floor must be in"):
        ControllerConfig(tau_floor=math.inf)
    with pytest.raises(ValueError, match="rebalance_threshold must be in"):
        ControllerConfig(rebalance_threshold=-0.1)
    with pytest.raises(ValueError, match="rebalance_threshold must be in"):
        ControllerConfig(rebalance_threshold=1.1)
    with pytest.raises(ValueError, match="rebalance_threshold must be in"):
        ControllerConfig(rebalance_threshold=math.inf)
    with pytest.raises(ValueError, match="tau_confirm_threshold must be finite"):
        ControllerConfig(tau_confirm_threshold=math.inf)
    with pytest.raises(ValueError, match="trend_persistence_window must be positive"):
        ControllerConfig(trend_persistence_window=0)
    with pytest.raises(ValueError, match="persistence_floor must be in"):
        ControllerConfig(persistence_floor=-0.1)
    with pytest.raises(ValueError, match="persistence_floor must be in"):
        ControllerConfig(persistence_floor=1.0)


def test_backtest_runs_with_minimal_controller_and_minimal_state_estimator() -> None:
    result = BacktestEngine(
        controller=MinimalContinuousController(),
        state_estimator=MinimalStateEstimator(),
        config=BacktestConfig(fee_rate=0.001),
    ).run(pd.Series([100.0, 101.0, 102.0, 101.0, 103.0]))

    assert not result.empty
    assert result["position"].between(0.0, 1.0).all()

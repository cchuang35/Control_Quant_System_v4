import pandas as pd
import pytest

from src.v4 import (
    ControllerConfig,
    MinimalContinuousController,
    StateVector,
    V42_CANDIDATE_D_DESCRIPTION,
    V42_CANDIDATE_D_VERSION_NAME,
    make_v42_candidate_b_config,
    make_v42_candidate_c_config,
    make_v42_candidate_d_config,
    run_v42_candidate_d_backtest,
)


def test_v42_candidate_d_uses_k_tau_tau_floor_and_deadband() -> None:
    config = make_v42_candidate_d_config()

    assert config.version_name == V42_CANDIDATE_D_VERSION_NAME == "v4.2-candidate-D"
    assert "rebalance deadband of 0.01" in V42_CANDIDATE_D_DESCRIPTION
    assert config.state_estimator.k_tau == pytest.approx(5.0)
    assert config.controller.tau_floor == pytest.approx(0.10)
    assert config.controller.rebalance_threshold == pytest.approx(0.01)


def test_v42_candidate_d_is_based_on_c_not_b() -> None:
    candidate_b = make_v42_candidate_b_config()
    candidate_c = make_v42_candidate_c_config()
    candidate_d = make_v42_candidate_d_config()

    assert candidate_b.controller.w_portfolio_risk == pytest.approx(0.90)
    assert candidate_d.controller.w_portfolio_risk == pytest.approx(candidate_c.controller.w_portfolio_risk)
    assert candidate_d.controller.w_portfolio_risk == pytest.approx(0.75)


def test_v42_candidate_c_remains_without_deadband() -> None:
    candidate_c = make_v42_candidate_c_config()

    assert candidate_c.controller.tau_floor == pytest.approx(0.10)
    assert candidate_c.controller.rebalance_threshold == pytest.approx(0.0)


def test_v42_candidate_d_only_changes_rebalance_threshold_from_c() -> None:
    candidate_c = make_v42_candidate_c_config()
    candidate_d = make_v42_candidate_d_config()

    assert candidate_d.filter == candidate_c.filter
    assert candidate_d.backtest == candidate_c.backtest
    assert candidate_d.periods_per_year == candidate_c.periods_per_year
    assert candidate_d.state_estimator == candidate_c.state_estimator
    assert candidate_d.controller.w_epsilon == pytest.approx(candidate_c.controller.w_epsilon)
    assert candidate_d.controller.w_volatility == pytest.approx(candidate_c.controller.w_volatility)
    assert candidate_d.controller.w_portfolio_risk == pytest.approx(candidate_c.controller.w_portfolio_risk)
    assert candidate_d.controller.max_position_change == pytest.approx(candidate_c.controller.max_position_change)
    assert candidate_d.controller.tau_floor == pytest.approx(candidate_c.controller.tau_floor)
    assert candidate_d.controller.rebalance_threshold == pytest.approx(0.01)


def test_deadband_skips_small_position_change() -> None:
    controller = MinimalContinuousController(
        ControllerConfig(max_position_change=1.0, tau_floor=0.10, rebalance_threshold=0.01)
    )

    trace = controller.explain(StateVector(tau=0.1081, nu=0.0, epsilon=0.0, rho=0.0, previous_position=0.0))

    assert trace["pre_deadband_target"] == pytest.approx(0.009)
    assert trace["deadband_skip"] == pytest.approx(1.0)
    assert trace["raw_target_position"] == pytest.approx(0.0)


def test_deadband_keeps_computed_target_at_threshold() -> None:
    controller = MinimalContinuousController(
        ControllerConfig(max_position_change=1.0, tau_floor=0.10, rebalance_threshold=0.01)
    )

    trace = controller.explain(StateVector(tau=0.118, nu=0.0, epsilon=0.0, rho=0.0, previous_position=0.0))

    assert trace["pre_deadband_target"] == pytest.approx(0.02)
    assert trace["deadband_skip"] == pytest.approx(0.0)
    assert trace["raw_target_position"] == pytest.approx(0.02)


def test_v42_candidate_d_output_is_bounded() -> None:
    controller = MinimalContinuousController(
        ControllerConfig(max_position_change=1.0, tau_floor=0.10, rebalance_threshold=0.01)
    )
    target = controller.decide(StateVector(tau=10.0, nu=-10.0, epsilon=10.0, rho=-10.0, previous_position=10.0))

    assert 0.0 <= target <= 1.0


def test_v42_candidate_d_can_run_inside_backtest_engine() -> None:
    prices = pd.Series(
        [100.0, 101.0, 102.0, 101.0, 103.0, 104.0],
        index=pd.date_range("2024-01-01", periods=6, freq="D"),
    )

    result = run_v42_candidate_d_backtest(prices)

    assert not result.empty
    assert result["position"].between(0.0, 1.0).all()
    assert (result["equity"] > 0.0).all()

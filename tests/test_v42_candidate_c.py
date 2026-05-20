import pandas as pd
import pytest

from src.v4 import (
    ControllerConfig,
    MinimalContinuousController,
    StateVector,
    V42_CANDIDATE_C_DESCRIPTION,
    V42_CANDIDATE_C_VERSION_NAME,
    make_v42_candidate_a_config,
    make_v42_candidate_b_config,
    make_v42_candidate_c_config,
    run_v42_candidate_c_backtest,
)


def test_v42_candidate_c_uses_k_tau_5_and_tau_floor() -> None:
    config = make_v42_candidate_c_config()

    assert config.version_name == V42_CANDIDATE_C_VERSION_NAME == "v4.2-candidate-C"
    assert "tau_floor base-exposure mapping" in V42_CANDIDATE_C_DESCRIPTION
    assert config.state_estimator.k_tau == pytest.approx(5.0)
    assert config.controller.tau_floor == pytest.approx(0.10)


def test_v42_candidate_a_remains_tau_floor_zero() -> None:
    candidate_a = make_v42_candidate_a_config()

    assert candidate_a.state_estimator.k_tau == pytest.approx(5.0)
    assert candidate_a.controller.tau_floor == pytest.approx(0.0)


def test_v42_candidate_c_does_not_use_candidate_b_as_base() -> None:
    candidate_b = make_v42_candidate_b_config()
    candidate_c = make_v42_candidate_c_config()

    assert candidate_b.controller.w_portfolio_risk == pytest.approx(0.90)
    assert candidate_c.controller.w_portfolio_risk == pytest.approx(0.75)


def test_v42_candidate_c_only_changes_tau_floor_from_candidate_a() -> None:
    candidate_a = make_v42_candidate_a_config()
    candidate_c = make_v42_candidate_c_config()

    assert candidate_c.filter == candidate_a.filter
    assert candidate_c.backtest == candidate_a.backtest
    assert candidate_c.periods_per_year == candidate_a.periods_per_year
    assert candidate_c.state_estimator == candidate_a.state_estimator
    assert candidate_c.controller.w_epsilon == pytest.approx(candidate_a.controller.w_epsilon)
    assert candidate_c.controller.w_volatility == pytest.approx(candidate_a.controller.w_volatility)
    assert candidate_c.controller.w_portfolio_risk == pytest.approx(candidate_a.controller.w_portfolio_risk)
    assert candidate_c.controller.max_position_change == pytest.approx(candidate_a.controller.max_position_change)
    assert candidate_c.controller.tau_floor == pytest.approx(0.10)
    assert candidate_a.controller.tau_floor == pytest.approx(0.0)


def test_tau_floor_base_exposure_mapping() -> None:
    controller = MinimalContinuousController(ControllerConfig(max_position_change=1.0, tau_floor=0.10))

    below = controller.explain(StateVector(tau=0.05, nu=0.0, epsilon=0.0, rho=0.0, previous_position=0.0))
    at_floor = controller.explain(StateVector(tau=0.10, nu=0.0, epsilon=0.0, rho=0.0, previous_position=0.0))
    midpoint = controller.explain(StateVector(tau=0.55, nu=0.0, epsilon=0.0, rho=0.0, previous_position=0.0))
    full = controller.explain(StateVector(tau=1.0, nu=0.0, epsilon=0.0, rho=0.0, previous_position=0.0))

    assert below["base_exposure"] == pytest.approx(0.0)
    assert at_floor["base_exposure"] == pytest.approx(0.0)
    assert midpoint["base_exposure"] == pytest.approx(0.5)
    assert full["base_exposure"] == pytest.approx(1.0)


def test_v42_candidate_c_output_is_bounded() -> None:
    controller = MinimalContinuousController(ControllerConfig(max_position_change=1.0, tau_floor=0.10))
    target = controller.decide(StateVector(tau=10.0, nu=-10.0, epsilon=10.0, rho=-10.0, previous_position=10.0))

    assert 0.0 <= target <= 1.0


def test_v42_candidate_c_can_run_inside_backtest_engine() -> None:
    prices = pd.Series(
        [100.0, 101.0, 102.0, 101.0, 103.0, 104.0],
        index=pd.date_range("2024-01-01", periods=6, freq="D"),
    )

    result = run_v42_candidate_c_backtest(prices)

    assert not result.empty
    assert result["position"].between(0.0, 1.0).all()
    assert (result["equity"] > 0.0).all()

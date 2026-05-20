import pandas as pd
import pytest

from src.v4 import (
    ControllerConfig,
    MinimalContinuousController,
    StateVector,
    V42_CANDIDATE_E_DESCRIPTION,
    V42_CANDIDATE_E_VERSION_NAME,
    make_v42_candidate_c_config,
    make_v42_candidate_d_config,
    make_v42_candidate_e_config,
    run_v42_candidate_e_backtest,
)


def test_v42_candidate_e_uses_defined_parameters() -> None:
    config = make_v42_candidate_e_config()

    assert config.version_name == V42_CANDIDATE_E_VERSION_NAME == "v4.2-candidate-E"
    assert "causal trend persistence gate" in V42_CANDIDATE_E_DESCRIPTION
    assert config.state_estimator.k_tau == pytest.approx(5.0)
    assert config.controller.tau_floor == pytest.approx(0.10)
    assert config.controller.tau_confirm_threshold == pytest.approx(0.25)
    assert config.controller.trend_persistence_window == 10
    assert config.controller.persistence_floor == pytest.approx(0.50)
    assert config.controller.use_trend_persistence_gate is True


def test_v42_candidate_e_is_based_on_c_not_d() -> None:
    candidate_c = make_v42_candidate_c_config()
    candidate_d = make_v42_candidate_d_config()
    candidate_e = make_v42_candidate_e_config()

    assert candidate_d.controller.rebalance_threshold == pytest.approx(0.01)
    assert candidate_e.controller.rebalance_threshold == pytest.approx(candidate_c.controller.rebalance_threshold)
    assert candidate_e.controller.rebalance_threshold == pytest.approx(0.0)
    assert candidate_e.controller.w_portfolio_risk == pytest.approx(candidate_c.controller.w_portfolio_risk)


def test_v42_candidate_c_remains_without_persistence_gate() -> None:
    candidate_c = make_v42_candidate_c_config()

    assert candidate_c.controller.tau_floor == pytest.approx(0.10)
    assert candidate_c.controller.use_trend_persistence_gate is False


def test_trend_persistence_state_starts_at_zero_and_resets() -> None:
    controller = MinimalContinuousController(make_v42_candidate_e_config().controller)

    assert controller.trend_persistence_state == pytest.approx(0.0)
    controller.decide(StateVector(tau=1.0, nu=0.0, epsilon=0.0, rho=0.0, previous_position=0.0))
    assert controller.trend_persistence_state > 0.0
    controller.reset()
    assert controller.trend_persistence_state == pytest.approx(0.0)


def test_low_tau_keeps_persistence_gate_and_base_exposure_zero() -> None:
    controller = MinimalContinuousController(make_v42_candidate_e_config().controller)
    trace = {}
    for _ in range(20):
        trace = controller.explain(StateVector(tau=0.20, nu=0.0, epsilon=0.0, rho=0.0, previous_position=0.0))

    assert trace["trend_persistence_state"] < 0.01
    assert trace["trend_persistence_gate"] == pytest.approx(0.0)
    assert trace["base_exposure"] == pytest.approx(0.0)


def test_sustained_confirmed_tau_allows_positive_gate_and_exposure() -> None:
    controller = MinimalContinuousController(make_v42_candidate_e_config().controller)
    trace = {}
    for _ in range(10):
        trace = controller.explain(StateVector(tau=0.60, nu=0.0, epsilon=0.0, rho=0.0, previous_position=0.0))

    assert trace["trend_persistence_state"] > 0.50
    assert trace["trend_persistence_gate"] > 0.0
    assert trace["base_exposure_C"] > 0.0
    assert trace["base_exposure"] > 0.0


def test_brief_confirmed_tau_spike_is_suppressed() -> None:
    controller = MinimalContinuousController(make_v42_candidate_e_config().controller)

    first = controller.explain(StateVector(tau=0.60, nu=0.0, epsilon=0.0, rho=0.0, previous_position=0.0))
    second = controller.explain(StateVector(tau=0.60, nu=0.0, epsilon=0.0, rho=0.0, previous_position=0.0))

    assert first["trend_persistence_gate"] == pytest.approx(0.0)
    assert second["trend_persistence_gate"] == pytest.approx(0.0)
    assert second["base_exposure"] == pytest.approx(0.0)


def test_v42_candidate_e_output_is_bounded() -> None:
    controller = MinimalContinuousController(make_v42_candidate_e_config().controller)
    target = controller.decide(StateVector(tau=10.0, nu=-10.0, epsilon=10.0, rho=-10.0, previous_position=10.0))

    assert 0.0 <= target <= 1.0


def test_v42_candidate_e_can_run_inside_backtest_engine() -> None:
    prices = pd.Series(
        [100.0, 101.0, 102.0, 101.0, 103.0, 104.0, 105.0, 106.0],
        index=pd.date_range("2024-01-01", periods=8, freq="D"),
    )

    result = run_v42_candidate_e_backtest(prices)

    assert not result.empty
    assert result["position"].between(0.0, 1.0).all()
    assert (result["equity"] > 0.0).all()

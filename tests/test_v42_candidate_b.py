import pandas as pd
import pytest

from src.v4 import (
    V42_CANDIDATE_B_DESCRIPTION,
    V42_CANDIDATE_B_VERSION_NAME,
    make_v42_candidate_a_config,
    make_v42_candidate_b_config,
    run_v42_candidate_b_backtest,
)


def test_v42_candidate_b_uses_k_tau_5() -> None:
    config = make_v42_candidate_b_config()

    assert config.version_name == V42_CANDIDATE_B_VERSION_NAME == "v4.2-candidate-B"
    assert "stronger portfolio risk feedback" in V42_CANDIDATE_B_DESCRIPTION
    assert config.state_estimator.k_tau == pytest.approx(5.0)


def test_v42_candidate_b_uses_stronger_portfolio_risk_feedback() -> None:
    config = make_v42_candidate_b_config()

    assert config.controller.w_portfolio_risk == pytest.approx(0.90)


def test_v42_candidate_a_remains_unchanged() -> None:
    candidate_a = make_v42_candidate_a_config()

    assert candidate_a.state_estimator.k_tau == pytest.approx(5.0)
    assert candidate_a.controller.w_portfolio_risk == pytest.approx(0.75)


def test_v42_candidate_b_only_changes_portfolio_risk_from_candidate_a() -> None:
    candidate_a = make_v42_candidate_a_config()
    candidate_b = make_v42_candidate_b_config()

    assert candidate_b.filter == candidate_a.filter
    assert candidate_b.backtest == candidate_a.backtest
    assert candidate_b.periods_per_year == candidate_a.periods_per_year
    assert candidate_b.state_estimator == candidate_a.state_estimator
    assert candidate_b.controller.w_epsilon == pytest.approx(candidate_a.controller.w_epsilon)
    assert candidate_b.controller.w_volatility == pytest.approx(candidate_a.controller.w_volatility)
    assert candidate_b.controller.max_position_change == pytest.approx(candidate_a.controller.max_position_change)
    assert candidate_b.controller.w_portfolio_risk == pytest.approx(0.90)
    assert candidate_a.controller.w_portfolio_risk == pytest.approx(0.75)


def test_v42_candidate_b_can_run_inside_backtest_engine() -> None:
    prices = pd.Series(
        [100.0, 101.0, 102.0, 101.0, 103.0, 104.0],
        index=pd.date_range("2024-01-01", periods=6, freq="D"),
    )

    result = run_v42_candidate_b_backtest(prices)

    assert not result.empty
    assert result["position"].between(0.0, 1.0).all()
    assert (result["equity"] > 0.0).all()

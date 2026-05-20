import pandas as pd
import pytest

from src.v4 import (
    V41_VERSION_NAME,
    V42_CANDIDATE_A_DESCRIPTION,
    V42_CANDIDATE_A_VERSION_NAME,
    create_v41_default_config,
    make_v42_candidate_a_config,
    run_v42_candidate_a_backtest,
)


def test_v42_candidate_a_uses_k_tau_5() -> None:
    config = make_v42_candidate_a_config()

    assert config.version_name == V42_CANDIDATE_A_VERSION_NAME == "v4.2-candidate-A"
    assert "changes only k_tau from 1.0 to 5.0" in V42_CANDIDATE_A_DESCRIPTION
    assert config.state_estimator.k_tau == pytest.approx(5.0)


def test_v41_default_still_uses_k_tau_1() -> None:
    config = create_v41_default_config()

    assert config.version_name == V41_VERSION_NAME
    assert config.state_estimator.k_tau == pytest.approx(1.0)


def test_v42_candidate_a_only_changes_k_tau_from_v41() -> None:
    v41 = create_v41_default_config()
    v42 = make_v42_candidate_a_config()

    assert v42.filter == v41.filter
    assert v42.controller == v41.controller
    assert v42.backtest == v41.backtest
    assert v42.periods_per_year == v41.periods_per_year
    assert v42.state_estimator.k_epsilon == pytest.approx(v41.state_estimator.k_epsilon)
    assert v42.state_estimator.vol_ref == pytest.approx(v41.state_estimator.vol_ref)
    assert v42.state_estimator.drawdown_ref == pytest.approx(v41.state_estimator.drawdown_ref)
    assert v42.state_estimator.epsilon == pytest.approx(v41.state_estimator.epsilon)
    assert v42.state_estimator.k_tau == pytest.approx(5.0)
    assert v41.state_estimator.k_tau == pytest.approx(1.0)


def test_v42_candidate_a_can_run_inside_backtest_engine() -> None:
    prices = pd.Series(
        [100.0, 101.0, 102.0, 101.0, 103.0, 104.0],
        index=pd.date_range("2024-01-01", periods=6, freq="D"),
    )

    result = run_v42_candidate_a_backtest(prices)

    assert not result.empty
    assert result["position"].between(0.0, 1.0).all()
    assert (result["equity"] > 0.0).all()

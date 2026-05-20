import pytest

from src.v3.data_types import MarketEstimateV3
from src.v3.long_term_controller import (
    AGGRESSIVE_BASE_POSITIONS,
    CONSERVATIVE_BASE_POSITIONS,
    LongTermControllerConfig,
    aggressive_long_term_config,
    conservative_long_term_config,
    decide_long_term_position,
)


def estimate(long_regime: str, *, short_regime: str = "noise", risk_state: str = "normal") -> MarketEstimateV3:
    return MarketEstimateV3(
        timestamp="t",
        long_regime=long_regime,
        short_regime=short_regime,
        trend_strength=0.0,
        volatility_state="normal",
        drawdown_state="normal",
        risk_state=risk_state,
        confidence_score=0.7,
        allow_entry=True,
        allow_hold=True,
        notes={},
    )


def test_v3_long_term_controller_conservative_mapping() -> None:
    config = conservative_long_term_config()
    for regime, expected_position in CONSERVATIVE_BASE_POSITIONS.items():
        decision = decide_long_term_position(estimate(regime), config=config)
        assert decision.base_position == expected_position
        assert decision.long_regime == regime
        assert "conservative_long_term_mapping" in decision.reason


def test_v3_long_term_controller_aggressive_mapping() -> None:
    config = aggressive_long_term_config()
    for regime, expected_position in AGGRESSIVE_BASE_POSITIONS.items():
        decision = decide_long_term_position(estimate(regime), config=config)
        assert decision.base_position == expected_position
        assert "aggressive_long_term_mapping" in decision.reason


def test_v3_long_term_controller_ignores_short_term_and_risk_state() -> None:
    base = decide_long_term_position(estimate("bull", short_regime="noise", risk_state="normal"))
    stressed = decide_long_term_position(estimate("bull", short_regime="breakdown", risk_state="risk_off"))

    assert base.base_position == stressed.base_position == 0.50


def test_v3_long_term_controller_rejects_invalid_config() -> None:
    with pytest.raises(ValueError, match="missing regimes"):
        decide_long_term_position(estimate("bull"), config=LongTermControllerConfig(base_positions={"bull": 0.50}))

    bad_position = dict(CONSERVATIVE_BASE_POSITIONS)
    bad_position["bull"] = 0.60
    with pytest.raises(ValueError, match="discrete positions"):
        decide_long_term_position(estimate("bull"), config=LongTermControllerConfig(base_positions=bad_position))

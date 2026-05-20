from dataclasses import dataclass

import pytest

from src.v3.data_types import LongTermDecisionV3, MarketEstimateV3, ShortTermDecisionV3
from src.v3.risk_supervisor import PortfolioRiskStateV3, RiskSupervisorConfig, supervise_risk


def estimate(*, volatility_state: str = "normal", risk_state: str = "normal") -> MarketEstimateV3:
    return MarketEstimateV3(
        timestamp="t",
        long_regime="bull",
        short_regime="noise",
        trend_strength=0.5,
        volatility_state=volatility_state,
        drawdown_state="normal",
        risk_state=risk_state,
        confidence_score=0.8,
        allow_entry=True,
        allow_hold=True,
        notes={},
    )


def long_decision() -> LongTermDecisionV3:
    return LongTermDecisionV3("t", 0.50, "test", "bull", 0.8)


def short_decision(adjustment: float = 0.0) -> ShortTermDecisionV3:
    return ShortTermDecisionV3("t", adjustment, "test", "noise", False)


def state(drawdown: float, *, losses: int = 0, realized_volatility: float = 0.02) -> PortfolioRiskStateV3:
    return PortfolioRiskStateV3(
        portfolio_drawdown=drawdown,
        realized_volatility=realized_volatility,
        consecutive_losses=losses,
        current_position=0.5,
    )


def test_v3_risk_supervisor_drawdown_caps() -> None:
    cases = [
        (-0.01, 1.00, "normal"),
        (-0.05, 0.75, "no_new_entry"),
        (-0.10, 0.50, "reduce_only"),
        (-0.15, 0.25, "force_deleverage"),
        (-0.20, 0.00, "risk_off"),
    ]
    for drawdown, expected_cap, expected_action in cases:
        decision = supervise_risk(estimate(), state(drawdown), long_decision(), short_decision())
        assert decision.risk_cap == expected_cap
        assert decision.risk_action == expected_action
        assert decision.portfolio_drawdown == drawdown


def test_v3_risk_supervisor_volatility_caps() -> None:
    high = supervise_risk(estimate(volatility_state="high"), state(-0.01), long_decision(), short_decision())
    extreme = supervise_risk(estimate(volatility_state="extreme"), state(-0.01), long_decision(), short_decision())
    extreme_025 = supervise_risk(
        estimate(volatility_state="extreme"),
        state(-0.01),
        long_decision(),
        short_decision(),
        config=RiskSupervisorConfig(extreme_volatility_cap=0.25),
    )

    assert high.risk_cap == 0.75
    assert high.risk_action == "normal"
    assert extreme.risk_cap == 0.50
    assert extreme.risk_action == "reduce_only"
    assert extreme_025.risk_cap == 0.25


def test_v3_risk_supervisor_consecutive_loss_rules() -> None:
    two = supervise_risk(estimate(), state(-0.01, losses=2), long_decision(), short_decision(0.25))
    three = supervise_risk(estimate(), state(-0.01, losses=3), long_decision(), short_decision())
    four = supervise_risk(estimate(), state(-0.01, losses=4), long_decision(), short_decision())

    assert two.risk_cap == 1.0
    assert two.risk_action == "no_new_entry"
    assert "block_short_addition" in two.reason
    assert three.risk_cap == 0.50
    assert three.risk_action == "reduce_only"
    assert four.risk_cap == 0.25
    assert four.risk_action == "risk_off"


def test_v3_risk_supervisor_market_risk_off_has_highest_authority() -> None:
    decision = supervise_risk(
        estimate(risk_state="risk_off"),
        state(-0.01),
        long_decision(),
        short_decision(0.25),
    )

    assert decision.risk_action == "risk_off"
    assert decision.risk_cap == 0.0
    assert "market_estimate_risk_off" in decision.reason


def test_v3_risk_supervisor_feature_switches_disable_targeted_rules() -> None:
    no_risk = RiskSupervisorConfig(
        enable_drawdown_cap=False,
        enable_volatility_cap=False,
        enable_consecutive_loss_rules=False,
        enable_market_risk_state=False,
    )
    decision = supervise_risk(
        estimate(volatility_state="extreme", risk_state="risk_off"),
        state(-0.25, losses=4),
        long_decision(),
        short_decision(0.25),
        config=no_risk,
    )

    assert decision.risk_cap == 1.0
    assert decision.risk_action == "normal"
    assert "disabled" in decision.reason


def test_v3_risk_supervisor_supports_dict_and_external_portfolio_state() -> None:
    @dataclass(frozen=True)
    class ExternalState:
        portfolio_drawdown: float
        realized_volatility: float
        consecutive_losses: int
        current_position: float
        recent_turnover: float
        fee_drag: float

    from_dict = supervise_risk(
        estimate(),
        {"portfolio_drawdown": -0.06, "realized_volatility": 0.02, "consecutive_losses": 0, "current_position": 0.5},
        long_decision(),
        short_decision(),
    )
    from_external = supervise_risk(
        estimate(),
        ExternalState(-0.01, 0.02, 0, 0.5, 1.0, 0.0),
        long_decision(),
        short_decision(),
        config=RiskSupervisorConfig(turnover_caution=0.5),
    )

    assert from_dict.risk_cap == 0.75
    assert from_external.risk_action == "no_new_entry"
    assert "turnover_caution" in from_external.reason


def test_v3_risk_supervisor_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="missing required fields"):
        supervise_risk(estimate(), {"portfolio_drawdown": -0.01}, long_decision(), short_decision())

    with pytest.raises(ValueError, match="risk_off_cap"):
        supervise_risk(
            estimate(),
            state(-0.01),
            long_decision(),
            short_decision(),
            config=RiskSupervisorConfig(risk_off_cap=0.5),
        )

import pytest

from src.v3.data_types import LongTermDecisionV3, RiskDecisionV3, ShortTermDecisionV3
from src.v3.position_composer import PositionComposerConfig, compose_target_position


def long(base_position: float) -> LongTermDecisionV3:
    return LongTermDecisionV3("t", base_position, "long", "bull", 0.8)


def short(adjustment: float) -> ShortTermDecisionV3:
    return ShortTermDecisionV3("t", adjustment, "short", "noise", False)


def risk(cap: float, action: str = "normal") -> RiskDecisionV3:
    return RiskDecisionV3("t", cap, action, "risk", -0.01, 0.02)


def test_v3_position_composer_floor_rounds_discrete_target() -> None:
    decision = compose_target_position(long(0.50), short(0.25), risk(1.00))

    assert decision.raw_target_position == 0.75
    assert decision.risk_cap == 1.00
    assert decision.target_position == 0.75
    assert decision.executed_position == decision.target_position
    assert decision.trade_amount == 0.0
    assert "pending_execution" in decision.execution_reason


def test_v3_position_composer_applies_risk_cap_before_rounding() -> None:
    decision = compose_target_position(long(0.75), short(0.25), risk(0.60, "reduce_only"))

    assert decision.raw_target_position == 1.00
    assert decision.target_position == 0.50
    assert "risk_action=reduce_only" in decision.execution_reason


def test_v3_position_composer_floor_vs_nearest() -> None:
    floor_decision = compose_target_position(long(0.50), short(0.25), risk(0.62))
    nearest_decision = compose_target_position(
        long(0.50),
        short(0.25),
        risk(0.62),
        config=PositionComposerConfig(rounding_mode="nearest"),
    )

    assert floor_decision.target_position == 0.50
    assert nearest_decision.target_position == 0.50

    nearest_up = compose_target_position(
        long(0.50),
        short(0.25),
        risk(0.64),
        config=PositionComposerConfig(rounding_mode="nearest"),
    )
    assert nearest_up.target_position == 0.75


def test_v3_position_composer_clips_to_min_and_max() -> None:
    below_min = compose_target_position(long(0.00), short(-0.25), risk(1.00))
    above_max = compose_target_position(long(1.00), short(0.25), risk(2.00))

    assert below_min.target_position == 0.00
    assert above_max.target_position == 1.00


def test_v3_position_composer_optional_leverage() -> None:
    leveraged = compose_target_position(
        long(1.00),
        short(0.25),
        risk(1.25),
        config=PositionComposerConfig(max_position=1.25, allow_leverage=True),
    )

    assert leveraged.target_position == 1.25


def test_v3_position_composer_rejects_invalid_config() -> None:
    with pytest.raises(ValueError, match="rounding_mode"):
        compose_target_position(long(0.50), short(0.0), risk(1.0), config=PositionComposerConfig(rounding_mode="ceil"))

    with pytest.raises(ValueError, match="allow_leverage"):
        compose_target_position(
            long(0.50),
            short(0.0),
            risk(1.0),
            config=PositionComposerConfig(allowed_positions=(0.0, 0.25, 1.25)),
        )

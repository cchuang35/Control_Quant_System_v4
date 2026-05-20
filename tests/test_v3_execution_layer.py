import pytest

from src.v3.data_types import FinalPositionDecisionV3
from src.v3.execution_layer import (
    ExecutionConfig,
    apply_execution,
    compute_strategy_return_net,
    execute_target_position,
)


def decision(target_position: float, risk_cap: float = 1.0) -> FinalPositionDecisionV3:
    return FinalPositionDecisionV3(
        timestamp="t",
        base_position=target_position,
        position_adjustment=0.0,
        raw_target_position=target_position,
        risk_cap=risk_cap,
        target_position=target_position,
        executed_position=target_position,
        trade_amount=0.0,
        execution_reason="pending_execution",
    )


def test_v3_execution_executes_target_outside_no_trade_zone() -> None:
    executed = apply_execution(decision(0.75), current_position=0.25, risk_action="normal")

    assert executed.executed_position == 0.75
    assert executed.trade_amount == 0.50
    assert "execute_target" in executed.execution_reason


def test_v3_execution_no_trade_zone_keeps_current_position() -> None:
    executed = apply_execution(
        decision(0.40),
        current_position=0.25,
        risk_action="normal",
        minimum_position_step=0.25,
    )

    assert executed.executed_position == 0.25
    assert executed.trade_amount == 0.0
    assert "no_trade_zone" in executed.execution_reason


def test_v3_execution_exact_target_hold_is_not_no_trade_zone() -> None:
    executed = apply_execution(decision(0.25), current_position=0.25, risk_action="normal")

    assert executed.executed_position == 0.25
    assert executed.trade_amount == 0.0
    assert "hold_target" in executed.execution_reason


def test_v3_execution_risk_actions() -> None:
    risk_off = apply_execution(decision(0.75), current_position=0.50, risk_action="risk_off")
    reduce_blocks = apply_execution(decision(0.75), current_position=0.50, risk_action="reduce_only")
    reduce_allows = apply_execution(decision(0.25), current_position=0.75, risk_action="reduce_only")
    no_new_entry = apply_execution(decision(0.25), current_position=0.0, risk_action="no_new_entry")
    force = apply_execution(decision(0.75, risk_cap=0.25), current_position=0.75, risk_action="force_deleverage")

    assert risk_off.executed_position == 0.0
    assert reduce_blocks.executed_position == 0.50
    assert reduce_allows.executed_position == 0.25
    assert no_new_entry.executed_position == 0.0
    assert force.executed_position == 0.25


def test_v3_execute_target_position_caps_target() -> None:
    executed, reason = execute_target_position(
        current_position=0.25,
        target_position=1.0,
        risk_action="normal",
        risk_cap=0.50,
    )

    assert executed == 0.50
    assert "execute_target" in reason


def test_v3_strategy_return_net_uses_previous_position_and_costs() -> None:
    net = compute_strategy_return_net(
        previous_position=0.50,
        current_position=0.75,
        asset_return=0.02,
        fee_rate=0.001,
        slippage_rate=0.0005,
    )

    expected = 0.50 * 0.02 - 0.25 * 0.001 - 0.25 * 0.0005
    assert net == pytest.approx(expected)


def test_v3_execution_rejects_invalid_costs() -> None:
    with pytest.raises(ValueError, match="fee_rate"):
        compute_strategy_return_net(previous_position=0.0, current_position=0.25, asset_return=0.0, fee_rate=-0.1)

    with pytest.raises(ValueError, match="minimum_position_step"):
        execute_target_position(
            current_position=0.0,
            target_position=0.25,
            risk_action="normal",
            risk_cap=1.0,
            config=ExecutionConfig(minimum_position_step=0.0),
        )

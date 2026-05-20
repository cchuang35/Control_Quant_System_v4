"""v3 fee-aware execution layer.

The execution layer converts a composed target position into an executed
position using risk-action constraints and a no-trade zone. It also provides a
fee-aware return helper that preserves the no-look-ahead convention.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .data_types import FinalPositionDecisionV3


@dataclass(frozen=True)
class ExecutionConfig:
    """Configuration for v3 position execution decisions."""

    fee_rate: float = 0.001
    slippage_rate: float = 0.0
    minimum_position_step: float = 0.25
    risk_off_position: float = 0.0


def apply_execution(
    decision: FinalPositionDecisionV3,
    *,
    current_position: float,
    risk_action: str,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0,
    minimum_position_step: float = 0.25,
    confidence_score: float | None = None,
    risk_cap: float | None = None,
) -> FinalPositionDecisionV3:
    """Return ``decision`` with executed position and trade amount updated."""

    config = ExecutionConfig(
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        minimum_position_step=minimum_position_step,
    )
    executed_position, reason = execute_target_position(
        current_position=current_position,
        target_position=decision.target_position,
        risk_action=risk_action,
        risk_cap=decision.risk_cap if risk_cap is None else risk_cap,
        config=config,
        confidence_score=confidence_score,
    )
    trade_amount = abs(executed_position - float(current_position))
    return replace(
        decision,
        executed_position=float(executed_position),
        trade_amount=float(trade_amount),
        execution_reason=reason,
    )


def execute_target_position(
    *,
    current_position: float,
    target_position: float,
    risk_action: str,
    risk_cap: float,
    config: ExecutionConfig | None = None,
    confidence_score: float | None = None,
) -> tuple[float, str]:
    """Decide the executed position for one bar.

    This function does not inspect asset returns. It can therefore be used in a
    backtest where the signal at bar ``t`` determines position for bar ``t+1``.
    """

    config = config or ExecutionConfig()
    _validate_config(config)
    current = float(current_position)
    target = float(target_position)
    cap = max(0.0, float(risk_cap))
    risk_action = str(risk_action)

    if risk_action == "risk_off":
        executed = min(config.risk_off_position, cap)
        return executed, _reason("risk_off", current, target, executed, confidence_score)

    if risk_action == "force_deleverage":
        executed = min(target, cap, current)
        return executed, _reason("force_deleverage", current, target, executed, confidence_score)

    if risk_action == "reduce_only" and target > current:
        return current, _reason("reduce_only_blocks_increase", current, target, current, confidence_score)

    if risk_action == "no_new_entry" and current <= 0.0 and target > current:
        return current, _reason("no_new_entry_blocks_entry", current, target, current, confidence_score)

    capped_target = min(target, cap)
    if abs(capped_target - current) <= 1e-12:
        return current, _reason("hold_target", current, capped_target, current, confidence_score)

    if abs(capped_target - current) < config.minimum_position_step:
        return current, _reason("no_trade_zone", current, capped_target, current, confidence_score)

    return capped_target, _reason("execute_target", current, target, capped_target, confidence_score)


def compute_strategy_return_net(
    *,
    previous_position: float,
    current_position: float,
    asset_return: float,
    fee_rate: float,
    slippage_rate: float = 0.0,
) -> float:
    """Compute fee-aware net return without look-ahead bias.

    ``previous_position`` earns ``asset_return``. The position change from
    ``previous_position`` to ``current_position`` pays transaction costs.
    """

    if fee_rate < 0.0:
        raise ValueError("fee_rate must be non-negative")
    if slippage_rate < 0.0:
        raise ValueError("slippage_rate must be non-negative")
    trade_amount = abs(float(current_position) - float(previous_position))
    gross_return = float(previous_position) * float(asset_return)
    return gross_return - trade_amount * float(fee_rate) - trade_amount * float(slippage_rate)


def _reason(
    action: str,
    current_position: float,
    target_position: float,
    executed_position: float,
    confidence_score: float | None,
) -> str:
    confidence = "" if confidence_score is None else f", confidence={float(confidence_score):.2f}"
    return (
        f"{action}: current={float(current_position):.2f}, target={float(target_position):.2f},"
        f" executed={float(executed_position):.2f}{confidence}"
    )


def _validate_config(config: ExecutionConfig) -> None:
    if config.fee_rate < 0.0:
        raise ValueError("fee_rate must be non-negative")
    if config.slippage_rate < 0.0:
        raise ValueError("slippage_rate must be non-negative")
    if config.minimum_position_step <= 0.0:
        raise ValueError("minimum_position_step must be positive")
    if config.risk_off_position < 0.0:
        raise ValueError("risk_off_position must be non-negative")


__all__ = [
    "ExecutionConfig",
    "FinalPositionDecisionV3",
    "apply_execution",
    "compute_strategy_return_net",
    "execute_target_position",
]

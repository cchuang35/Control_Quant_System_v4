"""v3 position composer.

The composer combines long-term base exposure, short-term auxiliary adjustment,
and the risk supervisor's cap into a discrete target position. It does not
apply fees, no-trade zones, or execution rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

from .data_types import FinalPositionDecisionV3, LongTermDecisionV3, RiskDecisionV3, ShortTermDecisionV3


DEFAULT_ALLOWED_POSITIONS = (0.0, 0.25, 0.50, 0.75, 1.0)
LEVERAGED_ALLOWED_POSITIONS = (0.0, 0.25, 0.50, 0.75, 1.0, 1.25)
ROUNDING_MODES = ("floor", "nearest")


@dataclass(frozen=True)
class PositionComposerConfig:
    """Configuration for discrete v3 target position composition."""

    min_position: float = 0.0
    max_position: float = 1.0
    allowed_positions: tuple[float, ...] = field(default_factory=lambda: DEFAULT_ALLOWED_POSITIONS)
    rounding_mode: str = "floor"
    allow_leverage: bool = False


def compose_target_position(
    long_decision: LongTermDecisionV3,
    short_decision: ShortTermDecisionV3,
    risk_decision: RiskDecisionV3,
    config: PositionComposerConfig | None = None,
) -> FinalPositionDecisionV3:
    """Compose a risk-limited discrete target position.

    ``executed_position`` is set equal to ``target_position`` as a placeholder
    for the later execution layer. No fee or no-trade-zone logic is applied
    here.
    """

    config = config or PositionComposerConfig()
    allowed_positions = _validated_allowed_positions(config)
    raw_target_position = float(long_decision.base_position) + float(short_decision.position_adjustment)
    risk_limited_position = min(raw_target_position, float(risk_decision.risk_cap))
    clipped_position = _clip(risk_limited_position, config.min_position, config.max_position)
    target_position = _round_to_allowed(clipped_position, allowed_positions, config.rounding_mode)
    reason = (
        "pending_execution:"
        f" raw={raw_target_position:.2f}, risk_limited={risk_limited_position:.2f},"
        f" clipped={clipped_position:.2f}, rounded={target_position:.2f},"
        f" rounding_mode={config.rounding_mode}, risk_action={risk_decision.risk_action}"
    )
    return FinalPositionDecisionV3(
        timestamp=long_decision.timestamp,
        base_position=float(long_decision.base_position),
        position_adjustment=float(short_decision.position_adjustment),
        raw_target_position=float(raw_target_position),
        risk_cap=float(risk_decision.risk_cap),
        target_position=float(target_position),
        executed_position=float(target_position),
        trade_amount=0.0,
        execution_reason=reason,
    )


def _round_to_allowed(value: float, allowed_positions: tuple[float, ...], rounding_mode: str) -> float:
    if rounding_mode == "floor":
        candidates = [position for position in allowed_positions if position <= value + 1e-12]
        return candidates[-1] if candidates else allowed_positions[0]
    if rounding_mode == "nearest":
        return min(allowed_positions, key=lambda position: (abs(position - value), position))
    raise ValueError("rounding_mode must be 'floor' or 'nearest'")


def _validated_allowed_positions(config: PositionComposerConfig) -> tuple[float, ...]:
    if config.rounding_mode not in ROUNDING_MODES:
        raise ValueError("rounding_mode must be 'floor' or 'nearest'")
    if config.min_position > config.max_position:
        raise ValueError("min_position must be <= max_position")

    positions = config.allowed_positions
    if config.allow_leverage and max(positions, default=0.0) <= 1.0:
        positions = LEVERAGED_ALLOWED_POSITIONS
    if not config.allow_leverage and any(position > 1.0 for position in positions):
        raise ValueError("allowed_positions above 1.0 require allow_leverage=True")

    cleaned = tuple(sorted(float(position) for position in positions))
    if not cleaned:
        raise ValueError("allowed_positions must not be empty")
    if cleaned[0] < config.min_position - 1e-12 or cleaned[-1] > config.max_position + 1e-12:
        raise ValueError("allowed_positions must stay within min_position and max_position")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("allowed_positions must be unique")
    if any(not isfinite(position) for position in cleaned):
        raise ValueError("allowed_positions must be finite")
    return cleaned


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


__all__ = [
    "DEFAULT_ALLOWED_POSITIONS",
    "LEVERAGED_ALLOWED_POSITIONS",
    "FinalPositionDecisionV3",
    "PositionComposerConfig",
    "ROUNDING_MODES",
    "compose_target_position",
]

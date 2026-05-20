"""v3 long-term primary exposure controller.

This controller maps the long-term market regime into the base exposure for
v3. It deliberately ignores short-term signals and drawdown caps; those belong
to the short-term auxiliary controller and risk supervisor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .data_types import LongTermDecisionV3, MarketEstimateV3
from .market_estimator import LONG_REGIMES


CONSERVATIVE_BASE_POSITIONS = {
    "strong_bull": 0.75,
    "bull": 0.50,
    "neutral": 0.25,
    "bear": 0.00,
    "strong_bear": 0.00,
}
AGGRESSIVE_BASE_POSITIONS = {
    "strong_bull": 1.00,
    "bull": 0.75,
    "neutral": 0.50,
    "bear": 0.25,
    "strong_bear": 0.00,
}


@dataclass(frozen=True)
class LongTermControllerConfig:
    """Configuration for long-term regime-to-base-position mapping."""

    base_positions: dict[str, float] = field(default_factory=lambda: dict(CONSERVATIVE_BASE_POSITIONS))
    name: str = "conservative"


def conservative_long_term_config() -> LongTermControllerConfig:
    """Return the default conservative v3 long-term mapping."""

    return LongTermControllerConfig(base_positions=dict(CONSERVATIVE_BASE_POSITIONS), name="conservative")


def aggressive_long_term_config() -> LongTermControllerConfig:
    """Return the optional aggressive v3 long-term mapping."""

    return LongTermControllerConfig(base_positions=dict(AGGRESSIVE_BASE_POSITIONS), name="aggressive")


def decide_long_term_position(
    estimate: MarketEstimateV3,
    config: LongTermControllerConfig | None = None,
) -> LongTermDecisionV3:
    """Map a market estimate into the v3 long-term base position.

    Only the long-term regime is used to choose exposure. The estimate's
    short-term regime, drawdown state, and risk state are intentionally not
    applied here.
    """

    config = config or conservative_long_term_config()
    _validate_config(config)
    long_regime = str(estimate.long_regime)
    if long_regime not in config.base_positions:
        raise ValueError(f"unknown long_regime: {long_regime}")

    base_position = float(config.base_positions[long_regime])
    reason = (
        f"{config.name}_long_term_mapping:"
        f" long_regime={long_regime}, base_position={base_position:.2f}"
    )
    return LongTermDecisionV3(
        timestamp=estimate.timestamp,
        base_position=base_position,
        reason=reason,
        long_regime=long_regime,
        confidence_score=float(estimate.confidence_score),
    )


def _validate_config(config: LongTermControllerConfig) -> None:
    missing = set(LONG_REGIMES).difference(config.base_positions)
    if missing:
        raise ValueError(f"base_positions missing regimes: {sorted(missing)}")
    for regime, position in config.base_positions.items():
        if regime not in LONG_REGIMES:
            raise ValueError(f"unknown regime in base_positions: {regime}")
        if position not in {0.0, 0.25, 0.50, 0.75, 1.0}:
            raise ValueError(f"base position for {regime} must be one of the v3 discrete positions")


__all__ = [
    "AGGRESSIVE_BASE_POSITIONS",
    "CONSERVATIVE_BASE_POSITIONS",
    "LongTermControllerConfig",
    "LongTermDecisionV3",
    "aggressive_long_term_config",
    "conservative_long_term_config",
    "decide_long_term_position",
]

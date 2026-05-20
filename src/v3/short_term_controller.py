"""v3 short-term auxiliary controller.

This controller suggests a small tactical adjustment around the long-term base
position. It never overrides the long-term controller, never applies portfolio
drawdown caps, and never executes trades.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .data_types import MarketEstimateV3, ShortTermDecisionV3
from .market_estimator import LONG_REGIMES, SHORT_REGIMES


ADJUSTMENT_DOWN = -0.25
ADJUSTMENT_NONE = 0.0
ADJUSTMENT_UP = 0.25
ALLOWED_ADJUSTMENTS = {ADJUSTMENT_DOWN, ADJUSTMENT_NONE, ADJUSTMENT_UP}


@dataclass(frozen=True)
class ShortTermControllerConfig:
    """Thresholds and switches for the auxiliary short-term controller."""

    high_confidence: float = 0.70
    very_high_confidence: float = 0.85
    enable_pullback_add: bool = True
    enable_recovery_add: bool = True
    enable_overheat_reduce: bool = True
    enable_breakdown_reduce: bool = True
    allow_neutral_recovery_add: bool = True
    experimental_mode: bool = False


@dataclass(frozen=True)
class CooldownStateV3:
    """Minimal cooldown state for blocking new bull-like additions."""

    cooldown_active: bool = False
    reason: str = ""


def decide_short_term_adjustment(
    estimate: MarketEstimateV3,
    cooldown_state: CooldownStateV3 | bool | Any = False,
    config: ShortTermControllerConfig | None = None,
) -> ShortTermDecisionV3:
    """Return a v3 auxiliary position adjustment.

    The output is always one of ``-0.25``, ``0.0``, or ``+0.25``. Active
    cooldown blocks new bullish additions in bull-like regimes, preserving the
    v2 weak-bull cooldown idea without forcing an immediate exit.
    """

    config = config or ShortTermControllerConfig()
    _validate_config(config)
    long_regime = str(estimate.long_regime)
    short_regime = str(estimate.short_regime)
    if long_regime not in LONG_REGIMES:
        raise ValueError(f"unknown long_regime: {long_regime}")
    if short_regime not in SHORT_REGIMES:
        raise ValueError(f"unknown short_regime: {short_regime}")

    cooldown_active = _cooldown_active(cooldown_state)
    adjustment, reason = _raw_adjustment(estimate, config)
    bull_like_addition = adjustment > 0.0 and long_regime in {"strong_bull", "bull"}
    if cooldown_active and bull_like_addition:
        adjustment = ADJUSTMENT_NONE
        reason = f"cooldown_blocks_bull_like_addition: {reason}"

    if adjustment not in ALLOWED_ADJUSTMENTS:
        raise ValueError(f"position_adjustment must be one of {sorted(ALLOWED_ADJUSTMENTS)}")
    return ShortTermDecisionV3(
        timestamp=estimate.timestamp,
        position_adjustment=adjustment,
        reason=reason,
        short_regime=short_regime,
        cooldown_active=cooldown_active,
    )


def _raw_adjustment(estimate: MarketEstimateV3, config: ShortTermControllerConfig) -> tuple[float, str]:
    long_regime = str(estimate.long_regime)
    short_regime = str(estimate.short_regime)
    confidence = float(estimate.confidence_score)

    if long_regime in {"strong_bull", "bull"}:
        if short_regime == "pullback":
            if config.enable_pullback_add:
                return ADJUSTMENT_UP, "bullish_pullback_add"
            return ADJUSTMENT_NONE, "bullish_pullback_add_disabled"
        if short_regime == "recovery" and config.enable_recovery_add and confidence >= config.high_confidence:
            return ADJUSTMENT_UP, "bullish_recovery_high_confidence_add"
        if short_regime == "recovery":
            return ADJUSTMENT_NONE, "bullish_recovery_confidence_too_low"
        if short_regime == "overheat":
            if config.enable_overheat_reduce:
                return ADJUSTMENT_DOWN, "bullish_overheat_reduce"
            return ADJUSTMENT_NONE, "bullish_overheat_reduce_disabled"
        if short_regime == "breakdown":
            if config.enable_breakdown_reduce:
                return ADJUSTMENT_DOWN, "bullish_breakdown_reduce"
            return ADJUSTMENT_NONE, "bullish_breakdown_reduce_disabled"
        return ADJUSTMENT_NONE, "bullish_noise_hold_adjustment"

    if long_regime == "neutral":
        if (
            short_regime == "recovery"
            and config.enable_recovery_add
            and config.allow_neutral_recovery_add
            and confidence >= config.high_confidence
        ):
            return ADJUSTMENT_UP, "neutral_recovery_high_confidence_add"
        if short_regime == "breakdown":
            if config.enable_breakdown_reduce:
                return ADJUSTMENT_DOWN, "neutral_breakdown_reduce"
            return ADJUSTMENT_NONE, "neutral_breakdown_reduce_disabled"
        return ADJUSTMENT_NONE, "neutral_no_clear_auxiliary_edge"

    if long_regime in {"bear", "strong_bear"}:
        if (
            config.experimental_mode
            and short_regime == "recovery"
            and confidence >= config.very_high_confidence
        ):
            return ADJUSTMENT_UP, "experimental_bear_recovery_very_high_confidence_add"
        if short_regime in {"breakdown", "overheat"}:
            enabled = (
                config.enable_breakdown_reduce
                if short_regime == "breakdown"
                else config.enable_overheat_reduce
            )
            if enabled:
                return ADJUSTMENT_DOWN, f"{long_regime}_{short_regime}_reduce"
            return ADJUSTMENT_NONE, f"{long_regime}_{short_regime}_reduce_disabled"
        return ADJUSTMENT_NONE, f"{long_regime}_default_no_add"

    raise ValueError(f"unknown long_regime: {long_regime}")


def _cooldown_active(cooldown_state: CooldownStateV3 | bool | Any) -> bool:
    if isinstance(cooldown_state, bool):
        return cooldown_state
    if isinstance(cooldown_state, CooldownStateV3):
        return cooldown_state.cooldown_active
    if hasattr(cooldown_state, "cooldown_active"):
        return bool(cooldown_state.cooldown_active)
    if hasattr(cooldown_state, "active"):
        return bool(cooldown_state.active)
    return bool(cooldown_state)


def _validate_config(config: ShortTermControllerConfig) -> None:
    if not 0.0 <= config.high_confidence <= 1.0:
        raise ValueError("high_confidence must be in [0, 1]")
    if not 0.0 <= config.very_high_confidence <= 1.0:
        raise ValueError("very_high_confidence must be in [0, 1]")
    if config.very_high_confidence < config.high_confidence:
        raise ValueError("very_high_confidence must be >= high_confidence")


__all__ = [
    "ADJUSTMENT_DOWN",
    "ADJUSTMENT_NONE",
    "ADJUSTMENT_UP",
    "ALLOWED_ADJUSTMENTS",
    "CooldownStateV3",
    "ShortTermControllerConfig",
    "ShortTermDecisionV3",
    "decide_short_term_adjustment",
]

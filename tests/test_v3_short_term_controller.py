from dataclasses import dataclass

import pytest

from src.v3.data_types import MarketEstimateV3
from src.v3.short_term_controller import (
    ADJUSTMENT_DOWN,
    ADJUSTMENT_NONE,
    ADJUSTMENT_UP,
    CooldownStateV3,
    ShortTermControllerConfig,
    decide_short_term_adjustment,
)


def estimate(
    long_regime: str,
    short_regime: str,
    *,
    confidence_score: float = 0.75,
) -> MarketEstimateV3:
    return MarketEstimateV3(
        timestamp="t",
        long_regime=long_regime,
        short_regime=short_regime,
        trend_strength=0.0,
        volatility_state="normal",
        drawdown_state="normal",
        risk_state="normal",
        confidence_score=confidence_score,
        allow_entry=True,
        allow_hold=True,
        notes={},
    )


def test_v3_short_term_bullish_rules_and_cooldown_block_additions() -> None:
    pullback = decide_short_term_adjustment(estimate("bull", "pullback"), cooldown_state=False)
    assert pullback.position_adjustment == ADJUSTMENT_UP
    assert "pullback" in pullback.reason

    blocked = decide_short_term_adjustment(estimate("bull", "pullback"), cooldown_state=True)
    assert blocked.position_adjustment == ADJUSTMENT_NONE
    assert blocked.cooldown_active
    assert "cooldown_blocks" in blocked.reason

    overheat = decide_short_term_adjustment(estimate("strong_bull", "overheat"), cooldown_state=True)
    assert overheat.position_adjustment == ADJUSTMENT_DOWN
    assert overheat.cooldown_active

    breakdown = decide_short_term_adjustment(estimate("bull", "breakdown"))
    assert breakdown.position_adjustment == ADJUSTMENT_DOWN


def test_v3_short_term_recovery_requires_high_confidence() -> None:
    low = decide_short_term_adjustment(estimate("bull", "recovery", confidence_score=0.60))
    high = decide_short_term_adjustment(estimate("bull", "recovery", confidence_score=0.80))

    assert low.position_adjustment == ADJUSTMENT_NONE
    assert high.position_adjustment == ADJUSTMENT_UP


def test_v3_short_term_rule_switches_disable_targeted_auxiliary_rules() -> None:
    disabled = ShortTermControllerConfig(
        enable_pullback_add=False,
        enable_recovery_add=False,
        enable_overheat_reduce=False,
        enable_breakdown_reduce=False,
    )

    pullback = decide_short_term_adjustment(estimate("bull", "pullback"), config=disabled)
    recovery = decide_short_term_adjustment(estimate("bull", "recovery", confidence_score=0.90), config=disabled)
    overheat = decide_short_term_adjustment(estimate("bull", "overheat"), config=disabled)
    breakdown = decide_short_term_adjustment(estimate("bull", "breakdown"), config=disabled)

    assert pullback.position_adjustment == ADJUSTMENT_NONE
    assert recovery.position_adjustment == ADJUSTMENT_NONE
    assert overheat.position_adjustment == ADJUSTMENT_NONE
    assert breakdown.position_adjustment == ADJUSTMENT_NONE


def test_v3_short_term_neutral_rules() -> None:
    recovery = decide_short_term_adjustment(estimate("neutral", "recovery", confidence_score=0.80))
    breakdown = decide_short_term_adjustment(estimate("neutral", "breakdown", confidence_score=0.80))
    noise = decide_short_term_adjustment(estimate("neutral", "noise", confidence_score=0.80))

    assert recovery.position_adjustment == ADJUSTMENT_UP
    assert breakdown.position_adjustment == ADJUSTMENT_DOWN
    assert noise.position_adjustment == ADJUSTMENT_NONE

    disabled = decide_short_term_adjustment(
        estimate("neutral", "recovery", confidence_score=0.90),
        config=ShortTermControllerConfig(allow_neutral_recovery_add=False),
    )
    assert disabled.position_adjustment == ADJUSTMENT_NONE


def test_v3_short_term_bear_rules_do_not_add_without_experimental_mode() -> None:
    default = decide_short_term_adjustment(estimate("bear", "recovery", confidence_score=0.95))
    experimental = decide_short_term_adjustment(
        estimate("bear", "recovery", confidence_score=0.95),
        config=ShortTermControllerConfig(experimental_mode=True),
    )
    breakdown = decide_short_term_adjustment(estimate("strong_bear", "breakdown"))

    assert default.position_adjustment == ADJUSTMENT_NONE
    assert experimental.position_adjustment == ADJUSTMENT_UP
    assert "experimental" in experimental.reason
    assert breakdown.position_adjustment == ADJUSTMENT_DOWN


def test_v3_short_term_accepts_cooldown_state_objects() -> None:
    @dataclass(frozen=True)
    class ExternalCooldown:
        active: bool

    own_state = decide_short_term_adjustment(estimate("bull", "pullback"), CooldownStateV3(True))
    external_state = decide_short_term_adjustment(estimate("bull", "pullback"), ExternalCooldown(True))

    assert own_state.position_adjustment == ADJUSTMENT_NONE
    assert external_state.position_adjustment == ADJUSTMENT_NONE


def test_v3_short_term_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="unknown long_regime"):
        decide_short_term_adjustment(estimate("moon", "noise"))

    with pytest.raises(ValueError, match="unknown short_regime"):
        decide_short_term_adjustment(estimate("bull", "maybe"))

    with pytest.raises(ValueError, match="very_high_confidence"):
        decide_short_term_adjustment(
            estimate("bull", "noise"),
            config=ShortTermControllerConfig(high_confidence=0.9, very_high_confidence=0.8),
        )

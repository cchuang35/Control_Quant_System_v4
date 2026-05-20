"""Rule-based v3 market estimator.

This module converts v3 features into decision-ready market estimates. It is a
simple, transparent v3 estimator; particle-filter state estimation is
explicitly deferred to v4.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .data_types import MarketEstimateV3, MarketFeaturesV3


LONG_REGIMES = ("strong_bull", "bull", "neutral", "bear", "strong_bear")
SHORT_REGIMES = ("pullback", "recovery", "overheat", "breakdown", "noise")
VOLATILITY_STATES = ("low", "normal", "high", "extreme")
DRAW_DOWN_STATES = ("normal", "caution", "danger", "severe")
RISK_STATES = ("normal", "caution", "risk_off")
ESTIMATE_COLUMNS = [
    "timestamp",
    "long_regime",
    "short_regime",
    "trend_strength",
    "volatility_state",
    "drawdown_state",
    "risk_state",
    "confidence_score",
    "allow_entry",
    "allow_hold",
    "notes",
]


@dataclass(frozen=True)
class MarketEstimatorConfig:
    """Thresholds for the initial v3 rule-based estimator."""

    bull_trend_threshold: float = 0.03
    strong_bull_trend_threshold: float = 0.08
    bear_trend_threshold: float = -0.03
    strong_bear_trend_threshold: float = -0.10
    bull_momentum_threshold: float = 0.01
    strong_bull_momentum_threshold: float = 0.04
    bear_momentum_threshold: float = -0.01
    strong_bear_momentum_threshold: float = -0.06
    pullback_momentum_threshold: float = -0.005
    pullback_drawdown_threshold: float = -0.01
    recovery_momentum_threshold: float = 0.005
    recovery_drawdown_threshold: float = -0.01
    overheat_momentum_threshold: float = 0.04
    overheat_ma_stretch: float = 0.03
    breakdown_momentum_threshold: float = -0.035
    breakdown_shock_threshold: float = 0.75
    low_volatility_ratio: float = 0.70
    high_volatility_ratio: float = 1.50
    extreme_volatility_ratio: float = 2.50
    caution_drawdown: float = -0.05
    danger_drawdown: float = -0.12
    severe_drawdown: float = -0.20
    risk_off_drawdown: float = -0.15
    risk_off_shock: float = 0.90
    min_confidence: float = 0.0
    max_confidence: float = 1.0


def estimate_market(
    features: MarketFeaturesV3 | pd.Series | dict[str, Any],
    config: MarketEstimatorConfig | None = None,
) -> MarketEstimateV3:
    """Estimate v3 market state for one feature snapshot."""

    config = config or MarketEstimatorConfig()
    values = _feature_values(features)

    trend_strength = _clip(_trend_strength(values), -1.0, 1.0)
    volatility_state = _classify_volatility(values["volatility_ratio"], config)
    drawdown_state = _classify_drawdown(values["drawdown_long"], config)
    risk_state = _classify_risk(values, volatility_state, drawdown_state, config)
    long_regime = _classify_long_regime(values, trend_strength, volatility_state, drawdown_state, config)
    short_regime = _classify_short_regime(values, long_regime, config)
    confidence_score = _confidence_score(values, trend_strength, volatility_state, drawdown_state, risk_state, config)
    allow_entry, allow_hold = _entry_hold_permissions(long_regime, risk_state)

    notes = {
        "estimator": "rule_based_v3",
        "particle_filter": "deferred_to_v4",
        "price_vs_ma_long_term": _safe_divide_scalar(values["close"], values["ma_long_term"], 1.0) - 1.0,
        "ma_alignment": _ma_alignment(values),
        "risk_inputs": {
            "drawdown_long": values["drawdown_long"],
            "volatility_ratio": values["volatility_ratio"],
            "shock_score": values["shock_score"],
        },
    }

    return MarketEstimateV3(
        timestamp=values["timestamp"],
        long_regime=long_regime,
        short_regime=short_regime,
        trend_strength=trend_strength,
        volatility_state=volatility_state,
        drawdown_state=drawdown_state,
        risk_state=risk_state,
        confidence_score=confidence_score,
        allow_entry=allow_entry,
        allow_hold=allow_hold,
        notes=notes,
    )


def estimate_market_records(
    features: Iterable[MarketFeaturesV3],
    config: MarketEstimatorConfig | None = None,
) -> list[MarketEstimateV3]:
    """Estimate market state for an iterable of ``MarketFeaturesV3`` records."""

    return [estimate_market(feature, config=config) for feature in features]


def estimate_market_frame(
    features: pd.DataFrame,
    config: MarketEstimatorConfig | None = None,
) -> pd.DataFrame:
    """Estimate v3 market state for each row in a feature DataFrame."""

    estimates = [estimate_market(row, config=config) for _, row in features.iterrows()]
    return pd.DataFrame(
        [
            {
                "timestamp": estimate.timestamp,
                "long_regime": estimate.long_regime,
                "short_regime": estimate.short_regime,
                "trend_strength": estimate.trend_strength,
                "volatility_state": estimate.volatility_state,
                "drawdown_state": estimate.drawdown_state,
                "risk_state": estimate.risk_state,
                "confidence_score": estimate.confidence_score,
                "allow_entry": estimate.allow_entry,
                "allow_hold": estimate.allow_hold,
                "notes": estimate.notes,
            }
            for estimate in estimates
        ],
        columns=ESTIMATE_COLUMNS,
    )


def _classify_long_regime(
    values: dict[str, Any],
    trend_strength: float,
    volatility_state: str,
    drawdown_state: str,
    config: MarketEstimatorConfig,
) -> str:
    price_above_long_term = values["close"] > values["ma_long_term"]
    bullish_alignment = values["ma_short"] > values["ma_long"] > values["ma_long_term"]
    bearish_alignment = values["ma_short"] < values["ma_long"] < values["ma_long_term"]
    long_momentum = values["momentum_long"]
    not_extreme_vol = volatility_state != "extreme"

    if (
        price_above_long_term
        and bullish_alignment
        and trend_strength >= config.strong_bull_trend_threshold
        and long_momentum >= config.strong_bull_momentum_threshold
        and not_extreme_vol
    ):
        return "strong_bull"
    if (
        trend_strength >= config.bull_trend_threshold
        and long_momentum >= config.bull_momentum_threshold
        and price_above_long_term
    ):
        return "bull"
    if (
        bearish_alignment
        and (
            trend_strength <= config.strong_bear_trend_threshold
            or long_momentum <= config.strong_bear_momentum_threshold
            or drawdown_state == "severe"
        )
    ):
        return "strong_bear"
    if trend_strength <= config.bear_trend_threshold or long_momentum <= config.bear_momentum_threshold:
        return "bear"
    return "neutral"


def _classify_short_regime(
    values: dict[str, Any],
    long_regime: str,
    config: MarketEstimatorConfig,
) -> str:
    short_momentum = values["momentum_short"]
    short_drawdown = values["drawdown_short"]
    ma_stretch = _safe_divide_scalar(values["close"], values["ma_short"], 1.0) - 1.0
    negative_shock = values["return_1"] < 0.0 and values["shock_score"] >= config.breakdown_shock_threshold

    if short_momentum <= config.breakdown_momentum_threshold or negative_shock:
        return "breakdown"
    if (
        long_regime in {"strong_bull", "bull"}
        and (short_momentum <= config.pullback_momentum_threshold or short_drawdown <= config.pullback_drawdown_threshold)
    ):
        return "pullback"
    if short_drawdown <= config.recovery_drawdown_threshold and short_momentum >= config.recovery_momentum_threshold:
        return "recovery"
    if short_momentum >= config.overheat_momentum_threshold and ma_stretch >= config.overheat_ma_stretch:
        return "overheat"
    return "noise"


def _classify_volatility(volatility_ratio: float, config: MarketEstimatorConfig) -> str:
    if volatility_ratio >= config.extreme_volatility_ratio:
        return "extreme"
    if volatility_ratio >= config.high_volatility_ratio:
        return "high"
    if volatility_ratio <= config.low_volatility_ratio:
        return "low"
    return "normal"


def _classify_drawdown(drawdown_long: float, config: MarketEstimatorConfig) -> str:
    if drawdown_long <= config.severe_drawdown:
        return "severe"
    if drawdown_long <= config.danger_drawdown:
        return "danger"
    if drawdown_long <= config.caution_drawdown:
        return "caution"
    return "normal"


def _classify_risk(
    values: dict[str, Any],
    volatility_state: str,
    drawdown_state: str,
    config: MarketEstimatorConfig,
) -> str:
    if (
        values["drawdown_long"] <= config.risk_off_drawdown
        or values["shock_score"] >= config.risk_off_shock
        or drawdown_state == "severe"
    ):
        return "risk_off"
    if volatility_state in {"high", "extreme"} or drawdown_state in {"caution", "danger"}:
        return "caution"
    return "normal"


def _entry_hold_permissions(long_regime: str, risk_state: str) -> tuple[bool, bool]:
    if risk_state == "risk_off":
        return False, False
    if long_regime in {"strong_bull", "bull"}:
        return True, True
    if long_regime == "neutral":
        return False, True
    if long_regime == "bear":
        return False, True
    return False, False


def _confidence_score(
    values: dict[str, Any],
    trend_strength: float,
    volatility_state: str,
    drawdown_state: str,
    risk_state: str,
    config: MarketEstimatorConfig,
) -> float:
    trend_component = min(abs(trend_strength), 1.0)
    momentum_component = min(abs(values["momentum_long"]) / 0.10, 1.0)
    vol_penalty = {"low": 0.05, "normal": 0.0, "high": 0.20, "extreme": 0.45}[volatility_state]
    drawdown_penalty = {"normal": 0.0, "caution": 0.12, "danger": 0.28, "severe": 0.45}[drawdown_state]
    risk_penalty = {"normal": 0.0, "caution": 0.10, "risk_off": 0.35}[risk_state]
    shock_penalty = 0.20 * values["shock_score"]
    score = 0.35 + 0.35 * trend_component + 0.20 * momentum_component
    score -= vol_penalty + drawdown_penalty + risk_penalty + shock_penalty
    return _clip(score, config.min_confidence, config.max_confidence)


def _trend_strength(values: dict[str, Any]) -> float:
    ma_signal = _safe_divide_scalar(values["ma_short"] - values["ma_long_term"], values["ma_long_term"], 0.0)
    momentum_signal = values["momentum_long"]
    return float(np.tanh(6.0 * ma_signal + 4.0 * momentum_signal))


def _ma_alignment(values: dict[str, Any]) -> str:
    if values["ma_short"] > values["ma_long"] > values["ma_long_term"]:
        return "bullish"
    if values["ma_short"] < values["ma_long"] < values["ma_long_term"]:
        return "bearish"
    return "mixed"


def _feature_values(features: MarketFeaturesV3 | pd.Series | dict[str, Any]) -> dict[str, Any]:
    if isinstance(features, MarketFeaturesV3):
        values = asdict(features)
    elif isinstance(features, pd.Series):
        values = features.to_dict()
    else:
        values = dict(features)

    required = {
        "timestamp",
        "close",
        "ma_short",
        "ma_long",
        "ma_long_term",
        "momentum_short",
        "momentum_long",
        "volatility_ratio",
        "drawdown_short",
        "drawdown_long",
        "return_1",
    }
    missing = required.difference(values)
    if missing:
        raise ValueError(f"features are missing required fields: {sorted(missing)}")

    normalized = dict(values)
    for key in required.difference({"timestamp"}):
        normalized[key] = _finite_float(normalized[key], key)
    normalized["volatility_short"] = _finite_float(normalized.get("volatility_short", 0.0), "volatility_short")
    normalized["volatility_long"] = _finite_float(normalized.get("volatility_long", 0.0), "volatility_long")
    normalized["shock_score"] = _clip(_finite_float(normalized.get("shock_score", 0.0), "shock_score"), 0.0, 1.0)
    return normalized


def _finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _safe_divide_scalar(numerator: float, denominator: float, default: float) -> float:
    if abs(denominator) <= 1e-12:
        return default
    return numerator / denominator


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


__all__ = [
    "DRAW_DOWN_STATES",
    "ESTIMATE_COLUMNS",
    "LONG_REGIMES",
    "MarketEstimateV3",
    "MarketEstimatorConfig",
    "SHORT_REGIMES",
    "RISK_STATES",
    "VOLATILITY_STATES",
    "estimate_market",
    "estimate_market_frame",
    "estimate_market_records",
]

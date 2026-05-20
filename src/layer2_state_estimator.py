"""Layer 2: rule-based regime belief estimation."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log

import numpy as np
import pandas as pd

from .layer1_market_model import MarketStateV1


REGIMES = ("bull", "bear", "sideways", "high_vol", "crash_risk")


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass(frozen=True)
class StateEstimatorConfigV1:
    softmax_temperature: float = 0.7
    smoothing_alpha: float = 0.2


@dataclass(frozen=True)
class EstimatedMarketStateV1:
    p_bull: float
    p_bear: float
    p_sideways: float
    p_high_vol: float
    p_crash_risk: float
    dominant_regime: str
    state_confidence: float
    regime_uncertainty: float
    transition_risk: float
    danger_score: float

    def probabilities(self) -> dict[str, float]:
        return {
            "bull": self.p_bull,
            "bear": self.p_bear,
            "sideways": self.p_sideways,
            "high_vol": self.p_high_vol,
            "crash_risk": self.p_crash_risk,
        }


ESTIMATED_MARKET_STATE_COLUMNS = [
    "p_bull",
    "p_bear",
    "p_sideways",
    "p_high_vol",
    "p_crash_risk",
    "dominant_regime",
    "state_confidence",
    "regime_uncertainty",
    "transition_risk",
    "danger_score",
]


def estimate_market_state_frame(
    market_states: pd.DataFrame,
    config: StateEstimatorConfigV1 | None = None,
) -> pd.DataFrame:
    """Compute EstimatedMarketStateV1 columns for a MarketStateV1 DataFrame."""

    config = config or StateEstimatorConfigV1()
    required = {
        "trend_score",
        "volatility_score",
        "volume_score",
        "liquidity_score",
        "drawdown",
        "shock_score",
        "confidence",
    }
    missing = required.difference(market_states.columns)
    if missing:
        raise ValueError(f"market_states is missing required columns: {sorted(missing)}")

    scores = _regime_score_frame(market_states)
    raw_probabilities = _softmax_frame(scores, config.softmax_temperature)
    smoothed = _smooth_probabilities(raw_probabilities, config.smoothing_alpha)
    smoothed = smoothed.div(smoothed.sum(axis=1), axis=0)

    dominant_regime = smoothed.idxmax(axis=1).str.removeprefix("p_")
    regime_clarity = smoothed.max(axis=1)
    confidence = pd.to_numeric(market_states["confidence"], errors="coerce").fillna(0.0)
    state_confidence = (confidence * regime_clarity).clip(0.0, 1.0)

    entropy_terms = smoothed.where(smoothed > 0.0, 1.0)
    entropy = -(smoothed * np.log(entropy_terms)).sum(axis=1)
    regime_uncertainty = (entropy / np.log(len(REGIMES))).clip(0.0, 1.0)

    previous_probs = smoothed.shift(1).fillna(smoothed)
    probability_shift = 0.5 * (smoothed - previous_probs).abs().sum(axis=1)
    shock_score = pd.to_numeric(market_states["shock_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    transition_risk = (0.7 * probability_shift + 0.3 * shock_score).clip(0.0, 1.0)

    liquidity_score = pd.to_numeric(market_states["liquidity_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    drawdown = pd.to_numeric(market_states["drawdown"], errors="coerce").fillna(0.0)
    drawdown_severity = (drawdown.abs() / 0.25).clip(0.0, 1.0)
    danger_score = (
        0.35 * smoothed["p_crash_risk"]
        + 0.25 * smoothed["p_high_vol"]
        + 0.20 * shock_score
        + 0.10 * (1.0 - liquidity_score)
        + 0.10 * drawdown_severity
    ).clip(0.0, 1.0)

    result = pd.DataFrame(
        {
            "p_bull": smoothed["p_bull"],
            "p_bear": smoothed["p_bear"],
            "p_sideways": smoothed["p_sideways"],
            "p_high_vol": smoothed["p_high_vol"],
            "p_crash_risk": smoothed["p_crash_risk"],
            "dominant_regime": dominant_regime,
            "state_confidence": state_confidence,
            "regime_uncertainty": regime_uncertainty,
            "transition_risk": transition_risk,
            "danger_score": danger_score,
        },
        index=market_states.index,
    )
    return _clean_estimated_market_state_frame(result)


def estimate_market_state(
    market: MarketStateV1,
    previous: EstimatedMarketStateV1 | None = None,
    config: StateEstimatorConfigV1 | None = None,
) -> EstimatedMarketStateV1:
    config = config or StateEstimatorConfigV1()
    scores = _regime_scores(market)
    raw = _softmax(scores, config.softmax_temperature)

    if previous is None:
        smoothed = raw
        previous_probs = raw
    else:
        previous_probs = previous.probabilities()
        alpha = _clip(config.smoothing_alpha, 0.0, 1.0)
        smoothed = {name: alpha * raw[name] + (1.0 - alpha) * previous_probs[name] for name in REGIMES}

    dominant_regime = max(smoothed, key=smoothed.get)
    regime_clarity = smoothed[dominant_regime]
    state_confidence = _clip(market.confidence * regime_clarity, 0.0, 1.0)
    entropy = -sum(p * log(p) for p in smoothed.values() if p > 0)
    regime_uncertainty = _clip(entropy / log(len(REGIMES)), 0.0, 1.0)
    probability_shift = 0.5 * sum(abs(smoothed[name] - previous_probs[name]) for name in REGIMES)
    transition_risk = _clip(0.7 * probability_shift + 0.3 * market.shock_score, 0.0, 1.0)
    drawdown_severity = _clip(abs(market.drawdown) / 0.25, 0.0, 1.0)
    danger_score = _clip(
        0.35 * smoothed["crash_risk"]
        + 0.25 * smoothed["high_vol"]
        + 0.20 * market.shock_score
        + 0.10 * (1.0 - market.liquidity_score)
        + 0.10 * drawdown_severity,
        0.0,
        1.0,
    )

    return EstimatedMarketStateV1(
        p_bull=smoothed["bull"],
        p_bear=smoothed["bear"],
        p_sideways=smoothed["sideways"],
        p_high_vol=smoothed["high_vol"],
        p_crash_risk=smoothed["crash_risk"],
        dominant_regime=dominant_regime,
        state_confidence=state_confidence,
        regime_uncertainty=regime_uncertainty,
        transition_risk=transition_risk,
        danger_score=danger_score,
    )


def _regime_scores(market: MarketStateV1) -> dict[str, float]:
    up_trend = max(market.trend_score, 0.0)
    down_trend = max(-market.trend_score, 0.0)
    flatness = 1.0 - abs(market.trend_score)
    high_vol = _clip((market.volatility_score - 1.0) / 1.5, 0.0, 1.0)
    extreme_vol = _clip((market.volatility_score - 2.0) / 1.5, 0.0, 1.0)
    illiquidity = 1.0 - market.liquidity_score
    drawdown_sev = _clip(abs(market.drawdown) / 0.25, 0.0, 1.0)
    shock = market.shock_score

    return {
        "bull": 1.8 * up_trend + 0.4 * market.confidence + 0.2 * market.liquidity_score - 0.5 * high_vol - 0.7 * shock - 0.5 * drawdown_sev,
        "bear": 1.8 * down_trend + 0.4 * drawdown_sev + 0.3 * high_vol - 0.4 * shock - 0.2 * illiquidity,
        "sideways": 0.8 * flatness + 0.3 * market.liquidity_score + 0.2 * market.confidence - 0.6 * high_vol - 0.8 * shock,
        "high_vol": 1.2 * high_vol + 0.4 * market.volume_score + 0.3 * abs(market.trend_score) - 0.4 * shock,
        "crash_risk": 1.2 * shock + 1.0 * extreme_vol + 0.8 * illiquidity + 0.8 * drawdown_sev + 0.4 * market.volume_score - 0.3 * market.confidence,
    }


def _softmax(scores: dict[str, float], temperature: float) -> dict[str, float]:
    temperature = max(temperature, 1e-9)
    max_score = max(scores.values())
    weights = {name: exp((score - max_score) / temperature) for name, score in scores.items()}
    total = sum(weights.values())
    return {name: weight / total for name, weight in weights.items()}


def _regime_score_frame(market_states: pd.DataFrame) -> pd.DataFrame:
    trend_score = pd.to_numeric(market_states["trend_score"], errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    volatility_score = pd.to_numeric(market_states["volatility_score"], errors="coerce").fillna(1.0).clip(lower=0.0)
    volume_score = pd.to_numeric(market_states["volume_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    liquidity_score = pd.to_numeric(market_states["liquidity_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    drawdown = pd.to_numeric(market_states["drawdown"], errors="coerce").fillna(0.0)
    shock_score = pd.to_numeric(market_states["shock_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    confidence = pd.to_numeric(market_states["confidence"], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    up_trend = trend_score.clip(lower=0.0)
    down_trend = (-trend_score).clip(lower=0.0)
    flatness = (1.0 - trend_score.abs()).clip(0.0, 1.0)
    high_vol = ((volatility_score - 1.0) / 1.5).clip(0.0, 1.0)
    extreme_vol = ((volatility_score - 2.0) / 1.5).clip(0.0, 1.0)
    illiquidity = (1.0 - liquidity_score).clip(0.0, 1.0)
    drawdown_sev = (drawdown.abs() / 0.25).clip(0.0, 1.0)

    return pd.DataFrame(
        {
            "bull": 1.8 * up_trend + 0.4 * confidence + 0.2 * liquidity_score - 0.5 * high_vol - 0.7 * shock_score - 0.5 * drawdown_sev,
            "bear": 1.8 * down_trend + 0.4 * drawdown_sev + 0.3 * high_vol - 0.4 * shock_score - 0.2 * illiquidity,
            "sideways": 0.8 * flatness + 0.3 * liquidity_score + 0.2 * confidence - 0.6 * high_vol - 0.8 * shock_score,
            "high_vol": 1.2 * high_vol + 0.4 * volume_score + 0.3 * trend_score.abs() - 0.4 * shock_score,
            "crash_risk": 1.2 * shock_score + 1.0 * extreme_vol + 0.8 * illiquidity + 0.8 * drawdown_sev + 0.4 * volume_score - 0.3 * confidence,
        },
        index=market_states.index,
    )


def _softmax_frame(scores: pd.DataFrame, temperature: float) -> pd.DataFrame:
    temperature = max(float(temperature), 1e-9)
    centered = scores.sub(scores.max(axis=1), axis=0)
    weights = np.exp(centered / temperature)
    probabilities = weights.div(weights.sum(axis=1), axis=0)
    probabilities.columns = [f"p_{column}" for column in probabilities.columns]
    return probabilities


def _smooth_probabilities(probabilities: pd.DataFrame, alpha: float) -> pd.DataFrame:
    alpha = _clip(float(alpha), 0.0, 1.0)
    if probabilities.empty:
        return probabilities.copy()

    rows = []
    previous = probabilities.iloc[0]
    for idx, current in probabilities.iterrows():
        if not rows:
            smoothed = current
        else:
            smoothed = alpha * current + (1.0 - alpha) * previous
        rows.append(smoothed)
        previous = smoothed
    return pd.DataFrame(rows, index=probabilities.index, columns=probabilities.columns)


def _clean_estimated_market_state_frame(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [column for column in frame.columns if column != "dominant_regime"]
    frame[numeric_columns] = frame[numeric_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    probability_columns = [f"p_{regime}" for regime in REGIMES]
    frame[probability_columns] = frame[probability_columns].clip(0.0, 1.0)
    probability_sum = frame[probability_columns].sum(axis=1).replace(0.0, 1.0)
    frame[probability_columns] = frame[probability_columns].div(probability_sum, axis=0)
    frame["state_confidence"] = frame["state_confidence"].clip(0.0, 1.0)
    frame["regime_uncertainty"] = frame["regime_uncertainty"].clip(0.0, 1.0)
    frame["transition_risk"] = frame["transition_risk"].clip(0.0, 1.0)
    frame["danger_score"] = frame["danger_score"].clip(0.0, 1.0)
    frame["dominant_regime"] = frame["dominant_regime"].where(frame["dominant_regime"].isin(REGIMES), "sideways")
    return frame[ESTIMATED_MARKET_STATE_COLUMNS]

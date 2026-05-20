"""Layer 1: rule-based market modeling from OHLCV bars."""

from __future__ import annotations

from dataclasses import dataclass
from math import log, tanh
from statistics import mean, median, pstdev
from typing import Sequence

import numpy as np
import pandas as pd


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _safe_mean(values: Sequence[float], default: float = 0.0) -> float:
    return mean(values) if values else default


def _safe_std(values: Sequence[float], default: float = 0.0) -> float:
    return pstdev(values) if len(values) > 1 else default


def _safe_median(values: Sequence[float], default: float = 1.0) -> float:
    return median(values) if values else default


@dataclass(frozen=True)
class OHLCVBar:
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class MarketStateV1:
    timestamp: float
    close: float
    return_1: float
    volatility: float
    volatility_score: float
    trend_raw: float
    trend_score: float
    volume_z: float
    volume_score: float
    price_range: float
    liquidity_score: float
    drawdown: float
    shock_score: float
    confidence: float
    market_mode: str


MARKET_STATE_COLUMNS = [
    "timestamp",
    "close",
    "return_1",
    "volatility",
    "volatility_score",
    "trend_raw",
    "trend_score",
    "volume_z",
    "volume_score",
    "price_range",
    "liquidity_score",
    "drawdown",
    "shock_score",
    "confidence",
    "market_mode",
]


def build_market_state_frame(ohlcv: pd.DataFrame, trend_k: float = 10.0) -> pd.DataFrame:
    """Compute MarketStateV1 columns for an OHLCV DataFrame.

    Required input columns: open, high, low, close, volume.
    """

    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(ohlcv.columns)
    if missing:
        raise ValueError(f"ohlcv is missing required columns: {sorted(missing)}")

    data = ohlcv.copy()
    close = pd.to_numeric(data["close"], errors="coerce")
    high = pd.to_numeric(data["high"], errors="coerce")
    low = pd.to_numeric(data["low"], errors="coerce")
    volume = pd.to_numeric(data["volume"], errors="coerce").clip(lower=0.0)
    timestamp = data["timestamp"] if "timestamp" in data.columns else pd.Series(data.index, index=data.index)

    safe_close = close.where(close > 0)
    return_1 = np.log(safe_close / safe_close.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    volatility = return_1.rolling(20, min_periods=2).std(ddof=0).fillna(0.0)
    volatility_baseline = volatility.rolling(120, min_periods=1).median()
    volatility_score = _safe_divide_series(volatility, volatility_baseline, default=1.0)

    ma_short = close.rolling(20, min_periods=1).mean()
    ma_long = close.rolling(60, min_periods=1).mean()
    trend_raw = _safe_divide_series(ma_short - ma_long, ma_long, default=0.0)
    trend_score = np.tanh(trend_k * trend_raw)

    volume_mean = volume.rolling(60, min_periods=1).mean()
    volume_std = volume.rolling(60, min_periods=2).std(ddof=0)
    volume_z = _safe_divide_series(volume - volume_mean, volume_std, default=0.0)
    volume_score = (volume_z.abs() / 3.0).clip(0.0, 1.0)

    price_range = _safe_divide_series(high - low, close, default=0.0).clip(lower=0.0)
    normalized_volume = _safe_divide_series(volume, volume_mean, default=1.0)
    illiquidity = _safe_divide_series(price_range, normalized_volume, default=1.0).clip(lower=0.0)
    illiquidity_baseline = illiquidity.rolling(120, min_periods=1).median()
    normalized_illiquidity = _safe_divide_series(illiquidity, 3.0 * illiquidity_baseline, default=1.0)
    liquidity_score = (1.0 - normalized_illiquidity.clip(0.0, 1.0)).clip(0.0, 1.0)

    rolling_high = close.rolling(120, min_periods=1).max()
    drawdown = _safe_divide_series(close, rolling_high, default=1.0) - 1.0
    drawdown = drawdown.clip(upper=0.0)

    return_std = return_1.rolling(60, min_periods=2).std(ddof=0)
    return_z = _safe_divide_series(return_1.abs(), return_std, default=0.0)
    shock_score = (return_z / 4.0).clip(0.0, 1.0)

    valid_bar = close.gt(0) & high.ge(low) & volume.ge(0)
    missing_data_penalty = (~valid_bar).astype(float)
    volatility_penalty = ((volatility_score - 1.5) / 2.0).clip(0.0, 1.0)
    illiquidity_penalty = 1.0 - liquidity_score
    confidence = (
        1.0
        - 0.3 * shock_score
        - 0.3 * volatility_penalty
        - 0.2 * illiquidity_penalty
        - 0.2 * missing_data_penalty
    ).clip(0.0, 1.0)

    market_mode = pd.Series("normal", index=data.index, dtype="object")
    market_mode = market_mode.mask(trend_score > 0.4, "trending_up")
    market_mode = market_mode.mask(trend_score < -0.4, "trending_down")
    market_mode = market_mode.mask(volatility_score > 1.8, "high_volatility")
    market_mode = market_mode.mask((volatility_score > 2.5) & (liquidity_score < 0.4), "stressed")
    market_mode = market_mode.mask(shock_score > 0.8, "shock")

    result = pd.DataFrame(
        {
            "timestamp": timestamp,
            "close": close,
            "return_1": return_1,
            "volatility": volatility,
            "volatility_score": volatility_score,
            "trend_raw": trend_raw,
            "trend_score": trend_score,
            "volume_z": volume_z,
            "volume_score": volume_score,
            "price_range": price_range,
            "liquidity_score": liquidity_score,
            "drawdown": drawdown,
            "shock_score": shock_score,
            "confidence": confidence,
            "market_mode": market_mode,
        },
        index=data.index,
    )
    return _clean_market_state_frame(result)


def build_market_state(history: Sequence[OHLCVBar], trend_k: float = 10.0) -> MarketStateV1:
    """Build the latest MarketStateV1 from OHLCV history."""

    if not history:
        raise ValueError("history must contain at least one OHLCVBar")

    bar = history[-1]
    closes = [b.close for b in history if b.close > 0]
    volumes = [max(b.volume, 0.0) for b in history]
    missing_data_penalty = 0.0 if _is_valid_bar(bar) else 1.0

    previous_close = history[-2].close if len(history) > 1 and history[-2].close > 0 else bar.close
    return_1 = log(bar.close / previous_close) if bar.close > 0 and previous_close > 0 else 0.0

    returns = _log_returns(history)
    recent_returns_20 = returns[-20:]
    recent_returns_60 = returns[-60:]
    volatility = _safe_std(recent_returns_20)
    vol_history = [_safe_std(returns[max(0, i - 19) : i + 1]) for i in range(len(returns))]
    vol_baseline = _safe_median(vol_history[-120:], default=volatility or 1.0)
    volatility_score = volatility / vol_baseline if vol_baseline > 0 else 1.0

    ma_short = _safe_mean(closes[-20:], default=bar.close)
    ma_long = _safe_mean(closes[-60:], default=bar.close)
    trend_raw = (ma_short - ma_long) / ma_long if ma_long else 0.0
    trend_score = tanh(trend_k * trend_raw)

    volume_mean = _safe_mean(volumes[-60:], default=bar.volume)
    volume_std = _safe_std(volumes[-60:], default=0.0)
    volume_z = (bar.volume - volume_mean) / volume_std if volume_std > 0 else 0.0
    volume_score = _clip(abs(volume_z) / 3.0, 0.0, 1.0)

    price_range = (bar.high - bar.low) / bar.close if bar.close > 0 else 0.0
    normalized_volume = bar.volume / volume_mean if volume_mean > 0 else 1.0
    illiquidity = price_range / normalized_volume if normalized_volume > 0 else 1.0
    illiquidity_history = _illiquidity_history(history)
    illiquidity_baseline = _safe_median(illiquidity_history[-120:], default=illiquidity or 1.0)
    normalized_illiquidity = illiquidity / (3.0 * illiquidity_baseline) if illiquidity_baseline > 0 else 1.0
    liquidity_score = 1.0 - _clip(normalized_illiquidity, 0.0, 1.0)

    rolling_high = max(closes[-120:]) if closes else bar.close
    drawdown = bar.close / rolling_high - 1.0 if rolling_high > 0 else 0.0

    return_std = _safe_std(recent_returns_60, default=0.0)
    return_z = abs(return_1) / return_std if return_std > 0 else 0.0
    shock_score = _clip(return_z / 4.0, 0.0, 1.0)

    volatility_penalty = _clip((volatility_score - 1.5) / 2.0, 0.0, 1.0)
    illiquidity_penalty = 1.0 - liquidity_score
    confidence = 1.0 - 0.3 * shock_score - 0.3 * volatility_penalty - 0.2 * illiquidity_penalty - 0.2 * missing_data_penalty
    confidence = _clip(confidence, 0.0, 1.0)
    market_mode = classify_market_mode(volatility_score, trend_score, liquidity_score, shock_score)

    return MarketStateV1(
        timestamp=bar.timestamp,
        close=bar.close,
        return_1=return_1,
        volatility=volatility,
        volatility_score=volatility_score,
        trend_raw=trend_raw,
        trend_score=trend_score,
        volume_z=volume_z,
        volume_score=volume_score,
        price_range=price_range,
        liquidity_score=liquidity_score,
        drawdown=drawdown,
        shock_score=shock_score,
        confidence=confidence,
        market_mode=market_mode,
    )


def classify_market_mode(
    volatility_score: float,
    trend_score: float,
    liquidity_score: float,
    shock_score: float,
) -> str:
    if shock_score > 0.8:
        return "shock"
    if volatility_score > 2.5 and liquidity_score < 0.4:
        return "stressed"
    if volatility_score > 1.8:
        return "high_volatility"
    if trend_score > 0.4:
        return "trending_up"
    if trend_score < -0.4:
        return "trending_down"
    return "normal"


def _is_valid_bar(bar: OHLCVBar) -> bool:
    return bar.close > 0 and bar.high >= bar.low and bar.volume >= 0


def _log_returns(history: Sequence[OHLCVBar]) -> list[float]:
    returns: list[float] = []
    for previous, current in zip(history, history[1:]):
        if previous.close > 0 and current.close > 0:
            returns.append(log(current.close / previous.close))
    return returns


def _illiquidity_history(history: Sequence[OHLCVBar]) -> list[float]:
    values: list[float] = []
    for idx, bar in enumerate(history):
        window = history[max(0, idx - 59) : idx + 1]
        avg_volume = _safe_mean([max(b.volume, 0.0) for b in window], default=bar.volume)
        normalized_volume = bar.volume / avg_volume if avg_volume > 0 else 1.0
        price_range = (bar.high - bar.low) / bar.close if bar.close > 0 else 0.0
        values.append(price_range / normalized_volume if normalized_volume > 0 else 1.0)
    return values


def _safe_divide_series(numerator: pd.Series, denominator: pd.Series | float, default: float) -> pd.Series:
    result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan).fillna(default)


def _clean_market_state_frame(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [column for column in frame.columns if column != "market_mode"]
    frame[numeric_columns] = frame[numeric_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    frame["volatility_score"] = frame["volatility_score"].clip(lower=0.0)
    frame["trend_score"] = frame["trend_score"].clip(-1.0, 1.0)
    frame["volume_score"] = frame["volume_score"].clip(0.0, 1.0)
    frame["liquidity_score"] = frame["liquidity_score"].clip(0.0, 1.0)
    frame["drawdown"] = frame["drawdown"].clip(upper=0.0)
    frame["shock_score"] = frame["shock_score"].clip(0.0, 1.0)
    frame["confidence"] = frame["confidence"].clip(0.0, 1.0)
    return frame[MARKET_STATE_COLUMNS]

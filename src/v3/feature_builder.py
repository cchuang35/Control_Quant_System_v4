"""Build v3 market features from OHLCV data.

The feature builder is intentionally isolated from the existing v1/v2 pipeline.
All rolling calculations use pandas' default trailing windows, so each row uses
only the current and historical bars available at that timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .data_types import MarketFeaturesV3


REQUIRED_OHLCV_COLUMNS = ("open", "high", "low", "close")
FEATURE_COLUMNS = [
    "timestamp",
    "close",
    "return_1",
    "ma_short",
    "ma_long",
    "ma_long_term",
    "momentum_short",
    "momentum_long",
    "volatility_short",
    "volatility_long",
    "volatility_ratio",
    "drawdown_short",
    "drawdown_long",
    "shock_score",
]


@dataclass(frozen=True)
class FeatureWindowConfig:
    """Rolling-window configuration for v3 feature construction.

    Defaults target 1h crypto data, especially BTCUSDT 1h research, while
    remaining explicit and easy to replace for other markets or bar sizes.
    """

    ma_short: int = 24
    ma_long: int = 168
    ma_long_term: int = 720
    momentum_short: int = 24
    momentum_long: int = 168
    volatility_short: int = 24
    volatility_long: int = 168
    drawdown_short: int = 168
    drawdown_long: int = 720
    shock_z: float = 4.0


def build_feature_frame(
    ohlcv: pd.DataFrame,
    config: FeatureWindowConfig | None = None,
) -> pd.DataFrame:
    """Return a v3 feature DataFrame from OHLCV input.

    Input must include ``open``, ``high``, ``low``, and ``close`` columns.
    ``volume`` is accepted but not required by the initial v3 feature set.
    Timestamps are taken from a ``timestamp`` column when present; otherwise
    the DataFrame index is used, which supports datetime-indexed input.
    """

    config = config or FeatureWindowConfig()
    _validate_config(config)
    _require_ohlcv_columns(ohlcv)

    data = ohlcv.copy()
    timestamp = data["timestamp"] if "timestamp" in data.columns else pd.Series(data.index, index=data.index)
    close = pd.to_numeric(data["close"], errors="coerce").astype(float)
    if close.isna().any():
        raise ValueError("close must be numeric and non-null")
    if (close <= 0.0).any():
        raise ValueError("close must be positive")

    returns = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    volatility_short = returns.rolling(config.volatility_short, min_periods=2).std(ddof=0).fillna(0.0)
    volatility_long = returns.rolling(config.volatility_long, min_periods=2).std(ddof=0).fillna(0.0)

    result = pd.DataFrame(index=data.index)
    result["timestamp"] = timestamp
    result["close"] = close
    result["return_1"] = returns
    result["ma_short"] = close.rolling(config.ma_short, min_periods=1).mean()
    result["ma_long"] = close.rolling(config.ma_long, min_periods=1).mean()
    result["ma_long_term"] = close.rolling(config.ma_long_term, min_periods=1).mean()
    result["momentum_short"] = _momentum(close, config.momentum_short)
    result["momentum_long"] = _momentum(close, config.momentum_long)
    result["volatility_short"] = volatility_short
    result["volatility_long"] = volatility_long
    result["volatility_ratio"] = _safe_divide(volatility_short, volatility_long, default=1.0)
    result["drawdown_short"] = _drawdown(close, config.drawdown_short)
    result["drawdown_long"] = _drawdown(close, config.drawdown_long)
    result["shock_score"] = _shock_score(returns, volatility_long, config.shock_z)
    return result.loc[:, FEATURE_COLUMNS]


def build_feature_records(
    ohlcv: pd.DataFrame,
    config: FeatureWindowConfig | None = None,
) -> list[MarketFeaturesV3]:
    """Return v3 feature dataclass records from OHLCV input."""

    frame = build_feature_frame(ohlcv, config=config)
    return [
        MarketFeaturesV3(
            timestamp=row.timestamp,
            close=float(row.close),
            return_1=float(row.return_1),
            ma_short=float(row.ma_short),
            ma_long=float(row.ma_long),
            ma_long_term=float(row.ma_long_term),
            momentum_short=float(row.momentum_short),
            momentum_long=float(row.momentum_long),
            volatility_short=float(row.volatility_short),
            volatility_long=float(row.volatility_long),
            volatility_ratio=float(row.volatility_ratio),
            drawdown_short=float(row.drawdown_short),
            drawdown_long=float(row.drawdown_long),
            shock_score=float(row.shock_score),
        )
        for row in frame.itertuples(index=False)
    ]


def validate_feature_frame(frame: pd.DataFrame) -> None:
    """Validate that a feature frame has the required v3 feature columns."""

    missing = set(FEATURE_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"feature frame is missing columns: {sorted(missing)}")
    numeric_columns = [column for column in FEATURE_COLUMNS if column != "timestamp"]
    numeric = frame.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("feature frame contains NaN values")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("feature frame contains infinite values")
    if not frame["shock_score"].between(0.0, 1.0).all():
        raise ValueError("shock_score must be in [0, 1]")
    if not frame["drawdown_short"].le(0.0).all() or not frame["drawdown_long"].le(0.0).all():
        raise ValueError("drawdown features must be non-positive")


def _momentum(close: pd.Series, window: int) -> pd.Series:
    momentum = close / close.shift(window) - 1.0
    return momentum.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _drawdown(close: pd.Series, window: int) -> pd.Series:
    rolling_high = close.rolling(window, min_periods=1).max()
    return (close / rolling_high - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(upper=0.0)


def _shock_score(returns: pd.Series, volatility_long: pd.Series, shock_z: float) -> pd.Series:
    denominator = shock_z * volatility_long
    return _safe_divide(returns.abs(), denominator, default=0.0).clip(0.0, 1.0)


def _safe_divide(numerator: pd.Series, denominator: pd.Series | float, default: float) -> pd.Series:
    denominator_series = denominator if isinstance(denominator, pd.Series) else pd.Series(denominator, index=numerator.index)
    result = numerator / denominator_series.where(denominator_series.abs() > 1e-12)
    return result.replace([np.inf, -np.inf], np.nan).fillna(default).astype(float)


def _require_ohlcv_columns(ohlcv: pd.DataFrame) -> None:
    missing = set(REQUIRED_OHLCV_COLUMNS).difference(ohlcv.columns)
    if missing:
        raise ValueError(f"ohlcv is missing required columns: {sorted(missing)}")


def _validate_config(config: FeatureWindowConfig) -> None:
    values: dict[str, Any] = {
        "ma_short": config.ma_short,
        "ma_long": config.ma_long,
        "ma_long_term": config.ma_long_term,
        "momentum_short": config.momentum_short,
        "momentum_long": config.momentum_long,
        "volatility_short": config.volatility_short,
        "volatility_long": config.volatility_long,
        "drawdown_short": config.drawdown_short,
        "drawdown_long": config.drawdown_long,
    }
    invalid = [name for name, value in values.items() if int(value) <= 0]
    if invalid:
        raise ValueError(f"feature windows must be positive: {invalid}")
    if config.shock_z <= 0.0:
        raise ValueError("shock_z must be positive")


__all__ = [
    "FEATURE_COLUMNS",
    "FeatureWindowConfig",
    "MarketFeaturesV3",
    "build_feature_frame",
    "build_feature_records",
    "validate_feature_frame",
]

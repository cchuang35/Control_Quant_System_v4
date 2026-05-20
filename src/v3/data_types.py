"""Shared v3 data contracts.

These dataclasses define the boundaries between the planned v3 feature,
estimation, controller, risk, composition, and execution layers. They are not
wired into the current v1/v2 backtests yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MarketFeaturesV3:
    """Feature snapshot built from OHLCV data for one decision timestamp.

    The fields are designed to support a long-term primary controller and a
    short-term auxiliary controller while keeping feature calculation separate
    from strategy decisions.
    """

    timestamp: Any
    close: float
    return_1: float
    ma_short: float
    ma_long: float
    ma_long_term: float
    momentum_short: float
    momentum_long: float
    volatility_short: float
    volatility_long: float
    volatility_ratio: float
    drawdown_short: float
    drawdown_long: float
    shock_score: float | None = None


@dataclass(frozen=True)
class MarketEstimateV3:
    """Regime and risk estimate consumed by v3 controllers.

    Market estimation should summarize features into decision-ready state. v3
    keeps this estimator simple; particle-filter estimation is deferred to v4.
    """

    timestamp: Any
    long_regime: str
    short_regime: str
    trend_strength: float
    volatility_state: str
    drawdown_state: str
    risk_state: str
    confidence_score: float
    allow_entry: bool
    allow_hold: bool
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LongTermDecisionV3:
    """Primary strategic exposure decision.

    The long-term controller owns the base exposure for v3. Short-term logic
    may adjust this value, but should not become the primary strategy path.
    """

    timestamp: Any
    base_position: float
    reason: str
    long_regime: str
    confidence_score: float


@dataclass(frozen=True)
class ShortTermDecisionV3:
    """Auxiliary tactical adjustment to the long-term base position.

    This decision can add, reduce, or delay exposure in small increments based
    on short-horizon context and cooldown state.
    """

    timestamp: Any
    position_adjustment: float
    reason: str
    short_regime: str
    cooldown_active: bool


@dataclass(frozen=True)
class RiskDecisionV3:
    """Highest-authority risk constraint for v3 exposure.

    The risk supervisor can cap, reduce, or block exposure regardless of the
    long-term and short-term controller outputs.
    """

    timestamp: Any
    risk_cap: float
    risk_action: str
    reason: str
    portfolio_drawdown: float
    realized_volatility: float


@dataclass(frozen=True)
class FinalPositionDecisionV3:
    """Composed and execution-aware final v3 position decision.

    This object records the path from base position and auxiliary adjustment to
    risk-limited target, executed position, trade amount, and execution reason.
    """

    timestamp: Any
    base_position: float
    position_adjustment: float
    raw_target_position: float
    risk_cap: float
    target_position: float
    executed_position: float
    trade_amount: float
    execution_reason: str

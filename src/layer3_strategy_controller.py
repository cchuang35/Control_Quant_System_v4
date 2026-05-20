"""Layer 3: rule-based risk-aware exposure controller."""

from __future__ import annotations

from dataclasses import dataclass

from .layer1_market_model import MarketStateV1
from .layer2_state_estimator import EstimatedMarketStateV1


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass(frozen=True)
class PortfolioStateV1:
    current_exposure: float
    current_position: float
    equity: float
    cash: float
    unrealized_pnl: float
    portfolio_drawdown: float
    leverage: float
    available_margin: float = 0.0


@dataclass(frozen=True)
class ControlActionV1:
    target_exposure: float
    exposure_change: float
    max_leverage: float
    rebalance_speed: float
    trade_allowed: bool
    reduce_only: bool
    action_type: str
    reason_code: str
    raw_target_exposure: float = 0.0
    smoothed_target_exposure: float = 0.0


def compute_control_action(
    market: MarketStateV1,
    estimated: EstimatedMarketStateV1,
    portfolio: PortfolioStateV1,
    long_only: bool = False,
    base_exposure_multiplier: float = 2.0,
    previous_target_exposure: float | None = None,
    use_target_smoothing: bool = True,
    target_deadband: float = 0.05,
    beta_increase_risk: float = 0.30,
    beta_decrease_risk: float = 0.60,
) -> ControlActionV1:
    directional_signal = 0.6 * (estimated.p_bull - estimated.p_bear) + 0.4 * market.trend_score
    directional_signal = _clip(directional_signal, -1.0, 1.0)

    risk_scaler = (
        0.35 * (1.0 - estimated.danger_score)
        + 0.20 * (1.0 - 0.5 * estimated.p_high_vol)
        + 0.20 * (1.0 - 0.8 * estimated.p_crash_risk)
        + 0.15 * estimated.state_confidence
        + 0.10 * (1.0 - 0.6 * estimated.transition_risk)
    )
    risk_scaler = _clip(risk_scaler, 0.0, 1.0)
    sideways_scaler = 1.0 - 0.3 * estimated.p_sideways
    raw_target_exposure = base_exposure_multiplier * directional_signal * risk_scaler * sideways_scaler

    if not use_target_smoothing or previous_target_exposure is None:
        target_exposure = raw_target_exposure
    else:
        diff = raw_target_exposure - previous_target_exposure
        if abs(diff) < target_deadband:
            target_exposure = previous_target_exposure
        else:
            if abs(raw_target_exposure) > abs(previous_target_exposure):
                beta = _clip(beta_increase_risk, 0.0, 1.0)
            else:
                beta = _clip(beta_decrease_risk, 0.0, 1.0)
            target_exposure = beta * raw_target_exposure + (1.0 - beta) * previous_target_exposure

    target_exposure = _clip(target_exposure, 0.0, 1.0) if long_only else _clip(target_exposure, -1.0, 1.0)

    trade_allowed = False
    reduce_only = False
    reason_code = "risk_block"
    if estimated.danger_score < 0.5 and market.shock_score < 0.6 and estimated.state_confidence > 0.4:
        trade_allowed = True
        reason_code = "normal_trade"
    if estimated.danger_score >= 0.75 or market.shock_score >= 0.8 or estimated.p_crash_risk >= 0.5:
        trade_allowed = True
        reduce_only = True
        target_exposure = _reduce_only_target(portfolio.current_exposure, target_exposure)
        reason_code = "reduce_only"

    rebalance_speed = 0.8 * estimated.state_confidence * (1.0 - estimated.transition_risk) * market.liquidity_score
    rebalance_speed = _clip(rebalance_speed, 0.05, 0.8)
    if market.shock_score > 0.8:
        rebalance_speed = min(rebalance_speed, 0.2)

    exposure_change = rebalance_speed * (target_exposure - portfolio.current_exposure)
    max_leverage = risk_scaler
    if estimated.danger_score > 0.75:
        max_leverage = min(max_leverage, 0.3)
    if estimated.p_crash_risk > 0.5:
        max_leverage = 0.0

    action_type = "hold"
    if trade_allowed and abs(exposure_change) > 1e-12:
        action_type = "reduce" if reduce_only else "rebalance"

    return ControlActionV1(
        target_exposure=target_exposure,
        exposure_change=exposure_change,
        max_leverage=max_leverage,
        rebalance_speed=rebalance_speed,
        trade_allowed=trade_allowed,
        reduce_only=reduce_only,
        action_type=action_type,
        reason_code=reason_code,
        raw_target_exposure=raw_target_exposure,
        smoothed_target_exposure=target_exposure,
    )


def _reduce_only_target(current_exposure: float, target_exposure: float) -> float:
    if current_exposure > 0:
        return _clip(target_exposure, 0.0, abs(current_exposure))
    if current_exposure < 0:
        return _clip(target_exposure, -abs(current_exposure), 0.0)
    return 0.0

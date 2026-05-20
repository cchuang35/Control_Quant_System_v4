"""Layer 4: safety filter for desired control actions."""

from __future__ import annotations

from dataclasses import dataclass

from .layer1_market_model import MarketStateV1
from .layer2_state_estimator import EstimatedMarketStateV1
from .layer3_strategy_controller import ControlActionV1, PortfolioStateV1


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass(frozen=True)
class RiskConfigV1:
    base_max_exposure: float = 1.0
    base_max_leverage: float = 1.0
    max_allowed_drawdown: float = 0.20
    reduce_only_drawdown: float = 0.15
    kill_switch_drawdown: float = 0.20
    base_turnover_limit: float = 0.20
    min_turnover: float = 0.02
    danger_reduce_threshold: float = 0.75
    danger_kill_threshold: float = 0.90
    shock_reduce_threshold: float = 0.80
    shock_kill_threshold: float = 0.90
    crash_reduce_threshold: float = 0.50
    crash_kill_threshold: float = 0.70
    liquidity_stress_threshold: float = 0.25
    liquidity_kill_threshold: float = 0.20
    min_trade_threshold: float = 0.03


@dataclass(frozen=True)
class SafeControlActionV1:
    safe_target_exposure: float
    safe_exposure_change: float
    allowed_max_exposure: float
    allowed_max_leverage: float
    allowed_turnover: float
    trade_allowed: bool
    reduce_only: bool
    emergency_deleveraging: bool
    kill_switch: bool
    final_action_type: str
    risk_reason_code: str


def apply_risk_filter(
    market: MarketStateV1,
    estimated: EstimatedMarketStateV1,
    portfolio: PortfolioStateV1,
    action: ControlActionV1,
    config: RiskConfigV1 | None = None,
    long_only: bool = False,
) -> SafeControlActionV1:
    config = config or RiskConfigV1()

    vol_penalty = _clip((market.volatility_score - 1.0) / 2.0, 0.0, 1.0)
    dd_severity = _clip(abs(portfolio.portfolio_drawdown) / config.max_allowed_drawdown, 0.0, 1.0)
    allowed_max_exposure = (
        config.base_max_exposure
        * (1.0 - estimated.danger_score)
        * (1.0 - 0.6 * vol_penalty)
        * (1.0 - 0.9 * estimated.p_crash_risk)
        * (1.0 - 0.8 * dd_severity)
    )
    allowed_max_exposure = _clip(allowed_max_exposure, 0.0, config.base_max_exposure)

    allowed_max_leverage = min(config.base_max_leverage, action.max_leverage)
    safe_target_exposure = _clip(action.target_exposure, 0.0, allowed_max_exposure) if long_only else _clip(action.target_exposure, -allowed_max_exposure, allowed_max_exposure)

    if portfolio.current_exposure * safe_target_exposure < 0 and (
        estimated.state_confidence < 0.7 or estimated.transition_risk > 0.4
    ):
        safe_target_exposure = 0.0

    allowed_turnover = (
        config.base_turnover_limit
        * market.liquidity_score
        * (1.0 - 0.7 * market.shock_score)
        * (1.0 - 0.5 * estimated.transition_risk)
    )
    allowed_turnover = _clip(allowed_turnover, config.min_turnover, config.base_turnover_limit)

    reduce_only = action.reduce_only or _should_reduce_only(market, estimated, portfolio, config)
    risk_reason_code = "ok"
    if reduce_only:
        safe_target_exposure = _reduce_only_target(portfolio.current_exposure, safe_target_exposure)
        risk_reason_code = "reduce_only"

    kill_switch = _should_kill_switch(market, estimated, portfolio, allowed_max_leverage, config)
    emergency_deleveraging = False
    if kill_switch:
        reduce_only = True
        emergency_deleveraging = True
        safe_target_exposure = 0.0
        risk_reason_code = "kill_switch"

    raw_change = safe_target_exposure - portfolio.current_exposure
    safe_exposure_change = _clip(raw_change, -allowed_turnover, allowed_turnover)
    if abs(safe_exposure_change) < config.min_trade_threshold:
        safe_exposure_change = 0.0

    trade_allowed = action.trade_allowed
    if kill_switch:
        trade_allowed = True
    elif safe_exposure_change == 0.0:
        trade_allowed = False

    final_action_type = "hold"
    if kill_switch:
        final_action_type = "kill_switch"
    elif trade_allowed and reduce_only:
        final_action_type = "reduce"
    elif trade_allowed:
        final_action_type = "rebalance"

    return SafeControlActionV1(
        safe_target_exposure=safe_target_exposure,
        safe_exposure_change=safe_exposure_change,
        allowed_max_exposure=allowed_max_exposure,
        allowed_max_leverage=allowed_max_leverage,
        allowed_turnover=allowed_turnover,
        trade_allowed=trade_allowed,
        reduce_only=reduce_only,
        emergency_deleveraging=emergency_deleveraging,
        kill_switch=kill_switch,
        final_action_type=final_action_type,
        risk_reason_code=risk_reason_code,
    )


def _should_reduce_only(
    market: MarketStateV1,
    estimated: EstimatedMarketStateV1,
    portfolio: PortfolioStateV1,
    config: RiskConfigV1,
) -> bool:
    liquidity_stress = market.liquidity_score <= config.liquidity_stress_threshold and market.volatility_score > 1.8
    return (
        estimated.danger_score >= config.danger_reduce_threshold
        or market.shock_score >= config.shock_reduce_threshold
        or estimated.p_crash_risk >= config.crash_reduce_threshold
        or portfolio.portfolio_drawdown <= -config.reduce_only_drawdown
        or liquidity_stress
    )


def _should_kill_switch(
    market: MarketStateV1,
    estimated: EstimatedMarketStateV1,
    portfolio: PortfolioStateV1,
    allowed_max_leverage: float,
    config: RiskConfigV1,
) -> bool:
    return (
        portfolio.portfolio_drawdown <= -config.kill_switch_drawdown
        or estimated.danger_score >= config.danger_kill_threshold
        or estimated.p_crash_risk >= config.crash_kill_threshold
        or (market.shock_score >= config.shock_kill_threshold and market.liquidity_score <= config.liquidity_kill_threshold)
        or (allowed_max_leverage > 0 and portfolio.leverage > allowed_max_leverage * 1.2)
    )


def _reduce_only_target(current_exposure: float, target_exposure: float) -> float:
    if current_exposure > 0:
        return _clip(target_exposure, 0.0, abs(current_exposure))
    if current_exposure < 0:
        return _clip(target_exposure, -abs(current_exposure), 0.0)
    return 0.0

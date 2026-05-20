"""Layer 5: rule-based adaptive supervisor and health monitor."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass(frozen=True)
class AdaptiveConfigV1:
    performance_window: int = 100
    regime_performance_window: int = 300
    min_model_health: float = 0.4
    min_strategy_health: float = 0.4
    max_param_update_step: float = 0.05
    min_softmax_temperature: float = 0.5
    max_softmax_temperature: float = 1.5
    min_smoothing_alpha: float = 0.05
    max_smoothing_alpha: float = 0.40
    min_base_turnover_limit: float = 0.02
    max_base_turnover_limit: float = 0.25
    max_allowed_parameter_drift: float = 0.30


@dataclass(frozen=True)
class SystemHistoryV1:
    strategy_returns: list[float] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    drawdowns: list[float] = field(default_factory=list)
    regime_uncertainties: list[float] = field(default_factory=list)
    state_confidences: list[float] = field(default_factory=list)
    shock_scores: list[float] = field(default_factory=list)
    turnovers: list[float] = field(default_factory=list)
    transaction_costs: list[float] = field(default_factory=list)
    intervention_flags: list[bool] = field(default_factory=list)
    return_by_regime: dict[str, float] = field(default_factory=dict)
    drawdown_by_regime: dict[str, float] = field(default_factory=dict)
    turnover_by_regime: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class AdaptiveUpdateV1:
    model_health_score: float
    strategy_health_score: float
    overfit_risk_score: float
    new_softmax_temperature: float
    new_smoothing_alpha: float
    new_sideways_penalty: float
    new_volatility_penalty: float
    new_base_turnover_limit: float
    new_danger_reduce_threshold: float
    new_danger_kill_threshold: float
    adaptation_mode: str
    reason_code: str


def supervise_adaptation(
    history: SystemHistoryV1,
    config: AdaptiveConfigV1 | None = None,
    current_softmax_temperature: float = 0.7,
    current_smoothing_alpha: float = 0.2,
    current_sideways_penalty: float = 0.5,
    current_volatility_penalty: float = 0.5,
    current_base_turnover_limit: float = 0.20,
    current_danger_reduce_threshold: float = 0.75,
    current_danger_kill_threshold: float = 0.90,
) -> AdaptiveUpdateV1:
    config = config or AdaptiveConfigV1()
    intervention_rate = _mean_bool(history.intervention_flags)
    avg_uncertainty = _avg(history.regime_uncertainties)
    avg_confidence = _avg(history.state_confidences, default=1.0)
    shock_frequency = sum(1 for value in history.shock_scores if value >= 0.7) / len(history.shock_scores) if history.shock_scores else 0.0
    low_confidence_penalty = _clip((0.4 - avg_confidence) / 0.4, 0.0, 1.0)
    abnormal_shock_frequency = _clip(shock_frequency / 0.3, 0.0, 1.0)
    model_health = _clip(
        1.0 - 0.3 * avg_uncertainty - 0.3 * intervention_rate - 0.2 * low_confidence_penalty - 0.2 * abnormal_shock_frequency,
        0.0,
        1.0,
    )

    rolling_sharpe = _rolling_sharpe(history.strategy_returns[-config.performance_window :])
    rolling_drawdown = abs(min(history.drawdowns[-config.performance_window :], default=0.0))
    turnover = _avg(history.turnovers[-config.performance_window :])
    net_return = sum(history.strategy_returns[-config.performance_window :]) - sum(history.transaction_costs[-config.performance_window :])
    strategy_health = _clip(
        1.0
        + 0.3 * _clip(rolling_sharpe / 2.0, -1.0, 1.0)
        - 0.4 * _clip(rolling_drawdown / 0.2, 0.0, 1.0)
        - 0.2 * _turnover_cost_penalty(turnover, net_return)
        - 0.1 * intervention_rate,
        0.0,
        1.0,
    )
    overfit_risk = _clip(0.4 * intervention_rate + 0.3 * avg_uncertainty + 0.3 * max(0.0, -rolling_sharpe), 0.0, 1.0)

    mode = "normal"
    reason = "healthy"
    if avg_confidence < 0.4:
        mode, reason = "cautious", "low_state_confidence"
    if rolling_sharpe < 0 and rolling_drawdown >= 0.10:
        mode, reason = "defensive", "performance_degradation"
    if model_health < config.min_model_health:
        mode, reason = "retrain_required", "model_health_low"
    if strategy_health < config.min_strategy_health or rolling_drawdown >= 0.20:
        mode, reason = "disabled", "strategy_health_low"

    new_sideways_penalty = current_sideways_penalty
    new_volatility_penalty = current_volatility_penalty
    new_turnover_limit = current_base_turnover_limit
    new_reduce_threshold = current_danger_reduce_threshold
    new_kill_threshold = current_danger_kill_threshold

    step = config.max_param_update_step
    if history.return_by_regime.get("sideways", 0.0) < -0.02 and history.turnover_by_regime.get("sideways", 0.0) > 0.2:
        new_sideways_penalty += step
        new_turnover_limit -= step
        reason = "sideways_underperformance"
    if abs(history.drawdown_by_regime.get("high_vol", 0.0)) > 0.10:
        new_volatility_penalty += step
        reason = "high_vol_drawdown"
    if abs(history.drawdown_by_regime.get("crash_risk", 0.0)) > 0.08:
        new_reduce_threshold -= step
        new_kill_threshold -= step
        reason = "crash_risk_loss"
    if turnover > 0.2 and net_return <= 0:
        new_turnover_limit -= step
        reason = "turnover_too_high"
    if intervention_rate > 0.5:
        new_volatility_penalty += step
        new_sideways_penalty += step
        reason = "intervention_rate_high"

    if mode in {"cautious", "defensive", "retrain_required", "disabled"}:
        new_smoothing_alpha = current_smoothing_alpha - step
        new_softmax_temperature = current_softmax_temperature + step
    else:
        new_smoothing_alpha = current_smoothing_alpha
        new_softmax_temperature = current_softmax_temperature

    return AdaptiveUpdateV1(
        model_health_score=model_health,
        strategy_health_score=strategy_health,
        overfit_risk_score=overfit_risk,
        new_softmax_temperature=_clip(new_softmax_temperature, config.min_softmax_temperature, config.max_softmax_temperature),
        new_smoothing_alpha=_clip(new_smoothing_alpha, config.min_smoothing_alpha, config.max_smoothing_alpha),
        new_sideways_penalty=_clip(new_sideways_penalty, 0.0, 1.0),
        new_volatility_penalty=_clip(new_volatility_penalty, 0.0, 1.0),
        new_base_turnover_limit=_clip(new_turnover_limit, config.min_base_turnover_limit, config.max_base_turnover_limit),
        new_danger_reduce_threshold=_clip(new_reduce_threshold, 0.50, 0.90),
        new_danger_kill_threshold=_clip(new_kill_threshold, 0.60, 0.99),
        adaptation_mode=mode,
        reason_code=reason,
    )


def _avg(values: list[float], default: float = 0.0) -> float:
    return mean(values) if values else default


def _mean_bool(values: list[bool]) -> float:
    return sum(1 for value in values if value) / len(values) if values else 0.0


def _rolling_sharpe(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    avg_return = mean(returns)
    variance = mean([(value - avg_return) ** 2 for value in returns])
    std = variance ** 0.5
    return avg_return / std if std > 0 else 0.0


def _turnover_cost_penalty(turnover: float, net_return: float) -> float:
    if net_return <= 0:
        return _clip(turnover / 0.3, 0.0, 1.0)
    return _clip(turnover / max(abs(net_return), 1e-9), 0.0, 1.0)


"""v3 highest-authority risk supervisor.

The risk supervisor does not create alpha and does not execute trades. It only
limits risk by returning a risk cap and action that downstream composition and
execution layers must respect.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .data_types import LongTermDecisionV3, MarketEstimateV3, RiskDecisionV3, ShortTermDecisionV3


RISK_ACTIONS = ("normal", "no_new_entry", "reduce_only", "force_deleverage", "risk_off")


@dataclass(frozen=True)
class PortfolioRiskStateV3:
    """Portfolio fields required by the v3 risk supervisor."""

    portfolio_drawdown: float
    realized_volatility: float
    consecutive_losses: int
    current_position: float
    recent_turnover: float = 0.0
    fee_drag: float = 0.0


@dataclass(frozen=True)
class RiskSupervisorConfig:
    """Thresholds for initial v3 risk caps and actions."""

    enable_drawdown_cap: bool = True
    enable_volatility_cap: bool = True
    enable_consecutive_loss_rules: bool = True
    enable_market_risk_state: bool = True
    enable_cost_guards: bool = True
    drawdown_caution: float = -0.05
    drawdown_danger: float = -0.10
    drawdown_severe: float = -0.15
    drawdown_risk_off: float = -0.20
    risk_off_cap: float = 0.0
    high_volatility_cap: float = 0.75
    extreme_volatility_cap: float = 0.50
    losses_no_new_entry: int = 2
    losses_reduce_cap: int = 3
    losses_risk_off: int = 4
    consecutive_loss_cap: float = 0.50
    fee_drag_caution: float | None = None
    turnover_caution: float | None = None


def supervise_risk(
    estimate: MarketEstimateV3,
    portfolio_state: PortfolioRiskStateV3 | dict[str, Any] | Any,
    long_decision: LongTermDecisionV3,
    short_decision: ShortTermDecisionV3,
    config: RiskSupervisorConfig | None = None,
) -> RiskDecisionV3:
    """Return the highest-authority v3 risk decision.

    The long-term and short-term decisions are accepted for diagnostics and
    intent awareness, but this module only limits risk. It never calculates
    signal alpha and never executes trades.
    """

    config = config or RiskSupervisorConfig()
    _validate_config(config)
    state = _portfolio_values(portfolio_state)

    caps: list[tuple[float, str]] = []
    actions: list[tuple[str, str]] = []

    dd_cap, dd_action, dd_reason = (
        _drawdown_cap(state["portfolio_drawdown"], config)
        if config.enable_drawdown_cap
        else (1.00, "normal", "drawdown_cap_disabled")
    )
    caps.append((dd_cap, dd_reason))
    actions.append((dd_action, dd_reason))

    vol_cap, vol_action, vol_reason = (
        _volatility_cap(str(estimate.volatility_state), config)
        if config.enable_volatility_cap
        else (1.00, "normal", "volatility_cap_disabled")
    )
    caps.append((vol_cap, vol_reason))
    actions.append((vol_action, vol_reason))

    loss_cap, loss_action, loss_reason = (
        _loss_cap(state["consecutive_losses"], short_decision, config)
        if config.enable_consecutive_loss_rules
        else (1.00, "normal", "consecutive_loss_rules_disabled")
    )
    caps.append((loss_cap, loss_reason))
    actions.append((loss_action, loss_reason))

    optional_cap, optional_action, optional_reason = (
        _optional_cost_cap(state, config)
        if config.enable_cost_guards
        else (1.00, "normal", "cost_guards_disabled")
    )
    caps.append((optional_cap, optional_reason))
    actions.append((optional_action, optional_reason))

    if config.enable_market_risk_state and str(estimate.risk_state) == "risk_off":
        caps.append((min(config.risk_off_cap, 0.25), "market_estimate_risk_off"))
        actions.append(("risk_off", "market_estimate_risk_off"))

    risk_cap = min(cap for cap, _ in caps)
    risk_action = _most_severe_action(action for action, _ in actions)
    reason_parts = [reason for _, reason in caps + actions if reason and reason != "normal"]
    if not reason_parts:
        reason_parts = ["normal"]
    reason = "; ".join(dict.fromkeys(reason_parts))

    return RiskDecisionV3(
        timestamp=estimate.timestamp,
        risk_cap=float(_clip_to_v3_cap(risk_cap)),
        risk_action=risk_action,
        reason=reason,
        portfolio_drawdown=state["portfolio_drawdown"],
        realized_volatility=state["realized_volatility"],
    )


def _drawdown_cap(drawdown: float, config: RiskSupervisorConfig) -> tuple[float, str, str]:
    if drawdown <= config.drawdown_risk_off:
        return config.risk_off_cap, "risk_off", "drawdown_risk_off"
    if drawdown <= config.drawdown_severe:
        return 0.25, "force_deleverage", "drawdown_severe_cap_025"
    if drawdown <= config.drawdown_danger:
        return 0.50, "reduce_only", "drawdown_danger_cap_050"
    if drawdown <= config.drawdown_caution:
        return 0.75, "no_new_entry", "drawdown_caution_cap_075"
    return 1.00, "normal", "normal"


def _volatility_cap(volatility_state: str, config: RiskSupervisorConfig) -> tuple[float, str, str]:
    if volatility_state == "extreme":
        return config.extreme_volatility_cap, "reduce_only", "volatility_extreme_cap"
    if volatility_state == "high":
        # High volatility should reduce maximum exposure, not erase the long-term
        # controller's ability to enter. Extreme volatility remains reduce-only.
        return config.high_volatility_cap, "normal", "volatility_high_cap"
    return 1.00, "normal", "normal"


def _loss_cap(
    consecutive_losses: int,
    short_decision: ShortTermDecisionV3,
    config: RiskSupervisorConfig,
) -> tuple[float, str, str]:
    if consecutive_losses >= config.losses_risk_off:
        return 0.25, "risk_off", "consecutive_losses_risk_off"
    if consecutive_losses >= config.losses_reduce_cap:
        return config.consecutive_loss_cap, "reduce_only", "consecutive_losses_reduce_cap"
    if consecutive_losses >= config.losses_no_new_entry:
        if short_decision.position_adjustment > 0.0:
            return 1.00, "no_new_entry", "consecutive_losses_block_short_addition"
        return 1.00, "no_new_entry", "consecutive_losses_no_new_entry"
    return 1.00, "normal", "normal"


def _optional_cost_cap(state: dict[str, Any], config: RiskSupervisorConfig) -> tuple[float, str, str]:
    if config.fee_drag_caution is not None and state["fee_drag"] >= config.fee_drag_caution:
        return 0.75, "no_new_entry", "fee_drag_caution"
    if config.turnover_caution is not None and state["recent_turnover"] >= config.turnover_caution:
        return 0.75, "no_new_entry", "turnover_caution"
    return 1.00, "normal", "normal"


def _most_severe_action(actions: Any) -> str:
    severity = {
        "normal": 0,
        "no_new_entry": 1,
        "reduce_only": 2,
        "force_deleverage": 3,
        "risk_off": 4,
    }
    return max(actions, key=lambda action: severity[str(action)])


def _portfolio_values(portfolio_state: PortfolioRiskStateV3 | dict[str, Any] | Any) -> dict[str, Any]:
    if isinstance(portfolio_state, PortfolioRiskStateV3):
        values = asdict(portfolio_state)
    elif isinstance(portfolio_state, dict):
        values = dict(portfolio_state)
    else:
        values = {
            "portfolio_drawdown": getattr(portfolio_state, "portfolio_drawdown"),
            "realized_volatility": getattr(portfolio_state, "realized_volatility"),
            "consecutive_losses": getattr(portfolio_state, "consecutive_losses"),
            "current_position": getattr(portfolio_state, "current_position"),
            "recent_turnover": getattr(portfolio_state, "recent_turnover", 0.0),
            "fee_drag": getattr(portfolio_state, "fee_drag", 0.0),
        }

    required = {"portfolio_drawdown", "realized_volatility", "consecutive_losses", "current_position"}
    missing = required.difference(values)
    if missing:
        raise ValueError(f"portfolio_state is missing required fields: {sorted(missing)}")
    return {
        "portfolio_drawdown": float(values["portfolio_drawdown"]),
        "realized_volatility": float(values["realized_volatility"]),
        "consecutive_losses": int(values["consecutive_losses"]),
        "current_position": float(values["current_position"]),
        "recent_turnover": float(values.get("recent_turnover", 0.0)),
        "fee_drag": float(values.get("fee_drag", 0.0)),
    }


def _clip_to_v3_cap(value: float) -> float:
    return max(0.0, min(1.0, value))


def _validate_config(config: RiskSupervisorConfig) -> None:
    if config.risk_off_cap not in {0.0, 0.25}:
        raise ValueError("risk_off_cap must be 0.0 or 0.25")
    if config.extreme_volatility_cap not in {0.25, 0.50}:
        raise ValueError("extreme_volatility_cap must be 0.25 or 0.50")
    if config.high_volatility_cap > 1.0 or config.high_volatility_cap < 0.0:
        raise ValueError("high_volatility_cap must be in [0, 1]")
    if not (config.losses_no_new_entry <= config.losses_reduce_cap <= config.losses_risk_off):
        raise ValueError("loss thresholds must be non-decreasing")


__all__ = [
    "PortfolioRiskStateV3",
    "RISK_ACTIONS",
    "RiskDecisionV3",
    "RiskSupervisorConfig",
    "supervise_risk",
]

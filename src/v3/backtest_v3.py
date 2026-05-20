"""v3 isolated backtest pipeline.

The v3 backtest wires together the v3 feature builder, estimator, controllers,
cooldown manager, risk supervisor, position composer, and execution layer. It
does not modify or replace the existing v1/v2 backtests.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from .cooldown_manager import RegimeCooldownManagerV3, TradeCloseInfoV3
from .diagnostics import build_v3_diagnostics, calculate_v3_metrics, write_v3_diagnostics
from .execution_layer import apply_execution, compute_strategy_return_net
from .feature_builder import FeatureWindowConfig, build_feature_frame
from .long_term_controller import LongTermControllerConfig, decide_long_term_position
from .market_estimator import MarketEstimatorConfig, estimate_market
from .position_composer import PositionComposerConfig, compose_target_position
from .risk_supervisor import PortfolioRiskStateV3, RiskSupervisorConfig, supervise_risk
from .short_term_controller import ShortTermControllerConfig, decide_short_term_adjustment


@dataclass(frozen=True)
class ConditionalLeverageConfigV3:
    """Optional v3.5 leverage overlay, disabled by default.

    This is an experiment-only overlay. It can add a small exposure increment
    only when the long-term regime, confidence, volatility, portfolio drawdown,
    cooldown, and recent loss conditions are all clean.
    """

    enabled: bool = False
    max_position: float = 1.25
    leverage_increment: float = 0.25
    high_confidence: float = 0.80
    max_drawdown_for_leverage: float = -0.05
    require_raw_target_at_least: float = 1.0


@dataclass(frozen=True)
class BacktestV3Config:
    """Configuration bundle for the isolated v3 backtest."""

    fee_rate: float = 0.001
    slippage_rate: float = 0.0
    initial_equity: float = 1.0
    cooldown_bars: int = 120
    minimum_position_step: float = 0.25
    recent_turnover_window: int = 24
    feature_config: FeatureWindowConfig | None = None
    estimator_config: MarketEstimatorConfig | None = None
    long_term_config: LongTermControllerConfig | None = None
    short_term_config: ShortTermControllerConfig | None = None
    risk_config: RiskSupervisorConfig | None = None
    composer_config: PositionComposerConfig | None = None
    leverage_config: ConditionalLeverageConfigV3 | None = None


def run_v3_backtest(
    ohlcv: pd.DataFrame,
    config: BacktestV3Config | None = None,
) -> pd.DataFrame:
    """Run the initial v3 backtest and return a diagnostics DataFrame.

    Signal at bar ``t`` determines ``executed_position[t]`` for the next bar.
    Return at bar ``t`` is earned by ``executed_position[t-1]``.
    """

    config = config or BacktestV3Config()
    _validate_config(config)
    features = build_feature_frame(ohlcv, config=config.feature_config)

    cooldown = RegimeCooldownManagerV3(cooldown_bars=config.cooldown_bars)
    equity = float(config.initial_equity)
    equity_peak = equity
    current_position = 0.0
    open_trade_entry_regime = ""
    open_trade_net_return = 0.0
    consecutive_losses = 0
    recent_trade_amounts: list[float] = []
    rows: list[dict[str, Any]] = []

    for feature_row in features.itertuples(index=False):
        asset_return = float(feature_row.return_1)
        portfolio_drawdown = equity / equity_peak - 1.0
        recent_turnover = sum(recent_trade_amounts[-config.recent_turnover_window :])

        estimate = estimate_market(pd.Series(feature_row._asdict()), config=config.estimator_config)
        long_decision = decide_long_term_position(estimate, config=config.long_term_config)
        cooldown_active = cooldown.is_active(estimate.long_regime)
        short_decision = decide_short_term_adjustment(
            estimate,
            cooldown_state=cooldown_active,
            config=config.short_term_config,
        )
        risk_decision = supervise_risk(
            estimate,
            PortfolioRiskStateV3(
                portfolio_drawdown=portfolio_drawdown,
                realized_volatility=float(feature_row.volatility_short),
                consecutive_losses=consecutive_losses,
                current_position=current_position,
                recent_turnover=recent_turnover,
                fee_drag=0.0,
            ),
            long_decision,
            short_decision,
            config=config.risk_config,
        )
        leverage_allowed, leverage_reason = _conditional_leverage_allowed(
            estimate=estimate,
            portfolio_drawdown=portfolio_drawdown,
            cooldown_active=cooldown_active,
            consecutive_losses=consecutive_losses,
            config=config.leverage_config,
        )
        leverage_increment = 0.0
        effective_long_decision = long_decision
        effective_risk_decision = risk_decision
        composer_config = config.composer_config
        raw_without_leverage = float(long_decision.base_position) + float(short_decision.position_adjustment)
        if (
            leverage_allowed
            and config.leverage_config is not None
            and raw_without_leverage >= config.leverage_config.require_raw_target_at_least - 1e-12
            and risk_decision.risk_action == "normal"
        ):
            leverage_increment = config.leverage_config.leverage_increment
            effective_long_decision = replace(
                long_decision,
                base_position=float(long_decision.base_position) + leverage_increment,
                reason=f"{long_decision.reason}; conditional_leverage_overlay_{leverage_increment:.2f}",
            )
            effective_risk_decision = replace(
                risk_decision,
                risk_cap=max(float(risk_decision.risk_cap), float(config.leverage_config.max_position)),
            )
            composer_config = _leverage_composer_config(config.composer_config, config.leverage_config)
            leverage_reason = "conditional_leverage_allowed"
        elif leverage_allowed and raw_without_leverage < (config.leverage_config.require_raw_target_at_least if config.leverage_config else 1.0):
            leverage_reason = "blocked_raw_target_below_leverage_threshold"
        elif leverage_allowed and risk_decision.risk_action != "normal":
            leverage_reason = f"blocked_risk_action_{risk_decision.risk_action}"

        composed = compose_target_position(
            effective_long_decision,
            short_decision,
            effective_risk_decision,
            config=composer_config,
        )
        executed = apply_execution(
            composed,
            current_position=current_position,
            risk_action=effective_risk_decision.risk_action,
            fee_rate=config.fee_rate,
            slippage_rate=config.slippage_rate,
            minimum_position_step=config.minimum_position_step,
            confidence_score=estimate.confidence_score,
            risk_cap=effective_risk_decision.risk_cap,
        )

        previous_position = current_position
        executed_position = executed.executed_position
        trade_amount = abs(executed_position - previous_position)
        raw_target_position = float(long_decision.base_position) + float(short_decision.position_adjustment)
        risk_limited_position = min(raw_target_position, float(effective_risk_decision.risk_cap))
        fee_cost = trade_amount * config.fee_rate
        slippage_cost = trade_amount * config.slippage_rate
        strategy_return_gross = previous_position * asset_return
        strategy_return_net = compute_strategy_return_net(
            previous_position=previous_position,
            current_position=executed_position,
            asset_return=asset_return,
            fee_rate=config.fee_rate,
            slippage_rate=config.slippage_rate,
        )

        equity *= 1.0 + strategy_return_net
        equity = max(equity, 1e-12)
        equity_peak = max(equity_peak, equity)
        drawdown = equity / equity_peak - 1.0
        recent_trade_amounts.append(trade_amount)

        if previous_position > 0.0:
            open_trade_net_return += strategy_return_net

        cooldown.update_on_bar()
        cooldown_triggered = False
        if previous_position <= 0.0 and executed_position > 0.0:
            open_trade_entry_regime = estimate.long_regime
            open_trade_net_return = strategy_return_net
        elif previous_position > 0.0 and executed_position <= 0.0:
            cooldown.update_on_trade_close(
                TradeCloseInfoV3(
                    entry_regime=open_trade_entry_regime,
                    exit_regime=estimate.long_regime,
                    net_trade_return=open_trade_net_return,
                )
            )
            cooldown_triggered = cooldown.is_active(open_trade_entry_regime) and open_trade_net_return < 0.0
            consecutive_losses = consecutive_losses + 1 if open_trade_net_return < 0.0 else 0
            open_trade_entry_regime = ""
            open_trade_net_return = 0.0

        cooldown_state = cooldown.get_state()
        cooldown_regime = ";".join(
            f"{regime}:{remaining}"
            for regime, remaining in sorted(cooldown_state.remaining_by_regime.items())
        )
        rows.append(
            {
                "timestamp": feature_row.timestamp,
                "close": float(feature_row.close),
                "asset_return": asset_return,
                "long_regime": estimate.long_regime,
                "short_regime": estimate.short_regime,
                "trend_strength": estimate.trend_strength,
                "volatility_state": estimate.volatility_state,
                "drawdown_state": estimate.drawdown_state,
                "risk_state": estimate.risk_state,
                "confidence_score": estimate.confidence_score,
                "allow_entry": estimate.allow_entry,
                "allow_hold": estimate.allow_hold,
                "base_position": long_decision.base_position,
                "position_adjustment": short_decision.position_adjustment,
                "short_adjustment": short_decision.position_adjustment,
                "leverage_increment": leverage_increment,
                "leverage_allowed": leverage_allowed,
                "leverage_used": executed_position > 1.0,
                "leverage_reason": leverage_reason,
                "risk_cap": effective_risk_decision.risk_cap,
                "raw_target_position": raw_target_position,
                "risk_limited_position": risk_limited_position,
                "target_position": executed.target_position,
                "current_position": previous_position,
                "executed_position": executed_position,
                "trade_amount": trade_amount,
                "fee_cost": fee_cost,
                "slippage_cost": slippage_cost,
                "strategy_return_gross": strategy_return_gross,
                "strategy_return_net": strategy_return_net,
                "equity_curve": equity,
                "drawdown": drawdown,
                "risk_action": risk_decision.risk_action,
                "long_reason": long_decision.reason,
                "short_reason": short_decision.reason,
                "risk_reason": risk_decision.reason,
                "execution_reason": executed.execution_reason,
                "cooldown_active": cooldown_active,
                "cooldown_triggered": cooldown_triggered,
                "cooldown_remaining": max(cooldown_state.remaining_by_regime.values(), default=0),
                "cooldown_regime": cooldown_regime,
                "consecutive_losses": consecutive_losses,
            }
        )
        current_position = executed_position

    return pd.DataFrame(rows)


def run_v3_backtest_from_csv(
    csv_path: str | Path,
    config: BacktestV3Config | None = None,
) -> pd.DataFrame:
    """Load an OHLCV CSV and run the isolated v3 backtest."""

    return run_v3_backtest(pd.read_csv(csv_path), config=config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the isolated v3 backtest.")
    parser.add_argument("--csv", type=Path, required=True, help="OHLCV CSV with timestamp,open,high,low,close,volume columns.")
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--slippage-rate", type=float, default=0.0)
    parser.add_argument("--cooldown-bars", type=int, default=120)
    parser.add_argument("--output", type=Path, default=None, help="Optional output CSV path for the v3 result frame.")
    parser.add_argument("--diagnostics-dir", type=Path, default=None, help="Optional directory for v3 diagnostic CSV tables.")
    args = parser.parse_args()

    result = run_v3_backtest_from_csv(
        args.csv,
        config=BacktestV3Config(
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
            cooldown_bars=args.cooldown_bars,
        ),
    )
    metrics = calculate_v3_metrics(result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
    if args.diagnostics_dir:
        write_v3_diagnostics(build_v3_diagnostics(result), args.diagnostics_dir)
    for key, value in metrics.items():
        print(f"{key}: {value}")
    if args.output:
        print(f"result_csv: {args.output}")
    if args.diagnostics_dir:
        print(f"diagnostics_dir: {args.diagnostics_dir}")


def _validate_config(config: BacktestV3Config) -> None:
    if config.fee_rate < 0.0:
        raise ValueError("fee_rate must be non-negative")
    if config.slippage_rate < 0.0:
        raise ValueError("slippage_rate must be non-negative")
    if config.initial_equity <= 0.0:
        raise ValueError("initial_equity must be positive")
    if config.cooldown_bars < 0:
        raise ValueError("cooldown_bars must be non-negative")
    if config.minimum_position_step <= 0.0:
        raise ValueError("minimum_position_step must be positive")
    if config.recent_turnover_window <= 0:
        raise ValueError("recent_turnover_window must be positive")
    if config.leverage_config is not None:
        if config.leverage_config.max_position > 1.25:
            raise ValueError("v3.5 experimental max_position must not exceed 1.25")
        if config.leverage_config.max_position <= 1.0 and config.leverage_config.enabled:
            raise ValueError("enabled leverage_config requires max_position above 1.0")
        if config.leverage_config.leverage_increment <= 0.0:
            raise ValueError("leverage_increment must be positive")
        if not 0.0 <= config.leverage_config.high_confidence <= 1.0:
            raise ValueError("high_confidence must be in [0, 1]")


def _conditional_leverage_allowed(
    *,
    estimate: Any,
    portfolio_drawdown: float,
    cooldown_active: bool,
    consecutive_losses: int,
    config: ConditionalLeverageConfigV3 | None,
) -> tuple[bool, str]:
    if config is None or not config.enabled:
        return False, "leverage_disabled"
    if estimate.long_regime != "strong_bull":
        return False, "blocked_not_strong_bull"
    if float(estimate.confidence_score) < config.high_confidence:
        return False, "blocked_confidence_below_threshold"
    if estimate.volatility_state not in {"low", "normal"}:
        return False, "blocked_volatility_not_low_or_normal"
    if float(portfolio_drawdown) <= config.max_drawdown_for_leverage:
        return False, "blocked_drawdown_at_or_below_threshold"
    if cooldown_active:
        return False, "blocked_cooldown_active"
    if int(consecutive_losses) > 0:
        return False, "blocked_recent_consecutive_losses"
    return True, "conditional_leverage_precheck_passed"


def _leverage_composer_config(
    base_config: PositionComposerConfig | None,
    leverage_config: ConditionalLeverageConfigV3,
) -> PositionComposerConfig:
    base = base_config or PositionComposerConfig()
    allowed = tuple(sorted(set(base.allowed_positions + (float(leverage_config.max_position),))))
    return replace(
        base,
        max_position=float(leverage_config.max_position),
        allowed_positions=allowed,
        allow_leverage=True,
    )


if __name__ == "__main__":
    main()

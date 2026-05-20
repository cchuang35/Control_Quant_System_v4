from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtester import load_ohlcv_csv, run_backtest_fast
from scripts.run_v3_decision_waterfall import add_waterfall_fields
from scripts.run_v3_final_candidate import (
    PERIODS_PER_YEAR,
    V1_ENTRY_THRESHOLD,
    build_backtest_config,
    build_buy_and_hold_frame,
    build_ma_crossover_frame,
    input_frame_from_v1,
    load_final_candidate_config,
    summarize_frame,
)
from src.v3.backtest_v3 import BacktestV3Config
from src.v3.cooldown_manager import RegimeCooldownManagerV3, TradeCloseInfoV3
from src.v3.execution_layer import apply_execution, compute_strategy_return_net
from src.v3.feature_builder import build_feature_frame
from src.v3.long_term_controller import decide_long_term_position
from src.v3.market_estimator import estimate_market
from src.v3.position_composer import compose_target_position
from src.v3.risk_supervisor import PortfolioRiskStateV3, RiskDecisionV3, RiskSupervisorConfig, supervise_risk
from src.v3.short_term_controller import decide_short_term_adjustment
from v2_small_cap import backtest_v2_btc_final_candidate_a, backtest_v2_final_candidate_a


REPORT_PATH = Path("reports") / "v3_risk_action_semantics_experiment.md"
SUMMARY_CSV = Path("reports") / "v3_risk_action_semantics_experiment_summary.csv"
DIAGNOSTICS_CSV = Path("reports") / "v3_risk_action_semantics_experiment_diagnostics.csv"
STRONG_BULL_CSV = Path("reports") / "v3_risk_action_semantics_experiment_strong_bull.csv"


@dataclass(frozen=True)
class VariantSpec:
    name: str
    policy: str
    regimes: tuple[str, ...] = ()
    cap_value: float | None = None
    probation_bars: int | None = None


@dataclass
class VariantState:
    equity: float = 1.0
    equity_peak: float = 1.0
    current_position: float = 0.0
    open_trade_entry_regime: str = ""
    open_trade_net_return: float = 0.0
    consecutive_losses: int = 0
    recent_trade_amounts: list[float] | None = None
    cooldown: RegimeCooldownManagerV3 | None = None
    flat_bars: int = 0
    consecutive_loss_active_bars: int = 0

    def __post_init__(self) -> None:
        if self.recent_trade_amounts is None:
            self.recent_trade_amounts = []
        if self.cooldown is None:
            self.cooldown = RegimeCooldownManagerV3(cooldown_bars=120)


VARIANTS = (
    VariantSpec("A_current_final_candidate", "current"),
    VariantSpec("B_probation_cap_0p25_strong_bull_only", "probation_cap", regimes=("strong_bull",), cap_value=0.25),
    VariantSpec("C_probation_cap_0p25_bull_and_strong_bull", "probation_cap", regimes=("strong_bull", "bull"), cap_value=0.25),
    VariantSpec("D_probation_cap_0p50_strong_bull_only", "probation_cap", regimes=("strong_bull",), cap_value=0.50),
    VariantSpec("E_two_stage_probation", "two_stage_probation", regimes=("strong_bull",), cap_value=0.25, probation_bars=72),
    VariantSpec("F_probation_requires_positive_recent_return", "positive_recent_return", regimes=("strong_bull",), cap_value=0.25),
)


def main() -> None:
    config = load_final_candidate_config()
    fee_rates = tuple(float(value) for value in config["execution"]["fee_rates_to_validate"])
    summary_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    strong_bull_rows: list[dict[str, Any]] = []

    for asset, dataset_key in [("BTC", "btc_datasets"), ("ETH", "eth_datasets")]:
        paths = [Path(path) for path in config["validation"].get(dataset_key, []) if Path(path).exists()]
        for path in paths:
            dataset = path.stem
            print(f"asset={asset} dataset={dataset}")
            data = load_ohlcv_csv(path)
            for fee_rate in fee_rates:
                print(f"  fee={fee_rate:g}")
                backtest_config = build_backtest_config(config, fee_rate=fee_rate)
                variant_frames = run_variants_for_dataset(data, backtest_config)
                for variant, frame in variant_frames.items():
                    enriched = enrich_with_waterfall_fields(with_keys(frame, asset, dataset, fee_rate))
                    summary_rows.append({"asset": asset, "dataset": dataset, "fee_rate": fee_rate, "version": variant, **summarize_ablation_frame(enriched)})
                    diagnostic_rows.extend(build_distribution_rows(enriched, variant))
                    strong_bull_rows.append({"asset": asset, "dataset": dataset, "fee_rate": fee_rate, "version": variant, **strong_bull_summary(enriched)})
                summary_rows.extend(reference_rows(asset, dataset, data, fee_rate))

    summary = pd.DataFrame(summary_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    strong_bull = pd.DataFrame(strong_bull_rows)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)
    diagnostics.to_csv(DIAGNOSTICS_CSV, index=False)
    strong_bull.to_csv(STRONG_BULL_CSV, index=False)
    write_report(summary, diagnostics, strong_bull)
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {DIAGNOSTICS_CSV}")
    print(f"Wrote {STRONG_BULL_CSV}")


def run_variants_for_dataset(data: pd.DataFrame, config: BacktestV3Config) -> dict[str, pd.DataFrame]:
    features = build_feature_frame(data, config=config.feature_config)
    no_consecutive_config = _risk_config_without_consecutive(config.risk_config)
    states = {variant.name: VariantState(cooldown=RegimeCooldownManagerV3(cooldown_bars=config.cooldown_bars)) for variant in VARIANTS}
    rows = {variant.name: [] for variant in VARIANTS}

    for feature_row in features.itertuples(index=False):
        estimate = estimate_market(pd.Series(feature_row._asdict()), config=config.estimator_config)
        long_decision = decide_long_term_position(estimate, config=config.long_term_config)

        for variant in VARIANTS:
            state = states[variant.name]
            assert state.recent_trade_amounts is not None
            assert state.cooldown is not None
            update_consecutive_loss_age(state, config.risk_config)
            asset_return = float(feature_row.return_1)
            portfolio_drawdown = state.equity / state.equity_peak - 1.0
            recent_turnover = sum(state.recent_trade_amounts[-config.recent_turnover_window :])
            cooldown_active = state.cooldown.is_active(estimate.long_regime)
            short_decision = decide_short_term_adjustment(estimate, cooldown_state=cooldown_active, config=config.short_term_config)
            effective_consecutive_losses = adjusted_consecutive_losses(
                variant,
                state,
                estimate,
                portfolio_drawdown,
                config,
                no_consecutive_config,
                long_decision,
                short_decision,
                feature_row,
            )
            risk_decision = variant_risk_decision(
                variant,
                estimate,
                state,
                portfolio_drawdown,
                recent_turnover,
                config,
                no_consecutive_config,
                long_decision,
                short_decision,
                effective_consecutive_losses,
                feature_row,
            )
            composed = compose_target_position(long_decision, short_decision, risk_decision, config=config.composer_config)
            executed = apply_execution(
                composed,
                current_position=state.current_position,
                risk_action=risk_decision.risk_action,
                fee_rate=config.fee_rate,
                slippage_rate=config.slippage_rate,
                minimum_position_step=config.minimum_position_step,
                confidence_score=estimate.confidence_score,
                risk_cap=risk_decision.risk_cap,
            )
            previous_position = state.current_position
            executed_position = executed.executed_position
            trade_amount = abs(executed_position - previous_position)
            strategy_return_net = compute_strategy_return_net(
                previous_position=previous_position,
                current_position=executed_position,
                asset_return=asset_return,
                fee_rate=config.fee_rate,
                slippage_rate=config.slippage_rate,
            )
            state.equity *= 1.0 + strategy_return_net
            state.equity = max(state.equity, 1e-12)
            state.equity_peak = max(state.equity_peak, state.equity)
            state.recent_trade_amounts.append(trade_amount)
            drawdown = state.equity / state.equity_peak - 1.0
            if previous_position > 0.0:
                state.open_trade_net_return += strategy_return_net
            state.cooldown.update_on_bar()
            cooldown_triggered = False
            if previous_position <= 0.0 and executed_position > 0.0:
                state.open_trade_entry_regime = estimate.long_regime
                state.open_trade_net_return = strategy_return_net
            elif previous_position > 0.0 and executed_position <= 0.0:
                state.cooldown.update_on_trade_close(
                    TradeCloseInfoV3(
                        entry_regime=state.open_trade_entry_regime,
                        exit_regime=estimate.long_regime,
                        net_trade_return=state.open_trade_net_return,
                    )
                )
                cooldown_triggered = state.cooldown.is_active(state.open_trade_entry_regime) and state.open_trade_net_return < 0.0
                state.consecutive_losses = state.consecutive_losses + 1 if state.open_trade_net_return < 0.0 else 0
                state.open_trade_entry_regime = ""
                state.open_trade_net_return = 0.0
            if executed_position <= 0.0:
                state.flat_bars += 1
            else:
                state.flat_bars = 0
            cooldown_state = state.cooldown.get_state()
            raw_target_position = float(long_decision.base_position) + float(short_decision.position_adjustment)
            risk_limited_position = min(raw_target_position, float(risk_decision.risk_cap))
            rows[variant.name].append(
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
                    "risk_cap": risk_decision.risk_cap,
                    "raw_target_position": raw_target_position,
                    "risk_limited_position": risk_limited_position,
                    "target_position": executed.target_position,
                    "current_position": previous_position,
                    "executed_position": executed_position,
                    "trade_amount": trade_amount,
                    "fee_cost": trade_amount * config.fee_rate,
                    "strategy_return_gross": previous_position * asset_return,
                    "strategy_return_net": strategy_return_net,
                    "equity_curve": state.equity,
                    "drawdown": drawdown,
                    "risk_action": risk_decision.risk_action,
                    "risk_reason": risk_decision.reason,
                    "short_reason": short_decision.reason,
                    "execution_reason": executed.execution_reason,
                    "cooldown_active": cooldown_active,
                    "cooldown_triggered": cooldown_triggered,
                    "cooldown_remaining": max(cooldown_state.remaining_by_regime.values(), default=0),
                    "cooldown_regime": ";".join(f"{regime}:{remaining}" for regime, remaining in sorted(cooldown_state.remaining_by_regime.items())),
                    "consecutive_losses": state.consecutive_losses,
                    "consecutive_loss_active_bars": state.consecutive_loss_active_bars,
                    "flat_bars": state.flat_bars,
                }
            )
            state.current_position = executed_position

    return {variant: pd.DataFrame(records) for variant, records in rows.items()}


def adjusted_consecutive_losses(
    variant: VariantSpec,
    state: VariantState,
    estimate: Any,
    portfolio_drawdown: float,
    config: BacktestV3Config,
    no_consecutive_config: RiskSupervisorConfig,
    long_decision: Any,
    short_decision: Any,
    feature_row: Any,
) -> int:
    return state.consecutive_losses


def update_consecutive_loss_age(state: VariantState, config: RiskSupervisorConfig | None) -> None:
    risk_config = config or RiskSupervisorConfig()
    if state.consecutive_losses >= risk_config.losses_no_new_entry:
        state.consecutive_loss_active_bars += 1
    else:
        state.consecutive_loss_active_bars = 0


def variant_risk_decision(
    variant: VariantSpec,
    estimate: Any,
    state: VariantState,
    portfolio_drawdown: float,
    recent_turnover: float,
    config: BacktestV3Config,
    no_consecutive_config: RiskSupervisorConfig,
    long_decision: Any,
    short_decision: Any,
    consecutive_losses: int,
    feature_row: Any,
) -> RiskDecisionV3:
    risk_config = config.risk_config or RiskSupervisorConfig()
    risk_state = PortfolioRiskStateV3(
        portfolio_drawdown=portfolio_drawdown,
        realized_volatility=0.0,
        consecutive_losses=consecutive_losses,
        current_position=state.current_position,
        recent_turnover=recent_turnover,
        fee_drag=0.0,
    )
    if should_use_probation_cap(variant, estimate, state, consecutive_losses, risk_config, feature_row):
        base = supervise_risk(estimate, risk_state, long_decision, short_decision, config=no_consecutive_config)
        if consecutive_losses >= risk_config.losses_risk_off and base.risk_action != "risk_off":
            return replace(base, risk_cap=min(base.risk_cap, 0.25), risk_action="risk_off", reason=f"{base.reason}; consecutive_losses_risk_off")
        if base.risk_action == "risk_off" or base.risk_cap <= 0.0:
            return base
        cap = float(variant.cap_value if variant.cap_value is not None else risk_config.consecutive_loss_cap)
        return replace(base, risk_cap=min(base.risk_cap, cap), reason=f"{base.reason}; consecutive_losses_probation_cap_{cap:.2f}")
    adjusted_state = replace(risk_state, consecutive_losses=consecutive_losses)
    return supervise_risk(estimate, adjusted_state, long_decision, short_decision, config=risk_config)


def should_use_probation_cap(
    variant: VariantSpec,
    estimate: Any,
    state: VariantState,
    consecutive_losses: int,
    risk_config: RiskSupervisorConfig,
    feature_row: Any,
) -> bool:
    if consecutive_losses < risk_config.losses_no_new_entry:
        return False
    if estimate.long_regime not in variant.regimes:
        return False
    if variant.policy == "probation_cap":
        return True
    if variant.policy == "two_stage_probation":
        return (
            state.consecutive_loss_active_bars > int(variant.probation_bars or 72)
            and estimate.volatility_state != "extreme"
        )
    if variant.policy == "positive_recent_return":
        return has_positive_recent_return_or_momentum(feature_row)
    return False


def has_positive_recent_return_or_momentum(feature_row: Any) -> bool:
    return_1 = float(getattr(feature_row, "return_1", 0.0) or 0.0)
    momentum_short = float(getattr(feature_row, "momentum_short", 0.0) or 0.0)
    return return_1 > 0.0 or momentum_short > 0.0


def _risk_config_without_consecutive(config: RiskSupervisorConfig | None) -> RiskSupervisorConfig:
    base = config or RiskSupervisorConfig()
    return replace(base, enable_consecutive_loss_rules=False)


def with_keys(frame: pd.DataFrame, asset: str, dataset: str, fee_rate: float) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, "fee_rate", fee_rate)
    result.insert(0, "dataset", dataset)
    result.insert(0, "asset", asset)
    return result


def enrich_with_waterfall_fields(frame: pd.DataFrame) -> pd.DataFrame:
    """Add decision-waterfall diagnostics while preserving return/equity columns."""
    waterfall = add_waterfall_fields(frame)
    result = frame.copy()
    for column in waterfall.columns:
        if column not in result.columns:
            result[column] = waterfall[column].to_numpy()
    return result


def summarize_ablation_frame(frame: pd.DataFrame) -> dict[str, float | int]:
    metrics = summarize_frame(frame)
    target = pd.to_numeric(frame["target_position"], errors="coerce").fillna(0.0)
    executed = pd.to_numeric(frame["executed_position"], errors="coerce").fillna(0.0)
    metrics.update(
        {
            "average_target_position": float(target.mean()),
            "average_executed_position": float(executed.mean()),
            "target_to_executed_gap": float((target - executed).mean()),
            "no_new_entry_count": int((frame["risk_action"] == "no_new_entry").sum()),
            "consecutive_loss_active_count": int(frame["risk_reason"].astype(str).str.contains("consecutive_losses", regex=False).sum()),
            "target_gt_zero_executed_zero_count": int(((target > 0.0) & (executed == 0.0)).sum()),
        }
    )
    return metrics


def reference_rows(asset: str, dataset: str, data: pd.DataFrame, fee_rate: float) -> list[dict[str, Any]]:
    v1_result = run_backtest_fast(
        data,
        fee_rate=fee_rate,
        periods_per_year=PERIODS_PER_YEAR,
        progress_every=10000 if len(data) > 15000 else None,
    )
    v2_input = input_frame_from_v1(v1_result)
    v2_label = "v2.btc_final_candidate_A" if asset == "BTC" else "v2.final_candidate_A_cd120_on_ETH"
    v2_func = backtest_v2_btc_final_candidate_a if asset == "BTC" else backtest_v2_final_candidate_a
    frames = {
        v2_label: v2_func(v2_input, fee_rate=fee_rate, v1_entry_threshold=V1_ENTRY_THRESHOLD, cooldown_bars=120),
        "buy_and_hold": build_buy_and_hold_frame(data, fee_rate),
        "ma20_ma60": build_ma_crossover_frame(data, fee_rate),
    }
    rows = []
    for version, frame in frames.items():
        row = {"asset": asset, "dataset": dataset, "fee_rate": fee_rate, "version": version, **summarize_frame(frame)}
        row.update(
            {
                "average_target_position": np.nan,
                "average_executed_position": np.nan,
                "target_to_executed_gap": np.nan,
                "no_new_entry_count": np.nan,
                "consecutive_loss_active_count": np.nan,
                "target_gt_zero_executed_zero_count": np.nan,
            }
        )
        rows.append(row)
    return rows


def build_distribution_rows(frame: pd.DataFrame, variant: str) -> list[dict[str, Any]]:
    rows = []
    for diagnostic, column in [
        ("exposure_distribution", "executed_position"),
        ("risk_action_distribution", "risk_action"),
        ("binding_block_reason_distribution", "binding_block_reason"),
    ]:
        counts = frame[column].value_counts(dropna=False).sort_index()
        for bucket, count in counts.items():
            rows.append(
                {
                    "asset": frame["asset"].iloc[0],
                    "dataset": frame["dataset"].iloc[0],
                    "fee_rate": frame["fee_rate"].iloc[0],
                    "version": variant,
                    "diagnostic": diagnostic,
                    "bucket": bucket,
                    "count": int(count),
                    "percentage": float(count / max(len(frame), 1)),
                }
            )
    return rows


def strong_bull_summary(frame: pd.DataFrame) -> dict[str, Any]:
    strong = frame[frame["long_regime"] == "strong_bull"].copy()
    blocked = strong[(pd.to_numeric(strong["target_position"], errors="coerce") > 0.0) & (pd.to_numeric(strong["executed_position"], errors="coerce") == 0.0)]
    return {
        "strong_bull_bars": int(len(strong)),
        "strong_bull_no_new_entry_count": int((strong["risk_action"] == "no_new_entry").sum()) if len(strong) else 0,
        "strong_bull_average_executed_position": float(pd.to_numeric(strong["executed_position"], errors="coerce").mean()) if len(strong) else 0.0,
        "strong_bull_turnover": float(pd.to_numeric(strong["trade_amount"], errors="coerce").sum()) if len(strong) else 0.0,
        "strong_bull_fee_drag": float(pd.to_numeric(strong["fee_cost"], errors="coerce").sum()) if len(strong) else 0.0,
        "strong_bull_strategy_return_net": float(pd.to_numeric(strong["strategy_return_net"], errors="coerce").sum()) if len(strong) else 0.0,
        "strong_bull_blocked_count": int(len(blocked)),
        "strong_bull_blocked_next_24_bar_return": float(pd.to_numeric(blocked["next_24_bar_return"], errors="coerce").mean()) if len(blocked) else np.nan,
        "strong_bull_target_distribution": distribution_string(strong, "target_position"),
        "strong_bull_executed_distribution": distribution_string(strong, "executed_position"),
    }


def distribution_string(frame: pd.DataFrame, column: str) -> str:
    if frame.empty:
        return ""
    counts = frame[column].value_counts(dropna=False).sort_index()
    return "; ".join(f"{bucket}:{count}" for bucket, count in counts.items())


def write_report(summary: pd.DataFrame, diagnostics: pd.DataFrame, strong_bull: pd.DataFrame) -> None:
    variant_summary = summary[summary["version"].isin([variant.name for variant in VARIANTS])].copy()
    reference_summary = summary[~summary["version"].isin([variant.name for variant in VARIANTS])].copy()
    btc_variants = aggregate_variants(variant_summary[variant_summary["asset"] == "BTC"])
    eth_variants = aggregate_variants(variant_summary[variant_summary["asset"] == "ETH"])
    reference_table = aggregate_references(reference_summary)
    risk_table = aggregate_risk_controls(variant_summary)
    exposure_table = aggregate_exposure(variant_summary)
    strong_table = aggregate_strong_bull(strong_bull)
    btc_best_return = best_variant(btc_variants, "avg_total_return")
    btc_best_sharpe = best_variant(btc_variants, "avg_sharpe_ratio")
    btc_tradeoff = tradeoff_variant(btc_variants)
    btc_worst_tradeoff = worst_tradeoff_variant(btc_variants)
    lines = [
        "# v3 Risk-Action Semantics Experiment",
        "",
        "This experiment isolates how consecutive-loss state is translated into risk actions or capped probation exposure. Feature windows, estimator thresholds, long-term mapping, short-term rules, drawdown caps, volatility caps, fee-aware execution, cooldown behavior, no-leverage behavior, and v2 behavior are unchanged.",
        "",
        "## 1. Executive Summary",
        "",
        f"- Best BTC return variant: `{btc_best_return}`.",
        f"- Best BTC Sharpe variant: `{btc_best_sharpe}`.",
        f"- Best BTC drawdown/Sharpe tradeoff variant: `{btc_tradeoff}`.",
        "- This tests probation semantics after consecutive loss, not a blind disable/reset of the loss rule.",
        "- A useful variant must improve exposure capture without letting drawdown, turnover, or fee drag drift toward v2 levels.",
        f"- Worst BTC drawdown/Sharpe tradeoff variant: `{btc_worst_tradeoff}`.",
        "",
        "## 2. Why Previous Consecutive-Loss Ablation Was Not Enough",
        "",
        "The prior ablation confirmed that consecutive-loss logic is the main underexposure mechanism, but naive relaxation raised drawdown, turnover, and fee drag enough to damage BTC Sharpe. This run asks whether consecutive loss should become capped probation exposure in selected bullish regimes instead of a broad `no_new_entry` state.",
        "",
        "## 3. Variant Definitions",
        "",
        variant_definitions_markdown(),
        "",
        "## 4. BTC Comparison Table",
        "",
        _frame_to_markdown(btc_variants),
        "",
        "## 5. ETH Comparison Table",
        "",
        _frame_to_markdown(eth_variants),
        "",
        "## Reference Comparison Table",
        "",
        "These rows keep the requested reference systems visible beside the semantics variants. Reference rows do not have v3-only decision-waterfall fields.",
        "",
        _frame_to_markdown(reference_table),
        "",
        "## 6. Risk-Control Comparison",
        "",
        _frame_to_markdown(risk_table),
        "",
        "## 7. Exposure-Capture Comparison",
        "",
        _frame_to_markdown(exposure_table),
        "",
        "## 8. Strong-Bull Specific Comparison",
        "",
        _frame_to_markdown(strong_table),
        "",
        "## 9. Which Risk-Action Semantic Is Best",
        "",
        f"`{btc_best_return}` has the best BTC average total return, while `{btc_best_sharpe}` has the best BTC average Sharpe. The selection rule favors Sharpe and drawdown-controlled exposure capture over return alone.",
        "",
        "## 10. Should Any Variant Replace v3.final_candidate?",
        "",
        replacement_text(btc_variants, btc_tradeoff),
        "",
        "## 11. Is v3 Still Only An Architecture Checkpoint?",
        "",
        "Yes. This semantics experiment is diagnostic. Even if a probation variant improves exposure capture, v3 still needs a follow-up final-candidate run and rolling validation before replacing v2.",
        "",
        "## 12. Recommended Next Step",
        "",
        "Run a targeted follow-up only if one probation variant improves BTC Sharpe or keeps it nearly flat while materially increasing executed exposure. Otherwise, inspect whether the alpha side, not the Risk Supervisor, is too weak after costs.",
        "",
        "## Files",
        "",
        f"- Summary CSV: `{SUMMARY_CSV}`",
        f"- Diagnostics CSV: `{DIAGNOSTICS_CSV}`",
        f"- Strong-bull CSV: `{STRONG_BULL_CSV}`",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def aggregate_variants(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("version", dropna=False)
        .agg(
            rows=("dataset", "count"),
            avg_total_return=("total_return", "mean"),
            avg_annual_return=("annual_return", "mean"),
            avg_max_drawdown=("max_drawdown", "mean"),
            worst_max_drawdown=("max_drawdown", "min"),
            avg_sharpe_ratio=("sharpe_ratio", "mean"),
            avg_number_of_trades=("number_of_trades", "mean"),
            avg_turnover=("turnover", "mean"),
            avg_fee_drag=("fee_drag", "mean"),
            avg_average_target_position=("average_target_position", "mean"),
            avg_average_executed_position=("average_executed_position", "mean"),
            avg_target_to_executed_gap=("target_to_executed_gap", "mean"),
            avg_average_exposure=("average_exposure", "mean"),
            max_exposure=("max_exposure", "max"),
            avg_no_new_entry_count=("no_new_entry_count", "mean"),
            avg_consecutive_loss_active_count=("consecutive_loss_active_count", "mean"),
            avg_blocked_nonzero_target_count=("target_gt_zero_executed_zero_count", "mean"),
        )
        .reset_index()
        .sort_values("avg_sharpe_ratio", ascending=False)
    )


def aggregate_references(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    return (
        frame.groupby(["asset", "version"], dropna=False)
        .agg(
            rows=("dataset", "count"),
            avg_total_return=("total_return", "mean"),
            avg_annual_return=("annual_return", "mean"),
            avg_max_drawdown=("max_drawdown", "mean"),
            worst_max_drawdown=("max_drawdown", "min"),
            avg_sharpe_ratio=("sharpe_ratio", "mean"),
            avg_number_of_trades=("number_of_trades", "mean"),
            avg_turnover=("turnover", "mean"),
            avg_fee_drag=("fee_drag", "mean"),
            avg_average_exposure=("average_exposure", "mean"),
            max_exposure=("max_exposure", "max"),
        )
        .reset_index()
        .sort_values(["asset", "avg_sharpe_ratio"], ascending=[True, False])
    )


def aggregate_risk_controls(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["asset", "version"], dropna=False)
        .agg(
            avg_max_drawdown=("max_drawdown", "mean"),
            worst_max_drawdown=("max_drawdown", "min"),
            avg_turnover=("turnover", "mean"),
            avg_fee_drag=("fee_drag", "mean"),
            avg_no_new_entry_count=("no_new_entry_count", "mean"),
            avg_consecutive_loss_active_count=("consecutive_loss_active_count", "mean"),
        )
        .reset_index()
    )


def aggregate_exposure(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["asset", "version"], dropna=False)
        .agg(
            avg_target=("average_target_position", "mean"),
            avg_executed=("average_executed_position", "mean"),
            avg_gap=("target_to_executed_gap", "mean"),
            avg_exposure=("average_exposure", "mean"),
            max_exposure=("max_exposure", "max"),
            avg_blocked_nonzero_target_count=("target_gt_zero_executed_zero_count", "mean"),
        )
        .reset_index()
    )


def aggregate_strong_bull(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["asset", "version"], dropna=False)
        .agg(
            avg_strong_bull_bars=("strong_bull_bars", "mean"),
            avg_strong_bull_no_new_entry_count=("strong_bull_no_new_entry_count", "mean"),
            avg_strong_bull_executed_position=("strong_bull_average_executed_position", "mean"),
            avg_strong_bull_turnover=("strong_bull_turnover", "mean"),
            avg_strong_bull_fee_drag=("strong_bull_fee_drag", "mean"),
            avg_strong_bull_strategy_return=("strong_bull_strategy_return_net", "mean"),
            avg_strong_bull_blocked_count=("strong_bull_blocked_count", "mean"),
            avg_blocked_next_24_return=("strong_bull_blocked_next_24_bar_return", "mean"),
        )
        .reset_index()
    )


def best_variant(frame: pd.DataFrame, metric: str) -> str:
    variants = frame[frame["version"].astype(str).str.startswith(("A_", "B_", "C_", "D", "E", "F_"))]
    if variants.empty:
        return "unavailable"
    return str(variants.sort_values(metric, ascending=False).iloc[0]["version"])


def tradeoff_variant(frame: pd.DataFrame) -> str:
    variants = frame[frame["version"].astype(str).str.startswith(("A_", "B_", "C_", "D", "E", "F_"))].copy()
    if variants.empty:
        return "unavailable"
    variants["sharpe_rank"] = variants["avg_sharpe_ratio"].rank(ascending=False, method="min")
    variants["drawdown_rank"] = variants["avg_max_drawdown"].rank(ascending=False, method="min")
    variants["turnover_rank"] = variants["avg_turnover"].rank(ascending=True, method="min")
    variants["score"] = variants["sharpe_rank"] + variants["drawdown_rank"] + 0.5 * variants["turnover_rank"]
    return str(variants.sort_values("score").iloc[0]["version"])


def variant_definitions_markdown() -> str:
    definitions = [
        ("A_current_final_candidate", "Current behavior: consecutive-loss state becomes `no_new_entry` as in v3.final_candidate."),
        ("B_probation_cap_0p25_strong_bull_only", "In `strong_bull`, consecutive-loss `no_new_entry` becomes capped probation exposure at 0.25; other regimes keep current behavior."),
        ("C_probation_cap_0p25_bull_and_strong_bull", "In `strong_bull` and `bull`, consecutive-loss `no_new_entry` becomes capped probation exposure at 0.25; other regimes keep current behavior."),
        ("D_probation_cap_0p50_strong_bull_only", "In `strong_bull`, consecutive-loss `no_new_entry` becomes capped probation exposure at 0.50; other regimes keep current behavior."),
        ("E_two_stage_probation", "For the first 72 consecutive-loss bars, keep `no_new_entry`; after that, allow `strong_bull` probation exposure capped at 0.25 if volatility is not extreme."),
        ("F_probation_requires_positive_recent_return", "In `strong_bull`, allow capped 0.25 probation only when current return or short momentum is positive."),
    ]
    return "\n".join(f"- `{name}`: {description}" for name, description in definitions)


def worst_tradeoff_variant(frame: pd.DataFrame) -> str:
    variants = frame[frame["version"].astype(str).str.startswith(("A_", "B_", "C_", "D", "E", "F_"))].copy()
    if variants.empty:
        return "unavailable"
    variants["sharpe_rank"] = variants["avg_sharpe_ratio"].rank(ascending=False, method="min")
    variants["drawdown_rank"] = variants["avg_max_drawdown"].rank(ascending=False, method="min")
    variants["turnover_rank"] = variants["avg_turnover"].rank(ascending=True, method="min")
    variants["score"] = variants["sharpe_rank"] + variants["drawdown_rank"] + 0.5 * variants["turnover_rank"]
    return str(variants.sort_values("score", ascending=False).iloc[0]["version"])


def replacement_text(btc_variants: pd.DataFrame, tradeoff: str) -> str:
    baseline = btc_variants[btc_variants["version"] == "A_current_final_candidate"]
    candidate = btc_variants[btc_variants["version"] == tradeoff]
    if baseline.empty or candidate.empty or tradeoff == "A_current_final_candidate":
        return "No semantics variant clearly replaces the current final candidate yet."
    b = baseline.iloc[0]
    c = candidate.iloc[0]
    drawdown_not_much_worse = abs(float(c["avg_max_drawdown"])) <= abs(float(b["avg_max_drawdown"])) * 1.5
    if c["avg_sharpe_ratio"] > b["avg_sharpe_ratio"] and drawdown_not_much_worse:
        return f"`{tradeoff}` is a candidate for follow-up validation, but it should not replace v3.final_candidate until rolling validation confirms the drawdown remains controlled."
    return f"`{tradeoff}` improves part of the tradeoff, but the evidence is not sufficient to replace v3.final_candidate without a dedicated final-candidate rerun."


def _frame_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No data_"
    display = frame.copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(lambda value: f"{value:.6g}")
    columns = [str(column) for column in display.columns]
    rows = display.astype(object).where(pd.notna(display), "").astype(str).values.tolist()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator] + body)


if __name__ == "__main__":
    main()

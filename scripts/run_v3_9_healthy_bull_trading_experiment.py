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
from scripts.run_v3_6_strong_bull_deweight_validation import compare_versions, rolling_windows
from scripts.run_v3_8_estimator_label_redesign_audit import assign_trend_health_label, forward_compound_return
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
from src.v3.backtest_v3 import BacktestV3Config, run_v3_backtest
from src.v3.cooldown_manager import RegimeCooldownManagerV3, TradeCloseInfoV3
from src.v3.data_types import ShortTermDecisionV3
from src.v3.execution_layer import apply_execution, compute_strategy_return_net
from src.v3.feature_builder import build_feature_frame
from src.v3.long_term_controller import decide_long_term_position
from src.v3.market_estimator import estimate_market
from src.v3.position_composer import compose_target_position
from src.v3.risk_supervisor import PortfolioRiskStateV3, supervise_risk
from src.v3.short_term_controller import decide_short_term_adjustment
from v2_small_cap import backtest_v2_btc_final_candidate_a, backtest_v2_final_candidate_a


REPORT_PATH = Path("reports") / "v3_9_healthy_bull_trading_experiment.md"
SUMMARY_CSV = Path("reports") / "v3_9_healthy_bull_trading_experiment_summary.csv"
ROLLING_CSV = Path("reports") / "v3_9_healthy_bull_trading_experiment_rolling.csv"
DIAGNOSTICS_CSV = Path("reports") / "v3_9_healthy_bull_trading_experiment_diagnostics.csv"


@dataclass(frozen=True)
class VariantSpec:
    name: str
    policy: str


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

    def __post_init__(self) -> None:
        if self.recent_trade_amounts is None:
            self.recent_trade_amounts = []
        if self.cooldown is None:
            self.cooldown = RegimeCooldownManagerV3(cooldown_bars=120)


VARIANTS = (
    VariantSpec("A_current_v3_final_candidate", "current"),
    VariantSpec("B_healthy_bull_only_core", "healthy_bull_only"),
    VariantSpec("C_healthy_bull_core_with_strong_bull_deweight", "healthy_bull_with_deweight"),
    VariantSpec("D_healthy_bull_plus_oversold_probe_0p25", "healthy_bull_plus_oversold"),
    VariantSpec("E_healthy_bull_requires_v2_confirmation", "healthy_bull_v2_confirmed"),
    VariantSpec("G_healthy_bull_probation_cap_after_consecutive_loss", "healthy_bull_probation_cap"),
)

REFERENCE_V3_6 = "v3.6_strong_bull_deweight"
SKIPPED_VARIANTS = {"F_healthy_bull_without_consecutive_loss_override": "Duplicate of B_healthy_bull_only_core because B already preserves the existing consecutive-loss no_new_entry behavior."}


def main() -> None:
    config = load_final_candidate_config()
    fee_rates = tuple(float(value) for value in config["execution"]["fee_rates_to_validate"])
    summary_rows: list[dict[str, Any]] = []
    rolling_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []

    for asset, dataset_key in [("BTC", "btc_datasets"), ("ETH", "eth_datasets")]:
        paths = [Path(path) for path in config["validation"].get(dataset_key, []) if Path(path).exists()]
        for path in paths:
            dataset = path.stem
            print(f"asset={asset} dataset={dataset}")
            data = load_ohlcv_csv(path)
            for fee_rate in fee_rates:
                print(f"  fee={fee_rate:g}")
                backtest_config = build_backtest_config(config, fee_rate=fee_rate)
                v2_frame = run_v2_reference(asset, data, fee_rate)
                variant_frames = run_variants_for_dataset(data, backtest_config, v2_frame)
                reference_frames = build_reference_frames(asset, data, backtest_config, v2_frame, fee_rate)
                all_frames = {**variant_frames, **reference_frames}
                for version, frame in all_frames.items():
                    keyed = with_keys(add_forward_returns(enrich_with_waterfall_fields(frame)), asset, dataset, fee_rate)
                    summary_rows.append({"asset": asset, "dataset": dataset, "fee_rate": fee_rate, "version": version, **summarize_experiment_frame(keyed)})
                    diagnostic_rows.extend(build_diagnostic_rows(keyed, version))
                    for window_name, window in rolling_windows(keyed):
                        rolling_rows.append({"asset": asset, "dataset": dataset, "fee_rate": fee_rate, "version": version, "window": window_name, **summarize_experiment_frame(window)})

    summary = pd.DataFrame(summary_rows)
    rolling = pd.DataFrame(rolling_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)
    rolling.to_csv(ROLLING_CSV, index=False)
    diagnostics.to_csv(DIAGNOSTICS_CSV, index=False)
    write_report(summary, rolling, diagnostics)
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {ROLLING_CSV}")
    print(f"Wrote {DIAGNOSTICS_CSV}")


def run_v2_reference(asset: str, data: pd.DataFrame, fee_rate: float) -> pd.DataFrame:
    v1_result = run_backtest_fast(
        data,
        fee_rate=fee_rate,
        periods_per_year=PERIODS_PER_YEAR,
        progress_every=10000 if len(data) > 15000 else None,
    )
    v2_input = input_frame_from_v1(v1_result)
    v2_func = backtest_v2_btc_final_candidate_a if asset == "BTC" else backtest_v2_final_candidate_a
    return v2_func(v2_input, fee_rate=fee_rate, v1_entry_threshold=V1_ENTRY_THRESHOLD, cooldown_bars=120).reset_index(drop=True)


def build_reference_frames(asset: str, data: pd.DataFrame, config: BacktestV3Config, v2_frame: pd.DataFrame, fee_rate: float) -> dict[str, pd.DataFrame]:
    v2_label = "v2.btc_final_candidate_A" if asset == "BTC" else "v2.final_candidate_A_cd120_on_ETH"
    return {
        REFERENCE_V3_6: run_v3_backtest(data, config=strong_bull_deweight_config(config)),
        v2_label: v2_frame,
        "buy_and_hold": build_buy_and_hold_frame(data, fee_rate),
        "ma20_ma60": build_ma_crossover_frame(data, fee_rate),
    }


def strong_bull_deweight_config(config: BacktestV3Config) -> BacktestV3Config:
    long_config = replace(config.long_term_config, base_positions={**config.long_term_config.base_positions, "strong_bull": 0.50})
    return replace(config, long_term_config=long_config)


def run_variants_for_dataset(data: pd.DataFrame, config: BacktestV3Config, v2_frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    features = build_feature_frame(data, config=config.feature_config)
    v2_position = pd.to_numeric(v2_frame.get("final_position", pd.Series(0.0, index=v2_frame.index)), errors="coerce").fillna(0.0).reset_index(drop=True)
    states = {variant.name: VariantState(cooldown=RegimeCooldownManagerV3(cooldown_bars=config.cooldown_bars)) for variant in VARIANTS}
    rows = {variant.name: [] for variant in VARIANTS}

    for idx, feature_row in enumerate(features.itertuples(index=False)):
        estimate = estimate_market(pd.Series(feature_row._asdict()), config=config.estimator_config)
        base_long_decision = decide_long_term_position(estimate, config=config.long_term_config)
        base_short_decision = decide_short_term_adjustment(estimate, cooldown_state=False, config=config.short_term_config)
        v2_pos = float(v2_position.iloc[idx]) if idx < len(v2_position) else 0.0

        for variant in VARIANTS:
            state = states[variant.name]
            assert state.recent_trade_amounts is not None
            assert state.cooldown is not None
            asset_return = float(feature_row.return_1)
            portfolio_drawdown = state.equity / state.equity_peak - 1.0
            recent_turnover = sum(state.recent_trade_amounts[-config.recent_turnover_window :])
            cooldown_active = state.cooldown.is_active(estimate.long_regime)

            baseline_short_decision = decide_short_term_adjustment(estimate, cooldown_state=cooldown_active, config=config.short_term_config)
            baseline_risk = supervise_risk(
                estimate,
                PortfolioRiskStateV3(portfolio_drawdown, 0.0, state.consecutive_losses, state.current_position, recent_turnover, 0.0),
                base_long_decision,
                baseline_short_decision,
                config=config.risk_config,
            )
            baseline_composed = compose_target_position(base_long_decision, baseline_short_decision, baseline_risk, config=config.composer_config)
            label = diagnostic_label(estimate, baseline_composed.target_position, v2_pos)

            long_decision, short_decision = variant_decisions(variant, base_long_decision, base_short_decision, estimate, label, v2_pos)
            risk_decision = supervise_risk(
                estimate,
                PortfolioRiskStateV3(portfolio_drawdown, 0.0, state.consecutive_losses, state.current_position, recent_turnover, 0.0),
                long_decision,
                short_decision,
                config=config.risk_config,
            )
            composed = compose_target_position(long_decision, short_decision, risk_decision, config=config.composer_config)
            adjusted, override_reason, execution_risk_action = apply_variant_override(
                variant,
                composed,
                estimate,
                risk_decision,
                label=label,
                consecutive_losses=state.consecutive_losses,
            )
            executed = apply_execution(
                adjusted,
                current_position=state.current_position,
                risk_action=execution_risk_action,
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

            raw_target_position = float(long_decision.base_position) + float(short_decision.position_adjustment)
            consecutive_loss_block = "consecutive_losses" in str(risk_decision.reason) and str(risk_decision.risk_action) == "no_new_entry"
            healthy_blocked = bool(label == "healthy_bull" and composed.target_position > 0.0 and executed_position == 0.0)
            rows[variant.name].append(
                {
                    "timestamp": feature_row.timestamp,
                    "close": float(feature_row.close),
                    "asset_return": asset_return,
                    "long_regime": estimate.long_regime,
                    "short_regime": estimate.short_regime,
                    "volatility_state": estimate.volatility_state,
                    "drawdown_state": estimate.drawdown_state,
                    "risk_state": estimate.risk_state,
                    "confidence_score": estimate.confidence_score,
                    "allow_entry": estimate.allow_entry,
                    "allow_hold": estimate.allow_hold,
                    "trend_health_label": label,
                    "base_position": long_decision.base_position,
                    "position_adjustment": short_decision.position_adjustment,
                    "risk_cap": risk_decision.risk_cap,
                    "raw_target_position": raw_target_position,
                    "risk_limited_position": min(raw_target_position, float(risk_decision.risk_cap)),
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
                    "execution_risk_action": execution_risk_action,
                    "risk_reason": risk_decision.reason,
                    "execution_reason": executed.execution_reason,
                    "variant_override_reason": override_reason,
                    "v2_position": v2_pos,
                    "consecutive_losses": state.consecutive_losses,
                    "consecutive_loss_block": consecutive_loss_block,
                    "healthy_bull_blocked": healthy_blocked,
                    "cooldown_active": cooldown_active,
                    "cooldown_triggered": cooldown_triggered,
                }
            )
            state.current_position = executed_position

    return {variant: pd.DataFrame(records) for variant, records in rows.items()}


def diagnostic_label(estimate: Any, baseline_target: float, v2_position: float) -> str:
    return assign_trend_health_label(
        pd.Series(
            {
                "long_regime": estimate.long_regime,
                "short_regime": estimate.short_regime,
                "volatility_state": estimate.volatility_state,
                "drawdown_state": estimate.drawdown_state,
                "confidence_score": estimate.confidence_score,
                "target_position": baseline_target,
                "v2_position": v2_position,
            }
        )
    )


def variant_decisions(variant: VariantSpec, long_decision: Any, short_decision: ShortTermDecisionV3, estimate: Any, label: str, v2_position: float) -> tuple[Any, ShortTermDecisionV3]:
    if variant.policy == "current":
        return long_decision, short_decision
    base = label_base_position(variant, label, v2_position)
    mapped_long = replace(long_decision, base_position=base, reason=f"v3.9_{variant.policy}_{label}_base_{base:g}")
    zero_short = replace(short_decision, position_adjustment=0.0, reason=f"{short_decision.reason}; v3.9_label_mapping_disables_short_adjustment")
    return mapped_long, zero_short


def label_base_position(variant: VariantSpec, label: str, v2_position: float) -> float:
    if variant.policy in {"healthy_bull_only", "healthy_bull_probation_cap"}:
        return 0.50 if label == "healthy_bull" else 0.0
    if variant.policy == "healthy_bull_v2_confirmed":
        return 0.50 if label == "healthy_bull" and v2_position > 0.0 else 0.0
    mapping = {
        "healthy_bull": 0.50,
        "late_bull": 0.25,
        "weak_bull": 0.25,
        "overextended_bull": 0.0,
        "neutral_range": 0.0,
        "true_bear": 0.0,
        "true_strong_bear": 0.0,
        "early_recovery_candidate": 0.0,
        "oversold_rebound_candidate": 0.25 if variant.policy == "healthy_bull_plus_oversold" else 0.0,
    }
    return mapping.get(label, 0.0)


def apply_variant_override(variant: VariantSpec, composed: Any, estimate: Any, risk_decision: Any, *, label: str, consecutive_losses: int) -> tuple[Any, str, str]:
    if variant.policy != "healthy_bull_probation_cap":
        return composed, "none", str(risk_decision.risk_action)
    hard_block = str(risk_decision.risk_action) == "risk_off" or float(risk_decision.risk_cap) <= 0.0 or estimate.volatility_state == "extreme"
    loss_block = str(risk_decision.risk_action) == "no_new_entry" and "consecutive_losses" in str(risk_decision.reason)
    if label == "healthy_bull" and loss_block and not hard_block and consecutive_losses >= 2:
        return replace(composed, target_position=min(0.25, float(risk_decision.risk_cap))), "healthy_bull_consecutive_loss_probation_cap_0p25", "normal"
    return composed, "none", str(risk_decision.risk_action)


def add_forward_returns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "asset_return" in result.columns:
        returns = pd.to_numeric(result["asset_return"], errors="coerce").fillna(0.0)
        result["next_24_bar_return"] = forward_compound_return(returns, 24)
    return result


def enrich_with_waterfall_fields(frame: pd.DataFrame) -> pd.DataFrame:
    try:
        return add_waterfall_fields(frame)
    except Exception:
        result = frame.copy()
        result["binding_block_reason"] = "unavailable"
        return result


def with_keys(frame: pd.DataFrame, asset: str, dataset: str, fee_rate: float) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, "asset", asset)
    result.insert(1, "dataset", dataset)
    result.insert(2, "fee_rate", fee_rate)
    return result


def summarize_experiment_frame(frame: pd.DataFrame) -> dict[str, Any]:
    metrics = summarize_frame(frame)
    target = pd.to_numeric(frame.get("target_position", pd.Series(np.nan, index=frame.index)), errors="coerce")
    executed = pd.to_numeric(frame.get("executed_position", frame.get("final_position", pd.Series(0.0, index=frame.index))), errors="coerce").fillna(0.0)
    metrics.update(
        {
            "average_target_position": float(target.fillna(0.0).mean()) if "target_position" in frame.columns else np.nan,
            "average_executed_position": float(executed.mean()),
            "target_to_executed_gap": float((target.fillna(0.0) - executed).mean()) if "target_position" in frame.columns else np.nan,
            "exposure_distribution": distribution_string(executed),
            "risk_action_distribution": distribution_string(frame["risk_action"].astype(str)) if "risk_action" in frame.columns else "",
            "binding_block_reason_distribution": distribution_string(frame["binding_block_reason"].astype(str)) if "binding_block_reason" in frame.columns else "",
            "healthy_bull_exposure": label_exposure(frame, "healthy_bull"),
            "healthy_bull_contribution": label_contribution(frame, "healthy_bull"),
            "late_bull_exposure": label_exposure(frame, "late_bull"),
            "late_bull_contribution": label_contribution(frame, "late_bull"),
            "overextended_bull_exposure": label_exposure(frame, "overextended_bull"),
            "overextended_bull_contribution": label_contribution(frame, "overextended_bull"),
            "oversold_rebound_exposure": label_exposure(frame, "oversold_rebound_candidate"),
            "oversold_rebound_contribution": label_contribution(frame, "oversold_rebound_candidate"),
            "consecutive_loss_block_count": int(pd.Series(frame.get("consecutive_loss_block", False)).fillna(False).sum()),
            "healthy_bull_blocked_count": int(pd.Series(frame.get("healthy_bull_blocked", False)).fillna(False).sum()),
            "healthy_bull_executed_count": int(((frame.get("trend_health_label", pd.Series("", index=frame.index)) == "healthy_bull") & (executed > 0.0)).sum()) if "trend_health_label" in frame.columns else 0,
            "blocked_healthy_bull_avg_next_24": blocked_forward_mean(frame, "healthy_bull"),
        }
    )
    return metrics


def label_exposure(frame: pd.DataFrame, label: str) -> float:
    if "trend_health_label" not in frame.columns or "executed_position" not in frame.columns:
        return np.nan
    subset = frame[frame["trend_health_label"] == label]
    return mean(subset["executed_position"]) if len(subset) else 0.0


def label_contribution(frame: pd.DataFrame, label: str) -> float:
    if "trend_health_label" not in frame.columns or "strategy_return_net" not in frame.columns:
        return np.nan
    subset = frame[frame["trend_health_label"] == label]
    return float(pd.to_numeric(subset["strategy_return_net"], errors="coerce").fillna(0.0).sum())


def blocked_forward_mean(frame: pd.DataFrame, label: str) -> float:
    if "next_24_bar_return" not in frame.columns or "healthy_bull_blocked" not in frame.columns:
        return np.nan
    subset = frame[(frame["trend_health_label"] == label) & (frame["healthy_bull_blocked"])]
    return mean(subset["next_24_bar_return"]) if len(subset) else np.nan


def build_diagnostic_rows(frame: pd.DataFrame, version: str) -> list[dict[str, Any]]:
    rows = []
    for diagnostic, column in [
        ("exposure_distribution", "executed_position"),
        ("risk_action_distribution", "risk_action"),
        ("binding_block_reason_distribution", "binding_block_reason"),
        ("trend_health_label_distribution", "trend_health_label"),
        ("variant_override_reason_distribution", "variant_override_reason"),
    ]:
        if column not in frame.columns:
            continue
        counts = frame[column].value_counts(dropna=False).sort_index()
        for bucket, count in counts.items():
            rows.append({"asset": frame["asset"].iloc[0], "dataset": frame["dataset"].iloc[0], "fee_rate": frame["fee_rate"].iloc[0], "version": version, "diagnostic": diagnostic, "bucket": bucket, "count": int(count), "percentage": float(count / max(len(frame), 1))})
    return rows


def aggregate_versions(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    return frame.groupby("version", dropna=False).agg(
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
        avg_healthy_bull_exposure=("healthy_bull_exposure", "mean"),
        avg_healthy_bull_contribution=("healthy_bull_contribution", "mean"),
        avg_late_bull_exposure=("late_bull_exposure", "mean"),
        avg_late_bull_contribution=("late_bull_contribution", "mean"),
        avg_overextended_bull_exposure=("overextended_bull_exposure", "mean"),
        avg_overextended_bull_contribution=("overextended_bull_contribution", "mean"),
        avg_oversold_rebound_exposure=("oversold_rebound_exposure", "mean"),
        avg_oversold_rebound_contribution=("oversold_rebound_contribution", "mean"),
        avg_consecutive_loss_block_count=("consecutive_loss_block_count", "mean"),
        avg_healthy_bull_blocked_count=("healthy_bull_blocked_count", "mean"),
        avg_healthy_bull_executed_count=("healthy_bull_executed_count", "mean"),
        avg_blocked_healthy_bull_next_24=("blocked_healthy_bull_avg_next_24", "mean"),
    ).reset_index().sort_values("avg_sharpe_ratio", ascending=False)


def aggregate_reference_versions(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    return frame.groupby(["asset", "version"], dropna=False).agg(
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
    ).reset_index().sort_values(["asset", "avg_sharpe_ratio"], ascending=[True, False])


def rolling_comparison_table(rolling: pd.DataFrame, asset: str, baseline: str) -> pd.DataFrame:
    candidates = [variant.name for variant in VARIANTS if variant.name != baseline]
    asset_rolling = rolling[rolling["asset"] == asset]
    rows = []
    for candidate in candidates:
        comparison = compare_versions(asset_rolling, candidate, baseline)
        if not comparison.empty:
            rows.append(comparison)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def write_report(summary: pd.DataFrame, rolling: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    variant_names = [variant.name for variant in VARIANTS] + [REFERENCE_V3_6]
    btc = aggregate_versions(summary[(summary["asset"] == "BTC") & (summary["version"].isin(variant_names))])
    eth = aggregate_versions(summary[(summary["asset"] == "ETH") & (summary["version"].isin(variant_names))])
    refs = aggregate_reference_versions(summary[~summary["version"].isin(variant_names)])
    btc_roll_v3 = rolling_comparison_table(rolling, "BTC", "A_current_v3_final_candidate")
    btc_roll_g = rolling_comparison_table(rolling, "BTC", REFERENCE_V3_6)
    eth_roll_v3 = rolling_comparison_table(rolling, "ETH", "A_current_v3_final_candidate")
    eth_roll_g = rolling_comparison_table(rolling, "ETH", REFERENCE_V3_6)
    best_return = best_variant(btc, "avg_total_return")
    best_sharpe = best_variant(btc, "avg_sharpe_ratio")
    tradeoff = tradeoff_variant(btc)
    lines = [
        "# v3.9 Healthy-Bull Trading Experiment",
        "",
        "This is a narrow isolated experiment. v2 behavior, v3.final_candidate behavior, particle filter status, leverage settings, and Risk Supervisor configuration are unchanged.",
        "",
        "## 1. Executive Summary",
        "",
        f"- Best BTC return variant: `{best_return}`.",
        f"- Best BTC Sharpe variant: `{best_sharpe}`.",
        f"- Best BTC drawdown/Sharpe tradeoff: `{tradeoff}`.",
        replacement_text(btc, btc_roll_v3, tradeoff),
        "",
        "## 2. Why Healthy-Bull Experiment Is Needed",
        "",
        "v3.8 found `healthy_bull` to be the cleanest diagnostic label while old `strong_bull` split into late/overextended states. v3.9 tests whether that label quality survives actual fee-aware execution and Risk Supervisor constraints.",
        "",
        "## 3. Variant Definitions",
        "",
        variant_definitions(),
        "",
        "## 4. BTC Full-Period Comparison",
        "",
        frame_to_markdown(btc),
        "",
        "## 5. BTC Rolling Comparison vs v3.final_candidate",
        "",
        frame_to_markdown(btc_roll_v3),
        "",
        "## BTC Rolling Comparison vs v3.6_strong_bull_deweight",
        "",
        frame_to_markdown(btc_roll_g),
        "",
        "## 6. ETH Full-Period Comparison",
        "",
        frame_to_markdown(eth),
        "",
        "## 7. ETH Rolling Comparison vs v3.final_candidate",
        "",
        frame_to_markdown(eth_roll_v3),
        "",
        "## ETH Rolling Comparison vs v3.6_strong_bull_deweight",
        "",
        frame_to_markdown(eth_roll_g),
        "",
        "## Reference Baselines",
        "",
        frame_to_markdown(refs),
        "",
        "## 8. Healthy-Bull Execution Diagnostics",
        "",
        healthy_bull_text(btc),
        "",
        "## 9. Late / Overextended Bull Suppression Diagnostics",
        "",
        suppression_text(btc),
        "",
        "## 10. Oversold Rebound Probe Diagnostics",
        "",
        oversold_text(btc, eth),
        "",
        "## 11. Best Variant By BTC Return",
        "",
        f"`{best_return}`.",
        "",
        "## 12. Best Variant By BTC Sharpe",
        "",
        f"`{best_sharpe}`.",
        "",
        "## 13. Best Drawdown / Sharpe Tradeoff",
        "",
        f"`{tradeoff}`.",
        "",
        "## 14. Should Any Variant Replace v3.final_candidate?",
        "",
        replacement_text(btc, btc_roll_v3, tradeoff),
        "",
        "## 15. Should v3.9 Become A Candidate Or Remain Diagnostic?",
        "",
        candidate_text(btc, btc_roll_v3, tradeoff),
        "",
        "## 16. Recommended Next Step",
        "",
        next_step_text(best_sharpe, tradeoff),
        "",
        "## Skipped Duplicate Variant",
        "",
        "\n".join(f"- `{name}`: {reason}" for name, reason in SKIPPED_VARIANTS.items()),
        "",
        "## Files",
        "",
        f"- Summary CSV: `{SUMMARY_CSV}`",
        f"- Rolling CSV: `{ROLLING_CSV}`",
        f"- Diagnostics CSV: `{DIAGNOSTICS_CSV}`",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def variant_definitions() -> str:
    return "\n".join(
        [
            "- `A_current_v3_final_candidate`: existing v3 final candidate.",
            "- `B_healthy_bull_only_core`: only healthy_bull receives base 0.50; all other labels base 0.",
            "- `C_healthy_bull_core_with_strong_bull_deweight`: healthy_bull 0.50, late_bull/weak_bull 0.25, overextended/neutral/bear labels 0.",
            "- `D_healthy_bull_plus_oversold_probe_0p25`: C plus oversold_rebound_candidate 0.25.",
            "- `E_healthy_bull_requires_v2_confirmation`: healthy_bull 0.50 only when v2_position is long.",
            "- `G_healthy_bull_probation_cap_after_consecutive_loss`: B plus healthy_bull can enter with 0.25 cap when consecutive-loss alone would block no_new_entry.",
        ]
    )


def healthy_bull_text(btc: pd.DataFrame) -> str:
    cols = ["version", "avg_healthy_bull_exposure", "avg_healthy_bull_contribution", "avg_healthy_bull_executed_count", "avg_healthy_bull_blocked_count", "avg_blocked_healthy_bull_next_24"]
    return frame_to_markdown(btc[[c for c in cols if c in btc.columns]])


def suppression_text(btc: pd.DataFrame) -> str:
    cols = ["version", "avg_late_bull_exposure", "avg_late_bull_contribution", "avg_overextended_bull_exposure", "avg_overextended_bull_contribution", "avg_max_drawdown", "avg_sharpe_ratio"]
    return frame_to_markdown(btc[[c for c in cols if c in btc.columns]])


def oversold_text(btc: pd.DataFrame, eth: pd.DataFrame) -> str:
    btc_d = btc[["version", "avg_oversold_rebound_exposure", "avg_oversold_rebound_contribution"]].copy()
    btc_d.insert(0, "asset", "BTC")
    eth_d = eth[["version", "avg_oversold_rebound_exposure", "avg_oversold_rebound_contribution"]].copy()
    eth_d.insert(0, "asset", "ETH")
    return frame_to_markdown(pd.concat([btc_d, eth_d], ignore_index=True))


def replacement_text(btc: pd.DataFrame, btc_roll: pd.DataFrame, tradeoff: str) -> str:
    baseline = btc[btc["version"] == "A_current_v3_final_candidate"]
    candidate = btc[btc["version"] == tradeoff]
    if baseline.empty or candidate.empty or tradeoff == "A_current_v3_final_candidate":
        return "No v3.9 variant clearly replaces v3.final_candidate."
    sharpe_rate = metric_rate(btc_roll, tradeoff, "sharpe_ratio", "win_rate")
    changed_rate = metric_rate(btc_roll, tradeoff, "sharpe_ratio", "changed_rate")
    c = candidate.iloc[0]
    b = baseline.iloc[0]
    if c["avg_sharpe_ratio"] > b["avg_sharpe_ratio"] and c["avg_max_drawdown"] >= b["avg_max_drawdown"] and sharpe_rate >= 0.25 and changed_rate >= 0.10:
        return f"`{tradeoff}` is eligible for deeper robustness review, but not automatic promotion."
    return f"`{tradeoff}` does not satisfy the v3.9 replacement criteria."


def candidate_text(btc: pd.DataFrame, btc_roll: pd.DataFrame, tradeoff: str) -> str:
    return replacement_text(btc, btc_roll, tradeoff) + " Treat v3.9 as diagnostic unless a later focused robustness review says otherwise."


def next_step_text(best_sharpe: str, tradeoff: str) -> str:
    if best_sharpe in {"B_healthy_bull_only_core", "C_healthy_bull_core_with_strong_bull_deweight", "G_healthy_bull_probation_cap_after_consecutive_loss"}:
        return f"Run a narrower robustness review on `{best_sharpe}` only if rolling changed-window evidence is broad enough; otherwise return to estimator design and v2 alpha extraction."
    return "Do not expand v3.9. Use the result to refine estimator labels and continue v2 alpha extraction as the stronger BTC path."


def best_variant(frame: pd.DataFrame, metric: str) -> str:
    if frame.empty:
        return "unavailable"
    return str(frame.sort_values(metric, ascending=False).iloc[0]["version"])


def tradeoff_variant(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "unavailable"
    variants = frame.copy()
    variants["sharpe_rank"] = variants["avg_sharpe_ratio"].rank(ascending=False, method="min")
    variants["drawdown_rank"] = variants["avg_max_drawdown"].rank(ascending=False, method="min")
    variants["turnover_rank"] = variants["avg_turnover"].rank(ascending=True, method="min")
    variants["score"] = variants["sharpe_rank"] + variants["drawdown_rank"] + 0.5 * variants["turnover_rank"]
    return str(variants.sort_values("score").iloc[0]["version"])


def metric_rate(table: pd.DataFrame, candidate: str, metric: str, column: str) -> float:
    row = table[(table["candidate"] == candidate) & (table["metric"] == metric)]
    return float(row.iloc[0][column]) if not row.empty else np.nan


def distribution_string(values: pd.Series) -> str:
    counts = values.value_counts(dropna=False).sort_index()
    return "; ".join(f"{bucket}:{count}" for bucket, count in counts.items())


def mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else np.nan


def frame_to_markdown(frame: pd.DataFrame) -> str:
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

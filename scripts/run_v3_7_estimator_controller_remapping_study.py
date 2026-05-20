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
from scripts.run_v3_6_strong_bull_deweight_validation import compare_versions, rolling_windows
from src.v3.backtest_v3 import BacktestV3Config
from src.v3.cooldown_manager import RegimeCooldownManagerV3, TradeCloseInfoV3
from src.v3.execution_layer import apply_execution, compute_strategy_return_net
from src.v3.feature_builder import build_feature_frame
from src.v3.long_term_controller import decide_long_term_position
from src.v3.market_estimator import estimate_market
from src.v3.position_composer import compose_target_position
from src.v3.risk_supervisor import PortfolioRiskStateV3, supervise_risk
from src.v3.short_term_controller import decide_short_term_adjustment
from v2_small_cap import backtest_v2_btc_final_candidate_a, backtest_v2_final_candidate_a


REPORT_PATH = Path("reports") / "v3_7_estimator_controller_remapping_study.md"
SUMMARY_CSV = Path("reports") / "v3_7_estimator_controller_remapping_summary.csv"
ROLLING_CSV = Path("reports") / "v3_7_estimator_controller_remapping_rolling.csv"
AUDIT_CSV = Path("reports") / "v3_7_estimator_controller_remapping_audit.csv"
DIAGNOSTICS_CSV = Path("reports") / "v3_7_estimator_controller_remapping_diagnostics.csv"


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
    VariantSpec("B_strong_bull_deweight", "strong_bull_deweight"),
    VariantSpec("C_strong_bull_to_bull_mapping", "strong_bull_to_bull"),
    VariantSpec("D1_strong_bull_overheat_pullback_cap_0p25", "strong_bull_overheat_pullback_cap_0p25"),
    VariantSpec("D2_strong_bull_overheat_pullback_cap_0p50", "strong_bull_overheat_pullback_cap_0p50"),
    VariantSpec("E_bull_noise_core_only", "bull_noise_core_only"),
    VariantSpec("F_bull_noise_v2_confirmed_core", "bull_noise_v2_confirmed_core"),
    VariantSpec("G_neutral_recovery_watchlist_not_trade", "neutral_recovery_watchlist"),
    VariantSpec("H_revised_mapping_conservative", "revised_mapping_conservative"),
    VariantSpec("I_bull_noise_plus_deweighted_strong_bull", "bull_noise_plus_deweighted_strong_bull"),
)


def main() -> None:
    config = load_final_candidate_config()
    fee_rates = tuple(float(value) for value in config["execution"]["fee_rates_to_validate"])
    summary_rows: list[dict[str, Any]] = []
    rolling_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
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
                for variant, frame in variant_frames.items():
                    keyed = with_keys(add_forward_returns(enrich_with_waterfall_fields(frame)), asset, dataset, fee_rate)
                    summary_rows.append({"asset": asset, "dataset": dataset, "fee_rate": fee_rate, "version": variant, **summarize_variant_frame(keyed)})
                    diagnostic_rows.extend(build_distribution_rows(keyed, variant))
                    for window_name, window in rolling_windows(keyed):
                        rolling_rows.append({"asset": asset, "dataset": dataset, "fee_rate": fee_rate, "version": variant, "window": window_name, **summarize_variant_frame(window)})
                    if variant in {"A_current_v3_final_candidate", "B_strong_bull_deweight"}:
                        audit_rows.extend(build_regime_audit_rows(keyed, variant))
                summary_rows.extend(reference_rows(asset, dataset, data, fee_rate, v2_frame))

    summary = pd.DataFrame(summary_rows)
    rolling = pd.DataFrame(rolling_rows)
    audits = pd.DataFrame(audit_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)
    rolling.to_csv(ROLLING_CSV, index=False)
    audits.to_csv(AUDIT_CSV, index=False)
    diagnostics.to_csv(DIAGNOSTICS_CSV, index=False)
    write_report(summary, rolling, audits, diagnostics)
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {ROLLING_CSV}")
    print(f"Wrote {AUDIT_CSV}")
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


def run_variants_for_dataset(data: pd.DataFrame, config: BacktestV3Config, v2_frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    features = build_feature_frame(data, config=config.feature_config)
    v2_position = pd.to_numeric(v2_frame.get("final_position", pd.Series(0.0, index=v2_frame.index)), errors="coerce").fillna(0.0).reset_index(drop=True)
    states = {variant.name: VariantState(cooldown=RegimeCooldownManagerV3(cooldown_bars=config.cooldown_bars)) for variant in VARIANTS}
    rows = {variant.name: [] for variant in VARIANTS}

    for idx, feature_row in enumerate(features.itertuples(index=False)):
        estimate = estimate_market(pd.Series(feature_row._asdict()), config=config.estimator_config)
        base_long_decision = decide_long_term_position(estimate, config=config.long_term_config)
        v2_pos = float(v2_position.iloc[idx]) if idx < len(v2_position) else 0.0

        for variant in VARIANTS:
            state = states[variant.name]
            assert state.recent_trade_amounts is not None
            assert state.cooldown is not None
            asset_return = float(feature_row.return_1)
            portfolio_drawdown = state.equity / state.equity_peak - 1.0
            recent_turnover = sum(state.recent_trade_amounts[-config.recent_turnover_window :])
            cooldown_active = state.cooldown.is_active(estimate.long_regime)
            long_decision = variant_long_decision(variant, base_long_decision, estimate)
            short_decision = decide_short_term_adjustment(estimate, cooldown_state=cooldown_active, config=config.short_term_config)
            risk_decision = supervise_risk(
                estimate,
                PortfolioRiskStateV3(
                    portfolio_drawdown=portfolio_drawdown,
                    realized_volatility=0.0,
                    consecutive_losses=state.consecutive_losses,
                    current_position=state.current_position,
                    recent_turnover=recent_turnover,
                    fee_drag=0.0,
                ),
                long_decision,
                short_decision,
                config=config.risk_config,
            )
            composed = compose_target_position(long_decision, short_decision, risk_decision, config=config.composer_config)
            normal_target_position = composed.target_position
            adjusted, override_reason, execution_risk_action = apply_variant_target_override(
                variant,
                composed,
                estimate,
                risk_decision,
                current_position=state.current_position,
                v2_position=v2_pos,
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
                    "risk_cap": risk_decision.risk_cap,
                    "raw_target_position": raw_target_position,
                    "risk_limited_position": risk_limited_position,
                    "normal_target_position": normal_target_position,
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
                    "early_recovery_watchlist": is_early_recovery_watchlist(estimate, v2_pos),
                    "v2_position": v2_pos,
                    "cooldown_active": cooldown_active,
                    "cooldown_triggered": cooldown_triggered,
                }
            )
            state.current_position = executed_position

    return {variant: pd.DataFrame(records) for variant, records in rows.items()}


def variant_long_decision(variant: VariantSpec, long_decision: Any, estimate: Any) -> Any:
    if variant.policy in {"strong_bull_deweight", "strong_bull_to_bull"} and estimate.long_regime == "strong_bull":
        return replace(long_decision, base_position=0.50, reason=f"{long_decision.reason}; experiment_strong_bull_to_0p50")
    if variant.policy == "revised_mapping_conservative":
        mapping = {"strong_bull": 0.25, "bull": 0.50, "neutral": 0.0, "bear": 0.0, "strong_bear": 0.0}
        return replace(long_decision, base_position=mapping.get(estimate.long_regime, long_decision.base_position), reason=f"{long_decision.reason}; experiment_revised_mapping_conservative")
    if variant.policy == "bull_noise_plus_deweighted_strong_bull":
        mapping = {"strong_bull": 0.25, "bull": 0.50, "neutral": 0.0, "bear": 0.0, "strong_bear": 0.0}
        return replace(long_decision, base_position=mapping.get(estimate.long_regime, long_decision.base_position), reason=f"{long_decision.reason}; experiment_bull_noise_plus_deweighted_strong_bull")
    return long_decision


def apply_variant_target_override(
    variant: VariantSpec,
    composed: Any,
    estimate: Any,
    risk_decision: Any,
    *,
    current_position: float,
    v2_position: float,
) -> tuple[Any, str, str]:
    target = float(composed.target_position)
    risk_action = str(risk_decision.risk_action)
    if hard_blocked(estimate, risk_decision):
        return composed, "hard_risk_block", risk_action

    if variant.policy == "strong_bull_overheat_pullback_cap_0p25" and is_hot_or_pullback_strong_bull(estimate):
        return replace(composed, target_position=min(target, 0.25, float(risk_decision.risk_cap))), "strong_bull_hot_pullback_cap_0p25", risk_action
    if variant.policy == "strong_bull_overheat_pullback_cap_0p50" and is_hot_or_pullback_strong_bull(estimate):
        return replace(composed, target_position=min(target, 0.50, float(risk_decision.risk_cap))), "strong_bull_hot_pullback_cap_0p50", risk_action

    if variant.policy == "bull_noise_core_only":
        if current_position <= 0.0 and target > 0.0 and not is_bull_noise_core(estimate, target):
            return replace(composed, target_position=0.0), "bull_noise_core_blocks_other_new_entries", "no_new_entry"
        return composed, "bull_noise_core_allows_slice" if is_bull_noise_core(estimate, target) else "none", risk_action

    if variant.policy == "bull_noise_v2_confirmed_core":
        if current_position <= 0.0 and target > 0.0 and not (is_bull_noise_core(estimate, target) and v2_position > 0.0):
            return replace(composed, target_position=0.0), "bull_noise_v2_confirmed_blocks_other_new_entries", "no_new_entry"
        return composed, "bull_noise_v2_confirmed_allows_slice" if is_bull_noise_core(estimate, target) and v2_position > 0.0 else "none", risk_action

    if variant.policy == "bull_noise_plus_deweighted_strong_bull" and estimate.long_regime == "bull" and estimate.short_regime != "noise":
        return replace(composed, target_position=min(target, 0.25, float(risk_decision.risk_cap))), "bull_non_noise_cap_0p25", risk_action

    return composed, "none", risk_action


def hard_blocked(estimate: Any, risk_decision: Any) -> bool:
    return str(risk_decision.risk_action) == "risk_off" or float(risk_decision.risk_cap) <= 0.0 or estimate.volatility_state == "extreme"


def is_hot_or_pullback_strong_bull(estimate: Any) -> bool:
    return estimate.long_regime == "strong_bull" and estimate.short_regime in {"overheat", "pullback"}


def is_bull_noise_core(estimate: Any, target: float) -> bool:
    return estimate.long_regime == "bull" and estimate.short_regime == "noise" and abs(target - 0.50) <= 1e-12


def is_early_recovery_watchlist(estimate: Any, v2_position: float) -> bool:
    return estimate.long_regime == "neutral" and estimate.short_regime == "recovery" and v2_position > 0.0


def add_forward_returns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    returns = pd.to_numeric(result["asset_return"], errors="coerce").fillna(0.0)
    for horizon in (6, 24, 72):
        result[f"next_{horizon}_bar_return"] = forward_compound_return(returns, horizon)
    return result


def forward_compound_return(returns: pd.Series, horizon: int) -> pd.Series:
    values = (1.0 + returns).to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    for idx in range(len(values)):
        end = idx + horizon
        if end < len(values):
            out[idx] = float(np.prod(values[idx + 1 : end + 1]) - 1.0)
    return pd.Series(out, index=returns.index)


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


def summarize_variant_frame(frame: pd.DataFrame) -> dict[str, float | int | str]:
    metrics = summarize_frame(frame)
    target = pd.to_numeric(frame["target_position"], errors="coerce").fillna(0.0)
    executed = pd.to_numeric(frame["executed_position"], errors="coerce").fillna(0.0)
    metrics.update(
        {
            "average_target_position": float(target.mean()),
            "average_executed_position": float(executed.mean()),
            "target_to_executed_gap": float((target - executed).mean()),
            "exposure_distribution": distribution_string(executed),
            "risk_action_distribution": distribution_string(frame["risk_action"].astype(str)),
            "binding_block_reason_distribution": distribution_string(frame.get("binding_block_reason", pd.Series("unavailable", index=frame.index)).astype(str)),
            "strong_bull_exposure": conditional_mean_exposure(frame, frame["long_regime"] == "strong_bull"),
            "strong_bull_contribution": conditional_contribution(frame, frame["long_regime"] == "strong_bull"),
            "bull_noise_exposure": conditional_mean_exposure(frame, (frame["long_regime"] == "bull") & (frame["short_regime"] == "noise")),
            "bull_noise_contribution": conditional_contribution(frame, (frame["long_regime"] == "bull") & (frame["short_regime"] == "noise")),
            "neutral_exposure": conditional_mean_exposure(frame, frame["long_regime"] == "neutral"),
            "neutral_contribution": conditional_contribution(frame, frame["long_regime"] == "neutral"),
            "early_recovery_watchlist_count": int(pd.Series(frame.get("early_recovery_watchlist", False)).sum()),
        }
    )
    return metrics


def reference_rows(asset: str, dataset: str, data: pd.DataFrame, fee_rate: float, v2_frame: pd.DataFrame) -> list[dict[str, Any]]:
    frames = {
        "v2.btc_final_candidate_A" if asset == "BTC" else "v2.final_candidate_A_cd120_on_ETH": v2_frame,
        "buy_and_hold": build_buy_and_hold_frame(data, fee_rate),
        "ma20_ma60": build_ma_crossover_frame(data, fee_rate),
    }
    rows = []
    for version, frame in frames.items():
        row = {"asset": asset, "dataset": dataset, "fee_rate": fee_rate, "version": version, **summarize_frame(frame)}
        for key in [
            "average_target_position",
            "average_executed_position",
            "target_to_executed_gap",
            "strong_bull_exposure",
            "strong_bull_contribution",
            "bull_noise_exposure",
            "bull_noise_contribution",
            "neutral_exposure",
            "neutral_contribution",
        ]:
            row[key] = np.nan
        rows.append(row)
    return rows


def build_distribution_rows(frame: pd.DataFrame, variant: str) -> list[dict[str, Any]]:
    rows = []
    for diagnostic, column in [
        ("exposure_distribution", "executed_position"),
        ("risk_action_distribution", "risk_action"),
        ("binding_block_reason_distribution", "binding_block_reason"),
        ("variant_override_reason_distribution", "variant_override_reason"),
    ]:
        if column not in frame.columns:
            continue
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


def build_regime_audit_rows(frame: pd.DataFrame, version: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    audit_specs = [
        ("strong_bull_by_short_regime", frame["long_regime"] == "strong_bull", ["short_regime"]),
        ("bull_noise_by_v2_position", (frame["long_regime"] == "bull") & (frame["short_regime"] == "noise") & (frame["target_position"].round(6) == 0.50), ["v2_bucket"]),
        ("missed_v2_long_v3_zero", (frame["v2_position"] > 0.0) & (frame["target_position"] <= 0.0), ["long_short_pair"]),
    ]
    prepared = frame.copy()
    prepared["v2_bucket"] = np.where(prepared["v2_position"] > 0.0, "v2_long", "v2_flat")
    prepared["long_short_pair"] = prepared["long_regime"].astype(str) + "+" + prepared["short_regime"].astype(str)
    for audit, mask, group_cols in audit_specs:
        subset = prepared[mask].copy()
        if subset.empty:
            continue
        grouped = subset.groupby(group_cols, dropna=False)
        for key, group in grouped:
            key_text = key if isinstance(key, str) else "+".join(str(item) for item in key)
            rows.append(
                {
                    "asset": frame["asset"].iloc[0],
                    "dataset": frame["dataset"].iloc[0],
                    "fee_rate": frame["fee_rate"].iloc[0],
                    "version": version,
                    "audit": audit,
                    "bucket": key_text,
                    "count": int(len(group)),
                    "avg_next_6": mean(group["next_6_bar_return"]),
                    "avg_next_24": mean(group["next_24_bar_return"]),
                    "avg_next_72": mean(group["next_72_bar_return"]),
                    "hit_rate_next_24": hit_rate(group["next_24_bar_return"]),
                    "realized_contribution": float(pd.to_numeric(group["strategy_return_net"], errors="coerce").fillna(0.0).sum()),
                    "average_target_position": mean(group["target_position"]),
                    "average_executed_position": mean(group["executed_position"]),
                    "expected_net_edge_0p001": mean(group["next_24_bar_return"]) * mean(group["target_position"]) - 2.0 * 0.001 * mean(group["target_position"]),
                    "later_target_positive_24_rate": later_target_positive_rate(prepared, group.index, 24),
                }
            )
    return rows


def later_target_positive_rate(frame: pd.DataFrame, indexes: pd.Index, horizon: int) -> float:
    target = pd.to_numeric(frame["target_position"], errors="coerce").fillna(0.0).to_numpy()
    values = []
    for idx in indexes:
        loc = int(idx)
        end = min(len(target), loc + horizon + 1)
        values.append(bool((target[loc + 1 : end] > 0.0).any()))
    return float(np.mean(values)) if values else np.nan


def conditional_mean_exposure(frame: pd.DataFrame, mask: pd.Series) -> float:
    subset = frame[mask]
    return mean(subset["executed_position"]) if len(subset) else 0.0


def conditional_contribution(frame: pd.DataFrame, mask: pd.Series) -> float:
    subset = frame[mask]
    return float(pd.to_numeric(subset["strategy_return_net"], errors="coerce").fillna(0.0).sum())


def aggregate_variants(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
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
            avg_strong_bull_exposure=("strong_bull_exposure", "mean"),
            avg_strong_bull_contribution=("strong_bull_contribution", "mean"),
            avg_bull_noise_exposure=("bull_noise_exposure", "mean"),
            avg_bull_noise_contribution=("bull_noise_contribution", "mean"),
            avg_neutral_exposure=("neutral_exposure", "mean"),
            avg_neutral_contribution=("neutral_contribution", "mean"),
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
            avg_max_drawdown=("max_drawdown", "mean"),
            avg_sharpe_ratio=("sharpe_ratio", "mean"),
            avg_turnover=("turnover", "mean"),
            avg_fee_drag=("fee_drag", "mean"),
            avg_average_exposure=("average_exposure", "mean"),
        )
        .reset_index()
        .sort_values(["asset", "avg_sharpe_ratio"], ascending=[True, False])
    )


def rolling_comparison_table(rolling: pd.DataFrame, asset: str) -> pd.DataFrame:
    rows = []
    candidates = [variant.name for variant in VARIANTS if variant.name != "A_current_v3_final_candidate"]
    asset_rolling = rolling[rolling["asset"] == asset]
    for candidate in candidates:
        comparison = compare_versions(asset_rolling, candidate, "A_current_v3_final_candidate")
        if not comparison.empty:
            rows.append(comparison)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def write_report(summary: pd.DataFrame, rolling: pd.DataFrame, audits: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    variants = [variant.name for variant in VARIANTS]
    variant_summary = summary[summary["version"].isin(variants)].copy()
    reference_summary = summary[~summary["version"].isin(variants)].copy()
    btc = aggregate_variants(variant_summary[variant_summary["asset"] == "BTC"])
    eth = aggregate_variants(variant_summary[variant_summary["asset"] == "ETH"])
    refs = aggregate_references(reference_summary)
    btc_roll = rolling_comparison_table(rolling, "BTC")
    eth_roll = rolling_comparison_table(rolling, "ETH")
    btc_best_return = best_variant(btc, "avg_total_return")
    btc_best_sharpe = best_variant(btc, "avg_sharpe_ratio")
    btc_tradeoff = tradeoff_variant(btc)
    lines = [
        "# v3.7 Estimator / Controller Remapping Study",
        "",
        "This diagnostics-first study keeps v2 behavior, v3.final_candidate behavior, Risk Supervisor logic, feature windows, estimator thresholds, particle filter status, and leverage settings unchanged. Remaps are isolated named experiment variants.",
        "",
        "## 1. Executive Summary",
        "",
        f"- Best BTC return variant: `{btc_best_return}`.",
        f"- Best BTC Sharpe variant: `{btc_best_sharpe}`.",
        f"- Best BTC drawdown/Sharpe tradeoff variant: `{btc_tradeoff}`.",
        replacement_text(btc, btc_roll, btc_tradeoff),
        "",
        "## 2. Why Remapping Is Needed",
        "",
        "Prior diagnostics showed `strong_bull` has weak BTC forward-return behavior, while `bull + noise` around target 0.50 is the most promising v3 slice. Risk relaxation increased exposure but worsened Sharpe/drawdown, so this step tests estimator/controller semantics instead.",
        "",
        "## 3. Diagnostics-Only Regime Audit",
        "",
        frame_to_markdown(audit_focus(audits)),
        "",
        "## 4. Variant Definitions",
        "",
        variant_definitions_markdown(),
        "",
        "## 5. BTC Full-Period Comparison",
        "",
        frame_to_markdown(btc),
        "",
        "## 6. BTC Rolling Comparison",
        "",
        frame_to_markdown(btc_roll),
        "",
        "## 7. ETH Full-Period Comparison",
        "",
        frame_to_markdown(eth),
        "",
        "## 8. ETH Rolling Comparison",
        "",
        frame_to_markdown(eth_roll),
        "",
        "## Reference Comparison",
        "",
        frame_to_markdown(refs),
        "",
        "## 9. Strong-Bull Semantics Conclusion",
        "",
        strong_bull_text(audits, btc),
        "",
        "## 10. Bull + Noise Slice Conclusion",
        "",
        bull_noise_text(audits, btc),
        "",
        "## 11. v2 Diagnostic Feature Conclusion",
        "",
        "v2 remains useful as a diagnostic timing feature. In this study it is only used as a filter in `F_bull_noise_v2_confirmed_core`; it is not allowed to create targets outside the diagnosed bull + noise slice.",
        "",
        "## 12. Best Variant By BTC Return",
        "",
        f"`{btc_best_return}`.",
        "",
        "## 13. Best Variant By BTC Sharpe",
        "",
        f"`{btc_best_sharpe}`.",
        "",
        "## 14. Best Variant By Drawdown/Sharpe Tradeoff",
        "",
        f"`{btc_tradeoff}`.",
        "",
        "## 15. Should Any Variant Replace v3.final_candidate?",
        "",
        replacement_text(btc, btc_roll, btc_tradeoff),
        "",
        "## 16. Should G Remain Only A Cleanup Candidate?",
        "",
        "Yes, unless a later robustness pass shows broader changed-window improvement. The previous strong-bull deweighting effect remains useful but sparse.",
        "",
        "## 17. Recommended Next Step",
        "",
        next_step_text(btc, btc_roll, btc_best_sharpe),
        "",
        "## Files",
        "",
        f"- Summary CSV: `{SUMMARY_CSV}`",
        f"- Rolling CSV: `{ROLLING_CSV}`",
        f"- Audit CSV: `{AUDIT_CSV}`",
        f"- Diagnostics CSV: `{DIAGNOSTICS_CSV}`",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def audit_focus(audits: pd.DataFrame) -> pd.DataFrame:
    if audits.empty:
        return pd.DataFrame()
    btc = audits[
        (audits["asset"] == "BTC")
        & (audits["version"].isin(["A_current_v3_final_candidate", "B_strong_bull_deweight"]))
    ].copy()
    buckets = {
        "strong_bull_by_short_regime": ["noise", "pullback", "overheat", "breakdown"],
        "bull_noise_by_v2_position": ["v2_long", "v2_flat"],
        "missed_v2_long_v3_zero": ["neutral+noise", "neutral+recovery", "strong_bear+noise", "strong_bear+recovery", "bear+noise", "bear+recovery"],
    }
    mask = pd.Series(False, index=btc.index)
    for audit, wanted in buckets.items():
        mask |= (btc["audit"] == audit) & (btc["bucket"].isin(wanted))
    return btc[mask].groupby(["version", "audit", "bucket"], dropna=False).agg(
        rows=("dataset", "count"),
        avg_count=("count", "mean"),
        avg_next_6=("avg_next_6", "mean"),
        avg_next_24=("avg_next_24", "mean"),
        avg_next_72=("avg_next_72", "mean"),
        hit_rate_next_24=("hit_rate_next_24", "mean"),
        avg_realized_contribution=("realized_contribution", "mean"),
        expected_net_edge_0p001=("expected_net_edge_0p001", "mean"),
        later_target_positive_24_rate=("later_target_positive_24_rate", "mean"),
    ).reset_index()


def variant_definitions_markdown() -> str:
    definitions = [
        ("A_current_v3_final_candidate", "Existing explicit v3.final_candidate control."),
        ("B_strong_bull_deweight", "Previous G reference: strong_bull base position 0.75 -> 0.50."),
        ("C_strong_bull_to_bull_mapping", "Treat strong_bull like bull for base-position sizing, kept separately named for diagnostics."),
        ("D1_strong_bull_overheat_pullback_cap_0p25", "Keep strong_bull 0.75, but cap strong_bull + overheat/pullback targets to 0.25."),
        ("D2_strong_bull_overheat_pullback_cap_0p50", "Keep strong_bull 0.75, but cap strong_bull + overheat/pullback targets to 0.50."),
        ("E_bull_noise_core_only", "Only allow new nonzero entries from bull + noise + target 0.50; existing holds are not forcibly exited by this rule."),
        ("F_bull_noise_v2_confirmed_core", "Same bull + noise core, but new entries also require v2_position == 1 as confirmation only."),
        ("G_neutral_recovery_watchlist_not_trade", "No behavior change; records neutral + recovery + v2 long as a diagnostics-only watchlist."),
        ("H_revised_mapping_conservative", "strong_bull 0.25, bull 0.50, neutral 0.00, bear 0.00, strong_bear 0.00."),
        ("I_bull_noise_plus_deweighted_strong_bull", "strong_bull 0.25, bull 0.50, neutral 0.00; bull non-noise targets capped to 0.25."),
    ]
    return "\n".join(f"- `{name}`: {description}" for name, description in definitions)


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


def replacement_text(btc: pd.DataFrame, btc_roll: pd.DataFrame, tradeoff: str) -> str:
    if btc.empty or tradeoff == "A_current_v3_final_candidate":
        return "No remapping variant clearly replaces v3.final_candidate."
    candidate = btc[btc["version"] == tradeoff]
    baseline = btc[btc["version"] == "A_current_v3_final_candidate"]
    if candidate.empty or baseline.empty:
        return "No remapping variant clearly replaces v3.final_candidate."
    sharpe_non_worse = metric_rate(btc_roll, tradeoff, "sharpe_ratio", "non_worse_rate")
    sharpe_changed = metric_rate(btc_roll, tradeoff, "sharpe_ratio", "changed_rate")
    if float(candidate.iloc[0]["avg_sharpe_ratio"]) >= float(baseline.iloc[0]["avg_sharpe_ratio"]) and sharpe_non_worse >= 0.8 and sharpe_changed >= 0.1:
        return f"`{tradeoff}` is eligible for a narrower robustness pass, but should not replace v3.final_candidate until reviewed."
    return f"`{tradeoff}` does not have enough rolling evidence to replace v3.final_candidate."


def strong_bull_text(audits: pd.DataFrame, btc: pd.DataFrame) -> str:
    focus = audit_focus(audits)
    strong = focus[focus["audit"] == "strong_bull_by_short_regime"]
    if strong.empty:
        return "Strong-bull diagnostics were unavailable."
    worst = strong.sort_values("avg_next_24").head(1).iloc[0]
    return f"BTC strong_bull remains suspect. The weakest audited strong_bull short-regime bucket is `{worst['bucket']}` with average next-24 return {worst['avg_next_24']:.6g}; this supports treating strong_bull as potentially late/overheated rather than automatically highest exposure."


def bull_noise_text(audits: pd.DataFrame, btc: pd.DataFrame) -> str:
    focus = audit_focus(audits)
    bull_noise = focus[focus["audit"] == "bull_noise_by_v2_position"]
    if bull_noise.empty:
        return "Bull + noise diagnostics were unavailable."
    best = bull_noise.sort_values("avg_next_24", ascending=False).head(1).iloc[0]
    return f"The best audited bull + noise split is `{best['bucket']}` with average next-24 return {best['avg_next_24']:.6g} and hit rate {best['hit_rate_next_24']:.6g}. This remains the most interpretable candidate slice, but it still needs rolling support before becoming core v3 logic."


def next_step_text(btc: pd.DataFrame, btc_roll: pd.DataFrame, best_sharpe: str) -> str:
    if best_sharpe in {"E_bull_noise_core_only", "F_bull_noise_v2_confirmed_core", "I_bull_noise_plus_deweighted_strong_bull"}:
        return f"Run a narrower no-lookahead robustness pass on `{best_sharpe}` and inspect changed windows before promoting anything. Keep v2 as a diagnostic/filter only."
    return "No remap is ready for promotion. Next, redesign the estimator labels for strong_bull versus bull/noise, then rerun the target-alpha audit before any new trading variant."


def metric_rate(table: pd.DataFrame, candidate: str, metric: str, column: str) -> float:
    row = table[(table["candidate"] == candidate) & (table["metric"] == metric)]
    return float(row.iloc[0][column]) if not row.empty else np.nan


def distribution_string(values: pd.Series) -> str:
    counts = values.value_counts(dropna=False).sort_index()
    return "; ".join(f"{bucket}:{count}" for bucket, count in counts.items())


def mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else np.nan


def hit_rate(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float((values > 0.0).mean()) if len(values) else np.nan


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

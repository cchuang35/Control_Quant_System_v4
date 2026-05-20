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
from src.v3.risk_supervisor import PortfolioRiskStateV3, supervise_risk
from src.v3.short_term_controller import decide_short_term_adjustment
from v2_small_cap import backtest_v2_btc_final_candidate_a, backtest_v2_final_candidate_a


REPORT_PATH = Path("reports") / "v3_6_v2_assisted_early_entry_experiment.md"
SUMMARY_CSV = Path("reports") / "v3_6_v2_assisted_early_entry_summary.csv"
DIAGNOSTICS_CSV = Path("reports") / "v3_6_v2_assisted_early_entry_diagnostics.csv"
EARLY_ENTRY_CSV = Path("reports") / "v3_6_v2_assisted_early_entry_slice.csv"
CAPTURE_CSV = Path("reports") / "v3_6_v2_assisted_early_entry_capture.csv"
STRONG_BULL_CSV = Path("reports") / "v3_6_v2_assisted_early_entry_strong_bull.csv"


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
    previous_target_positive_history: list[bool] | None = None

    def __post_init__(self) -> None:
        if self.recent_trade_amounts is None:
            self.recent_trade_amounts = []
        if self.cooldown is None:
            self.cooldown = RegimeCooldownManagerV3(cooldown_bars=120)
        if self.previous_target_positive_history is None:
            self.previous_target_positive_history = []


VARIANTS = (
    VariantSpec("A_current_v3_final_candidate", "current"),
    VariantSpec("B_v2_early_entry_neutral_only_cap_0p25", "neutral_only"),
    VariantSpec("C_v2_early_entry_neutral_and_danger_cap_0p25", "neutral_and_danger"),
    VariantSpec("D_v2_early_entry_neutral_strong_bear_cap_0p25", "neutral_strong_bear"),
    VariantSpec("E_v2_confirmed_bull_noise_boost", "bull_noise_boost"),
    VariantSpec("F_v2_early_entry_delay_bridge_24", "delay_bridge_24"),
    VariantSpec("G_strong_bull_deweight_mapping", "strong_bull_deweight"),
    VariantSpec("H_bull_noise_only_v3_slice", "bull_noise_only"),
)


def main() -> None:
    config = load_final_candidate_config()
    fee_rates = tuple(float(value) for value in config["execution"]["fee_rates_to_validate"])
    summary_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    early_rows: list[dict[str, Any]] = []
    capture_rows: list[dict[str, Any]] = []
    strong_rows: list[dict[str, Any]] = []

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
                    enriched = enrich_with_waterfall_fields(with_keys(frame, asset, dataset, fee_rate))
                    summary_rows.append({"asset": asset, "dataset": dataset, "fee_rate": fee_rate, "version": variant, **summarize_variant_frame(enriched)})
                    diagnostic_rows.extend(build_distribution_rows(enriched, variant))
                    early_rows.append({"asset": asset, "dataset": dataset, "fee_rate": fee_rate, "version": variant, **early_entry_summary(enriched)})
                    capture_rows.append({"asset": asset, "dataset": dataset, "fee_rate": fee_rate, "version": variant, **capture_summary(enriched)})
                    strong_rows.append({"asset": asset, "dataset": dataset, "fee_rate": fee_rate, "version": variant, **strong_bull_summary(enriched)})
                summary_rows.extend(reference_rows(asset, dataset, data, fee_rate, v2_frame))

    summary = pd.DataFrame(summary_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    early = pd.DataFrame(early_rows)
    capture = pd.DataFrame(capture_rows)
    strong = pd.DataFrame(strong_rows)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)
    diagnostics.to_csv(DIAGNOSTICS_CSV, index=False)
    early.to_csv(EARLY_ENTRY_CSV, index=False)
    capture.to_csv(CAPTURE_CSV, index=False)
    strong.to_csv(STRONG_BULL_CSV, index=False)
    write_report(summary, diagnostics, early, capture, strong)
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {DIAGNOSTICS_CSV}")
    print(f"Wrote {EARLY_ENTRY_CSV}")
    print(f"Wrote {CAPTURE_CSV}")
    print(f"Wrote {STRONG_BULL_CSV}")


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
            assert state.previous_target_positive_history is not None
            asset_return = float(feature_row.return_1)
            portfolio_drawdown = state.equity / state.equity_peak - 1.0
            recent_turnover = sum(state.recent_trade_amounts[-config.recent_turnover_window :])
            cooldown_active = state.cooldown.is_active(estimate.long_regime)
            long_decision = variant_long_decision(variant, base_long_decision)
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
                previous_target_positive_history=state.previous_target_positive_history,
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
            early_entry_active = override_reason.startswith("early_entry") or override_reason.startswith("bull_noise_boost")
            bull_noise_entry_active = override_reason.startswith("bull_noise")
            missed_v2_long_baseline = bool(v2_pos > 0.0 and normal_target_position <= 0.0)
            captured_missed_v2_long = bool(missed_v2_long_baseline and adjusted.target_position > 0.0)
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
                    "short_reason": short_decision.reason,
                    "execution_reason": executed.execution_reason,
                    "variant_override_reason": override_reason,
                    "early_entry_active": early_entry_active,
                    "bull_noise_entry_active": bull_noise_entry_active,
                    "missed_v2_long_baseline": missed_v2_long_baseline,
                    "captured_missed_v2_long": captured_missed_v2_long,
                    "v2_position": v2_pos,
                    "cooldown_active": cooldown_active,
                    "cooldown_triggered": cooldown_triggered,
                }
            )
            state.current_position = executed_position
            state.previous_target_positive_history.append(bool(normal_target_position > 0.0))

    return {variant: pd.DataFrame(records) for variant, records in rows.items()}


def variant_long_decision(variant: VariantSpec, long_decision: Any) -> Any:
    if variant.policy == "strong_bull_deweight" and long_decision.long_regime == "strong_bull":
        return replace(long_decision, base_position=0.50, reason=f"{long_decision.reason}; experiment_strong_bull_deweight_to_0p50")
    return long_decision


def apply_variant_target_override(
    variant: VariantSpec,
    composed: Any,
    estimate: Any,
    risk_decision: Any,
    *,
    current_position: float,
    v2_position: float,
    previous_target_positive_history: list[bool],
) -> tuple[Any, str, str]:
    target = float(composed.target_position)
    risk_action = str(risk_decision.risk_action)
    if variant.policy == "current" or variant.policy == "strong_bull_deweight":
        return composed, "none", risk_action
    if hard_blocked(estimate, risk_decision):
        if variant.policy == "bull_noise_only" and target > 0.0 and current_position <= 0.0:
            return replace(composed, target_position=0.0), "bull_noise_only_hard_block", risk_action
        return composed, "hard_risk_block", risk_action

    if variant.policy == "bull_noise_only":
        condition = (
            estimate.long_regime == "bull"
            and estimate.short_regime == "noise"
            and abs(target - 0.50) <= 1e-12
            and v2_position > 0.0
        )
        if target > 0.0 and current_position <= 0.0 and not condition:
            return replace(composed, target_position=0.0), "bull_noise_only_blocks_other_new_entries", "no_new_entry"
        return composed, "bull_noise_only_allows_slice" if condition else "none", risk_action

    if variant.policy == "bull_noise_boost" and is_bull_noise_boost(estimate, v2_position):
        new_target = min(0.50, float(risk_decision.risk_cap))
        return replace(composed, target_position=new_target), "bull_noise_boost_to_0p50", "normal"

    if target > 0.0:
        return composed, "none", risk_action
    if v2_position <= 0.0:
        return composed, "none", risk_action

    if variant.policy == "neutral_only" and is_neutral_noise_recovery(estimate):
        return replace(composed, target_position=min(0.25, float(risk_decision.risk_cap))), "early_entry_neutral_cap_0p25", "normal"
    if variant.policy == "neutral_and_danger" and (
        is_neutral_noise_recovery(estimate)
        or (estimate.drawdown_state == "danger" and estimate.long_regime not in {"bear", "strong_bear"} and estimate.short_regime in {"noise", "recovery"})
    ):
        return replace(composed, target_position=min(0.25, float(risk_decision.risk_cap))), "early_entry_neutral_or_danger_cap_0p25", "normal"
    if variant.policy == "neutral_strong_bear" and (
        estimate.long_regime in {"neutral", "strong_bear"}
        and estimate.short_regime in {"noise", "recovery"}
    ):
        return replace(composed, target_position=min(0.25, float(risk_decision.risk_cap))), "early_entry_neutral_strong_bear_cap_0p25", "normal"
    if variant.policy == "delay_bridge_24" and is_delay_bridge_condition(estimate, previous_target_positive_history):
        return replace(composed, target_position=min(0.25, float(risk_decision.risk_cap))), "early_entry_delay_bridge_24_cap_0p25", "normal"
    return composed, "none", risk_action


def hard_blocked(estimate: Any, risk_decision: Any) -> bool:
    return str(risk_decision.risk_action) == "risk_off" or float(risk_decision.risk_cap) <= 0.0 or estimate.volatility_state == "extreme"


def is_neutral_noise_recovery(estimate: Any) -> bool:
    return estimate.long_regime == "neutral" and estimate.short_regime in {"noise", "recovery"} and estimate.volatility_state != "extreme"


def is_bull_noise_boost(estimate: Any, v2_position: float) -> bool:
    return estimate.long_regime == "bull" and estimate.short_regime == "noise" and v2_position > 0.0 and estimate.volatility_state != "extreme"


def is_delay_bridge_condition(estimate: Any, history: list[bool]) -> bool:
    previous_24_all_zero = not any(history[-24:]) if history else True
    return (
        previous_24_all_zero
        and estimate.long_regime in {"neutral", "bear", "strong_bear"}
        and estimate.short_regime in {"noise", "recovery"}
        and float(estimate.confidence_score) <= 0.50
        and estimate.volatility_state != "extreme"
    )


def with_keys(frame: pd.DataFrame, asset: str, dataset: str, fee_rate: float) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, "fee_rate", fee_rate)
    result.insert(0, "dataset", dataset)
    result.insert(0, "asset", asset)
    return result


def enrich_with_waterfall_fields(frame: pd.DataFrame) -> pd.DataFrame:
    waterfall = add_waterfall_fields(frame)
    result = frame.copy()
    for column in waterfall.columns:
        if column not in result.columns:
            result[column] = waterfall[column].to_numpy()
    return result


def summarize_variant_frame(frame: pd.DataFrame) -> dict[str, float | int]:
    metrics = summarize_frame(frame)
    target = pd.to_numeric(frame["target_position"], errors="coerce").fillna(0.0)
    executed = pd.to_numeric(frame["executed_position"], errors="coerce").fillna(0.0)
    metrics.update(
        {
            "average_target_position": float(target.mean()),
            "average_executed_position": float(executed.mean()),
            "target_to_executed_gap": float((target - executed).mean()),
            "no_new_entry_count": int((frame["risk_action"] == "no_new_entry").sum()),
            "early_entry_count": int(frame["early_entry_active"].sum()),
            "early_entry_return_contribution": float(frame.loc[frame["early_entry_active"], "strategy_return_net"].sum()),
            "bull_noise_entry_count": int(frame["bull_noise_entry_active"].sum()),
            "captured_missed_v2_long_count": int(frame["captured_missed_v2_long"].sum()),
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
        row.update(
            {
                "average_target_position": np.nan,
                "average_executed_position": np.nan,
                "target_to_executed_gap": np.nan,
                "no_new_entry_count": np.nan,
                "early_entry_count": np.nan,
                "early_entry_return_contribution": np.nan,
                "bull_noise_entry_count": np.nan,
                "captured_missed_v2_long_count": np.nan,
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


def early_entry_summary(frame: pd.DataFrame) -> dict[str, Any]:
    early = frame[frame["early_entry_active"]].copy()
    next24 = pd.to_numeric(early.get("next_24_bar_return", pd.Series(dtype=float)), errors="coerce")
    return {
        "early_entry_count": int(len(early)),
        "early_entry_avg_next_24_return": mean(next24),
        "early_entry_hit_rate_next_24": hit_rate(next24),
        "early_entry_strategy_contribution": float(pd.to_numeric(early.get("strategy_return_net", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()),
        "early_entry_turnover": float(pd.to_numeric(early.get("trade_amount", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()),
        "early_entry_fee_drag": float(pd.to_numeric(early.get("fee_cost", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()),
    }


def capture_summary(frame: pd.DataFrame) -> dict[str, Any]:
    missed = frame[frame["missed_v2_long_baseline"]].copy()
    captured = missed[missed["captured_missed_v2_long"]].copy()
    still = missed[~missed["captured_missed_v2_long"]].copy()
    return {
        "missed_v2_long_count": int(len(missed)),
        "captured_missed_v2_long_count": int(len(captured)),
        "captured_missed_v2_long_rate": float(len(captured) / len(missed)) if len(missed) else np.nan,
        "captured_avg_next_24_return": mean(pd.to_numeric(captured.get("next_24_bar_return", pd.Series(dtype=float)), errors="coerce")),
        "still_missed_avg_next_24_return": mean(pd.to_numeric(still.get("next_24_bar_return", pd.Series(dtype=float)), errors="coerce")),
    }


def strong_bull_summary(frame: pd.DataFrame) -> dict[str, Any]:
    strong = frame[frame["long_regime"] == "strong_bull"].copy()
    return {
        "strong_bull_bars": int(len(strong)),
        "strong_bull_average_exposure": mean(pd.to_numeric(strong.get("executed_position", pd.Series(dtype=float)), errors="coerce")),
        "strong_bull_strategy_contribution": float(pd.to_numeric(strong.get("strategy_return_net", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()),
        "strong_bull_turnover": float(pd.to_numeric(strong.get("trade_amount", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()),
        "strong_bull_fee_drag": float(pd.to_numeric(strong.get("fee_cost", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()),
    }


def write_report(summary: pd.DataFrame, diagnostics: pd.DataFrame, early: pd.DataFrame, capture: pd.DataFrame, strong: pd.DataFrame) -> None:
    variants = [variant.name for variant in VARIANTS]
    variant_summary = summary[summary["version"].isin(variants)].copy()
    reference_summary = summary[~summary["version"].isin(variants)].copy()
    btc = aggregate_variants(variant_summary[variant_summary["asset"] == "BTC"])
    eth = aggregate_variants(variant_summary[variant_summary["asset"] == "ETH"])
    refs = aggregate_references(reference_summary)
    early_table = aggregate_auxiliary(early, "early")
    capture_table = aggregate_auxiliary(capture, "capture")
    strong_table = aggregate_auxiliary(strong, "strong")
    btc_best_return = best_variant(btc, "avg_total_return")
    btc_best_sharpe = best_variant(btc, "avg_sharpe_ratio")
    btc_tradeoff = tradeoff_variant(btc)
    lines = [
        "# v3.6 v2-Assisted Early-Entry Experiment",
        "",
        "This is an isolated experiment runner. v2 behavior, v3.final_candidate config, Risk Supervisor logic, feature windows, estimator thresholds, particle-filter status, and leverage settings are unchanged.",
        "",
        "## 1. Executive Summary",
        "",
        f"- Best BTC return variant: `{btc_best_return}`.",
        f"- Best BTC Sharpe variant: `{btc_best_sharpe}`.",
        f"- Best BTC drawdown/Sharpe tradeoff variant: `{btc_tradeoff}`.",
        "- v2 is used only as an early-entry/filter feature in named variants, not as a full target replacement.",
        "",
        "## 2. Why This Experiment Is Needed",
        "",
        "Prior diagnostics showed `v2_long_v3_target_zero` has strong BTC forward returns while naive Risk Supervisor relaxation worsens Sharpe. v3.6 tests whether v2 can help v3 enter earlier in specific diagnosed states without blindly copying v2 exposure.",
        "",
        "## 3. Variant Definitions",
        "",
        variant_definitions_markdown(),
        "",
        "## 4. BTC Comparison Table",
        "",
        _frame_to_markdown(btc),
        "",
        "## 5. ETH Comparison Table",
        "",
        _frame_to_markdown(eth),
        "",
        "## Reference Comparison Table",
        "",
        _frame_to_markdown(refs),
        "",
        "## 6. Early-Entry Diagnostics",
        "",
        _frame_to_markdown(early_table),
        "",
        "## 7. v2 Missed-Slice Capture Diagnostics",
        "",
        _frame_to_markdown(capture_table),
        "",
        "## 8. Strong-Bull Deweighting Diagnostics",
        "",
        _frame_to_markdown(strong_table),
        "",
        "## 9. Best Variant By BTC Return",
        "",
        f"`{btc_best_return}` has the highest average BTC total return among v3.6 variants.",
        "",
        "## 10. Best Variant By BTC Sharpe",
        "",
        f"`{btc_best_sharpe}` has the highest average BTC Sharpe among v3.6 variants.",
        "",
        "## 11. Best Variant By Drawdown/Sharpe Tradeoff",
        "",
        f"`{btc_tradeoff}` has the best simple drawdown/Sharpe/turnover tradeoff score.",
        "",
        "## 12. Should Any Variant Replace v3.final_candidate?",
        "",
        replacement_text(btc, btc_tradeoff),
        "",
        "## 13. Should v3.6 Continue Toward v2-Assisted Architecture?",
        "",
        continue_text(btc, btc_best_sharpe),
        "",
        "## 14. Recommended Next Step",
        "",
        "If a v2-assisted variant improves BTC Sharpe without drawdown/fee explosion, run rolling validation on that exact variant. Otherwise, keep v2 fields as diagnostics and redesign the v3 estimator/controller around the proven `bull + noise + v2_long` slice.",
        "",
        "## Files",
        "",
        f"- Summary CSV: `{SUMMARY_CSV}`",
        f"- Diagnostics CSV: `{DIAGNOSTICS_CSV}`",
        f"- Early-entry CSV: `{EARLY_ENTRY_CSV}`",
        f"- Capture CSV: `{CAPTURE_CSV}`",
        f"- Strong-bull CSV: `{STRONG_BULL_CSV}`",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


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
            avg_no_new_entry_count=("no_new_entry_count", "mean"),
            avg_early_entry_count=("early_entry_count", "mean"),
            avg_early_entry_return_contribution=("early_entry_return_contribution", "mean"),
            avg_bull_noise_entry_count=("bull_noise_entry_count", "mean"),
            avg_captured_missed_v2_long_count=("captured_missed_v2_long_count", "mean"),
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


def aggregate_auxiliary(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    aggregations = {column: (column, "mean") for column in frame.columns if column not in {"asset", "dataset", "fee_rate", "version"}}
    return frame.groupby(["asset", "version"], dropna=False).agg(**aggregations).reset_index()


def variant_definitions_markdown() -> str:
    definitions = [
        ("A_current_v3_final_candidate", "Existing v3.final_candidate control."),
        ("B_v2_early_entry_neutral_only_cap_0p25", "v2 long can open 0.25 only in neutral + noise/recovery with non-extreme volatility and no hard risk block."),
        ("C_v2_early_entry_neutral_and_danger_cap_0p25", "B plus drawdown_state=danger, excluding bear/strong_bear."),
        ("D_v2_early_entry_neutral_strong_bear_cap_0p25", "v2 long can open 0.25 in neutral or strong_bear with noise/recovery."),
        ("E_v2_confirmed_bull_noise_boost", "bull + noise + v2 long can target 0.50 even when no_new_entry would otherwise block entry."),
        ("F_v2_early_entry_delay_bridge_24", "No-lookahead bridge heuristic: v2 long, previous 24 v3 targets flat, low confidence, neutral/bear/strong_bear + noise/recovery."),
        ("G_strong_bull_deweight_mapping", "No v2 early entry; strong_bull base position is reduced from 0.75 to 0.50."),
        ("H_bull_noise_only_v3_slice", "Only new nonzero v3 entries in bull + noise + target 0.50 + v2 long are allowed."),
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


def replacement_text(btc: pd.DataFrame, tradeoff: str) -> str:
    baseline = btc[btc["version"] == "A_current_v3_final_candidate"]
    candidate = btc[btc["version"] == tradeoff]
    if baseline.empty or candidate.empty or tradeoff == "A_current_v3_final_candidate":
        return "No v3.6 variant clearly replaces v3.final_candidate based on this diagnostic run."
    b = baseline.iloc[0]
    c = candidate.iloc[0]
    drawdown_ok = abs(float(c["avg_max_drawdown"])) <= abs(float(b["avg_max_drawdown"])) * 1.5
    if c["avg_sharpe_ratio"] >= b["avg_sharpe_ratio"] and drawdown_ok:
        return f"`{tradeoff}` is eligible for rolling validation, but should not replace v3.final_candidate until robustness is checked."
    return f"`{tradeoff}` improves part of the tradeoff, but not enough to replace v3.final_candidate."


def continue_text(btc: pd.DataFrame, best_sharpe: str) -> str:
    if best_sharpe == "A_current_v3_final_candidate":
        return "Not yet. v2-assisted signals should remain diagnostic until a variant improves BTC Sharpe without sacrificing the risk-control gains."
    if best_sharpe == "G_strong_bull_deweight_mapping":
        return "Not as a v2-assisted architecture yet. The best risk-adjusted result came from deweighting v3 strong_bull, not from using v2 as an early-entry feature."
    return f"Possibly. `{best_sharpe}` should be rolled forward as an isolated v3.6 candidate before any architecture change."


def mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else np.nan


def hit_rate(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float((values > 0.0).mean()) if len(values) else np.nan


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

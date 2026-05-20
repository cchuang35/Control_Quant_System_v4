from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtester import load_ohlcv_csv, run_backtest_fast
from scripts.run_v3_8_estimator_label_redesign_audit import assign_trend_health_label
from scripts.run_v3_final_candidate import (
    PERIODS_PER_YEAR,
    V1_ENTRY_THRESHOLD,
    input_frame_from_v1,
    load_final_candidate_config,
)
from v2_small_cap import backtest_v2_btc_final_candidate_a, backtest_v2_final_candidate_a


REPORT_PATH = Path("reports") / "v2_alpha_extraction_study.md"
TRADE_LEVEL_CSV = Path("reports") / "v2_alpha_extraction_trade_level.csv"
BAR_LEVEL_CSV = Path("reports") / "v2_alpha_extraction_bar_level.csv"
REGIME_TIMING_CSV = Path("reports") / "v2_alpha_extraction_regime_timing.csv"
SIGNAL_COMPONENTS_CSV = Path("reports") / "v2_alpha_extraction_signal_components.csv"

BTC_WATERFALL_CSV = Path("reports") / "v3_decision_waterfall_btc_1h.csv"
ETH_WATERFALL_CSV = Path("reports") / "v3_decision_waterfall_eth_1h.csv"


def main() -> None:
    config = load_final_candidate_config()
    bar_level = build_bar_level(config)
    trade_level = build_trade_level(bar_level)
    regime_timing = build_regime_timing(bar_level, trade_level)
    signal_components = build_signal_component_tables(bar_level, trade_level)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    trade_level.to_csv(TRADE_LEVEL_CSV, index=False)
    bar_level.to_csv(BAR_LEVEL_CSV, index=False)
    regime_timing.to_csv(REGIME_TIMING_CSV, index=False)
    signal_components.to_csv(SIGNAL_COMPONENTS_CSV, index=False)
    write_report(bar_level, trade_level, regime_timing, signal_components)

    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {TRADE_LEVEL_CSV}")
    print(f"Wrote {BAR_LEVEL_CSV}")
    print(f"Wrote {REGIME_TIMING_CSV}")
    print(f"Wrote {SIGNAL_COMPONENTS_CSV}")


def build_bar_level(config: dict[str, Any]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    fee_rates = tuple(float(value) for value in config["execution"]["fee_rates_to_validate"])
    for asset, dataset_key, waterfall_path in [
        ("BTC", "btc_datasets", BTC_WATERFALL_CSV),
        ("ETH", "eth_datasets", ETH_WATERFALL_CSV),
    ]:
        if not waterfall_path.exists():
            print(f"missing v3 waterfall CSV for {asset}: {waterfall_path}")
            continue
        print(f"loading {waterfall_path}")
        v3_all = pd.read_csv(waterfall_path, low_memory=False)
        for data_path in [Path(path) for path in config["validation"].get(dataset_key, []) if Path(path).exists()]:
            dataset = data_path.stem
            data = load_ohlcv_csv(data_path)
            for fee_rate in fee_rates:
                print(f"asset={asset} dataset={dataset} fee={fee_rate:g}")
                v3 = v3_all[
                    (v3_all["dataset"].astype(str) == dataset)
                    & (pd.to_numeric(v3_all["fee_rate"], errors="coerce") == fee_rate)
                ].copy()
                if v3.empty:
                    print(f"  missing v3 rows for {asset} {dataset} {fee_rate:g}")
                    continue
                v2 = run_v2_frame(asset, data, fee_rate)
                frames.append(align_frames(asset, dataset, fee_rate, v2, v3))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_v2_frame(asset: str, data: pd.DataFrame, fee_rate: float) -> pd.DataFrame:
    v1_result = run_backtest_fast(
        data,
        fee_rate=fee_rate,
        periods_per_year=PERIODS_PER_YEAR,
        progress_every=10000 if len(data) > 15000 else None,
    )
    v2_input = input_frame_from_v1(v1_result)
    v2_func = backtest_v2_btc_final_candidate_a if asset == "BTC" else backtest_v2_final_candidate_a
    return v2_func(v2_input, fee_rate=fee_rate, v1_entry_threshold=V1_ENTRY_THRESHOLD, cooldown_bars=120).reset_index(drop=True)


def align_frames(asset: str, dataset: str, fee_rate: float, v2: pd.DataFrame, v3: pd.DataFrame) -> pd.DataFrame:
    length = min(len(v2), len(v3))
    v2 = v2.iloc[:length].copy().reset_index(drop=True)
    v3 = v3.iloc[:length].copy().reset_index(drop=True)

    result = pd.DataFrame(
        {
            "asset": asset,
            "dataset": dataset,
            "fee_rate": fee_rate,
            "bar_index": np.arange(length),
            "timestamp": pick_string(v3, "timestamp", v2, fallback=""),
            "close": numeric(v2, "close"),
            "asset_return": numeric(v2, "asset_return"),
            "v2_position": numeric(v2, "final_position"),
            "v2_strategy_return_net": numeric(v2, "strategy_return_net"),
            "v2_strategy_return_gross": numeric(v2, "strategy_return_gross"),
            "v2_fee_cost": numeric(v2, "fee_cost"),
            "v2_trade_size": numeric(v2, "trade_size"),
            "v2_regime": string(v2, "confirmed_regime"),
            "v2_raw_regime": string(v2, "raw_regime"),
            "v2_v1_position": numeric(v2, "v1_position"),
            "v2_position_before_control": numeric(v2, "position_before_control"),
            "v2_position_before_dd_gate": numeric(v2, "position_before_dd_gate"),
            "v2_weak_bull_cooldown_active": bool_series(v2, "weak_bull_cooldown_active"),
            "v2_weak_bull_cooldown_remaining": numeric(v2, "weak_bull_cooldown_remaining"),
            "v2_weak_bull_entry_attempt": bool_series(v2, "weak_bull_entry_attempt"),
            "v2_weak_bull_entry_allowed": bool_series(v2, "weak_bull_entry_allowed"),
            "v2_weak_bull_entry_blocked": bool_series(v2, "weak_bull_entry_blocked"),
            "v2_exit_reason": string(v2, "exit_reason"),
            "v3_target_position": numeric(v3, "target_position"),
            "v3_executed_position": numeric(v3, "executed_position"),
            "v3_risk_action": string(v3, "risk_action"),
            "v3_binding_block_reason": string(v3, "binding_block_reason"),
            "v3_long_regime": string(v3, "long_regime"),
            "v3_short_regime": string(v3, "short_regime"),
            "v3_risk_state": string(v3, "risk_state"),
            "v3_volatility_state": string(v3, "volatility_state"),
            "v3_drawdown_state": string(v3, "drawdown_state"),
            "v3_confidence_score": numeric(v3, "confidence_score"),
            "v3_allow_entry": bool_series(v3, "allow_entry"),
            "v3_allow_hold": bool_series(v3, "allow_hold"),
        }
    )
    result["v2_new_entry"] = (result["v2_position"] > 0.0) & (result["v2_position"].shift(1).fillna(0.0) <= 0.0)
    result["v2_exit"] = (result["v2_position"] <= 0.0) & (result["v2_position"].shift(1).fillna(0.0) > 0.0)
    result["v2_hold"] = (result["v2_position"] > 0.0) & (~result["v2_new_entry"])
    result["v2_long"] = result["v2_position"] > 0.0
    result["v3_target_positive"] = result["v3_target_position"] > 0.0
    result["v3_executed_positive"] = result["v3_executed_position"] > 0.0
    result["v2_gate_state"] = result["v2_regime"].map(v2_gate_state).fillna("unknown")
    result["v2_sideways_hold_context"] = (result["v2_regime"] == "sideways") & (result["v2_position"] > 0.0)
    result["v2_bear_exit_context"] = result["v2_regime"].isin(["bear", "strong_bear"]) & result["v2_exit"]
    result["v3_trend_health_label"] = result.apply(v3_trend_label_from_prefixed_row, axis=1)
    result["overlap_group"] = np.select(
        [
            result["v2_long"] & result["v3_target_positive"],
            result["v2_long"] & (~result["v3_target_positive"]),
            (~result["v2_long"]) & result["v3_target_positive"],
            (~result["v2_long"]) & (~result["v3_target_positive"]),
        ],
        [
            "v2_long_v3_target_positive",
            "v2_long_v3_target_zero",
            "v2_flat_v3_target_positive",
            "v2_flat_v3_target_zero",
        ],
        default="unknown",
    )
    for horizon in (1, 6, 24, 72):
        result[f"next_{horizon}_bar_return"] = forward_compound_return(result["asset_return"], horizon)
    return result


def build_trade_level(bar_level: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if bar_level.empty:
        return pd.DataFrame()
    for keys, group in bar_level.groupby(["asset", "dataset", "fee_rate"], dropna=False):
        asset, dataset, fee_rate = keys
        group = group.reset_index(drop=True).copy()
        position = group["v2_position"] > 0.0
        entry_idx: int | None = None
        for idx, is_active in enumerate(position):
            was_active = bool(position.iloc[idx - 1]) if idx > 0 else False
            if bool(is_active) and not was_active:
                entry_idx = idx
            if was_active and not bool(is_active) and entry_idx is not None:
                rows.append(build_trade_row(asset, dataset, float(fee_rate), group, entry_idx, idx))
                entry_idx = None
        if entry_idx is not None:
            rows.append(build_trade_row(asset, dataset, float(fee_rate), group, entry_idx, len(group) - 1))
    return pd.DataFrame(rows)


def build_trade_row(asset: str, dataset: str, fee_rate: float, frame: pd.DataFrame, entry_idx: int, exit_idx: int) -> dict[str, Any]:
    trade = frame.iloc[entry_idx : exit_idx + 1].copy()
    entry = frame.iloc[entry_idx]
    exit_row = frame.iloc[exit_idx]
    entry_price = float(entry["close"])
    exit_price = float(exit_row["close"])
    path_return = pd.to_numeric(trade["close"], errors="coerce") / entry_price - 1.0 if entry_price > 0.0 else pd.Series(0.0, index=trade.index)
    return {
        "asset": asset,
        "dataset": dataset,
        "fee_rate": fee_rate,
        "entry_idx": entry_idx,
        "exit_idx": exit_idx,
        "entry_time": entry["timestamp"],
        "exit_time": exit_row["timestamp"],
        "holding_bars": exit_idx - entry_idx + 1,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "gross_return": exit_price / entry_price - 1.0 if entry_price > 0.0 else 0.0,
        "net_return": float(pd.to_numeric(trade["v2_strategy_return_net"], errors="coerce").fillna(0.0).sum()),
        "max_favorable_excursion": float(path_return.max()),
        "max_adverse_excursion": float(path_return.min()),
        "entry_regime": entry["v2_regime"],
        "exit_regime": exit_row["v2_regime"],
        "entry_v2_gate_state": entry["v2_gate_state"],
        "exit_reason": first_nonempty(exit_row.get("v2_exit_reason", ""), "position_exit"),
        "weak_bull_cooldown_state_at_entry": bool(entry["v2_weak_bull_cooldown_active"]),
        "cooldown_remaining_at_entry": float(entry["v2_weak_bull_cooldown_remaining"]),
        "entry_after_cooldown": bool(entry_idx > 0 and frame["v2_weak_bull_cooldown_active"].iloc[max(0, entry_idx - 24) : entry_idx].any()),
        "strong_bull_context": bool(entry["v2_regime"] == "strong_bull"),
        "weak_bull_context": bool(entry["v2_regime"] == "weak_bull"),
        "sideways_hold_context": bool(trade["v2_sideways_hold_context"].any()),
        "bear_exit_context": bool(trade["v2_bear_exit_context"].any()),
        "v3_long_regime_at_entry": entry["v3_long_regime"],
        "v3_short_regime_at_entry": entry["v3_short_regime"],
        "v3_trend_health_label_at_entry": entry["v3_trend_health_label"],
        "v3_target_position_at_entry": float(entry["v3_target_position"]),
        "v3_risk_action_at_entry": entry["v3_risk_action"],
        "v3_binding_block_reason_at_entry": entry["v3_binding_block_reason"],
    }


def build_regime_timing(bar_level: pd.DataFrame, trade_level: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.extend(group_forward_rows(bar_level, ["asset", "dataset", "fee_rate", "v2_regime"], "v2_regime_bar_forward"))
    rows.extend(group_forward_rows(bar_level, ["asset", "dataset", "fee_rate", "v2_gate_state"], "v2_gate_state_bar_forward"))
    rows.extend(group_forward_rows(bar_level, ["asset", "dataset", "fee_rate", "overlap_group"], "v2_v3_overlap_forward"))
    rows.extend(group_forward_rows(bar_level, ["asset", "dataset", "fee_rate", "v3_long_regime"], "v3_long_regime_forward"))
    rows.extend(group_forward_rows(bar_level, ["asset", "dataset", "fee_rate", "v3_trend_health_label"], "v3_label_forward"))
    rows.extend(special_group_rows(bar_level))
    rows.extend(entry_delay_rows(bar_level))
    rows.extend(exit_timing_rows(bar_level))
    rows.extend(trade_summary_rows(trade_level))
    return pd.DataFrame(rows)


def build_signal_component_tables(bar_level: pd.DataFrame, trade_level: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.extend(component_proxy_rows(bar_level))
    rows.extend(btc_eth_contrast_rows(bar_level, trade_level))
    rows.extend(hidden_state_target_rows())
    return pd.DataFrame(rows)


def group_forward_rows(frame: pd.DataFrame, group_cols: list[str], audit_type: str) -> list[dict[str, Any]]:
    rows = []
    for keys, group in frame.groupby(group_cols, dropna=False):
        row = keys_to_row(group_cols, keys)
        row["audit_type"] = audit_type
        row.update(forward_stats(group))
        row["v2_realized_contribution"] = sum_numeric(group, "v2_strategy_return_net")
        row["v3_avg_target_position"] = mean_numeric(group, "v3_target_position")
        row["v3_avg_executed_position"] = mean_numeric(group, "v3_executed_position")
        rows.append(row)
    return rows


def special_group_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    definitions = {
        "v2_long": frame["v2_long"],
        "v2_flat": ~frame["v2_long"],
        "v2_new_entry": frame["v2_new_entry"],
        "v2_hold": frame["v2_hold"],
        "v2_exit": frame["v2_exit"],
        "v2_long_v3_target_zero": frame["overlap_group"] == "v2_long_v3_target_zero",
        "v2_long_v3_target_positive": frame["overlap_group"] == "v2_long_v3_target_positive",
        "v2_flat_v3_target_positive": frame["overlap_group"] == "v2_flat_v3_target_positive",
        "v2_long_healthy_bull": frame["v2_long"] & (frame["v3_trend_health_label"] == "healthy_bull"),
        "v2_long_non_healthy_bull": frame["v2_long"] & (frame["v3_trend_health_label"] != "healthy_bull"),
        "v2_long_v3_neutral": frame["v2_long"] & (frame["v3_long_regime"] == "neutral"),
        "v2_long_v3_bear_or_strong_bear": frame["v2_long"] & frame["v3_long_regime"].isin(["bear", "strong_bear"]),
    }
    rows = []
    for (asset, dataset, fee_rate), group in frame.groupby(["asset", "dataset", "fee_rate"], dropna=False):
        base_index = group.index
        for name, mask in definitions.items():
            subset = group[mask.loc[base_index]]
            row = {
                "audit_type": "special_bar_slice",
                "asset": asset,
                "dataset": dataset,
                "fee_rate": fee_rate,
                "slice": name,
            }
            row.update(forward_stats(subset))
            row["v2_realized_contribution"] = sum_numeric(subset, "v2_strategy_return_net")
            rows.append(row)
    return rows


def entry_delay_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    horizons = (6, 12, 24, 48, 72)
    for keys, group in frame.groupby(["asset", "dataset", "fee_rate"], dropna=False):
        group = group.reset_index(drop=True)
        entries = group[group["v2_new_entry"]].copy()
        target = group["v3_target_positive"].astype(bool)
        for _, entry in entries.iterrows():
            idx = int(entry["bar_index"])
            future_target_positions = np.flatnonzero(target.iloc[idx:].to_numpy())
            delay = int(future_target_positions[0]) if len(future_target_positions) else np.nan
            if pd.isna(delay):
                delay_return = np.nan
            else:
                delay_end = min(idx + int(delay), len(group) - 1)
                delay_return = compound_return(group["asset_return"].iloc[idx + 1 : delay_end + 1])
            row = keys_to_row(["asset", "dataset", "fee_rate"], keys)
            row.update(
                {
                    "audit_type": "v2_entry_delay_to_v3_target",
                    "entry_idx": idx,
                    "v3_target_positive_at_entry": bool(entry["v3_target_positive"]),
                    "delay_to_v3_target_positive": delay,
                    "return_before_v3_target_positive": delay_return,
                    "v3_long_regime_at_entry": entry["v3_long_regime"],
                    "v3_trend_health_label_at_entry": entry["v3_trend_health_label"],
                    "v2_regime_at_entry": entry["v2_regime"],
                }
            )
            for horizon in horizons:
                future = target.shift(-1).rolling(horizon, min_periods=1).max().shift(-(horizon - 1)).fillna(False)
                row[f"v3_target_positive_within_{horizon}"] = bool(future.iloc[idx])
            rows.append(row)
    return rows


def exit_timing_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for keys, group in frame.groupby(["asset", "dataset", "fee_rate"], dropna=False):
        group = group.reset_index(drop=True)
        exits = group[group["v2_exit"]].copy()
        for _, exit_row in exits.iterrows():
            idx = int(exit_row["bar_index"])
            row = keys_to_row(["asset", "dataset", "fee_rate"], keys)
            row.update(
                {
                    "audit_type": "v2_exit_timing",
                    "exit_idx": idx,
                    "v2_regime_at_exit": exit_row["v2_regime"],
                    "v2_exit_reason": exit_row["v2_exit_reason"],
                    "v3_target_positive_at_exit": bool(exit_row["v3_target_positive"]),
                    "v3_executed_positive_at_exit": bool(exit_row["v3_executed_positive"]),
                    "v3_long_regime_at_exit": exit_row["v3_long_regime"],
                    "next_6_return_after_exit": exit_row["next_6_bar_return"],
                    "next_24_return_after_exit": exit_row["next_24_bar_return"],
                    "next_72_return_after_exit": exit_row["next_72_bar_return"],
                }
            )
            rows.append(row)
    return rows


def trade_summary_rows(trade_level: pd.DataFrame) -> list[dict[str, Any]]:
    if trade_level.empty:
        return []
    rows = []
    for column in ["entry_regime", "entry_v2_gate_state", "v3_long_regime_at_entry", "v3_trend_health_label_at_entry"]:
        for keys, group in trade_level.groupby(["asset", "dataset", "fee_rate", column], dropna=False):
            row = keys_to_row(["asset", "dataset", "fee_rate", "bucket"], keys)
            row["audit_type"] = f"trade_summary_by_{column}"
            row["trade_count"] = int(len(group))
            row["avg_net_return"] = mean_numeric(group, "net_return")
            row["win_rate"] = mean_bool(pd.to_numeric(group["net_return"], errors="coerce") > 0.0)
            row["avg_holding_bars"] = mean_numeric(group, "holding_bars")
            row["total_net_contribution"] = sum_numeric(group, "net_return")
            row["avg_mfe"] = mean_numeric(group, "max_favorable_excursion")
            row["avg_mae"] = mean_numeric(group, "max_adverse_excursion")
            rows.append(row)
    return rows


def component_proxy_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    definitions = {
        "base_v1_core_signal_long": frame["v2_v1_position"] > 0.0,
        "v2_regime_gate_entry_allowed": frame["v2_gate_state"].isin(["entry_and_hold"]),
        "weak_bull_cooldown_active": frame["v2_weak_bull_cooldown_active"],
        "sideways_hold_only_active": frame["v2_sideways_hold_context"],
        "bear_no_hold_exit": frame["v2_bear_exit_context"],
        "strong_bull_entry_permission": frame["v2_regime"].eq("strong_bull") & (frame["v2_v1_position"] > 0.0),
        "weak_bull_entry_permission": frame["v2_regime"].eq("weak_bull") & (frame["v2_v1_position"] > 0.0),
    }
    for (asset, dataset, fee_rate), group in frame.groupby(["asset", "dataset", "fee_rate"], dropna=False):
        idx = group.index
        for component, mask in definitions.items():
            subset = group[mask.loc[idx]]
            row = {
                "audit_type": "component_proxy_forward",
                "asset": asset,
                "dataset": dataset,
                "fee_rate": fee_rate,
                "component": component,
                "available_as": "logged_state_proxy",
            }
            row.update(forward_stats(subset))
            row["v2_realized_contribution"] = sum_numeric(subset, "v2_strategy_return_net")
            rows.append(row)
    rows.append(
        {
            "audit_type": "component_availability",
            "component": "continuous_v1_core_score",
            "available_as": "unavailable; v2 output exposes binary v1_position proxy only",
        }
    )
    return rows


def btc_eth_contrast_rows(bar_level: pd.DataFrame, trade_level: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    rows.extend(group_forward_rows(bar_level, ["asset", "fee_rate", "v2_regime"], "btc_eth_v2_regime_contrast"))
    if not trade_level.empty:
        for keys, group in trade_level.groupby(["asset", "fee_rate", "entry_regime"], dropna=False):
            row = keys_to_row(["asset", "fee_rate", "entry_regime"], keys)
            row["audit_type"] = "btc_eth_trade_regime_contrast"
            row["trade_count"] = int(len(group))
            row["avg_net_return"] = mean_numeric(group, "net_return")
            row["win_rate"] = mean_bool(pd.to_numeric(group["net_return"], errors="coerce") > 0.0)
            row["total_net_contribution"] = sum_numeric(group, "net_return")
            rows.append(row)
    return rows


def hidden_state_target_rows() -> list[dict[str, Any]]:
    states = [
        ("early_trend_recovery", "v2 entries during v3 neutral/bear before v3 target confirmation", "entry"),
        ("healthy_continuation", "v3 healthy_bull and v2-long continuation slices", "entry/hold"),
        ("overextended_trend", "v3 strong_bull split into late/overheated diagnostics", "exit/risk control"),
        ("oversold_rebound", "v2-long while v3 bear/strong_bear with recovery/noise", "cautious entry"),
        ("failed_weak_bull", "weak_bull losing-trade cooldown mechanism", "risk control"),
        ("sideways_hold_state", "v2 sideways-hold-only behavior preserves positions without fresh entries", "hold"),
        ("trend_exhaustion", "v2 exits and v3 strong_bull/overheat weakness diagnostics", "exit"),
    ]
    return [
        {
            "audit_type": "candidate_hidden_state_target",
            "hidden_state": state,
            "evidence": evidence,
            "possible_observable_features": "v2_position/v1_position proxy, v2 regime, v3 regime, trend_health_label, momentum, drawdown, volatility",
            "likely_scope": "BTC-primary until ETH validates",
            "intended_use": intended_use,
        }
        for state, evidence, intended_use in states
    ]


def write_report(bar_level: pd.DataFrame, trade_level: pd.DataFrame, regime_timing: pd.DataFrame, signal_components: pd.DataFrame) -> None:
    btc_slices = aggregate_special_slices(regime_timing, "BTC")
    eth_slices = aggregate_special_slices(regime_timing, "ETH")
    btc_trades = aggregate_trade_summary(regime_timing, "BTC", "trade_summary_by_entry_regime")
    eth_trades = aggregate_trade_summary(regime_timing, "ETH", "trade_summary_by_entry_regime")
    delay_summary = summarize_entry_delays(regime_timing)
    exit_summary = summarize_exits(regime_timing)
    component_summary = summarize_components(signal_components)

    best_trade = best_row(btc_trades, "total_net_contribution")
    best_slice = best_row(btc_slices, "avg_next_24_bar_return")
    missed = btc_slices[btc_slices.get("slice", pd.Series(dtype=str)).eq("v2_long_v3_target_zero")]
    missed_next24 = weighted_mean(missed, "avg_next_24_bar_return", "count") if not missed.empty else np.nan
    missed_hit = weighted_mean(missed, "hit_rate_next_24", "count") if not missed.empty else np.nan

    lines = [
        "# v2 Alpha Extraction Study",
        "",
        "Diagnostics-only study. v2 and v3 trading behavior were not changed. Forward returns are post-run analysis only and were not used to make trading decisions.",
        "",
        "## Executive summary",
        f"- CSV outputs: `{TRADE_LEVEL_CSV}`, `{BAR_LEVEL_CSV}`, `{REGIME_TIMING_CSV}`, `{SIGNAL_COMPONENTS_CSV}`.",
        f"- BTC clearest missed-opportunity slice remains `v2_long_v3_target_zero`: avg next-24 return `{fmt(missed_next24)}`, hit rate `{fmt(missed_hit)}`.",
        f"- Best BTC trade contribution bucket by entry regime: `{best_trade.get('bucket', 'unavailable')}` with total net contribution `{fmt(best_trade.get('total_net_contribution'))}`.",
        f"- Best BTC bar slice by next-24 forward return: `{best_slice.get('slice', best_slice.get('v2_regime', 'unavailable'))}` with avg next-24 `{fmt(best_slice.get('avg_next_24_bar_return'))}`.",
        "- v2 internals expose binary `v1_position`, v2 regime, cooldown, entry/exit flags, and final position. A continuous v1/core score is unavailable, so this report uses `v1_position` as the v1.final proxy.",
        "",
        "## v2 trade-level decomposition",
        table_md(select_cols(btc_trades, ["bucket", "trade_count", "avg_net_return", "win_rate", "avg_holding_bars", "total_net_contribution"]).head(12)),
        "",
        "## v2 bar-level signal slice audit",
        "BTC slices:",
        table_md(select_cols(btc_slices, ["slice", "count", "avg_next_1_bar_return", "avg_next_6_bar_return", "avg_next_24_bar_return", "avg_next_72_bar_return", "hit_rate_next_24", "sharpe_like_next_24", "v2_realized_contribution"]).head(20)),
        "",
        "ETH contrast slices:",
        table_md(select_cols(eth_slices, ["slice", "count", "avg_next_24_bar_return", "hit_rate_next_24", "v2_realized_contribution"]).head(20)),
        "",
        "## v2 entry timing versus v3",
        table_md(select_cols(delay_summary, ["asset", "avg_delay_to_v3_target_positive", "median_delay_to_v3_target_positive", "avg_return_before_v3_target_positive", "target_positive_at_entry_rate", "target_positive_within_24_rate"]).head(10)),
        "",
        "Interpretation: low same-bar v3 confirmation plus positive delay-period returns supports the view that v2's BTC edge is meaningfully entry-timing driven.",
        "",
        "## v2 exit timing versus v3",
        table_md(select_cols(exit_summary, ["asset", "exit_count", "avg_next_24_return_after_exit", "negative_next_24_rate_after_exit", "v3_target_positive_at_exit_rate"]).head(10)),
        "",
        "## v2 component importance",
        table_md(select_cols(component_summary, ["asset", "component", "count", "avg_next_24_bar_return", "hit_rate_next_24", "v2_realized_contribution"]).head(30)),
        "",
        "The component section is proxy-based, not a true causal ablation. Existing logs are enough to inspect v1 binary signal, v2 regimes, sideways hold, weak-bull cooldown, and exit contexts, but not a continuous v1 score or every internal gate threshold.",
        "",
        "## BTC vs ETH contrast",
        "BTC remains the primary validated asset. ETH rows are included as contrast; any v2 behavior that looks strong on BTC but weak or inconsistent on ETH should be treated as BTC-specific until independently validated.",
        "",
        "BTC trade buckets:",
        table_md(select_cols(btc_trades, ["bucket", "trade_count", "avg_net_return", "win_rate", "total_net_contribution"]).head(10)),
        "",
        "ETH trade buckets:",
        table_md(select_cols(eth_trades, ["bucket", "trade_count", "avg_net_return", "win_rate", "total_net_contribution"]).head(10)),
        "",
        "## Candidate hidden-state targets for v4",
        table_md(select_cols(signal_components[signal_components["audit_type"].eq("candidate_hidden_state_target")], ["hidden_state", "evidence", "intended_use", "likely_scope"])),
        "",
        "## Main conclusions",
        "1. v2's BTC alpha appears more entry-timing driven than pure exposure-driven: the missed `v2_long_v3_target_zero` slice has positive next-24 diagnostics, while prior v3 audits showed `v2_long_v3_target_positive` was weak.",
        "2. v2's useful behavior is not simply higher exposure. It enters in states that v3 often labels neutral/bearish or not-yet-confirmed, then v3 becomes positive later.",
        "3. The most importable ideas are hidden-state targets: early trend recovery, healthy continuation, sideways hold, failed weak-bull control, and trend exhaustion.",
        "4. What should not be imported blindly: direct v2-position replacement, broad risk relaxation, or BTC-only weak-bull behavior as a cross-asset rule.",
        "5. Recommended next step: design a v4 hidden-state estimator specification using these extracted target states before running new trading variants.",
        "",
        "## Files generated",
        f"- `{TRADE_LEVEL_CSV}`",
        f"- `{BAR_LEVEL_CSV}`",
        f"- `{REGIME_TIMING_CSV}`",
        f"- `{SIGNAL_COMPONENTS_CSV}`",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_special_slices(regime_timing: pd.DataFrame, asset: str) -> pd.DataFrame:
    subset = regime_timing[(regime_timing["audit_type"] == "special_bar_slice") & (regime_timing["asset"] == asset)].copy()
    if subset.empty:
        return subset
    return (
        subset.groupby("slice", dropna=False)
        .apply(weighted_aggregate, include_groups=False)
        .reset_index()
        .sort_values("avg_next_24_bar_return", ascending=False)
    )


def aggregate_trade_summary(regime_timing: pd.DataFrame, asset: str, audit_type: str) -> pd.DataFrame:
    subset = regime_timing[(regime_timing["audit_type"] == audit_type) & (regime_timing["asset"] == asset)].copy()
    if subset.empty:
        return subset
    return (
        subset.groupby("bucket", dropna=False)
        .agg(
            trade_count=("trade_count", "sum"),
            avg_net_return=("avg_net_return", "mean"),
            win_rate=("win_rate", "mean"),
            avg_holding_bars=("avg_holding_bars", "mean"),
            total_net_contribution=("total_net_contribution", "sum"),
        )
        .reset_index()
        .sort_values("total_net_contribution", ascending=False)
    )


def summarize_entry_delays(regime_timing: pd.DataFrame) -> pd.DataFrame:
    subset = regime_timing[regime_timing["audit_type"] == "v2_entry_delay_to_v3_target"].copy()
    if subset.empty:
        return subset
    rows = []
    for asset, group in subset.groupby("asset", dropna=False):
        rows.append(
            {
                "asset": asset,
                "entry_count": int(len(group)),
                "avg_delay_to_v3_target_positive": mean_numeric(group, "delay_to_v3_target_positive"),
                "median_delay_to_v3_target_positive": median_numeric(group, "delay_to_v3_target_positive"),
                "avg_return_before_v3_target_positive": mean_numeric(group, "return_before_v3_target_positive"),
                "target_positive_at_entry_rate": mean_bool(group["v3_target_positive_at_entry"]),
                "target_positive_within_24_rate": mean_bool(group["v3_target_positive_within_24"]),
            }
        )
    return pd.DataFrame(rows)


def summarize_exits(regime_timing: pd.DataFrame) -> pd.DataFrame:
    subset = regime_timing[regime_timing["audit_type"] == "v2_exit_timing"].copy()
    if subset.empty:
        return subset
    rows = []
    for asset, group in subset.groupby("asset", dropna=False):
        next24 = pd.to_numeric(group["next_24_return_after_exit"], errors="coerce")
        rows.append(
            {
                "asset": asset,
                "exit_count": int(len(group)),
                "avg_next_24_return_after_exit": mean_numeric(group, "next_24_return_after_exit"),
                "negative_next_24_rate_after_exit": mean_bool(next24 < 0.0),
                "v3_target_positive_at_exit_rate": mean_bool(group["v3_target_positive_at_exit"]),
            }
        )
    return pd.DataFrame(rows)


def summarize_components(signal_components: pd.DataFrame) -> pd.DataFrame:
    subset = signal_components[signal_components["audit_type"] == "component_proxy_forward"].copy()
    if subset.empty:
        return subset
    return (
        subset.groupby(["asset", "component"], dropna=False)
        .apply(weighted_aggregate, include_groups=False)
        .reset_index()
        .sort_values(["asset", "avg_next_24_bar_return"], ascending=[True, False])
    )


def weighted_aggregate(group: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            "count": int(pd.to_numeric(group["count"], errors="coerce").fillna(0).sum()),
            "avg_next_1_bar_return": weighted_mean(group, "avg_next_1_bar_return", "count"),
            "avg_next_6_bar_return": weighted_mean(group, "avg_next_6_bar_return", "count"),
            "avg_next_24_bar_return": weighted_mean(group, "avg_next_24_bar_return", "count"),
            "avg_next_72_bar_return": weighted_mean(group, "avg_next_72_bar_return", "count"),
            "hit_rate_next_24": weighted_mean(group, "hit_rate_next_24", "count"),
            "sharpe_like_next_24": weighted_mean(group, "sharpe_like_next_24", "count"),
            "v2_realized_contribution": sum_numeric(group, "v2_realized_contribution"),
        }
    )


def forward_stats(group: pd.DataFrame) -> dict[str, Any]:
    next24 = pd.to_numeric(group.get("next_24_bar_return", pd.Series(dtype=float)), errors="coerce")
    return {
        "count": int(len(group)),
        "avg_next_1_bar_return": mean_numeric(group, "next_1_bar_return"),
        "avg_next_6_bar_return": mean_numeric(group, "next_6_bar_return"),
        "avg_next_24_bar_return": mean_numeric(group, "next_24_bar_return"),
        "avg_next_72_bar_return": mean_numeric(group, "next_72_bar_return"),
        "median_next_24_bar_return": float(next24.median()) if next24.notna().any() else np.nan,
        "hit_rate_next_24": mean_bool(next24 > 0.0),
        "vol_next_24_return": float(next24.std(ddof=0)) if next24.notna().any() else np.nan,
        "sharpe_like_next_24": safe_ratio(float(next24.mean()) if next24.notna().any() else np.nan, float(next24.std(ddof=0)) if next24.notna().any() else np.nan),
    }


def v3_trend_label_from_prefixed_row(row: pd.Series) -> str:
    proxy = pd.Series(
        {
            "long_regime": row.get("v3_long_regime"),
            "short_regime": row.get("v3_short_regime"),
            "volatility_state": row.get("v3_volatility_state"),
            "drawdown_state": row.get("v3_drawdown_state"),
            "confidence_score": row.get("v3_confidence_score"),
            "target_position": row.get("v3_target_position"),
            "v2_position": row.get("v2_position"),
        }
    )
    return assign_trend_health_label(proxy)


def v2_gate_state(regime: str) -> str:
    if regime in {"strong_bull", "weak_bull"}:
        return "entry_and_hold"
    if regime == "sideways":
        return "hold_only"
    if regime in {"bear", "strong_bear"}:
        return "blocked"
    return "unknown"


def forward_compound_return(returns: pd.Series, horizon: int) -> pd.Series:
    values = pd.to_numeric(returns, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    for idx in range(len(values)):
        end = idx + horizon
        if end < len(values):
            out[idx] = float(np.prod(1.0 + values[idx + 1 : end + 1]) - 1.0)
    return pd.Series(out, index=returns.index)


def compound_return(values: Iterable[float]) -> float:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if len(arr) == 0:
        return 0.0
    return float(np.prod(1.0 + arr) - 1.0)


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def string(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series("", index=frame.index, dtype=object)
    return frame[column].fillna("").astype(str)


def bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    values = frame[column]
    if values.dtype == bool:
        return values.fillna(False).astype(bool)
    return values.astype(str).str.lower().isin(["true", "1", "yes"])


def pick_string(preferred: pd.DataFrame, column: str, fallback_frame: pd.DataFrame, fallback: str = "") -> pd.Series:
    if column in preferred.columns:
        return preferred[column].fillna(fallback).astype(str)
    if column in fallback_frame.columns:
        return fallback_frame[column].fillna(fallback).astype(str)
    return pd.Series(fallback, index=preferred.index, dtype=object)


def keys_to_row(columns: list[str], keys: Any) -> dict[str, Any]:
    if not isinstance(keys, tuple):
        keys = (keys,)
    return dict(zip(columns, keys))


def mean_numeric(frame_or_series: pd.DataFrame | pd.Series, column: str | None = None) -> float:
    series = frame_or_series if column is None else frame_or_series.get(column, pd.Series(dtype=float))
    values = pd.to_numeric(series, errors="coerce")
    return float(values.mean()) if values.notna().any() else np.nan


def median_numeric(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame.get(column, pd.Series(dtype=float)), errors="coerce")
    return float(values.median()) if values.notna().any() else np.nan


def sum_numeric(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).sum())


def mean_bool(values: pd.Series) -> float:
    if len(values) == 0:
        return np.nan
    return float(pd.Series(values).fillna(False).astype(bool).mean())


def weighted_mean(frame: pd.DataFrame, value_col: str, weight_col: str) -> float:
    if frame.empty or value_col not in frame.columns or weight_col not in frame.columns:
        return np.nan
    values = pd.to_numeric(frame[value_col], errors="coerce")
    weights = pd.to_numeric(frame[weight_col], errors="coerce").fillna(0.0)
    mask = values.notna() & (weights > 0.0)
    if not mask.any():
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0.0:
        return np.nan
    return float(numerator / denominator)


def first_nonempty(value: Any, fallback: str) -> str:
    text = str(value) if value is not None else ""
    return text if text else fallback


def best_row(frame: pd.DataFrame, column: str) -> pd.Series:
    if frame.empty or column not in frame.columns:
        return pd.Series(dtype=object)
    values = pd.to_numeric(frame[column], errors="coerce")
    if not values.notna().any():
        return pd.Series(dtype=object)
    return frame.loc[values.idxmax()]


def select_cols(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame[[column for column in columns if column in frame.columns]].copy() if not frame.empty else pd.DataFrame(columns=columns)


def table_md(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows available._"
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_numeric_dtype(display[column]):
            display[column] = display[column].map(lambda value: fmt(value))
    headers = [str(column) for column in display.columns]
    rows = []
    rows.append("| " + " | ".join(headers) + " |")
    rows.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in display.iterrows():
        values = [str(row[column]).replace("|", "\\|") for column in display.columns]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "nan"
    return f"{number:.6g}"


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtester import load_ohlcv_csv, run_backtest_fast
from scripts.run_v3_final_candidate import (
    PERIODS_PER_YEAR,
    V1_ENTRY_THRESHOLD,
    input_frame_from_v1,
    load_final_candidate_config,
)
from v2_small_cap import backtest_v2_btc_final_candidate_a, backtest_v2_final_candidate_a


BTC_WATERFALL_CSV = Path("reports") / "v3_decision_waterfall_btc_1h.csv"
ETH_WATERFALL_CSV = Path("reports") / "v3_decision_waterfall_eth_1h.csv"
REPORT_PATH = Path("reports") / "v3_v2_signal_gap_analysis.md"
OVERLAP_CSV = Path("reports") / "v3_v2_signal_gap_overlap.csv"
MISSED_V2_LONG_CSV = Path("reports") / "v3_v2_signal_gap_missed_v2_long.csv"
TRANSITION_TIMING_CSV = Path("reports") / "v3_v2_signal_gap_transition_timing.csv"
REGIME_BREAKDOWN_CSV = Path("reports") / "v3_v2_signal_gap_regime_breakdown.csv"


def main() -> None:
    config = load_final_candidate_config()
    aligned = build_aligned_frames(config)
    overlap = build_overlap_table(aligned)
    missed = build_missed_v2_long_table(aligned)
    transitions = build_transition_timing_table(aligned)
    regime = build_regime_breakdown_table(aligned)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    overlap.to_csv(OVERLAP_CSV, index=False)
    missed.to_csv(MISSED_V2_LONG_CSV, index=False)
    transitions.to_csv(TRANSITION_TIMING_CSV, index=False)
    regime.to_csv(REGIME_BREAKDOWN_CSV, index=False)
    write_report(overlap, missed, transitions, regime)
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {OVERLAP_CSV}")
    print(f"Wrote {MISSED_V2_LONG_CSV}")
    print(f"Wrote {TRANSITION_TIMING_CSV}")
    print(f"Wrote {REGIME_BREAKDOWN_CSV}")


def build_aligned_frames(config: dict[str, Any]) -> pd.DataFrame:
    frames = []
    v3_sources = {"BTC": BTC_WATERFALL_CSV, "ETH": ETH_WATERFALL_CSV}
    for asset, dataset_key in [("BTC", "btc_datasets"), ("ETH", "eth_datasets")]:
        source = v3_sources[asset]
        if not source.exists():
            print(f"missing v3 waterfall CSV for {asset}: {source}")
            continue
        print(f"loading {source}")
        v3_all = pd.read_csv(source, low_memory=False)
        v3_all["asset"] = asset
        v3_all = add_forward_returns(v3_all)
        paths = [Path(path) for path in config["validation"].get(dataset_key, []) if Path(path).exists()]
        for path in paths:
            dataset = path.stem
            data = load_ohlcv_csv(path)
            for fee_rate in sorted(v3_all.loc[v3_all["dataset"] == dataset, "fee_rate"].unique()):
                subset = v3_all[(v3_all["dataset"] == dataset) & (v3_all["fee_rate"] == fee_rate)].copy().reset_index(drop=True)
                if subset.empty:
                    continue
                print(f"align asset={asset} dataset={dataset} fee={fee_rate:g}")
                v2 = run_v2_frame(asset, data, float(fee_rate)).reset_index(drop=True)
                frames.append(align_v2_v3(subset, v2, asset, dataset, float(fee_rate)))
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
    return v2_func(v2_input, fee_rate=fee_rate, v1_entry_threshold=V1_ENTRY_THRESHOLD, cooldown_bars=120)


def align_v2_v3(v3: pd.DataFrame, v2: pd.DataFrame, asset: str, dataset: str, fee_rate: float) -> pd.DataFrame:
    length = min(len(v3), len(v2))
    result = v3.iloc[:length].copy().reset_index(drop=True)
    v2 = v2.iloc[:length].copy().reset_index(drop=True)
    result["asset"] = asset
    result["dataset"] = dataset
    result["fee_rate"] = fee_rate
    result["v2_position"] = numeric_column(v2, "final_position")
    result["v2_strategy_return_net"] = numeric_column(v2, "strategy_return_net")
    result["v2_regime"] = string_column(v2, "confirmed_regime")
    result["v2_raw_regime"] = string_column(v2, "raw_regime")
    result["v2_v1_position"] = numeric_column(v2, "v1_position")
    result["v2_cooldown_active"] = bool_column(v2, "cooldown_active")
    result["v2_cooldown_remaining"] = numeric_column(v2, "cooldown_remaining")
    result["v2_entries_disabled_by_dd"] = bool_column(v2, "entries_disabled_by_dd")
    result["v3_strategy_return_net_proxy"] = (
        numeric_column(result, "current_position") * numeric_column(result, "asset_return")
        - numeric_column(result, "trade_amount") * fee_rate
    )
    result["v2_long"] = result["v2_position"] > 0.0
    result["v3_target_positive"] = pd.to_numeric(result["target_position"], errors="coerce").fillna(0.0) > 0.0
    result["v3_target_zero"] = ~result["v3_target_positive"]
    result["overlap_group"] = np.select(
        [
            result["v2_long"] & result["v3_target_positive"],
            result["v2_long"] & result["v3_target_zero"],
            (~result["v2_long"]) & result["v3_target_positive"],
            (~result["v2_long"]) & result["v3_target_zero"],
        ],
        [
            "v2_long_v3_target_positive",
            "v2_long_v3_target_zero",
            "v2_flat_v3_target_positive",
            "v2_flat_v3_target_zero",
        ],
        default="unknown",
    )
    result["confidence_bucket"] = pd.cut(
        pd.to_numeric(result["confidence_score"], errors="coerce"),
        bins=[-np.inf, 0.25, 0.50, 0.70, 0.85, np.inf],
        labels=["<=0.25", "0.25-0.50", "0.50-0.70", "0.70-0.85", ">0.85"],
    ).astype(str)
    return result


def add_forward_returns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["asset_return"] = pd.to_numeric(result["asset_return"], errors="coerce").fillna(0.0)
    keys = ["asset", "dataset", "fee_rate"]
    for horizon in (1, 6, 24, 72):
        result[f"next_{horizon}_bar_return"] = result.groupby(keys, group_keys=False)["asset_return"].transform(
            lambda series, h=horizon: series.shift(-1).rolling(h, min_periods=1).sum().shift(-(h - 1))
        )
    return result


def build_overlap_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(["asset", "dataset", "fee_rate", "overlap_group"], dropna=False):
        row = keys_to_row(["asset", "dataset", "fee_rate", "overlap_group"], keys)
        row.update(forward_stats(group))
        row["percentage_of_bars"] = len(group) / max(len(frame[(frame["asset"] == row["asset"]) & (frame["dataset"] == row["dataset"]) & (frame["fee_rate"] == row["fee_rate"])]), 1)
        row["v2_realized_contribution"] = sum_numeric(group, "v2_strategy_return_net")
        row["v3_realized_contribution"] = sum_numeric(group, "v3_strategy_return_net_proxy")
        rows.append(row)
    return pd.DataFrame(rows)


def build_missed_v2_long_table(frame: pd.DataFrame) -> pd.DataFrame:
    missed = frame[frame["overlap_group"] == "v2_long_v3_target_zero"].copy()
    rows = []
    for column in ["long_regime", "short_regime", "risk_state", "volatility_state", "drawdown_state", "confidence_bucket", "allow_entry", "allow_hold"]:
        for keys, group in missed.groupby(["asset", "dataset", "fee_rate", column], dropna=False):
            row = keys_to_row(["asset", "dataset", "fee_rate", "bucket"], keys)
            row["breakdown"] = column
            row.update(forward_stats(group, horizons=(24, 72)))
            rows.append(row)
    return pd.DataFrame(rows)


def build_transition_timing_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    horizons = (6, 12, 24, 48, 72)
    for keys, group in frame.groupby(["asset", "dataset", "fee_rate"], dropna=False):
        group = group.reset_index(drop=True).copy()
        target = group["v3_target_positive"].astype(bool)
        missed_mask = group["overlap_group"].eq("v2_long_v3_target_zero")
        for horizon in horizons:
            future = target.shift(-1).rolling(horizon, min_periods=1).max().shift(-(horizon - 1)).fillna(False).astype(bool)
            previous = target.shift(1).rolling(horizon, min_periods=1).max().fillna(False).astype(bool)
            delay_return = group["asset_return"].shift(-1).rolling(horizon, min_periods=1).sum().shift(-(horizon - 1))
            subset = group[missed_mask]
            row = keys_to_row(["asset", "dataset", "fee_rate"], keys)
            row.update(
                {
                    "horizon_bars": horizon,
                    "missed_count": int(len(subset)),
                    "becomes_target_positive_within_horizon_count": int(future[missed_mask].sum()),
                    "becomes_target_positive_within_horizon_rate": mean(future[missed_mask].astype(float)),
                    "was_target_positive_previous_horizon_count": int(previous[missed_mask].sum()),
                    "was_target_positive_previous_horizon_rate": mean(previous[missed_mask].astype(float)),
                    "avg_return_during_delay": mean(delay_return[missed_mask]),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def build_regime_breakdown_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    btc = frame[frame["asset"] == "BTC"].copy()
    rows.extend(bull_noise_rows(btc))
    rows.extend(strong_bull_failure_rows(btc))
    rows.extend(target_half_rows(btc))
    rows.extend(v2_condition_rows(frame[frame["overlap_group"] == "v2_long_v3_target_zero"].copy()))
    return pd.DataFrame(rows)


def bull_noise_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    subset = frame[(frame["long_regime"] == "bull") & (frame["short_regime"] == "noise")]
    rows = []
    for keys, group in subset.groupby(["dataset", "fee_rate", "overlap_group"], dropna=False):
        row = keys_to_row(["dataset", "fee_rate", "overlap_group"], keys)
        row["asset"] = "BTC"
        row["audit_type"] = "btc_bull_noise_vs_v2"
        row["v2_long_count"] = int(group["v2_long"].sum())
        row["v2_flat_count"] = int((~group["v2_long"]).sum())
        row["v3_target_0p50_count"] = int((pd.to_numeric(group["target_position"], errors="coerce") == 0.5).sum())
        row["v3_executed_positive_count"] = int((pd.to_numeric(group["executed_position"], errors="coerce") > 0.0).sum())
        row.update(forward_stats(group, horizons=(24,)))
        rows.append(row)
    return rows


def strong_bull_failure_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    subset = frame[frame["long_regime"] == "strong_bull"]
    rows = []
    for keys, group in subset.groupby(["dataset", "fee_rate", "short_regime", "v2_long"], dropna=False):
        row = keys_to_row(["dataset", "fee_rate", "short_regime", "v2_long"], keys)
        row["asset"] = "BTC"
        row["audit_type"] = "btc_strong_bull_failure"
        row.update(forward_stats(group, horizons=(24, 72)))
        row["avg_v3_target_position"] = mean_numeric(group, "target_position")
        row["avg_v3_executed_position"] = mean_numeric(group, "executed_position")
        rows.append(row)
    return rows


def target_half_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    subset = frame[pd.to_numeric(frame["target_position"], errors="coerce") == 0.5]
    rows = []
    for keys, group in subset.groupby(["dataset", "fee_rate", "long_regime", "short_regime", "v2_long"], dropna=False):
        row = keys_to_row(["dataset", "fee_rate", "long_regime", "short_regime", "v2_long"], keys)
        row["asset"] = "BTC"
        row["audit_type"] = "btc_target_0p50"
        row.update(forward_stats(group, horizons=(24,)))
        target = mean_numeric(group, "target_position")
        fee = float(row["fee_rate"])
        weighted_gross = target * row["avg_next_24_bar_return"]
        row["target_weighted_expected_gross_next_24"] = weighted_gross
        row["approx_round_trip_fee_cost"] = 2.0 * target * fee
        row["expected_net_edge_after_fee"] = weighted_gross - row["approx_round_trip_fee_cost"]
        rows.append(row)
    return rows


def v2_condition_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for column in ["v2_regime", "v2_raw_regime", "v2_v1_position", "v2_cooldown_active", "v2_entries_disabled_by_dd"]:
        for keys, group in frame.groupby(["asset", "dataset", "fee_rate", column], dropna=False):
            row = keys_to_row(["asset", "dataset", "fee_rate", "bucket"], keys)
            row["audit_type"] = "v2_long_v3_zero_v2_condition"
            row["condition"] = column
            row.update(forward_stats(group, horizons=(24,)))
            row["avg_v2_position"] = mean_numeric(group, "v2_position")
            row["avg_v3_target_position"] = mean_numeric(group, "target_position")
            rows.append(row)
    return rows


def write_report(overlap: pd.DataFrame, missed: pd.DataFrame, transitions: pd.DataFrame, regime: pd.DataFrame) -> None:
    overlap = normalize_next_bar_column(overlap)
    missed = normalize_next_bar_column(missed)
    regime = normalize_next_bar_column(regime)
    btc_overlap = aggregate_overlap(overlap, "BTC")
    eth_overlap = aggregate_overlap(overlap, "ETH")
    missed_highlights = aggregate_missed(missed, "BTC")
    transition = aggregate_transitions(transitions, "BTC")
    bull_noise = aggregate_regime(regime, "btc_bull_noise_vs_v2")
    strong_bull = aggregate_regime(regime, "btc_strong_bull_failure")
    target_half = aggregate_regime(regime, "btc_target_0p50")
    v2_conditions = aggregate_v2_conditions(regime)
    conclusions = build_conclusions(btc_overlap, missed_highlights, transition, strong_bull, target_half)

    lines = [
        "# v3 v2 Signal Gap Analysis",
        "",
        "This diagnostics-only report aligns v2 final-candidate positions with v3.final_candidate target decisions. No trading behavior, v2/v3 logic, risk rules, thresholds, particle filter, or leverage settings were changed. Forward returns are post-run diagnostics only.",
        "",
        "## 1. Main Conclusions",
        "",
        conclusions,
        "",
        "## 2. BTC Signal Overlap Matrix",
        "",
        _frame_to_markdown(btc_overlap),
        "",
        "## 3. ETH Signal Overlap Matrix",
        "",
        _frame_to_markdown(eth_overlap),
        "",
        "## 4. v2 Long / v3 Target Zero Breakdown",
        "",
        _frame_to_markdown(missed_highlights),
        "",
        "## 5. Transition Timing For v2 Long / v3 Target Zero",
        "",
        _frame_to_markdown(transition),
        "",
        "## 6. v2 Entry Logic Fields Available",
        "",
        "Available v2 proxy fields: `final_position`, `confirmed_regime`, `raw_regime`, `v1_position`, `cooldown_active`, `cooldown_remaining`, and `entries_disabled_by_dd`. A continuous v1 core score was not available in the v2 output, so `v1_position` is used as the v1.final signal proxy.",
        "",
        _frame_to_markdown(v2_conditions),
        "",
        "## 7. BTC Bull + Noise Audit Versus v2",
        "",
        _frame_to_markdown(bull_noise),
        "",
        "## 8. BTC Strong-Bull Failure Audit",
        "",
        _frame_to_markdown(strong_bull),
        "",
        "## 9. BTC Target=0.50 Audit",
        "",
        _frame_to_markdown(target_half),
        "",
        "## 10. Files",
        "",
        f"- Overlap CSV: `{OVERLAP_CSV}`",
        f"- Missed v2 long CSV: `{MISSED_V2_LONG_CSV}`",
        f"- Transition timing CSV: `{TRANSITION_TIMING_CSV}`",
        f"- Regime breakdown CSV: `{REGIME_BREAKDOWN_CSV}`",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def normalize_next_bar_column(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "avg_next_bar_return" not in result.columns and "avg_next_1_bar_return" in result.columns:
        result["avg_next_bar_return"] = result["avg_next_1_bar_return"]
    return result


def aggregate_overlap(frame: pd.DataFrame, asset: str) -> pd.DataFrame:
    subset = frame[frame["asset"] == asset]
    return (
        subset.groupby("overlap_group", dropna=False)
        .agg(
            avg_count=("count", "mean"),
            avg_percentage_of_bars=("percentage_of_bars", "mean"),
            avg_next_bar_return=("avg_next_bar_return", "mean"),
            avg_next_6_bar_return=("avg_next_6_bar_return", "mean"),
            avg_next_24_bar_return=("avg_next_24_bar_return", "mean"),
            avg_next_72_bar_return=("avg_next_72_bar_return", "mean"),
            median_next_24_bar_return=("median_next_24_bar_return", "mean"),
            hit_rate_next_24_gt_0=("hit_rate_next_24_gt_0", "mean"),
            volatility_next_24_return=("volatility_next_24_return", "mean"),
            avg_v2_realized_contribution=("v2_realized_contribution", "mean"),
            avg_v3_realized_contribution=("v3_realized_contribution", "mean"),
        )
        .reset_index()
        .sort_values("avg_next_24_bar_return", ascending=False)
    )


def aggregate_missed(frame: pd.DataFrame, asset: str) -> pd.DataFrame:
    subset = frame[frame["asset"] == asset]
    if subset.empty:
        return pd.DataFrame()
    return (
        subset.groupby(["breakdown", "bucket"], dropna=False)
        .agg(
            avg_count=("count", "mean"),
            avg_next_24_bar_return=("avg_next_24_bar_return", "mean"),
            hit_rate_next_24_gt_0=("hit_rate_next_24_gt_0", "mean"),
            avg_next_72_bar_return=("avg_next_72_bar_return", "mean"),
        )
        .reset_index()
        .sort_values(["avg_next_24_bar_return"], ascending=False)
        .head(30)
    )


def aggregate_transitions(frame: pd.DataFrame, asset: str) -> pd.DataFrame:
    subset = frame[frame["asset"] == asset]
    return (
        subset.groupby("horizon_bars", dropna=False)
        .agg(
            avg_missed_count=("missed_count", "mean"),
            becomes_target_positive_rate=("becomes_target_positive_within_horizon_rate", "mean"),
            was_target_positive_previous_rate=("was_target_positive_previous_horizon_rate", "mean"),
            avg_return_during_delay=("avg_return_during_delay", "mean"),
        )
        .reset_index()
    )


def aggregate_regime(frame: pd.DataFrame, audit_type: str) -> pd.DataFrame:
    subset = frame[frame["audit_type"] == audit_type].copy()
    if subset.empty:
        return pd.DataFrame()
    group_cols = [column for column in ["overlap_group", "short_regime", "v2_long", "long_regime"] if column in subset.columns]
    return (
        subset.groupby(group_cols, dropna=False)
        .agg(
            avg_count=("count", "mean"),
            avg_next_24_bar_return=("avg_next_24_bar_return", "mean"),
            hit_rate_next_24_gt_0=("hit_rate_next_24_gt_0", "mean"),
            avg_next_72_bar_return=("avg_next_72_bar_return", "mean") if "avg_next_72_bar_return" in subset.columns else ("avg_next_24_bar_return", "mean"),
            avg_expected_net_edge_after_fee=("expected_net_edge_after_fee", "mean") if "expected_net_edge_after_fee" in subset.columns else ("avg_next_24_bar_return", "mean"),
            avg_v2_long_count=("v2_long_count", "mean") if "v2_long_count" in subset.columns else ("count", "mean"),
            avg_v3_target_0p50_count=("v3_target_0p50_count", "mean") if "v3_target_0p50_count" in subset.columns else ("count", "mean"),
            avg_v3_executed_positive_count=("v3_executed_positive_count", "mean") if "v3_executed_positive_count" in subset.columns else ("count", "mean"),
        )
        .reset_index()
        .sort_values("avg_next_24_bar_return", ascending=False)
        .head(30)
    )


def aggregate_v2_conditions(frame: pd.DataFrame) -> pd.DataFrame:
    subset = frame[frame["audit_type"] == "v2_long_v3_zero_v2_condition"]
    return (
        subset.groupby(["asset", "condition", "bucket"], dropna=False)
        .agg(
            avg_count=("count", "mean"),
            avg_next_24_bar_return=("avg_next_24_bar_return", "mean"),
            hit_rate_next_24_gt_0=("hit_rate_next_24_gt_0", "mean"),
            avg_v2_position=("avg_v2_position", "mean"),
            avg_v3_target_position=("avg_v3_target_position", "mean"),
        )
        .reset_index()
        .sort_values(["asset", "avg_next_24_bar_return"], ascending=[True, False])
        .head(40)
    )


def build_conclusions(
    btc_overlap: pd.DataFrame,
    missed: pd.DataFrame,
    transitions: pd.DataFrame,
    strong_bull: pd.DataFrame,
    target_half: pd.DataFrame,
) -> str:
    lines = []
    missed_row = row_for(btc_overlap, "overlap_group", "v2_long_v3_target_zero")
    both_long = row_for(btc_overlap, "overlap_group", "v2_long_v3_target_positive")
    if missed_row is not None:
        lines.append(f"- BTC `v2_long_v3_target_zero` avg_next_24_bar_return is `{missed_row['avg_next_24_bar_return']:.6g}` with hit rate `{missed_row['hit_rate_next_24_gt_0']:.6g}`.")
    if both_long is not None:
        lines.append(f"- BTC `v2_long_v3_target_positive` avg_next_24_bar_return is `{both_long['avg_next_24_bar_return']:.6g}`; v2's missed-by-v3 slice is stronger than the overlap slice.")
    if not missed.empty:
        top = missed.iloc[0]
        lines.append(f"- Strongest missed-v2-long v3 state bucket: `{top['breakdown']}={top['bucket']}` with avg_next_24 `{top['avg_next_24_bar_return']:.6g}`.")
    if not transitions.empty:
        h24 = transitions[transitions["horizon_bars"] == 24]
        if not h24.empty:
            lines.append(f"- Within 24 bars after a missed v2-long bar, v3 becomes target-positive at rate `{float(h24['becomes_target_positive_rate'].iloc[0]):.6g}`; previous-24 target-positive rate is `{float(h24['was_target_positive_previous_rate'].iloc[0]):.6g}`.")
    if not strong_bull.empty:
        worst = strong_bull.sort_values("avg_next_24_bar_return").iloc[0]
        lines.append(f"- Weak strong-bull slice: short_regime `{worst.get('short_regime', 'unknown')}`, v2_long `{worst.get('v2_long', 'unknown')}`, avg_next_24 `{worst['avg_next_24_bar_return']:.6g}`.")
    if not target_half.empty:
        best = target_half.sort_values("avg_expected_net_edge_after_fee", ascending=False).iloc[0]
        lines.append(f"- Best BTC target=0.50 cost-adjusted slice: `{best.get('long_regime', '')} + {best.get('short_regime', '')}`, v2_long `{best.get('v2_long', '')}`.")
    lines.append("- Evidence points to a signal-definition/timing gap, not just insufficient exposure. The next step should test adding v2 signal/regime fields as diagnostic input features before changing v3 rules.")
    return "\n".join(lines)


def row_for(frame: pd.DataFrame, column: str, value: str) -> pd.Series | None:
    subset = frame[frame[column] == value]
    return None if subset.empty else subset.iloc[0]


def forward_stats(group: pd.DataFrame, horizons: tuple[int, ...] = (1, 6, 24, 72)) -> dict[str, Any]:
    result: dict[str, Any] = {"count": int(len(group))}
    for horizon in horizons:
        output_name = "avg_next_bar_return" if horizon == 1 else f"avg_next_{horizon}_bar_return"
        result[output_name] = mean_numeric(group, f"next_{horizon}_bar_return")
    next24 = pd.to_numeric(group.get("next_24_bar_return", pd.Series(dtype=float)), errors="coerce").dropna()
    if 24 in horizons:
        result["median_next_24_bar_return"] = float(next24.median()) if len(next24) else np.nan
        result["hit_rate_next_24_gt_0"] = float((next24 > 0.0).mean()) if len(next24) else np.nan
        result["volatility_next_24_return"] = float(next24.std(ddof=0)) if len(next24) else np.nan
    return result


def numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype(float)


def string_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series("unavailable", index=frame.index)
    return frame[column].astype(str).fillna("unavailable")


def bool_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame[column].fillna(False).astype(bool)


def keys_to_row(groups: list[str], keys: Any) -> dict[str, Any]:
    if not isinstance(keys, tuple):
        keys = (keys,)
    return {group: key for group, key in zip(groups, keys)}


def mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else np.nan


def mean_numeric(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return np.nan
    return mean(frame[column])


def sum_numeric(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).sum())


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

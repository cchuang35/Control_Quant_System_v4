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
REPORT_PATH = Path("reports") / "v3_target_alpha_quality_audit.md"
SUMMARY_CSV = Path("reports") / "v3_target_alpha_quality_audit_summary.csv"
COMBINED_REGIMES_CSV = Path("reports") / "v3_target_alpha_quality_audit_combined_regimes.csv"
BLOCKED_TARGETS_CSV = Path("reports") / "v3_target_alpha_quality_audit_blocked_targets.csv"


def main() -> None:
    config = load_final_candidate_config()
    frames = []
    for asset, path in [("BTC", BTC_WATERFALL_CSV), ("ETH", ETH_WATERFALL_CSV)]:
        if not path.exists():
            print(f"missing waterfall CSV for {asset}: {path}")
            continue
        print(f"loading {path}")
        frame = pd.read_csv(path)
        frame["asset"] = asset
        frames.append(add_forward_returns(frame))
    if not frames:
        raise FileNotFoundError("no v3 decision-waterfall CSVs were available")

    all_data = pd.concat(frames, ignore_index=True)
    summary = build_summary_tables(all_data)
    combined = build_combined_regime_table(all_data)
    blocked = build_blocked_target_table(all_data)
    v2_compare = build_v2_comparison_table(all_data, config)
    summary = pd.concat([summary, v2_compare], ignore_index=True)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)
    combined.to_csv(COMBINED_REGIMES_CSV, index=False)
    blocked.to_csv(BLOCKED_TARGETS_CSV, index=False)
    write_report(summary, combined, blocked)
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {COMBINED_REGIMES_CSV}")
    print(f"Wrote {BLOCKED_TARGETS_CSV}")


def add_forward_returns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    keys = ["asset", "dataset", "fee_rate"]
    result["asset_return"] = pd.to_numeric(result["asset_return"], errors="coerce").fillna(0.0)
    for horizon in (1, 6, 24, 72):
        result[f"next_{horizon}_bar_return"] = result.groupby(keys, group_keys=False)["asset_return"].transform(
            lambda series, h=horizon: series.shift(-1).rolling(h, min_periods=1).sum().shift(-(h - 1))
        )
    result["target_position"] = pd.to_numeric(result["target_position"], errors="coerce").fillna(0.0)
    result["executed_position"] = pd.to_numeric(result["executed_position"], errors="coerce").fillna(0.0)
    result["target_bucket"] = result["target_position"].round(2)
    result["target_weighted_next_bar_return"] = result["target_position"] * result["next_1_bar_return"]
    result["target_weighted_next_24_bar_return"] = result["target_position"] * result["next_24_bar_return"]
    return result


def build_summary_tables(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    rows.append(group_forward_stats(frame, ["asset", "dataset", "fee_rate", "target_bucket"], "target_position"))
    rows.append(group_forward_stats(frame, ["asset", "dataset", "fee_rate", "long_regime"], "long_regime"))
    rows.append(group_forward_stats(frame, ["asset", "dataset", "fee_rate", "short_regime"], "short_regime"))
    rows.append(cost_adjusted_target_stats(frame))
    rows.append(cost_adjusted_combined_regime_stats(frame))
    return pd.concat(rows, ignore_index=True)


def group_forward_stats(frame: pd.DataFrame, groups: list[str], audit_type: str) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(groups, dropna=False):
        row = keys_to_row(groups, keys)
        next24 = pd.to_numeric(group["next_24_bar_return"], errors="coerce")
        row.update(
            {
                "audit_type": audit_type,
                "count": int(len(group)),
                "avg_next_bar_return": mean(group["next_1_bar_return"]),
                "avg_next_6_bar_return": mean(group["next_6_bar_return"]),
                "avg_next_24_bar_return": mean(next24),
                "avg_next_72_bar_return": mean(group["next_72_bar_return"]),
                "median_next_24_bar_return": float(next24.median()) if len(next24.dropna()) else np.nan,
                "hit_rate_next_24_gt_0": hit_rate(next24),
                "volatility_next_24_return": std(next24),
                "sharpe_like_next_24": sharpe_like(next24),
                "avg_target_position": mean(group["target_position"]),
                "avg_executed_position": mean(group["executed_position"]),
                "target_to_executed_gap": mean(group["target_position"] - group["executed_position"]),
                "avg_target_weighted_next_bar_return": mean(group["target_weighted_next_bar_return"]),
                "avg_target_weighted_next_24_bar_return": mean(group["target_weighted_next_24_bar_return"]),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def cost_adjusted_target_stats(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = ["asset", "dataset", "fee_rate", "target_bucket"]
    for keys, group in frame.groupby(groups, dropna=False):
        row = keys_to_row(groups, keys)
        avg_target = mean(group["target_position"])
        gross_next24 = mean(group["next_24_bar_return"])
        weighted_gross = mean(group["target_weighted_next_24_bar_return"])
        round_trip_fee = 2.0 * avg_target * float(row["fee_rate"])
        row.update(
            {
                "audit_type": "cost_adjusted_target",
                "count": int(len(group)),
                "avg_next_24_bar_return": gross_next24,
                "avg_target_position": avg_target,
                "target_weighted_expected_gross_next_24": weighted_gross,
                "approx_round_trip_fee_cost": round_trip_fee,
                "expected_net_edge_after_fee": weighted_gross - round_trip_fee,
                "expected_net_edge_positive": bool(weighted_gross - round_trip_fee > 0.0),
                "fee_assumption": "enter_and_exit_full_target_once_over_24_bars",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def cost_adjusted_combined_regime_stats(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = ["asset", "dataset", "fee_rate", "target_bucket", "long_regime", "short_regime"]
    for keys, group in frame.groupby(groups, dropna=False):
        row = keys_to_row(groups, keys)
        avg_target = mean(group["target_position"])
        gross_next24 = mean(group["next_24_bar_return"])
        weighted_gross = mean(group["target_weighted_next_24_bar_return"])
        round_trip_fee = 2.0 * avg_target * float(row["fee_rate"])
        row.update(
            {
                "audit_type": "cost_adjusted_combined_regime",
                "combo": f"{row['long_regime']} + {row['short_regime']}",
                "count": int(len(group)),
                "avg_next_24_bar_return": gross_next24,
                "avg_target_position": avg_target,
                "target_weighted_expected_gross_next_24": weighted_gross,
                "approx_round_trip_fee_cost": round_trip_fee,
                "expected_net_edge_after_fee": weighted_gross - round_trip_fee,
                "expected_net_edge_positive": bool(weighted_gross - round_trip_fee > 0.0),
                "fee_assumption": "enter_and_exit_full_target_once_over_24_bars",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_combined_regime_table(frame: pd.DataFrame) -> pd.DataFrame:
    groups = ["asset", "dataset", "fee_rate", "long_regime", "short_regime"]
    result = group_forward_stats(frame, groups, "combined_regime")
    result["combo"] = result["long_regime"].astype(str) + " + " + result["short_regime"].astype(str)
    return result.sort_values(["asset", "dataset", "fee_rate", "avg_next_24_bar_return"], ascending=[True, True, True, False])


def build_blocked_target_table(frame: pd.DataFrame) -> pd.DataFrame:
    blocked = frame[(frame["target_position"] > 0.0) & (frame["executed_position"] == 0.0)].copy()
    executed = frame[(frame["target_position"] > 0.0) & (frame["executed_position"] > 0.0)].copy()
    rows = []
    for label, subset in [("blocked_nonzero_target", blocked), ("executed_nonzero_target", executed)]:
        rows.extend(blocked_summary_rows(subset, label))
        if label == "blocked_nonzero_target" and not subset.empty:
            rows.extend(distribution_rows(subset, label, "binding_block_reason"))
            rows.extend(distribution_rows(subset, label, "long_regime"))
            rows.extend(distribution_rows(subset, label, "short_regime"))
    return pd.DataFrame(rows)


def blocked_summary_rows(frame: pd.DataFrame, audit_type: str) -> list[dict[str, Any]]:
    rows = []
    for keys, group in frame.groupby(["asset", "dataset", "fee_rate"], dropna=False):
        row = keys_to_row(["asset", "dataset", "fee_rate"], keys)
        next24 = pd.to_numeric(group["next_24_bar_return"], errors="coerce")
        row.update(
            {
                "audit_type": audit_type,
                "diagnostic": "summary",
                "bucket": "all",
                "count": int(len(group)),
                "avg_next_24_bar_return": mean(next24),
                "hit_rate_next_24_gt_0": hit_rate(next24),
                "avg_target_position": mean(group["target_position"]),
                "avg_executed_position": mean(group["executed_position"]),
            }
        )
        rows.append(row)
    return rows


def distribution_rows(frame: pd.DataFrame, audit_type: str, column: str) -> list[dict[str, Any]]:
    rows = []
    for keys, group in frame.groupby(["asset", "dataset", "fee_rate"], dropna=False):
        base = keys_to_row(["asset", "dataset", "fee_rate"], keys)
        counts = group[column].astype(str).value_counts(dropna=False)
        for bucket, count in counts.items():
            row = dict(base)
            row.update(
                {
                    "audit_type": audit_type,
                    "diagnostic": f"{column}_distribution",
                    "bucket": bucket,
                    "count": int(count),
                    "percentage": float(count / max(len(group), 1)),
                }
            )
            rows.append(row)
    return rows


def build_v2_comparison_table(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for asset, dataset_key in [("BTC", "btc_datasets"), ("ETH", "eth_datasets")]:
        paths = [Path(path) for path in config["validation"].get(dataset_key, []) if Path(path).exists()]
        v2_func = backtest_v2_btc_final_candidate_a if asset == "BTC" else backtest_v2_final_candidate_a
        for path in paths:
            dataset = path.stem
            data = load_ohlcv_csv(path)
            for fee_rate in sorted(frame.loc[(frame["asset"] == asset) & (frame["dataset"] == dataset), "fee_rate"].unique()):
                subset = frame[(frame["asset"] == asset) & (frame["dataset"] == dataset) & (frame["fee_rate"] == fee_rate)].copy().reset_index(drop=True)
                if subset.empty:
                    continue
                print(f"v2 compare asset={asset} dataset={dataset} fee={fee_rate:g}")
                v1_result = run_backtest_fast(
                    data,
                    fee_rate=float(fee_rate),
                    periods_per_year=PERIODS_PER_YEAR,
                    progress_every=10000 if len(data) > 15000 else None,
                )
                v2_frame = v2_func(input_frame_from_v1(v1_result), fee_rate=float(fee_rate), v1_entry_threshold=V1_ENTRY_THRESHOLD, cooldown_bars=120)
                v2_position = pd.to_numeric(v2_frame.get("final_position", pd.Series(0.0, index=v2_frame.index)), errors="coerce").fillna(0.0).reset_index(drop=True)
                v2_return = pd.to_numeric(v2_frame.get("strategy_return_net", pd.Series(0.0, index=v2_frame.index)), errors="coerce").fillna(0.0).reset_index(drop=True)
                length = min(len(subset), len(v2_position))
                aligned = subset.iloc[:length].copy()
                aligned["v2_position"] = v2_position.iloc[:length].to_numpy()
                aligned["v2_strategy_return_net"] = v2_return.iloc[:length].to_numpy()
                rows.extend(v2_group_rows(aligned))
    return pd.DataFrame(rows)


def v2_group_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    masks = {
        "v2_long_v3_target_zero": (frame["v2_position"] > 0.0) & (frame["target_position"] == 0.0),
        "v2_long_v3_target_positive": (frame["v2_position"] > 0.0) & (frame["target_position"] > 0.0),
        "v2_flat_v3_target_positive": (frame["v2_position"] == 0.0) & (frame["target_position"] > 0.0),
    }
    rows = []
    base = {
        "asset": frame["asset"].iloc[0],
        "dataset": frame["dataset"].iloc[0],
        "fee_rate": frame["fee_rate"].iloc[0],
    }
    for label, mask in masks.items():
        subset = frame[mask]
        next24 = pd.to_numeric(subset["next_24_bar_return"], errors="coerce")
        row = dict(base)
        row.update(
            {
                "audit_type": "v2_comparison",
                "comparison_group": label,
                "count": int(len(subset)),
                "avg_next_24_bar_return": mean(next24),
                "hit_rate_next_24_gt_0": hit_rate(next24),
                "realized_v2_strategy_contribution": float(pd.to_numeric(subset["v2_strategy_return_net"], errors="coerce").fillna(0.0).sum()),
                "avg_target_position": mean(subset["target_position"]) if len(subset) else np.nan,
                "avg_executed_position": mean(subset["executed_position"]) if len(subset) else np.nan,
            }
        )
        rows.append(row)
    return rows


def write_report(summary: pd.DataFrame, combined: pd.DataFrame, blocked: pd.DataFrame) -> None:
    btc_target = aggregate(summary, "target_position", "BTC", "target_bucket")
    eth_target = aggregate(summary, "target_position", "ETH", "target_bucket")
    btc_long = aggregate(summary, "long_regime", "BTC", "long_regime")
    btc_short = aggregate(summary, "short_regime", "BTC", "short_regime")
    btc_cost = aggregate(summary, "cost_adjusted_target", "BTC", "target_bucket")
    btc_cost_combos = aggregate_cost_combos(summary, "BTC")
    btc_v2 = aggregate_v2(summary, "BTC")
    eth_v2 = aggregate_v2(summary, "ETH")
    best_combos = aggregate_combined(combined, best=True)
    worst_combos = aggregate_combined(combined, best=False)
    blocked_compare = aggregate_blocked(blocked)
    conclusions = build_conclusions(btc_target, btc_long, btc_short, blocked_compare)

    lines = [
        "# v3 Target Alpha Quality Audit",
        "",
        "This is a diagnostics-only audit of v3.final_candidate target quality. No trading behavior, Risk Supervisor behavior, v2 behavior, feature windows, estimator thresholds, or controller mappings were changed. Forward returns are post-run diagnostics only.",
        "",
        "## 1. Main Conclusions",
        "",
        conclusions,
        "",
        "## 2. BTC Target-Position Forward Returns",
        "",
        _frame_to_markdown(btc_target),
        "",
        "## 3. ETH Target-Position Forward Returns",
        "",
        _frame_to_markdown(eth_target),
        "",
        "## 4. BTC Long-Regime Audit",
        "",
        _frame_to_markdown(btc_long),
        "",
        "## 5. BTC Short-Regime Audit",
        "",
        _frame_to_markdown(btc_short),
        "",
        "## 6. Combined Regime Highlights",
        "",
        "Best combinations by average next-24-bar return:",
        "",
        _frame_to_markdown(best_combos),
        "",
        "Worst combinations by average next-24-bar return:",
        "",
        _frame_to_markdown(worst_combos),
        "",
        "## 7. Blocked Target Audit",
        "",
        _frame_to_markdown(blocked_compare),
        "",
        "## 8. Cost-Adjusted Target Audit",
        "",
        "Fee assumption: enter and exit the full target exposure once over the next 24 bars, so approximate fee cost is `2 * target_position * fee_rate`.",
        "",
        _frame_to_markdown(btc_cost),
        "",
        "Cost-adjusted BTC combinations with positive net edge:",
        "",
        _frame_to_markdown(btc_cost_combos),
        "",
        "## 9. v2 Comparison Audit",
        "",
        "BTC:",
        "",
        _frame_to_markdown(btc_v2),
        "",
        "ETH:",
        "",
        _frame_to_markdown(eth_v2),
        "",
        "## 10. Files",
        "",
        f"- Summary CSV: `{SUMMARY_CSV}`",
        f"- Combined regimes CSV: `{COMBINED_REGIMES_CSV}`",
        f"- Blocked targets CSV: `{BLOCKED_TARGETS_CSV}`",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def aggregate(summary: pd.DataFrame, audit_type: str, asset: str, bucket_col: str) -> pd.DataFrame:
    subset = summary[(summary["audit_type"] == audit_type) & (summary["asset"] == asset)].copy()
    if subset.empty:
        return pd.DataFrame()
    if audit_type.startswith("cost_adjusted"):
        columns = [
            "count",
            "avg_next_24_bar_return",
            "avg_target_position",
            "target_weighted_expected_gross_next_24",
            "approx_round_trip_fee_cost",
            "expected_net_edge_after_fee",
            "expected_net_edge_positive",
        ]
    else:
        columns = [
            "count",
            "avg_next_bar_return",
            "avg_next_6_bar_return",
            "avg_next_24_bar_return",
            "avg_next_72_bar_return",
            "median_next_24_bar_return",
            "hit_rate_next_24_gt_0",
            "volatility_next_24_return",
            "sharpe_like_next_24",
            "avg_target_position",
            "avg_executed_position",
            "target_to_executed_gap",
            "avg_target_weighted_next_bar_return",
            "avg_target_weighted_next_24_bar_return",
        ]
    agg_spec = {column: "mean" for column in columns if column in subset.columns}
    result = subset.groupby(bucket_col, dropna=False).agg(agg_spec).reset_index()
    return result.sort_values(bucket_col)


def aggregate_v2(summary: pd.DataFrame, asset: str) -> pd.DataFrame:
    subset = summary[(summary["audit_type"] == "v2_comparison") & (summary["asset"] == asset)].copy()
    if subset.empty:
        return pd.DataFrame()
    return (
        subset.groupby("comparison_group", dropna=False)
        .agg(
            avg_count=("count", "mean"),
            avg_next_24_bar_return=("avg_next_24_bar_return", "mean"),
            hit_rate_next_24_gt_0=("hit_rate_next_24_gt_0", "mean"),
            avg_realized_v2_strategy_contribution=("realized_v2_strategy_contribution", "mean"),
            avg_target_position=("avg_target_position", "mean"),
            avg_executed_position=("avg_executed_position", "mean"),
        )
        .reset_index()
    )


def aggregate_cost_combos(summary: pd.DataFrame, asset: str) -> pd.DataFrame:
    subset = summary[(summary["audit_type"] == "cost_adjusted_combined_regime") & (summary["asset"] == asset)].copy()
    if subset.empty:
        return pd.DataFrame()
    grouped = (
        subset.groupby(["target_bucket", "combo"], dropna=False)
        .agg(
            avg_count=("count", "mean"),
            avg_next_24_bar_return=("avg_next_24_bar_return", "mean"),
            avg_target_position=("avg_target_position", "mean"),
            target_weighted_expected_gross_next_24=("target_weighted_expected_gross_next_24", "mean"),
            approx_round_trip_fee_cost=("approx_round_trip_fee_cost", "mean"),
            expected_net_edge_after_fee=("expected_net_edge_after_fee", "mean"),
            positive_edge_rate=("expected_net_edge_positive", "mean"),
        )
        .reset_index()
    )
    positive = grouped[grouped["expected_net_edge_after_fee"] > 0.0]
    if positive.empty:
        return grouped.sort_values("expected_net_edge_after_fee", ascending=False).head(12)
    return positive.sort_values("expected_net_edge_after_fee", ascending=False).head(12)


def aggregate_combined(combined: pd.DataFrame, *, best: bool) -> pd.DataFrame:
    if combined.empty:
        return pd.DataFrame()
    grouped = (
        combined.groupby(["asset", "combo"], dropna=False)
        .agg(
            avg_count=("count", "mean"),
            avg_next_24_bar_return=("avg_next_24_bar_return", "mean"),
            hit_rate_next_24_gt_0=("hit_rate_next_24_gt_0", "mean"),
            avg_target_position=("avg_target_position", "mean"),
            avg_executed_position=("avg_executed_position", "mean"),
        )
        .reset_index()
    )
    return grouped.sort_values(["asset", "avg_next_24_bar_return"], ascending=[True, not best]).groupby("asset").head(8)


def aggregate_blocked(blocked: pd.DataFrame) -> pd.DataFrame:
    subset = blocked[blocked["diagnostic"] == "summary"].copy()
    if subset.empty:
        return pd.DataFrame()
    return (
        subset.groupby(["asset", "audit_type"], dropna=False)
        .agg(
            avg_count=("count", "mean"),
            avg_next_24_bar_return=("avg_next_24_bar_return", "mean"),
            hit_rate_next_24_gt_0=("hit_rate_next_24_gt_0", "mean"),
            avg_target_position=("avg_target_position", "mean"),
            avg_executed_position=("avg_executed_position", "mean"),
        )
        .reset_index()
    )


def build_conclusions(
    btc_target: pd.DataFrame,
    btc_long: pd.DataFrame,
    btc_short: pd.DataFrame,
    blocked_compare: pd.DataFrame,
) -> str:
    lines = []
    positive_targets = btc_target[pd.to_numeric(btc_target.get("target_bucket", 0), errors="coerce") > 0.0] if not btc_target.empty else pd.DataFrame()
    if not positive_targets.empty:
        best_target = positive_targets.sort_values("avg_next_24_bar_return", ascending=False).iloc[0]
        positive_count = int((positive_targets["avg_next_24_bar_return"] > 0.0).sum())
        total_count = len(positive_targets)
        if positive_count == total_count:
            target_quality = "positive across all nonzero buckets"
        elif positive_count > 0:
            target_quality = f"mixed: {positive_count}/{total_count} nonzero buckets are positive"
        else:
            target_quality = "weak or negative across nonzero buckets"
        lines.append(f"- BTC target_position quality is {target_quality} on average next-24-bar return.")
        lines.append(f"- Best BTC target bucket by next-24-bar return: `{best_target['target_bucket']}` with avg_next_24_bar_return `{best_target['avg_next_24_bar_return']:.6g}`.")
    if not btc_long.empty:
        best_long = btc_long.sort_values("avg_next_24_bar_return", ascending=False).iloc[0]
        lines.append(f"- Best BTC long_regime by next-24-bar return: `{best_long['long_regime']}`.")
    if not btc_short.empty:
        best_short = btc_short.sort_values("avg_next_24_bar_return", ascending=False).iloc[0]
        worst_short = btc_short.sort_values("avg_next_24_bar_return", ascending=True).iloc[0]
        lines.append(f"- Best BTC short_regime: `{best_short['short_regime']}`; weakest BTC short_regime: `{worst_short['short_regime']}`.")
    btc_blocked = blocked_compare[(blocked_compare["asset"] == "BTC") & (blocked_compare["audit_type"] == "blocked_nonzero_target")]
    btc_executed = blocked_compare[(blocked_compare["asset"] == "BTC") & (blocked_compare["audit_type"] == "executed_nonzero_target")]
    if not btc_blocked.empty and not btc_executed.empty:
        blocked_ret = float(btc_blocked["avg_next_24_bar_return"].iloc[0])
        executed_ret = float(btc_executed["avg_next_24_bar_return"].iloc[0])
        lines.append(f"- BTC blocked nonzero targets avg_next_24_bar_return `{blocked_ret:.6g}` versus executed nonzero targets `{executed_ret:.6g}`.")
    lines.append("- Next step should compare v2 signal timing versus v3 target signal quality before relaxing risk rules again.")
    return "\n".join(lines)


def keys_to_row(groups: list[str], keys: Any) -> dict[str, Any]:
    if not isinstance(keys, tuple):
        keys = (keys,)
    return {group: key for group, key in zip(groups, keys)}


def mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    return float(values.mean()) if len(values.dropna()) else np.nan


def std(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.std(ddof=0)) if len(values) else np.nan


def hit_rate(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float((values > 0.0).mean()) if len(values) else np.nan


def sharpe_like(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    sigma = float(values.std(ddof=0)) if len(values) else 0.0
    return float(values.mean() / sigma) if sigma > 0.0 else np.nan


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

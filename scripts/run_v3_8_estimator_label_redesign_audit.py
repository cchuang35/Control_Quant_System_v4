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
    build_backtest_config,
    input_frame_from_v1,
    load_final_candidate_config,
)
from src.v3.backtest_v3 import run_v3_backtest
from v2_small_cap import backtest_v2_btc_final_candidate_a, backtest_v2_final_candidate_a


REPORT_PATH = Path("reports") / "v3_8_estimator_label_redesign_audit.md"
SUMMARY_CSV = Path("reports") / "v3_8_estimator_label_redesign_audit_summary.csv"
FORWARD_CSV = Path("reports") / "v3_8_estimator_label_redesign_regime_forward_returns.csv"
TRANSITION_CSV = Path("reports") / "v3_8_estimator_label_redesign_transition_audit.csv"
COMPARISON_CSV = Path("reports") / "v3_8_estimator_label_redesign_old_new_comparison.csv"


LABEL_ORDER = [
    "healthy_bull",
    "overextended_bull",
    "late_bull",
    "weak_bull",
    "neutral_range",
    "early_recovery_candidate",
    "oversold_rebound_candidate",
    "true_bear",
    "true_strong_bear",
    "unknown",
]


def main() -> None:
    config = load_final_candidate_config()
    fee_rates = tuple(float(value) for value in config["execution"]["fee_rates_to_validate"])
    frames: list[pd.DataFrame] = []

    for asset, dataset_key in [("BTC", "btc_datasets"), ("ETH", "eth_datasets")]:
        paths = [Path(path) for path in config["validation"].get(dataset_key, []) if Path(path).exists()]
        for path in paths:
            dataset = path.stem
            print(f"asset={asset} dataset={dataset}")
            data = load_ohlcv_csv(path)
            for fee_rate in fee_rates:
                print(f"  fee={fee_rate:g}")
                v3 = run_v3_backtest(data, config=build_backtest_config(config, fee_rate=fee_rate)).reset_index(drop=True)
                v2 = run_v2_reference(asset, data, fee_rate)
                frames.append(prepare_audit_frame(v3, v2, asset, dataset, fee_rate))

    audit_frame = pd.concat(frames, ignore_index=True)
    summary = build_summary(audit_frame)
    forward = build_forward_return_audit(audit_frame)
    transitions = build_transition_audit(audit_frame)
    comparison = build_old_new_comparison(audit_frame)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)
    forward.to_csv(FORWARD_CSV, index=False)
    transitions.to_csv(TRANSITION_CSV, index=False)
    comparison.to_csv(COMPARISON_CSV, index=False)
    write_report(summary, forward, transitions, comparison)
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {FORWARD_CSV}")
    print(f"Wrote {TRANSITION_CSV}")
    print(f"Wrote {COMPARISON_CSV}")


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


def prepare_audit_frame(v3: pd.DataFrame, v2: pd.DataFrame, asset: str, dataset: str, fee_rate: float) -> pd.DataFrame:
    frame = v3.copy().reset_index(drop=True)
    frame.insert(0, "asset", asset)
    frame.insert(1, "dataset", dataset)
    frame.insert(2, "fee_rate", fee_rate)
    frame["bar_index"] = np.arange(len(frame))
    frame["v2_position"] = pd.to_numeric(v2.get("final_position", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0).reindex(frame.index, fill_value=0.0)
    frame["trend_health_label"] = frame.apply(assign_trend_health_label, axis=1)
    returns = pd.to_numeric(frame["asset_return"], errors="coerce").fillna(0.0)
    for horizon in (1, 6, 24, 72):
        frame[f"next_{horizon}_bar_return"] = forward_compound_return(returns, horizon)
    frame["target_positive"] = pd.to_numeric(frame["target_position"], errors="coerce").fillna(0.0) > 0.0
    return frame


def assign_trend_health_label(row: pd.Series) -> str:
    long_regime = str(row.get("long_regime", ""))
    short_regime = str(row.get("short_regime", ""))
    volatility_state = str(row.get("volatility_state", ""))
    drawdown_state = str(row.get("drawdown_state", ""))
    confidence = float(row.get("confidence_score", np.nan))
    target = float(row.get("target_position", 0.0))
    v2_position = float(row.get("v2_position", 0.0))

    if long_regime == "strong_bear" and short_regime == "breakdown" and v2_position <= 0.0:
        return "true_strong_bear"
    if long_regime == "bear" and short_regime == "breakdown" and v2_position <= 0.0:
        return "true_bear"
    if (
        long_regime in {"bear", "strong_bear"}
        and short_regime in {"noise", "recovery"}
        and v2_position > 0.0
        and drawdown_state in {"caution", "danger"}
        and volatility_state != "extreme"
    ):
        return "oversold_rebound_candidate"
    if long_regime in {"neutral", "bear"} and short_regime == "recovery" and v2_position > 0.0 and volatility_state != "extreme":
        return "early_recovery_candidate"
    if long_regime == "strong_bull" and short_regime in {"overheat", "pullback"}:
        return "overextended_bull"
    if long_regime == "strong_bull" and short_regime == "noise":
        return "late_bull"
    if (
        long_regime == "bull"
        and short_regime == "noise"
        and volatility_state != "extreme"
        and drawdown_state in {"normal", "caution"}
        and abs(target - 0.50) <= 1e-12
    ):
        return "healthy_bull"
    if long_regime == "bull" and (short_regime != "noise" or confidence < 0.55):
        return "weak_bull"
    if long_regime == "neutral":
        return "neutral_range"
    if long_regime == "bear":
        return "true_bear"
    if long_regime == "strong_bear":
        return "true_strong_bear"
    return "unknown"


def forward_compound_return(returns: pd.Series, horizon: int) -> pd.Series:
    values = (1.0 + returns).to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    for idx in range(len(values)):
        end = idx + horizon
        if end < len(values):
            out[idx] = float(np.prod(values[idx + 1 : end + 1]) - 1.0)
    return pd.Series(out, index=returns.index)


def build_summary(frame: pd.DataFrame) -> pd.DataFrame:
    forward = build_forward_return_audit(frame)
    btc = forward[forward["asset"] == "BTC"].copy()
    eth = forward[forward["asset"] == "ETH"].copy()
    rows = []
    for label in LABEL_ORDER:
        b = btc[btc["trend_health_label"] == label]
        e = eth[eth["trend_health_label"] == label]
        rows.append(
            {
                "trend_health_label": label,
                "btc_rows": int(b["count"].sum()) if not b.empty else 0,
                "btc_avg_next_24": weighted_mean(b, "avg_next_24", "count"),
                "btc_hit_rate_next_24": weighted_mean(b, "hit_rate_next_24", "count"),
                "btc_positive_edge_rate": positive_edge_rate(b),
                "eth_rows": int(e["count"].sum()) if not e.empty else 0,
                "eth_avg_next_24": weighted_mean(e, "avg_next_24", "count"),
                "eth_hit_rate_next_24": weighted_mean(e, "hit_rate_next_24", "count"),
                "eth_positive_edge_rate": positive_edge_rate(e),
                "promising": is_promising_label(b, e),
            }
        )
    return pd.DataFrame(rows)


def build_forward_return_audit(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total_by_key = frame.groupby(["asset", "dataset", "fee_rate"]).size().rename("total")
    grouped = frame.groupby(["asset", "dataset", "fee_rate", "trend_health_label"], dropna=False)
    for (asset, dataset, fee_rate, label), group in grouped:
        next24 = pd.to_numeric(group["next_24_bar_return"], errors="coerce")
        count = int(len(group))
        cost = 2.0 * float(fee_rate)
        gross24 = mean(next24)
        rows.append(
            {
                "asset": asset,
                "dataset": dataset,
                "fee_rate": fee_rate,
                "trend_health_label": label,
                "count": count,
                "percentage": count / int(total_by_key.loc[(asset, dataset, fee_rate)]),
                "avg_next_1": mean(group["next_1_bar_return"]),
                "avg_next_6": mean(group["next_6_bar_return"]),
                "avg_next_24": gross24,
                "avg_next_72": mean(group["next_72_bar_return"]),
                "median_next_24": median(next24),
                "hit_rate_next_24": hit_rate(next24),
                "vol_next_24": std(next24),
                "sharpe_like_next_24": safe_ratio(gross24, std(next24)),
                "avg_target_position": mean(group["target_position"]),
                "avg_executed_position": mean(group["executed_position"]),
                "risk_action_distribution": distribution_string(group["risk_action"].astype(str)),
                "avg_v2_position": mean(group["v2_position"]),
                "expected_gross_next_24": gross24,
                "approx_entry_exit_cost": cost,
                "expected_net_edge": gross24 - cost,
                "positive_edge": bool(gross24 - cost > 0.0) if not np.isnan(gross24) else False,
            }
        )
    return pd.DataFrame(rows)


def build_old_new_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (asset, dataset, fee_rate, old_regime), group in frame.groupby(["asset", "dataset", "fee_rate", "long_regime"], dropna=False):
        counts = group["trend_health_label"].value_counts(dropna=False)
        total = max(int(counts.sum()), 1)
        for label, count in counts.items():
            rows.append(
                {
                    "asset": asset,
                    "dataset": dataset,
                    "fee_rate": fee_rate,
                    "old_long_regime": old_regime,
                    "trend_health_label": label,
                    "count": int(count),
                    "percentage": float(count / total),
                    "avg_next_24": mean(group.loc[group["trend_health_label"] == label, "next_24_bar_return"]),
                    "hit_rate_next_24": hit_rate(group.loc[group["trend_health_label"] == label, "next_24_bar_return"]),
                }
            )
    return pd.DataFrame(rows)


def build_transition_audit(frame: pd.DataFrame) -> pd.DataFrame:
    labels = {"healthy_bull", "early_recovery_candidate", "oversold_rebound_candidate"}
    rows: list[dict[str, Any]] = []
    for (asset, dataset, fee_rate, label), group in frame[frame["trend_health_label"].isin(labels)].groupby(["asset", "dataset", "fee_rate", "trend_health_label"]):
        source = frame[(frame["asset"] == asset) & (frame["dataset"] == dataset) & (frame["fee_rate"] == fee_rate)].reset_index(drop=True)
        indexes = group["bar_index"].astype(int).to_list()
        row: dict[str, Any] = {"asset": asset, "dataset": dataset, "fee_rate": fee_rate, "trend_health_label": label, "count": int(len(group))}
        for horizon in (6, 12, 24, 48, 72):
            row[f"target_positive_within_{horizon}_rate"] = target_positive_within_rate(source, indexes, horizon)
            row[f"avg_return_until_{horizon}"] = mean_return_until(source, indexes, horizon)
            row[f"v2_long_during_{horizon}_rate"] = v2_long_during_rate(source, indexes, horizon)
        rows.append(row)
    return pd.DataFrame(rows)


def target_positive_within_rate(frame: pd.DataFrame, indexes: list[int], horizon: int) -> float:
    target = frame["target_positive"].to_numpy(dtype=bool)
    values = []
    for idx in indexes:
        end = min(len(target), idx + horizon + 1)
        values.append(bool(target[idx + 1 : end].any()))
    return float(np.mean(values)) if values else np.nan


def mean_return_until(frame: pd.DataFrame, indexes: list[int], horizon: int) -> float:
    returns = pd.to_numeric(frame["asset_return"], errors="coerce").fillna(0.0).to_numpy()
    values = []
    for idx in indexes:
        end = min(len(returns), idx + horizon + 1)
        if end > idx + 1:
            values.append(float(np.prod(1.0 + returns[idx + 1 : end]) - 1.0))
    return float(np.mean(values)) if values else np.nan


def v2_long_during_rate(frame: pd.DataFrame, indexes: list[int], horizon: int) -> float:
    v2 = pd.to_numeric(frame["v2_position"], errors="coerce").fillna(0.0).to_numpy()
    values = []
    for idx in indexes:
        end = min(len(v2), idx + horizon + 1)
        values.append(bool((v2[idx : end] > 0.0).any()))
    return float(np.mean(values)) if values else np.nan


def write_report(summary: pd.DataFrame, forward: pd.DataFrame, transitions: pd.DataFrame, comparison: pd.DataFrame) -> None:
    btc_forward = aggregate_forward(forward[forward["asset"] == "BTC"])
    eth_forward = aggregate_forward(forward[forward["asset"] == "ETH"])
    old_new = aggregate_old_new(comparison)
    transition_summary = aggregate_transitions(transitions)
    strongest = strongest_labels(summary)
    weakest = weakest_labels(summary)
    lines = [
        "# v3.8 Estimator Label Redesign Audit",
        "",
        "This is diagnostics only. v2 and v3.final_candidate trading behavior are unchanged; forward returns are post-run analysis only.",
        "",
        "## 1. Executive Summary",
        "",
        executive_summary(summary),
        "",
        "## 2. Why Estimator Label Redesign Is Needed",
        "",
        "Current `strong_bull` often behaves like a late or overextended state rather than a clean high-conviction trend. Prior trading variants did not improve enough, so v3.8 separates diagnostic labels before changing any strategy logic.",
        "",
        "## 3. New Diagnostic Label Definitions",
        "",
        label_definitions(),
        "",
        "## 4. BTC Forward-Return Audit",
        "",
        frame_to_markdown(btc_forward),
        "",
        "## 5. ETH Forward-Return Audit",
        "",
        frame_to_markdown(eth_forward),
        "",
        "## 6. Old-Label vs New-Label Comparison",
        "",
        frame_to_markdown(old_new),
        "",
        "## 7. Transition Timing Audit",
        "",
        frame_to_markdown(transition_summary),
        "",
        "## 8. Cost-Adjusted Edge Audit",
        "",
        frame_to_markdown(cost_edge_summary(forward)),
        "",
        "## 9. Best Diagnostic Labels",
        "",
        strongest,
        "",
        "## 10. Weak Or Dangerous Labels",
        "",
        weakest,
        "",
        "## 11. Are New Labels Better Than Old v3 Regimes?",
        "",
        label_quality_text(summary, comparison),
        "",
        "## 12. Recommended v3.9 Trading-Variant Candidates",
        "",
        recommended_candidates(summary),
        "",
        "## 13. Proceed With v3.9 Or Return To v2 Alpha Extraction?",
        "",
        proceed_text(summary),
        "",
        "## Files",
        "",
        f"- Summary CSV: `{SUMMARY_CSV}`",
        f"- Forward-return CSV: `{FORWARD_CSV}`",
        f"- Transition CSV: `{TRANSITION_CSV}`",
        f"- Old/new comparison CSV: `{COMPARISON_CSV}`",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def aggregate_forward(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for label, group in frame.groupby("trend_health_label", dropna=False):
        rows.append(
            {
                "trend_health_label": label,
                "rows": int(len(group)),
                "bar_count": int(group["count"].sum()),
                "avg_count": mean(group["count"]),
                "avg_percentage": mean(group["percentage"]),
                "avg_next_1": weighted_mean(group, "avg_next_1", "count"),
                "avg_next_6": weighted_mean(group, "avg_next_6", "count"),
                "avg_next_24": weighted_mean(group, "avg_next_24", "count"),
                "avg_next_72": weighted_mean(group, "avg_next_72", "count"),
                "avg_hit_rate_next_24": weighted_mean(group, "hit_rate_next_24", "count"),
                "avg_sharpe_like_next_24": weighted_mean(group, "sharpe_like_next_24", "count"),
                "avg_target_position": weighted_mean(group, "avg_target_position", "count"),
                "avg_executed_position": weighted_mean(group, "avg_executed_position", "count"),
                "avg_v2_position": weighted_mean(group, "avg_v2_position", "count"),
                "positive_edge_rate": positive_edge_rate(group),
            }
        )
    return pd.DataFrame(rows).sort_values("avg_next_24", ascending=False)


def aggregate_old_new(comparison: pd.DataFrame) -> pd.DataFrame:
    if comparison.empty:
        return pd.DataFrame()
    return comparison.groupby(["asset", "old_long_regime", "trend_health_label"], dropna=False).agg(
        rows=("dataset", "count"),
        avg_percentage=("percentage", "mean"),
        avg_next_24=("avg_next_24", "mean"),
        avg_hit_rate_next_24=("hit_rate_next_24", "mean"),
    ).reset_index().sort_values(["asset", "old_long_regime", "avg_percentage"], ascending=[True, True, False])


def aggregate_transitions(transitions: pd.DataFrame) -> pd.DataFrame:
    if transitions.empty:
        return pd.DataFrame()
    return transitions.groupby(["asset", "trend_health_label"], dropna=False).mean(numeric_only=True).reset_index()


def cost_edge_summary(forward: pd.DataFrame) -> pd.DataFrame:
    if forward.empty:
        return pd.DataFrame()
    rows = []
    for (asset, label), group in forward.groupby(["asset", "trend_health_label"], dropna=False):
        gross = weighted_mean(group, "expected_gross_next_24", "count")
        cost = weighted_mean(group, "approx_entry_exit_cost", "count")
        rows.append(
            {
                "asset": asset,
                "trend_health_label": label,
                "rows": int(len(group)),
                "bar_count": int(group["count"].sum()),
                "avg_expected_gross_next_24": gross,
                "avg_approx_entry_exit_cost": cost,
                "avg_expected_net_edge": gross - cost if not np.isnan(gross) and not np.isnan(cost) else np.nan,
                "positive_edge_rate": positive_edge_rate(group),
            }
        )
    return pd.DataFrame(rows).sort_values(["asset", "avg_expected_net_edge"], ascending=[True, False])


def executive_summary(summary: pd.DataFrame) -> str:
    promising = summary[summary["promising"] == True]["trend_health_label"].tolist()
    if promising:
        return f"- Promising labels under the stated screen: {', '.join(f'`{label}`' for label in promising)}.\n- These labels are diagnostic only and are not yet trading rules."
    return "- No redesigned label cleanly passes all future-label criteria across BTC and ETH.\n- Some BTC labels are diagnostically useful, but they need more robustness checks before v3.9 trading variants."


def label_definitions() -> str:
    return "\n".join(
        [
            "- `healthy_bull`: bull + noise, non-extreme volatility, normal/caution drawdown, target 0.50.",
            "- `overextended_bull`: strong_bull with overheat or pullback short regime.",
            "- `late_bull`: strong_bull + noise.",
            "- `weak_bull`: bull with non-noise short regime or lower confidence.",
            "- `neutral_range`: neutral not classified as early recovery.",
            "- `early_recovery_candidate`: neutral/bear + recovery, v2 long, non-extreme volatility.",
            "- `oversold_rebound_candidate`: bear/strong_bear + noise/recovery, v2 long, caution/danger drawdown, non-extreme volatility.",
            "- `true_bear`: bear + breakdown with v2 flat, or unconfirmed bear.",
            "- `true_strong_bear`: strong_bear + breakdown with v2 flat, or unconfirmed strong_bear.",
            "- `unknown`: fallback when fields are insufficient.",
        ]
    )


def strongest_labels(summary: pd.DataFrame) -> str:
    btc = summary.sort_values("btc_avg_next_24", ascending=False).head(5)
    return frame_to_markdown(btc[["trend_health_label", "btc_rows", "btc_avg_next_24", "btc_hit_rate_next_24", "btc_positive_edge_rate", "eth_avg_next_24", "eth_hit_rate_next_24"]])


def weakest_labels(summary: pd.DataFrame) -> str:
    btc = summary.sort_values("btc_avg_next_24", ascending=True).head(5)
    return frame_to_markdown(btc[["trend_health_label", "btc_rows", "btc_avg_next_24", "btc_hit_rate_next_24", "btc_positive_edge_rate", "eth_avg_next_24", "eth_hit_rate_next_24"]])


def label_quality_text(summary: pd.DataFrame, comparison: pd.DataFrame) -> str:
    strong = comparison[(comparison["old_long_regime"] == "strong_bull") & (comparison["trend_health_label"].isin(["overextended_bull", "late_bull"]))]
    strong_share = float(strong.groupby(["asset", "dataset", "fee_rate"])["percentage"].sum().mean()) if not strong.empty else np.nan
    healthy = comparison[(comparison["old_long_regime"] == "bull") & (comparison["trend_health_label"] == "healthy_bull")]
    healthy_share = float(healthy["percentage"].mean()) if not healthy.empty else np.nan
    return f"Old strong_bull is largely decomposed into late/overextended labels; average split share across rows is {strong_share:.6g}. Old bull maps to healthy_bull with average share {healthy_share:.6g}. This is a cleaner semantic split, but predictive value remains mixed and must not be promoted directly."


def recommended_candidates(summary: pd.DataFrame) -> str:
    promising = summary[summary["promising"] == True]["trend_health_label"].tolist()
    if promising:
        return "Candidate labels for v3.9 study: " + ", ".join(f"`{label}`" for label in promising) + ". Keep all others diagnostic-only."
    return "`healthy_bull`, `early_recovery_candidate`, and `oversold_rebound_candidate` can be studied as diagnostics, but none should become trading logic without a controlled v3.9 experiment."


def proceed_text(summary: pd.DataFrame) -> str:
    if bool(summary["promising"].any()):
        return "Proceed with a narrow v3.9 estimator-relabeling experiment only, using the promising labels as inputs and keeping Risk Supervisor unchanged. v2_position should remain diagnostic confirmation unless separately validated."
    return "Pause trading-rule expansion. Use these labels to guide estimator redesign, while continuing v2 alpha extraction as the stronger BTC alpha path."


def is_promising_label(btc: pd.DataFrame, eth: pd.DataFrame) -> bool:
    if btc.empty or float(btc["count"].sum()) < 300:
        return False
    btc_next24 = weighted_mean(btc, "avg_next_24", "count")
    btc_hit = weighted_mean(btc, "hit_rate_next_24", "count")
    btc_edge = positive_edge_rate(btc)
    eth_next24 = weighted_mean(eth, "avg_next_24", "count") if not eth.empty else np.nan
    eth_ok = np.isnan(eth_next24) or eth_next24 > -0.01
    return bool(btc_next24 > 0.0 and btc_hit > 0.52 and btc_edge > 0.5 and eth_ok)


def weighted_mean(frame: pd.DataFrame, value: str, weight: str) -> float:
    if frame.empty:
        return np.nan
    values = pd.to_numeric(frame[value], errors="coerce")
    weights = pd.to_numeric(frame[weight], errors="coerce").fillna(0.0)
    mask = values.notna() & weights.gt(0)
    return float(np.average(values[mask], weights=weights[mask])) if mask.any() else np.nan


def positive_edge_rate(frame: pd.DataFrame) -> float:
    if frame.empty:
        return np.nan
    return float(pd.Series(frame["positive_edge"]).astype(bool).mean())


def mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else np.nan


def median(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.median()) if len(values) else np.nan


def std(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.std(ddof=0)) if len(values) else np.nan


def hit_rate(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float((values > 0.0).mean()) if len(values) else np.nan


def safe_ratio(numerator: float, denominator: float) -> float:
    return np.nan if denominator == 0.0 or np.isnan(denominator) else float(numerator / denominator)


def distribution_string(values: pd.Series) -> str:
    counts = values.value_counts(dropna=False).sort_index()
    return "; ".join(f"{bucket}:{count}" for bucket, count in counts.items())


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

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REPORT_PATH = Path("reports") / "v3_baseline_small_improvements.md"
INPUTS = {
    "BTC": {
        "before": Path("reports") / "v3_baseline_btc_1h_before_improvement.csv",
        "after": Path("reports") / "v3_baseline_btc_1h_summary.csv",
    },
    "ETH": {
        "before": Path("reports") / "v3_baseline_eth_1h_before_improvement.csv",
        "after": Path("reports") / "v3_baseline_eth_1h_summary.csv",
    },
}
METRICS = [
    "total_return",
    "annual_return",
    "max_drawdown",
    "sharpe_ratio",
    "number_of_trades",
    "turnover",
    "fee_drag",
    "average_exposure",
]


def load_v3(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["version"] == "v3.baseline"].copy()
    for metric in METRICS:
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    return frame


def build_comparison(asset: str, before: pd.DataFrame, after: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = ["dataset", "fee_rate"]
    joined = before[key + METRICS].merge(after[key + METRICS], on=key, suffixes=("_before", "_after"))
    joined.insert(0, "asset", asset)
    for metric in METRICS:
        joined[f"{metric}_change"] = joined[f"{metric}_after"] - joined[f"{metric}_before"]

    aggregate_rows = []
    for metric in METRICS:
        aggregate_rows.append(
            {
                "asset": asset,
                "metric": metric,
                "before_avg": float(before[metric].mean()),
                "after_avg": float(after[metric].mean()),
                "change_avg": float(after[metric].mean() - before[metric].mean()),
                "improved_rows": _improved_rows(metric, joined),
                "rows": len(joined),
            }
        )
    return joined, pd.DataFrame(aggregate_rows)


def _improved_rows(metric: str, joined: pd.DataFrame) -> int:
    change = joined[f"{metric}_change"]
    if metric in {"max_drawdown", "sharpe_ratio", "total_return", "annual_return"}:
        return int((change >= 0.0).sum())
    if metric in {"fee_drag", "turnover"}:
        return int((change <= 0.0).sum())
    return int((change >= 0.0).sum())


def write_report() -> None:
    comparisons = []
    aggregates = []
    for asset, paths in INPUTS.items():
        before = load_v3(paths["before"])
        after = load_v3(paths["after"])
        comparison, aggregate = build_comparison(asset, before, after)
        comparisons.append(comparison)
        aggregates.append(aggregate)

    comparison_table = pd.concat(comparisons, ignore_index=True)
    aggregate_table = pd.concat(aggregates, ignore_index=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# v3 Baseline Small Improvements Review",
        "",
        "This report compares the v3 baseline before and after two small, explainable changes. No particle filter, ML, extra indicators, ETH-specific tuning, or leverage was added.",
        "",
        "## Changes Made",
        "",
        "1. High-volatility risk behavior changed from hard `no_new_entry` to cap-only `normal` with `risk_cap = 0.75`.",
        "   - Why: baseline diagnostics showed v3 spent most bars in `no_new_entry`, which prevented the long-term controller from taking even capped exposure. High volatility should reduce exposure, while `extreme` volatility still remains `reduce_only`.",
        "2. Execution now reports exact no-change targets as `hold_target` instead of `no_trade_zone`.",
        "   - Why: the previous diagnostics counted many bars where target already equaled current position as skipped no-trade-zone events. This did not change returns directly, but it made diagnostics misleading.",
        "",
        "## Outcome Summary",
        "",
        *_build_outcome_summary(aggregate_table),
        "",
        "## Aggregate Before/After",
        "",
        _frame_to_markdown(aggregate_table),
        "",
        "## Detailed v3 Rows",
        "",
        _frame_to_markdown(
            comparison_table[
                [
                    "asset",
                    "dataset",
                    "fee_rate",
                    "total_return_before",
                    "total_return_after",
                    "total_return_change",
                    "max_drawdown_before",
                    "max_drawdown_after",
                    "max_drawdown_change",
                    "sharpe_ratio_before",
                    "sharpe_ratio_after",
                    "sharpe_ratio_change",
                    "turnover_before",
                    "turnover_after",
                    "turnover_change",
                    "fee_drag_before",
                    "fee_drag_after",
                    "fee_drag_change",
                    "average_exposure_before",
                    "average_exposure_after",
                    "average_exposure_change",
                ]
            ]
        ),
        "",
        "## Interpretation",
        "",
        "- BTC: the change tests whether v3 was over-blocked by high-volatility gating. Improvement should be accepted only if exposure rises without drawdown/fee drag exploding.",
        "- ETH: no ETH-specific parameters were tuned. If ETH improves, that supports v3 architecture generalization; if not, ETH remains weak and should stay a separate research path.",
        "- This is still a baseline architecture check, not parameter optimization.",
        "",
        "## Source Files",
        "",
        "- `src/v3/risk_supervisor.py`",
        "- `src/v3/execution_layer.py`",
        "- `tests/test_v3_risk_supervisor.py`",
        "- `tests/test_v3_execution_layer.py`",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


def _build_outcome_summary(aggregate_table: pd.DataFrame) -> list[str]:
    rows = []
    for asset in sorted(aggregate_table["asset"].unique()):
        local = aggregate_table[aggregate_table["asset"] == asset].set_index("metric")
        ret_change = float(local.loc["total_return", "change_avg"])
        sharpe_change = float(local.loc["sharpe_ratio", "change_avg"])
        mdd_change = float(local.loc["max_drawdown", "change_avg"])
        exposure_change = float(local.loc["average_exposure", "change_avg"])
        tone = "improved" if ret_change > 0.0 and sharpe_change >= 0.0 else "mixed or worse"
        rows.append(
            "- "
            f"{asset}: {tone}. Average total_return change={ret_change:.6g}, "
            f"Sharpe change={sharpe_change:.6g}, max_drawdown change={mdd_change:.6g}, "
            f"average_exposure change={exposure_change:.6g}."
        )
    rows.append(
        "- Keep this as a small baseline revision only if the goal is clearer risk semantics and ETH generalization; "
        "it is not a proven BTC performance upgrade."
    )
    return rows


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
    write_report()

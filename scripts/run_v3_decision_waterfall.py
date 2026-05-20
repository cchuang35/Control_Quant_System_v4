from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtester import load_ohlcv_csv
from scripts.run_v3_final_candidate import build_backtest_config, load_final_candidate_config
from src.v3.backtest_v3 import run_v3_backtest


BTC_WATERFALL_CSV = Path("reports") / "v3_decision_waterfall_btc_1h.csv"
ETH_WATERFALL_CSV = Path("reports") / "v3_decision_waterfall_eth_1h.csv"
REPORT_PATH = Path("reports") / "v3_exposure_blocking_diagnosis.md"


def run_waterfall() -> tuple[pd.DataFrame, pd.DataFrame]:
    config = load_final_candidate_config()
    fee_rates = tuple(float(value) for value in config["execution"]["fee_rates_to_validate"])
    btc = run_asset_waterfall("BTC", config, fee_rates)
    eth = run_asset_waterfall("ETH", config, fee_rates)
    BTC_WATERFALL_CSV.parent.mkdir(parents=True, exist_ok=True)
    btc.to_csv(BTC_WATERFALL_CSV, index=False)
    eth.to_csv(ETH_WATERFALL_CSV, index=False)
    write_report(btc, eth)
    return btc, eth


def run_asset_waterfall(asset: str, config: dict[str, Any], fee_rates: tuple[float, ...]) -> pd.DataFrame:
    dataset_key = "btc_datasets" if asset == "BTC" else "eth_datasets"
    paths = [Path(path) for path in config["validation"].get(dataset_key, []) if Path(path).exists()]
    frames: list[pd.DataFrame] = []
    for path in paths:
        dataset = path.stem
        print(f"asset={asset} dataset={dataset}")
        data = load_ohlcv_csv(path)
        for fee_rate in fee_rates:
            print(f"  fee={fee_rate:g}")
            result = run_v3_backtest(data, config=build_backtest_config(config, fee_rate=fee_rate))
            result.insert(0, "fee_rate", fee_rate)
            result.insert(0, "dataset", dataset)
            result.insert(0, "asset", asset)
            frames.append(add_waterfall_fields(result))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def add_waterfall_fields(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["risk_subreason"] = result["risk_reason"].astype(str)
    result["drawdown_cap_triggered"] = result["risk_reason"].astype(str).str.contains("drawdown_", regex=False)
    result["volatility_cap_triggered"] = result["risk_reason"].astype(str).str.contains("volatility_", regex=False)
    result["market_risk_state_triggered"] = result["risk_reason"].astype(str).str.contains("market_estimate_risk_off", regex=False)
    result["consecutive_loss_triggered"] = result["risk_reason"].astype(str).str.contains("consecutive_losses", regex=False)
    result["risk_off_triggered"] = result["risk_action"].astype(str).eq("risk_off")
    result["no_new_entry_triggered"] = result["risk_action"].astype(str).eq("no_new_entry")
    result["cooldown_blocks_addition"] = result["short_reason"].astype(str).str.contains("cooldown_blocks", regex=False)
    result["cooldown_blocks_entry"] = False
    result["no_trade_zone_block"] = result["execution_reason"].astype(str).str.contains("no_trade_zone", regex=False)
    result["cost_guard_block"] = result["risk_reason"].astype(str).str.contains("fee_drag_caution|turnover_caution", regex=True)
    result["reduce_only_block"] = result["execution_reason"].astype(str).str.contains("reduce_only_blocks_increase", regex=False)
    result["no_new_entry_block"] = result["execution_reason"].astype(str).str.contains("no_new_entry_blocks_entry", regex=False)
    result["risk_off_block"] = result["execution_reason"].astype(str).str.contains("risk_off:", regex=False)
    result["next_bar_return"] = result.groupby(["asset", "dataset", "fee_rate"])["asset_return"].shift(-1)
    result["next_24_bar_return"] = (
        result.groupby(["asset", "dataset", "fee_rate"])["asset_return"]
        .transform(lambda series: series.shift(-1).rolling(24, min_periods=1).sum().shift(-23))
    )
    classifications = result.apply(classify_binding_reason, axis=1)
    result["binding_block_reason"] = [item[0] for item in classifications]
    result["all_block_reasons"] = [item[1] for item in classifications]
    return result[waterfall_columns(result)]


def classify_binding_reason(row: pd.Series) -> tuple[str, str]:
    reasons: list[str] = []
    target = float(row.get("target_position", 0.0))
    executed = float(row.get("executed_position", 0.0))
    raw_target = float(row.get("raw_target_position", 0.0))
    risk_cap = float(row.get("risk_cap", 0.0))
    risk_action = str(row.get("risk_action", ""))
    risk_reason = str(row.get("risk_reason", ""))
    execution_reason = str(row.get("execution_reason", ""))

    if target <= 0.0:
        reasons.append("target_position_zero")
    if raw_target > 0.0 and target <= 0.0 and risk_cap <= 0.0:
        reasons.append("risk_cap_zero")
    if risk_action == "risk_off":
        reasons.append("risk_action_risk_off")
    if risk_action == "no_new_entry":
        reasons.append("risk_action_no_new_entry")
    if not bool(row.get("allow_entry", True)) and float(row.get("current_position", 0.0)) <= 0.0 and raw_target > 0.0:
        reasons.append("estimator_allow_entry_false")
    if not bool(row.get("allow_hold", True)) and float(row.get("current_position", 0.0)) > 0.0:
        reasons.append("estimator_allow_hold_false")
    if bool(row.get("cooldown_blocks_addition", False)) or bool(row.get("cooldown_blocks_entry", False)):
        reasons.append("cooldown_block")
    if "consecutive_losses" in risk_reason:
        reasons.append("consecutive_loss_block")
    if "market_estimate_risk_off" in risk_reason:
        reasons.append("market_risk_state_block")
    if "volatility_" in risk_reason:
        reasons.append("volatility_block")
    if "drawdown_" in risk_reason:
        reasons.append("drawdown_block")
    if raw_target > 0.0 and target <= 0.0 and risk_cap > 0.0:
        reasons.append("floor_rounding_to_zero")
    if "no_trade_zone" in execution_reason:
        reasons.append("no_trade_zone")
    if "fee_drag_caution" in risk_reason or "turnover_caution" in risk_reason:
        reasons.append("cost_guard")
    if "reduce_only_blocks_increase" in execution_reason or risk_action == "reduce_only":
        reasons.append("reduce_only")

    direct = direct_binding_reason(reasons, target, executed)
    all_reasons = ";".join(dict.fromkeys(reasons)) if reasons else "none"
    if target > 0.0 and executed == 0.0 and direct == "none":
        direct = "unknown"
        all_reasons = f"{all_reasons};unknown" if all_reasons != "none" else "unknown"
    return direct, all_reasons


def direct_binding_reason(reasons: list[str], target: float, executed: float) -> str:
    if executed != 0.0:
        return "none"
    priority = [
        "risk_cap_zero",
        "risk_action_risk_off",
        "risk_action_no_new_entry",
        "cooldown_block",
        "consecutive_loss_block",
        "market_risk_state_block",
        "volatility_block",
        "drawdown_block",
        "estimator_allow_entry_false",
        "estimator_allow_hold_false",
        "reduce_only",
        "no_trade_zone",
        "cost_guard",
        "floor_rounding_to_zero",
    ]
    for reason in priority:
        if reason in reasons:
            return reason
    if target <= 0.0:
        return "target_position_zero"
    return "unknown" if target > 0.0 and executed == 0.0 else "none"


def waterfall_columns(frame: pd.DataFrame) -> list[str]:
    desired = [
        "timestamp",
        "dataset",
        "asset",
        "fee_rate",
        "close",
        "asset_return",
        "next_bar_return",
        "next_24_bar_return",
        "long_regime",
        "short_regime",
        "trend_strength",
        "volatility_state",
        "drawdown_state",
        "risk_state",
        "confidence_score",
        "allow_entry",
        "allow_hold",
        "base_position",
        "short_adjustment",
        "raw_target_position",
        "risk_limited_position",
        "target_position",
        "risk_action",
        "risk_cap",
        "risk_reason",
        "risk_subreason",
        "drawdown_cap_triggered",
        "volatility_cap_triggered",
        "market_risk_state_triggered",
        "consecutive_loss_triggered",
        "cooldown_triggered",
        "risk_off_triggered",
        "no_new_entry_triggered",
        "cooldown_active",
        "cooldown_remaining",
        "cooldown_regime",
        "cooldown_blocks_entry",
        "cooldown_blocks_addition",
        "current_position",
        "executed_position",
        "trade_amount",
        "execution_reason",
        "no_trade_zone_block",
        "cost_guard_block",
        "reduce_only_block",
        "no_new_entry_block",
        "risk_off_block",
        "binding_block_reason",
        "all_block_reasons",
    ]
    return [column for column in desired if column in frame.columns]


def write_report(btc: pd.DataFrame, eth: pd.DataFrame) -> None:
    combined = pd.concat([btc, eth], ignore_index=True)
    lines = [
        "# v3.final_candidate Exposure Blocking Diagnosis",
        "",
        "This is a diagnostics-only decision waterfall for the explicit `v3.final_candidate`. It preserves trading behavior and uses forward returns only for post-run analysis.",
        "",
        "## A. Overall Summary",
        "",
        _frame_to_markdown(overall_summary(combined)),
        "",
        "### Target Position Distribution",
        "",
        _frame_to_markdown(distribution(combined, "target_position")),
        "",
        "### Executed Position Distribution",
        "",
        _frame_to_markdown(distribution(combined, "executed_position")),
        "",
        "## B. Binding Block Reason Counts",
        "",
        "### All Bars",
        "",
        _frame_to_markdown(distribution(combined, "binding_block_reason")),
        "",
        "### Bars With target_position > 0 And executed_position == 0",
        "",
        _frame_to_markdown(distribution(blocked_target_rows(combined), "binding_block_reason")),
        "",
        "## C. no_new_entry Diagnosis",
        "",
        "### Long Regime Distribution",
        "",
        _frame_to_markdown(distribution(no_new_entry_rows(combined), "long_regime")),
        "",
        "### Short Regime Distribution",
        "",
        _frame_to_markdown(distribution(no_new_entry_rows(combined), "short_regime")),
        "",
        "### Risk State Distribution",
        "",
        _frame_to_markdown(distribution(no_new_entry_rows(combined), "risk_state")),
        "",
        "### Confidence Buckets",
        "",
        _frame_to_markdown(confidence_bucket_distribution(no_new_entry_rows(combined))),
        "",
        "### no_new_entry Trigger Counts",
        "",
        _frame_to_markdown(no_new_entry_trigger_counts(no_new_entry_rows(combined))),
        "",
        "## D. Strong-Bull Blocking Diagnosis",
        "",
        _regime_blocking_section(combined, "strong_bull"),
        "",
        "## E. Bull Blocking Diagnosis",
        "",
        _regime_blocking_section(combined, "bull"),
        "",
        "## F. Risk Cap Versus Risk Action",
        "",
        _frame_to_markdown(risk_cap_vs_action(combined)),
        "",
        "## G. Execution Layer Diagnosis",
        "",
        _execution_layer_text(combined),
        "",
        "## H. Top 5 Underexposure Causes",
        "",
        _top_causes_text(combined),
        "",
        "## I. Recommended Next Experiments",
        "",
        "1. Risk-action ablation: keep risk caps unchanged, but compare current `no_new_entry` semantics against a capped-entry version where high-volatility/caution states cap exposure instead of blocking first entry.",
        "2. Market-risk-state audit: isolate bars where `risk_state == risk_off` and measure whether blocked bullish bars had positive forward returns.",
        "3. Consecutive-loss rule ablation: keep drawdown and volatility caps unchanged, but test consecutive-loss rules off, no-new-entry-only, and reduce-only variants.",
        "4. Estimator permission audit: report `allow_entry` and `allow_hold` as independent blockers before changing thresholds.",
        "5. Floor-rounding diagnostic: compare floor versus nearest rounding only after risk-action blockers are understood.",
        "",
        "## Output Files",
        "",
        f"- BTC waterfall CSV: `{BTC_WATERFALL_CSV}`",
        f"- ETH waterfall CSV: `{ETH_WATERFALL_CSV}`",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def overall_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in grouped_assets(frame):
        target = pd.to_numeric(group["target_position"], errors="coerce").fillna(0.0)
        executed = pd.to_numeric(group["executed_position"], errors="coerce").fillna(0.0)
        rows.append(
            {
                **keys,
                "bars": len(group),
                "average_target_position": float(target.mean()),
                "average_executed_position": float(executed.mean()),
                "target_to_executed_gap": float((target - executed).mean()),
                "target_gt_zero_executed_zero_bars": int(((target > 0.0) & (executed == 0.0)).sum()),
            }
        )
    return pd.DataFrame(rows)


def grouped_assets(frame: pd.DataFrame) -> list[tuple[dict[str, str], pd.DataFrame]]:
    groups: list[tuple[dict[str, str], pd.DataFrame]] = [({"asset": "ALL"}, frame)]
    for asset, group in frame.groupby("asset", dropna=False):
        groups.append(({"asset": str(asset)}, group))
    return groups


def distribution(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if frame.empty or column not in frame.columns:
        return pd.DataFrame()
    rows = []
    for keys, group in grouped_assets(frame):
        counts = group[column].value_counts(dropna=False).sort_index()
        for bucket, count in counts.items():
            rows.append({**keys, "bucket": bucket, "count": int(count), "percentage": float(count / max(len(group), 1))})
    return pd.DataFrame(rows)


def confidence_bucket_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    data = frame.copy()
    data["confidence_bucket"] = pd.cut(
        pd.to_numeric(data["confidence_score"], errors="coerce"),
        bins=[-0.001, 0.25, 0.50, 0.70, 0.85, 1.001],
        labels=["0-0.25", "0.25-0.50", "0.50-0.70", "0.70-0.85", "0.85-1.00"],
    )
    return distribution(data, "confidence_bucket")


def no_new_entry_trigger_counts(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    fields = [
        "allow_entry",
        "allow_hold",
        "cooldown_active",
        "consecutive_loss_triggered",
        "market_risk_state_triggered",
        "volatility_cap_triggered",
        "drawdown_cap_triggered",
    ]
    rows = []
    for keys, group in grouped_assets(frame):
        rows.append(
            {
                **keys,
                "bars": len(group),
                "allow_entry_false": int((~group["allow_entry"].astype(bool)).sum()),
                "allow_hold_false": int((~group["allow_hold"].astype(bool)).sum()),
                "cooldown_active": int(group["cooldown_active"].astype(bool).sum()),
                "consecutive_loss_active": int(group["consecutive_loss_triggered"].astype(bool).sum()),
                "market_risk_state_triggered": int(group["market_risk_state_triggered"].astype(bool).sum()),
                "volatility_cap_triggered": int(group["volatility_cap_triggered"].astype(bool).sum()),
                "drawdown_cap_triggered": int(group["drawdown_cap_triggered"].astype(bool).sum()),
            }
        )
    return pd.DataFrame(rows)


def _regime_blocking_section(frame: pd.DataFrame, regime: str) -> str:
    subset = frame[frame["long_regime"] == regime].copy()
    blocked = blocked_target_rows(subset)
    lines = [
        f"Bars with `long_regime == {regime}`: `{len(subset)}`.",
        "",
        "### Target Position Distribution",
        "",
        _frame_to_markdown(distribution(subset, "target_position")),
        "",
        "### Executed Position Distribution",
        "",
        _frame_to_markdown(distribution(subset, "executed_position")),
        "",
        "### Risk Action Distribution",
        "",
        _frame_to_markdown(distribution(subset, "risk_action")),
        "",
        "### Binding Block Reason Distribution",
        "",
        _frame_to_markdown(distribution(subset, "binding_block_reason")),
        "",
        "### Forward Return After Blocked Bars",
        "",
        _frame_to_markdown(forward_return_summary(blocked)),
    ]
    return "\n".join(lines)


def forward_return_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in grouped_assets(frame):
        rows.append(
            {
                **keys,
                "blocked_bars": len(group),
                "average_next_bar_return": float(pd.to_numeric(group["next_bar_return"], errors="coerce").mean()) if len(group) else np.nan,
                "average_next_24_bar_return": float(pd.to_numeric(group["next_24_bar_return"], errors="coerce").mean()) if len(group) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def risk_cap_vs_action(frame: pd.DataFrame) -> pd.DataFrame:
    target = pd.to_numeric(frame["target_position"], errors="coerce").fillna(0.0)
    executed = pd.to_numeric(frame["executed_position"], errors="coerce").fillna(0.0)
    cap = pd.to_numeric(frame["risk_cap"], errors="coerce").fillna(0.0)
    blocked = (target > 0.0) & (executed == 0.0)
    rows = []
    for keys, group in grouped_assets(frame.assign(_blocked=blocked, _cap=cap)):
        blocked_group = group[group["_blocked"]]
        rows.append(
            {
                **keys,
                "blocked_target_gt_zero_bars": len(blocked_group),
                "risk_cap_zero": int((blocked_group["_cap"] <= 0.0).sum()),
                "no_new_entry_with_risk_cap_gt_zero": int(((blocked_group["risk_action"] == "no_new_entry") & (blocked_group["_cap"] > 0.0)).sum()),
                "risk_off": int((blocked_group["risk_action"] == "risk_off").sum()),
                "reduce_only": int((blocked_group["risk_action"] == "reduce_only").sum()),
            }
        )
    return pd.DataFrame(rows)


def _execution_layer_text(frame: pd.DataFrame) -> str:
    no_trade_zone = int(frame["no_trade_zone_block"].astype(bool).sum())
    cost_guard = int(frame["cost_guard_block"].astype(bool).sum())
    reduce_only = int(frame["reduce_only_block"].astype(bool).sum())
    no_new_entry = int(frame["no_new_entry_block"].astype(bool).sum())
    risk_off = int(frame["risk_off_block"].astype(bool).sum())
    if no_trade_zone == 0 and cost_guard == 0:
        conclusion = "No-trade-zone and cost-guard logic are not the main blockers in this run."
    else:
        conclusion = "Execution-layer cost logic has nonzero blocks and should be inspected before changing strategy rules."
    return "\n".join(
        [
            f"- no_trade_zone_block bars: `{no_trade_zone}`",
            f"- cost_guard_block bars: `{cost_guard}`",
            f"- reduce_only_block bars: `{reduce_only}`",
            f"- no_new_entry_block bars: `{no_new_entry}`",
            f"- risk_off_block bars: `{risk_off}`",
            f"- Conclusion: {conclusion}",
        ]
    )


def _top_causes_text(frame: pd.DataFrame) -> str:
    blocked = blocked_target_rows(frame)
    counts = blocked["binding_block_reason"].value_counts()
    lines = []
    for rank, (reason, count) in enumerate(counts.head(5).items(), start=1):
        pct = count / max(len(blocked), 1)
        lines.append(f"{rank}. `{reason}`: {count} blocked target bars ({pct:.2%}).")
    return "\n".join(lines) if lines else "_No blocked nonzero targets found._"


def blocked_target_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame[(pd.to_numeric(frame["target_position"], errors="coerce") > 0.0) & (pd.to_numeric(frame["executed_position"], errors="coerce") == 0.0)]


def no_new_entry_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame[frame["risk_action"] == "no_new_entry"]


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


def main() -> None:
    run_waterfall()
    print(f"Wrote {BTC_WATERFALL_CSV}")
    print(f"Wrote {ETH_WATERFALL_CSV}")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()

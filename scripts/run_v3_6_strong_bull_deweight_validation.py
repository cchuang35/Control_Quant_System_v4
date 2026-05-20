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
    build_buy_and_hold_frame,
    build_ma_crossover_frame,
    input_frame_from_v1,
    load_final_candidate_config,
)
from src.v3.backtest_v3 import run_v3_backtest
from v2_small_cap import backtest_v2_btc_final_candidate_a, backtest_v2_final_candidate_a


BASE_CONFIG_PATH = Path("configs") / "v3_final_candidate.yaml"
G_CONFIG_PATH = Path("configs") / "v3_6_strong_bull_deweight.yaml"
BTC_REPORT = Path("reports") / "v3_6_strong_bull_deweight_rolling_btc_1h.md"
ETH_REPORT = Path("reports") / "v3_6_strong_bull_deweight_rolling_eth_1h.md"
COMPARISON_REPORT = Path("reports") / "v3_6_strong_bull_deweight_vs_v3_vs_v2.md"
FULL_CSV = Path("reports") / "v3_6_strong_bull_deweight_full_period.csv"
ROLLING_CSV = Path("reports") / "v3_6_strong_bull_deweight_rolling.csv"
DIAGNOSTICS_CSV = Path("reports") / "v3_6_strong_bull_deweight_diagnostics.csv"

ROLLING_WINDOWS = {
    "90d": 90 * 24,
    "180d": 180 * 24,
    "365d": 365 * 24,
    "2y": 2 * 365 * 24,
}


def main() -> None:
    base_config = load_final_candidate_config(BASE_CONFIG_PATH)
    g_config = load_final_candidate_config(G_CONFIG_PATH)
    fee_rates = tuple(float(value) for value in g_config["execution"]["fee_rates_to_validate"])
    full_rows: list[dict[str, Any]] = []
    rolling_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []

    for asset, dataset_key in [("BTC", "btc_datasets"), ("ETH", "eth_datasets")]:
        paths = [Path(path) for path in g_config["validation"].get(dataset_key, []) if Path(path).exists()]
        for path in paths:
            dataset = path.stem
            print(f"asset={asset} dataset={dataset}")
            data = load_ohlcv_csv(path)
            for fee_rate in fee_rates:
                print(f"  fee={fee_rate:g}")
                frames = build_frames(asset, data, fee_rate, base_config, g_config)
                for version, frame in frames.items():
                    full_rows.append({"asset": asset, "dataset": dataset, "fee_rate": fee_rate, "version": version, **summarize_window(frame)})
                    diagnostic_rows.extend(diagnostic_rows_for(asset, dataset, fee_rate, version, frame))
                    for window_name, window in rolling_windows(frame):
                        rolling_rows.append(
                            {
                                "asset": asset,
                                "dataset": dataset,
                                "fee_rate": fee_rate,
                                "version": version,
                                "window": window_name,
                                **summarize_window(window),
                            }
                        )

    full = pd.DataFrame(full_rows)
    rolling = pd.DataFrame(rolling_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    BTC_REPORT.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(FULL_CSV, index=False)
    rolling.to_csv(ROLLING_CSV, index=False)
    diagnostics.to_csv(DIAGNOSTICS_CSV, index=False)
    write_asset_report("BTC", full, rolling, diagnostics, BTC_REPORT)
    write_asset_report("ETH", full, rolling, diagnostics, ETH_REPORT)
    write_comparison_report(full, rolling, diagnostics)
    print(f"Wrote {BTC_REPORT}")
    print(f"Wrote {ETH_REPORT}")
    print(f"Wrote {COMPARISON_REPORT}")
    print(f"Wrote {FULL_CSV}")
    print(f"Wrote {ROLLING_CSV}")
    print(f"Wrote {DIAGNOSTICS_CSV}")


def build_frames(asset: str, data: pd.DataFrame, fee_rate: float, base_config: dict[str, Any], g_config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    v1 = run_backtest_fast(
        data,
        fee_rate=fee_rate,
        periods_per_year=PERIODS_PER_YEAR,
        progress_every=10000 if len(data) > 15000 else None,
    )
    v2_input = input_frame_from_v1(v1)
    v2_func = backtest_v2_btc_final_candidate_a if asset == "BTC" else backtest_v2_final_candidate_a
    v2_label = "v2.btc_final_candidate_A" if asset == "BTC" else "v2.final_candidate_A_cd120_on_ETH"
    return {
        "v3.final_candidate": run_v3_backtest(data, config=build_backtest_config(base_config, fee_rate=fee_rate)),
        "v3.6_strong_bull_deweight": run_v3_backtest(data, config=build_backtest_config(g_config, fee_rate=fee_rate)),
        v2_label: v2_func(v2_input, fee_rate=fee_rate, v1_entry_threshold=V1_ENTRY_THRESHOLD, cooldown_bars=120),
        "buy_and_hold": build_buy_and_hold_frame(data, fee_rate),
        "ma20_ma60": build_ma_crossover_frame(data, fee_rate),
    }


def rolling_windows(frame: pd.DataFrame, step_days: int = 30) -> list[tuple[str, pd.DataFrame]]:
    windows = []
    for label, size in ROLLING_WINDOWS.items():
        if len(frame) < size:
            continue
        step = step_days * 24
        index = 0
        number = 1
        while index + size <= len(frame):
            windows.append((f"{label}_{number}", frame.iloc[index : index + size].copy()))
            index += step
            number += 1
        if (len(frame) - size) % step != 0:
            windows.append((f"{label}_final", frame.iloc[-size:].copy()))
    return windows


def summarize_window(frame: pd.DataFrame) -> dict[str, float | int | str]:
    returns = pd.to_numeric(column(frame, "strategy_return_net"), errors="coerce").fillna(0.0)
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0 if len(equity) else pd.Series(dtype=float)
    position = position_series(frame)
    trade_amount = trade_amount_series(frame, position)
    fee_cost = pd.to_numeric(frame.get("fee_cost", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    std = float(returns.std(ddof=0))
    periods = max(len(returns), 1)
    final_equity = float(equity.iloc[-1]) if len(equity) else 1.0
    target = pd.to_numeric(frame.get("target_position", pd.Series(np.nan, index=frame.index)), errors="coerce")
    executed = pd.to_numeric(frame.get("executed_position", position), errors="coerce").fillna(0.0)
    return {
        "window_start": str(frame["timestamp"].iloc[0]) if "timestamp" in frame.columns and len(frame) else "",
        "window_end": str(frame["timestamp"].iloc[-1]) if "timestamp" in frame.columns and len(frame) else "",
        "bars": int(len(frame)),
        "total_return": final_equity - 1.0,
        "annual_return": final_equity ** (PERIODS_PER_YEAR / periods) - 1.0,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "sharpe_ratio": 0.0 if std == 0.0 else float(returns.mean() / std * np.sqrt(PERIODS_PER_YEAR)),
        "turnover": float(trade_amount.sum()),
        "fee_drag": float(fee_cost.sum()),
        "average_exposure": float(position.abs().mean()) if len(position) else 0.0,
        "max_exposure": float(position.abs().max()) if len(position) else 0.0,
        "number_of_trades": int((trade_amount > 0.0).sum()),
        "target_to_executed_gap": float((target.fillna(0.0) - executed).mean()) if "target_position" in frame.columns else np.nan,
        "strong_bull_exposure": strong_bull_exposure(frame),
        "strong_bull_strategy_contribution": strong_bull_contribution(frame),
        "exposure_distribution": distribution_string(position),
        "risk_action_distribution": risk_action_distribution(frame),
    }


def diagnostic_rows_for(asset: str, dataset: str, fee_rate: float, version: str, frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    rows.extend(distribution_rows(asset, dataset, fee_rate, version, "exposure_distribution", position_series(frame)))
    if "risk_action" in frame.columns:
        rows.extend(distribution_rows(asset, dataset, fee_rate, version, "risk_action_distribution", frame["risk_action"].astype(str)))
    if "long_regime" in frame.columns:
        strong = frame[frame["long_regime"] == "strong_bull"]
        rows.append(
            {
                "asset": asset,
                "dataset": dataset,
                "fee_rate": fee_rate,
                "version": version,
                "diagnostic": "strong_bull_summary",
                "bucket": "strong_bull",
                "count": int(len(strong)),
                "average_exposure": strong_bull_exposure(frame),
                "strategy_contribution": strong_bull_contribution(frame),
            }
        )
    return rows


def distribution_rows(asset: str, dataset: str, fee_rate: float, version: str, diagnostic: str, values: pd.Series) -> list[dict[str, Any]]:
    counts = values.value_counts(dropna=False).sort_index()
    total = max(int(counts.sum()), 1)
    return [
        {
            "asset": asset,
            "dataset": dataset,
            "fee_rate": fee_rate,
            "version": version,
            "diagnostic": diagnostic,
            "bucket": bucket,
            "count": int(count),
            "percentage": float(count / total),
        }
        for bucket, count in counts.items()
    ]


def write_asset_report(asset: str, full: pd.DataFrame, rolling: pd.DataFrame, diagnostics: pd.DataFrame, path: Path) -> None:
    asset_full = full[full["asset"] == asset].copy()
    asset_rolling = rolling[rolling["asset"] == asset].copy()
    g_vs_v3 = compare_versions(asset_rolling, "v3.6_strong_bull_deweight", "v3.final_candidate")
    aggregate = aggregate_versions(asset_full)
    rolling_aggregate = aggregate_versions(asset_rolling)
    lines = [
        f"# v3.6 Strong-Bull Deweight Rolling Validation {asset}USDT 1h",
        "",
        "This validates only the G variant: strong_bull base position 0.50 instead of 0.75. No Risk Supervisor, leverage, particle filter, or v2-assisted early-entry changes are included.",
        "",
        "## Full-Period Aggregate",
        "",
        frame_to_markdown(aggregate),
        "",
        "## Rolling Aggregate",
        "",
        frame_to_markdown(rolling_aggregate),
        "",
        "## v3.6 Versus v3.final_candidate Rolling Win Rates",
        "",
        frame_to_markdown(g_vs_v3),
        "",
        "## Rolling Rows Sample",
        "",
        frame_to_markdown(asset_rolling.head(80)),
        "",
        "## Diagnostics",
        "",
        frame_to_markdown(diagnostics[diagnostics["asset"] == asset].head(80)),
        "",
        "## Files",
        "",
        f"- Full-period CSV: `{FULL_CSV}`",
        f"- Rolling CSV: `{ROLLING_CSV}`",
        f"- Diagnostics CSV: `{DIAGNOSTICS_CSV}`",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_comparison_report(full: pd.DataFrame, rolling: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    btc_full = full[full["asset"] == "BTC"]
    eth_full = full[full["asset"] == "ETH"]
    btc_roll = rolling[rolling["asset"] == "BTC"]
    eth_roll = rolling[rolling["asset"] == "ETH"]
    btc_g_v3 = compare_versions(btc_roll, "v3.6_strong_bull_deweight", "v3.final_candidate")
    btc_g_v2 = compare_versions(btc_roll, "v3.6_strong_bull_deweight", "v2.btc_final_candidate_A")
    eth_g_v3 = compare_versions(eth_roll, "v3.6_strong_bull_deweight", "v3.final_candidate")
    diagnostics_table = strong_bull_diagnostics(diagnostics)
    decision = replacement_decision(btc_g_v3, eth_g_v3)
    lines = [
        "# v3.6 Strong-Bull Deweight vs v3 vs v2",
        "",
        "## 1. Executive Summary",
        "",
        decision,
        "",
        "## 2. Variant Definition",
        "",
        "`v3.6_strong_bull_deweight` changes only the long-term controller base mapping: `strong_bull` goes from `0.75` to `0.50`; `bull=0.50`, `neutral=0.25`, `bear=0.00`, and `strong_bear=0.00` remain unchanged.",
        "",
        "## 3. BTC Full-Period Comparison",
        "",
        frame_to_markdown(aggregate_versions(btc_full)),
        "",
        "## 4. BTC Rolling Validation Comparison",
        "",
        "v3.6 versus v3.final_candidate:",
        "",
        frame_to_markdown(btc_g_v3),
        "",
        "v3.6 versus v2:",
        "",
        frame_to_markdown(btc_g_v2),
        "",
        "## 5. ETH Full-Period Comparison",
        "",
        frame_to_markdown(aggregate_versions(eth_full)),
        "",
        "## 6. ETH Rolling Validation Comparison",
        "",
        frame_to_markdown(eth_g_v3),
        "",
        "## 7. Strong-Bull Deweighting Diagnostics",
        "",
        frame_to_markdown(diagnostics_table),
        "",
        "## 8. Risk / Turnover / Fee Analysis",
        "",
        risk_fee_analysis(btc_g_v3, btc_g_v2, eth_g_v3),
        "",
        "## 9. Should G Replace v3.final_candidate?",
        "",
        replacement_answer(btc_g_v3, eth_g_v3),
        "",
        "## 10. Is v3.6 Still An Architecture Checkpoint?",
        "",
        "Yes. This is a small controller cleanup, not evidence that v3 has become a BTC alpha upgrade over v2.",
        "",
        "## 11. Recommended Next Step",
        "",
        "If accepted, freeze `v3.6_strong_bull_deweight` as a candidate and run a targeted estimator/controller remapping study focused on replacing the weak `strong_bull` semantics and isolating the durable `bull + noise` slice.",
        "",
        "## Files",
        "",
        f"- BTC rolling report: `{BTC_REPORT}`",
        f"- ETH rolling report: `{ETH_REPORT}`",
        f"- Full-period CSV: `{FULL_CSV}`",
        f"- Rolling CSV: `{ROLLING_CSV}`",
        f"- Diagnostics CSV: `{DIAGNOSTICS_CSV}`",
    ]
    COMPARISON_REPORT.write_text("\n".join(lines), encoding="utf-8")


def aggregate_versions(frame: pd.DataFrame) -> pd.DataFrame:
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
            avg_turnover=("turnover", "mean"),
            avg_fee_drag=("fee_drag", "mean"),
            avg_average_exposure=("average_exposure", "mean"),
            max_exposure=("max_exposure", "max"),
            avg_number_of_trades=("number_of_trades", "mean"),
            avg_strong_bull_exposure=("strong_bull_exposure", "mean"),
            avg_strong_bull_contribution=("strong_bull_strategy_contribution", "mean"),
            avg_target_to_executed_gap=("target_to_executed_gap", "mean"),
        )
        .reset_index()
        .sort_values("avg_sharpe_ratio", ascending=False)
    )


def compare_versions(frame: pd.DataFrame, candidate: str, baseline: str) -> pd.DataFrame:
    keys = ["asset", "dataset", "fee_rate", "window"]
    c = frame[frame["version"] == candidate]
    b = frame[frame["version"] == baseline]
    if c.empty or b.empty:
        return pd.DataFrame()
    metrics = ["total_return", "annual_return", "max_drawdown", "sharpe_ratio", "turnover", "fee_drag", "average_exposure", "number_of_trades", "target_to_executed_gap"]
    joined = c[keys + metrics].merge(b[keys + metrics], on=keys, suffixes=("_candidate", "_baseline"))
    rows = []
    for metric in metrics:
        delta = joined[f"{metric}_candidate"] - joined[f"{metric}_baseline"]
        higher_is_better = metric not in {"turnover", "fee_drag", "number_of_trades", "target_to_executed_gap"}
        wins = delta > 0 if higher_is_better else delta < 0
        non_worse = delta >= 0 if higher_is_better else delta <= 0
        changed = delta.abs() > 1e-12
        rows.append(
            {
                "candidate": candidate,
                "baseline": baseline,
                "metric": metric,
                "windows": int(len(joined)),
                "changed_count": int(changed.sum()),
                "changed_rate": float(changed.mean()) if len(joined) else np.nan,
                "win_count": int(wins.sum()),
                "win_rate": float(wins.mean()) if len(joined) else np.nan,
                "non_worse_count": int(non_worse.sum()),
                "non_worse_rate": float(non_worse.mean()) if len(joined) else np.nan,
                "avg_candidate": float(joined[f"{metric}_candidate"].mean()) if len(joined) else np.nan,
                "avg_baseline": float(joined[f"{metric}_baseline"].mean()) if len(joined) else np.nan,
                "avg_delta": float(delta.mean()) if len(joined) else np.nan,
                "worst_delta": float(delta.min()) if len(joined) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def strong_bull_diagnostics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    subset = diagnostics[diagnostics["diagnostic"] == "strong_bull_summary"].copy()
    if subset.empty:
        return pd.DataFrame()
    return (
        subset.groupby(["asset", "version"], dropna=False)
        .agg(
            avg_count=("count", "mean"),
            avg_strong_bull_exposure=("average_exposure", "mean"),
            avg_strong_bull_strategy_contribution=("strategy_contribution", "mean"),
        )
        .reset_index()
    )


def replacement_decision(btc_g_v3: pd.DataFrame, eth_g_v3: pd.DataFrame) -> str:
    return "\n".join(
        [
            f"- BTC rolling Sharpe win rate versus v3.final_candidate: {metric_win_rate(btc_g_v3, 'sharpe_ratio'):.3g}.",
            f"- BTC rolling Sharpe changed-window rate versus v3.final_candidate: {metric_changed_rate(btc_g_v3, 'sharpe_ratio'):.3g}.",
            f"- BTC rolling Sharpe non-worse rate versus v3.final_candidate: {metric_non_worse_rate(btc_g_v3, 'sharpe_ratio'):.3g}.",
            f"- BTC rolling max-drawdown win rate versus v3.final_candidate: {metric_win_rate(btc_g_v3, 'max_drawdown'):.3g}.",
            f"- BTC rolling turnover win rate versus v3.final_candidate: {metric_win_rate(btc_g_v3, 'turnover'):.3g}.",
            f"- BTC rolling fee-drag win rate versus v3.final_candidate: {metric_win_rate(btc_g_v3, 'fee_drag'):.3g}.",
            f"- ETH rolling Sharpe win rate versus v3.final_candidate: {metric_win_rate(eth_g_v3, 'sharpe_ratio'):.3g}.",
        ]
    )


def replacement_answer(btc_g_v3: pd.DataFrame, eth_g_v3: pd.DataFrame) -> str:
    sharpe = metric_win_rate(btc_g_v3, "sharpe_ratio")
    sharpe_changed = metric_changed_rate(btc_g_v3, "sharpe_ratio")
    sharpe_non_worse = metric_non_worse_rate(btc_g_v3, "sharpe_ratio")
    drawdown = metric_win_rate(btc_g_v3, "max_drawdown")
    turnover = metric_win_rate(btc_g_v3, "turnover")
    fee = metric_win_rate(btc_g_v3, "fee_drag")
    eth_dd = metric_win_rate(eth_g_v3, "max_drawdown")
    if sharpe >= 0.5 and drawdown >= 0.5 and turnover >= 0.5 and fee >= 0.5 and eth_dd >= 0.4:
        return "G passes the mechanical rolling criteria for replacing v3.final_candidate as a v3.6 candidate, but it remains a risk/control cleanup rather than a proven alpha upgrade."
    if sharpe_non_worse >= 0.95 and sharpe_changed < 0.05:
        return "G is mostly non-worse because most rolling windows are unchanged, but its active improvement is sparse. Do not promote it to v3.6_final_candidate on rolling evidence alone."
    return "G does not cleanly pass the rolling replacement criteria. Treat it as a useful diagnostic cleanup unless a deeper review accepts the tradeoff."


def risk_fee_analysis(btc_g_v3: pd.DataFrame, btc_g_v2: pd.DataFrame, eth_g_v3: pd.DataFrame) -> str:
    return "\n".join(
        [
            f"- BTC vs v3: turnover win rate {metric_win_rate(btc_g_v3, 'turnover'):.3g}; fee-drag win rate {metric_win_rate(btc_g_v3, 'fee_drag'):.3g}.",
            f"- BTC vs v2: max-drawdown win rate {metric_win_rate(btc_g_v2, 'max_drawdown'):.3g}; turnover win rate {metric_win_rate(btc_g_v2, 'turnover'):.3g}; fee-drag win rate {metric_win_rate(btc_g_v2, 'fee_drag'):.3g}.",
            f"- ETH vs v3: max-drawdown win rate {metric_win_rate(eth_g_v3, 'max_drawdown'):.3g}; Sharpe win rate {metric_win_rate(eth_g_v3, 'sharpe_ratio'):.3g}.",
        ]
    )


def metric_win_rate(frame: pd.DataFrame, metric: str) -> float:
    if frame.empty:
        return float("nan")
    row = frame[frame["metric"] == metric]
    return float(row["win_rate"].iloc[0]) if not row.empty else float("nan")


def metric_changed_rate(frame: pd.DataFrame, metric: str) -> float:
    if frame.empty:
        return float("nan")
    row = frame[frame["metric"] == metric]
    return float(row["changed_rate"].iloc[0]) if not row.empty else float("nan")


def metric_non_worse_rate(frame: pd.DataFrame, metric: str) -> float:
    if frame.empty:
        return float("nan")
    row = frame[frame["metric"] == metric]
    return float(row["non_worse_rate"].iloc[0]) if not row.empty else float("nan")


def column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        raise ValueError(f"frame missing column: {name}")
    return frame[name]


def position_series(frame: pd.DataFrame) -> pd.Series:
    for name in ("executed_position", "final_position", "position", "current_exposure"):
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce").fillna(0.0)
    raise ValueError("frame missing position column")


def trade_amount_series(frame: pd.DataFrame, position: pd.Series) -> pd.Series:
    for name in ("trade_amount", "trade_size"):
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce").fillna(0.0).abs()
    return position.diff().abs().fillna(position.abs())


def strong_bull_exposure(frame: pd.DataFrame) -> float:
    if "long_regime" not in frame.columns:
        return np.nan
    strong = frame[frame["long_regime"] == "strong_bull"]
    if strong.empty:
        return 0.0
    return float(position_series(strong).abs().mean())


def strong_bull_contribution(frame: pd.DataFrame) -> float:
    if "long_regime" not in frame.columns or "strategy_return_net" not in frame.columns:
        return np.nan
    strong = frame[frame["long_regime"] == "strong_bull"]
    return float(pd.to_numeric(strong["strategy_return_net"], errors="coerce").fillna(0.0).sum())


def risk_action_distribution(frame: pd.DataFrame) -> str:
    if "risk_action" not in frame.columns:
        return ""
    return distribution_string(frame["risk_action"].astype(str))


def distribution_string(values: pd.Series) -> str:
    counts = values.value_counts(dropna=False).sort_index()
    return "; ".join(f"{bucket}:{count}" for bucket, count in counts.items())


def frame_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No data_"
    display = frame.copy()
    for col in display.select_dtypes(include=[np.number]).columns:
        display[col] = display[col].map(lambda value: f"{value:.6g}")
    columns = [str(col) for col in display.columns]
    rows = display.astype(object).where(pd.notna(display), "").astype(str).values.tolist()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator] + body)


if __name__ == "__main__":
    main()

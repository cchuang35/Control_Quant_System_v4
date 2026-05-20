from __future__ import annotations

from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtester import load_ohlcv_csv, run_backtest_fast
from src.v3.backtest_v3 import BacktestV3Config, run_v3_backtest
from src.v3.diagnostics import build_v3_diagnostics
from v2_small_cap import backtest_v2_final_candidate_a


FEE_RATES = (0.0005, 0.0010, 0.0020)
PERIODS_PER_YEAR = 365 * 24
V1_ENTRY_THRESHOLD = 0.10
REPORT_PATH = Path("reports") / "v3_baseline_eth_1h.md"
SUMMARY_CSV = Path("reports") / "v3_baseline_eth_1h_summary.csv"
DIAGNOSTICS_CSV = Path("reports") / "v3_baseline_eth_1h_diagnostics.csv"
ETH_DATASETS = (
    "ethusdt_1h_365d.csv",
    "ethusdt_1h_2y.csv",
    "ethusdt_1h_3y.csv",
    "ethusdt_1h_5y.csv",
)


def discover_eth_datasets(data_dir: Path = Path("data")) -> dict[str, Path]:
    return {
        path.stem: path
        for name in ETH_DATASETS
        for path in [data_dir / name]
        if path.exists()
    }


def run_baseline() -> tuple[pd.DataFrame, pd.DataFrame]:
    datasets = discover_eth_datasets()
    if not datasets:
        raise FileNotFoundError("No ETHUSDT 1h datasets found under data/")

    summary_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []

    for dataset, path in datasets.items():
        print(f"dataset={dataset}")
        data = load_ohlcv_csv(path)
        for fee_rate in FEE_RATES:
            print(f"  fee={fee_rate:g}")
            start = perf_counter()
            v1 = run_backtest_fast(
                data,
                fee_rate=fee_rate,
                periods_per_year=PERIODS_PER_YEAR,
                progress_every=10000 if len(data) > 15000 else None,
            )
            v1_runtime = perf_counter() - start
            v2_input = input_frame_from_v1(v1)

            frames = {
                "v2.final_candidate_A_cd120_on_ETH": backtest_v2_final_candidate_a(
                    v2_input,
                    fee_rate=fee_rate,
                    v1_entry_threshold=V1_ENTRY_THRESHOLD,
                    cooldown_bars=120,
                ),
                "v3.baseline": run_v3_backtest(data, config=BacktestV3Config(fee_rate=fee_rate, cooldown_bars=120)),
                "buy_and_hold": build_buy_and_hold_frame(data, fee_rate),
                "ma20_ma60": build_ma_crossover_frame(data, fee_rate),
            }

            for version, frame in frames.items():
                summary_rows.append(
                    {
                        "dataset": dataset,
                        "fee_rate": fee_rate,
                        "version": version,
                        **summarize_frame(frame),
                        "runtime_sec": v1_runtime if version == "v2.final_candidate_A_cd120_on_ETH" else 0.0,
                    }
                )

            v3_diagnostics = build_v3_diagnostics(frames["v3.baseline"])
            diagnostic_rows.extend(flatten_v3_diagnostics(dataset, fee_rate, v3_diagnostics))

    return pd.DataFrame(summary_rows), pd.DataFrame(diagnostic_rows)


def input_frame_from_v1(result: Any) -> pd.DataFrame:
    exposure = result.exposure_history.reset_index(drop=True).copy()
    return pd.DataFrame(
        {
            "timestamp": exposure["execution_timestamp"],
            "close": exposure["close"].astype(float),
            "current_exposure": exposure["current_exposure"].astype(float),
        }
    )


def build_buy_and_hold_frame(data: pd.DataFrame, fee_rate: float) -> pd.DataFrame:
    close = pd.to_numeric(data["close"], errors="coerce").astype(float).reset_index(drop=True)
    timestamp = data["timestamp"].reset_index(drop=True) if "timestamp" in data.columns else pd.Series(data.index)
    asset_return = close.pct_change().fillna(0.0)
    position = pd.Series(1.0, index=close.index)
    trade_amount = position.diff().abs().fillna(position.abs())
    fee_cost = trade_amount * fee_rate
    gross = position.shift(1).fillna(0.0) * asset_return
    net = gross - fee_cost
    equity = (1.0 + net).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "close": close,
            "asset_return": asset_return,
            "position": position,
            "trade_amount": trade_amount,
            "fee_cost": fee_cost,
            "strategy_return_net": net,
            "equity_curve": equity,
            "drawdown": drawdown,
        }
    )


def build_ma_crossover_frame(data: pd.DataFrame, fee_rate: float) -> pd.DataFrame:
    close = pd.to_numeric(data["close"], errors="coerce").astype(float).reset_index(drop=True)
    timestamp = data["timestamp"].reset_index(drop=True) if "timestamp" in data.columns else pd.Series(data.index)
    asset_return = close.pct_change().fillna(0.0)
    ma20 = close.rolling(20, min_periods=1).mean()
    ma60 = close.rolling(60, min_periods=1).mean()
    position = (ma20 > ma60).astype(float)
    trade_amount = position.diff().abs().fillna(position.abs())
    fee_cost = trade_amount * fee_rate
    gross = position.shift(1).fillna(0.0) * asset_return
    net = gross - fee_cost
    equity = (1.0 + net).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "close": close,
            "asset_return": asset_return,
            "position": position,
            "trade_amount": trade_amount,
            "fee_cost": fee_cost,
            "strategy_return_net": net,
            "equity_curve": equity,
            "drawdown": drawdown,
        }
    )


def summarize_frame(frame: pd.DataFrame) -> dict[str, float | int]:
    returns = pd.to_numeric(_column(frame, "strategy_return_net"), errors="coerce").fillna(0.0)
    equity = _equity(frame, returns)
    drawdown = _drawdown(frame, equity)
    position = _position(frame)
    trade_amount = _trade_amount(frame, position)
    fee_cost = pd.to_numeric(frame.get("fee_cost", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    final_equity = float(equity.iloc[-1]) if len(equity) else 1.0
    periods = max(len(returns), 1)
    return_std = float(returns.std(ddof=0))
    return {
        "total_return": final_equity - 1.0,
        "annual_return": final_equity ** (PERIODS_PER_YEAR / periods) - 1.0,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "sharpe_ratio": 0.0 if return_std == 0.0 else float(returns.mean() / return_std * np.sqrt(PERIODS_PER_YEAR)),
        "number_of_trades": int((trade_amount > 0.0).sum()),
        "turnover": float(trade_amount.sum()),
        "fee_drag": float(fee_cost.sum()),
        "average_exposure": float(position.abs().mean()) if len(position) else 0.0,
    }


def flatten_v3_diagnostics(dataset: str, fee_rate: float, diagnostics: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for diagnostic, frame in {
        "long_regime_performance": diagnostics["long_regime_performance"],
        "short_regime_performance": diagnostics["short_regime_performance"],
        "executed_position_distribution": diagnostics["executed_position_distribution"],
        "risk_action_counts": diagnostics["risk_action_counts"],
    }.items():
        for row in frame.to_dict("records"):
            normalized = {
                "dataset": dataset,
                "fee_rate": fee_rate,
                "diagnostic": diagnostic,
                **row,
            }
            rows.append(normalized)
    execution_summary = diagnostics["execution_summary"].iloc[0].to_dict()
    rows.append(
        {
            "dataset": dataset,
            "fee_rate": fee_rate,
            "diagnostic": "execution_summary",
            **execution_summary,
        }
    )
    return rows


def write_report(summary: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)
    diagnostics.to_csv(DIAGNOSTICS_CSV, index=False)

    v3_summary = summary[summary["version"] == "v3.baseline"].copy()
    v2_summary = summary[summary["version"] == "v2.final_candidate_A_cd120_on_ETH"].copy()
    conclusion = build_conclusion(v3_summary, v2_summary)
    poor_regimes = build_poor_regime_table(diagnostics)

    lines = [
        "# v3 Baseline ETHUSDT 1h Validation",
        "",
        "This is an untuned ETH generalization check. It uses the current v3 default ETH-facing 1h settings and does not tune ETH-specific parameters.",
        "",
        "## Scope",
        "",
        f"- Datasets: {', '.join(sorted(summary['dataset'].unique()))}",
        f"- Fee rates: {', '.join(f'{fee:g}' for fee in FEE_RATES)}",
        "- Versions: v2.final_candidate_A_cd120_on_ETH, v3.baseline, buy_and_hold, ma20_ma60",
        "- v2 comparison note: the v2 candidate is applied to ETH as a reference only; existing docs say it is not an ETH-validated final strategy.",
        "",
        "## Direct Answer",
        "",
        conclusion,
        "",
        "## Comparison Table",
        "",
        _frame_to_markdown(
            summary[
                [
                    "dataset",
                    "fee_rate",
                    "version",
                    "total_return",
                    "annual_return",
                    "max_drawdown",
                    "sharpe_ratio",
                    "number_of_trades",
                    "turnover",
                    "fee_drag",
                    "average_exposure",
                ]
            ]
        ),
        "",
        "## Poor v3 Regimes",
        "",
        _frame_to_markdown(poor_regimes),
        "",
        "## v3 Long Regime Performance",
        "",
        _frame_to_markdown(diagnostics[diagnostics["diagnostic"] == "long_regime_performance"]),
        "",
        "## v3 Short Regime Performance",
        "",
        _frame_to_markdown(diagnostics[diagnostics["diagnostic"] == "short_regime_performance"]),
        "",
        "## v3 Executed Position Distribution",
        "",
        _frame_to_markdown(diagnostics[diagnostics["diagnostic"] == "executed_position_distribution"]),
        "",
        "## v3 Risk Action Counts",
        "",
        _frame_to_markdown(diagnostics[diagnostics["diagnostic"] == "risk_action_counts"]),
        "",
        "## Execution Summary",
        "",
        _frame_to_markdown(diagnostics[diagnostics["diagnostic"] == "execution_summary"]),
        "",
        "## Files",
        "",
        f"- Summary CSV: `{SUMMARY_CSV}`",
        f"- Diagnostics CSV: `{DIAGNOSTICS_CSV}`",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def build_conclusion(v3_summary: pd.DataFrame, v2_summary: pd.DataFrame) -> str:
    v3_avg_return = float(v3_summary["total_return"].mean()) if len(v3_summary) else 0.0
    v3_positive_rate = float((v3_summary["total_return"] > 0.0).mean()) if len(v3_summary) else 0.0
    v3_avg_exposure = float(v3_summary["average_exposure"].mean()) if len(v3_summary) else 0.0
    v3_vs_v2 = compare_v3_to_v2(v3_summary, v2_summary)

    works = (
        "The v3 pipeline works mechanically on ETH: all datasets completed and produced fee-aware positions, returns, and diagnostics."
        if len(v3_summary)
        else "The v3 pipeline did not produce ETH results."
    )
    weakness = (
        "ETH remains weak for v3 if judged by absolute return: the average v3 total return is negative or near flat."
        if v3_avg_return <= 0.01
        else "ETH is not uniformly weak for v3 on this baseline, but the result is still an untuned research check."
    )
    specificity = (
        "Treat v3 as potentially cross-asset in architecture only, not as a proven cross-asset strategy yet."
        if v3_positive_rate < 0.75 or v3_avg_exposure < 0.05
        else "v3 shows some cross-asset promise, but it still needs separate ETH validation before any non-BTC claim."
    )
    return "\n".join(
        [
            f"- Whether v3 works on ETH: {works}",
            f"- Whether ETH remains weak like v2: {weakness} {v3_vs_v2}",
            f"- BTC-specific or cross-asset: {specificity}",
            f"- Average v3 total_return: {v3_avg_return:.6g}; positive-result rate: {v3_positive_rate:.3g}; average exposure: {v3_avg_exposure:.6g}.",
            "- Bad results are not filtered out; the full comparison table and regime diagnostics below include every tested fee and dataset.",
        ]
    )


def compare_v3_to_v2(v3_summary: pd.DataFrame, v2_summary: pd.DataFrame) -> str:
    if v2_summary.empty:
        return "No v2 ETH candidate comparison was available."
    key = ["dataset", "fee_rate"]
    joined = v3_summary.merge(v2_summary, on=key, suffixes=("_v3", "_v2"))
    if joined.empty:
        return "The v2 comparison did not align with the v3 ETH result rows."
    win_rate = float((joined["total_return_v3"] >= joined["total_return_v2"]).mean())
    return f"Against the ETH-applied v2 candidate, v3 total_return win rate is {win_rate:.3g}."


def build_poor_regime_table(diagnostics: pd.DataFrame) -> pd.DataFrame:
    long_perf = diagnostics[diagnostics["diagnostic"] == "long_regime_performance"].copy()
    return_column = "strategy_return_net"
    if long_perf.empty or return_column not in long_perf.columns:
        return pd.DataFrame()
    long_perf[return_column] = pd.to_numeric(long_perf[return_column], errors="coerce").fillna(0.0)
    grouped = (
        long_perf.groupby("regime", dropna=False)
        .agg(
            observations=("dataset", "count"),
            avg_strategy_return_net=(return_column, "mean"),
            worst_strategy_return_net=(return_column, "min"),
            avg_bars=("bars", "mean"),
        )
        .reset_index()
        .sort_values(["avg_strategy_return_net", "worst_strategy_return_net"], ascending=True)
    )
    return grouped[grouped["avg_strategy_return_net"] < 0.0]


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        raise ValueError(f"frame is missing column: {name}")
    return frame[name]


def _equity(frame: pd.DataFrame, returns: pd.Series) -> pd.Series:
    for column in ("equity_curve", "equity_net", "equity"):
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce").fillna(1.0)
    return (1.0 + returns).cumprod()


def _drawdown(frame: pd.DataFrame, equity: pd.Series) -> pd.Series:
    if "drawdown" in frame.columns:
        return pd.to_numeric(frame["drawdown"], errors="coerce").fillna(0.0)
    return equity / equity.cummax() - 1.0


def _position(frame: pd.DataFrame) -> pd.Series:
    for column in ("executed_position", "final_position", "position", "current_exposure"):
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    raise ValueError("frame is missing a position column")


def _trade_amount(frame: pd.DataFrame, position: pd.Series) -> pd.Series:
    for column in ("trade_amount", "trade_size"):
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return position.diff().abs().fillna(position.abs())


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
    summary, diagnostics = run_baseline()
    write_report(summary, diagnostics)
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {DIAGNOSTICS_CSV}")


if __name__ == "__main__":
    main()

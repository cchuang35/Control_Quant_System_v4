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
from src.v3.backtest_v3 import BacktestV3Config, run_v3_backtest
from src.v3.diagnostics import build_v3_diagnostics
from src.v3.position_composer import PositionComposerConfig


FEE_RATES = (0.0005, 0.0010, 0.0020)
PERIODS_PER_YEAR = 365 * 24
REPORT_PATH = Path("reports") / "v3_2_discrete_position_experiment.md"
SUMMARY_CSV = Path("reports") / "v3_2_discrete_position_experiment_summary.csv"
DIAGNOSTICS_CSV = Path("reports") / "v3_2_discrete_position_experiment_diagnostics.csv"
BTC_DATASETS = (
    "btcusdt_1h.csv",
    "btcusdt_1h_365d.csv",
    "btcusdt_1h_2y.csv",
    "btcusdt_1h_3y.csv",
    "btcusdt_1h_5y.csv",
)
ETH_DATASETS = (
    "ethusdt_1h_365d.csv",
    "ethusdt_1h_2y.csv",
    "ethusdt_1h_3y.csv",
    "ethusdt_1h_5y.csv",
)
POSITION_SCHEMES = {
    "A_binary_0_1": (0.0, 1.0),
    "B_conservative_0_025_05_075_1": (0.0, 0.25, 0.50, 0.75, 1.0),
    "C_coarse_0_05_1": (0.0, 0.50, 1.0),
}


def discover_datasets(data_dir: Path = Path("data")) -> dict[str, dict[str, Path]]:
    return {
        "BTC": {
            path.stem: path
            for name in BTC_DATASETS
            for path in [data_dir / name]
            if path.exists()
        },
        "ETH": {
            path.stem: path
            for name in ETH_DATASETS
            for path in [data_dir / name]
            if path.exists()
        },
    }


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame]:
    datasets = discover_datasets()
    if not datasets["BTC"]:
        raise FileNotFoundError("No BTCUSDT 1h datasets found under data/")

    summary_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []

    for asset, asset_datasets in datasets.items():
        if asset == "ETH" and not asset_datasets:
            continue
        for dataset, path in asset_datasets.items():
            print(f"asset={asset} dataset={dataset}")
            data = load_ohlcv_csv(path)
            for fee_rate in FEE_RATES:
                print(f"  fee={fee_rate:g}")
                for scheme_name, allowed_positions in POSITION_SCHEMES.items():
                    print(f"    scheme={scheme_name}")
                    result = run_v3_backtest(
                        data,
                        config=BacktestV3Config(
                            fee_rate=fee_rate,
                            cooldown_bars=120,
                            composer_config=PositionComposerConfig(
                                allowed_positions=allowed_positions,
                                rounding_mode="floor",
                            ),
                        ),
                    )
                    summary_rows.append(
                        {
                            "asset": asset,
                            "dataset": dataset,
                            "fee_rate": fee_rate,
                            "scheme": scheme_name,
                            "allowed_positions": str(list(allowed_positions)),
                            **summarize_frame(result),
                        }
                    )
                    diagnostics = build_v3_diagnostics(result)
                    diagnostic_rows.extend(flatten_diagnostics(asset, dataset, fee_rate, scheme_name, diagnostics))

    return pd.DataFrame(summary_rows), pd.DataFrame(diagnostic_rows)


def summarize_frame(frame: pd.DataFrame) -> dict[str, float | int]:
    returns = pd.to_numeric(frame["strategy_return_net"], errors="coerce").fillna(0.0)
    equity = pd.to_numeric(frame["equity_curve"], errors="coerce").fillna(1.0)
    drawdown = pd.to_numeric(frame["drawdown"], errors="coerce").fillna(0.0)
    position = pd.to_numeric(frame["executed_position"], errors="coerce").fillna(0.0)
    trade_amount = pd.to_numeric(frame["trade_amount"], errors="coerce").fillna(0.0)
    fee_cost = pd.to_numeric(frame["fee_cost"], errors="coerce").fillna(0.0)
    final_equity = float(equity.iloc[-1]) if len(equity) else 1.0
    periods = max(len(frame), 1)
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
        "max_exposure": float(position.abs().max()) if len(position) else 0.0,
    }


def flatten_diagnostics(
    asset: str,
    dataset: str,
    fee_rate: float,
    scheme: str,
    diagnostics: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for diagnostic, frame in {
        "executed_position_distribution": diagnostics["executed_position_distribution"],
        "risk_action_counts": diagnostics["risk_action_counts"],
        "execution_summary": diagnostics["execution_summary"],
    }.items():
        for row in frame.to_dict("records"):
            rows.append(
                {
                    "asset": asset,
                    "dataset": dataset,
                    "fee_rate": fee_rate,
                    "scheme": scheme,
                    "diagnostic": diagnostic,
                    **row,
                }
            )
    return rows


def write_report(summary: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)
    diagnostics.to_csv(DIAGNOSTICS_CSV, index=False)

    btc = summary[summary["asset"] == "BTC"].copy()
    eth = summary[summary["asset"] == "ETH"].copy()
    btc_aggregate = aggregate_by_scheme(btc)
    eth_aggregate = aggregate_by_scheme(eth) if not eth.empty else pd.DataFrame()
    tradeoff = rank_drawdown_sharpe_tradeoff(btc)
    turnover = turnover_comparison(btc)
    binary = binary_comparison(btc)

    lines = [
        "# v3.2 Discrete Position Experiment",
        "",
        "This experiment changes only the v3 position-composer allowed positions. The estimator, long-term controller, short-term controller, cooldown manager, risk supervisor, execution layer, fee rates, and no-leverage rule stay the same.",
        "",
        "## Position Schemes",
        "",
        "- A. Binary: `[0, 1]`",
        "- B. Conservative discrete: `[0, 0.25, 0.5, 0.75, 1.0]`",
        "- C. Coarse discrete: `[0, 0.5, 1.0]`",
        "",
        "All schemes use the current conservative `floor` rounding. This means binary is intentionally strict: a raw target below `1.0` rounds to `0`.",
        "",
        "## Direct Answers",
        "",
        *direct_answers(btc_aggregate, tradeoff, turnover, binary),
        "",
        "## BTC Aggregate By Scheme",
        "",
        _frame_to_markdown(btc_aggregate),
        "",
        "## BTC Drawdown / Sharpe Tradeoff Ranking",
        "",
        _frame_to_markdown(tradeoff),
        "",
        "## BTC Detailed Results",
        "",
        _frame_to_markdown(
            btc[
                [
                    "dataset",
                    "fee_rate",
                    "scheme",
                    "total_return",
                    "annual_return",
                    "max_drawdown",
                    "sharpe_ratio",
                    "number_of_trades",
                    "turnover",
                    "fee_drag",
                    "average_exposure",
                    "max_exposure",
                ]
            ]
        ),
        "",
        "## ETH Validation",
        "",
        "ETH is included only as validation. No ETH-specific tuning was applied.",
        "",
        _frame_to_markdown(eth_aggregate),
        "",
        "## Diagnostics",
        "",
        "### BTC Executed Position Distribution",
        "",
        _frame_to_markdown(
            diagnostics[
                (diagnostics["asset"] == "BTC")
                & (diagnostics["diagnostic"] == "executed_position_distribution")
            ]
        ),
        "",
        "## Files",
        "",
        f"- Summary CSV: `{SUMMARY_CSV}`",
        f"- Diagnostics CSV: `{DIAGNOSTICS_CSV}`",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def aggregate_by_scheme(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    return (
        frame.groupby("scheme", sort=True)
        .agg(
            rows=("scheme", "count"),
            avg_total_return=("total_return", "mean"),
            avg_annual_return=("annual_return", "mean"),
            avg_max_drawdown=("max_drawdown", "mean"),
            worst_max_drawdown=("max_drawdown", "min"),
            avg_sharpe=("sharpe_ratio", "mean"),
            avg_trades=("number_of_trades", "mean"),
            avg_turnover=("turnover", "mean"),
            avg_fee_drag=("fee_drag", "mean"),
            avg_exposure=("average_exposure", "mean"),
            max_exposure=("max_exposure", "max"),
        )
        .reset_index()
    )


def rank_drawdown_sharpe_tradeoff(frame: pd.DataFrame) -> pd.DataFrame:
    aggregate = aggregate_by_scheme(frame)
    if aggregate.empty:
        return aggregate
    ranked = aggregate.copy()
    ranked["sharpe_rank"] = ranked["avg_sharpe"].rank(ascending=False, method="min")
    ranked["drawdown_rank"] = ranked["avg_max_drawdown"].rank(ascending=False, method="min")
    ranked["turnover_rank"] = ranked["avg_turnover"].rank(ascending=True, method="min")
    ranked["tradeoff_score"] = ranked["sharpe_rank"] + ranked["drawdown_rank"] + 0.5 * ranked["turnover_rank"]
    return ranked.sort_values(["tradeoff_score", "sharpe_rank", "drawdown_rank"])


def turnover_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    return aggregate_by_scheme(frame)[["scheme", "avg_turnover", "avg_fee_drag", "avg_trades", "avg_exposure"]]


def binary_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    return aggregate_by_scheme(frame)[["scheme", "avg_total_return", "avg_max_drawdown", "avg_sharpe", "avg_turnover"]]


def direct_answers(
    btc_aggregate: pd.DataFrame,
    tradeoff: pd.DataFrame,
    turnover: pd.DataFrame,
    binary: pd.DataFrame,
) -> list[str]:
    if btc_aggregate.empty:
        return ["- No BTC results were produced."]
    best_scheme = str(tradeoff.iloc[0]["scheme"])
    fine = btc_aggregate[btc_aggregate["scheme"] == "B_conservative_0_025_05_075_1"].iloc[0]
    coarse = btc_aggregate[btc_aggregate["scheme"] == "C_coarse_0_05_1"].iloc[0]
    binary_row = btc_aggregate[btc_aggregate["scheme"] == "A_binary_0_1"].iloc[0]
    best_sharpe_scheme = str(btc_aggregate.sort_values("avg_sharpe", ascending=False).iloc[0]["scheme"])
    best_drawdown_scheme = str(btc_aggregate.sort_values("avg_max_drawdown", ascending=False).iloc[0]["scheme"])
    fine_turnover_delta = float(fine["avg_turnover"] - coarse["avg_turnover"])
    binary_best = bool(
        binary_row["avg_sharpe"] >= btc_aggregate["avg_sharpe"].max()
        or binary_row["avg_max_drawdown"] >= btc_aggregate["avg_max_drawdown"].max()
    )
    return [
        f"- Best drawdown/Sharpe tradeoff on BTC: `{best_scheme}` by the simple rank score.",
        f"- Best average Sharpe: `{best_sharpe_scheme}`; best average max drawdown: `{best_drawdown_scheme}`.",
        f"- Finer discrete positions turnover check: conservative discrete average turnover minus coarse average turnover = `{fine_turnover_delta:.6g}`.",
        "- Finer positions do not create extra turnover by themselves in this setup; the execution layer and controller target changes are still the main turnover constraints."
        if fine_turnover_delta <= 0.0
        else "- Finer positions increased turnover in this setup, so any Sharpe/drawdown benefit must be judged net of fee drag.",
        "- Binary still performs better on at least one risk-adjusted dimension."
        if binary_best
        else "- Binary did not outperform on the main drawdown/Sharpe dimensions; with floor rounding it was mostly too restrictive rather than simply lower-turnover.",
    ]


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
    summary, diagnostics = run_experiment()
    write_report(summary, diagnostics)
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {DIAGNOSTICS_CSV}")


if __name__ == "__main__":
    main()

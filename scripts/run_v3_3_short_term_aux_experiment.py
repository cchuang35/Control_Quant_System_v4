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
from src.v3.short_term_controller import ShortTermControllerConfig


FEE_RATES = (0.0005, 0.0010, 0.0020)
PERIODS_PER_YEAR = 365 * 24
REPORT_PATH = Path("reports") / "v3_3_short_term_aux_experiment.md"
SUMMARY_CSV = Path("reports") / "v3_3_short_term_aux_experiment_summary.csv"
REGIME_CSV = Path("reports") / "v3_3_short_term_aux_experiment_regime_performance.csv"
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


VARIANTS = {
    "A_long_term_only": ShortTermControllerConfig(
        enable_pullback_add=False,
        enable_recovery_add=False,
        enable_overheat_reduce=False,
        enable_breakdown_reduce=False,
        allow_neutral_recovery_add=False,
    ),
    "B_pullback_add_only": ShortTermControllerConfig(
        enable_pullback_add=True,
        enable_recovery_add=False,
        enable_overheat_reduce=False,
        enable_breakdown_reduce=False,
        allow_neutral_recovery_add=False,
    ),
    "C_overheat_breakdown_reduce_only": ShortTermControllerConfig(
        enable_pullback_add=False,
        enable_recovery_add=False,
        enable_overheat_reduce=True,
        enable_breakdown_reduce=True,
        allow_neutral_recovery_add=False,
    ),
    "D_full_short_term_aux": ShortTermControllerConfig(),
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
    regime_rows: list[dict[str, Any]] = []

    for asset, asset_datasets in datasets.items():
        if asset == "ETH" and not asset_datasets:
            continue
        for dataset, path in asset_datasets.items():
            print(f"asset={asset} dataset={dataset}")
            data = load_ohlcv_csv(path)
            for fee_rate in FEE_RATES:
                print(f"  fee={fee_rate:g}")
                for variant, short_config in VARIANTS.items():
                    print(f"    variant={variant}")
                    result = run_v3_backtest(
                        data,
                        config=BacktestV3Config(
                            fee_rate=fee_rate,
                            cooldown_bars=120,
                            short_term_config=short_config,
                        ),
                    )
                    summary_rows.append(
                        {
                            "asset": asset,
                            "dataset": dataset,
                            "fee_rate": fee_rate,
                            "variant": variant,
                            **summarize_frame(result),
                        }
                    )
                    diagnostics = build_v3_diagnostics(result)
                    regime_rows.extend(flatten_regime_performance(asset, dataset, fee_rate, variant, diagnostics))

    return pd.DataFrame(summary_rows), pd.DataFrame(regime_rows)


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
        "annual_return": final_equity ** (PERIODS_PER_YEAR / periods) - 1.0,
        "total_return": final_equity - 1.0,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "sharpe_ratio": 0.0 if return_std == 0.0 else float(returns.mean() / return_std * np.sqrt(PERIODS_PER_YEAR)),
        "turnover": float(trade_amount.sum()),
        "fee_drag": float(fee_cost.sum()),
        "number_of_trades": int((trade_amount > 0.0).sum()),
        "average_exposure": float(position.abs().mean()) if len(position) else 0.0,
    }


def flatten_regime_performance(
    asset: str,
    dataset: str,
    fee_rate: float,
    variant: str,
    diagnostics: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for diagnostic_name in ("long_regime_performance", "short_regime_performance"):
        frame = diagnostics[diagnostic_name]
        for row in frame.to_dict("records"):
            rows.append(
                {
                    "asset": asset,
                    "dataset": dataset,
                    "fee_rate": fee_rate,
                    "variant": variant,
                    "diagnostic": diagnostic_name,
                    **row,
                }
            )
    return rows


def write_report(summary: pd.DataFrame, regimes: pd.DataFrame) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)
    regimes.to_csv(REGIME_CSV, index=False)

    btc = summary[summary["asset"] == "BTC"].copy()
    eth = summary[summary["asset"] == "ETH"].copy()
    btc_aggregate = aggregate_by_variant(btc)
    eth_aggregate = aggregate_by_variant(eth) if not eth.empty else pd.DataFrame()
    btc_short_regimes = aggregate_regime_performance(regimes, "BTC", "short_regime_performance")
    answers = direct_answers(btc_aggregate)

    lines = [
        "# v3.3 Short-Term Auxiliary Controller Experiment",
        "",
        "This experiment keeps the v3 estimator, long-term controller, Risk Supervisor, Position Composer, Execution Layer, fee model, and cooldown manager the same. Only short-term auxiliary rule switches change.",
        "",
        "## Variants",
        "",
        "- A. `A_long_term_only`: all short-term adjustments disabled.",
        "- B. `B_pullback_add_only`: only bullish pullback add enabled.",
        "- C. `C_overheat_breakdown_reduce_only`: only overheat/breakdown reductions enabled.",
        "- D. `D_full_short_term_aux`: current full auxiliary controller.",
        "",
        "## Direct Answers",
        "",
        *answers,
        "",
        "## BTC Aggregate By Variant",
        "",
        _frame_to_markdown(btc_aggregate),
        "",
        "## BTC Regime-Specific Performance",
        "",
        _frame_to_markdown(btc_short_regimes),
        "",
        "## BTC Detailed Results",
        "",
        _frame_to_markdown(
            btc[
                [
                    "dataset",
                    "fee_rate",
                    "variant",
                    "annual_return",
                    "max_drawdown",
                    "sharpe_ratio",
                    "turnover",
                    "fee_drag",
                    "number_of_trades",
                    "average_exposure",
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
        "## Files",
        "",
        f"- Summary CSV: `{SUMMARY_CSV}`",
        f"- Regime performance CSV: `{REGIME_CSV}`",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def aggregate_by_variant(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    return (
        frame.groupby("variant", sort=True)
        .agg(
            rows=("variant", "count"),
            avg_annual_return=("annual_return", "mean"),
            avg_total_return=("total_return", "mean"),
            avg_max_drawdown=("max_drawdown", "mean"),
            worst_max_drawdown=("max_drawdown", "min"),
            avg_sharpe=("sharpe_ratio", "mean"),
            avg_turnover=("turnover", "mean"),
            avg_fee_drag=("fee_drag", "mean"),
            avg_trades=("number_of_trades", "mean"),
            avg_exposure=("average_exposure", "mean"),
        )
        .reset_index()
    )


def aggregate_regime_performance(regimes: pd.DataFrame, asset: str, diagnostic: str) -> pd.DataFrame:
    local = regimes[(regimes["asset"] == asset) & (regimes["diagnostic"] == diagnostic)].copy()
    if local.empty:
        return pd.DataFrame()
    for column in ["strategy_return_net", "trade_count", "turnover", "fees", "avg_exposure", "bars"]:
        local[column] = pd.to_numeric(local[column], errors="coerce").fillna(0.0)
    return (
        local.groupby(["variant", "regime"], sort=True)
        .agg(
            observations=("dataset", "count"),
            avg_strategy_return_net=("strategy_return_net", "mean"),
            total_strategy_return_net=("strategy_return_net", "sum"),
            avg_trade_count=("trade_count", "mean"),
            avg_turnover=("turnover", "mean"),
            avg_fees=("fees", "mean"),
            avg_exposure=("avg_exposure", "mean"),
            avg_bars=("bars", "mean"),
        )
        .reset_index()
        .sort_values(["variant", "avg_strategy_return_net"], ascending=[True, False])
    )


def direct_answers(btc_aggregate: pd.DataFrame) -> list[str]:
    if btc_aggregate.empty:
        return ["- No BTC results were produced."]

    indexed = btc_aggregate.set_index("variant")
    long_only = indexed.loc["A_long_term_only"]
    pullback = indexed.loc["B_pullback_add_only"]
    reduce = indexed.loc["C_overheat_breakdown_reduce_only"]
    full = indexed.loc["D_full_short_term_aux"]

    pullback_return_change = float(pullback["avg_annual_return"] - long_only["avg_annual_return"])
    pullback_dd_change = float(pullback["avg_max_drawdown"] - long_only["avg_max_drawdown"])
    reduce_dd_change = float(reduce["avg_max_drawdown"] - long_only["avg_max_drawdown"])
    full_turnover_change = float(full["avg_turnover"] - long_only["avg_turnover"])

    best = btc_aggregate.copy()
    best["sharpe_rank"] = best["avg_sharpe"].rank(ascending=False, method="min")
    best["drawdown_rank"] = best["avg_max_drawdown"].rank(ascending=False, method="min")
    best["turnover_rank"] = best["avg_turnover"].rank(ascending=True, method="min")
    best["score"] = best["sharpe_rank"] + best["drawdown_rank"] + 0.5 * best["turnover_rank"]
    best_variant = str(best.sort_values(["score", "sharpe_rank"]).iloc[0]["variant"])

    keep = best_variant
    if pullback_return_change > 0.0 and pullback_dd_change >= -0.002:
        pullback_answer = "Pullback add improved average annual return without a large drawdown penalty."
    elif pullback_return_change > 0.0:
        pullback_answer = "Pullback add improved return but paid for it with worse drawdown."
    else:
        pullback_answer = "Pullback add did not improve returns on average."

    reduce_answer = (
        "Overheat/breakdown reduction reduced average drawdown versus long-term-only."
        if reduce_dd_change > 0.0
        else "Overheat/breakdown reduction did not reduce average drawdown versus long-term-only."
    )
    full_answer = (
        "Full short-term controller appears to overtrade versus long-term-only."
        if full_turnover_change > 1.0
        else "Full short-term controller does not materially overtrade versus long-term-only."
    )

    return [
        f"- Does short-term pullback add improve returns or just increase risk? {pullback_answer} annual_return_delta={pullback_return_change:.6g}, max_drawdown_delta={pullback_dd_change:.6g}.",
        f"- Does overheat/breakdown reduction reduce drawdown? {reduce_answer} max_drawdown_delta={reduce_dd_change:.6g}.",
        f"- Does the full short-term controller overtrade? {full_answer} turnover_delta={full_turnover_change:.6g}.",
        f"- Rule candidate for `v3.final_candidate`: `{keep}` by the simple Sharpe/drawdown/turnover score.",
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
    summary, regimes = run_experiment()
    write_report(summary, regimes)
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {REGIME_CSV}")


if __name__ == "__main__":
    main()

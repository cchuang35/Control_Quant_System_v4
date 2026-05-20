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
from src.v3.backtest_v3 import BacktestV3Config, ConditionalLeverageConfigV3, run_v3_backtest


FEE_RATES = (0.0005, 0.0010, 0.0020)
PERIODS_PER_YEAR = 365 * 24
REPORT_PATH = Path("reports") / "v3_5_conditional_leverage_experiment.md"
SUMMARY_CSV = Path("reports") / "v3_5_conditional_leverage_experiment_summary.csv"
WORST_TRADES_CSV = Path("reports") / "v3_5_conditional_leverage_worst_trades.csv"
REASONS_CSV = Path("reports") / "v3_5_conditional_leverage_reasons.csv"
BTC_DATASETS = (
    "btcusdt_1h.csv",
    "btcusdt_1h_365d.csv",
    "btcusdt_1h_2y.csv",
    "btcusdt_1h_3y.csv",
    "btcusdt_1h_5y.csv",
)
VARIANTS = {
    "A_no_leverage_max_1": None,
    "B_conditional_leverage_max_1_25": ConditionalLeverageConfigV3(
        enabled=True,
        max_position=1.25,
        leverage_increment=0.25,
        high_confidence=0.70,
        max_drawdown_for_leverage=-0.05,
        require_raw_target_at_least=1.0,
    ),
}


def discover_btc_datasets(data_dir: Path = Path("data")) -> dict[str, Path]:
    return {
        path.stem: path
        for name in BTC_DATASETS
        for path in [data_dir / name]
        if path.exists()
    }


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    datasets = discover_btc_datasets()
    if not datasets:
        raise FileNotFoundError("No BTCUSDT 1h datasets found under data/")

    summary_rows: list[dict[str, Any]] = []
    worst_rows: list[dict[str, Any]] = []
    reason_rows: list[dict[str, Any]] = []

    for dataset, path in datasets.items():
        print(f"dataset={dataset}")
        data = load_ohlcv_csv(path)
        for fee_rate in FEE_RATES:
            print(f"  fee={fee_rate:g}")
            for variant, leverage_config in VARIANTS.items():
                print(f"    variant={variant}")
                result = run_v3_backtest(
                    data,
                    config=BacktestV3Config(
                        fee_rate=fee_rate,
                        cooldown_bars=120,
                        leverage_config=leverage_config,
                    ),
                )
                summary_rows.append(
                    {
                        "dataset": dataset,
                        "fee_rate": fee_rate,
                        "variant": variant,
                        **summarize_frame(result),
                    }
                )
                worst_rows.append(
                    {
                        "dataset": dataset,
                        "fee_rate": fee_rate,
                        "variant": variant,
                        **worst_leverage_trade(result),
                    }
                )
                reason_rows.extend(leverage_reason_counts(dataset, fee_rate, variant, result))

    return pd.DataFrame(summary_rows), pd.DataFrame(worst_rows), pd.DataFrame(reason_rows)


def summarize_frame(frame: pd.DataFrame) -> dict[str, float | int]:
    returns = pd.to_numeric(frame["strategy_return_net"], errors="coerce").fillna(0.0)
    equity = pd.to_numeric(frame["equity_curve"], errors="coerce").fillna(1.0)
    drawdown = pd.to_numeric(frame["drawdown"], errors="coerce").fillna(0.0)
    position = pd.to_numeric(frame["executed_position"], errors="coerce").fillna(0.0)
    trade_amount = pd.to_numeric(frame["trade_amount"], errors="coerce").fillna(0.0)
    fee_cost = pd.to_numeric(frame["fee_cost"], errors="coerce").fillna(0.0)
    leverage_used = frame.get("leverage_used", pd.Series(False, index=frame.index)).astype(bool)
    leverage_position = position.where(position > 1.0, 0.0)
    final_equity = float(equity.iloc[-1]) if len(equity) else 1.0
    periods = max(len(frame), 1)
    return_std = float(returns.std(ddof=0))
    leverage_entries = leverage_used.astype(int).diff().fillna(leverage_used.astype(int)).clip(lower=0)
    return {
        "annual_return": final_equity ** (PERIODS_PER_YEAR / periods) - 1.0,
        "total_return": final_equity - 1.0,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "sharpe_ratio": 0.0 if return_std == 0.0 else float(returns.mean() / return_std * np.sqrt(PERIODS_PER_YEAR)),
        "turnover": float(trade_amount.sum()),
        "fee_drag": float(fee_cost.sum()),
        "number_of_trades": int((trade_amount > 0.0).sum()),
        "average_exposure": float(position.abs().mean()) if len(position) else 0.0,
        "max_exposure": float(position.abs().max()) if len(position) else 0.0,
        "leverage_usage_frequency": float(leverage_used.mean()) if len(leverage_used) else 0.0,
        "leverage_bars": int(leverage_used.sum()),
        "leverage_entries": int(leverage_entries.sum()),
        "average_leverage_exposure": float(leverage_position[leverage_position > 0.0].mean()) if (leverage_position > 0.0).any() else 0.0,
    }


def worst_leverage_trade(frame: pd.DataFrame) -> dict[str, Any]:
    leverage_used = frame.get("leverage_used", pd.Series(False, index=frame.index)).astype(bool).reset_index(drop=True)
    if not leverage_used.any():
        return {
            "has_leverage_trade": False,
            "start_time": "",
            "end_time": "",
            "bars": 0,
            "net_return": 0.0,
            "min_bar_return": 0.0,
            "max_position": 0.0,
            "fees": 0.0,
        }

    temp = frame.reset_index(drop=True).copy()
    groups: list[pd.DataFrame] = []
    start: int | None = None
    for idx, active in enumerate(leverage_used):
        if active and start is None:
            start = idx
        elif not active and start is not None:
            groups.append(temp.iloc[start:idx].copy())
            start = None
    if start is not None:
        groups.append(temp.iloc[start:].copy())

    worst = min(groups, key=lambda group: float(pd.to_numeric(group["strategy_return_net"], errors="coerce").fillna(0.0).sum()))
    returns = pd.to_numeric(worst["strategy_return_net"], errors="coerce").fillna(0.0)
    return {
        "has_leverage_trade": True,
        "start_time": worst["timestamp"].iloc[0],
        "end_time": worst["timestamp"].iloc[-1],
        "bars": int(len(worst)),
        "net_return": float(returns.sum()),
        "min_bar_return": float(returns.min()),
        "max_position": float(pd.to_numeric(worst["executed_position"], errors="coerce").fillna(0.0).max()),
        "fees": float(pd.to_numeric(worst["fee_cost"], errors="coerce").fillna(0.0).sum()),
    }


def leverage_reason_counts(dataset: str, fee_rate: float, variant: str, frame: pd.DataFrame) -> list[dict[str, Any]]:
    reasons = frame.get("leverage_reason", pd.Series("missing", index=frame.index)).astype(str)
    counts = reasons.value_counts(dropna=False).sort_index()
    total = max(len(frame), 1)
    return [
        {
            "dataset": dataset,
            "fee_rate": fee_rate,
            "variant": variant,
            "leverage_reason": reason,
            "bars": int(count),
            "ratio": float(count / total),
        }
        for reason, count in counts.items()
    ]


def write_report(summary: pd.DataFrame, worst_trades: pd.DataFrame, reasons: pd.DataFrame) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)
    worst_trades.to_csv(WORST_TRADES_CSV, index=False)
    reasons.to_csv(REASONS_CSV, index=False)

    aggregate = aggregate_by_variant(summary)
    comparison = compare_variants(summary)
    worst_leverage = worst_trades[
        (worst_trades["variant"] == "B_conditional_leverage_max_1_25")
        & (worst_trades["has_leverage_trade"].astype(bool))
    ].copy()
    if not worst_leverage.empty:
        worst_leverage["net_return"] = pd.to_numeric(worst_leverage["net_return"], errors="coerce").fillna(0.0)
        worst_leverage = worst_leverage.sort_values("net_return").head(10)

    lines = [
        "# v3.5 Conditional Leverage Experiment",
        "",
        "Leverage remains disabled by default in v3. This is an optional experiment that tests `max_position = 1.25` only under strict strong-bull conditions.",
        "",
        "## Strict Leverage Conditions",
        "",
        "- `long_regime == strong_bull`",
        "- `confidence_score >= 0.70`",
        "- `volatility_state` is `low` or `normal`",
        "- `portfolio_drawdown > -5%`",
        "- no active weak_bull/bull-like cooldown",
        "- no recent consecutive losses",
        "- normal risk action and raw controller target at least `1.0`",
        "",
        "## Direct Answer",
        "",
        *direct_answer(aggregate, comparison),
        "",
        "## Aggregate By Variant",
        "",
        _frame_to_markdown(aggregate),
        "",
        "## Conditional Leverage Impact Versus No Leverage",
        "",
        _frame_to_markdown(comparison),
        "",
        "## Detailed Results",
        "",
        _frame_to_markdown(
            summary[
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
                    "max_exposure",
                    "leverage_usage_frequency",
                    "leverage_bars",
                    "leverage_entries",
                ]
            ]
        ),
        "",
        "## Worst Leverage Trades",
        "",
        _frame_to_markdown(worst_leverage),
        "",
        "## Leverage Condition Diagnostics",
        "",
        _frame_to_markdown(
            aggregate_reason_counts(reasons[reasons["variant"] == "B_conditional_leverage_max_1_25"])
        ),
        "",
        "## Files",
        "",
        f"- Summary CSV: `{SUMMARY_CSV}`",
        f"- Worst leverage trades CSV: `{WORST_TRADES_CSV}`",
        f"- Leverage reason diagnostics CSV: `{REASONS_CSV}`",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def aggregate_by_variant(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby("variant", sort=True)
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
            max_exposure=("max_exposure", "max"),
            avg_leverage_usage=("leverage_usage_frequency", "mean"),
            total_leverage_bars=("leverage_bars", "sum"),
            total_leverage_entries=("leverage_entries", "sum"),
        )
        .reset_index()
    )


def compare_variants(summary: pd.DataFrame) -> pd.DataFrame:
    key = ["dataset", "fee_rate"]
    base = summary[summary["variant"] == "A_no_leverage_max_1"][key + [
        "annual_return",
        "max_drawdown",
        "sharpe_ratio",
        "turnover",
        "fee_drag",
        "average_exposure",
    ]]
    leveraged = summary[summary["variant"] == "B_conditional_leverage_max_1_25"][key + [
        "annual_return",
        "max_drawdown",
        "sharpe_ratio",
        "turnover",
        "fee_drag",
        "average_exposure",
        "leverage_usage_frequency",
        "leverage_bars",
    ]]
    joined = base.merge(leveraged, on=key, suffixes=("_no_leverage", "_conditional_leverage"))
    for metric in ["annual_return", "max_drawdown", "sharpe_ratio", "turnover", "fee_drag", "average_exposure"]:
        joined[f"{metric}_delta"] = joined[f"{metric}_conditional_leverage"] - joined[f"{metric}_no_leverage"]
    return joined


def aggregate_reason_counts(reasons: pd.DataFrame) -> pd.DataFrame:
    if reasons.empty:
        return pd.DataFrame()
    return (
        reasons.groupby("leverage_reason", sort=True)
        .agg(
            observations=("dataset", "count"),
            total_bars=("bars", "sum"),
            avg_ratio=("ratio", "mean"),
        )
        .reset_index()
        .sort_values(["total_bars", "leverage_reason"], ascending=[False, True])
    )


def direct_answer(aggregate: pd.DataFrame, comparison: pd.DataFrame) -> list[str]:
    indexed = aggregate.set_index("variant")
    base = indexed.loc["A_no_leverage_max_1"]
    leveraged = indexed.loc["B_conditional_leverage_max_1_25"]
    annual_delta = float(leveraged["avg_annual_return"] - base["avg_annual_return"])
    drawdown_delta = float(leveraged["avg_max_drawdown"] - base["avg_max_drawdown"])
    sharpe_delta = float(leveraged["avg_sharpe"] - base["avg_sharpe"])
    usage = float(leveraged["avg_leverage_usage"])
    total_entries = int(leveraged["total_leverage_entries"])
    recommendation = (
        "remain experimental"
        if usage == 0.0 or drawdown_delta < -0.002 or sharpe_delta <= 0.0
        else "be considered only as a later candidate branch"
    )
    return [
        f"- Conditional leverage average annual_return delta: `{annual_delta:.6g}`.",
        f"- Conditional leverage average max_drawdown delta: `{drawdown_delta:.6g}`.",
        f"- Conditional leverage average Sharpe delta: `{sharpe_delta:.6g}`.",
        f"- Leverage usage frequency averaged `{usage:.6g}` with `{total_entries}` total leverage entries.",
        f"- Recommendation: leverage should `{recommendation}`, not be included in v3 default.",
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
    summary, worst_trades, reasons = run_experiment()
    write_report(summary, worst_trades, reasons)
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {WORST_TRADES_CSV}")
    print(f"Wrote {REASONS_CSV}")


if __name__ == "__main__":
    main()

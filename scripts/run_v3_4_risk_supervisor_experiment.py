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
from src.v3.risk_supervisor import RiskSupervisorConfig


FEE_RATES = (0.0005, 0.0010, 0.0020)
PERIODS_PER_YEAR = 365 * 24
REPORT_PATH = Path("reports") / "v3_4_risk_supervisor_experiment.md"
SUMMARY_CSV = Path("reports") / "v3_4_risk_supervisor_experiment_summary.csv"
DIAGNOSTICS_CSV = Path("reports") / "v3_4_risk_supervisor_experiment_diagnostics.csv"
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
    "A_no_risk_supervisor": RiskSupervisorConfig(
        enable_drawdown_cap=False,
        enable_volatility_cap=False,
        enable_consecutive_loss_rules=False,
        enable_market_risk_state=False,
        enable_cost_guards=False,
    ),
    "B_drawdown_cap_only": RiskSupervisorConfig(
        enable_drawdown_cap=True,
        enable_volatility_cap=False,
        enable_consecutive_loss_rules=False,
        enable_market_risk_state=False,
        enable_cost_guards=False,
    ),
    "C_volatility_cap_only": RiskSupervisorConfig(
        enable_drawdown_cap=False,
        enable_volatility_cap=True,
        enable_consecutive_loss_rules=False,
        enable_market_risk_state=False,
        enable_cost_guards=False,
    ),
    "D_drawdown_plus_volatility": RiskSupervisorConfig(
        enable_drawdown_cap=True,
        enable_volatility_cap=True,
        enable_consecutive_loss_rules=False,
        enable_market_risk_state=False,
        enable_cost_guards=False,
    ),
    "E_full_risk_supervisor": RiskSupervisorConfig(),
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
                for variant, risk_config in VARIANTS.items():
                    print(f"    variant={variant}")
                    result = run_v3_backtest(
                        data,
                        config=BacktestV3Config(
                            fee_rate=fee_rate,
                            cooldown_bars=120,
                            risk_config=risk_config,
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
                    diagnostic_rows.extend(flatten_diagnostics(asset, dataset, fee_rate, variant, diagnostics))

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
        "annual_return": final_equity ** (PERIODS_PER_YEAR / periods) - 1.0,
        "total_return": final_equity - 1.0,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "sharpe_ratio": 0.0 if return_std == 0.0 else float(returns.mean() / return_std * np.sqrt(PERIODS_PER_YEAR)),
        "turnover": float(trade_amount.sum()),
        "fee_drag": float(fee_cost.sum()),
        "number_of_trades": int((trade_amount > 0.0).sum()),
        "average_exposure": float(position.abs().mean()) if len(position) else 0.0,
    }


def flatten_diagnostics(
    asset: str,
    dataset: str,
    fee_rate: float,
    variant: str,
    diagnostics: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for diagnostic_name in ("risk_action_counts", "risk_cap_distribution", "executed_position_distribution"):
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


def write_report(summary: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)
    diagnostics.to_csv(DIAGNOSTICS_CSV, index=False)

    btc = summary[summary["asset"] == "BTC"].copy()
    eth = summary[summary["asset"] == "ETH"].copy()
    btc_aggregate = aggregate_by_variant(btc)
    eth_aggregate = aggregate_by_variant(eth) if not eth.empty else pd.DataFrame()
    impacts = impact_vs_no_risk(btc_aggregate)
    answers = direct_answers(btc_aggregate, impacts)

    lines = [
        "# v3.4 Risk Supervisor Experiment",
        "",
        "This experiment keeps the v3 estimator, long-term controller, short-term controller, Position Composer, Execution Layer, cooldown manager, fee model, and discrete positions fixed. Only Risk Supervisor rules are enabled or disabled.",
        "",
        "## Variants",
        "",
        "- A. `A_no_risk_supervisor`: controller target only, risk cap/action disabled.",
        "- B. `B_drawdown_cap_only`: drawdown caps only.",
        "- C. `C_volatility_cap_only`: volatility caps only.",
        "- D. `D_drawdown_plus_volatility`: drawdown and volatility caps.",
        "- E. `E_full_risk_supervisor`: drawdown, volatility, market risk state, and consecutive loss rules.",
        "",
        "## Direct Answers",
        "",
        *answers,
        "",
        "## BTC Aggregate By Variant",
        "",
        _frame_to_markdown(btc_aggregate),
        "",
        "## BTC Impact Versus No Risk Supervisor",
        "",
        _frame_to_markdown(impacts),
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
        "## BTC Risk Diagnostics",
        "",
        _frame_to_markdown(
            diagnostics[
                (diagnostics["asset"] == "BTC")
                & (diagnostics["diagnostic"].isin(["risk_action_counts", "risk_cap_distribution"]))
            ]
        ),
        "",
        "## Files",
        "",
        f"- Summary CSV: `{SUMMARY_CSV}`",
        f"- Diagnostics CSV: `{DIAGNOSTICS_CSV}`",
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


def impact_vs_no_risk(aggregate: pd.DataFrame) -> pd.DataFrame:
    if aggregate.empty:
        return pd.DataFrame()
    base = aggregate[aggregate["variant"] == "A_no_risk_supervisor"].iloc[0]
    rows = []
    for row in aggregate.to_dict("records"):
        rows.append(
            {
                "variant": row["variant"],
                "annual_return_delta": row["avg_annual_return"] - base["avg_annual_return"],
                "max_drawdown_delta": row["avg_max_drawdown"] - base["avg_max_drawdown"],
                "sharpe_delta": row["avg_sharpe"] - base["avg_sharpe"],
                "turnover_delta": row["avg_turnover"] - base["avg_turnover"],
                "fee_drag_delta": row["avg_fee_drag"] - base["avg_fee_drag"],
                "exposure_delta": row["avg_exposure"] - base["avg_exposure"],
            }
        )
    return pd.DataFrame(rows)


def direct_answers(aggregate: pd.DataFrame, impacts: pd.DataFrame) -> list[str]:
    if aggregate.empty:
        return ["- No BTC results were produced."]
    best_drawdown = aggregate.sort_values("avg_max_drawdown", ascending=False).iloc[0]
    best_sharpe = aggregate.sort_values("avg_sharpe", ascending=False).iloc[0]
    best_annual = aggregate.sort_values("avg_annual_return", ascending=False).iloc[0]

    ranked = aggregate.copy()
    ranked["sharpe_rank"] = ranked["avg_sharpe"].rank(ascending=False, method="min")
    ranked["drawdown_rank"] = ranked["avg_max_drawdown"].rank(ascending=False, method="min")
    ranked["annual_rank"] = ranked["avg_annual_return"].rank(ascending=False, method="min")
    ranked["turnover_rank"] = ranked["avg_turnover"].rank(ascending=True, method="min")
    ranked["score"] = ranked["sharpe_rank"] + ranked["drawdown_rank"] + ranked["annual_rank"] + 0.5 * ranked["turnover_rank"]
    useful = ranked.sort_values(["score", "sharpe_rank"]).iloc[0]
    single_rule_rows = aggregate[aggregate["variant"].isin(["B_drawdown_cap_only", "C_volatility_cap_only"])]
    best_single = single_rule_rows.sort_values(["avg_sharpe", "avg_max_drawdown"], ascending=False).iloc[0]
    drawdown_row = aggregate[aggregate["variant"] == "B_drawdown_cap_only"].iloc[0]
    volatility_row = aggregate[aggregate["variant"] == "C_volatility_cap_only"].iloc[0]

    base = aggregate[aggregate["variant"] == "A_no_risk_supervisor"].iloc[0]
    conservative = aggregate[
        (aggregate["avg_exposure"] < base["avg_exposure"] * 0.75)
        & (aggregate["avg_annual_return"] < base["avg_annual_return"])
    ]
    too_conservative = (
        "No rule is obviously too conservative on BTC by the exposure/annual-return check."
        if conservative.empty
        else f"`{conservative.iloc[0]['variant']}` looks potentially too conservative."
    )

    return [
        f"- Impact on max drawdown: best average max drawdown is `{best_drawdown['variant']}` at {best_drawdown['avg_max_drawdown']:.6g}.",
        f"- Impact on Sharpe: best average Sharpe is `{best_sharpe['variant']}` at {best_sharpe['avg_sharpe']:.6g}.",
        f"- Impact on annual return: best average annual return is `{best_annual['variant']}` at {best_annual['avg_annual_return']:.6g}.",
        f"- Impact on turnover: lowest average turnover is `{aggregate.sort_values('avg_turnover').iloc[0]['variant']}`.",
        f"- Most useful single risk rule: `{best_single['variant']}`. Drawdown-only Sharpe={drawdown_row['avg_sharpe']:.6g}; volatility-only Sharpe={volatility_row['avg_sharpe']:.6g}.",
        f"- Most useful risk rule set by combined score: `{useful['variant']}`.",
        f"- Too conservative check: {too_conservative}",
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

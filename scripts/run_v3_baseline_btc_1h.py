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
from v2_small_cap import backtest_v2_btc_final_candidate_a


FEE_RATES = (0.0005, 0.0010, 0.0020)
PERIODS_PER_YEAR = 365 * 24
V1_ENTRY_THRESHOLD = 0.10
REPORT_PATH = Path("reports") / "v3_baseline_btc_1h.md"
SUMMARY_CSV = Path("reports") / "v3_baseline_btc_1h_summary.csv"
DIAGNOSTICS_CSV = Path("reports") / "v3_baseline_btc_1h_diagnostics.csv"
BTC_DATASETS = (
    "btcusdt_1h.csv",
    "btcusdt_1h_365d.csv",
    "btcusdt_1h_2y.csv",
    "btcusdt_1h_3y.csv",
    "btcusdt_1h_5y.csv",
)


def discover_btc_datasets(data_dir: Path = Path("data")) -> dict[str, Path]:
    return {
        path.stem: path
        for name in BTC_DATASETS
        for path in [data_dir / name]
        if path.exists()
    }


def run_baseline() -> tuple[pd.DataFrame, pd.DataFrame]:
    datasets = discover_btc_datasets()
    if not datasets:
        raise FileNotFoundError("No BTCUSDT 1h datasets found under data/")

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
            v1_frame = v1_frame_from_result(v1, fee_rate)
            v2_input = input_frame_from_v1(v1)

            frames = {
                "v1.final": v1_frame,
                "v2.btc_final_candidate_A": backtest_v2_btc_final_candidate_a(
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
                row = {
                    "dataset": dataset,
                    "fee_rate": fee_rate,
                    "version": version,
                    **summarize_frame(frame),
                    "runtime_sec": v1_runtime if version == "v1.final" else 0.0,
                }
                summary_rows.append(row)

            v3_diagnostics = build_v3_diagnostics(frames["v3.baseline"])
            diagnostic_rows.extend(flatten_v3_diagnostics(dataset, fee_rate, v3_diagnostics))

    summary = pd.DataFrame(summary_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    return summary, diagnostics


def v1_frame_from_result(result: Any, fee_rate: float) -> pd.DataFrame:
    exposure = result.exposure_history.reset_index(drop=True).copy()
    equity = result.equity_curve.iloc[1:].reset_index(drop=True).copy()
    current_exposure = exposure["current_exposure"].astype(float)
    frame = pd.DataFrame(
        {
            "timestamp": exposure["execution_timestamp"],
            "close": exposure["close"].astype(float),
            "position": current_exposure,
            "trade_amount": exposure["safe_exposure_change"].abs().astype(float),
            "fee_cost": exposure["safe_exposure_change"].abs().astype(float) * fee_rate,
            "strategy_return_net": equity["period_return"].astype(float),
            "equity_curve": equity["equity"].astype(float),
            "drawdown": equity["drawdown"].astype(float),
        }
    )
    frame["asset_return"] = frame["close"].pct_change().fillna(0.0)
    return frame


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
    mappings = {
        "long_regime_distribution": diagnostics["long_regime_performance"].rename(columns={"regime": "bucket", "bars": "value"}),
        "executed_position_distribution": diagnostics["executed_position_distribution"].rename(columns={"executed_position": "bucket", "bars": "value"}),
        "risk_action_counts": diagnostics["risk_action_counts"].rename(columns={"risk_action": "bucket", "bars": "value"}),
    }
    for diagnostic, frame in mappings.items():
        for row in frame.to_dict("records"):
            rows.append(
                {
                    "dataset": dataset,
                    "fee_rate": fee_rate,
                    "diagnostic": diagnostic,
                    "bucket": row.get("bucket"),
                    "value": row.get("value"),
                    "ratio": row.get("ratio", np.nan),
                }
            )
    execution_summary = diagnostics["execution_summary"].iloc[0].to_dict()
    rows.append(
        {
            "dataset": dataset,
            "fee_rate": fee_rate,
            "diagnostic": "skipped_trades_no_trade_zone",
            "bucket": "no_trade_zone",
            "value": execution_summary["skipped_trades_no_trade_zone"],
            "ratio": np.nan,
        }
    )
    return rows


def write_report(summary: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)
    diagnostics.to_csv(DIAGNOSTICS_CSV, index=False)

    lines = [
        "# v3 Baseline BTCUSDT 1h Validation",
        "",
        "This is an untuned v3 architecture baseline. It checks whether the v3 pipeline runs and compares it against existing references; parameters were not tuned to beat v2.",
        "",
        "## Scope",
        "",
        f"- Datasets: {', '.join(sorted(summary['dataset'].unique()))}",
        f"- Fee rates: {', '.join(f'{fee:g}' for fee in FEE_RATES)}",
        "- Versions: v1.final, v2.btc_final_candidate_A, v3.baseline, buy_and_hold, ma20_ma60",
        "",
        "## Comparison Table",
        "",
        _frame_to_markdown(summary[[
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
        ]]),
        "",
        "## v3 Diagnostics",
        "",
        "### Long Regime Distribution",
        "",
        _frame_to_markdown(diagnostics[diagnostics["diagnostic"] == "long_regime_distribution"]),
        "",
        "### Executed Position Distribution",
        "",
        _frame_to_markdown(diagnostics[diagnostics["diagnostic"] == "executed_position_distribution"]),
        "",
        "### Risk Action Counts",
        "",
        _frame_to_markdown(diagnostics[diagnostics["diagnostic"] == "risk_action_counts"]),
        "",
        "### Skipped Trades Due To No-Trade Zone",
        "",
        _frame_to_markdown(diagnostics[diagnostics["diagnostic"] == "skipped_trades_no_trade_zone"]),
        "",
        "## Files",
        "",
        f"- Summary CSV: `{SUMMARY_CSV}`",
        f"- Diagnostics CSV: `{DIAGNOSTICS_CSV}`",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


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

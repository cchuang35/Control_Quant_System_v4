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
from src.v3.backtest_v3 import BacktestV3Config, run_v3_backtest
from v2_small_cap import backtest_v2_btc_final_candidate_a


FEE_RATES = (0.0005, 0.0010, 0.0020)
PERIODS_PER_YEAR = 365 * 24
V1_ENTRY_THRESHOLD = 0.10
OUT_CSV = Path("reports") / "v3_rolling_validation_btc_1h.csv"
REPORT_PATH = Path("reports") / "v3_rolling_validation_btc_1h.md"
BTC_DATASETS = (
    "btcusdt_1h.csv",
    "btcusdt_1h_365d.csv",
    "btcusdt_1h_2y.csv",
    "btcusdt_1h_3y.csv",
    "btcusdt_1h_5y.csv",
)
ROLLING_WINDOWS = {
    "90d": 90 * 24,
    "180d": 180 * 24,
    "365d": 365 * 24,
    "2y": 2 * 365 * 24,
}


def discover_btc_datasets(data_dir: Path = Path("data")) -> dict[str, Path]:
    return {
        path.stem: path
        for name in BTC_DATASETS
        for path in [data_dir / name]
        if path.exists()
    }


def run_rolling_validation() -> pd.DataFrame:
    datasets = discover_btc_datasets()
    if not datasets:
        raise FileNotFoundError("No BTCUSDT 1h datasets found under data/")

    rows: list[dict[str, Any]] = []
    for dataset, path in datasets.items():
        print(f"dataset={dataset}")
        data = load_ohlcv_csv(path)
        for fee_rate in FEE_RATES:
            print(f"  fee={fee_rate:g}")
            v1 = run_backtest_fast(
                data,
                fee_rate=fee_rate,
                periods_per_year=PERIODS_PER_YEAR,
                progress_every=10000 if len(data) > 15000 else None,
            )
            v2_input = input_frame_from_v1(v1)
            frames = {
                "v2.btc_final_candidate_A": normalize_v2_frame(
                    backtest_v2_btc_final_candidate_a(
                        v2_input,
                        fee_rate=fee_rate,
                        v1_entry_threshold=V1_ENTRY_THRESHOLD,
                        cooldown_bars=120,
                    )
                ),
                "v3.baseline": normalize_v3_frame(
                    run_v3_backtest(data, config=BacktestV3Config(fee_rate=fee_rate, cooldown_bars=120))
                ),
            }
            for version, frame in frames.items():
                for window_name, window_frame in rolling_windows(frame):
                    rows.append(
                        {
                            "dataset": dataset,
                            "fee_rate": fee_rate,
                            "version": version,
                            "window": window_name,
                            **summarize_frame(window_frame),
                        }
                    )

    result = pd.DataFrame(rows)
    return add_v2_comparison(result)


def input_frame_from_v1(result: Any) -> pd.DataFrame:
    exposure = result.exposure_history.reset_index(drop=True).copy()
    return pd.DataFrame(
        {
            "timestamp": exposure["execution_timestamp"],
            "close": exposure["close"].astype(float),
            "current_exposure": exposure["current_exposure"].astype(float),
        }
    )


def normalize_v2_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": frame["timestamp"] if "timestamp" in frame.columns else frame.index,
            "strategy_return_net": frame["strategy_return_net"].astype(float),
            "drawdown": frame["drawdown"].astype(float),
            "position": frame["final_position"].astype(float),
            "trade_amount": frame["trade_size"].astype(float),
            "fee_cost": frame["fee_cost"].astype(float),
        }
    )


def normalize_v3_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": frame["timestamp"] if "timestamp" in frame.columns else frame.index,
            "strategy_return_net": frame["strategy_return_net"].astype(float),
            "drawdown": frame["drawdown"].astype(float),
            "position": frame["executed_position"].astype(float),
            "trade_amount": frame["trade_amount"].astype(float),
            "fee_cost": frame["fee_cost"].astype(float),
        }
    )


def rolling_windows(frame: pd.DataFrame, step_days: int = 30) -> list[tuple[str, pd.DataFrame]]:
    windows: list[tuple[str, pd.DataFrame]] = []
    for label, size in ROLLING_WINDOWS.items():
        step = step_days * 24
        if len(frame) < size:
            continue
        idx = 0
        number = 1
        while idx + size <= len(frame):
            windows.append((f"rolling_{label}_{number}", frame.iloc[idx : idx + size].copy()))
            idx += step
            number += 1
        if (len(frame) - size) % step != 0:
            windows.append((f"rolling_{label}_final", frame.iloc[-size:].copy()))
    return windows


def summarize_frame(frame: pd.DataFrame) -> dict[str, float | int | str]:
    returns = pd.to_numeric(frame["strategy_return_net"], errors="coerce").fillna(0.0)
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    position = pd.to_numeric(frame["position"], errors="coerce").fillna(0.0)
    trade_amount = pd.to_numeric(frame["trade_amount"], errors="coerce").fillna(0.0)
    fee_cost = pd.to_numeric(frame["fee_cost"], errors="coerce").fillna(0.0)
    std = float(returns.std(ddof=0))
    start = str(frame["timestamp"].iloc[0]) if "timestamp" in frame.columns and not frame.empty else ""
    end = str(frame["timestamp"].iloc[-1]) if "timestamp" in frame.columns and not frame.empty else ""
    return {
        "window_start": start,
        "window_end": end,
        "bars": int(len(frame)),
        "total_return": float(equity.iloc[-1] - 1.0) if len(equity) else 0.0,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "sharpe_ratio": 0.0 if std == 0.0 else float(returns.mean() / std * np.sqrt(PERIODS_PER_YEAR)),
        "turnover": float(trade_amount.sum()),
        "average_exposure": float(position.abs().mean()) if len(position) else 0.0,
        "fee_drag": float(fee_cost.sum()),
    }


def add_v2_comparison(result: pd.DataFrame) -> pd.DataFrame:
    key = ["dataset", "fee_rate", "window"]
    v2 = result[result["version"] == "v2.btc_final_candidate_A"][
        key + ["total_return", "max_drawdown", "sharpe_ratio", "turnover", "average_exposure", "fee_drag"]
    ].rename(
        columns={
            "total_return": "v2_total_return",
            "max_drawdown": "v2_max_drawdown",
            "sharpe_ratio": "v2_sharpe_ratio",
            "turnover": "v2_turnover",
            "average_exposure": "v2_average_exposure",
            "fee_drag": "v2_fee_drag",
        }
    )
    joined = result.merge(v2, on=key, how="left")
    joined["total_return_vs_v2"] = joined["total_return"] - joined["v2_total_return"]
    joined["max_drawdown_vs_v2"] = joined["max_drawdown"] - joined["v2_max_drawdown"]
    joined["sharpe_ratio_vs_v2"] = joined["sharpe_ratio"] - joined["v2_sharpe_ratio"]
    joined["turnover_vs_v2"] = joined["turnover"] - joined["v2_turnover"]
    joined["fee_drag_vs_v2"] = joined["fee_drag"] - joined["v2_fee_drag"]
    return joined


def write_report(result: pd.DataFrame) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_CSV, index=False)

    v3 = result[result["version"] == "v3.baseline"].copy()
    win_summary = build_win_summary(v3)
    aggregate = build_aggregate_summary(result)
    lines = [
        "# v3 Rolling Validation BTCUSDT 1h",
        "",
        "This is a robustness evaluation only. No v3 parameters were optimized for this run.",
        "",
        "## Scope",
        "",
        f"- Datasets: {', '.join(sorted(result['dataset'].unique()))}",
        f"- Fee rates: {', '.join(f'{fee:g}' for fee in FEE_RATES)}",
        "- Rolling windows: 90d, 180d, 365d, 2y when enough data exists",
        "- Comparison baseline: v2.btc_final_candidate_A",
        "",
        "## Aggregate Metrics",
        "",
        _frame_to_markdown(aggregate),
        "",
        "## v3 Win Rates Versus v2",
        "",
        _frame_to_markdown(win_summary),
        "",
        "## Full Rolling Result Sample",
        "",
        _frame_to_markdown(result.head(40)),
        "",
        "## Files",
        "",
        f"- Rolling CSV: `{OUT_CSV}`",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def build_aggregate_summary(result: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for version, group in result.groupby("version"):
        rows.append(
            {
                "version": version,
                "windows": len(group),
                "avg_total_return": group["total_return"].mean(),
                "avg_max_drawdown": group["max_drawdown"].mean(),
                "avg_sharpe_ratio": group["sharpe_ratio"].mean(),
                "avg_turnover": group["turnover"].mean(),
                "avg_average_exposure": group["average_exposure"].mean(),
                "avg_fee_drag": group["fee_drag"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("version").reset_index(drop=True)


def build_win_summary(v3: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, fee_rate), group in v3.groupby(["dataset", "fee_rate"]):
        rows.append(
            {
                "dataset": dataset,
                "fee_rate": fee_rate,
                "windows": len(group),
                "total_return_win_rate": float((group["total_return_vs_v2"] >= 0.0).mean()) if len(group) else 0.0,
                "max_drawdown_win_rate": float((group["max_drawdown_vs_v2"] >= 0.0).mean()) if len(group) else 0.0,
                "sharpe_win_rate": float((group["sharpe_ratio_vs_v2"] >= 0.0).mean()) if len(group) else 0.0,
                "lower_turnover_rate": float((group["turnover_vs_v2"] <= 0.0).mean()) if len(group) else 0.0,
                "lower_fee_drag_rate": float((group["fee_drag_vs_v2"] <= 0.0).mean()) if len(group) else 0.0,
            }
        )
    return pd.DataFrame(rows)


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
    result = run_rolling_validation()
    write_report(result)
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()

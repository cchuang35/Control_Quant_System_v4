from __future__ import annotations

from pathlib import Path
import sys
from time import perf_counter

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtester import load_ohlcv_csv, run_backtest_fast
from v2_small_cap import (
    backtest_v24_small_cap,
    backtest_v2_final_candidate_a,
    calculate_performance_stats,
    compute_regime_features,
)


FEE_RATES = (0.0005, 0.0010, 0.0020)
PERIODS = 365
THRESHOLD = 0.10
OUT_DIR = Path("reports") / "v211_oos_validation"
REQUESTED_DATASETS = (
    "btcusdt_1h.csv",
    "btcusdt_1h_365d.csv",
    "btcusdt_1h_2y.csv",
    "btcusdt_1h_3y.csv",
    "btcusdt_1h_5y.csv",
    "ethusdt_1h_365d.csv",
    "ethusdt_1h_2y.csv",
    "ethusdt_1h_3y.csv",
    "ethusdt_1h_5y.csv",
)


def discover_datasets(data_dir: Path = Path("data")) -> dict[str, Path]:
    available: dict[str, Path] = {}
    requested = {name.lower() for name in REQUESTED_DATASETS}
    for path in sorted(data_dir.glob("*.csv")):
        lower_name = path.name.lower()
        if lower_name in requested or (("btcusdt" in lower_name or "ethusdt" in lower_name) and "1h" in lower_name):
            available[path.stem] = path
    return available


def v1_frame_from_result(result, fee_rate: float) -> pd.DataFrame:
    exposure = result.exposure_history.reset_index(drop=True).copy()
    equity = result.equity_curve.iloc[1:].reset_index(drop=True).copy()
    current_exposure = exposure["current_exposure"].astype(float)
    frame = pd.DataFrame(
        {
            "timestamp": exposure["execution_timestamp"],
            "close": exposure["close"].astype(float),
            "current_exposure": current_exposure,
            "v1_position": (current_exposure.abs() >= THRESHOLD).astype(int),
            "final_position": current_exposure,
            "trade_size": exposure["safe_exposure_change"].abs().astype(float),
            "fee_cost": exposure["safe_exposure_change"].abs().astype(float) * fee_rate,
            "strategy_return_net": equity["period_return"].astype(float),
            "equity_net": equity["equity"].astype(float),
            "drawdown": equity["drawdown"].astype(float),
        }
    )
    frame["asset_return"] = frame["close"].pct_change().fillna(0.0)
    result_frame = compute_regime_features(frame)
    for column in [
        "timestamp",
        "current_exposure",
        "v1_position",
        "final_position",
        "trade_size",
        "fee_cost",
        "strategy_return_net",
        "equity_net",
        "drawdown",
    ]:
        result_frame[column] = frame[column].values
    return result_frame


def input_frame_from_v1(result) -> pd.DataFrame:
    exposure = result.exposure_history.reset_index(drop=True).copy()
    return pd.DataFrame(
        {
            "timestamp": exposure["execution_timestamp"],
            "close": exposure["close"].astype(float),
            "current_exposure": exposure["current_exposure"].astype(float),
        }
    )


def flag_sum(frame: pd.DataFrame, column: str) -> int:
    return int(frame.get(column, pd.Series(False, index=frame.index)).sum())


def window_bounds(frame: pd.DataFrame) -> tuple[str, str]:
    if "timestamp" not in frame.columns or frame.empty:
        return "", ""
    return str(frame["timestamp"].iloc[0]), str(frame["timestamp"].iloc[-1])


def summarize(
    *,
    dataset: str,
    fee_rate: float,
    window_name: str,
    window_type: str,
    version: str,
    frame: pd.DataFrame,
    runtime_sec: float = 0.0,
) -> dict[str, float | int | str]:
    stats = calculate_performance_stats(frame, annualization_factor=PERIODS)
    attempts = flag_sum(frame, "weak_bull_entry_attempt")
    allowed = flag_sum(frame, "weak_bull_entry_allowed")
    blocked = flag_sum(frame, "weak_bull_entry_blocked")
    start, end = window_bounds(frame)
    return {
        "dataset": dataset,
        "fee_rate": fee_rate,
        "window_name": window_name,
        "window_type": window_type,
        "window_start": start,
        "window_end": end,
        "version": version,
        "runtime_sec": runtime_sec,
        "total_return_net": stats["total_return_net"],
        "ann_return_net": stats["annualized_return_net"],
        "max_drawdown": stats["max_drawdown"],
        "Sharpe": stats["Sharpe_net"],
        "avg_exposure": stats["average_exposure"],
        "turnover": stats["turnover"],
        "trades": stats["total_trades"],
        "fees": stats["total_fee_paid"],
        "entries": stats["number_of_entries"],
        "exits": stats["number_of_exits"],
        "avg_holding_days": stats["average_holding_days"],
        "weak_bull_entry_attempts": attempts,
        "weak_bull_entries_allowed": allowed,
        "weak_bull_entries_blocked": blocked,
        "weak_bull_block_ratio": blocked / attempts if attempts else 0.0,
        "weak_bull_cooldown_triggers": flag_sum(frame, "weak_bull_cooldown_trigger"),
        "weak_bull_cooldown_bars": flag_sum(frame, "weak_bull_cooldown_active"),
    }


def fixed_windows(frame: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    windows: list[tuple[str, str, pd.DataFrame]] = [("Full period", "full", frame)]
    n = len(frame)
    if n >= 2:
        windows.extend(
            [
                ("First half", "fixed", frame.iloc[: n // 2].copy()),
                ("Second half", "fixed", frame.iloc[n // 2 :].copy()),
            ]
        )
    if n >= 4:
        windows.extend(
            [
                ("Quarter 1", "fixed", frame.iloc[: n // 4].copy()),
                ("Quarter 2", "fixed", frame.iloc[n // 4 : n // 2].copy()),
                ("Quarter 3", "fixed", frame.iloc[n // 2 : 3 * n // 4].copy()),
                ("Quarter 4", "fixed", frame.iloc[3 * n // 4 :].copy()),
            ]
        )
    return windows


def rolling_windows(frame: pd.DataFrame, days: int, step_days: int = 30) -> list[tuple[str, str, pd.DataFrame]]:
    size = days * 24
    step = step_days * 24
    if len(frame) < size:
        return []
    windows = []
    idx = 0
    number = 1
    while idx + size <= len(frame):
        windows.append((f"Rolling {days}d #{number}", f"rolling_{days}d", frame.iloc[idx : idx + size].copy()))
        idx += step
        number += 1
    if (len(frame) - size) % step != 0:
        windows.append((f"Rolling {days}d final", f"rolling_{days}d", frame.iloc[-size:].copy()))
    return windows


def monthly_windows(frame: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    if "timestamp" not in frame.columns:
        return []
    temp = frame.copy()
    timestamp = pd.to_datetime(temp["timestamp"], utc=True, errors="coerce")
    temp["_month"] = timestamp.dt.strftime("%Y-%m")
    return [(f"Month {month}", "monthly", group.drop(columns=["_month"]).copy()) for month, group in temp.groupby("_month", sort=True)]


def run_version(version: str, v1_frame: pd.DataFrame, v2_input: pd.DataFrame, fee_rate: float) -> pd.DataFrame:
    if version == "v1.final":
        return v1_frame
    if version == "v2.4.pause_4_24_exit_7":
        return backtest_v24_small_cap(
            v2_input,
            fee_rate=fee_rate,
            v1_entry_threshold=THRESHOLD,
            pause_dd=-0.04,
            pause_bars=24,
            exit_dd=-0.07,
        )
    cooldown = int(version.rsplit("cd", 1)[1])
    return backtest_v2_final_candidate_a(
        v2_input,
        fee_rate=fee_rate,
        v1_entry_threshold=THRESHOLD,
        cooldown_bars=cooldown,
    )


def build_win_rates(summary: pd.DataFrame, baseline_version: str, compare_versions: list[str], metric_set: str) -> pd.DataFrame:
    rolling = summary[summary["window_type"].isin(["rolling_90d", "rolling_180d"])]
    baseline = rolling[rolling["version"] == baseline_version][
        ["dataset", "fee_rate", "window_name", "window_type", "Sharpe", "max_drawdown", "total_return_net"]
    ].rename(columns={"Sharpe": "base_sharpe", "max_drawdown": "base_mdd", "total_return_net": "base_ret"})
    rows = []
    for version in compare_versions:
        joined = rolling[rolling["version"] == version].merge(
            baseline,
            on=["dataset", "fee_rate", "window_name", "window_type"],
            how="inner",
        )
        for (dataset, fee_rate, window_type), group in joined.groupby(["dataset", "fee_rate", "window_type"]):
            count = len(group)
            rows.append(
                {
                    "baseline": baseline_version,
                    "metric_set": metric_set,
                    "dataset": dataset,
                    "fee_rate": fee_rate,
                    "version": version,
                    "window_type": window_type,
                    "windows": count,
                    "sharpe_win_rate": float((group["Sharpe"] >= group["base_sharpe"]).mean()) if count else 0.0,
                    "max_drawdown_win_rate": float((group["max_drawdown"] >= group["base_mdd"]).mean()) if count else 0.0,
                    "total_return_win_rate": float((group["total_return_net"] >= group["base_ret"]).mean()) if count else 0.0,
                }
            )
    return pd.DataFrame(rows)


def build_period_contribution(summary: pd.DataFrame) -> pd.DataFrame:
    contribution = summary[summary["window_type"].isin(["fixed", "monthly"])].copy()
    full = summary[summary["window_type"] == "full"][
        ["dataset", "fee_rate", "version", "total_return_net"]
    ].rename(columns={"total_return_net": "full_total_return_net"})
    contribution = contribution.merge(full, on=["dataset", "fee_rate", "version"], how="left")
    contribution["return_share"] = contribution["total_return_net"] / contribution["full_total_return_net"].replace(0, pd.NA)
    return contribution


def build_cooldown_similarity(summary: pd.DataFrame, candidate_versions: list[str]) -> pd.DataFrame:
    full = summary[(summary["window_type"] == "full") & (summary["version"].isin(candidate_versions))]
    rows = []
    for (dataset, fee_rate), group in full.groupby(["dataset", "fee_rate"]):
        metrics = group.set_index("version")[["total_return_net", "max_drawdown", "Sharpe", "trades", "fees"]]
        rows.append(
            {
                "dataset": dataset,
                "fee_rate": fee_rate,
                "versions": ",".join(metrics.index),
                "total_return_range": float(metrics["total_return_net"].max() - metrics["total_return_net"].min()),
                "max_drawdown_range": float(metrics["max_drawdown"].max() - metrics["max_drawdown"].min()),
                "sharpe_range": float(metrics["Sharpe"].max() - metrics["Sharpe"].min()),
                "trades_range": float(metrics["trades"].max() - metrics["trades"].min()),
                "fees_range": float(metrics["fees"].max() - metrics["fees"].min()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = discover_datasets()
    missing = [name for name in REQUESTED_DATASETS if name not in {path.name for path in datasets.values()}]
    versions = [
        "v1.final",
        "v2.4.pause_4_24_exit_7",
        "v2.final_candidate_A_cd120",
        "v2.final_candidate_A_cd144",
        "v2.final_candidate_A_cd168",
    ]
    candidate_versions = [version for version in versions if version.startswith("v2.final_candidate_A")]
    rows: list[dict[str, float | int | str]] = []
    runtime_rows: list[dict[str, float | str]] = []

    for fee_rate in FEE_RATES:
        for dataset, path in datasets.items():
            print(f"fee={fee_rate} dataset={dataset}")
            data = load_ohlcv_csv(path)
            start = perf_counter()
            v1 = run_backtest_fast(
                data,
                fee_rate=fee_rate,
                periods_per_year=PERIODS,
                progress_every=2000 if len(data) > 3000 else None,
            )
            v1_runtime = perf_counter() - start
            v1_frame = v1_frame_from_result(v1, fee_rate)
            v2_input = input_frame_from_v1(v1)
            for version in versions:
                start = perf_counter()
                frame = run_version(version, v1_frame, v2_input, fee_rate)
                runtime = v1_runtime if version == "v1.final" else perf_counter() - start
                runtime_rows.append({"dataset": dataset, "fee_rate": fee_rate, "version": version, "runtime_sec": runtime})
                safe_version = version.replace(".", "_")
                frame.to_csv(OUT_DIR / f"{dataset}__fee_{fee_rate:g}__{safe_version}_frame.csv", index=False)
                for window_name, window_type, window_frame in (
                    fixed_windows(frame)
                    + rolling_windows(frame, 90)
                    + rolling_windows(frame, 180)
                    + monthly_windows(frame)
                ):
                    rows.append(
                        summarize(
                            dataset=dataset,
                            fee_rate=fee_rate,
                            window_name=window_name,
                            window_type=window_type,
                            version=version,
                            frame=window_frame,
                            runtime_sec=runtime if window_type == "full" else 0.0,
                        )
                    )

    summary = pd.DataFrame(rows)
    runtime_summary = pd.DataFrame(runtime_rows)
    win_vs_v24 = build_win_rates(summary, "v2.4.pause_4_24_exit_7", candidate_versions, "vs_v24")
    win_vs_v1 = build_win_rates(summary, "v1.final", candidate_versions, "vs_v1")
    win_rates = pd.concat([win_vs_v24, win_vs_v1], ignore_index=True)
    contribution = build_period_contribution(summary)
    similarity = build_cooldown_similarity(summary, candidate_versions)
    availability = pd.DataFrame(
        {
            "available_dataset": list(datasets),
            "path": [str(path) for path in datasets.values()],
        }
    )
    missing_frame = pd.DataFrame({"missing_requested_dataset": missing})

    summary.to_csv(OUT_DIR / "expanded_oos_summary.csv", index=False)
    runtime_summary.to_csv(OUT_DIR / "runtime_summary.csv", index=False)
    win_rates.to_csv(OUT_DIR / "rolling_win_rates.csv", index=False)
    contribution.to_csv(OUT_DIR / "period_contribution.csv", index=False)
    similarity.to_csv(OUT_DIR / "cooldown_similarity.csv", index=False)
    availability.to_csv(OUT_DIR / "available_datasets.csv", index=False)
    missing_frame.to_csv(OUT_DIR / "missing_requested_datasets.csv", index=False)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 40)
    print("\nFULL PERIOD")
    print(summary[summary["window_type"] == "full"].to_string(index=False))
    print("\nROLLING WIN RATES")
    print(win_rates.to_string(index=False))
    print("\nCOOLDOWN SIMILARITY")
    print(similarity.to_string(index=False))
    print("\nMISSING REQUESTED DATASETS")
    print(missing_frame.to_string(index=False))
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()

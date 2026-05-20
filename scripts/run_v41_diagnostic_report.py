from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v4 import (  # noqa: E402
    BacktestConfig,
    BacktestEngine,
    FilteredSignals,
    MinimalContinuousController,
    MinimalFilterLayer,
    MinimalStateEstimator,
    Observation,
    StateVector,
    create_v41_default_config,
)


FEE_RATE = 0.001
OUT_DIR = Path("reports") / "v41_diagnostics"
DATASETS = (
    ("BTC", "365d", Path("data") / "btcusdt_1h_365d.csv"),
    ("BTC", "2y", Path("data") / "btcusdt_1h_2y.csv"),
    ("BTC", "3y", Path("data") / "btcusdt_1h_3y.csv"),
    ("BTC", "5y", Path("data") / "btcusdt_1h_5y.csv"),
    ("ETH", "365d", Path("data") / "ethusdt_1h_365d.csv"),
    ("ETH", "2y", Path("data") / "ethusdt_1h_2y.csv"),
    ("ETH", "3y", Path("data") / "ethusdt_1h_3y.csv"),
    ("ETH", "5y", Path("data") / "ethusdt_1h_5y.csv"),
)
STAT_COLUMNS = ["mean", "median", "std", "min", "p05", "p25", "p75", "p95", "max"]
STATE_COLUMNS = ["tau", "nu", "epsilon", "rho", "state_previous_position"]
FILTER_COLUMNS = ["long_trend", "volatility", "short_timing", "filtered_drawdown", "filtered_previous_position"]
CONTROLLER_COLUMNS = [
    "base_exposure",
    "timing_multiplier",
    "market_risk_multiplier",
    "portfolio_risk_multiplier",
    "unsmoothed_target",
    "clipped_delta",
    "raw_target_position",
    "final_position",
]
THRESHOLDS = (
    ("pct_tau_gt_0", "tau", 0.0),
    ("pct_tau_gt_0_1", "tau", 0.1),
    ("pct_tau_gt_0_25", "tau", 0.25),
    ("pct_tau_gt_0_5", "tau", 0.5),
    ("pct_position_gt_0", "final_position", 0.0),
    ("pct_position_gt_0_05", "final_position", 0.05),
    ("pct_position_gt_0_10", "final_position", 0.10),
    ("pct_position_gt_0_20", "final_position", 0.20),
    ("pct_nu_gt_0_5", "nu", 0.5),
    ("pct_rho_gt_0_5", "rho", 0.5),
)


class RecordingStateEstimator:
    def __init__(self) -> None:
        config = create_v41_default_config()
        self.filter_layer = MinimalFilterLayer(config.filter)
        self.mapper = MinimalStateEstimator(config=config.state_estimator)
        self.filtered_history: list[FilteredSignals] = []
        self.state_history: list[StateVector] = []

    def update(self, observation: Observation) -> StateVector:
        filtered = self.filter_layer.update(observation)
        state = self.mapper.estimate_from_filtered(filtered)
        self.filtered_history.append(filtered)
        self.state_history.append(state)
        return state


class RecordingController:
    def __init__(self) -> None:
        config = create_v41_default_config()
        self.controller = MinimalContinuousController(config.controller)
        self.trace_history: list[dict[str, float]] = []

    def decide(self, state: StateVector) -> float:
        trace = self.controller.explain(state)
        self.trace_history.append(trace)
        return trace["raw_target_position"]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    detail_frames = []
    summary_rows = []
    threshold_rows = []

    for asset, window, path in DATASETS:
        prices = load_daily_close(path)
        detail = run_diagnostic_backtest(asset=asset, window=window, prices=prices)
        detail_frames.append(detail)
        summary_rows.extend(summarize_variables(detail))
        threshold_rows.append(summarize_thresholds(detail))

    details = pd.concat(detail_frames, ignore_index=True)
    variable_summary = pd.DataFrame(summary_rows)
    threshold_summary = pd.DataFrame(threshold_rows)

    details_path = OUT_DIR / "v41_diagnostic_period_details.csv"
    variable_summary_path = OUT_DIR / "v41_diagnostic_variable_summary.csv"
    threshold_summary_path = OUT_DIR / "v41_diagnostic_threshold_summary.csv"
    report_path = OUT_DIR / "v41_diagnostic_report.md"

    details.to_csv(details_path, index=False)
    variable_summary.to_csv(variable_summary_path, index=False)
    threshold_summary.to_csv(threshold_summary_path, index=False)
    report_path.write_text(build_markdown_report(variable_summary, threshold_summary), encoding="utf-8")

    print(f"period_details_csv: {details_path}")
    print(f"variable_summary_csv: {variable_summary_path}")
    print(f"threshold_summary_csv: {threshold_summary_path}")
    print(f"markdown_report: {report_path}")
    print("\nThreshold summary:")
    print(threshold_summary.to_string(index=False))
    print("\nVariable summary:")
    print(variable_summary.to_string(index=False))


def run_diagnostic_backtest(*, asset: str, window: str, prices: pd.Series) -> pd.DataFrame:
    estimator = RecordingStateEstimator()
    controller = RecordingController()
    result = BacktestEngine(
        controller=controller,
        state_estimator=estimator,
        config=BacktestConfig(fee_rate=FEE_RATE),
    ).run(prices)
    if not (len(result) == len(estimator.filtered_history) == len(estimator.state_history) == len(controller.trace_history)):
        raise RuntimeError("diagnostic histories are not aligned")

    diagnostic = result[["timestamp", "price", "position", "previous_position", "drawdown"]].copy()
    diagnostic.insert(0, "window", window)
    diagnostic.insert(0, "asset", asset)
    diagnostic.insert(2, "strategy_name", "v4.1-minimal-control-strategy")

    for idx, state in enumerate(estimator.state_history):
        diagnostic.loc[idx, "tau"] = state.tau
        diagnostic.loc[idx, "nu"] = state.nu
        diagnostic.loc[idx, "epsilon"] = state.epsilon
        diagnostic.loc[idx, "rho"] = state.rho
        diagnostic.loc[idx, "state_previous_position"] = state.previous_position

    for idx, filtered in enumerate(estimator.filtered_history):
        diagnostic.loc[idx, "long_trend"] = filtered.long_trend
        diagnostic.loc[idx, "volatility"] = filtered.volatility
        diagnostic.loc[idx, "short_timing"] = filtered.short_timing
        diagnostic.loc[idx, "filtered_drawdown"] = filtered.drawdown
        diagnostic.loc[idx, "filtered_previous_position"] = filtered.previous_position

    for idx, trace in enumerate(controller.trace_history):
        for key in CONTROLLER_COLUMNS:
            diagnostic.loc[idx, key] = trace[key]

    return diagnostic


def summarize_variables(detail: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    grouped_columns = {
        "state": STATE_COLUMNS,
        "controller": CONTROLLER_COLUMNS,
        "filtered_signals": FILTER_COLUMNS,
    }
    asset = str(detail["asset"].iloc[0])
    window = str(detail["window"].iloc[0])
    for group, columns in grouped_columns.items():
        for column in columns:
            series = detail[column].astype(float)
            rows.append(
                {
                    "asset": asset,
                    "window": window,
                    "group": group,
                    "variable": column,
                    "mean": float(series.mean()),
                    "median": float(series.median()),
                    "std": float(series.std(ddof=0)),
                    "min": float(series.min()),
                    "p05": float(series.quantile(0.05)),
                    "p25": float(series.quantile(0.25)),
                    "p75": float(series.quantile(0.75)),
                    "p95": float(series.quantile(0.95)),
                    "max": float(series.max()),
                }
            )
    return rows


def summarize_thresholds(detail: pd.DataFrame) -> dict[str, Any]:
    row: dict[str, Any] = {
        "asset": str(detail["asset"].iloc[0]),
        "window": str(detail["window"].iloc[0]),
        "strategy_name": "v4.1-minimal-control-strategy",
    }
    for name, column, threshold in THRESHOLDS:
        row[name] = float((detail[column].astype(float) > threshold).mean())
    return row


def load_daily_close(path: Path) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if "timestamp" not in frame.columns or "close" not in frame.columns:
        raise ValueError(f"{path} must contain timestamp and close columns")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("timestamp").set_index("timestamp")
    daily = frame["close"].astype(float).resample("1D").last().dropna()
    if len(daily) < 2:
        raise ValueError(f"{path} produced fewer than two daily closes")
    if (daily <= 0.0).any():
        raise ValueError(f"{path} contains non-positive daily closes")
    return daily


def build_markdown_report(variable_summary: pd.DataFrame, threshold_summary: pd.DataFrame) -> str:
    sections = [
        "# v4.1 Minimal Control Strategy Diagnostic Report",
        "",
        "No strategy logic, parameters, or optimization settings were changed. Daily series were derived from local 1h data using each UTC day's last close.",
        "",
        "## Threshold Summary",
        "",
        frame_to_markdown(threshold_summary),
        "",
        "## Variable Summary",
        "",
        frame_to_markdown(variable_summary),
        "",
    ]
    return "\n".join(sections)


def frame_to_markdown(frame: pd.DataFrame) -> str:
    rendered = frame.copy()
    for column in rendered.columns:
        if pd.api.types.is_float_dtype(rendered[column]):
            rendered[column] = rendered[column].map(lambda value: "nan" if pd.isna(value) else f"{value:.6f}")
        else:
            rendered[column] = rendered[column].astype(str)
    headers = list(rendered.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rendered.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()

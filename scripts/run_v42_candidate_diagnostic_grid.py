from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v4 import (  # noqa: E402
    BacktestConfig,
    BacktestEngine,
    ControllerConfig,
    FilterConfig,
    FilteredSignals,
    MinimalContinuousController,
    MinimalFilterLayer,
    MinimalStateEstimator,
    Observation,
    StateEstimatorConfig,
    StateVector,
    evaluate_metrics,
)


FEE_RATE = 0.001
PERIODS_PER_YEAR = 365
OUT_DIR = Path("reports") / "v42_candidate_diagnostic_grid"
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
METRIC_COLUMNS = [
    "asset",
    "window",
    "candidate_group",
    "candidate_name",
    "k_tau",
    "vol_ref",
    "total_return",
    "annualized_return",
    "max_drawdown",
    "sharpe_ratio",
    "total_turnover",
    "average_turnover",
    "average_exposure",
    "total_fee_cost",
    "trade_count",
    "final_equity",
    "min_position",
    "max_position",
    "position_std",
    "average_drawdown",
    "pct_position_gt_0",
    "pct_position_gt_0_05",
    "pct_position_gt_0_10",
    "pct_position_gt_0_20",
    "pct_tau_gt_0",
    "pct_tau_gt_0_1",
    "pct_tau_gt_0_25",
    "pct_tau_gt_0_5",
    "pct_nu_gt_0_5",
    "pct_rho_gt_0_5",
]
DELTA_BASE_COLUMNS = ["asset", "window", "candidate_group", "candidate_name", "k_tau", "vol_ref"]
SUMMARY_VARIABLES = [
    "tau",
    "nu",
    "rho",
    "base_exposure",
    "market_risk_multiplier",
    "unsmoothed_target",
    "final_position",
]
SUMMARY_STATS = ["mean", "median", "std", "min", "p05", "p25", "p75", "p95", "max"]
THRESHOLDS = (
    ("pct_position_gt_0", "final_position", 0.0),
    ("pct_position_gt_0_05", "final_position", 0.05),
    ("pct_position_gt_0_10", "final_position", 0.10),
    ("pct_position_gt_0_20", "final_position", 0.20),
    ("pct_tau_gt_0", "tau", 0.0),
    ("pct_tau_gt_0_1", "tau", 0.1),
    ("pct_tau_gt_0_25", "tau", 0.25),
    ("pct_tau_gt_0_5", "tau", 0.5),
    ("pct_nu_gt_0_5", "nu", 0.5),
    ("pct_rho_gt_0_5", "rho", 0.5),
)


@dataclass(frozen=True)
class Candidate:
    candidate_group: str
    candidate_name: str
    k_tau: float
    vol_ref: float


CANDIDATES = (
    Candidate("baseline", "v4.1_default", 1.0, 0.03),
    Candidate("A_trend_sensitivity", "k_tau_2_0", 2.0, 0.03),
    Candidate("A_trend_sensitivity", "k_tau_3_0", 3.0, 0.03),
    Candidate("A_trend_sensitivity", "k_tau_5_0", 5.0, 0.03),
    Candidate("B_volatility_reference", "vol_ref_0_04", 1.0, 0.04),
    Candidate("B_volatility_reference", "vol_ref_0_05", 1.0, 0.05),
    Candidate("B_volatility_reference", "vol_ref_0_06", 1.0, 0.06),
    Candidate("C_combined", "k_tau_2_0_vol_ref_0_04", 2.0, 0.04),
    Candidate("C_combined", "k_tau_3_0_vol_ref_0_05", 3.0, 0.05),
    Candidate("C_combined", "k_tau_5_0_vol_ref_0_06", 5.0, 0.06),
)


class RecordingStateEstimator:
    def __init__(self, *, k_tau: float, vol_ref: float) -> None:
        self.filter_layer = MinimalFilterLayer(FilterConfig())
        self.mapper = MinimalStateEstimator(
            config=StateEstimatorConfig(k_tau=k_tau, vol_ref=vol_ref),
        )
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
        self.controller = MinimalContinuousController(ControllerConfig())
        self.trace_history: list[dict[str, float]] = []

    def decide(self, state: StateVector) -> float:
        trace = self.controller.explain(state)
        self.trace_history.append(trace)
        return trace["raw_target_position"]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, Any]] = []
    variable_summary_rows: list[dict[str, Any]] = []

    for asset, window, path in DATASETS:
        prices = load_daily_close(path)
        for candidate in CANDIDATES:
            detail, result = run_candidate(asset=asset, window=window, prices=prices, candidate=candidate)
            metric_rows.append(summarize_metrics(asset=asset, window=window, candidate=candidate, result=result, detail=detail))
            variable_summary_rows.extend(summarize_variables(asset=asset, window=window, candidate=candidate, detail=detail))

    metrics = pd.DataFrame(metric_rows, columns=METRIC_COLUMNS)
    variable_summary = pd.DataFrame(variable_summary_rows)
    deltas = build_baseline_delta_table(metrics)

    metrics_path = OUT_DIR / "v42_candidate_metrics.csv"
    deltas_path = OUT_DIR / "v42_candidate_vs_baseline_deltas.csv"
    variable_summary_path = OUT_DIR / "v42_candidate_variable_summary.csv"
    report_path = OUT_DIR / "v42_candidate_diagnostic_grid.md"
    metrics.to_csv(metrics_path, index=False)
    deltas.to_csv(deltas_path, index=False)
    variable_summary.to_csv(variable_summary_path, index=False)
    report_path.write_text(build_markdown_report(metrics, deltas, variable_summary), encoding="utf-8")

    print(f"metrics_csv: {metrics_path}")
    print(f"baseline_deltas_csv: {deltas_path}")
    print(f"variable_summary_csv: {variable_summary_path}")
    print(f"markdown_report: {report_path}")
    print("\nCandidate metrics:")
    print(metrics.to_string(index=False))
    print("\nCandidate vs baseline deltas:")
    print(deltas.to_string(index=False))


def run_candidate(
    *,
    asset: str,
    window: str,
    prices: pd.Series,
    candidate: Candidate,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    estimator = RecordingStateEstimator(k_tau=candidate.k_tau, vol_ref=candidate.vol_ref)
    controller = RecordingController()
    result = BacktestEngine(
        controller=controller,
        state_estimator=estimator,
        config=BacktestConfig(fee_rate=FEE_RATE),
    ).run(prices)
    if not (len(result) == len(estimator.state_history) == len(controller.trace_history)):
        raise RuntimeError(f"diagnostic histories are not aligned for {asset} {window} {candidate.candidate_name}")

    detail = result[["timestamp", "position", "drawdown"]].copy()
    for idx, state in enumerate(estimator.state_history):
        detail.loc[idx, "tau"] = state.tau
        detail.loc[idx, "nu"] = state.nu
        detail.loc[idx, "rho"] = state.rho
    for idx, trace in enumerate(controller.trace_history):
        for column in ["base_exposure", "market_risk_multiplier", "unsmoothed_target", "final_position"]:
            detail.loc[idx, column] = trace[column]
    return detail, result


def summarize_metrics(
    *,
    asset: str,
    window: str,
    candidate: Candidate,
    result: pd.DataFrame,
    detail: pd.DataFrame,
) -> dict[str, Any]:
    metrics = evaluate_metrics(result, periods_per_year=PERIODS_PER_YEAR)
    row = {
        "asset": asset,
        "window": window,
        "candidate_group": candidate.candidate_group,
        "candidate_name": candidate.candidate_name,
        "k_tau": candidate.k_tau,
        "vol_ref": candidate.vol_ref,
        **metrics,
        "final_equity": float(result["equity"].iloc[-1]),
        "min_position": float(result["position"].min()),
        "max_position": float(result["position"].max()),
        "position_std": float(result["position"].std(ddof=0)),
        "average_drawdown": float(result["drawdown"].mean()),
    }
    for name, column, threshold in THRESHOLDS:
        row[name] = float((detail[column].astype(float) > threshold).mean())
    return row


def summarize_variables(
    *,
    asset: str,
    window: str,
    candidate: Candidate,
    detail: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows = []
    for variable in SUMMARY_VARIABLES:
        series = detail[variable].astype(float)
        rows.append(
            {
                "asset": asset,
                "window": window,
                "candidate_group": candidate.candidate_group,
                "candidate_name": candidate.candidate_name,
                "k_tau": candidate.k_tau,
                "vol_ref": candidate.vol_ref,
                "variable": variable,
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


def build_baseline_delta_table(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_names = [
        column
        for column in METRIC_COLUMNS
        if column not in {"asset", "window", "candidate_group", "candidate_name", "k_tau", "vol_ref"}
    ]
    baseline = metrics[metrics["candidate_name"] == "v4.1_default"].set_index(["asset", "window"])
    rows = []
    for row in metrics.itertuples(index=False):
        if row.candidate_name == "v4.1_default":
            continue
        base = baseline.loc[(row.asset, row.window)]
        delta_row: dict[str, Any] = {
            "asset": row.asset,
            "window": row.window,
            "candidate_group": row.candidate_group,
            "candidate_name": row.candidate_name,
            "k_tau": row.k_tau,
            "vol_ref": row.vol_ref,
        }
        for metric_name in metric_names:
            delta_row[f"delta_{metric_name}"] = float(getattr(row, metric_name) - base[metric_name])
        rows.append(delta_row)
    return pd.DataFrame(rows, columns=DELTA_BASE_COLUMNS + [f"delta_{name}" for name in metric_names])


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


def build_markdown_report(metrics: pd.DataFrame, deltas: pd.DataFrame, variable_summary: pd.DataFrame) -> str:
    sections = [
        "# v4.2 Candidate Diagnostic Grid",
        "",
        "No strategy architecture, controller formula, or optimization procedure was changed. Each candidate changes only the listed state-estimator diagnostic parameter(s).",
        "",
        "## Candidate Metrics",
        "",
        frame_to_markdown(metrics),
        "",
        "## Candidate vs Baseline Deltas",
        "",
        frame_to_markdown(deltas),
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

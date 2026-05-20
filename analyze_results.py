"""Analyze backtest outputs and write reports.

Expected input is a directory created by:

    python backtester.py --output-dir backtest_outputs/your_run
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_FILES = {
    "metrics": "metrics.csv",
    "equity_curve": "equity_curve.csv",
    "exposure_history": "exposure_history.csv",
    "estimated_state_history": "estimated_state_history.csv",
}


def analyze_backtest_output(input_dir: str | Path, reports_dir: str | Path = "reports") -> dict[str, pd.DataFrame]:
    """Read backtest output, write summary tables and charts, and return them."""

    input_path = Path(input_dir)
    report_path = Path(reports_dir)
    report_path.mkdir(parents=True, exist_ok=True)

    data = load_backtest_output(input_path)
    summary = build_performance_summary(data["metrics"], data["equity_curve"])
    regime_performance = build_regime_performance_table(
        data["equity_curve"],
        data["exposure_history"],
        data["estimated_state_history"],
    )

    summary.to_csv(report_path / "performance_summary.csv", index=False)
    regime_performance.to_csv(report_path / "regime_performance.csv", index=False)
    plot_all(data, report_path)

    return {
        "performance_summary": summary,
        "regime_performance": regime_performance,
    }


def load_backtest_output(input_dir: str | Path) -> dict[str, pd.DataFrame]:
    input_path = Path(input_dir)
    missing = [name for name in REQUIRED_FILES.values() if not (input_path / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing backtest output files in {input_path}: {missing}")
    return {
        key: pd.read_csv(input_path / filename)
        for key, filename in REQUIRED_FILES.items()
    }


def build_performance_summary(metrics: pd.DataFrame, equity_curve: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        raise ValueError("metrics.csv is empty")
    row = metrics.iloc[0].to_dict()
    summary_items: list[dict[str, Any]] = []
    for key, value in row.items():
        summary_items.append({"metric": key, "value": value})

    equity = pd.to_numeric(equity_curve["equity"], errors="coerce")
    period_return = pd.to_numeric(equity_curve["period_return"], errors="coerce").fillna(0.0)
    fee_paid = pd.to_numeric(equity_curve.get("fee_paid", pd.Series(0.0, index=equity_curve.index)), errors="coerce").fillna(0.0)
    summary_items.extend(
        [
            {"metric": "best_period_return", "value": float(period_return.max())},
            {"metric": "worst_period_return", "value": float(period_return.min())},
            {"metric": "average_period_return", "value": float(period_return.mean())},
            {"metric": "total_fees_paid", "value": float(fee_paid.sum())},
            {"metric": "equity_peak", "value": float(equity.max())},
            {"metric": "equity_trough", "value": float(equity.min())},
        ]
    )
    return pd.DataFrame(summary_items)


def build_regime_performance_table(
    equity_curve: pd.DataFrame,
    exposure_history: pd.DataFrame,
    estimated_state_history: pd.DataFrame,
) -> pd.DataFrame:
    if exposure_history.empty or estimated_state_history.empty:
        return pd.DataFrame(
            columns=[
                "dominant_regime",
                "periods",
                "total_return",
                "mean_return",
                "volatility",
                "sharpe_like",
                "max_drawdown",
                "avg_exposure",
                "turnover",
                "avg_danger_score",
                "avg_p_crash_risk",
            ]
        )

    period_returns = equity_curve.loc[:, ["timestamp", "period_return", "drawdown"]].rename(
        columns={"timestamp": "execution_timestamp"}
    )
    rows = (
        exposure_history.loc[:, ["decision_timestamp", "execution_timestamp", "current_exposure", "safe_exposure_change"]]
        .merge(
            estimated_state_history.loc[:, ["timestamp", "dominant_regime", "danger_score", "p_crash_risk"]],
            left_on="decision_timestamp",
            right_on="timestamp",
            how="left",
        )
        .merge(period_returns, on="execution_timestamp", how="left")
    )
    rows["period_return"] = pd.to_numeric(rows["period_return"], errors="coerce").fillna(0.0)
    rows["drawdown"] = pd.to_numeric(rows["drawdown"], errors="coerce").fillna(0.0)
    rows["current_exposure"] = pd.to_numeric(rows["current_exposure"], errors="coerce").fillna(0.0)
    rows["safe_exposure_change"] = pd.to_numeric(rows["safe_exposure_change"], errors="coerce").fillna(0.0)
    rows["danger_score"] = pd.to_numeric(rows["danger_score"], errors="coerce").fillna(0.0)
    rows["p_crash_risk"] = pd.to_numeric(rows["p_crash_risk"], errors="coerce").fillna(0.0)

    output_rows: list[dict[str, Any]] = []
    for regime, group in rows.groupby("dominant_regime", dropna=False):
        returns = group["period_return"]
        volatility = float(returns.std(ddof=0))
        output_rows.append(
            {
                "dominant_regime": regime,
                "periods": int(len(group)),
                "total_return": float((1.0 + returns).prod() - 1.0),
                "mean_return": float(returns.mean()),
                "volatility": volatility,
                "sharpe_like": 0.0 if volatility == 0.0 else float(returns.mean() / volatility),
                "max_drawdown": float(group["drawdown"].min()),
                "avg_exposure": float(group["current_exposure"].abs().mean()),
                "turnover": float(group["safe_exposure_change"].abs().sum()),
                "avg_danger_score": float(group["danger_score"].mean()),
                "avg_p_crash_risk": float(group["p_crash_risk"].mean()),
            }
        )
    return pd.DataFrame(output_rows).sort_values("dominant_regime").reset_index(drop=True)


def plot_all(data: dict[str, pd.DataFrame], reports_dir: Path) -> None:
    equity_curve = data["equity_curve"]
    exposure_history = data["exposure_history"]
    estimated = data["estimated_state_history"]

    _plot_line(
        equity_curve["timestamp"],
        pd.to_numeric(equity_curve["equity"], errors="coerce"),
        "Equity Curve",
        "Equity",
        reports_dir / "equity_curve.png",
    )
    _plot_line(
        equity_curve["timestamp"],
        pd.to_numeric(equity_curve["drawdown"], errors="coerce"),
        "Drawdown Curve",
        "Drawdown",
        reports_dir / "drawdown_curve.png",
    )
    _plot_line(
        exposure_history["execution_timestamp"],
        pd.to_numeric(exposure_history["current_exposure"], errors="coerce"),
        "Exposure Over Time",
        "Exposure",
        reports_dir / "exposure_over_time.png",
    )
    _plot_regime(estimated["timestamp"], estimated["dominant_regime"], reports_dir / "dominant_regime_over_time.png")
    _plot_line(
        estimated["timestamp"],
        pd.to_numeric(estimated["danger_score"], errors="coerce"),
        "Danger Score Over Time",
        "Danger Score",
        reports_dir / "danger_score_over_time.png",
    )
    _plot_line(
        estimated["timestamp"],
        pd.to_numeric(estimated["p_crash_risk"], errors="coerce"),
        "Crash Risk Probability Over Time",
        "P(Crash Risk)",
        reports_dir / "p_crash_risk_over_time.png",
    )


def _plot_line(x: pd.Series, y: pd.Series, title: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x.astype(str), y, linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    _thin_x_ticks(ax, len(x))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_regime(x: pd.Series, regimes: pd.Series, path: Path) -> None:
    labels = sorted(regimes.dropna().unique().tolist())
    mapping = {label: idx for idx, label in enumerate(labels)}
    y = regimes.map(mapping)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.step(x.astype(str), y, where="post", linewidth=1.6)
    ax.set_title("Dominant Regime Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Regime")
    ax.set_yticks(list(mapping.values()))
    ax.set_yticklabels(list(mapping.keys()))
    ax.grid(True, alpha=0.25)
    _thin_x_ticks(ax, len(x))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _thin_x_ticks(ax: plt.Axes, length: int) -> None:
    if length <= 12:
        return
    step = max(length // 8, 1)
    for idx, label in enumerate(ax.get_xticklabels()):
        label.set_visible(idx % step == 0 or idx == length - 1)
        label.set_rotation(30)
        label.set_horizontalalignment("right")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Layer 1-4 backtest outputs.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory created by backtester.py --output-dir.")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"), help="Directory where reports are saved.")
    args = parser.parse_args()

    tables = analyze_backtest_output(args.input_dir, args.reports_dir)
    print("Performance summary")
    print(tables["performance_summary"].to_string(index=False))
    print("\nRegime-wise performance")
    print(tables["regime_performance"].to_string(index=False))
    print(f"\nReports saved to: {args.reports_dir}")


if __name__ == "__main__":
    main()


"""Metrics and diagnostics for v3 backtest result frames."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def calculate_v3_metrics(result: pd.DataFrame, *, periods_per_year: int = 365 * 24) -> dict[str, float | int]:
    """Calculate headline v3 performance and exposure metrics."""

    if result.empty:
        return _empty_metrics()
    _require_columns(result, ["strategy_return_net", "equity_curve", "drawdown", "executed_position", "trade_amount", "fee_cost"])

    returns = pd.to_numeric(result["strategy_return_net"], errors="coerce").fillna(0.0)
    equity = pd.to_numeric(result["equity_curve"], errors="coerce").fillna(1.0)
    drawdown = pd.to_numeric(result["drawdown"], errors="coerce").fillna(0.0)
    position = pd.to_numeric(result["executed_position"], errors="coerce").fillna(0.0)
    trade_amount = pd.to_numeric(result["trade_amount"], errors="coerce").fillna(0.0)
    fee_cost = pd.to_numeric(result["fee_cost"], errors="coerce").fillna(0.0)

    final_equity = float(equity.iloc[-1])
    periods = max(len(result), 1)
    return_std = float(returns.std(ddof=0))
    active_returns = returns[returns != 0.0]
    return {
        "total_return": final_equity - 1.0,
        "annual_return": final_equity ** (periods_per_year / periods) - 1.0,
        "max_drawdown": float(drawdown.min()),
        "sharpe_ratio": 0.0 if return_std == 0.0 else float(returns.mean() / return_std * np.sqrt(periods_per_year)),
        "win_rate": float((active_returns > 0.0).mean()) if len(active_returns) else 0.0,
        "number_of_trades": int((trade_amount > 0.0).sum()),
        "turnover": float(trade_amount.sum()),
        "fee_drag": float(fee_cost.sum()),
        "average_exposure": float(position.abs().mean()),
        "max_exposure": float(position.abs().max()),
        "average_holding_period": _average_holding_period(position),
        # Backward-compatible aliases used by early v3 tests/CLI output.
        "total_return_net": final_equity - 1.0,
        "annualized_return_net": final_equity ** (periods_per_year / periods) - 1.0,
        "total_fee_paid": float(fee_cost.sum()),
        "total_trades": int((trade_amount > 0.0).sum()),
    }


def build_v3_diagnostics(result: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build CSV-friendly v3 diagnostics tables."""

    if result.empty:
        return _empty_diagnostics()
    return {
        "metrics": pd.DataFrame([calculate_v3_metrics(result)]),
        "exposure_distribution": _value_distribution(result, "executed_position", "executed_position"),
        "long_regime_performance": _regime_performance(result, "long_regime"),
        "short_regime_performance": _regime_performance(result, "short_regime"),
        "max_drawdown_period": _max_drawdown_period(result),
        "risk_action_counts": _value_distribution(result, "risk_action", "risk_action"),
        "risk_cap_distribution": _value_distribution(result, "risk_cap", "risk_cap"),
        "execution_summary": _execution_summary(result),
        "turnover_by_period": _turnover_by_period(result),
        "base_position_distribution": _value_distribution(result, "base_position", "base_position"),
        "short_adjustment_distribution": _value_distribution(result, "position_adjustment", "position_adjustment"),
        "executed_position_distribution": _value_distribution(result, "executed_position", "executed_position"),
    }


def write_v3_diagnostics(diagnostics: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    """Write diagnostics tables to CSV files."""

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    for name, frame in diagnostics.items():
        frame.to_csv(path / f"{name}.csv", index=False)


def build_v3_markdown_report(diagnostics: dict[str, pd.DataFrame]) -> str:
    """Build a compact markdown report from diagnostics tables."""

    lines = ["# v3 Backtest Diagnostics", ""]
    for name in [
        "metrics",
        "long_regime_performance",
        "short_regime_performance",
        "risk_action_counts",
        "execution_summary",
        "executed_position_distribution",
    ]:
        frame = diagnostics.get(name, pd.DataFrame())
        lines.extend([f"## {name}", "", _frame_to_markdown(frame) if not frame.empty else "_No data_", ""])
    return "\n".join(lines)


def _regime_performance(result: pd.DataFrame, regime_column: str) -> pd.DataFrame:
    _require_columns(result, [regime_column, "strategy_return_net", "asset_return", "trade_amount", "fee_cost", "executed_position"])
    rows: list[dict[str, Any]] = []
    for regime, group in result.groupby(regime_column, dropna=False):
        rows.append(
            {
                "regime": regime,
                "bars": int(len(group)),
                "strategy_return_net": float(group["strategy_return_net"].sum()),
                "asset_return": float(group["asset_return"].sum()),
                "trade_count": int((group["trade_amount"] > 0.0).sum()),
                "turnover": float(group["trade_amount"].sum()),
                "fees": float(group["fee_cost"].sum()),
                "avg_exposure": float(group["executed_position"].abs().mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("regime").reset_index(drop=True)


def _max_drawdown_period(result: pd.DataFrame) -> pd.DataFrame:
    _require_columns(result, ["timestamp", "equity_curve", "drawdown"])
    equity = pd.to_numeric(result["equity_curve"], errors="coerce").fillna(1.0).reset_index(drop=True)
    drawdown = pd.to_numeric(result["drawdown"], errors="coerce").fillna(0.0).reset_index(drop=True)
    trough_idx = int(drawdown.idxmin())
    peak_value = float(equity.iloc[: trough_idx + 1].max())
    peak_candidates = equity.iloc[: trough_idx + 1][equity.iloc[: trough_idx + 1] >= peak_value]
    start_idx = int(peak_candidates.index[-1]) if not peak_candidates.empty else 0
    recovery_idx: int | None = None
    for idx in range(trough_idx + 1, len(equity)):
        if float(equity.iloc[idx]) >= peak_value:
            recovery_idx = idx
            break
    episode = result.iloc[start_idx : trough_idx + 1]
    return pd.DataFrame(
        [
            {
                "dd_start_time": result["timestamp"].iloc[start_idx],
                "dd_trough_time": result["timestamp"].iloc[trough_idx],
                "dd_recovery_time": result["timestamp"].iloc[recovery_idx] if recovery_idx is not None else pd.NA,
                "dd_duration_bars": trough_idx - start_idx + 1,
                "dd_recovery_bars": recovery_idx - trough_idx if recovery_idx is not None else pd.NA,
                "equity_at_start": float(equity.iloc[start_idx]),
                "equity_at_trough": float(equity.iloc[trough_idx]),
                "max_drawdown": float(drawdown.iloc[trough_idx]),
                "strategy_return_during_dd": float(episode["strategy_return_net"].sum()),
                "fees_during_dd": float(episode["fee_cost"].sum()),
                "avg_exposure_during_dd": float(episode["executed_position"].abs().mean()) if not episode.empty else 0.0,
            }
        ]
    )


def _execution_summary(result: pd.DataFrame) -> pd.DataFrame:
    _require_columns(result, ["execution_reason", "fee_cost", "trade_amount"])
    no_trade_zone = result["execution_reason"].astype(str).str.contains("no_trade_zone", regex=False)
    return pd.DataFrame(
        [
            {
                "skipped_trades_no_trade_zone": int(no_trade_zone.sum()),
                "total_fee_cost": float(result["fee_cost"].sum()),
                "total_turnover": float(result["trade_amount"].sum()),
                "number_of_executed_trades": int((result["trade_amount"] > 0.0).sum()),
            }
        ]
    )


def _turnover_by_period(result: pd.DataFrame) -> pd.DataFrame:
    _require_columns(result, ["timestamp", "trade_amount", "fee_cost"])
    return pd.DataFrame(
        {
            "timestamp": result["timestamp"],
            "turnover": pd.to_numeric(result["trade_amount"], errors="coerce").fillna(0.0),
            "fee_cost": pd.to_numeric(result["fee_cost"], errors="coerce").fillna(0.0),
        }
    )


def _value_distribution(result: pd.DataFrame, column: str, output_name: str) -> pd.DataFrame:
    _require_columns(result, [column])
    counts = result[column].value_counts(dropna=False).sort_index().rename_axis(output_name).reset_index(name="bars")
    counts["ratio"] = counts["bars"] / max(len(result), 1)
    return counts


def _average_holding_period(position: pd.Series) -> float:
    holding_periods: list[int] = []
    current = 0
    for value in position.astype(float):
        if value > 0.0:
            current += 1
        elif current > 0:
            holding_periods.append(current)
            current = 0
    if current > 0:
        holding_periods.append(current)
    return float(np.mean(holding_periods)) if holding_periods else 0.0


def _frame_to_markdown(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    rows = frame.astype(object).where(pd.notna(frame), "").astype(str).values.tolist()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator] + body)


def _empty_metrics() -> dict[str, float | int]:
    return {
        "total_return": 0.0,
        "annual_return": 0.0,
        "max_drawdown": 0.0,
        "sharpe_ratio": 0.0,
        "win_rate": 0.0,
        "number_of_trades": 0,
        "turnover": 0.0,
        "fee_drag": 0.0,
        "average_exposure": 0.0,
        "max_exposure": 0.0,
        "average_holding_period": 0.0,
        "total_return_net": 0.0,
        "annualized_return_net": 0.0,
        "total_fee_paid": 0.0,
        "total_trades": 0,
    }


def _empty_diagnostics() -> dict[str, pd.DataFrame]:
    return {
        "metrics": pd.DataFrame([_empty_metrics()]),
        "exposure_distribution": pd.DataFrame(columns=["executed_position", "bars", "ratio"]),
        "long_regime_performance": pd.DataFrame(columns=["regime", "bars", "strategy_return_net", "asset_return", "trade_count", "turnover", "fees", "avg_exposure"]),
        "short_regime_performance": pd.DataFrame(columns=["regime", "bars", "strategy_return_net", "asset_return", "trade_count", "turnover", "fees", "avg_exposure"]),
        "max_drawdown_period": pd.DataFrame(),
        "risk_action_counts": pd.DataFrame(columns=["risk_action", "bars", "ratio"]),
        "risk_cap_distribution": pd.DataFrame(columns=["risk_cap", "bars", "ratio"]),
        "execution_summary": pd.DataFrame(columns=["skipped_trades_no_trade_zone", "total_fee_cost", "total_turnover", "number_of_executed_trades"]),
        "turnover_by_period": pd.DataFrame(columns=["timestamp", "turnover", "fee_cost"]),
        "base_position_distribution": pd.DataFrame(columns=["base_position", "bars", "ratio"]),
        "short_adjustment_distribution": pd.DataFrame(columns=["position_adjustment", "bars", "ratio"]),
        "executed_position_distribution": pd.DataFrame(columns=["executed_position", "bars", "ratio"]),
    }


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"result frame is missing required columns: {sorted(missing)}")

"""Metrics for the v4 minimal simulation framework."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


class MetricsEvaluator:
    """Compute framework-level performance, risk, cost, and activity metrics."""

    def evaluate(
        self,
        result: pd.DataFrame,
        *,
        periods_per_year: int | float,
        start_index: int | None = None,
    ) -> dict[str, Any]:
        if result.empty:
            raise ValueError("result must contain at least one return period")
        if periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")
        start = 0 if start_index is None else int(start_index)
        if start < 0:
            raise ValueError("start_index must be non-negative")
        if start >= len(result):
            raise ValueError("start_index must be less than the number of return periods")

        starting_equity = float(result.attrs.get("initial_equity", 1.0)) if start == 0 else float(result["equity"].iloc[start - 1])
        evaluation_result = result.iloc[start:]
        equity = evaluation_result["equity"].astype(float).to_numpy()
        equity_with_initial = np.concatenate([[starting_equity], equity])
        strategy_returns = equity_with_initial[1:] / equity_with_initial[:-1] - 1.0
        period_count = len(strategy_returns)

        mean_return = float(np.mean(strategy_returns))
        std_return = float(np.std(strategy_returns))
        sharpe = math.nan if std_return == 0.0 else mean_return / std_return * math.sqrt(float(periods_per_year))

        final_equity = float(equity_with_initial[-1])
        return {
            "total_return": final_equity / starting_equity - 1.0,
            "annualized_return": (final_equity / starting_equity) ** (float(periods_per_year) / period_count) - 1.0,
            "max_drawdown": float(evaluation_result["drawdown"].astype(float).max()),
            "sharpe_ratio": sharpe,
            "total_turnover": float(evaluation_result["turnover"].astype(float).sum()),
            "average_turnover": float(evaluation_result["turnover"].astype(float).sum() / period_count),
            "average_exposure": float(evaluation_result["previous_position"].astype(float).mean()),
            "total_fee_cost": float(evaluation_result["transaction_cost"].astype(float).sum()),
            "trade_count": int((evaluation_result["turnover"].astype(float) > 1e-6).sum()),
        }


def evaluate_metrics(
    result: pd.DataFrame,
    *,
    periods_per_year: int | float,
    start_index: int | None = None,
) -> dict[str, Any]:
    """Convenience wrapper around MetricsEvaluator."""

    return MetricsEvaluator().evaluate(result, periods_per_year=periods_per_year, start_index=start_index)


def evaluate_metrics_after_warmup(
    result: pd.DataFrame,
    *,
    periods_per_year: int | float,
    warmup_period: int,
) -> dict[str, Any]:
    """Evaluate metrics after the recursive filter warm-up period."""

    if warmup_period < 0:
        raise ValueError("warmup_period must be non-negative")
    return evaluate_metrics(result, periods_per_year=periods_per_year, start_index=warmup_period)

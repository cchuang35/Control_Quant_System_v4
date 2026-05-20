"""Run v4.2-candidate-C on a tiny daily price series."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.v4 import MetricsEvaluator, make_v42_candidate_c_config, run_v42_candidate_c_backtest


def main() -> None:
    prices = pd.Series(
        [100.0, 101.0, 102.0, 101.0, 103.0, 104.0, 102.0, 105.0, 106.0, 107.0],
        index=pd.date_range("2024-01-01", periods=10, freq="D"),
    )
    config = make_v42_candidate_c_config(fee_rate=0.001)
    result = run_v42_candidate_c_backtest(prices, config=config)
    metrics = MetricsEvaluator().evaluate(result, periods_per_year=365)
    print(result.tail())
    print(metrics)


if __name__ == "__main__":
    main()

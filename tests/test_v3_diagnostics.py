from pathlib import Path

import pandas as pd

from src.v3.backtest_v3 import BacktestV3Config, run_v3_backtest
from src.v3.diagnostics import build_v3_diagnostics, build_v3_markdown_report, calculate_v3_metrics, write_v3_diagnostics
from src.v3.feature_builder import FeatureWindowConfig


def sample_ohlcv(count: int = 160) -> pd.DataFrame:
    close = 100.0
    rows = []
    for idx in range(count):
        drift = 0.001 if idx < count // 2 else -0.0008
        shock = -0.04 if idx == count - 30 else 0.0
        close *= 1.0 + drift + shock
        rows.append(
            {
                "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(hours=idx),
                "open": close * 0.999,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1000.0 + idx,
            }
        )
    return pd.DataFrame(rows)


def fast_config() -> BacktestV3Config:
    return BacktestV3Config(
        fee_rate=0.001,
        feature_config=FeatureWindowConfig(
            ma_short=4,
            ma_long=8,
            ma_long_term=16,
            momentum_short=4,
            momentum_long=8,
            volatility_short=4,
            volatility_long=8,
            drawdown_short=8,
            drawdown_long=16,
        ),
    )


def test_v3_metrics_include_required_fields() -> None:
    result = run_v3_backtest(sample_ohlcv(), config=fast_config())
    metrics = calculate_v3_metrics(result)

    required = {
        "total_return",
        "annual_return",
        "max_drawdown",
        "sharpe_ratio",
        "win_rate",
        "number_of_trades",
        "turnover",
        "fee_drag",
        "average_exposure",
        "max_exposure",
        "average_holding_period",
    }
    assert required.issubset(metrics)
    assert 0.0 <= metrics["win_rate"] <= 1.0


def test_v3_diagnostics_build_csv_friendly_tables() -> None:
    result = run_v3_backtest(sample_ohlcv(), config=fast_config())
    diagnostics = build_v3_diagnostics(result)

    expected_tables = {
        "metrics",
        "exposure_distribution",
        "long_regime_performance",
        "short_regime_performance",
        "max_drawdown_period",
        "risk_action_counts",
        "risk_cap_distribution",
        "execution_summary",
        "turnover_by_period",
        "base_position_distribution",
        "short_adjustment_distribution",
        "executed_position_distribution",
    }
    assert expected_tables.issubset(diagnostics)
    assert {"regime", "strategy_return_net", "trade_count"}.issubset(diagnostics["long_regime_performance"].columns)
    assert diagnostics["execution_summary"]["total_fee_cost"].iloc[0] >= 0.0
    assert len(diagnostics["turnover_by_period"]) == len(result)


def test_v3_diagnostics_write_csv_and_markdown(tmp_path: Path) -> None:
    result = run_v3_backtest(sample_ohlcv(), config=fast_config())
    diagnostics = build_v3_diagnostics(result)

    write_v3_diagnostics(diagnostics, tmp_path)
    report = build_v3_markdown_report(diagnostics)

    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "long_regime_performance.csv").exists()
    assert "v3 Backtest Diagnostics" in report

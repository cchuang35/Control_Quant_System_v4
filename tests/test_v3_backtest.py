import numpy as np
import pandas as pd

from src.v3.backtest_v3 import BacktestV3Config, ConditionalLeverageConfigV3, calculate_v3_metrics, run_v3_backtest
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


def test_v3_backtest_outputs_required_columns() -> None:
    result = run_v3_backtest(sample_ohlcv(), config=fast_config())

    expected_columns = {
        "timestamp",
        "close",
        "asset_return",
        "long_regime",
        "short_regime",
        "confidence_score",
        "base_position",
        "position_adjustment",
        "risk_cap",
        "leverage_allowed",
        "leverage_used",
        "leverage_reason",
        "leverage_increment",
        "target_position",
        "executed_position",
        "trade_amount",
        "fee_cost",
        "strategy_return_gross",
        "strategy_return_net",
        "equity_curve",
        "drawdown",
        "risk_action",
        "long_reason",
        "short_reason",
        "risk_reason",
        "execution_reason",
    }
    assert expected_columns.issubset(result.columns)
    assert len(result) == 160
    assert set(result["executed_position"]).issubset({0.0, 0.25, 0.50, 0.75, 1.0})
    assert not result["leverage_used"].any()
    assert result["equity_curve"].iloc[-1] > 0.0


def test_v3_backtest_fee_aware_returns_use_lagged_position() -> None:
    result = run_v3_backtest(sample_ohlcv(), config=fast_config())

    expected_gross = result["executed_position"].shift(1).fillna(0.0) * result["asset_return"]
    expected_trade = result["executed_position"].diff().abs().fillna(result["executed_position"].abs())
    expected_fee = expected_trade * 0.001
    expected_net = expected_gross - expected_fee

    assert np.allclose(result["strategy_return_gross"], expected_gross)
    assert np.allclose(result["trade_amount"], expected_trade)
    assert np.allclose(result["fee_cost"], expected_fee)
    assert np.allclose(result["strategy_return_net"], expected_net)


def test_v3_backtest_metrics_are_available() -> None:
    result = run_v3_backtest(sample_ohlcv(), config=fast_config())
    metrics = calculate_v3_metrics(result)

    assert metrics["total_trades"] >= 0
    assert metrics["turnover"] >= 0.0
    assert metrics["total_fee_paid"] >= 0.0
    assert metrics["max_drawdown"] <= 0.0


def test_v3_backtest_rejects_leverage_above_experimental_cap() -> None:
    config = fast_config()
    bad_config = BacktestV3Config(
        fee_rate=config.fee_rate,
        feature_config=config.feature_config,
        leverage_config=ConditionalLeverageConfigV3(enabled=True, max_position=1.50),
    )

    import pytest

    with pytest.raises(ValueError, match="must not exceed 1.25"):
        run_v3_backtest(sample_ohlcv(), config=bad_config)

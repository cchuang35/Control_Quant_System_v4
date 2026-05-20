import numpy as np
import pandas as pd
import pytest

from src.v3.data_types import MarketFeaturesV3
from src.v3.feature_builder import (
    FEATURE_COLUMNS,
    FeatureWindowConfig,
    build_feature_frame,
    build_feature_records,
    validate_feature_frame,
)


def sample_ohlcv(count: int = 80) -> pd.DataFrame:
    close = 100.0
    rows = []
    for idx in range(count):
        close *= 1.0 + (0.001 if idx < count // 2 else -0.0005)
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


def test_v3_feature_builder_outputs_required_columns_and_records() -> None:
    features = build_feature_frame(sample_ohlcv(), config=FeatureWindowConfig(ma_long_term=48, drawdown_long=48))

    assert list(features.columns) == FEATURE_COLUMNS
    validate_feature_frame(features)
    assert features["return_1"].iloc[0] == 0.0
    assert features["shock_score"].between(0.0, 1.0).all()
    assert features["drawdown_short"].le(0.0).all()
    assert features["drawdown_long"].le(0.0).all()

    records = build_feature_records(sample_ohlcv(10), config=FeatureWindowConfig(ma_long_term=8, drawdown_long=8))
    assert len(records) == 10
    assert isinstance(records[0], MarketFeaturesV3)


def test_v3_feature_builder_uses_trailing_windows_only() -> None:
    frame = sample_ohlcv(80)
    config = FeatureWindowConfig(ma_long_term=48, drawdown_long=48)
    baseline = build_feature_frame(frame, config=config)

    mutated = frame.copy()
    mutated.loc[79, "close"] = mutated.loc[79, "close"] * 10.0
    changed = build_feature_frame(mutated, config=config)

    compare_columns = [column for column in FEATURE_COLUMNS if column != "timestamp"]
    pd.testing.assert_frame_equal(
        baseline.loc[:60, compare_columns],
        changed.loc[:60, compare_columns],
        check_exact=False,
        atol=1e-12,
        rtol=1e-12,
    )


def test_v3_feature_builder_supports_datetime_index() -> None:
    frame = sample_ohlcv(20).drop(columns=["timestamp"])
    frame.index = pd.date_range("2024-01-01", periods=len(frame), freq="h")

    features = build_feature_frame(frame, config=FeatureWindowConfig(ma_long_term=12, drawdown_long=12))

    assert features["timestamp"].iloc[0] == frame.index[0]
    validate_feature_frame(features)


def test_v3_feature_builder_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        build_feature_frame(pd.DataFrame({"close": [100.0]}))

    bad = sample_ohlcv(5)
    bad.loc[2, "close"] = np.nan
    with pytest.raises(ValueError, match="close must be numeric and non-null"):
        build_feature_frame(bad)

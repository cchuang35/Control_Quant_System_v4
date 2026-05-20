import pandas as pd
import pytest

from src.v3.data_types import MarketEstimateV3, MarketFeaturesV3
from src.v3.market_estimator import (
    DRAW_DOWN_STATES,
    ESTIMATE_COLUMNS,
    LONG_REGIMES,
    RISK_STATES,
    SHORT_REGIMES,
    VOLATILITY_STATES,
    MarketEstimatorConfig,
    estimate_market,
    estimate_market_frame,
    estimate_market_records,
)


def feature(**overrides) -> MarketFeaturesV3:
    values = {
        "timestamp": "t",
        "close": 120.0,
        "return_1": 0.001,
        "ma_short": 118.0,
        "ma_long": 112.0,
        "ma_long_term": 100.0,
        "momentum_short": 0.01,
        "momentum_long": 0.08,
        "volatility_short": 0.01,
        "volatility_long": 0.01,
        "volatility_ratio": 1.0,
        "drawdown_short": 0.0,
        "drawdown_long": 0.0,
        "shock_score": 0.0,
    }
    values.update(overrides)
    return MarketFeaturesV3(**values)


def test_v3_market_estimator_classifies_strong_bull_and_permissions() -> None:
    estimate = estimate_market(feature())

    assert estimate.long_regime == "strong_bull"
    assert estimate.short_regime == "noise"
    assert estimate.allow_entry
    assert estimate.allow_hold
    assert 0.0 <= estimate.confidence_score <= 1.0
    assert estimate.notes["particle_filter"] == "deferred_to_v4"


def test_v3_market_estimator_classifies_pullback_in_bullish_context() -> None:
    estimate = estimate_market(feature(momentum_short=-0.01, drawdown_short=-0.02))

    assert estimate.long_regime in {"strong_bull", "bull"}
    assert estimate.short_regime == "pullback"
    assert estimate.allow_entry
    assert estimate.allow_hold


def test_v3_market_estimator_classifies_overheat_and_breakdown() -> None:
    overheat = estimate_market(feature(close=126.0, ma_short=120.0, momentum_short=0.06))
    assert overheat.short_regime == "overheat"

    breakdown = estimate_market(feature(return_1=-0.06, momentum_short=-0.05, shock_score=0.95))
    assert breakdown.short_regime == "breakdown"
    assert breakdown.risk_state == "risk_off"


def test_v3_market_estimator_classifies_strong_bear_risk_off() -> None:
    estimate = estimate_market(
        feature(
            close=80.0,
            ma_short=82.0,
            ma_long=90.0,
            ma_long_term=100.0,
            momentum_short=-0.03,
            momentum_long=-0.12,
            drawdown_long=-0.22,
            volatility_ratio=2.0,
        )
    )

    assert estimate.long_regime == "strong_bear"
    assert estimate.drawdown_state == "severe"
    assert estimate.risk_state == "risk_off"
    assert not estimate.allow_entry
    assert not estimate.allow_hold


def test_v3_market_estimator_frame_and_records_outputs() -> None:
    records = [feature(timestamp=idx, close=120.0 + idx) for idx in range(3)]
    estimates = estimate_market_records(records)
    assert len(estimates) == 3
    assert isinstance(estimates[0], MarketEstimateV3)

    frame = pd.DataFrame([record.__dict__ for record in records])
    estimated_frame = estimate_market_frame(frame)

    assert list(estimated_frame.columns) == ESTIMATE_COLUMNS
    assert set(estimated_frame["long_regime"]).issubset(LONG_REGIMES)
    assert set(estimated_frame["short_regime"]).issubset(SHORT_REGIMES)
    assert set(estimated_frame["volatility_state"]).issubset(VOLATILITY_STATES)
    assert set(estimated_frame["drawdown_state"]).issubset(DRAW_DOWN_STATES)
    assert set(estimated_frame["risk_state"]).issubset(RISK_STATES)
    assert estimated_frame["confidence_score"].between(0.0, 1.0).all()


def test_v3_market_estimator_rejects_bad_features() -> None:
    with pytest.raises(ValueError, match="missing required fields"):
        estimate_market({"timestamp": "x", "close": 100.0})

    bad = feature(close=float("nan"))
    with pytest.raises(ValueError, match="close must be finite"):
        estimate_market(bad)

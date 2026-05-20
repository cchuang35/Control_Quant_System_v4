import numpy as np
import pandas as pd
import pytest

import v2_small_cap as v2sc
from analyze_results import analyze_backtest_output
from backtester import run_backtest_fast
from backtester import run_backtest as run_layer_backtester
from backtester import write_backtest_outputs
from download_data import kline_to_row, parse_datetime
from src.backtest import run_backtest
from src.layer1_market_model import MarketStateV1, OHLCVBar, build_market_state, build_market_state_frame
from src.layer2_state_estimator import EstimatedMarketStateV1, estimate_market_state, estimate_market_state_frame
from src.layer3_strategy_controller import ControlActionV1, PortfolioStateV1, compute_control_action
from src.layer4_risk_filter import RiskConfigV1, apply_risk_filter
from src.layer5_adaptive_supervisor import SystemHistoryV1, supervise_adaptation
from v2_small_cap import (
    apply_drawdown_risk_gate,
    apply_trade_gate,
    backtest_v21_small_cap,
    backtest_v22_small_cap,
    backtest_v23_small_cap,
    backtest_v24_small_cap,
    backtest_v25_small_cap,
    backtest_v26_regime_quality_small_cap,
    backtest_v27_weak_momentum_small_cap,
    backtest_v28_weak_bull_control_small_cap,
    backtest_v2_candidate_2,
    backtest_v2_btc_final_candidate_a,
    backtest_v2_final_candidate_a,
    backtest_v2_small_cap,
    build_performance_summary_table,
    build_regime_diagnostics,
    calculate_performance_stats,
    compute_soft_dd_scale,
    compute_regime_features,
    confirm_regime,
    decide_v21_position,
    regime_entry_hold_permissions,
)


def sample_bars(count: int = 160) -> list[OHLCVBar]:
    bars = []
    close = 100.0
    for idx in range(count):
        close *= 1.0 + 0.001
        bars.append(
            OHLCVBar(
                timestamp=float(idx),
                open=close * 0.999,
                high=close * 1.01,
                low=close * 0.99,
                close=close,
                volume=1000.0 + idx % 10,
            )
        )
    return bars


def sample_ohlcv_frame(count: int = 220) -> pd.DataFrame:
    close = 100.0
    rows = []
    for idx in range(count):
        drift = 0.001 if idx < count // 2 else -0.0005
        shock = -0.04 if idx == count - 20 else 0.0
        close *= 1.0 + drift + shock
        rows.append(
            {
                "timestamp": float(idx),
                "open": close * 0.998,
                "high": close * (1.012 + (0.015 if idx == count - 20 else 0.0)),
                "low": close * (0.988 - (0.015 if idx == count - 20 else 0.0)),
                "close": close,
                "volume": 1000.0 + (idx % 15) * 10.0,
            }
        )
    return pd.DataFrame(rows)


def assert_unit_interval(value: float) -> None:
    assert 0.0 <= value <= 1.0


def assert_no_nan_or_inf(frame: pd.DataFrame) -> None:
    numeric = frame.select_dtypes(include=[np.number])
    assert not numeric.isna().any().any()
    assert np.isfinite(numeric.to_numpy()).all()


def make_market_state(
    *,
    trend_score: float = 0.2,
    volatility_score: float = 1.0,
    liquidity_score: float = 0.8,
    shock_score: float = 0.1,
    confidence: float = 0.8,
    drawdown: float = 0.0,
) -> MarketStateV1:
    return MarketStateV1(
        timestamp=1.0,
        close=100.0,
        return_1=0.0,
        volatility=0.01,
        volatility_score=volatility_score,
        trend_raw=0.02,
        trend_score=trend_score,
        volume_z=0.0,
        volume_score=0.1,
        price_range=0.02,
        liquidity_score=liquidity_score,
        drawdown=drawdown,
        shock_score=shock_score,
        confidence=confidence,
        market_mode="normal",
    )


def make_estimated_state(
    *,
    p_bull: float = 0.6,
    p_bear: float = 0.1,
    p_sideways: float = 0.2,
    p_high_vol: float = 0.05,
    p_crash_risk: float = 0.05,
    state_confidence: float = 0.8,
    transition_risk: float = 0.1,
    danger_score: float = 0.2,
) -> EstimatedMarketStateV1:
    return EstimatedMarketStateV1(
        p_bull=p_bull,
        p_bear=p_bear,
        p_sideways=p_sideways,
        p_high_vol=p_high_vol,
        p_crash_risk=p_crash_risk,
        dominant_regime="bull",
        state_confidence=state_confidence,
        regime_uncertainty=0.3,
        transition_risk=transition_risk,
        danger_score=danger_score,
    )


def test_layer1_outputs_are_in_expected_ranges() -> None:
    state = build_market_state(sample_bars())
    assert -1.0 <= state.trend_score <= 1.0
    assert_unit_interval(state.volume_score)
    assert_unit_interval(state.liquidity_score)
    assert state.drawdown <= 0.0
    assert_unit_interval(state.shock_score)
    assert_unit_interval(state.confidence)
    assert state.market_mode in {"normal", "trending_up", "trending_down", "high_volatility", "stressed", "shock"}


def test_layer1_dataframe_outputs_are_bounded_without_nan_or_inf() -> None:
    states = build_market_state_frame(sample_ohlcv_frame())
    assert_no_nan_or_inf(states)
    assert states["return_1"].map(np.isfinite).all()
    assert states["volatility_score"].ge(0.0).all()
    assert states["trend_score"].between(-1.0, 1.0).all()
    assert states["volume_score"].between(0.0, 1.0).all()
    assert states["liquidity_score"].between(0.0, 1.0).all()
    assert states["drawdown"].le(0.0).all()
    assert states["shock_score"].between(0.0, 1.0).all()
    assert states["confidence"].between(0.0, 1.0).all()
    assert set(states["market_mode"]).issubset({"normal", "trending_up", "trending_down", "high_volatility", "stressed", "shock"})


def test_layer2_probabilities_and_scores_are_bounded() -> None:
    market = build_market_state(sample_bars())
    estimated = estimate_market_state(market)
    total_probability = estimated.p_bull + estimated.p_bear + estimated.p_sideways + estimated.p_high_vol + estimated.p_crash_risk
    assert abs(total_probability - 1.0) < 1e-9
    assert estimated.dominant_regime in {"bull", "bear", "sideways", "high_vol", "crash_risk"}
    assert_unit_interval(estimated.state_confidence)
    assert_unit_interval(estimated.regime_uncertainty)
    assert_unit_interval(estimated.transition_risk)
    assert_unit_interval(estimated.danger_score)


def test_layer2_dataframe_probabilities_scores_and_finiteness() -> None:
    market_states = build_market_state_frame(sample_ohlcv_frame())
    estimated = estimate_market_state_frame(market_states)
    assert_no_nan_or_inf(estimated)

    probability_columns = ["p_bull", "p_bear", "p_sideways", "p_high_vol", "p_crash_risk"]
    probability_sum = estimated[probability_columns].sum(axis=1)
    assert np.allclose(probability_sum.to_numpy(), 1.0, atol=1e-9)
    assert estimated[probability_columns].ge(0.0).all().all()
    assert estimated[probability_columns].le(1.0).all().all()
    assert set(estimated["dominant_regime"]).issubset({"bull", "bear", "sideways", "high_vol", "crash_risk"})
    assert estimated["state_confidence"].between(0.0, 1.0).all()
    assert estimated["regime_uncertainty"].between(0.0, 1.0).all()
    assert estimated["transition_risk"].between(0.0, 1.0).all()
    assert estimated["danger_score"].between(0.0, 1.0).all()


def test_layer3_control_action_is_bounded() -> None:
    market = build_market_state(sample_bars())
    estimated = estimate_market_state(market)
    portfolio = PortfolioStateV1(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    action = compute_control_action(market, estimated, portfolio)
    assert -1.0 <= action.target_exposure <= 1.0
    assert -1.0 <= action.exposure_change <= 1.0
    assert_unit_interval(action.max_leverage)
    assert 0.05 <= action.rebalance_speed <= 0.8


def test_layer3_reduce_only_does_not_increase_absolute_exposure() -> None:
    market = make_market_state(shock_score=0.85)
    estimated = make_estimated_state(danger_score=0.8, p_crash_risk=0.55)
    portfolio = PortfolioStateV1(0.4, 0.0, 1.0, 1.0, 0.0, 0.0, 0.4)
    action = compute_control_action(market, estimated, portfolio)
    assert action.reduce_only
    assert abs(action.target_exposure) <= abs(portfolio.current_exposure)
    assert abs(portfolio.current_exposure + action.exposure_change) <= abs(portfolio.current_exposure)


def test_layer4_safe_action_respects_limits() -> None:
    market = build_market_state(sample_bars())
    estimated = estimate_market_state(market)
    portfolio = PortfolioStateV1(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    action = compute_control_action(market, estimated, portfolio)
    safe = apply_risk_filter(market, estimated, portfolio, action)
    assert abs(safe.safe_target_exposure) <= safe.allowed_max_exposure
    assert abs(safe.safe_exposure_change) <= safe.allowed_turnover
    assert_unit_interval(safe.allowed_max_exposure)
    assert_unit_interval(safe.allowed_max_leverage)
    assert 0.01 <= safe.allowed_turnover <= 0.20


def test_layer4_reduce_only_does_not_increase_absolute_exposure() -> None:
    market = make_market_state(shock_score=0.85)
    estimated = make_estimated_state(danger_score=0.8, p_crash_risk=0.55)
    portfolio = PortfolioStateV1(0.5, 0.0, 1.0, 1.0, 0.0, 0.0, 0.5)
    action = ControlActionV1(
        target_exposure=0.9,
        exposure_change=0.4,
        max_leverage=0.3,
        rebalance_speed=0.2,
        trade_allowed=True,
        reduce_only=True,
        action_type="reduce",
        reason_code="reduce_only",
    )
    safe = apply_risk_filter(market, estimated, portfolio, action)
    post_exposure = portfolio.current_exposure + safe.safe_exposure_change
    assert safe.reduce_only
    assert abs(safe.safe_target_exposure) <= abs(portfolio.current_exposure)
    assert abs(post_exposure) <= abs(portfolio.current_exposure)


def test_layer4_kill_switch_targets_zero_exposure() -> None:
    market = make_market_state(shock_score=0.95, liquidity_score=0.1)
    estimated = make_estimated_state(danger_score=0.95, p_crash_risk=0.75)
    portfolio = PortfolioStateV1(0.6, 0.0, 1.0, 1.0, 0.0, -0.21, 0.6)
    action = ControlActionV1(
        target_exposure=0.6,
        exposure_change=0.0,
        max_leverage=0.2,
        rebalance_speed=0.2,
        trade_allowed=True,
        reduce_only=False,
        action_type="rebalance",
        reason_code="normal_trade",
    )
    safe = apply_risk_filter(market, estimated, portfolio, action)
    assert safe.kill_switch
    assert safe.emergency_deleveraging
    assert safe.reduce_only
    assert safe.safe_target_exposure == 0.0
    assert safe.safe_exposure_change <= 0.0


def test_layer4_direction_flip_requires_stability() -> None:
    market = make_market_state(trend_score=-0.6)
    estimated = make_estimated_state(
        p_bull=0.05,
        p_bear=0.7,
        p_sideways=0.1,
        p_high_vol=0.1,
        p_crash_risk=0.05,
        state_confidence=0.5,
        transition_risk=0.5,
        danger_score=0.2,
    )
    portfolio = PortfolioStateV1(0.5, 0.0, 1.0, 1.0, 0.0, 0.0, 0.5)
    action = ControlActionV1(
        target_exposure=-0.6,
        exposure_change=-1.1,
        max_leverage=0.8,
        rebalance_speed=0.8,
        trade_allowed=True,
        reduce_only=False,
        action_type="rebalance",
        reason_code="normal_trade",
    )
    safe = apply_risk_filter(market, estimated, portfolio, action)
    assert safe.safe_target_exposure == 0.0
    assert safe.safe_exposure_change < 0.0
    assert abs(safe.safe_exposure_change) <= safe.allowed_turnover


def test_layer4_min_trade_threshold_suppresses_small_changes() -> None:
    market = make_market_state()
    estimated = make_estimated_state()
    portfolio = PortfolioStateV1(0.5, 0.0, 1.0, 1.0, 0.0, 0.0, 0.5)
    action = ControlActionV1(
        target_exposure=0.52,
        exposure_change=0.02,
        max_leverage=0.8,
        rebalance_speed=0.2,
        trade_allowed=True,
        reduce_only=False,
        action_type="rebalance",
        reason_code="normal_trade",
    )
    safe = apply_risk_filter(market, estimated, portfolio, action, config=RiskConfigV1(min_trade_threshold=0.03))
    assert safe.safe_exposure_change == 0.0
    assert not safe.trade_allowed


def test_layer5_adaptive_update_is_bounded() -> None:
    history = SystemHistoryV1(
        strategy_returns=[0.001, -0.001, 0.002, 0.0],
        drawdowns=[0.0, -0.01, -0.02],
        regime_uncertainties=[0.2, 0.3],
        state_confidences=[0.8, 0.7],
        shock_scores=[0.1, 0.2],
        turnovers=[0.05, 0.04],
        transaction_costs=[0.0001, 0.0001],
        intervention_flags=[False, True, False],
    )
    update = supervise_adaptation(history)
    assert_unit_interval(update.model_health_score)
    assert_unit_interval(update.strategy_health_score)
    assert_unit_interval(update.overfit_risk_score)
    assert update.adaptation_mode in {"normal", "cautious", "defensive", "retrain_required", "disabled"}
    assert 0.5 <= update.new_softmax_temperature <= 1.5
    assert 0.05 <= update.new_smoothing_alpha <= 0.40
    assert 0.02 <= update.new_base_turnover_limit <= 0.25


def test_minimal_backtest_runs() -> None:
    result = run_backtest(sample_bars())
    assert result["bars"] == 160
    assert result["final_equity"] > 0.0
    assert -1.0 <= result["final_exposure"] <= 1.0


def test_layer_backtester_outputs_metrics_and_histories_without_lookahead_shape() -> None:
    frame = sample_ohlcv_frame(180)
    result = run_layer_backtester(frame, fee_rate=0.0005, periods_per_year=252)

    assert result.metrics["final_equity"] > 0.0
    assert -1.0 <= result.metrics["total_return"]
    assert result.metrics["trade_count"] >= 0
    assert result.metrics["turnover"] >= 0.0
    assert result.metrics["average_exposure"] >= 0.0
    assert 0.0 <= result.metrics["layer4_intervention_rate"] <= 1.0
    assert 0.0 <= result.metrics["minor_intervention_rate"] <= 1.0
    assert 0.0 <= result.metrics["target_clip_rate"] <= 1.0
    assert 0.0 <= result.metrics["turnover_clip_rate"] <= 1.0
    assert 0.0 <= result.metrics["hard_intervention_rate"] <= 1.0
    assert result.metrics["kill_switch_count"] >= 0
    assert result.metrics["reduce_only_count"] >= 0
    assert result.metrics["cooldown_blocked_trade_count"] >= 0

    assert len(result.equity_curve) == len(frame)
    assert len(result.exposure_history) == len(frame) - 1
    assert len(result.market_state_history) == len(frame) - 1
    assert len(result.estimated_state_history) == len(frame) - 1
    assert len(result.control_action_history) == len(frame) - 1
    assert len(result.safe_control_action_history) == len(frame) - 1
    assert result.exposure_history["decision_timestamp"].iloc[0] == frame["timestamp"].iloc[0]
    assert result.exposure_history["execution_timestamp"].iloc[0] == frame["timestamp"].iloc[1]

    assert_no_nan_or_inf(result.equity_curve)
    assert_no_nan_or_inf(result.exposure_history)
    assert_no_nan_or_inf(result.market_state_history)
    assert_no_nan_or_inf(result.estimated_state_history)
    assert_no_nan_or_inf(result.control_action_history)
    assert_no_nan_or_inf(result.safe_control_action_history)


def test_fast_layer_backtester_matches_standard_backtester_on_key_outputs() -> None:
    frame = sample_ohlcv_frame(220)
    standard = run_layer_backtester(frame, fee_rate=0.001, periods_per_year=365)
    fast = run_backtest_fast(frame, fee_rate=0.001, periods_per_year=365)

    assert len(fast.equity_curve) == len(standard.equity_curve)
    assert len(fast.exposure_history) == len(standard.exposure_history)
    for key in ["final_equity", "total_return", "max_drawdown", "turnover", "average_exposure"]:
        assert fast.metrics[key] == pytest.approx(standard.metrics[key], rel=0.02, abs=1e-6)


def test_analyze_results_writes_reports(tmp_path) -> None:
    frame = sample_ohlcv_frame(120)
    result = run_layer_backtester(frame, fee_rate=0.0005, periods_per_year=252)
    output_dir = tmp_path / "backtest_output"
    reports_dir = tmp_path / "reports"
    write_backtest_outputs(result, output_dir)

    tables = analyze_backtest_output(output_dir, reports_dir)

    assert not tables["performance_summary"].empty
    assert not tables["regime_performance"].empty
    expected_files = {
        "performance_summary.csv",
        "regime_performance.csv",
        "equity_curve.png",
        "drawdown_curve.png",
        "exposure_over_time.png",
        "dominant_regime_over_time.png",
        "danger_score_over_time.png",
        "p_crash_risk_over_time.png",
    }
    assert expected_files.issubset({path.name for path in reports_dir.iterdir()})


def test_download_data_converts_binance_kline_to_required_schema() -> None:
    row = kline_to_row(
        [
            1704067200000,
            "42280.00",
            "43100.00",
            "42100.00",
            "43000.00",
            "123.45",
            1704070799999,
            "0",
            0,
            "0",
            "0",
            "0",
        ]
    )
    assert list(row.keys()) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert row["timestamp"] == "2024-01-01T00:00:00Z"
    assert row["open"] == 42280.0
    assert row["high"] == 43100.0
    assert row["low"] == 42100.0
    assert row["close"] == 43000.0
    assert row["volume"] == 123.45


def test_download_data_parse_date_defaults_to_utc_midnight() -> None:
    assert parse_datetime("2024-01-01").isoformat() == "2024-01-01T00:00:00+00:00"


def test_v2_regime_features_and_confirmation_are_shape_stable() -> None:
    frame = pd.DataFrame({"close": np.linspace(100.0, 130.0, 80), "current_exposure": 0.6})
    features = compute_regime_features(frame)
    confirmed = confirm_regime(pd.Series(["sideways", "bear", "bear", "bear"]), confirmation_days=3)

    expected_columns = {
        "asset_return",
        "MA20",
        "MA60",
        "momentum_20",
        "vol20",
        "vol60",
        "raw_regime",
        "confirmed_regime",
        "regime_score",
    }
    assert expected_columns.issubset(features.columns)
    assert not features[["raw_regime", "confirmed_regime", "regime_score"]].isna().any().any()
    assert set(features["raw_regime"]).issubset({"strong_bull", "weak_bull", "sideways", "bear"})
    assert set(features["confirmed_regime"]).issubset({"strong_bull", "weak_bull", "sideways", "bear"})
    assert confirmed.tolist() == ["sideways", "sideways", "sideways", "bear"]


def test_v2_trade_and_drawdown_gates_are_discrete() -> None:
    assert apply_trade_gate(1, "strong_bull", 0) == 1
    assert apply_trade_gate(1, "sideways", 1) == 1
    assert apply_trade_gate(1, "sideways", 0) == 0
    assert apply_trade_gate(1, "bear", 1) == 0
    assert apply_trade_gate(1, "unknown", 1) == 0
    assert apply_trade_gate(1, "sideways", 1, gate_mode="strict_sideways") == 0
    assert apply_trade_gate(1, "weak_bull", 1, gate_mode="strict_sideways") == 1
    with pytest.raises(ValueError, match="v1_position must be 0 or 1"):
        apply_trade_gate(0.5, "strong_bull", 0)

    assert apply_drawdown_risk_gate(1, "weak_bull", -0.09) == 1
    assert apply_drawdown_risk_gate(1, "weak_bull", -0.12) == 0
    assert apply_drawdown_risk_gate(1, "strong_bull", -0.12) == 1
    assert apply_drawdown_risk_gate(1, "strong_bull", -0.16) == 0
    with pytest.raises(ValueError, match="position must be 0 or 1"):
        apply_drawdown_risk_gate(0.5, "strong_bull", 0.0)


def test_v21_regime_gate_uses_entry_hold_permissions() -> None:
    assert regime_entry_hold_permissions("strong_bull") == (True, True)
    assert regime_entry_hold_permissions("weak_bull") == (True, True)
    assert regime_entry_hold_permissions("sideways") == (False, True)
    assert regime_entry_hold_permissions("bear") == (False, False)

    assert decide_v21_position(previous_position=0, v1_position_signal=1, confirmed_regime="weak_bull") == 1
    assert decide_v21_position(previous_position=0, v1_position_signal=1, confirmed_regime="sideways") == 0
    assert decide_v21_position(previous_position=1, v1_position_signal=1, confirmed_regime="sideways") == 1
    assert decide_v21_position(previous_position=1, v1_position_signal=1, confirmed_regime="bear") == 0
    assert decide_v21_position(previous_position=1, v1_position_signal=0, confirmed_regime="weak_bull") == 0
    assert decide_v21_position(
        previous_position=0,
        v1_position_signal=1,
        confirmed_regime="weak_bull",
        allow_new_entries=False,
    ) == 0


def test_v21_drawdown_gate_forces_exit_and_cooldown_without_lookahead() -> None:
    close = [100.0]
    for _ in range(75):
        close.append(close[-1] * 1.01)
    close.extend([close[-1] * 0.90, close[-1] * 0.91, close[-1] * 0.92, close[-1] * 1.01])
    v1_position = [1] * len(close)
    v1_position[76] = 0
    frame = pd.DataFrame({"close": close, "v1_position": v1_position})
    result = backtest_v21_small_cap(
        frame,
        fee_rate=0.0,
        use_drawdown_gate=True,
        warning_drawdown=-0.03,
        exit_drawdown=-0.05,
        cooldown_bars=2,
        confirmation_days=1,
    )

    trigger_rows = result.index[result["drawdown_gate_triggered"]]
    assert len(trigger_rows) >= 1
    trigger_idx = int(trigger_rows[0])
    assert result["final_position"].iloc[trigger_idx] == 0
    assert result["cooldown_active"].iloc[trigger_idx]
    assert result["final_position"].iloc[trigger_idx + 1] == 0
    assert result["strategy_return_gross"].iloc[trigger_idx] == result["final_position"].shift(1).fillna(0).iloc[trigger_idx] * result["asset_return"].iloc[trigger_idx]


def test_v22_soft_drawdown_scale_modes() -> None:
    assert compute_soft_dd_scale(-0.01, mode="step") == 1.0
    assert compute_soft_dd_scale(-0.04, mode="step") == 0.5
    assert compute_soft_dd_scale(-0.06, mode="step") == 0.0
    assert compute_soft_dd_scale(-0.04, mode="linear") == pytest.approx(0.5)
    assert compute_soft_dd_scale(-0.035, mode="linear") == pytest.approx(0.75)
    with pytest.raises(ValueError, match="mode must be"):
        compute_soft_dd_scale(-0.04, mode="bad")


def test_v22_soft_drawdown_scaling_keeps_entries_available() -> None:
    close = [100.0]
    for _ in range(75):
        close.append(close[-1] * 1.01)
    close.extend([close[-1] * 0.965, close[-1] * 1.01, close[-1] * 1.01])
    frame = pd.DataFrame({"close": close, "v1_position": 1})

    result = backtest_v22_small_cap(
        frame,
        fee_rate=0.0,
        warning_dd=-0.03,
        hard_dd=-0.05,
        dd_scale_mode="step",
        confirmation_days=1,
    )

    scaled_rows = result.index[(result["dd_scale"] < 1.0) & (result["dd_scale"] > 0.0)]
    assert len(scaled_rows) >= 1
    scaled_idx = int(scaled_rows[0])
    assert result["position_before_dd_gate"].iloc[scaled_idx] == 1
    assert result["final_position"].iloc[scaled_idx] == 0.5
    assert not result["entries_disabled_by_dd"].any()


def test_v23_hysteresis_warning_mode_and_strong_reentry() -> None:
    close = [100.0]
    for _ in range(75):
        close.append(close[-1] * 1.01)
    close.append(close[-1] * 0.96)
    close.append(close[-1])
    close.append(close[-1])
    close.append(close[-1] * 1.002)
    close.append(close[-1] * 1.002)
    close.append(close[-1] * 1.01)
    v1_position = [1] * len(close)
    v1_position[76] = 0
    frame = pd.DataFrame({"close": close, "v1_position": v1_position})

    hysteresis = backtest_v23_small_cap(
        frame,
        fee_rate=0.0,
        warning_dd=-0.03,
        hard_dd=-0.20,
        recovery_dd=-0.02,
        cooldown_bars=2,
        confirmation_days=1,
    )
    warning_rows = hysteresis.index[hysteresis["warning_mode_active"]]
    assert len(warning_rows) >= 2
    assert not hysteresis["hard_stop"].any()
    assert hysteresis["entries_disabled_by_dd"].any()

    strong_reentry = backtest_v23_small_cap(
        frame,
        fee_rate=0.0,
        warning_dd=-0.03,
        hard_dd=-0.20,
        cooldown_bars=2,
        warning_entry_mode="strong_only",
        confirmation_days=1,
    )
    warning_strong_rows = strong_reentry.index[
        strong_reentry["warning_mode_active"] & (strong_reentry["confirmed_regime"] == "strong_bull")
    ]
    if len(warning_strong_rows) > 0:
        assert (strong_reentry.loc[warning_strong_rows, "entries_disabled_by_dd"] == False).all()


def test_v24_pause_gate_is_event_based_and_does_not_force_exit() -> None:
    close = [100.0]
    for _ in range(75):
        close.append(close[-1] * 1.01)
    close.extend([close[-1] * 0.96, close[-1], close[-1], close[-1] * 1.002, close[-1] * 1.002])
    v1_position = [1] * len(close)
    v1_position[77] = 0
    frame = pd.DataFrame({"close": close, "v1_position": v1_position})

    result = backtest_v24_small_cap(
        frame,
        fee_rate=0.0,
        pause_dd=-0.03,
        pause_bars=3,
        exit_dd=None,
        confirmation_days=1,
    )

    trigger_rows = result.index[result["pause_trigger"]]
    assert len(trigger_rows) == 1
    trigger_idx = int(trigger_rows[0])
    assert result["final_position"].iloc[trigger_idx] == 1
    assert result["pause_active"].iloc[trigger_idx]
    assert result["final_position"].iloc[trigger_idx + 1] == 0
    assert result["entries_disabled_by_dd"].iloc[trigger_idx + 2]
    assert result["final_position"].iloc[trigger_idx + 2] == 0


def test_v25_trade_stop_and_trailing_stop_force_next_position_exit() -> None:
    close = [100.0]
    for _ in range(75):
        close.append(close[-1] * 1.01)
    close.extend([close[-1] * 1.01, close[-1] * 0.80, close[-1] * 1.01])
    frame = pd.DataFrame({"close": close, "v1_position": 1})

    trade_stop = backtest_v25_small_cap(
        frame,
        fee_rate=0.0,
        stop_loss=-0.03,
        confirmation_days=1,
    )
    stop_rows = trade_stop.index[trade_stop["stop_exit"]]
    assert len(stop_rows) >= 1
    stop_idx = int(stop_rows[0])
    assert trade_stop["final_position"].iloc[stop_idx] == 0
    assert trade_stop["strategy_return_gross"].iloc[stop_idx] == trade_stop["asset_return"].iloc[stop_idx]

    trailing = backtest_v25_small_cap(
        frame,
        fee_rate=0.0,
        trailing_stop=-0.03,
        confirmation_days=1,
    )
    trailing_rows = trailing.index[trailing["trailing_exit"]]
    assert len(trailing_rows) >= 1
    trailing_idx = int(trailing_rows[0])
    assert trailing["final_position"].iloc[trailing_idx] == 0
    assert trailing["trade_peak_price"].shift(1).iloc[trailing_idx] >= trailing["entry_price"].shift(1).iloc[trailing_idx]


def test_v26_quality_filters_block_weak_entries_and_exit_deteriorating_sideways() -> None:
    close = [100.0]
    for _ in range(100):
        close.append(close[-1] * 1.0008)
    close.extend([close[-1] * 0.98, close[-1] * 0.98])
    v1_position = [0] * 80 + [1] * (len(close) - 80)
    frame = pd.DataFrame({"close": close, "v1_position": v1_position})

    weak_filter = backtest_v26_regime_quality_small_cap(
        frame,
        fee_rate=0.0,
        weak_entry_filter="mom_2",
        confirmation_days=1,
    )
    assert weak_filter["weak_entry_blocked"].any()
    first_blocked = int(weak_filter.index[weak_filter["weak_entry_blocked"]][0])
    assert weak_filter["confirmed_regime"].iloc[first_blocked] == "weak_bull"
    assert weak_filter["final_position"].iloc[first_blocked] == 0

    sideways_exit = backtest_v26_regime_quality_small_cap(
        frame,
        fee_rate=0.0,
        sideways_exit_filter="ma20",
        confirmation_days=1,
    )
    exit_rows = sideways_exit.index[sideways_exit["sideways_exit"]]
    if len(exit_rows) > 0:
        exit_idx = int(exit_rows[0])
        assert sideways_exit["confirmed_regime"].iloc[exit_idx] == "sideways"
        assert sideways_exit["final_position"].iloc[exit_idx] == 0


def test_v27_weak_momentum_threshold_sweep_tracks_entry_attempts() -> None:
    close = [100.0]
    for _ in range(100):
        close.append(close[-1] * 1.0008)
    v1_position = [0] * 80 + [1] * (len(close) - 80)
    frame = pd.DataFrame({"close": close, "v1_position": v1_position})

    loose = backtest_v27_weak_momentum_small_cap(
        frame,
        fee_rate=0.0,
        weak_momentum_threshold=0.005,
        confirmation_days=1,
    )
    strict = backtest_v27_weak_momentum_small_cap(
        frame,
        fee_rate=0.0,
        weak_momentum_threshold=0.02,
        confirmation_days=1,
    )

    assert loose["weak_bull_entry_attempt"].sum() >= 1
    assert loose["weak_bull_entry_allowed"].sum() >= 1
    assert strict["weak_bull_entry_blocked"].sum() >= loose["weak_bull_entry_blocked"].sum()


def test_v28_weak_bull_controls_block_only_targeted_weak_entries() -> None:
    close = [100.0]
    for _ in range(70):
        close.append(close[-1] * 1.001)
    close.extend([close[-1] * 0.99, close[-1] * 1.001, close[-1] * 1.001])
    close.extend([close[-1] * 1.001 for _ in range(30)])
    v1_position = [0] * len(close)
    for idx in range(60, 72):
        v1_position[idx] = 1
    for idx in range(73, len(close)):
        v1_position[idx] = 1
    frame = pd.DataFrame({"close": close, "v1_position": v1_position})

    cooldown = backtest_v28_weak_bull_control_small_cap(
        frame,
        fee_rate=0.0,
        weak_loss_cooldown_bars=24,
        confirmation_days=1,
    )
    assert cooldown["weak_bull_entry_attempt"].sum() >= 1

    confirm = backtest_v28_weak_bull_control_small_cap(
        frame,
        fee_rate=0.0,
        weak_confirm_bars=24,
        confirmation_days=1,
    )
    assert confirm["weak_bull_entry_blocked"].sum() >= 1
    blocked = confirm[confirm["weak_bull_entry_blocked"]]
    assert (blocked["confirmed_regime"] == "weak_bull").all()


def test_v2_candidate_2_wraps_weak_loss_cooldown_120() -> None:
    frame = pd.DataFrame({"close": np.linspace(100.0, 130.0, 140), "v1_position": 1})
    candidate = backtest_v2_candidate_2(frame, fee_rate=0.001, confirmation_days=1)
    direct = backtest_v28_weak_bull_control_small_cap(
        frame,
        fee_rate=0.001,
        weak_loss_cooldown_bars=120,
        confirmation_days=1,
    )

    pd.testing.assert_series_equal(candidate["final_position"], direct["final_position"])
    pd.testing.assert_series_equal(candidate["strategy_return_net"], direct["strategy_return_net"])


def test_v2_final_candidate_a_supports_cooldown_range() -> None:
    frame = pd.DataFrame({"close": np.linspace(100.0, 130.0, 140), "v1_position": 1})
    candidate = backtest_v2_final_candidate_a(frame, fee_rate=0.001, cooldown_bars=144, confirmation_days=1)
    direct = backtest_v28_weak_bull_control_small_cap(
        frame,
        fee_rate=0.001,
        weak_loss_cooldown_bars=144,
        confirmation_days=1,
    )

    pd.testing.assert_series_equal(candidate["final_position"], direct["final_position"])
    pd.testing.assert_series_equal(candidate["weak_bull_cooldown_active"], direct["weak_bull_cooldown_active"])


def test_btc_final_candidate_a_freezes_wrapper_name() -> None:
    frame = pd.DataFrame({"close": np.linspace(100.0, 130.0, 140), "v1_position": 1})
    frozen = backtest_v2_btc_final_candidate_a(frame, fee_rate=0.001, cooldown_bars=120, confirmation_days=1)
    candidate = backtest_v2_final_candidate_a(frame, fee_rate=0.001, cooldown_bars=120, confirmation_days=1)

    pd.testing.assert_series_equal(frozen["final_position"], candidate["final_position"])
    pd.testing.assert_series_equal(frozen["strategy_return_net"], candidate["strategy_return_net"])


def test_btc_final_candidate_a_cooldown_scope_fee_and_lag(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "close": [100.0, 98.0, 99.0, 105.0, 104.0, 106.0, 103.0],
            "v1_position": [1, 0, 1, 1, 1, 1, 1],
            "confirmed_regime": [
                "weak_bull",
                "weak_bull",
                "weak_bull",
                "strong_bull",
                "sideways",
                "sideways",
                "bear",
            ],
        }
    )

    def fake_regime_features(df: pd.DataFrame, confirmation_days: int = 3) -> pd.DataFrame:
        result = df.copy()
        result["asset_return"] = result["close"].pct_change()
        result["MA20"] = result["close"]
        result["MA60"] = result["close"]
        result["momentum_20"] = 0.01
        result["vol20"] = 0.0
        result["vol60"] = 0.0
        result["raw_regime"] = result["confirmed_regime"]
        result["regime_score"] = result["confirmed_regime"].map(v2sc.REGIME_SCORE)
        return result

    monkeypatch.setattr(v2sc, "compute_regime_features", fake_regime_features)
    result = v2sc.backtest_v2_btc_final_candidate_a(
        frame,
        fee_rate=0.001,
        cooldown_bars=120,
        confirmation_days=1,
    )

    assert set(result["final_position"]).issubset({0, 1})
    assert result["final_position"].tolist() == [1, 0, 0, 1, 1, 1, 0]
    assert result["weak_bull_cooldown_trigger"].iloc[1]
    assert result["weak_bull_entry_blocked"].iloc[2]
    assert result["confirmed_regime"].iloc[2] == "weak_bull"
    assert result["final_position"].iloc[3] == 1
    assert result["confirmed_regime"].iloc[3] == "strong_bull"
    assert result["final_position"].iloc[4] == 1
    assert result["confirmed_regime"].iloc[4] == "sideways"
    assert result["final_position"].iloc[5] == 1
    assert result["final_position"].iloc[6] == 0
    assert result["confirmed_regime"].iloc[6] == "bear"

    expected_trade_size = result["final_position"].diff().abs().fillna(result["final_position"].abs())
    pd.testing.assert_series_equal(result["trade_size"], expected_trade_size.astype(float), check_names=False)
    pd.testing.assert_series_equal(result["fee_cost"], expected_trade_size.astype(float) * 0.001, check_names=False)
    expected_gross = result["final_position"].shift(1).fillna(0) * result["asset_return"].fillna(0.0)
    pd.testing.assert_series_equal(result["strategy_return_gross"], expected_gross.astype(float), check_names=False)


def test_sideways_and_bear_base_permissions_are_frozen() -> None:
    assert regime_entry_hold_permissions("sideways") == (False, True)
    assert regime_entry_hold_permissions("bear") == (False, False)
    assert decide_v21_position(previous_position=0, v1_position_signal=1, confirmed_regime="sideways") == 0
    assert decide_v21_position(previous_position=1, v1_position_signal=1, confirmed_regime="sideways") == 1
    assert decide_v21_position(previous_position=1, v1_position_signal=1, confirmed_regime="bear") == 0


def test_v2_small_cap_backtest_builds_required_columns_from_v1_exposure() -> None:
    close = pd.Series(np.linspace(100.0, 150.0, 100))
    frame = pd.DataFrame(
        {
            "close": close,
            "current_exposure": [0.2] * 10 + [0.7] * 80 + [0.0] * 10,
        }
    )
    result = backtest_v2_small_cap(frame, fee_rate=0.001, use_drawdown_gate=True)

    expected_columns = {
        "close",
        "asset_return",
        "v1_position",
        "MA20",
        "MA60",
        "momentum_20",
        "vol20",
        "vol60",
        "raw_regime",
        "confirmed_regime",
        "regime_score",
        "position_before_dd_gate",
        "final_position",
        "trade_size",
        "fee_cost",
        "strategy_return_gross",
        "strategy_return_net",
        "equity_net",
        "equity_peak",
        "strategy_drawdown",
        "drawdown",
    }
    assert expected_columns.issubset(result.columns)
    assert set(result["v1_position"]).issubset({0, 1})
    assert set(result["final_position"]).issubset({0, 1})
    assert result["v1_position"].iloc[0] == 0
    assert result["v1_position"].iloc[20] == 1
    assert_no_nan_or_inf(
        result[
            [
                "v1_position",
                "regime_score",
                "position_before_dd_gate",
                "final_position",
                "trade_size",
                "fee_cost",
                "strategy_return_gross",
                "strategy_return_net",
                "equity_net",
                "equity_peak",
                "strategy_drawdown",
                "drawdown",
            ]
        ]
    )


def test_v2_fee_aware_backtest_uses_lagged_position_return() -> None:
    frame = pd.DataFrame({"close": np.linspace(100.0, 180.0, 100), "v1_position": 1})
    result = backtest_v2_small_cap(
        frame,
        fee_rate=0.001,
        use_drawdown_gate=False,
        confirmation_days=1,
        gate_mode="sideways_hold",
    )

    entry_idx = int(result.index[result["trade_size"] > 0.0][0])
    assert result["strategy_return_gross"].iloc[entry_idx] == 0.0
    assert result["fee_cost"].iloc[entry_idx] == 0.001
    assert result["strategy_return_gross"].iloc[entry_idx + 1] == result["asset_return"].iloc[entry_idx + 1]


def test_v2_fee_aware_backtest_matches_vectorized_formula() -> None:
    frame = pd.DataFrame({"close": np.linspace(100.0, 180.0, 100), "v1_position": 1})
    result = backtest_v2_small_cap(
        frame,
        fee_rate=0.001,
        use_drawdown_gate=False,
        confirmation_days=1,
        gate_mode="sideways_hold",
    )

    expected_asset_return = frame["close"].pct_change().fillna(0.0)
    expected_gross = result["final_position"].shift(1).fillna(0).astype(int) * expected_asset_return
    expected_trade_size = result["final_position"].diff().abs().fillna(result["final_position"].abs())
    expected_fee = expected_trade_size * 0.001
    expected_net = expected_gross - expected_fee
    expected_equity = (1.0 + expected_net).cumprod()
    expected_drawdown = expected_equity / expected_equity.cummax() - 1.0

    assert result["asset_return"].iloc[0] == 0.0
    assert np.allclose(result["strategy_return_gross"], expected_gross)
    assert np.allclose(result["trade_size"], expected_trade_size)
    assert np.allclose(result["fee_cost"], expected_fee)
    assert np.allclose(result["strategy_return_net"], expected_net)
    assert np.allclose(result["equity_net"], expected_equity)
    assert np.allclose(result["drawdown"], expected_drawdown)


def test_v2_drawdown_gate_uses_prior_v2_equity_and_can_be_disabled() -> None:
    close = [100.0]
    for _ in range(70):
        close.append(close[-1] * 1.01)
    for _ in range(8):
        close.append(close[-1] * 0.96)
    for _ in range(20):
        close.append(close[-1] * 1.005)
    frame = pd.DataFrame({"close": close, "v1_position": 1})

    gated = backtest_v2_small_cap(frame, fee_rate=0.0, use_drawdown_gate=True, confirmation_days=1)
    ungated = backtest_v2_small_cap(frame, fee_rate=0.0, use_drawdown_gate=False, confirmation_days=1)

    deep_drawdown_rows = gated["strategy_drawdown"] <= -0.15
    assert deep_drawdown_rows.any()
    assert (gated.loc[deep_drawdown_rows, "final_position"] == 0).all()
    assert (ungated.loc[deep_drawdown_rows, "position_before_dd_gate"] == ungated.loc[deep_drawdown_rows, "final_position"]).all()
    assert np.allclose(gated["strategy_drawdown"], gated["equity_net"].shift(1).fillna(1.0) / gated["equity_peak"].shift(1).fillna(1.0) - 1.0)


def test_v2_performance_stats_and_summary_windows_support_v1_and_v2_shapes() -> None:
    dates = pd.date_range("2024-01-01", periods=420, freq="D")
    close = pd.Series(np.linspace(100.0, 220.0, 420))
    v2 = backtest_v2_small_cap(
        pd.DataFrame({"timestamp": dates, "close": close, "v1_position": 1}),
        fee_rate=0.001,
        use_drawdown_gate=False,
        confirmation_days=1,
    )
    stats = calculate_performance_stats(v2, annualization_factor=365)
    expected_metric_keys = {
        "total_return_net",
        "annualized_return_net",
        "max_drawdown",
        "Sharpe_net",
        "Sortino_net",
        "Calmar",
        "average_exposure",
        "turnover",
        "number_of_entries",
        "number_of_exits",
        "total_trades",
        "average_holding_days",
        "total_fee_paid",
    }
    assert expected_metric_keys.issubset(stats)
    assert stats["total_trades"] >= 1
    assert stats["total_fee_paid"] >= 0.0

    v1 = pd.DataFrame(
        {
            "timestamp": dates,
            "equity": (1.0 + pd.Series([0.0] + [0.001] * 419)).cumprod(),
            "period_return": [0.0] + [0.001] * 419,
            "current_exposure": 1.0,
            "safe_exposure_change": [1.0] + [0.0] * 419,
            "fee_paid": 0.001,
        }
    )
    summary = build_performance_summary_table(
        {"v1.final": v1, "v2.0-small-cap": v2},
        annualization_factor=365,
    )

    assert list(summary.columns) == [
        "version",
        "window",
        "total_return_net",
        "annualized_return_net",
        "max_drawdown",
        "Sharpe_net",
        "Sortino_net",
        "Calmar",
        "average_exposure",
        "turnover",
        "total_trades",
        "total_fee_paid",
    ]
    assert set(summary["version"]) == {"v1.final", "v2.0-small-cap"}
    assert set(summary["window"]) == {"Full period", "Recent 180d", "Recent 365d", "Recent 2y", "Recent 3y"}
    assert len(summary) == 10


def test_v2_regime_diagnostics_outputs_distribution_performance_and_trade_behavior() -> None:
    frame = pd.DataFrame(
        {
            "confirmed_regime": ["strong_bull", "strong_bull", "sideways", "bear", "weak_bull"],
            "final_position": [0, 1, 1, 0, 1],
            "trade_size": [0, 1, 0, 1, 1],
            "fee_cost": [0.0, 0.001, 0.0, 0.001, 0.001],
            "strategy_return_net": [0.0, 0.01, 0.005, -0.001, 0.003],
            "asset_return": [0.0, 0.02, 0.01, -0.02, 0.01],
        }
    )
    diagnostics = build_regime_diagnostics(frame)

    distribution = diagnostics["regime_distribution"]
    performance = diagnostics["regime_performance"]
    trade_behavior = diagnostics["trade_behavior"]

    assert list(distribution.columns) == ["regime", "days", "ratio"]
    assert distribution["days"].sum() == len(frame)
    assert np.isclose(distribution["ratio"].sum(), 1.0)

    assert list(performance.columns) == [
        "regime",
        "avg_position",
        "strategy_return_net",
        "asset_return",
        "trades",
        "fees",
    ]
    bear_row = performance.loc[performance["regime"] == "bear"].iloc[0]
    assert bear_row["avg_position"] == 0.0
    assert bear_row["trades"] == 1
    assert bear_row["fees"] == 0.001

    assert list(trade_behavior.columns) == [
        "number_of_entries",
        "number_of_exits",
        "total_trades",
        "average_holding_days",
        "average_exposure",
        "turnover",
        "total_fee_paid",
    ]
    assert trade_behavior["number_of_entries"].iloc[0] == 2
    assert trade_behavior["number_of_exits"].iloc[0] == 1
    assert trade_behavior["total_trades"].iloc[0] == 3
    assert trade_behavior["total_fee_paid"].iloc[0] == 0.003

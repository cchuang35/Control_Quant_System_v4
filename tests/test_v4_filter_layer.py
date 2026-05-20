import math

import pytest

from src.v4 import FilterConfig, MinimalFilterLayer, Observation


def observation(log_return: float, drawdown: float = 0.0, previous_position: float = 0.0) -> Observation:
    return Observation(
        log_return=log_return,
        pre_trade_drawdown=drawdown,
        previous_position=previous_position,
    )


def test_zero_return_sequence_produces_zero_filtered_signals() -> None:
    filter_layer = MinimalFilterLayer()

    for _ in range(20):
        filtered = filter_layer.update(observation(0.0))

    assert filtered.long_trend == pytest.approx(0.0)
    assert filtered.volatility == pytest.approx(0.0)
    assert filtered.short_timing == pytest.approx(0.0)
    assert filter_layer.step_count == 20


def test_constant_positive_return_sequence_reacts_with_positive_short_timing() -> None:
    filter_layer = MinimalFilterLayer(FilterConfig(short_window=3, vol_window=5, long_window=10))

    first = filter_layer.update(observation(0.01))
    later = first
    for _ in range(4):
        later = filter_layer.update(observation(0.01))

    assert later.long_trend > 0.0
    assert later.volatility > 0.0
    assert first.short_timing > 0.0
    assert later.short_timing > 0.0


def test_alternating_returns_increase_volatility() -> None:
    filter_layer = MinimalFilterLayer(FilterConfig(short_window=3, vol_window=5, long_window=10))

    filtered = None
    for value in [0.02, -0.02, 0.02, -0.02]:
        filtered = filter_layer.update(observation(value))

    assert filtered is not None
    assert filtered.volatility > 0.0


def test_drawdown_and_position_are_passed_through() -> None:
    filtered = MinimalFilterLayer().update(observation(0.01, drawdown=0.2, previous_position=0.7))

    assert filtered.drawdown == pytest.approx(0.2)
    assert filtered.previous_position == pytest.approx(0.7)
    assert filtered.as_tuple()[3:] == pytest.approx((0.2, 0.7))


def test_recursive_update_matches_manual_causal_equations() -> None:
    config = FilterConfig(short_window=2, vol_window=4, long_window=6)
    filter_layer = MinimalFilterLayer(config)
    returns = [0.01, -0.02, 0.03]
    alpha_l = 2.0 / (config.long_window + 1.0)
    alpha_s = 2.0 / (config.short_window + 1.0)
    alpha_v = 2.0 / (config.vol_window + 1.0)
    expected_l = 0.0
    expected_m = 0.0
    expected_q = 0.0

    for value in returns:
        filtered = filter_layer.update(observation(value))
        expected_l = (1.0 - alpha_l) * expected_l + alpha_l * value
        expected_m = (1.0 - alpha_s) * expected_m + alpha_s * value
        expected_q = (1.0 - alpha_v) * expected_q + alpha_v * (value**2)

        assert filtered.long_trend == pytest.approx(expected_l)
        assert filtered.short_timing == pytest.approx(expected_m - expected_l)
        assert filtered.volatility == pytest.approx(math.sqrt(expected_q))


def test_reset_restores_initial_filter_state() -> None:
    filter_layer = MinimalFilterLayer()
    filter_layer.update(observation(0.02))

    filter_layer.reset()

    assert filter_layer.long_trend == pytest.approx(0.0)
    assert filter_layer.short_momentum == pytest.approx(0.0)
    assert filter_layer.variance == pytest.approx(0.0)
    assert filter_layer.step_count == 0
    assert filter_layer.is_warmup


def test_filter_config_validates_windows() -> None:
    with pytest.raises(ValueError, match="short_window must be positive"):
        FilterConfig(short_window=0)
    with pytest.raises(ValueError, match="vol_window must be positive"):
        FilterConfig(vol_window=0)
    with pytest.raises(ValueError, match="long_window must be positive"):
        FilterConfig(long_window=0)
    with pytest.raises(ValueError, match="short_window < vol_window < long_window"):
        FilterConfig(short_window=10, vol_window=5, long_window=20)

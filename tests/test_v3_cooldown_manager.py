from dataclasses import dataclass

import pytest

from src.v3.cooldown_manager import (
    CooldownStateSnapshotV3,
    RegimeCooldownManagerV3,
    TradeCloseInfoV3,
)


def test_v3_cooldown_starts_on_losing_bull_like_trade() -> None:
    manager = RegimeCooldownManagerV3(cooldown_bars=3)

    manager.update_on_trade_close(TradeCloseInfoV3(entry_regime="bull", net_trade_return=-0.01))

    assert manager.is_active()
    assert manager.is_active("bull")
    assert manager.is_active("weak_bull")
    assert manager.is_active("strong_bull")
    assert not manager.is_active("neutral")
    assert manager.get_state().remaining_by_regime["bull"] == 3


def test_v3_cooldown_counts_down_by_bar_without_forcing_exit() -> None:
    manager = RegimeCooldownManagerV3(cooldown_bars=2)
    manager.update_on_trade_close({"entry_regime": "weak_bull", "net_trade_return": -0.02})

    manager.update_on_bar()
    assert manager.is_active("weak_bull")
    assert manager.get_state().remaining_by_regime["weak_bull"] == 1

    manager.update_on_bar()
    assert not manager.is_active()
    assert manager.get_state().remaining_by_regime == {}


def test_v3_cooldown_does_not_trigger_on_winner_or_non_bull_like_trade() -> None:
    manager = RegimeCooldownManagerV3(cooldown_bars=120)

    manager.update_on_trade_close({"entry_regime": "bull", "net_trade_return": 0.01})
    assert not manager.is_active()

    manager.update_on_trade_close({"entry_regime": "bear", "net_trade_return": -0.01})
    assert not manager.is_active()


def test_v3_cooldown_supports_alternative_tested_values() -> None:
    for bars in (120, 144, 168):
        manager = RegimeCooldownManagerV3(cooldown_bars=bars)
        manager.update_on_trade_close({"entry_regime": "weak_bull", "net_trade_return": -0.01})
        assert manager.get_state().remaining_by_regime["weak_bull"] == bars


def test_v3_cooldown_supports_external_trade_objects_and_snapshot() -> None:
    @dataclass(frozen=True)
    class ExternalTrade:
        entry_regime: str
        net_trade_return: float
        exit_regime: str
        bars_held: int

    manager = RegimeCooldownManagerV3(cooldown_bars=5)
    manager.update_on_trade_close(ExternalTrade("bull", -0.03, "neutral", 12))
    state = manager.get_state()

    assert isinstance(state, CooldownStateSnapshotV3)
    assert state.last_trigger["entry_regime"] == "bull"
    assert state.last_trigger["exit_regime"] == "neutral"
    assert state.last_trigger["bars_held"] == 12


def test_v3_cooldown_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="cooldown_bars"):
        RegimeCooldownManagerV3(cooldown_bars=-1)

    manager = RegimeCooldownManagerV3()
    with pytest.raises(ValueError, match="missing required fields"):
        manager.update_on_trade_close({"entry_regime": "bull"})

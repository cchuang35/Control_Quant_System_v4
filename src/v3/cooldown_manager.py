"""v3 regime-specific cooldown manager.

This module preserves the v2 weak-bull losing-trade cooldown idea. A cooldown
blocks repeated weak/bull-like additions after a losing trade, but it does not
force existing positions to exit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


DEFAULT_BULL_LIKE_REGIMES = frozenset({"weak_bull", "bull", "strong_bull"})


@dataclass(frozen=True)
class TradeCloseInfoV3:
    """Information needed when a trade is completed."""

    entry_regime: str
    net_trade_return: float
    exit_regime: str | None = None
    bars_held: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CooldownStateSnapshotV3:
    """Serializable snapshot of active v3 cooldown state."""

    cooldown_bars: int
    remaining_by_regime: dict[str, int]
    bull_like_regimes: tuple[str, ...]
    last_trigger: dict[str, Any] | None = None


@dataclass
class RegimeCooldownManagerV3:
    """Track regime-specific cooldowns by bar.

    Defaults match the v2 BTCUSDT 1h final-candidate cooldown of 120 bars.
    Other tested values such as 144 and 168 can be passed through
    ``cooldown_bars``.
    """

    cooldown_bars: int = 120
    bull_like_regimes: frozenset[str] = DEFAULT_BULL_LIKE_REGIMES
    remaining_by_regime: dict[str, int] = field(default_factory=dict)
    last_trigger: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.cooldown_bars < 0:
            raise ValueError("cooldown_bars must be non-negative")
        self.bull_like_regimes = frozenset(str(regime) for regime in self.bull_like_regimes)
        self.remaining_by_regime = {
            str(regime): int(remaining)
            for regime, remaining in self.remaining_by_regime.items()
            if int(remaining) > 0
        }

    def is_active(self, regime: str | None = None) -> bool:
        """Return whether cooldown is active globally or for one regime."""

        if regime is None:
            return any(remaining > 0 for remaining in self.remaining_by_regime.values())
        regime_name = str(regime)
        if regime_name in self.bull_like_regimes:
            return any(self.remaining_by_regime.get(name, 0) > 0 for name in self.bull_like_regimes)
        return self.remaining_by_regime.get(regime_name, 0) > 0

    def update_on_bar(self) -> None:
        """Advance cooldown state by one bar."""

        next_remaining = {}
        for regime, remaining in self.remaining_by_regime.items():
            updated = max(0, int(remaining) - 1)
            if updated > 0:
                next_remaining[regime] = updated
        self.remaining_by_regime = next_remaining

    def update_on_trade_close(self, trade_info: TradeCloseInfoV3 | dict[str, Any] | Any) -> None:
        """Start cooldown when a bull-like trade closes at a loss."""

        info = _trade_info_values(trade_info)
        entry_regime = str(info["entry_regime"])
        net_trade_return = float(info["net_trade_return"])
        if entry_regime not in self.bull_like_regimes or net_trade_return >= 0.0:
            return

        if self.cooldown_bars <= 0:
            return

        for regime in self.bull_like_regimes:
            self.remaining_by_regime[regime] = self.cooldown_bars
        self.last_trigger = {
            "entry_regime": entry_regime,
            "exit_regime": info.get("exit_regime"),
            "net_trade_return": net_trade_return,
            "cooldown_bars": self.cooldown_bars,
            "bars_held": info.get("bars_held"),
        }

    def get_state(self) -> CooldownStateSnapshotV3:
        """Return a snapshot of current cooldown state."""

        return CooldownStateSnapshotV3(
            cooldown_bars=self.cooldown_bars,
            remaining_by_regime=dict(self.remaining_by_regime),
            bull_like_regimes=tuple(sorted(self.bull_like_regimes)),
            last_trigger=dict(self.last_trigger) if self.last_trigger else None,
        )


def _trade_info_values(trade_info: TradeCloseInfoV3 | dict[str, Any] | Any) -> dict[str, Any]:
    if isinstance(trade_info, TradeCloseInfoV3):
        values = asdict(trade_info)
    elif isinstance(trade_info, dict):
        values = dict(trade_info)
    else:
        values = {
            "entry_regime": getattr(trade_info, "entry_regime"),
            "net_trade_return": getattr(trade_info, "net_trade_return"),
            "exit_regime": getattr(trade_info, "exit_regime", None),
            "bars_held": getattr(trade_info, "bars_held", None),
        }

    missing = {"entry_regime", "net_trade_return"}.difference(values)
    if missing:
        raise ValueError(f"trade_info is missing required fields: {sorted(missing)}")
    return values


__all__ = [
    "DEFAULT_BULL_LIKE_REGIMES",
    "CooldownStateSnapshotV3",
    "RegimeCooldownManagerV3",
    "TradeCloseInfoV3",
]

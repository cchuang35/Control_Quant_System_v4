"""Minimal causal filter layer for v4 observations."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .data_types import FilteredSignals, Observation


@dataclass(frozen=True)
class FilterConfig:
    """EWMA window configuration for the minimal filter layer."""

    short_window: int = 10
    vol_window: int = 30
    long_window: int = 60

    def __post_init__(self) -> None:
        if self.short_window <= 0:
            raise ValueError("short_window must be positive")
        if self.vol_window <= 0:
            raise ValueError("vol_window must be positive")
        if self.long_window <= 0:
            raise ValueError("long_window must be positive")
        if not self.short_window < self.vol_window < self.long_window:
            raise ValueError("filter windows must satisfy short_window < vol_window < long_window")


class MinimalFilterLayer:
    """Recursive, causal EWMA filter for one observation at a time."""

    def __init__(self, config: FilterConfig | None = None) -> None:
        self.config = config or FilterConfig()
        self.alpha_long = 2.0 / (self.config.long_window + 1.0)
        self.alpha_short = 2.0 / (self.config.short_window + 1.0)
        self.alpha_vol = 2.0 / (self.config.vol_window + 1.0)
        self.reset()

    @property
    def is_warmup(self) -> bool:
        return self.step_count < self.config.long_window

    def update(self, observation: Observation) -> FilteredSignals:
        r_t = float(observation.log_return)
        self.long_trend = (1.0 - self.alpha_long) * self.long_trend + self.alpha_long * r_t
        self.short_momentum = (1.0 - self.alpha_short) * self.short_momentum + self.alpha_short * r_t
        self.variance = (1.0 - self.alpha_vol) * self.variance + self.alpha_vol * (r_t**2)
        self.step_count += 1

        volatility = math.sqrt(max(self.variance, 0.0))
        return FilteredSignals(
            long_trend=self.long_trend,
            volatility=volatility,
            short_timing=self.short_momentum - self.long_trend,
            drawdown=float(observation.pre_trade_drawdown),
            previous_position=float(observation.previous_position),
        )

    def reset(self) -> None:
        self.long_trend = 0.0
        self.short_momentum = 0.0
        self.variance = 0.0
        self.step_count = 0

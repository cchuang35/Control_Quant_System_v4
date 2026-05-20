"""Observation construction for the v4 minimal simulation framework."""

from __future__ import annotations

from .data_types import Observation


class ObservationBuilder:
    """Build the minimal observation vector y_t."""

    def build(
        self,
        *,
        log_return: float,
        pre_trade_drawdown: float,
        previous_position: float,
    ) -> Observation:
        return Observation(
            log_return=float(log_return),
            pre_trade_drawdown=float(pre_trade_drawdown),
            previous_position=float(previous_position),
        )

"""Interfaces and placeholders for v4 estimators and controllers."""

from __future__ import annotations

import math
from typing import Protocol

from .data_types import Observation, StateVector
from .filters import FilterConfig, MinimalFilterLayer


class StateEstimator(Protocol):
    """State estimator interface.

    Real filters can replace this later. The update method receives the current
    observation after it has been appended to the estimator's own history.
    """

    def update(self, observation: Observation) -> StateVector:
        """Return the current minimal state vector."""


class Controller(Protocol):
    """Trading controller interface."""

    def decide(self, state: StateVector) -> float:
        """Return a raw target position before long-only clipping."""


class PositionConstraint(Protocol):
    """Position constraint interface."""

    def apply(self, raw_position: float) -> float:
        """Return a feasible position."""


class DummyStateEstimator:
    """Neutral placeholder estimator for framework tests.

    It records observations and maps pre-trade drawdown into the portfolio risk
    state. No market trend, timing, volatility, or strategy logic is estimated.
    """

    def __init__(self) -> None:
        self.observations: list[Observation] = []

    def update(self, observation: Observation) -> StateVector:
        self.observations.append(observation)
        return StateVector(
            tau=0.0,
            nu=0.0,
            epsilon=0.0,
            rho=LongOnlyPositionConstraint.clip(observation.pre_trade_drawdown),
            previous_position=observation.previous_position,
        )


class FilteredDummyStateEstimator:
    """Placeholder estimator that consumes filtered signals without strategy logic."""

    def __init__(self, filter_layer: MinimalFilterLayer | None = None, config: FilterConfig | None = None) -> None:
        self.filter_layer = filter_layer or MinimalFilterLayer(config=config)

    def update(self, observation: Observation) -> StateVector:
        filtered = self.filter_layer.update(observation)
        return StateVector(
            tau=0.0,
            nu=0.0,
            epsilon=0.0,
            rho=LongOnlyPositionConstraint.clip(filtered.drawdown),
            previous_position=filtered.previous_position,
        )


class LongOnlyPositionConstraint:
    """Clip positions to the v4 minimal long-only, no-leverage range [0, 1]."""

    @staticmethod
    def clip(value: float) -> float:
        if not math.isfinite(float(value)):
            raise ValueError("raw target position must be finite")
        return min(1.0, max(0.0, float(value)))

    def apply(self, raw_position: float) -> float:
        return self.clip(raw_position)

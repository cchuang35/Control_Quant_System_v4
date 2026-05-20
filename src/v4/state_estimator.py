"""Minimal state estimator for v4 filtered signals."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .data_types import FilteredSignals, Observation, StateVector
from .filters import MinimalFilterLayer


@dataclass(frozen=True)
class StateEstimatorConfig:
    """Normalization parameters for mapping filtered signals to state."""

    k_tau: float = 1.0
    k_epsilon: float = 1.0
    vol_ref: float = 0.03
    drawdown_ref: float = 0.20
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if not math.isfinite(self.k_tau):
            raise ValueError("k_tau must be finite")
        if not math.isfinite(self.k_epsilon):
            raise ValueError("k_epsilon must be finite")
        if self.vol_ref <= 0.0:
            raise ValueError("vol_ref must be positive")
        if self.drawdown_ref <= 0.0:
            raise ValueError("drawdown_ref must be positive")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")


class MinimalStateEstimator:
    """Map causal filtered signals into bounded v4 state vectors."""

    def __init__(
        self,
        filter_layer: MinimalFilterLayer | None = None,
        config: StateEstimatorConfig | None = None,
    ) -> None:
        self.filter_layer = filter_layer or MinimalFilterLayer()
        self.config = config or StateEstimatorConfig()

    def update(self, observation: Observation) -> StateVector:
        filtered = self.filter_layer.update(observation)
        return self.estimate_from_filtered(filtered)

    def estimate_from_filtered(self, filtered: FilteredSignals) -> StateVector:
        volatility_denom = float(filtered.volatility) + self.config.epsilon
        tau = math.tanh(self.config.k_tau * float(filtered.long_trend) / volatility_denom)
        nu = _clip(float(filtered.volatility) / self.config.vol_ref, 0.0, 1.0)
        timing = math.tanh(self.config.k_epsilon * float(filtered.short_timing) / volatility_denom)
        rho = _clip(float(filtered.drawdown) / self.config.drawdown_ref, 0.0, 1.0)
        previous_position = _clip(float(filtered.previous_position), 0.0, 1.0)
        return StateVector(
            tau=tau,
            nu=nu,
            epsilon=timing,
            rho=rho,
            previous_position=previous_position,
        )

    def reset(self) -> None:
        self.filter_layer.reset()


def _clip(value: float, lower: float, upper: float) -> float:
    if not math.isfinite(value):
        raise ValueError("state estimator input must be finite")
    return min(upper, max(lower, value))

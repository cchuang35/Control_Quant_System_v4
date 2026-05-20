"""Controllers for the v4 framework."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .data_types import StateVector


@dataclass(frozen=True)
class ControllerConfig:
    """Configuration for the minimal continuous long-only controller."""

    w_epsilon: float = 0.25
    w_volatility: float = 0.50
    w_portfolio_risk: float = 0.75
    max_position_change: float = 0.20
    tau_floor: float = 0.0
    rebalance_threshold: float = 0.0
    use_trend_persistence_gate: bool = False
    tau_confirm_threshold: float = 0.25
    trend_persistence_window: int = 10
    persistence_floor: float = 0.50

    def __post_init__(self) -> None:
        _validate_unit_interval("w_epsilon", self.w_epsilon)
        _validate_unit_interval("w_volatility", self.w_volatility)
        _validate_unit_interval("w_portfolio_risk", self.w_portfolio_risk)
        if not math.isfinite(self.max_position_change) or not 0.0 < self.max_position_change <= 1.0:
            raise ValueError("max_position_change must be in (0, 1]")
        if not math.isfinite(self.tau_floor) or not 0.0 <= self.tau_floor < 1.0:
            raise ValueError("tau_floor must be in [0, 1)")
        if not math.isfinite(self.rebalance_threshold) or not 0.0 <= self.rebalance_threshold <= 1.0:
            raise ValueError("rebalance_threshold must be in [0, 1]")
        if not math.isfinite(self.tau_confirm_threshold):
            raise ValueError("tau_confirm_threshold must be finite")
        if self.trend_persistence_window <= 0:
            raise ValueError("trend_persistence_window must be positive")
        if not math.isfinite(self.persistence_floor) or not 0.0 <= self.persistence_floor < 1.0:
            raise ValueError("persistence_floor must be in [0, 1)")


class MinimalContinuousController:
    """Long-only continuous controller driven by bounded state variables."""

    def __init__(self, config: ControllerConfig | None = None) -> None:
        self.config = config or ControllerConfig()
        self.trend_persistence_state = 0.0

    def reset(self) -> None:
        self.trend_persistence_state = 0.0

    def decide(self, state: StateVector) -> float:
        return self.explain(state)["raw_target_position"]

    def explain(self, state: StateVector) -> dict[str, float]:
        """Return controller intermediates and update enabled recursive controller state."""

        tau = float(state.tau)
        nu = float(state.nu)
        epsilon = float(state.epsilon)
        rho = float(state.rho)
        previous_position = _clip(float(state.previous_position), 0.0, 1.0)

        base_exposure_c = _clip(
            (tau - self.config.tau_floor) / (1.0 - self.config.tau_floor),
            0.0,
            1.0,
        )
        trend_confirm_indicator = 1.0 if tau > self.config.tau_confirm_threshold else 0.0
        if self.config.use_trend_persistence_gate:
            alpha_p = 2.0 / (float(self.config.trend_persistence_window) + 1.0)
            self.trend_persistence_state = (
                (1.0 - alpha_p) * self.trend_persistence_state
                + alpha_p * trend_confirm_indicator
            )
            trend_persistence_gate = _clip(
                (self.trend_persistence_state - self.config.persistence_floor)
                / (1.0 - self.config.persistence_floor),
                0.0,
                1.0,
            )
        else:
            trend_persistence_gate = 1.0
        base_exposure = base_exposure_c * trend_persistence_gate
        timing_multiplier = _clip(
            1.0 + self.config.w_epsilon * epsilon,
            1.0 - self.config.w_epsilon,
            1.0 + self.config.w_epsilon,
        )
        market_risk_multiplier = _clip(
            1.0 - self.config.w_volatility * nu,
            0.0,
            1.0,
        )
        portfolio_risk_multiplier = _clip(
            1.0 - self.config.w_portfolio_risk * rho,
            0.0,
            1.0,
        )
        unsmoothed_target = _clip(
            base_exposure * timing_multiplier * market_risk_multiplier * portfolio_risk_multiplier,
            0.0,
            1.0,
        )
        clipped_delta = _clip(
            unsmoothed_target - previous_position,
            -self.config.max_position_change,
            self.config.max_position_change,
        )
        pre_deadband_target = _clip(previous_position + clipped_delta, 0.0, 1.0)
        deadband_skip = abs(pre_deadband_target - previous_position) < self.config.rebalance_threshold
        raw_target_position = previous_position if deadband_skip else pre_deadband_target
        return {
            "base_exposure_C": base_exposure_c,
            "base_exposure": base_exposure,
            "base_exposure_E": base_exposure,
            "trend_confirm_indicator": trend_confirm_indicator,
            "trend_persistence_state": self.trend_persistence_state,
            "trend_persistence_gate": trend_persistence_gate,
            "exposure_reduction_from_gate": base_exposure_c - base_exposure,
            "timing_multiplier": timing_multiplier,
            "market_risk_multiplier": market_risk_multiplier,
            "portfolio_risk_multiplier": portfolio_risk_multiplier,
            "unsmoothed_target": unsmoothed_target,
            "clipped_delta": clipped_delta,
            "pre_deadband_target": pre_deadband_target,
            "deadband_skip": float(deadband_skip),
            "raw_target_position": raw_target_position,
            "final_position": raw_target_position,
        }


class ZeroController:
    """Always target zero exposure."""

    def decide(self, state: StateVector) -> float:
        return 0.0


class BuyAndHoldController:
    """Always target full long exposure."""

    def decide(self, state: StateVector) -> float:
        return 1.0


@dataclass(frozen=True)
class FixedExposureController:
    """Always target a fixed raw exposure."""

    exposure: float = 0.5

    def decide(self, state: StateVector) -> float:
        return float(self.exposure)


def _clip(value: float, lower: float, upper: float) -> float:
    if not math.isfinite(value):
        raise ValueError("controller input must be finite")
    return min(upper, max(lower, value))


def _validate_unit_interval(name: str, value: float) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")

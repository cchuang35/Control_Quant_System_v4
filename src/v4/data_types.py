"""Data contracts for the v4 minimal simulation framework."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Observation:
    """Minimal observation vector y_t."""

    log_return: float
    pre_trade_drawdown: float
    previous_position: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.log_return, self.pre_trade_drawdown, self.previous_position)


@dataclass(frozen=True)
class StateVector:
    """Minimal state vector z_t consumed by controllers."""

    tau: float
    nu: float
    epsilon: float
    rho: float
    previous_position: float

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        return (self.tau, self.nu, self.epsilon, self.rho, self.previous_position)


@dataclass(frozen=True)
class FilteredSignals:
    """Minimal filtered signal vector phi_t."""

    long_trend: float
    volatility: float
    short_timing: float
    drawdown: float
    previous_position: float

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        return (
            self.long_trend,
            self.volatility,
            self.short_timing,
            self.drawdown,
            self.previous_position,
        )


@dataclass(frozen=True)
class PreTradeState:
    """Portfolio state after market movement and before the rebalance."""

    simple_return: float
    log_return: float
    pre_trade_equity: float
    pre_trade_high_watermark: float
    pre_trade_drawdown: float


@dataclass(frozen=True)
class AccountingResult:
    """Portfolio state after the rebalance and transaction cost."""

    turnover: float
    transaction_cost: float
    equity: float
    high_watermark: float
    drawdown: float


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for the v4 minimal backtest."""

    fee_rate: float = 0.001
    initial_equity: float = 1.0
    initial_high_watermark: float = 1.0
    initial_position: float = 0.0


BacktestRow = dict[str, Any]

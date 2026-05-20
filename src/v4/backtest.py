"""Backtest engine for the v4 minimal closed-loop simulation framework."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from .data_types import BacktestConfig, BacktestRow
from .interfaces import Controller, DummyStateEstimator, LongOnlyPositionConstraint, PositionConstraint, StateEstimator
from .observation import ObservationBuilder
from .portfolio import PortfolioAccounting


class BacktestEngine:
    """Run the v4 minimal long-only, no-leverage backtest loop."""

    def __init__(
        self,
        *,
        controller: Controller,
        state_estimator: StateEstimator | None = None,
        config: BacktestConfig | None = None,
        observation_builder: ObservationBuilder | None = None,
        accounting: PortfolioAccounting | None = None,
        position_constraint: PositionConstraint | None = None,
    ) -> None:
        self.controller = controller
        self.state_estimator = state_estimator or DummyStateEstimator()
        self.config = config or BacktestConfig()
        self.observation_builder = observation_builder or ObservationBuilder()
        self.accounting = accounting or PortfolioAccounting()
        self.position_constraint = position_constraint or LongOnlyPositionConstraint()
        self._validate_config()

    def run(self, prices: Sequence[float] | pd.Series | pd.DataFrame, *, price_column: str = "close") -> pd.DataFrame:
        _reset_if_supported(self.controller)
        _reset_if_supported(self.state_estimator)
        price_series = _coerce_price_series(prices, price_column=price_column)
        equity = float(self.config.initial_equity)
        high_watermark = float(self.config.initial_high_watermark)
        previous_position = float(self.config.initial_position)
        rows: list[BacktestRow] = []

        for step in range(1, len(price_series)):
            previous_price = float(price_series.iloc[step - 1])
            price = float(price_series.iloc[step])
            timestamp: Any = price_series.index[step]

            pre_trade = self.accounting.compute_pre_trade(
                previous_price=previous_price,
                price=price,
                previous_equity=equity,
                previous_high_watermark=high_watermark,
                previous_position=previous_position,
            )
            observation = self.observation_builder.build(
                log_return=pre_trade.log_return,
                pre_trade_drawdown=pre_trade.pre_trade_drawdown,
                previous_position=previous_position,
            )
            state = self.state_estimator.update(observation)
            raw_target_position = float(self.controller.decide(state))
            position = self.position_constraint.apply(raw_target_position)
            final_accounting = self.accounting.apply_rebalance(
                pre_trade_equity=pre_trade.pre_trade_equity,
                pre_trade_high_watermark=pre_trade.pre_trade_high_watermark,
                previous_position=previous_position,
                position=position,
                fee_rate=self.config.fee_rate,
            )

            rows.append(
                {
                    "timestamp": timestamp,
                    "price": price,
                    "simple_return": pre_trade.simple_return,
                    "log_return": pre_trade.log_return,
                    "pre_trade_equity": pre_trade.pre_trade_equity,
                    "pre_trade_high_watermark": pre_trade.pre_trade_high_watermark,
                    "pre_trade_drawdown": pre_trade.pre_trade_drawdown,
                    "previous_position": previous_position,
                    "observation": observation.as_tuple(),
                    "state": state.as_tuple(),
                    "raw_target_position": raw_target_position,
                    "position": position,
                    "turnover": final_accounting.turnover,
                    "transaction_cost": final_accounting.transaction_cost,
                    "equity": final_accounting.equity,
                    "high_watermark": final_accounting.high_watermark,
                    "drawdown": final_accounting.drawdown,
                }
            )

            equity = final_accounting.equity
            high_watermark = final_accounting.high_watermark
            previous_position = position

        result = pd.DataFrame(rows)
        result.attrs["initial_equity"] = float(self.config.initial_equity)
        result.attrs["initial_high_watermark"] = float(self.config.initial_high_watermark)
        result.attrs["initial_position"] = float(self.config.initial_position)
        return result

    def _validate_config(self) -> None:
        if self.config.fee_rate < 0.0:
            raise ValueError("fee_rate must be non-negative")
        if self.config.initial_equity <= 0.0:
            raise ValueError("initial_equity must be positive")
        if self.config.initial_high_watermark <= 0.0:
            raise ValueError("initial_high_watermark must be positive")
        if not 0.0 <= self.config.initial_position <= 1.0:
            raise ValueError("initial_position must be in [0, 1]")


def _coerce_price_series(prices: Sequence[float] | pd.Series | pd.DataFrame, *, price_column: str) -> pd.Series:
    if isinstance(prices, pd.DataFrame):
        if price_column not in prices.columns:
            raise ValueError(f"price_column '{price_column}' not found")
        series = prices[price_column]
    elif isinstance(prices, pd.Series):
        series = prices
    else:
        series = pd.Series(list(prices))

    if len(series) < 2:
        raise ValueError("at least two prices are required")
    if series.isna().any():
        raise ValueError("prices must not contain missing values")
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        raise ValueError("prices must be numeric")
    if (numeric <= 0.0).any():
        raise ValueError("prices must be positive")
    return numeric.astype(float)


def _reset_if_supported(component: Any) -> None:
    reset = getattr(component, "reset", None)
    if callable(reset):
        reset()


def run_backtest(
    prices: Sequence[float] | pd.Series | pd.DataFrame,
    *,
    controller: Controller,
    state_estimator: StateEstimator | None = None,
    config: BacktestConfig | None = None,
    price_column: str = "close",
) -> pd.DataFrame:
    """Convenience wrapper around BacktestEngine."""

    return BacktestEngine(controller=controller, state_estimator=state_estimator, config=config).run(
        prices,
        price_column=price_column,
    )

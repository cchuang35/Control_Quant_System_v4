"""Portfolio accounting for the v4 minimal long-only backtest."""

from __future__ import annotations

import math

from .data_types import AccountingResult, PreTradeState


class PortfolioAccounting:
    """Apply close-to-close returns and rebalance transaction costs."""

    def compute_pre_trade(
        self,
        *,
        previous_price: float,
        price: float,
        previous_equity: float,
        previous_high_watermark: float,
        previous_position: float,
    ) -> PreTradeState:
        if previous_price <= 0.0 or price <= 0.0:
            raise ValueError("prices must be positive")
        simple_return = price / previous_price - 1.0
        log_return = math.log(price / previous_price)
        pre_trade_equity = previous_equity * (1.0 + previous_position * simple_return)
        pre_trade_high_watermark = max(previous_high_watermark, pre_trade_equity)
        pre_trade_drawdown = 1.0 - pre_trade_equity / pre_trade_high_watermark
        return PreTradeState(
            simple_return=simple_return,
            log_return=log_return,
            pre_trade_equity=pre_trade_equity,
            pre_trade_high_watermark=pre_trade_high_watermark,
            pre_trade_drawdown=pre_trade_drawdown,
        )

    def apply_rebalance(
        self,
        *,
        pre_trade_equity: float,
        pre_trade_high_watermark: float,
        previous_position: float,
        position: float,
        fee_rate: float,
    ) -> AccountingResult:
        if fee_rate < 0.0:
            raise ValueError("fee_rate must be non-negative")
        turnover = abs(position - previous_position)
        transaction_cost = fee_rate * turnover * pre_trade_equity
        equity = pre_trade_equity - transaction_cost
        high_watermark = max(pre_trade_high_watermark, equity)
        drawdown = 1.0 - equity / high_watermark
        return AccountingResult(
            turnover=turnover,
            transaction_cost=transaction_cost,
            equity=equity,
            high_watermark=high_watermark,
            drawdown=drawdown,
        )

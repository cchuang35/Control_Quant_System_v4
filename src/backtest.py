"""Minimal backtest runner for the v1 five-layer pipeline."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .layer1_market_model import OHLCVBar, build_market_state
from .layer2_state_estimator import EstimatedMarketStateV1, estimate_market_state
from .layer3_strategy_controller import PortfolioStateV1, compute_control_action
from .layer4_risk_filter import apply_risk_filter


def load_ohlcv_csv(path: Path) -> list[OHLCVBar]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = csv.DictReader(file)
        return [
            OHLCVBar(
                timestamp=float(row.get("timestamp", idx)),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
            for idx, row in enumerate(rows)
        ]


def run_backtest(bars: list[OHLCVBar]) -> dict[str, float | int | str]:
    if len(bars) < 2:
        raise ValueError("at least two OHLCV bars are required")

    equity = 1.0
    peak_equity = 1.0
    current_exposure = 0.0
    previous_estimated: EstimatedMarketStateV1 | None = None
    last_mode = "normal"
    trades = 0

    for idx in range(1, len(bars)):
        market = build_market_state(bars[: idx + 1])
        estimated = estimate_market_state(market, previous_estimated)
        previous_estimated = estimated
        drawdown = equity / peak_equity - 1.0
        portfolio = PortfolioStateV1(
            current_exposure=current_exposure,
            current_position=current_exposure * equity / market.close if market.close else 0.0,
            equity=equity,
            cash=equity * (1.0 - abs(current_exposure)),
            unrealized_pnl=0.0,
            portfolio_drawdown=drawdown,
            leverage=abs(current_exposure),
            available_margin=max(0.0, 1.0 - abs(current_exposure)),
        )
        action = compute_control_action(market, estimated, portfolio)
        safe_action = apply_risk_filter(market, estimated, portfolio, action)
        if safe_action.trade_allowed and safe_action.safe_exposure_change:
            trades += 1
        current_exposure += safe_action.safe_exposure_change
        current_exposure = max(-1.0, min(1.0, current_exposure))
        equity *= 1.0 + current_exposure * market.return_1
        peak_equity = max(peak_equity, equity)
        last_mode = estimated.dominant_regime

    return {
        "bars": len(bars),
        "final_equity": equity,
        "total_return": equity - 1.0,
        "trades": trades,
        "final_exposure": current_exposure,
        "last_regime": last_mode,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal v1 five-layer backtest.")
    parser.add_argument("csv", type=Path, help="CSV with timestamp,open,high,low,close,volume columns")
    args = parser.parse_args()
    result = run_backtest(load_ohlcv_csv(args.csv))
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()


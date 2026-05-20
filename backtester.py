"""Sequential backtester for Layer 1 through Layer 4.

The event order avoids look-ahead bias:
1. At bar t, compute all layer states using data up to and including t.
2. Produce a SafeControlActionV1 at bar t.
3. Hold the previous exposure over close[t] -> close[t + 1].
4. Execute safe_exposure_change at close[t + 1] and charge fees there.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from math import log, tanh
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

import numpy as np
import pandas as pd

from src.layer1_market_model import MarketStateV1, OHLCVBar, build_market_state, classify_market_mode
from src.layer2_state_estimator import EstimatedMarketStateV1, estimate_market_state
from src.layer3_strategy_controller import ControlActionV1, PortfolioStateV1, compute_control_action
from src.layer4_risk_filter import RiskConfigV1, SafeControlActionV1, apply_risk_filter


REQUIRED_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class BacktestResult:
    metrics: dict[str, float | int]
    equity_curve: pd.DataFrame
    exposure_history: pd.DataFrame
    market_state_history: pd.DataFrame
    estimated_state_history: pd.DataFrame
    control_action_history: pd.DataFrame
    safe_control_action_history: pd.DataFrame
    trade_history: pd.DataFrame


def load_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    """Load one OHLCV CSV from data/ or an explicit path."""

    csv_path = Path(path)
    if not csv_path.exists():
        data_path = Path("data") / csv_path
        if data_path.exists():
            csv_path = data_path
    if not csv_path.exists():
        raise FileNotFoundError(f"OHLCV CSV not found: {path}")

    frame = pd.read_csv(csv_path)
    missing = set(REQUIRED_OHLCV_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")
    return frame


def find_data_csv(data_dir: str | Path = "data") -> Path:
    """Return the first CSV under data/ in lexical order."""

    paths = sorted(Path(data_dir).glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {Path(data_dir).resolve()}")
    return paths[0]


def run_backtest(
    ohlcv: pd.DataFrame,
    *,
    fee_rate: float = 0.0005,
    initial_equity: float = 1.0,
    periods_per_year: int = 252,
    long_only: bool = False,
    risk_config: RiskConfigV1 | None = None,
    minimum_rebalance_interval: int = 3,
    use_target_smoothing: bool = True,
    target_deadband: float = 0.05,
    beta_increase_risk: float = 0.30,
    beta_decrease_risk: float = 0.60,
) -> BacktestResult:
    """Run the Layer 1 -> Layer 4 pipeline without look-ahead bias."""

    _validate_ohlcv(ohlcv)
    if len(ohlcv) < 2:
        raise ValueError("at least two OHLCV rows are required")
    if fee_rate < 0:
        raise ValueError("fee_rate must be non-negative")
    if initial_equity <= 0:
        raise ValueError("initial_equity must be positive")

    bars = _frame_to_bars(ohlcv)
    closes = pd.to_numeric(ohlcv["close"], errors="coerce").astype(float).to_numpy()
    timestamps = _timestamps(ohlcv)

    equity = float(initial_equity)
    peak_equity = float(initial_equity)
    current_exposure = 0.0
    previous_estimated: EstimatedMarketStateV1 | None = None
    previous_target_exposure: float | None = None
    last_nonzero_trade_idx: int | None = None

    equity_rows: list[dict[str, Any]] = [
        {
            "timestamp": timestamps[0],
            "close": closes[0],
            "equity": equity,
            "period_return": 0.0,
            "drawdown": 0.0,
            "fee_paid": 0.0,
        }
    ]
    exposure_rows: list[dict[str, Any]] = []
    market_rows: list[dict[str, Any]] = []
    estimated_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    safe_action_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []

    trade_count = 0
    turnover = 0.0
    minor_intervention_count = 0
    target_clip_count = 0
    turnover_clip_count = 0
    hard_intervention_count = 0
    any_intervention_count = 0
    kill_switch_count = 0
    reduce_only_count = 0
    cooldown_blocked_trade_count = 0

    # The final row cannot produce a trade at next close, so decisions stop at n - 2.
    for idx in range(len(bars) - 1):
        decision_timestamp = timestamps[idx]
        execution_timestamp = timestamps[idx + 1]
        close_t = closes[idx]
        close_next = closes[idx + 1]

        market = build_market_state(bars[: idx + 1])
        estimated = estimate_market_state(market, previous_estimated)
        previous_estimated = estimated

        portfolio_drawdown = equity / peak_equity - 1.0
        portfolio = PortfolioStateV1(
            current_exposure=current_exposure,
            current_position=current_exposure * equity / close_t if close_t > 0 else 0.0,
            equity=equity,
            cash=equity * max(0.0, 1.0 - abs(current_exposure)),
            unrealized_pnl=0.0,
            portfolio_drawdown=portfolio_drawdown,
            leverage=abs(current_exposure),
            available_margin=max(0.0, 1.0 - abs(current_exposure)),
        )
        action = compute_control_action(
            market,
            estimated,
            portfolio,
            long_only=long_only,
            previous_target_exposure=previous_target_exposure,
            use_target_smoothing=use_target_smoothing,
            target_deadband=target_deadband,
            beta_increase_risk=beta_increase_risk,
            beta_decrease_risk=beta_decrease_risk,
        )
        previous_target_exposure = action.target_exposure
        safe_action = apply_risk_filter(market, estimated, portfolio, action, config=risk_config, long_only=long_only)

        requested_change = safe_action.safe_exposure_change if safe_action.trade_allowed else 0.0
        cooldown_blocked = False
        if requested_change != 0.0 and _should_apply_cooldown(
            idx=idx,
            last_nonzero_trade_idx=last_nonzero_trade_idx,
            minimum_rebalance_interval=minimum_rebalance_interval,
            current_exposure=current_exposure,
            safe_action=safe_action,
        ):
            requested_change = 0.0
            cooldown_blocked = True
            cooldown_blocked_trade_count += 1
        requested_change = _clip(requested_change, -1.0 - current_exposure, 1.0 - current_exposure)
        if long_only:
            requested_change = _clip(requested_change, -current_exposure, 1.0 - current_exposure)

        asset_return = close_next / close_t - 1.0 if close_t > 0 else 0.0
        equity_before_period = equity
        equity *= 1.0 + current_exposure * asset_return

        fee_paid = abs(requested_change) * fee_rate * equity
        equity -= fee_paid
        equity = max(equity, 1e-12)
        peak_equity = max(peak_equity, equity)

        previous_exposure = current_exposure
        current_exposure = _clip(current_exposure + requested_change, -1.0, 1.0)
        if long_only:
            current_exposure = _clip(current_exposure, 0.0, 1.0)

        if abs(requested_change) > 0.0:
            trade_count += 1
            turnover += abs(requested_change)
            last_nonzero_trade_idx = idx
        intervention = _classify_layer4_intervention(action, safe_action)
        if intervention["minor"]:
            minor_intervention_count += 1
        if intervention["target_clip"]:
            target_clip_count += 1
        if intervention["turnover_clip"]:
            turnover_clip_count += 1
        if intervention["hard"]:
            hard_intervention_count += 1
        if any(intervention.values()):
            any_intervention_count += 1
        if safe_action.kill_switch:
            kill_switch_count += 1
        if safe_action.reduce_only:
            reduce_only_count += 1

        period_return = equity / equity_before_period - 1.0
        drawdown = equity / peak_equity - 1.0

        market_rows.append(_record(decision_timestamp, market))
        estimated_rows.append(_record(decision_timestamp, estimated))
        action_rows.append(_record(decision_timestamp, action))
        safe_action_rows.append(_record(decision_timestamp, safe_action))
        exposure_rows.append(
            {
                "decision_timestamp": decision_timestamp,
                "execution_timestamp": execution_timestamp,
                "previous_exposure": previous_exposure,
                "safe_exposure_change": requested_change,
                "current_exposure": current_exposure,
                "position_value": current_exposure * equity,
                "close": close_next,
                "allowed_turnover": safe_action.allowed_turnover,
                "raw_target_exposure": action.raw_target_exposure,
                "smoothed_target_exposure": action.smoothed_target_exposure,
                "final_target_exposure": action.target_exposure,
                "safe_target_exposure": safe_action.safe_target_exposure,
                "cooldown_blocked": cooldown_blocked,
            }
        )
        equity_rows.append(
            {
                "timestamp": execution_timestamp,
                "close": close_next,
                "equity": equity,
                "period_return": period_return,
                "drawdown": drawdown,
                "fee_paid": fee_paid,
            }
        )
        if abs(requested_change) > 0.0:
            trade_rows.append(
                {
                    "decision_timestamp": decision_timestamp,
                    "execution_timestamp": execution_timestamp,
                    "execution_close": close_next,
                    "exposure_change": requested_change,
                    "exposure_before": previous_exposure,
                    "exposure_after": current_exposure,
                    "fee_paid": fee_paid,
                    "turnover": abs(requested_change),
                }
            )

    equity_curve = pd.DataFrame(equity_rows)
    metrics = _calculate_metrics(
        equity_curve,
        initial_equity=initial_equity,
        periods_per_year=periods_per_year,
        trade_count=trade_count,
        turnover=turnover,
        exposure_history=pd.DataFrame(exposure_rows),
        minor_intervention_count=minor_intervention_count,
        target_clip_count=target_clip_count,
        turnover_clip_count=turnover_clip_count,
        hard_intervention_count=hard_intervention_count,
        any_intervention_count=any_intervention_count,
        decision_count=max(len(bars) - 1, 1),
        kill_switch_count=kill_switch_count,
        reduce_only_count=reduce_only_count,
        cooldown_blocked_trade_count=cooldown_blocked_trade_count,
    )

    return BacktestResult(
        metrics=metrics,
        equity_curve=equity_curve,
        exposure_history=pd.DataFrame(exposure_rows),
        market_state_history=pd.DataFrame(market_rows),
        estimated_state_history=pd.DataFrame(estimated_rows),
        control_action_history=pd.DataFrame(action_rows),
        safe_control_action_history=pd.DataFrame(safe_action_rows),
        trade_history=pd.DataFrame(trade_rows),
    )


def run_backtest_fast(
    ohlcv: pd.DataFrame,
    *,
    fee_rate: float = 0.0005,
    initial_equity: float = 1.0,
    periods_per_year: int = 252,
    long_only: bool = False,
    risk_config: RiskConfigV1 | None = None,
    minimum_rebalance_interval: int = 3,
    use_target_smoothing: bool = True,
    target_deadband: float = 0.05,
    beta_increase_risk: float = 0.30,
    beta_decrease_risk: float = 0.60,
    progress_every: int | None = None,
) -> BacktestResult:
    """Run the v1 backtest with cached/vectorized Layer 1 and 2 state frames."""

    _validate_ohlcv(ohlcv)
    if len(ohlcv) < 2:
        raise ValueError("at least two OHLCV rows are required")
    if fee_rate < 0:
        raise ValueError("fee_rate must be non-negative")
    if initial_equity <= 0:
        raise ValueError("initial_equity must be positive")

    closes = pd.to_numeric(ohlcv["close"], errors="coerce").astype(float).to_numpy()
    timestamps = _timestamps(ohlcv)
    market_states = _build_exact_market_state_cache(ohlcv)
    estimated_states: list[EstimatedMarketStateV1] = []
    previous_estimated: EstimatedMarketStateV1 | None = None
    for market in market_states:
        estimated = estimate_market_state(market, previous_estimated)
        estimated_states.append(estimated)
        previous_estimated = estimated

    equity = float(initial_equity)
    peak_equity = float(initial_equity)
    current_exposure = 0.0
    previous_target_exposure: float | None = None
    last_nonzero_trade_idx: int | None = None

    equity_rows: list[dict[str, Any]] = [
        {
            "timestamp": timestamps[0],
            "close": closes[0],
            "equity": equity,
            "period_return": 0.0,
            "drawdown": 0.0,
            "fee_paid": 0.0,
        }
    ]
    exposure_rows: list[dict[str, Any]] = []
    market_rows: list[dict[str, Any]] = []
    estimated_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    safe_action_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []

    trade_count = 0
    turnover = 0.0
    minor_intervention_count = 0
    target_clip_count = 0
    turnover_clip_count = 0
    hard_intervention_count = 0
    any_intervention_count = 0
    kill_switch_count = 0
    reduce_only_count = 0
    cooldown_blocked_trade_count = 0
    decision_count = len(ohlcv) - 1

    for idx in range(decision_count):
        if progress_every and idx > 0 and idx % progress_every == 0:
            print(f"run_backtest_fast progress: {idx}/{decision_count}")

        decision_timestamp = timestamps[idx]
        execution_timestamp = timestamps[idx + 1]
        close_t = closes[idx]
        close_next = closes[idx + 1]

        market = market_states[idx]
        estimated = estimated_states[idx]

        portfolio_drawdown = equity / peak_equity - 1.0
        portfolio = PortfolioStateV1(
            current_exposure=current_exposure,
            current_position=current_exposure * equity / close_t if close_t > 0 else 0.0,
            equity=equity,
            cash=equity * max(0.0, 1.0 - abs(current_exposure)),
            unrealized_pnl=0.0,
            portfolio_drawdown=portfolio_drawdown,
            leverage=abs(current_exposure),
            available_margin=max(0.0, 1.0 - abs(current_exposure)),
        )
        action = compute_control_action(
            market,
            estimated,
            portfolio,
            long_only=long_only,
            previous_target_exposure=previous_target_exposure,
            use_target_smoothing=use_target_smoothing,
            target_deadband=target_deadband,
            beta_increase_risk=beta_increase_risk,
            beta_decrease_risk=beta_decrease_risk,
        )
        previous_target_exposure = action.target_exposure
        safe_action = apply_risk_filter(market, estimated, portfolio, action, config=risk_config, long_only=long_only)

        requested_change = safe_action.safe_exposure_change if safe_action.trade_allowed else 0.0
        cooldown_blocked = False
        if requested_change != 0.0 and _should_apply_cooldown(
            idx=idx,
            last_nonzero_trade_idx=last_nonzero_trade_idx,
            minimum_rebalance_interval=minimum_rebalance_interval,
            current_exposure=current_exposure,
            safe_action=safe_action,
        ):
            requested_change = 0.0
            cooldown_blocked = True
            cooldown_blocked_trade_count += 1
        requested_change = _clip(requested_change, -1.0 - current_exposure, 1.0 - current_exposure)
        if long_only:
            requested_change = _clip(requested_change, -current_exposure, 1.0 - current_exposure)

        asset_return = close_next / close_t - 1.0 if close_t > 0 else 0.0
        equity_before_period = equity
        equity *= 1.0 + current_exposure * asset_return

        fee_paid = abs(requested_change) * fee_rate * equity
        equity -= fee_paid
        equity = max(equity, 1e-12)
        peak_equity = max(peak_equity, equity)

        previous_exposure = current_exposure
        current_exposure = _clip(current_exposure + requested_change, -1.0, 1.0)
        if long_only:
            current_exposure = _clip(current_exposure, 0.0, 1.0)

        if abs(requested_change) > 0.0:
            trade_count += 1
            turnover += abs(requested_change)
            last_nonzero_trade_idx = idx
        intervention = _classify_layer4_intervention(action, safe_action)
        if intervention["minor"]:
            minor_intervention_count += 1
        if intervention["target_clip"]:
            target_clip_count += 1
        if intervention["turnover_clip"]:
            turnover_clip_count += 1
        if intervention["hard"]:
            hard_intervention_count += 1
        if any(intervention.values()):
            any_intervention_count += 1
        if safe_action.kill_switch:
            kill_switch_count += 1
        if safe_action.reduce_only:
            reduce_only_count += 1

        period_return = equity / equity_before_period - 1.0
        drawdown = equity / peak_equity - 1.0

        market_rows.append(_record(decision_timestamp, market))
        estimated_rows.append(_record(decision_timestamp, estimated))
        action_rows.append(_record(decision_timestamp, action))
        safe_action_rows.append(_record(decision_timestamp, safe_action))
        exposure_rows.append(
            {
                "decision_timestamp": decision_timestamp,
                "execution_timestamp": execution_timestamp,
                "previous_exposure": previous_exposure,
                "safe_exposure_change": requested_change,
                "current_exposure": current_exposure,
                "position_value": current_exposure * equity,
                "close": close_next,
                "allowed_turnover": safe_action.allowed_turnover,
                "raw_target_exposure": action.raw_target_exposure,
                "smoothed_target_exposure": action.smoothed_target_exposure,
                "final_target_exposure": action.target_exposure,
                "safe_target_exposure": safe_action.safe_target_exposure,
                "cooldown_blocked": cooldown_blocked,
            }
        )
        equity_rows.append(
            {
                "timestamp": execution_timestamp,
                "close": close_next,
                "equity": equity,
                "period_return": period_return,
                "drawdown": drawdown,
                "fee_paid": fee_paid,
            }
        )
        if abs(requested_change) > 0.0:
            trade_rows.append(
                {
                    "decision_timestamp": decision_timestamp,
                    "execution_timestamp": execution_timestamp,
                    "execution_close": close_next,
                    "exposure_change": requested_change,
                    "exposure_before": previous_exposure,
                    "exposure_after": current_exposure,
                    "fee_paid": fee_paid,
                    "turnover": abs(requested_change),
                }
            )

    equity_curve = pd.DataFrame(equity_rows)
    exposure_history = pd.DataFrame(exposure_rows)
    metrics = _calculate_metrics(
        equity_curve,
        initial_equity=initial_equity,
        periods_per_year=periods_per_year,
        trade_count=trade_count,
        turnover=turnover,
        exposure_history=exposure_history,
        minor_intervention_count=minor_intervention_count,
        target_clip_count=target_clip_count,
        turnover_clip_count=turnover_clip_count,
        hard_intervention_count=hard_intervention_count,
        any_intervention_count=any_intervention_count,
        decision_count=max(decision_count, 1),
        kill_switch_count=kill_switch_count,
        reduce_only_count=reduce_only_count,
        cooldown_blocked_trade_count=cooldown_blocked_trade_count,
    )

    return BacktestResult(
        metrics=metrics,
        equity_curve=equity_curve,
        exposure_history=exposure_history,
        market_state_history=pd.DataFrame(market_rows),
        estimated_state_history=pd.DataFrame(estimated_rows),
        control_action_history=pd.DataFrame(action_rows),
        safe_control_action_history=pd.DataFrame(safe_action_rows),
        trade_history=pd.DataFrame(trade_rows),
    )


def run_backtest_from_csv(
    csv_path: str | Path | None = None,
    *,
    fee_rate: float = 0.0005,
    initial_equity: float = 1.0,
    periods_per_year: int = 252,
    long_only: bool = False,
) -> BacktestResult:
    """Load a CSV from data/ and run the backtest."""

    path = find_data_csv() if csv_path is None else Path(csv_path)
    return run_backtest(
        load_ohlcv_csv(path),
        fee_rate=fee_rate,
        initial_equity=initial_equity,
        periods_per_year=periods_per_year,
        long_only=long_only,
    )


def write_backtest_outputs(result: BacktestResult, output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result.metrics]).to_csv(output_path / "metrics.csv", index=False)
    result.equity_curve.to_csv(output_path / "equity_curve.csv", index=False)
    result.exposure_history.to_csv(output_path / "exposure_history.csv", index=False)
    result.market_state_history.to_csv(output_path / "market_state_history.csv", index=False)
    result.estimated_state_history.to_csv(output_path / "estimated_state_history.csv", index=False)
    result.control_action_history.to_csv(output_path / "control_action_history.csv", index=False)
    result.safe_control_action_history.to_csv(output_path / "safe_control_action_history.csv", index=False)
    result.trade_history.to_csv(output_path / "trade_history.csv", index=False)


def _validate_ohlcv(ohlcv: pd.DataFrame) -> None:
    missing = set(REQUIRED_OHLCV_COLUMNS).difference(ohlcv.columns)
    if missing:
        raise ValueError(f"OHLCV data is missing required columns: {sorted(missing)}")
    numeric = ohlcv.loc[:, REQUIRED_OHLCV_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("OHLCV columns must be numeric and non-null")
    if (numeric["close"] <= 0).any():
        raise ValueError("close must be positive")
    if (numeric["volume"] < 0).any():
        raise ValueError("volume must be non-negative")


def _frame_to_bars(ohlcv: pd.DataFrame) -> list[OHLCVBar]:
    return [
        OHLCVBar(
            timestamp=float(idx),
            open=float(row.open),
            high=float(row.high),
            low=float(row.low),
            close=float(row.close),
            volume=float(row.volume),
        )
        for idx, row in enumerate(ohlcv.loc[:, REQUIRED_OHLCV_COLUMNS].itertuples(index=False))
    ]


def _timestamps(ohlcv: pd.DataFrame) -> list[Any]:
    if "timestamp" in ohlcv.columns:
        return ohlcv["timestamp"].tolist()
    return ohlcv.index.tolist()


def _record(timestamp: Any, state: Any) -> dict[str, Any]:
    row = asdict(state)
    row["timestamp"] = timestamp
    return row


def _build_exact_market_state_cache(ohlcv: pd.DataFrame, trend_k: float = 10.0) -> list[MarketStateV1]:
    """Precompute MarketStateV1 values with the same formulas as build_market_state."""

    close = pd.to_numeric(ohlcv["close"], errors="coerce").astype(float).to_numpy()
    high = pd.to_numeric(ohlcv["high"], errors="coerce").astype(float).to_numpy()
    low = pd.to_numeric(ohlcv["low"], errors="coerce").astype(float).to_numpy()
    volume = pd.to_numeric(ohlcv["volume"], errors="coerce").astype(float).clip(lower=0.0).to_numpy()

    returns: list[float] = []
    for idx in range(1, len(close)):
        if close[idx - 1] > 0.0 and close[idx] > 0.0:
            returns.append(log(close[idx] / close[idx - 1]))

    vol_history = [_safe_std_np(returns[max(0, idx - 19) : idx + 1]) for idx in range(len(returns))]
    illiquidity_history: list[float] = []
    for idx in range(len(close)):
        avg_volume = _safe_mean_np(volume[max(0, idx - 59) : idx + 1], default=volume[idx])
        normalized_volume = volume[idx] / avg_volume if avg_volume > 0.0 else 1.0
        price_range = (high[idx] - low[idx]) / close[idx] if close[idx] > 0.0 else 0.0
        illiquidity_history.append(price_range / normalized_volume if normalized_volume > 0.0 else 1.0)

    states: list[MarketStateV1] = []
    for idx in range(len(close)):
        bar_close = close[idx]
        previous_close = close[idx - 1] if idx > 0 and close[idx - 1] > 0.0 else bar_close
        return_1 = log(bar_close / previous_close) if bar_close > 0.0 and previous_close > 0.0 else 0.0
        recent_returns = returns[:idx]
        recent_returns_20 = recent_returns[-20:]
        recent_returns_60 = recent_returns[-60:]
        volatility = _safe_std_np(recent_returns_20)
        recent_vol_history = vol_history[:idx][-120:]
        vol_baseline = _safe_median_np(recent_vol_history, default=volatility or 1.0)
        volatility_score = volatility / vol_baseline if vol_baseline > 0.0 else 1.0

        positive_closes = close[: idx + 1][close[: idx + 1] > 0.0]
        ma_short = _safe_mean_np(positive_closes[-20:], default=bar_close)
        ma_long = _safe_mean_np(positive_closes[-60:], default=bar_close)
        trend_raw = (ma_short - ma_long) / ma_long if ma_long else 0.0
        trend_score = tanh(trend_k * trend_raw)

        volume_window = volume[max(0, idx - 59) : idx + 1]
        volume_mean = _safe_mean_np(volume_window, default=volume[idx])
        volume_std = _safe_std_np(volume_window)
        volume_z = (volume[idx] - volume_mean) / volume_std if volume_std > 0.0 else 0.0
        volume_score = _clip(abs(volume_z) / 3.0, 0.0, 1.0)

        price_range = (high[idx] - low[idx]) / bar_close if bar_close > 0.0 else 0.0
        normalized_volume = volume[idx] / volume_mean if volume_mean > 0.0 else 1.0
        illiquidity = price_range / normalized_volume if normalized_volume > 0.0 else 1.0
        illiquidity_baseline = _safe_median_np(illiquidity_history[max(0, idx - 119) : idx + 1], default=illiquidity or 1.0)
        normalized_illiquidity = illiquidity / (3.0 * illiquidity_baseline) if illiquidity_baseline > 0.0 else 1.0
        liquidity_score = 1.0 - _clip(normalized_illiquidity, 0.0, 1.0)

        rolling_high = float(np.max(positive_closes[-120:])) if len(positive_closes) else bar_close
        drawdown = bar_close / rolling_high - 1.0 if rolling_high > 0.0 else 0.0

        return_std = _safe_std_np(recent_returns_60)
        return_z = abs(return_1) / return_std if return_std > 0.0 else 0.0
        shock_score = _clip(return_z / 4.0, 0.0, 1.0)

        missing_data_penalty = 0.0 if bar_close > 0.0 and high[idx] >= low[idx] and volume[idx] >= 0.0 else 1.0
        volatility_penalty = _clip((volatility_score - 1.5) / 2.0, 0.0, 1.0)
        illiquidity_penalty = 1.0 - liquidity_score
        confidence = 1.0 - 0.3 * shock_score - 0.3 * volatility_penalty - 0.2 * illiquidity_penalty - 0.2 * missing_data_penalty
        confidence = _clip(confidence, 0.0, 1.0)
        market_mode = classify_market_mode(volatility_score, trend_score, liquidity_score, shock_score)

        states.append(
            MarketStateV1(
                timestamp=float(idx),
                close=float(bar_close),
                return_1=float(return_1),
                volatility=float(volatility),
                volatility_score=float(volatility_score),
                trend_raw=float(trend_raw),
                trend_score=float(trend_score),
                volume_z=float(volume_z),
                volume_score=float(volume_score),
                price_range=float(price_range),
                liquidity_score=float(liquidity_score),
                drawdown=float(drawdown),
                shock_score=float(shock_score),
                confidence=float(confidence),
                market_mode=market_mode,
            )
        )
    return states


def _safe_mean_np(values: Any, default: float = 0.0) -> float:
    return float(mean(values)) if len(values) else default


def _safe_std_np(values: Any, default: float = 0.0) -> float:
    return float(pstdev(values)) if len(values) > 1 else default


def _safe_median_np(values: Any, default: float = 1.0) -> float:
    return float(median(values)) if len(values) else default


def _market_state_from_row(row: pd.Series) -> MarketStateV1:
    return MarketStateV1(
        timestamp=row["timestamp"],
        close=float(row["close"]),
        return_1=float(row["return_1"]),
        volatility=float(row["volatility"]),
        volatility_score=float(row["volatility_score"]),
        trend_raw=float(row["trend_raw"]),
        trend_score=float(row["trend_score"]),
        volume_z=float(row["volume_z"]),
        volume_score=float(row["volume_score"]),
        price_range=float(row["price_range"]),
        liquidity_score=float(row["liquidity_score"]),
        drawdown=float(row["drawdown"]),
        shock_score=float(row["shock_score"]),
        confidence=float(row["confidence"]),
        market_mode=str(row["market_mode"]),
    )


def _estimated_state_from_row(row: pd.Series) -> EstimatedMarketStateV1:
    return EstimatedMarketStateV1(
        p_bull=float(row["p_bull"]),
        p_bear=float(row["p_bear"]),
        p_sideways=float(row["p_sideways"]),
        p_high_vol=float(row["p_high_vol"]),
        p_crash_risk=float(row["p_crash_risk"]),
        dominant_regime=str(row["dominant_regime"]),
        state_confidence=float(row["state_confidence"]),
        regime_uncertainty=float(row["regime_uncertainty"]),
        transition_risk=float(row["transition_risk"]),
        danger_score=float(row["danger_score"]),
    )


def _classify_layer4_intervention(action: ControlActionV1, safe_action: SafeControlActionV1) -> dict[str, bool]:
    target_delta = abs(safe_action.safe_target_exposure - action.target_exposure)
    turnover_delta = abs(safe_action.safe_exposure_change - action.exposure_change)
    hard = safe_action.reduce_only or safe_action.kill_switch or safe_action.emergency_deleveraging
    return {
        "minor": 1e-12 < target_delta < 0.01 and not hard,
        "target_clip": target_delta >= 0.01,
        "turnover_clip": turnover_delta >= 0.01,
        "hard": hard,
    }


def _should_apply_cooldown(
    *,
    idx: int,
    last_nonzero_trade_idx: int | None,
    minimum_rebalance_interval: int,
    current_exposure: float,
    safe_action: SafeControlActionV1,
) -> bool:
    if minimum_rebalance_interval <= 0 or last_nonzero_trade_idx is None:
        return False
    if idx - last_nonzero_trade_idx >= minimum_rebalance_interval:
        return False
    if safe_action.reduce_only or safe_action.kill_switch or safe_action.emergency_deleveraging:
        return False
    if abs(safe_action.safe_target_exposure) < abs(current_exposure):
        return False
    if abs(current_exposure) > safe_action.allowed_max_exposure:
        return False
    return True


def _calculate_metrics(
    equity_curve: pd.DataFrame,
    *,
    initial_equity: float,
    periods_per_year: int,
    trade_count: int,
    turnover: float,
    exposure_history: pd.DataFrame,
    minor_intervention_count: int,
    target_clip_count: int,
    turnover_clip_count: int,
    hard_intervention_count: int,
    any_intervention_count: int,
    decision_count: int,
    kill_switch_count: int,
    reduce_only_count: int,
    cooldown_blocked_trade_count: int,
) -> dict[str, float | int]:
    final_equity = float(equity_curve["equity"].iloc[-1])
    total_return = final_equity / initial_equity - 1.0
    periods = max(len(equity_curve) - 1, 1)
    annualized_return = (final_equity / initial_equity) ** (periods_per_year / periods) - 1.0
    max_drawdown = float(equity_curve["drawdown"].min())

    returns = equity_curve["period_return"].iloc[1:].astype(float)
    return_std = float(returns.std(ddof=0))
    sharpe = 0.0 if return_std == 0.0 else float(returns.mean() / return_std * np.sqrt(periods_per_year))
    average_exposure = 0.0
    if not exposure_history.empty:
        average_exposure = float(exposure_history["current_exposure"].abs().mean())
    average_allowed_turnover = 0.0
    average_raw_target_exposure = 0.0
    average_safe_target_exposure = 0.0
    average_smoothed_target_exposure = 0.0
    average_final_target_exposure = 0.0
    if "allowed_turnover" in exposure_history:
        average_allowed_turnover = float(exposure_history["allowed_turnover"].mean())
    if "raw_target_exposure" in exposure_history:
        average_raw_target_exposure = float(exposure_history["raw_target_exposure"].abs().mean())
    if "safe_target_exposure" in exposure_history:
        average_safe_target_exposure = float(exposure_history["safe_target_exposure"].abs().mean())
    if "smoothed_target_exposure" in exposure_history:
        average_smoothed_target_exposure = float(exposure_history["smoothed_target_exposure"].abs().mean())
    if "final_target_exposure" in exposure_history:
        average_final_target_exposure = float(exposure_history["final_target_exposure"].abs().mean())

    return {
        "initial_equity": float(initial_equity),
        "final_equity": final_equity,
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "trade_count": int(trade_count),
        "turnover": float(turnover),
        "average_exposure": average_exposure,
        "average_allowed_turnover": average_allowed_turnover,
        "average_raw_target_exposure": average_raw_target_exposure,
        "average_smoothed_target_exposure": average_smoothed_target_exposure,
        "average_final_target_exposure": average_final_target_exposure,
        "average_safe_target_exposure": average_safe_target_exposure,
        "layer4_intervention_rate": float(any_intervention_count / decision_count),
        "minor_intervention_rate": float(minor_intervention_count / decision_count),
        "target_clip_rate": float(target_clip_count / decision_count),
        "turnover_clip_rate": float(turnover_clip_count / decision_count),
        "hard_intervention_rate": float(hard_intervention_count / decision_count),
        "kill_switch_count": int(kill_switch_count),
        "reduce_only_count": int(reduce_only_count),
        "cooldown_blocked_trade_count": int(cooldown_blocked_trade_count),
        "bars": int(len(equity_curve)),
    }


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Layer 1-4 v1 backtester.")
    parser.add_argument("--csv", type=Path, default=None, help="OHLCV CSV path. Defaults to first CSV under data/.")
    parser.add_argument("--fee-rate", type=float, default=0.0005, help="Fee rate charged on absolute exposure change.")
    parser.add_argument("--initial-equity", type=float, default=1.0)
    parser.add_argument("--periods-per-year", type=int, default=252)
    parser.add_argument("--long-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional directory for metrics and history CSV outputs.")
    args = parser.parse_args()

    result = run_backtest_from_csv(
        args.csv,
        fee_rate=args.fee_rate,
        initial_equity=args.initial_equity,
        periods_per_year=args.periods_per_year,
        long_only=args.long_only,
    )
    if args.output_dir is not None:
        write_backtest_outputs(result, args.output_dir)

    for key, value in result.metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

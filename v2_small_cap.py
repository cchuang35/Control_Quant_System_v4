"""v2.0 small-cap wrapper around v1.final outputs.

This module intentionally does not modify the v1 Layer 1-4 pipeline.  It
expects a DataFrame that already contains v1 output columns, then adds an
outer regime detector, binary trade gate, drawdown gate, and fee-aware
backtest columns.

Observed v1.final outputs from ``backtester.BacktestResult``:
- ``exposure_history.current_exposure``
- ``exposure_history.safe_exposure_change``
- ``exposure_history.final_target_exposure``
- ``exposure_history.safe_target_exposure``
- ``equity_curve.period_return``
- ``equity_curve.drawdown``
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


REGIME_SCORE = {
    "strong_bull": 1.0,
    "weak_bull": 0.5,
    "sideways": 0.0,
    "bear": -1.0,
}

V1_POSITION_COLUMNS = ("v1_position", "position", "signal")
V1_EXPOSURE_COLUMNS = (
    "v1_exposure",
    "exposure",
    "current_exposure",
    "safe_target_exposure",
    "final_target_exposure",
)
SUMMARY_WINDOWS = {
    "Full period": None,
    "Recent 180d": 180,
    "Recent 365d": 365,
    "Recent 2y": 730,
    "Recent 3y": 1095,
}
SUMMARY_COLUMNS = [
    "version",
    "window",
    "total_return_net",
    "annualized_return_net",
    "max_drawdown",
    "Sharpe_net",
    "Sortino_net",
    "Calmar",
    "average_exposure",
    "turnover",
    "total_trades",
    "total_fee_paid",
]
TIME_COLUMNS = ("timestamp", "execution_timestamp", "decision_timestamp", "open_time", "date", "datetime", "time")


def compute_regime_features(df: pd.DataFrame, confirmation_days: int = 3) -> pd.DataFrame:
    """Add v2 regime detector columns using only current and historical closes."""

    _require_columns(df, ["close"])
    result = df.copy()
    close = pd.to_numeric(result["close"], errors="coerce").astype(float)
    if close.isna().any():
        raise ValueError("close must be numeric and non-null")
    if (close <= 0).any():
        raise ValueError("close must be positive")

    daily_return = close.pct_change()
    result["asset_return"] = daily_return
    result["MA20"] = close.rolling(20).mean()
    result["MA60"] = close.rolling(60).mean()
    result["momentum_20"] = close / close.shift(20) - 1.0
    result["vol20"] = daily_return.rolling(20).std() * np.sqrt(365)
    result["vol60"] = daily_return.rolling(60).std() * np.sqrt(365)
    result["raw_regime"] = result.apply(detect_raw_regime, axis=1)
    result["confirmed_regime"] = confirm_regime(result["raw_regime"], confirmation_days=confirmation_days)
    result["regime_score"] = result["confirmed_regime"].map(REGIME_SCORE).astype(float)
    return result


def detect_raw_regime(row: pd.Series | dict[str, Any]) -> str:
    """Classify one row into a raw v2 regime."""

    close = _as_float(row, "close")
    ma20 = _as_float(row, "MA20")
    ma60 = _as_float(row, "MA60")
    momentum_20 = _as_float(row, "momentum_20")
    vol20 = _as_float(row, "vol20")
    vol60 = _as_float(row, "vol60")

    if not all(np.isfinite(value) for value in [close, ma20, ma60, momentum_20, vol20, vol60]):
        return "sideways"

    if close > ma60 and ma20 > ma60 and momentum_20 > 0.05 and vol20 <= 1.5 * vol60:
        return "strong_bull"
    if close > ma60 and momentum_20 > 0.0:
        return "weak_bull"
    if close < ma60 and ma20 < ma60:
        return "bear"
    return "sideways"


def confirm_regime(raw_regime_series: pd.Series, confirmation_days: int = 3) -> pd.Series:
    """Confirm a regime only after it appears for consecutive observations."""

    if confirmation_days <= 0:
        raise ValueError("confirmation_days must be positive")
    if raw_regime_series.empty:
        return raw_regime_series.copy()

    confirmed: list[str] = []
    current_confirmed = str(raw_regime_series.iloc[0])
    streak_regime = current_confirmed
    streak_count = 0

    for raw_regime in raw_regime_series.astype(str):
        if raw_regime == streak_regime:
            streak_count += 1
        else:
            streak_regime = raw_regime
            streak_count = 1
        if streak_count >= confirmation_days:
            current_confirmed = streak_regime
        confirmed.append(current_confirmed)

    return pd.Series(confirmed, index=raw_regime_series.index, name="confirmed_regime")


def apply_trade_gate(
    v1_position: int | float,
    confirmed_regime: str,
    previous_position: int | float,
    *,
    gate_mode: str = "sideways_hold",
) -> int:
    """Apply the v2 trade gate to binary v1 and previous positions."""

    if gate_mode not in {"sideways_hold", "strict_sideways"}:
        raise ValueError("gate_mode must be 'sideways_hold' or 'strict_sideways'")

    v1_binary = _validate_binary_position(v1_position, name="v1_position")
    previous_binary = _validate_binary_position(previous_position, name="previous_position")

    if confirmed_regime in {"strong_bull", "weak_bull"}:
        return v1_binary
    if gate_mode == "strict_sideways":
        return 0
    if confirmed_regime == "sideways":
        return 1 if previous_binary == 1 and v1_binary == 1 else 0
    return 0


def apply_drawdown_risk_gate(position: int | float, confirmed_regime: str, strategy_drawdown: float) -> int:
    """Apply the discrete drawdown risk gate."""

    binary_position = _validate_binary_position(position, name="position")
    drawdown = float(strategy_drawdown)

    if drawdown > -0.10:
        return binary_position
    if drawdown > -0.15:
        return binary_position if confirmed_regime == "strong_bull" else 0
    return 0


def regime_entry_hold_permissions(confirmed_regime: str) -> tuple[bool, bool]:
    """Return ``(allow_entry, allow_hold)`` for the v2.1 regime gate."""

    if confirmed_regime in {"strong_bull", "weak_bull"}:
        return True, True
    if confirmed_regime == "sideways":
        return False, True
    return False, False


def decide_v21_position(
    *,
    previous_position: int | float,
    v1_position_signal: int | float,
    confirmed_regime: str,
    allow_new_entries: bool = True,
) -> int:
    """Decide the next binary position using v2.1 entry/hold regime permissions."""

    previous_binary = _validate_binary_position(previous_position, name="previous_position")
    signal_binary = _validate_binary_position(v1_position_signal, name="v1_position_signal")
    allow_entry, allow_hold = regime_entry_hold_permissions(confirmed_regime)

    if previous_binary == 0:
        return 1 if allow_new_entries and allow_entry and signal_binary == 1 else 0
    if not allow_hold:
        return 0
    if signal_binary == 0:
        return 0
    return 1


def backtest_v21_small_cap(
    df: pd.DataFrame,
    fee_rate: float = 0.001,
    *,
    v1_entry_threshold: float = 0.10,
    confirmation_days: int = 3,
    use_drawdown_gate: bool = True,
    warning_drawdown: float = -0.03,
    exit_drawdown: float = -0.05,
    cooldown_bars: int = 24,
) -> pd.DataFrame:
    """Run the v2.1 small-cap risk-gate backtest.

    The regime detector is unchanged from v2.0.  Signals at row ``t`` set the
    binary position held after row ``t``; row ``t`` returns are earned only by
    the prior row's position.
    """

    if fee_rate < 0.0:
        raise ValueError("fee_rate must be non-negative")
    if not 0.0 <= v1_entry_threshold <= 1.0:
        raise ValueError("v1_entry_threshold must be between 0 and 1")
    if cooldown_bars < 0:
        raise ValueError("cooldown_bars must be non-negative")
    if not exit_drawdown < warning_drawdown < 0.0:
        raise ValueError("drawdown thresholds must satisfy exit_drawdown < warning_drawdown < 0")

    result = compute_regime_features(df, confirmation_days=confirmation_days)
    result["v1_position"] = _extract_v1_position(result, v1_entry_threshold=v1_entry_threshold)
    result["asset_return"] = result["asset_return"].fillna(0.0)

    position_before_dd_gate: list[int] = []
    final_positions: list[int] = []
    trade_sizes: list[float] = []
    fee_costs: list[float] = []
    gross_returns: list[float] = []
    net_returns: list[float] = []
    equity_values: list[float] = []
    equity_peaks: list[float] = []
    strategy_drawdowns: list[float] = []
    drawdowns: list[float] = []
    drawdown_gate_triggered: list[bool] = []
    cooldown_active: list[bool] = []
    cooldown_remaining_values: list[int] = []
    entries_disabled_by_dd: list[bool] = []

    previous_position = 0
    equity = 1.0
    equity_peak = 1.0
    cooldown_remaining = 0

    for row in result.itertuples(index=False):
        asset_return = float(getattr(row, "asset_return"))
        confirmed_regime = str(getattr(row, "confirmed_regime"))
        v1_signal = int(getattr(row, "v1_position"))

        gross_return = previous_position * asset_return
        equity_before_trade = equity * (1.0 + gross_return)
        equity_peak = max(equity_peak, equity_before_trade)
        strategy_drawdown = equity_before_trade / equity_peak - 1.0

        in_cooldown = use_drawdown_gate and cooldown_remaining > 0
        disable_new_entries = use_drawdown_gate and strategy_drawdown <= warning_drawdown
        base_position = decide_v21_position(
            previous_position=previous_position,
            v1_position_signal=v1_signal,
            confirmed_regime=confirmed_regime,
            allow_new_entries=not disable_new_entries,
        )

        triggered = False
        final_position = base_position
        if use_drawdown_gate:
            if in_cooldown:
                final_position = 0
            elif strategy_drawdown <= exit_drawdown:
                final_position = 0
                cooldown_remaining = cooldown_bars
                triggered = True

        trade_size = abs(final_position - previous_position)
        fee_cost = trade_size * fee_rate
        net_return = gross_return - fee_cost
        equity = equity * (1.0 + net_return)
        equity_peak = max(equity_peak, equity)
        current_drawdown = equity / equity_peak - 1.0

        position_before_dd_gate.append(base_position)
        final_positions.append(final_position)
        trade_sizes.append(float(trade_size))
        fee_costs.append(float(fee_cost))
        gross_returns.append(float(gross_return))
        net_returns.append(float(net_return))
        equity_values.append(float(equity))
        equity_peaks.append(float(equity_peak))
        strategy_drawdowns.append(float(strategy_drawdown))
        drawdowns.append(float(current_drawdown))
        drawdown_gate_triggered.append(triggered)
        cooldown_active.append(in_cooldown or (triggered and cooldown_bars > 0))
        cooldown_remaining_values.append(int(cooldown_remaining))
        entries_disabled_by_dd.append(bool(disable_new_entries and previous_position == 0))

        previous_position = final_position
        if cooldown_remaining > 0:
            cooldown_remaining -= 1

    result["position_before_dd_gate"] = position_before_dd_gate
    result["final_position"] = final_positions
    result["trade_size"] = trade_sizes
    result["fee_cost"] = fee_costs
    result["strategy_return_gross"] = gross_returns
    result["strategy_return_net"] = net_returns
    result["equity_net"] = equity_values
    result["equity_peak"] = equity_peaks
    result["strategy_drawdown"] = strategy_drawdowns
    result["drawdown"] = drawdowns
    result["drawdown_gate_triggered"] = drawdown_gate_triggered
    result["cooldown_active"] = cooldown_active
    result["cooldown_remaining"] = cooldown_remaining_values
    result["entries_disabled_by_dd"] = entries_disabled_by_dd
    return result


def compute_soft_dd_scale(current_drawdown: float, *, warning_dd: float = -0.03, hard_dd: float = -0.05, mode: str = "step") -> float:
    """Compute a soft drawdown scale for v2.2."""

    if not hard_dd < warning_dd < 0.0:
        raise ValueError("drawdown thresholds must satisfy hard_dd < warning_dd < 0")
    drawdown = float(current_drawdown)
    if drawdown > warning_dd:
        return 1.0
    if drawdown <= hard_dd:
        return 0.0
    if mode == "step":
        return 0.5
    if mode == "linear":
        scale = (drawdown - hard_dd) / (warning_dd - hard_dd)
        return float(np.clip(scale, 0.0, 1.0))
    raise ValueError("mode must be 'step' or 'linear'")


def backtest_v22_small_cap(
    df: pd.DataFrame,
    fee_rate: float = 0.001,
    *,
    v1_entry_threshold: float = 0.10,
    confirmation_days: int = 3,
    warning_dd: float = -0.03,
    hard_dd: float = -0.05,
    cooldown_bars: int = 24,
    dd_scale_mode: str = "step",
) -> pd.DataFrame:
    """Run v2.2 with sideways-hold-only regime logic and soft drawdown scaling."""

    if fee_rate < 0.0:
        raise ValueError("fee_rate must be non-negative")
    if not 0.0 <= v1_entry_threshold <= 1.0:
        raise ValueError("v1_entry_threshold must be between 0 and 1")
    if cooldown_bars < 0:
        raise ValueError("cooldown_bars must be non-negative")

    result = compute_regime_features(df, confirmation_days=confirmation_days)
    result["v1_position"] = _extract_v1_position(result, v1_entry_threshold=v1_entry_threshold)
    result["asset_return"] = result["asset_return"].fillna(0.0)

    position_before_dd_gate: list[int] = []
    final_positions: list[float] = []
    dd_scales: list[float] = []
    trade_sizes: list[float] = []
    fee_costs: list[float] = []
    gross_returns: list[float] = []
    net_returns: list[float] = []
    equity_values: list[float] = []
    equity_peaks: list[float] = []
    strategy_drawdowns: list[float] = []
    drawdowns: list[float] = []
    forced_exit_flags: list[bool] = []
    cooldown_active: list[bool] = []
    cooldown_remaining_values: list[int] = []
    entries_disabled_by_dd: list[bool] = []

    previous_position = 0.0
    previous_binary_position = 0
    equity = 1.0
    equity_peak = 1.0
    cooldown_remaining = 0

    for row in result.itertuples(index=False):
        asset_return = float(getattr(row, "asset_return"))
        confirmed_regime = str(getattr(row, "confirmed_regime"))
        v1_signal = int(getattr(row, "v1_position"))

        gross_return = previous_position * asset_return
        equity_before_trade = equity * (1.0 + gross_return)
        equity_peak = max(equity_peak, equity_before_trade)
        strategy_drawdown = equity_before_trade / equity_peak - 1.0

        in_cooldown = cooldown_remaining > 0
        base_binary_position = decide_v21_position(
            previous_position=previous_binary_position,
            v1_position_signal=v1_signal,
            confirmed_regime=confirmed_regime,
            allow_new_entries=not in_cooldown,
        )
        scale = compute_soft_dd_scale(
            strategy_drawdown,
            warning_dd=warning_dd,
            hard_dd=hard_dd,
            mode=dd_scale_mode,
        )

        forced_exit = strategy_drawdown <= hard_dd
        if in_cooldown:
            final_position = 0.0
        elif forced_exit:
            final_position = 0.0
            cooldown_remaining = cooldown_bars
        else:
            final_position = base_binary_position * scale

        trade_size = abs(final_position - previous_position)
        fee_cost = trade_size * fee_rate
        net_return = gross_return - fee_cost
        equity = equity * (1.0 + net_return)
        equity_peak = max(equity_peak, equity)
        current_drawdown = equity / equity_peak - 1.0

        position_before_dd_gate.append(base_binary_position)
        final_positions.append(float(final_position))
        dd_scales.append(float(scale))
        trade_sizes.append(float(trade_size))
        fee_costs.append(float(fee_cost))
        gross_returns.append(float(gross_return))
        net_returns.append(float(net_return))
        equity_values.append(float(equity))
        equity_peaks.append(float(equity_peak))
        strategy_drawdowns.append(float(strategy_drawdown))
        drawdowns.append(float(current_drawdown))
        forced_exit_flags.append(bool(forced_exit and not in_cooldown))
        cooldown_active.append(bool(in_cooldown or (forced_exit and cooldown_bars > 0)))
        cooldown_remaining_values.append(int(cooldown_remaining))
        entries_disabled_by_dd.append(False)

        previous_position = float(final_position)
        previous_binary_position = 1 if final_position > 0.0 else 0
        if cooldown_remaining > 0:
            cooldown_remaining -= 1

    result["position_before_dd_gate"] = position_before_dd_gate
    result["final_position"] = final_positions
    result["dd_scale"] = dd_scales
    result["trade_size"] = trade_sizes
    result["fee_cost"] = fee_costs
    result["strategy_return_gross"] = gross_returns
    result["strategy_return_net"] = net_returns
    result["equity_net"] = equity_values
    result["equity_peak"] = equity_peaks
    result["strategy_drawdown"] = strategy_drawdowns
    result["drawdown"] = drawdowns
    result["drawdown_gate_triggered"] = forced_exit_flags
    result["forced_exit"] = forced_exit_flags
    result["cooldown_active"] = cooldown_active
    result["cooldown_remaining"] = cooldown_remaining_values
    result["entries_disabled_by_dd"] = entries_disabled_by_dd
    return result


def backtest_v23_small_cap(
    df: pd.DataFrame,
    fee_rate: float = 0.001,
    *,
    v1_entry_threshold: float = 0.10,
    confirmation_days: int = 3,
    warning_dd: float = -0.03,
    hard_dd: float = -0.05,
    recovery_dd: float | None = None,
    cooldown_bars: int = 24,
    warning_entry_mode: str = "block_all",
) -> pd.DataFrame:
    """Run v2.3 low-frequency binary drawdown gate variants."""

    if fee_rate < 0.0:
        raise ValueError("fee_rate must be non-negative")
    if not 0.0 <= v1_entry_threshold <= 1.0:
        raise ValueError("v1_entry_threshold must be between 0 and 1")
    if cooldown_bars < 0:
        raise ValueError("cooldown_bars must be non-negative")
    if not hard_dd < warning_dd < 0.0:
        raise ValueError("drawdown thresholds must satisfy hard_dd < warning_dd < 0")
    if recovery_dd is not None and recovery_dd <= warning_dd:
        raise ValueError("recovery_dd must be above warning_dd")
    if warning_entry_mode not in {"block_all", "strong_only"}:
        raise ValueError("warning_entry_mode must be 'block_all' or 'strong_only'")

    result = compute_regime_features(df, confirmation_days=confirmation_days)
    result["v1_position"] = _extract_v1_position(result, v1_entry_threshold=v1_entry_threshold)
    result["asset_return"] = result["asset_return"].fillna(0.0)

    position_before_dd_gate: list[int] = []
    final_positions: list[int] = []
    trade_sizes: list[float] = []
    fee_costs: list[float] = []
    gross_returns: list[float] = []
    net_returns: list[float] = []
    equity_values: list[float] = []
    equity_peaks: list[float] = []
    strategy_drawdowns: list[float] = []
    drawdowns: list[float] = []
    hard_stop_flags: list[bool] = []
    cooldown_active: list[bool] = []
    cooldown_remaining_values: list[int] = []
    entries_disabled_by_dd: list[bool] = []
    warning_mode_values: list[bool] = []

    previous_position = 0
    equity = 1.0
    equity_peak = 1.0
    cooldown_remaining = 0
    warning_mode_active = False

    for row in result.itertuples(index=False):
        asset_return = float(getattr(row, "asset_return"))
        confirmed_regime = str(getattr(row, "confirmed_regime"))
        v1_signal = int(getattr(row, "v1_position"))

        gross_return = previous_position * asset_return
        equity_before_trade = equity * (1.0 + gross_return)
        equity_peak = max(equity_peak, equity_before_trade)
        strategy_drawdown = equity_before_trade / equity_peak - 1.0

        if recovery_dd is None:
            warning_mode_active = strategy_drawdown <= warning_dd
        else:
            if warning_mode_active and strategy_drawdown > recovery_dd:
                warning_mode_active = False
            if strategy_drawdown <= warning_dd:
                warning_mode_active = True

        in_cooldown = cooldown_remaining > 0
        allow_new_entries = not in_cooldown
        if warning_mode_active:
            allow_new_entries = warning_entry_mode == "strong_only" and confirmed_regime == "strong_bull" and not in_cooldown

        base_position = decide_v21_position(
            previous_position=previous_position,
            v1_position_signal=v1_signal,
            confirmed_regime=confirmed_regime,
            allow_new_entries=allow_new_entries,
        )

        hard_stop = strategy_drawdown <= hard_dd and not in_cooldown
        final_position = base_position
        if in_cooldown:
            final_position = 0
        elif hard_stop:
            final_position = 0
            cooldown_remaining = cooldown_bars

        allow_entry_without_dd = decide_v21_position(
            previous_position=previous_position,
            v1_position_signal=v1_signal,
            confirmed_regime=confirmed_regime,
            allow_new_entries=True,
        )
        entry_disabled = previous_position == 0 and allow_entry_without_dd == 1 and final_position == 0 and (warning_mode_active or in_cooldown or hard_stop)

        trade_size = abs(final_position - previous_position)
        fee_cost = trade_size * fee_rate
        net_return = gross_return - fee_cost
        equity = equity * (1.0 + net_return)
        equity_peak = max(equity_peak, equity)
        current_drawdown = equity / equity_peak - 1.0

        position_before_dd_gate.append(base_position)
        final_positions.append(final_position)
        trade_sizes.append(float(trade_size))
        fee_costs.append(float(fee_cost))
        gross_returns.append(float(gross_return))
        net_returns.append(float(net_return))
        equity_values.append(float(equity))
        equity_peaks.append(float(equity_peak))
        strategy_drawdowns.append(float(strategy_drawdown))
        drawdowns.append(float(current_drawdown))
        hard_stop_flags.append(bool(hard_stop))
        cooldown_active.append(bool(in_cooldown or (hard_stop and cooldown_bars > 0)))
        cooldown_remaining_values.append(int(cooldown_remaining))
        entries_disabled_by_dd.append(bool(entry_disabled))
        warning_mode_values.append(bool(warning_mode_active))

        previous_position = final_position
        if cooldown_remaining > 0:
            cooldown_remaining -= 1

    result["position_before_dd_gate"] = position_before_dd_gate
    result["final_position"] = final_positions
    result["trade_size"] = trade_sizes
    result["fee_cost"] = fee_costs
    result["strategy_return_gross"] = gross_returns
    result["strategy_return_net"] = net_returns
    result["equity_net"] = equity_values
    result["equity_peak"] = equity_peaks
    result["strategy_drawdown"] = strategy_drawdowns
    result["drawdown"] = drawdowns
    result["drawdown_gate_triggered"] = hard_stop_flags
    result["hard_stop"] = hard_stop_flags
    result["forced_exit"] = hard_stop_flags
    result["cooldown_active"] = cooldown_active
    result["cooldown_remaining"] = cooldown_remaining_values
    result["entries_disabled_by_dd"] = entries_disabled_by_dd
    result["warning_mode_active"] = warning_mode_values
    return result


def backtest_v24_small_cap(
    df: pd.DataFrame,
    fee_rate: float = 0.001,
    *,
    v1_entry_threshold: float = 0.10,
    confirmation_days: int = 3,
    pause_dd: float = -0.03,
    pause_bars: int = 24,
    exit_dd: float | None = None,
) -> pd.DataFrame:
    """Run v2.4 with event-style drawdown pause gates.

    A pause disables new entries for a fixed number of bars only after the
    strategy drawdown crosses below ``pause_dd``. Existing positions are not
    forced out unless the optional ``exit_dd`` crossing event fires.
    """

    if fee_rate < 0.0:
        raise ValueError("fee_rate must be non-negative")
    if not 0.0 <= v1_entry_threshold <= 1.0:
        raise ValueError("v1_entry_threshold must be between 0 and 1")
    if pause_bars < 0:
        raise ValueError("pause_bars must be non-negative")
    if not pause_dd < 0.0:
        raise ValueError("pause_dd must be negative")
    if exit_dd is not None and not exit_dd < pause_dd:
        raise ValueError("exit_dd must be below pause_dd")

    result = compute_regime_features(df, confirmation_days=confirmation_days)
    result["v1_position"] = _extract_v1_position(result, v1_entry_threshold=v1_entry_threshold)
    result["asset_return"] = result["asset_return"].fillna(0.0)

    position_before_dd_gate: list[int] = []
    final_positions: list[int] = []
    trade_sizes: list[float] = []
    fee_costs: list[float] = []
    gross_returns: list[float] = []
    net_returns: list[float] = []
    equity_values: list[float] = []
    equity_peaks: list[float] = []
    strategy_drawdowns: list[float] = []
    drawdowns: list[float] = []
    pause_trigger_flags: list[bool] = []
    pause_active_flags: list[bool] = []
    pause_remaining_values: list[int] = []
    risk_exit_flags: list[bool] = []
    entries_disabled_by_dd: list[bool] = []

    previous_position = 0
    equity = 1.0
    equity_peak = 1.0
    previous_drawdown = 0.0
    pause_remaining = 0

    for row in result.itertuples(index=False):
        asset_return = float(getattr(row, "asset_return"))
        confirmed_regime = str(getattr(row, "confirmed_regime"))
        v1_signal = int(getattr(row, "v1_position"))

        gross_return = previous_position * asset_return
        equity_before_trade = equity * (1.0 + gross_return)
        equity_peak = max(equity_peak, equity_before_trade)
        strategy_drawdown = equity_before_trade / equity_peak - 1.0

        in_pause = pause_remaining > 0
        pause_trigger = (not in_pause) and previous_drawdown > pause_dd and strategy_drawdown <= pause_dd
        risk_exit = (
            exit_dd is not None
            and (not in_pause)
            and previous_drawdown > exit_dd
            and strategy_drawdown <= exit_dd
        )
        if pause_trigger or risk_exit:
            pause_remaining = pause_bars
            in_pause = pause_remaining > 0

        base_position = decide_v21_position(
            previous_position=previous_position,
            v1_position_signal=v1_signal,
            confirmed_regime=confirmed_regime,
            allow_new_entries=not in_pause,
        )

        final_position = 0 if risk_exit else base_position
        allow_entry_without_pause = decide_v21_position(
            previous_position=previous_position,
            v1_position_signal=v1_signal,
            confirmed_regime=confirmed_regime,
            allow_new_entries=True,
        )
        entry_disabled = previous_position == 0 and allow_entry_without_pause == 1 and final_position == 0 and in_pause

        trade_size = abs(final_position - previous_position)
        fee_cost = trade_size * fee_rate
        net_return = gross_return - fee_cost
        equity = equity * (1.0 + net_return)
        equity_peak = max(equity_peak, equity)
        current_drawdown = equity / equity_peak - 1.0

        position_before_dd_gate.append(base_position)
        final_positions.append(final_position)
        trade_sizes.append(float(trade_size))
        fee_costs.append(float(fee_cost))
        gross_returns.append(float(gross_return))
        net_returns.append(float(net_return))
        equity_values.append(float(equity))
        equity_peaks.append(float(equity_peak))
        strategy_drawdowns.append(float(strategy_drawdown))
        drawdowns.append(float(current_drawdown))
        pause_trigger_flags.append(bool(pause_trigger))
        pause_active_flags.append(bool(in_pause))
        pause_remaining_values.append(int(pause_remaining))
        risk_exit_flags.append(bool(risk_exit))
        entries_disabled_by_dd.append(bool(entry_disabled))

        previous_position = final_position
        previous_drawdown = current_drawdown
        if pause_remaining > 0:
            pause_remaining -= 1

    result["position_before_dd_gate"] = position_before_dd_gate
    result["final_position"] = final_positions
    result["trade_size"] = trade_sizes
    result["fee_cost"] = fee_costs
    result["strategy_return_gross"] = gross_returns
    result["strategy_return_net"] = net_returns
    result["equity_net"] = equity_values
    result["equity_peak"] = equity_peaks
    result["strategy_drawdown"] = strategy_drawdowns
    result["drawdown"] = drawdowns
    result["pause_trigger"] = pause_trigger_flags
    result["pause_active"] = pause_active_flags
    result["pause_remaining"] = pause_remaining_values
    result["risk_exit"] = risk_exit_flags
    result["forced_exit"] = risk_exit_flags
    result["entries_disabled_by_dd"] = entries_disabled_by_dd
    return result


def backtest_v25_small_cap(
    df: pd.DataFrame,
    fee_rate: float = 0.001,
    *,
    v1_entry_threshold: float = 0.10,
    confirmation_days: int = 3,
    stop_loss: float | None = None,
    trailing_stop: float | None = None,
    pause_dd: float | None = None,
    pause_bars: int = 0,
    exit_dd: float | None = None,
) -> pd.DataFrame:
    """Run v2.5 with trade-level stop-loss and trailing exit guards."""

    if fee_rate < 0.0:
        raise ValueError("fee_rate must be non-negative")
    if not 0.0 <= v1_entry_threshold <= 1.0:
        raise ValueError("v1_entry_threshold must be between 0 and 1")
    if stop_loss is not None and not stop_loss < 0.0:
        raise ValueError("stop_loss must be negative")
    if trailing_stop is not None and not trailing_stop < 0.0:
        raise ValueError("trailing_stop must be negative")
    if pause_bars < 0:
        raise ValueError("pause_bars must be non-negative")
    if pause_dd is not None and not pause_dd < 0.0:
        raise ValueError("pause_dd must be negative")
    if exit_dd is not None:
        if pause_dd is None:
            raise ValueError("pause_dd is required when exit_dd is set")
        if not exit_dd < pause_dd:
            raise ValueError("exit_dd must be below pause_dd")

    result = compute_regime_features(df, confirmation_days=confirmation_days)
    result["v1_position"] = _extract_v1_position(result, v1_entry_threshold=v1_entry_threshold)
    result["asset_return"] = result["asset_return"].fillna(0.0)

    position_before_dd_gate: list[int] = []
    final_positions: list[int] = []
    trade_sizes: list[float] = []
    fee_costs: list[float] = []
    gross_returns: list[float] = []
    net_returns: list[float] = []
    equity_values: list[float] = []
    equity_peaks: list[float] = []
    strategy_drawdowns: list[float] = []
    drawdowns: list[float] = []
    pause_trigger_flags: list[bool] = []
    pause_active_flags: list[bool] = []
    pause_remaining_values: list[int] = []
    risk_exit_flags: list[bool] = []
    stop_exit_flags: list[bool] = []
    trailing_exit_flags: list[bool] = []
    entries_disabled_by_dd: list[bool] = []
    entry_prices: list[float] = []
    trade_peak_prices: list[float] = []
    trade_returns: list[float] = []
    trailing_drawdowns: list[float] = []

    previous_position = 0
    equity = 1.0
    equity_peak = 1.0
    previous_drawdown = 0.0
    pause_remaining = 0
    entry_price: float | None = None
    trade_peak_price: float | None = None

    for row in result.itertuples(index=False):
        close = float(getattr(row, "close"))
        asset_return = float(getattr(row, "asset_return"))
        confirmed_regime = str(getattr(row, "confirmed_regime"))
        v1_signal = int(getattr(row, "v1_position"))

        gross_return = previous_position * asset_return
        equity_before_trade = equity * (1.0 + gross_return)
        equity_peak = max(equity_peak, equity_before_trade)
        strategy_drawdown = equity_before_trade / equity_peak - 1.0

        in_pause = pause_remaining > 0
        pause_trigger = (
            pause_dd is not None
            and (not in_pause)
            and previous_drawdown > pause_dd
            and strategy_drawdown <= pause_dd
        )
        risk_exit = (
            exit_dd is not None
            and (not in_pause)
            and previous_drawdown > exit_dd
            and strategy_drawdown <= exit_dd
        )
        if pause_trigger or risk_exit:
            pause_remaining = pause_bars
            in_pause = pause_remaining > 0

        base_position = decide_v21_position(
            previous_position=previous_position,
            v1_position_signal=v1_signal,
            confirmed_regime=confirmed_regime,
            allow_new_entries=not in_pause,
        )

        current_trade_return = 0.0
        current_trailing_drawdown = 0.0
        stop_exit = False
        trailing_exit = False
        if previous_position == 1 and entry_price is not None:
            current_trade_return = close / entry_price - 1.0
            if trade_peak_price is None:
                trade_peak_price = entry_price
            trade_peak_price = max(trade_peak_price, close)
            current_trailing_drawdown = close / trade_peak_price - 1.0
            stop_exit = stop_loss is not None and current_trade_return <= stop_loss
            trailing_exit = trailing_stop is not None and current_trailing_drawdown <= trailing_stop

        final_position = base_position
        if risk_exit or stop_exit or trailing_exit:
            final_position = 0

        allow_entry_without_pause = decide_v21_position(
            previous_position=previous_position,
            v1_position_signal=v1_signal,
            confirmed_regime=confirmed_regime,
            allow_new_entries=True,
        )
        entry_disabled = previous_position == 0 and allow_entry_without_pause == 1 and final_position == 0 and in_pause

        trade_size = abs(final_position - previous_position)
        fee_cost = trade_size * fee_rate
        net_return = gross_return - fee_cost
        equity = equity * (1.0 + net_return)
        equity_peak = max(equity_peak, equity)
        current_drawdown = equity / equity_peak - 1.0

        if previous_position == 0 and final_position == 1:
            entry_price = close
            trade_peak_price = close
        elif final_position == 0:
            entry_price = None
            trade_peak_price = None

        position_before_dd_gate.append(base_position)
        final_positions.append(final_position)
        trade_sizes.append(float(trade_size))
        fee_costs.append(float(fee_cost))
        gross_returns.append(float(gross_return))
        net_returns.append(float(net_return))
        equity_values.append(float(equity))
        equity_peaks.append(float(equity_peak))
        strategy_drawdowns.append(float(strategy_drawdown))
        drawdowns.append(float(current_drawdown))
        pause_trigger_flags.append(bool(pause_trigger))
        pause_active_flags.append(bool(in_pause))
        pause_remaining_values.append(int(pause_remaining))
        risk_exit_flags.append(bool(risk_exit))
        stop_exit_flags.append(bool(stop_exit))
        trailing_exit_flags.append(bool(trailing_exit))
        entries_disabled_by_dd.append(bool(entry_disabled))
        entry_prices.append(float(entry_price) if entry_price is not None else np.nan)
        trade_peak_prices.append(float(trade_peak_price) if trade_peak_price is not None else np.nan)
        trade_returns.append(float(current_trade_return))
        trailing_drawdowns.append(float(current_trailing_drawdown))

        previous_position = final_position
        previous_drawdown = current_drawdown
        if pause_remaining > 0:
            pause_remaining -= 1

    result["position_before_dd_gate"] = position_before_dd_gate
    result["final_position"] = final_positions
    result["trade_size"] = trade_sizes
    result["fee_cost"] = fee_costs
    result["strategy_return_gross"] = gross_returns
    result["strategy_return_net"] = net_returns
    result["equity_net"] = equity_values
    result["equity_peak"] = equity_peaks
    result["strategy_drawdown"] = strategy_drawdowns
    result["drawdown"] = drawdowns
    result["pause_trigger"] = pause_trigger_flags
    result["pause_active"] = pause_active_flags
    result["pause_remaining"] = pause_remaining_values
    result["risk_exit"] = risk_exit_flags
    result["stop_exit"] = stop_exit_flags
    result["trailing_exit"] = trailing_exit_flags
    result["forced_exit"] = [risk or stop or trailing for risk, stop, trailing in zip(risk_exit_flags, stop_exit_flags, trailing_exit_flags)]
    result["entries_disabled_by_dd"] = entries_disabled_by_dd
    result["entry_price"] = entry_prices
    result["trade_peak_price"] = trade_peak_prices
    result["trade_return"] = trade_returns
    result["trailing_drawdown"] = trailing_drawdowns
    return result


def backtest_v26_regime_quality_small_cap(
    df: pd.DataFrame,
    fee_rate: float = 0.001,
    *,
    v1_entry_threshold: float = 0.10,
    confirmation_days: int = 3,
    weak_entry_filter: str | None = None,
    sideways_exit_filter: str | None = None,
) -> pd.DataFrame:
    """Run v2.6 regime-quality filters on top of sideways-hold-only logic."""

    if fee_rate < 0.0:
        raise ValueError("fee_rate must be non-negative")
    if not 0.0 <= v1_entry_threshold <= 1.0:
        raise ValueError("v1_entry_threshold must be between 0 and 1")
    valid_weak_filters = {None, "price_ma20", "ma_align", "mom_2", "quality_combo"}
    valid_sideways_filters = {None, "ma20", "mom0", "combo"}
    if weak_entry_filter not in valid_weak_filters:
        raise ValueError(f"weak_entry_filter must be one of {sorted(str(v) for v in valid_weak_filters)}")
    if sideways_exit_filter not in valid_sideways_filters:
        raise ValueError(f"sideways_exit_filter must be one of {sorted(str(v) for v in valid_sideways_filters)}")

    result = compute_regime_features(df, confirmation_days=confirmation_days)
    result["v1_position"] = _extract_v1_position(result, v1_entry_threshold=v1_entry_threshold)
    result["asset_return"] = result["asset_return"].fillna(0.0)

    position_before_quality_gate: list[int] = []
    final_positions: list[int] = []
    trade_sizes: list[float] = []
    fee_costs: list[float] = []
    gross_returns: list[float] = []
    net_returns: list[float] = []
    equity_values: list[float] = []
    equity_peaks: list[float] = []
    strategy_drawdowns: list[float] = []
    drawdowns: list[float] = []
    weak_entry_blocked: list[bool] = []
    sideways_exit_flags: list[bool] = []
    exit_reasons: list[str] = []

    previous_position = 0
    equity = 1.0
    equity_peak = 1.0

    for row in result.itertuples(index=False):
        asset_return = float(getattr(row, "asset_return"))
        confirmed_regime = str(getattr(row, "confirmed_regime"))
        v1_signal = int(getattr(row, "v1_position"))

        allow_entry, allow_hold = regime_entry_hold_permissions(confirmed_regime)
        blocked_weak_entry = False
        if previous_position == 0 and confirmed_regime == "weak_bull" and allow_entry:
            if not _weak_bull_entry_quality_passes(row, weak_entry_filter=weak_entry_filter):
                allow_entry = False
                blocked_weak_entry = v1_signal == 1

        base_position = _decide_position_from_permissions(
            previous_position=previous_position,
            v1_position_signal=v1_signal,
            allow_entry=allow_entry,
            allow_hold=allow_hold,
        )

        sideways_exit = (
            previous_position == 1
            and confirmed_regime == "sideways"
            and _sideways_exit_quality_triggers(row, sideways_exit_filter=sideways_exit_filter)
        )
        final_position = 0 if sideways_exit else base_position

        trade_size = abs(final_position - previous_position)
        fee_cost = trade_size * fee_rate
        gross_return = previous_position * asset_return
        net_return = gross_return - fee_cost
        equity = equity * (1.0 + net_return)
        equity_peak = max(equity_peak, equity)
        current_drawdown = equity / equity_peak - 1.0

        position_before_quality_gate.append(base_position)
        final_positions.append(final_position)
        trade_sizes.append(float(trade_size))
        fee_costs.append(float(fee_cost))
        gross_returns.append(float(gross_return))
        net_returns.append(float(net_return))
        equity_values.append(float(equity))
        equity_peaks.append(float(equity_peak))
        strategy_drawdowns.append(float(current_drawdown))
        drawdowns.append(float(current_drawdown))
        weak_entry_blocked.append(bool(blocked_weak_entry))
        sideways_exit_flags.append(bool(sideways_exit))
        exit_reasons.append(
            _quality_exit_reason(
                previous_position=previous_position,
                final_position=final_position,
                v1_signal=v1_signal,
                confirmed_regime=confirmed_regime,
                sideways_exit=sideways_exit,
            )
        )
        previous_position = final_position

    result["position_before_quality_gate"] = position_before_quality_gate
    result["position_before_dd_gate"] = position_before_quality_gate
    result["final_position"] = final_positions
    result["trade_size"] = trade_sizes
    result["fee_cost"] = fee_costs
    result["strategy_return_gross"] = gross_returns
    result["strategy_return_net"] = net_returns
    result["equity_net"] = equity_values
    result["equity_peak"] = equity_peaks
    result["strategy_drawdown"] = strategy_drawdowns
    result["drawdown"] = drawdowns
    result["weak_entry_blocked"] = weak_entry_blocked
    result["sideways_exit"] = sideways_exit_flags
    result["exit_reason"] = exit_reasons
    return result


def backtest_v27_weak_momentum_small_cap(
    df: pd.DataFrame,
    fee_rate: float = 0.001,
    *,
    v1_entry_threshold: float = 0.10,
    confirmation_days: int = 3,
    weak_momentum_threshold: float = 0.01,
    pause_dd: float | None = None,
    pause_bars: int = 0,
    exit_dd: float | None = None,
) -> pd.DataFrame:
    """Run v2.7 weak-bull momentum threshold sweep variants."""

    if fee_rate < 0.0:
        raise ValueError("fee_rate must be non-negative")
    if not 0.0 <= v1_entry_threshold <= 1.0:
        raise ValueError("v1_entry_threshold must be between 0 and 1")
    if pause_bars < 0:
        raise ValueError("pause_bars must be non-negative")
    if pause_dd is not None and not pause_dd < 0.0:
        raise ValueError("pause_dd must be negative")
    if exit_dd is not None:
        if pause_dd is None:
            raise ValueError("pause_dd is required when exit_dd is set")
        if not exit_dd < pause_dd:
            raise ValueError("exit_dd must be below pause_dd")

    result = compute_regime_features(df, confirmation_days=confirmation_days)
    result["v1_position"] = _extract_v1_position(result, v1_entry_threshold=v1_entry_threshold)
    result["asset_return"] = result["asset_return"].fillna(0.0)

    position_before_quality_gate: list[int] = []
    final_positions: list[int] = []
    trade_sizes: list[float] = []
    fee_costs: list[float] = []
    gross_returns: list[float] = []
    net_returns: list[float] = []
    equity_values: list[float] = []
    equity_peaks: list[float] = []
    strategy_drawdowns: list[float] = []
    drawdowns: list[float] = []
    weak_entry_attempts: list[bool] = []
    weak_entries_allowed: list[bool] = []
    weak_entries_blocked: list[bool] = []
    pause_trigger_flags: list[bool] = []
    pause_active_flags: list[bool] = []
    risk_exit_flags: list[bool] = []
    entries_disabled_by_dd: list[bool] = []
    entry_regimes: list[str] = []
    exit_reasons: list[str] = []

    previous_position = 0
    equity = 1.0
    equity_peak = 1.0
    previous_drawdown = 0.0
    pause_remaining = 0

    for row in result.itertuples(index=False):
        asset_return = float(getattr(row, "asset_return"))
        confirmed_regime = str(getattr(row, "confirmed_regime"))
        v1_signal = int(getattr(row, "v1_position"))

        gross_return = previous_position * asset_return
        equity_before_trade = equity * (1.0 + gross_return)
        equity_peak = max(equity_peak, equity_before_trade)
        strategy_drawdown = equity_before_trade / equity_peak - 1.0

        in_pause = pause_remaining > 0
        pause_trigger = (
            pause_dd is not None
            and (not in_pause)
            and previous_drawdown > pause_dd
            and strategy_drawdown <= pause_dd
        )
        risk_exit = (
            exit_dd is not None
            and (not in_pause)
            and previous_drawdown > exit_dd
            and strategy_drawdown <= exit_dd
        )
        if pause_trigger or risk_exit:
            pause_remaining = pause_bars
            in_pause = pause_remaining > 0

        allow_entry, allow_hold = regime_entry_hold_permissions(confirmed_regime)
        weak_attempt = previous_position == 0 and confirmed_regime == "weak_bull" and v1_signal == 1
        weak_passes = _weak_momentum_passes(row, threshold=weak_momentum_threshold)
        if weak_attempt and not weak_passes:
            allow_entry = False
        if in_pause:
            allow_entry = False

        base_position = _decide_position_from_permissions(
            previous_position=previous_position,
            v1_position_signal=v1_signal,
            allow_entry=allow_entry,
            allow_hold=allow_hold,
        )
        final_position = 0 if risk_exit else base_position

        trade_size = abs(final_position - previous_position)
        fee_cost = trade_size * fee_rate
        net_return = gross_return - fee_cost
        equity = equity * (1.0 + net_return)
        equity_peak = max(equity_peak, equity)
        current_drawdown = equity / equity_peak - 1.0

        entry_regime = confirmed_regime if previous_position == 0 and final_position == 1 else ""
        exit_reason = _v27_exit_reason(
            previous_position=previous_position,
            final_position=final_position,
            v1_signal=v1_signal,
            confirmed_regime=confirmed_regime,
            risk_exit=risk_exit,
        )
        entry_disabled_by_dd = previous_position == 0 and in_pause and final_position == 0 and v1_signal == 1

        position_before_quality_gate.append(base_position)
        final_positions.append(final_position)
        trade_sizes.append(float(trade_size))
        fee_costs.append(float(fee_cost))
        gross_returns.append(float(gross_return))
        net_returns.append(float(net_return))
        equity_values.append(float(equity))
        equity_peaks.append(float(equity_peak))
        strategy_drawdowns.append(float(strategy_drawdown))
        drawdowns.append(float(current_drawdown))
        weak_entry_attempts.append(bool(weak_attempt))
        weak_entries_allowed.append(bool(weak_attempt and weak_passes and not in_pause and final_position == 1))
        weak_entries_blocked.append(bool(weak_attempt and (not weak_passes)))
        pause_trigger_flags.append(bool(pause_trigger))
        pause_active_flags.append(bool(in_pause))
        risk_exit_flags.append(bool(risk_exit))
        entries_disabled_by_dd.append(bool(entry_disabled_by_dd))
        entry_regimes.append(entry_regime)
        exit_reasons.append(exit_reason)

        previous_position = final_position
        previous_drawdown = current_drawdown
        if pause_remaining > 0:
            pause_remaining -= 1

    result["position_before_quality_gate"] = position_before_quality_gate
    result["position_before_dd_gate"] = position_before_quality_gate
    result["final_position"] = final_positions
    result["trade_size"] = trade_sizes
    result["fee_cost"] = fee_costs
    result["strategy_return_gross"] = gross_returns
    result["strategy_return_net"] = net_returns
    result["equity_net"] = equity_values
    result["equity_peak"] = equity_peaks
    result["strategy_drawdown"] = strategy_drawdowns
    result["drawdown"] = drawdowns
    result["weak_bull_entry_attempt"] = weak_entry_attempts
    result["weak_bull_entry_allowed"] = weak_entries_allowed
    result["weak_bull_entry_blocked"] = weak_entries_blocked
    result["weak_entry_blocked"] = weak_entries_blocked
    result["pause_trigger"] = pause_trigger_flags
    result["pause_active"] = pause_active_flags
    result["risk_exit"] = risk_exit_flags
    result["entries_disabled_by_dd"] = entries_disabled_by_dd
    result["entry_regime"] = entry_regimes
    result["exit_reason"] = exit_reasons
    return result


def backtest_v28_weak_bull_control_small_cap(
    df: pd.DataFrame,
    fee_rate: float = 0.001,
    *,
    v1_entry_threshold: float = 0.10,
    confirmation_days: int = 3,
    weak_loss_cooldown_bars: int | None = None,
    weak_confirm_bars: int | None = None,
) -> pd.DataFrame:
    """Run v2.8 targeted weak-bull cooldown or duration-confirmation controls."""

    if fee_rate < 0.0:
        raise ValueError("fee_rate must be non-negative")
    if not 0.0 <= v1_entry_threshold <= 1.0:
        raise ValueError("v1_entry_threshold must be between 0 and 1")
    if weak_loss_cooldown_bars is not None and weak_loss_cooldown_bars < 0:
        raise ValueError("weak_loss_cooldown_bars must be non-negative")
    if weak_confirm_bars is not None and weak_confirm_bars < 1:
        raise ValueError("weak_confirm_bars must be positive")
    if weak_loss_cooldown_bars is not None and weak_confirm_bars is not None:
        raise ValueError("use either weak_loss_cooldown_bars or weak_confirm_bars, not both")

    result = compute_regime_features(df, confirmation_days=confirmation_days)
    result["v1_position"] = _extract_v1_position(result, v1_entry_threshold=v1_entry_threshold)
    result["asset_return"] = result["asset_return"].fillna(0.0)

    position_before_control: list[int] = []
    final_positions: list[int] = []
    trade_sizes: list[float] = []
    fee_costs: list[float] = []
    gross_returns: list[float] = []
    net_returns: list[float] = []
    equity_values: list[float] = []
    equity_peaks: list[float] = []
    strategy_drawdowns: list[float] = []
    drawdowns: list[float] = []
    weak_entry_attempts: list[bool] = []
    weak_entries_allowed: list[bool] = []
    weak_entries_blocked: list[bool] = []
    cooldown_trigger_flags: list[bool] = []
    cooldown_active_flags: list[bool] = []
    cooldown_remaining_values: list[int] = []
    weak_duration_values: list[int] = []
    entry_regimes: list[str] = []
    exit_reasons: list[str] = []

    previous_position = 0
    equity = 1.0
    equity_peak = 1.0
    weak_cooldown_remaining = 0
    weak_bull_duration = 0
    open_trade_entry_regime = ""
    open_trade_net_return = 0.0

    for row in result.itertuples(index=False):
        asset_return = float(getattr(row, "asset_return"))
        confirmed_regime = str(getattr(row, "confirmed_regime"))
        v1_signal = int(getattr(row, "v1_position"))

        weak_bull_duration = weak_bull_duration + 1 if confirmed_regime == "weak_bull" else 0
        cooldown_active = weak_cooldown_remaining > 0

        allow_entry, allow_hold = regime_entry_hold_permissions(confirmed_regime)
        weak_attempt = previous_position == 0 and confirmed_regime == "weak_bull" and v1_signal == 1
        blocked_by_cooldown = weak_attempt and cooldown_active
        blocked_by_confirmation = (
            weak_attempt
            and weak_confirm_bars is not None
            and weak_bull_duration < weak_confirm_bars
        )
        if blocked_by_cooldown or blocked_by_confirmation:
            allow_entry = False

        base_position = _decide_position_from_permissions(
            previous_position=previous_position,
            v1_position_signal=v1_signal,
            allow_entry=allow_entry,
            allow_hold=allow_hold,
        )
        final_position = base_position

        trade_size = abs(final_position - previous_position)
        fee_cost = trade_size * fee_rate
        gross_return = previous_position * asset_return
        net_return = gross_return - fee_cost
        equity = equity * (1.0 + net_return)
        equity_peak = max(equity_peak, equity)
        current_drawdown = equity / equity_peak - 1.0

        if previous_position == 1:
            open_trade_net_return += net_return

        entry_regime = ""
        exit_reason = _v27_exit_reason(
            previous_position=previous_position,
            final_position=final_position,
            v1_signal=v1_signal,
            confirmed_regime=confirmed_regime,
            risk_exit=False,
        )
        cooldown_trigger = False
        if previous_position == 0 and final_position == 1:
            entry_regime = confirmed_regime
            open_trade_entry_regime = confirmed_regime
            open_trade_net_return = -fee_cost
        elif previous_position == 1 and final_position == 0:
            if (
                weak_loss_cooldown_bars is not None
                and open_trade_entry_regime == "weak_bull"
                and open_trade_net_return < 0.0
            ):
                weak_cooldown_remaining = weak_loss_cooldown_bars
                cooldown_trigger = weak_loss_cooldown_bars > 0
                cooldown_active = cooldown_active or cooldown_trigger
            open_trade_entry_regime = ""
            open_trade_net_return = 0.0

        position_before_control.append(base_position)
        final_positions.append(final_position)
        trade_sizes.append(float(trade_size))
        fee_costs.append(float(fee_cost))
        gross_returns.append(float(gross_return))
        net_returns.append(float(net_return))
        equity_values.append(float(equity))
        equity_peaks.append(float(equity_peak))
        strategy_drawdowns.append(float(current_drawdown))
        drawdowns.append(float(current_drawdown))
        weak_entry_attempts.append(bool(weak_attempt))
        weak_entries_allowed.append(bool(weak_attempt and final_position == 1))
        weak_entries_blocked.append(bool(blocked_by_cooldown or blocked_by_confirmation))
        cooldown_trigger_flags.append(bool(cooldown_trigger))
        cooldown_active_flags.append(bool(cooldown_active))
        cooldown_remaining_values.append(int(weak_cooldown_remaining))
        weak_duration_values.append(int(weak_bull_duration))
        entry_regimes.append(entry_regime)
        exit_reasons.append(exit_reason)

        previous_position = final_position
        if weak_cooldown_remaining > 0:
            weak_cooldown_remaining -= 1

    result["position_before_quality_gate"] = position_before_control
    result["position_before_dd_gate"] = position_before_control
    result["final_position"] = final_positions
    result["trade_size"] = trade_sizes
    result["fee_cost"] = fee_costs
    result["strategy_return_gross"] = gross_returns
    result["strategy_return_net"] = net_returns
    result["equity_net"] = equity_values
    result["equity_peak"] = equity_peaks
    result["strategy_drawdown"] = strategy_drawdowns
    result["drawdown"] = drawdowns
    result["weak_bull_entry_attempt"] = weak_entry_attempts
    result["weak_bull_entry_allowed"] = weak_entries_allowed
    result["weak_bull_entry_blocked"] = weak_entries_blocked
    result["weak_entry_blocked"] = weak_entries_blocked
    result["weak_bull_cooldown_trigger"] = cooldown_trigger_flags
    result["weak_bull_cooldown_active"] = cooldown_active_flags
    result["weak_bull_cooldown_remaining"] = cooldown_remaining_values
    result["weak_bull_duration"] = weak_duration_values
    result["entry_regime"] = entry_regimes
    result["exit_reason"] = exit_reasons
    return result


def backtest_v2_candidate_2(
    df: pd.DataFrame,
    fee_rate: float = 0.001,
    *,
    v1_entry_threshold: float = 0.10,
    confirmation_days: int = 3,
) -> pd.DataFrame:
    """Run v2.candidate_2: sideways-hold-only plus 120-bar weak-loss cooldown."""

    return backtest_v28_weak_bull_control_small_cap(
        df,
        fee_rate=fee_rate,
        v1_entry_threshold=v1_entry_threshold,
        confirmation_days=confirmation_days,
        weak_loss_cooldown_bars=120,
    )


def backtest_v2_final_candidate_a(
    df: pd.DataFrame,
    fee_rate: float = 0.001,
    *,
    v1_entry_threshold: float = 0.10,
    confirmation_days: int = 3,
    cooldown_bars: int = 120,
) -> pd.DataFrame:
    """Run v2.final_candidate_A with configurable weak-loss cooldown bars."""

    return backtest_v28_weak_bull_control_small_cap(
        df,
        fee_rate=fee_rate,
        v1_entry_threshold=v1_entry_threshold,
        confirmation_days=confirmation_days,
        weak_loss_cooldown_bars=cooldown_bars,
    )


def backtest_v2_btc_final_candidate_a(
    df: pd.DataFrame,
    fee_rate: float = 0.001,
    *,
    v1_entry_threshold: float = 0.10,
    confirmation_days: int = 3,
    cooldown_bars: int = 120,
) -> pd.DataFrame:
    """Run frozen v2.btc_final_candidate_A.

    Scope: BTCUSDT 1h validation candidate only.  The implementation is the
    sideways-hold-only regime gate plus weak-bull losing-trade cooldown.  The
    observed robust cooldown range is 120-168 one-hour bars, with 120 as the
    default candidate value.
    """

    return backtest_v2_final_candidate_a(
        df,
        fee_rate=fee_rate,
        v1_entry_threshold=v1_entry_threshold,
        confirmation_days=confirmation_days,
        cooldown_bars=cooldown_bars,
    )


def backtest_v2_small_cap(
    df: pd.DataFrame,
    fee_rate: float = 0.001,
    use_drawdown_gate: bool = True,
    *,
    v1_entry_threshold: float = 0.5,
    confirmation_days: int = 3,
    gate_mode: str = "sideways_hold",
) -> pd.DataFrame:
    """Build v2 columns from v1.final output and run a fee-aware backtest.

    ``df`` must contain ``close`` plus either a binary v1 output column
    (``v1_position``, ``position``, or ``signal``) or a continuous v1 exposure
    column (for the current v1 implementation, ``current_exposure``).
    Returns are computed with ``final_position[t - 1]`` so the position chosen
    at row ``t`` never earns row ``t`` returns.
    """

    if fee_rate < 0.0:
        raise ValueError("fee_rate must be non-negative")
    if not 0.0 <= v1_entry_threshold <= 1.0:
        raise ValueError("v1_entry_threshold must be between 0 and 1")
    if gate_mode not in {"sideways_hold", "strict_sideways"}:
        raise ValueError("gate_mode must be 'sideways_hold' or 'strict_sideways'")

    result = compute_regime_features(df, confirmation_days=confirmation_days)
    result["v1_position"] = _extract_v1_position(result, v1_entry_threshold=v1_entry_threshold)
    result["asset_return"] = result["asset_return"].fillna(0.0)

    position_before_dd_gate: list[int] = []
    final_positions: list[int] = []
    trade_sizes: list[float] = []
    fee_costs: list[float] = []
    gross_returns: list[float] = []
    net_returns: list[float] = []
    equity_values: list[float] = []
    equity_peaks: list[float] = []
    strategy_drawdowns: list[float] = []
    drawdowns: list[float] = []

    previous_position = 0
    equity = 1.0
    equity_peak = 1.0

    for row in result.itertuples(index=False):
        asset_return = float(getattr(row, "asset_return"))
        confirmed_regime = str(getattr(row, "confirmed_regime"))
        v1_position = int(getattr(row, "v1_position"))
        strategy_drawdown = equity / equity_peak - 1.0

        gated_position = apply_trade_gate(
            v1_position,
            confirmed_regime,
            previous_position,
            gate_mode=gate_mode,
        )

        final_position = (
            apply_drawdown_risk_gate(gated_position, confirmed_regime, strategy_drawdown)
            if use_drawdown_gate
            else gated_position
        )
        trade_size = abs(final_position - previous_position)
        fee_cost = trade_size * fee_rate
        gross_return = previous_position * asset_return
        net_return = gross_return - fee_cost

        equity *= 1.0 + net_return
        equity_peak = max(equity_peak, equity)
        current_drawdown = equity / equity_peak - 1.0

        position_before_dd_gate.append(gated_position)
        final_positions.append(final_position)
        trade_sizes.append(float(trade_size))
        fee_costs.append(float(fee_cost))
        gross_returns.append(float(gross_return))
        net_returns.append(float(net_return))
        equity_values.append(float(equity))
        equity_peaks.append(float(equity_peak))
        strategy_drawdowns.append(float(strategy_drawdown))
        drawdowns.append(float(current_drawdown))
        previous_position = final_position

    result["position_before_dd_gate"] = position_before_dd_gate
    result["final_position"] = final_positions
    result["trade_size"] = trade_sizes
    result["fee_cost"] = fee_costs
    result["strategy_return_gross"] = gross_returns
    result["strategy_return_net"] = net_returns
    result["equity_net"] = equity_values
    result["equity_peak"] = equity_peaks
    result["strategy_drawdown"] = strategy_drawdowns
    result["drawdown"] = drawdowns
    return result


def calculate_performance_stats(
    df: pd.DataFrame,
    *,
    annualization_factor: int = 365,
    window_days: int | None = None,
) -> dict[str, float | int]:
    """Calculate net performance and trade behavior metrics for one result frame."""

    if annualization_factor <= 0:
        raise ValueError("annualization_factor must be positive")
    frame = _window_frame(df, window_days=window_days)
    if frame.empty:
        return _empty_performance_stats()

    returns = _extract_net_returns(frame)
    equity_curve = (1.0 + returns).cumprod()
    drawdown = equity_curve / equity_curve.cummax() - 1.0
    periods = max(len(returns), 1)
    total_return = float(equity_curve.iloc[-1] - 1.0)
    annualized_return = float(equity_curve.iloc[-1] ** (annualization_factor / periods) - 1.0)
    max_drawdown = float(drawdown.min())
    sharpe = _sharpe_ratio(returns, annualization_factor=annualization_factor)
    sortino = _sortino_ratio(returns, annualization_factor=annualization_factor)
    calmar = 0.0 if max_drawdown == 0.0 else float(annualized_return / abs(max_drawdown))

    position = _extract_position(frame)
    trade_size = _extract_trade_size(frame, position)
    entries = int(((position.shift(1).fillna(0.0) <= 0.0) & (position > 0.0)).sum())
    exits = int(((position.shift(1).fillna(0.0) > 0.0) & (position <= 0.0)).sum())

    return {
        "total_return_net": total_return,
        "annualized_return_net": annualized_return,
        "max_drawdown": max_drawdown,
        "Sharpe_net": sharpe,
        "Sortino_net": sortino,
        "Calmar": calmar,
        "average_exposure": float(position.abs().mean()) if not position.empty else 0.0,
        "turnover": float(trade_size.sum()),
        "number_of_entries": entries,
        "number_of_exits": exits,
        "total_trades": int((trade_size > 0.0).sum()),
        "average_holding_days": _average_holding_days(position),
        "total_fee_paid": _extract_total_fee_paid(frame),
    }


def build_performance_summary_table(
    version_frames: dict[str, pd.DataFrame],
    *,
    annualization_factor: int = 365,
) -> pd.DataFrame:
    """Build a multi-window summary table for v1.final and v2 comparisons."""

    rows: list[dict[str, float | int | str]] = []
    for version, frame in version_frames.items():
        for window_name, window_days in SUMMARY_WINDOWS.items():
            stats = calculate_performance_stats(
                frame,
                annualization_factor=annualization_factor,
                window_days=window_days,
            )
            rows.append({"version": version, "window": window_name, **stats})
    return pd.DataFrame(rows).loc[:, SUMMARY_COLUMNS]


def build_regime_diagnostics(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build regime distribution, regime performance, and trade behavior tables."""

    _require_columns(df, ["confirmed_regime"])
    frame = df.copy()
    regime = frame["confirmed_regime"].astype(str)
    position = _extract_position(frame)
    trade_size = _extract_trade_size(frame, position)
    strategy_return = _extract_net_returns(frame)
    asset_return = _extract_asset_returns(frame)
    fee_cost = _extract_fee_series(frame)

    distribution = (
        regime.value_counts(sort=False)
        .rename_axis("regime")
        .reset_index(name="days")
    )
    distribution["ratio"] = distribution["days"] / max(len(frame), 1)
    distribution = _order_regime_table(distribution)

    performance_rows: list[dict[str, float | int | str]] = []
    for regime_name, indexes in regime.groupby(regime).groups.items():
        idx = list(indexes)
        performance_rows.append(
            {
                "regime": regime_name,
                "avg_position": float(position.loc[idx].abs().mean()) if idx else 0.0,
                "strategy_return_net": float(strategy_return.loc[idx].sum()),
                "asset_return": float(asset_return.loc[idx].sum()),
                "trades": int((trade_size.loc[idx] > 0.0).sum()),
                "fees": float(fee_cost.loc[idx].sum()),
            }
        )
    regime_performance = _order_regime_table(pd.DataFrame(performance_rows))

    stats = calculate_performance_stats(frame)
    trade_behavior = pd.DataFrame(
        [
            {
                "number_of_entries": stats["number_of_entries"],
                "number_of_exits": stats["number_of_exits"],
                "total_trades": stats["total_trades"],
                "average_holding_days": stats["average_holding_days"],
                "average_exposure": stats["average_exposure"],
                "turnover": stats["turnover"],
                "total_fee_paid": stats["total_fee_paid"],
            }
        ]
    )

    return {
        "regime_distribution": distribution.loc[:, ["regime", "days", "ratio"]],
        "regime_performance": regime_performance.loc[
            :, ["regime", "avg_position", "strategy_return_net", "asset_return", "trades", "fees"]
        ],
        "trade_behavior": trade_behavior,
    }


def build_drawdown_attribution_diagnostics(
    df: pd.DataFrame,
    *,
    trade_context: int = 3,
    bar_context: int = 48,
) -> dict[str, pd.DataFrame]:
    """Build v2.6 diagnostics for the max drawdown episode.

    The diagnostics are computed from already-produced backtest columns and do
    not alter strategy decisions.
    """

    if trade_context < 0:
        raise ValueError("trade_context must be non-negative")
    if bar_context < 0:
        raise ValueError("bar_context must be non-negative")
    _require_columns(df, ["close"])
    frame = df.copy().reset_index(drop=True)
    frame["_time"] = _extract_time_values(frame)
    frame["_position"] = _extract_position(frame)
    frame["_asset_return"] = _extract_asset_returns(frame)
    frame["_strategy_return_net"] = _extract_net_returns(frame)
    frame["_fee_cost"] = _extract_fee_series(frame)
    if "equity_net" in frame.columns:
        frame["_equity"] = pd.to_numeric(frame["equity_net"], errors="coerce").fillna(1.0).astype(float)
    elif "equity" in frame.columns:
        frame["_equity"] = pd.to_numeric(frame["equity"], errors="coerce").fillna(1.0).astype(float)
    else:
        frame["_equity"] = (1.0 + frame["_strategy_return_net"]).cumprod()
    if "drawdown" in frame.columns:
        frame["_drawdown"] = pd.to_numeric(frame["drawdown"], errors="coerce").fillna(0.0).astype(float)
    else:
        frame["_drawdown"] = frame["_equity"] / frame["_equity"].cummax() - 1.0

    start_idx, trough_idx, recovery_idx = _max_drawdown_episode_indexes(frame["_equity"])
    episode_slice = frame.iloc[start_idx : trough_idx + 1]
    summary = _build_drawdown_episode_summary(frame, start_idx, trough_idx, recovery_idx)
    regime_distribution = _build_episode_regime_distribution(episode_slice)
    trades = _build_trade_list_with_context(frame, start_idx, recovery_idx if recovery_idx is not None else trough_idx, trade_context)
    sample_start = max(0, start_idx - bar_context)
    sample_end = min(len(frame) - 1, trough_idx + bar_context)
    bar_sample = _build_drawdown_bar_sample(frame.iloc[sample_start : sample_end + 1])

    return {
        "max_drawdown_episode": summary,
        "regime_distribution_during_max_dd": regime_distribution,
        "trades_around_max_dd": trades,
        "bar_sample_around_max_dd": bar_sample,
    }


def _extract_v1_position(df: pd.DataFrame, *, v1_entry_threshold: float) -> pd.Series:
    for column in V1_POSITION_COLUMNS:
        if column in df.columns:
            return (pd.to_numeric(df[column], errors="coerce").fillna(0.0) >= 1.0).astype(int)

    for column in V1_EXPOSURE_COLUMNS:
        if column in df.columns:
            exposure = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
            return (exposure.abs() >= v1_entry_threshold).astype(int)

    raise ValueError(
        "df must include a v1 binary column "
        f"{list(V1_POSITION_COLUMNS)} or exposure column {list(V1_EXPOSURE_COLUMNS)}"
    )


def _extract_time_values(df: pd.DataFrame) -> pd.Series:
    for column in TIME_COLUMNS:
        if column in df.columns:
            return df[column].reset_index(drop=True)
    return pd.Series(df.index, index=df.index)


def _decide_position_from_permissions(
    *,
    previous_position: int,
    v1_position_signal: int,
    allow_entry: bool,
    allow_hold: bool,
) -> int:
    previous_binary = _validate_binary_position(previous_position, name="previous_position")
    signal_binary = _validate_binary_position(v1_position_signal, name="v1_position_signal")
    if previous_binary == 0:
        return 1 if allow_entry and signal_binary == 1 else 0
    if not allow_hold:
        return 0
    if signal_binary == 0:
        return 0
    return 1


def _weak_bull_entry_quality_passes(row: Any, *, weak_entry_filter: str | None) -> bool:
    if weak_entry_filter is None:
        return True
    close = float(getattr(row, "close"))
    ma20 = float(getattr(row, "MA20"))
    ma60 = float(getattr(row, "MA60"))
    momentum_20 = float(getattr(row, "momentum_20"))
    price_above_ma20 = np.isfinite(ma20) and close > ma20
    ma_aligned = np.isfinite(ma20) and np.isfinite(ma60) and ma20 > ma60
    momentum_passes = np.isfinite(momentum_20) and momentum_20 > 0.02
    if weak_entry_filter == "price_ma20":
        return price_above_ma20
    if weak_entry_filter == "ma_align":
        return ma_aligned
    if weak_entry_filter == "mom_2":
        return momentum_passes
    if weak_entry_filter == "quality_combo":
        return price_above_ma20 and momentum_passes
    return True


def _sideways_exit_quality_triggers(row: Any, *, sideways_exit_filter: str | None) -> bool:
    if sideways_exit_filter is None:
        return False
    close = float(getattr(row, "close"))
    ma20 = float(getattr(row, "MA20"))
    momentum_20 = float(getattr(row, "momentum_20"))
    price_below_ma20 = np.isfinite(ma20) and close < ma20
    momentum_negative = np.isfinite(momentum_20) and momentum_20 < 0.0
    if sideways_exit_filter == "ma20":
        return price_below_ma20
    if sideways_exit_filter == "mom0":
        return momentum_negative
    if sideways_exit_filter == "combo":
        return price_below_ma20 and momentum_negative
    return False


def _quality_exit_reason(
    *,
    previous_position: int,
    final_position: int,
    v1_signal: int,
    confirmed_regime: str,
    sideways_exit: bool,
) -> str:
    if previous_position <= 0 or final_position > 0:
        return ""
    if sideways_exit:
        return "sideways_quality_exit"
    if v1_signal == 0:
        return "v1_signal_off"
    _, allow_hold = regime_entry_hold_permissions(confirmed_regime)
    if not allow_hold:
        return "regime_hold_block"
    return "position_exit"


def _weak_momentum_passes(row: Any, *, threshold: float) -> bool:
    momentum_20 = float(getattr(row, "momentum_20"))
    return np.isfinite(momentum_20) and momentum_20 > threshold


def _v27_exit_reason(
    *,
    previous_position: int,
    final_position: int,
    v1_signal: int,
    confirmed_regime: str,
    risk_exit: bool,
) -> str:
    if previous_position <= 0 or final_position > 0:
        return ""
    if risk_exit:
        return "risk_exit"
    if v1_signal == 0:
        return "v1_signal_off"
    _, allow_hold = regime_entry_hold_permissions(confirmed_regime)
    if not allow_hold:
        return "regime_hold_block"
    return "position_exit"


def _max_drawdown_episode_indexes(equity: pd.Series) -> tuple[int, int, int | None]:
    values = pd.to_numeric(equity, errors="coerce").fillna(1.0).astype(float).reset_index(drop=True)
    if values.empty:
        return 0, 0, None
    peaks = values.cummax()
    drawdown = values / peaks - 1.0
    trough_idx = int(drawdown.idxmin())
    peak_value = float(peaks.iloc[trough_idx])
    peak_segment = values.iloc[: trough_idx + 1]
    start_candidates = peak_segment[peak_segment >= peak_value]
    start_idx = int(start_candidates.index[-1]) if not start_candidates.empty else 0
    recovery_idx: int | None = None
    for idx in range(trough_idx + 1, len(values)):
        if float(values.iloc[idx]) >= peak_value:
            recovery_idx = idx
            break
    return start_idx, trough_idx, recovery_idx


def _build_drawdown_episode_summary(
    frame: pd.DataFrame,
    start_idx: int,
    trough_idx: int,
    recovery_idx: int | None,
) -> pd.DataFrame:
    episode = frame.iloc[start_idx : trough_idx + 1]
    equity_at_start = float(frame["_equity"].iloc[start_idx])
    equity_at_trough = float(frame["_equity"].iloc[trough_idx])
    return pd.DataFrame(
        [
            {
                "dd_start_time": frame["_time"].iloc[start_idx],
                "dd_trough_time": frame["_time"].iloc[trough_idx],
                "dd_recovery_time": frame["_time"].iloc[recovery_idx] if recovery_idx is not None else pd.NA,
                "dd_duration_bars": trough_idx - start_idx + 1,
                "dd_recovery_bars": recovery_idx - trough_idx if recovery_idx is not None else pd.NA,
                "equity_at_start": equity_at_start,
                "equity_at_trough": equity_at_trough,
                "max_drawdown": equity_at_trough / equity_at_start - 1.0 if equity_at_start != 0.0 else 0.0,
                "asset_return_during_dd": float(episode["_asset_return"].sum()),
                "strategy_return_during_dd": float(episode["_strategy_return_net"].sum()),
                "fees_during_dd": float(episode["_fee_cost"].sum()),
                "avg_position_during_dd": float(episode["_position"].abs().mean()) if not episode.empty else 0.0,
            }
        ]
    )


def _build_episode_regime_distribution(episode: pd.DataFrame) -> pd.DataFrame:
    if episode.empty:
        return pd.DataFrame(
            columns=["regime", "bars", "ratio", "avg_position", "strategy_return_net", "asset_return"]
        )
    regime = episode["confirmed_regime"].astype(str) if "confirmed_regime" in episode.columns else pd.Series("unknown", index=episode.index)
    rows: list[dict[str, Any]] = []
    for regime_name, indexes in regime.groupby(regime).groups.items():
        idx = list(indexes)
        rows.append(
            {
                "regime": regime_name,
                "bars": len(idx),
                "ratio": len(idx) / len(episode),
                "avg_position": float(episode.loc[idx, "_position"].abs().mean()),
                "strategy_return_net": float(episode.loc[idx, "_strategy_return_net"].sum()),
                "asset_return": float(episode.loc[idx, "_asset_return"].sum()),
            }
        )
    return _order_regime_table(pd.DataFrame(rows)).loc[
        :, ["regime", "bars", "ratio", "avg_position", "strategy_return_net", "asset_return"]
    ]


def _build_trade_list_with_context(
    frame: pd.DataFrame,
    episode_start_idx: int,
    episode_end_idx: int,
    trade_context: int,
) -> pd.DataFrame:
    trades = _build_trade_list(frame)
    columns = [
        "entry_time",
        "exit_time",
        "holding_bars",
        "entry_regime",
        "exit_regime",
        "entry_price",
        "exit_price",
        "gross_trade_return",
        "net_trade_return",
        "max_adverse_excursion",
        "max_favorable_excursion",
        "exit_reason",
        "fees",
    ]
    if trades.empty:
        return pd.DataFrame(columns=columns)
    overlap_mask = (trades["entry_idx"] <= episode_end_idx) & (trades["exit_idx"] >= episode_start_idx)
    overlap_positions = list(np.flatnonzero(overlap_mask.to_numpy()))
    if not overlap_positions:
        closest = int((trades["entry_idx"] - episode_start_idx).abs().idxmin())
        first = max(0, closest - trade_context)
        last = min(len(trades) - 1, closest + trade_context)
    else:
        first = max(0, min(overlap_positions) - trade_context)
        last = min(len(trades) - 1, max(overlap_positions) + trade_context)
    return trades.iloc[first : last + 1].reset_index(drop=True).loc[:, columns]


def _build_trade_list(frame: pd.DataFrame) -> pd.DataFrame:
    position = frame["_position"].fillna(0.0).astype(float)
    active = position > 0.0
    trades: list[dict[str, Any]] = []
    entry_idx: int | None = None
    for idx, is_active in enumerate(active):
        was_active = bool(active.iloc[idx - 1]) if idx > 0 else False
        if bool(is_active) and not was_active:
            entry_idx = idx
        if was_active and (not bool(is_active)) and entry_idx is not None:
            trades.append(_build_trade_row(frame, entry_idx, idx))
            entry_idx = None
    if entry_idx is not None:
        trades.append(_build_trade_row(frame, entry_idx, len(frame) - 1))
    return pd.DataFrame(trades)


def _build_trade_row(frame: pd.DataFrame, entry_idx: int, exit_idx: int) -> dict[str, Any]:
    trade = frame.iloc[entry_idx : exit_idx + 1]
    entry_price = float(frame["close"].iloc[entry_idx])
    exit_price = float(frame["close"].iloc[exit_idx])
    close_path = pd.to_numeric(trade["close"], errors="coerce").astype(float)
    return {
        "entry_idx": entry_idx,
        "exit_idx": exit_idx,
        "entry_time": frame["_time"].iloc[entry_idx],
        "exit_time": frame["_time"].iloc[exit_idx],
        "holding_bars": exit_idx - entry_idx + 1,
        "entry_regime": _regime_at(frame, entry_idx),
        "exit_regime": _regime_at(frame, exit_idx),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "gross_trade_return": exit_price / entry_price - 1.0 if entry_price > 0.0 else 0.0,
        "net_trade_return": float(trade["_strategy_return_net"].sum()),
        "max_adverse_excursion": float((close_path / entry_price - 1.0).min()) if entry_price > 0.0 else 0.0,
        "max_favorable_excursion": float((close_path / entry_price - 1.0).max()) if entry_price > 0.0 else 0.0,
        "exit_reason": _exit_reason_at(frame, exit_idx),
        "fees": float(trade["_fee_cost"].sum()),
    }


def _regime_at(frame: pd.DataFrame, idx: int) -> str:
    if "confirmed_regime" not in frame.columns:
        return "unknown"
    return str(frame["confirmed_regime"].iloc[idx])


def _exit_reason_at(frame: pd.DataFrame, idx: int) -> str:
    for column, reason in (
        ("risk_exit", "risk_exit"),
        ("stop_exit", "stop_exit"),
        ("trailing_exit", "trailing_exit"),
        ("drawdown_gate_triggered", "drawdown_gate"),
        ("forced_exit", "forced_exit"),
    ):
        if column in frame.columns and bool(frame[column].iloc[idx]):
            return reason
    if "v1_position" in frame.columns and float(frame["v1_position"].iloc[idx]) <= 0.0:
        return "v1_signal_off"
    if "confirmed_regime" in frame.columns:
        _, allow_hold = regime_entry_hold_permissions(str(frame["confirmed_regime"].iloc[idx]))
        if not allow_hold:
            return "regime_hold_block"
    return "position_exit"


def _build_drawdown_bar_sample(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.reset_index(drop=True)
    regimes = frame["confirmed_regime"].astype(str) if "confirmed_regime" in frame.columns else pd.Series("unknown", index=frame.index)
    result = pd.DataFrame(
        {
            "timestamp": frame["_time"],
            "close": pd.to_numeric(frame["close"], errors="coerce").astype(float),
            "asset_return": frame["_asset_return"],
            "regime": regimes,
            "v1_signal": frame["v1_position"] if "v1_position" in frame.columns else pd.NA,
            "position": frame["_position"],
            "strategy_return_net": frame["_strategy_return_net"],
            "equity": frame["_equity"],
            "drawdown": frame["_drawdown"],
            "entry_allowed": [regime_entry_hold_permissions(str(regime))[0] for regime in regimes],
            "hold_allowed": [regime_entry_hold_permissions(str(regime))[1] for regime in regimes],
            "exit_reason": [_exit_reason_at(frame, idx) for idx in frame.index],
        }
    )
    previous_position = frame["_position"].shift(1).fillna(0.0)
    result.loc[frame["_position"].to_numpy() >= previous_position.to_numpy(), "exit_reason"] = ""
    return result.reset_index(drop=True)


def _window_frame(df: pd.DataFrame, *, window_days: int | None) -> pd.DataFrame:
    frame = df.copy()
    if window_days is None:
        return frame

    timestamp = _extract_timestamp(frame)
    if timestamp is not None:
        cutoff = timestamp.max() - pd.Timedelta(days=window_days)
        return frame.loc[timestamp >= cutoff].copy()
    return frame.tail(window_days).copy()


def _extract_timestamp(df: pd.DataFrame) -> pd.Series | None:
    source: pd.Series | pd.Index | None = None
    for column in ("timestamp", "date", "datetime"):
        if column in df.columns:
            source = df[column]
            break
    if source is None and isinstance(df.index, pd.DatetimeIndex):
        source = df.index
    if source is None:
        return None

    timestamp = pd.Series(pd.to_datetime(source, errors="coerce"), index=df.index)
    return None if timestamp.isna().all() else timestamp


def _extract_net_returns(df: pd.DataFrame) -> pd.Series:
    for column in ("strategy_return_net", "period_return", "return_net", "net_return"):
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce").fillna(0.0).astype(float)
    if "equity_net" in df.columns:
        equity = pd.to_numeric(df["equity_net"], errors="coerce").astype(float)
        return equity.pct_change().fillna(equity.iloc[0] - 1.0)
    if "equity" in df.columns:
        equity = pd.to_numeric(df["equity"], errors="coerce").astype(float)
        return equity.pct_change().fillna(equity.iloc[0] - 1.0)
    raise ValueError("df must include net return or equity columns")


def _extract_asset_returns(df: pd.DataFrame) -> pd.Series:
    if "asset_return" in df.columns:
        return pd.to_numeric(df["asset_return"], errors="coerce").fillna(0.0).astype(float)
    if "close" in df.columns:
        close = pd.to_numeric(df["close"], errors="coerce").astype(float)
        return close.pct_change().fillna(0.0)
    return pd.Series(0.0, index=df.index)


def _extract_position(df: pd.DataFrame) -> pd.Series:
    for column in ("final_position", "position", "current_exposure", "exposure", "v1_position"):
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce").fillna(0.0).astype(float)
    return pd.Series(0.0, index=df.index)


def _extract_trade_size(df: pd.DataFrame, position: pd.Series) -> pd.Series:
    for column in ("trade_size", "turnover", "safe_exposure_change"):
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce").fillna(0.0).astype(float)
            return values.abs()
    return position.diff().abs().fillna(position.abs())


def _extract_total_fee_paid(df: pd.DataFrame) -> float:
    return float(_extract_fee_series(df).sum())


def _extract_fee_series(df: pd.DataFrame) -> pd.Series:
    for column in ("fee_cost", "fee_paid", "total_fee_paid"):
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce").fillna(0.0).astype(float)
    return pd.Series(0.0, index=df.index)


def _order_regime_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "regime" not in frame.columns:
        return frame
    order = {regime: idx for idx, regime in enumerate(REGIME_SCORE)}
    result = frame.copy()
    result["_regime_order"] = result["regime"].map(order).fillna(len(order))
    return result.sort_values(["_regime_order", "regime"]).drop(columns="_regime_order").reset_index(drop=True)


def _sharpe_ratio(returns: pd.Series, *, annualization_factor: int) -> float:
    std = float(returns.std(ddof=0))
    if std == 0.0:
        return 0.0
    return float(returns.mean() / std * np.sqrt(annualization_factor))


def _sortino_ratio(returns: pd.Series, *, annualization_factor: int) -> float:
    downside = returns[returns < 0.0]
    downside_std = float(downside.std(ddof=0))
    if downside_std == 0.0:
        return 0.0
    return float(returns.mean() / downside_std * np.sqrt(annualization_factor))


def _average_holding_days(position: pd.Series) -> float:
    holding_lengths: list[int] = []
    current_length = 0
    for value in position:
        if value > 0.0:
            current_length += 1
        elif current_length > 0:
            holding_lengths.append(current_length)
            current_length = 0
    if current_length > 0:
        holding_lengths.append(current_length)
    if not holding_lengths:
        return 0.0
    return float(np.mean(holding_lengths))


def _empty_performance_stats() -> dict[str, float | int]:
    return {
        "total_return_net": 0.0,
        "annualized_return_net": 0.0,
        "max_drawdown": 0.0,
        "Sharpe_net": 0.0,
        "Sortino_net": 0.0,
        "Calmar": 0.0,
        "average_exposure": 0.0,
        "turnover": 0.0,
        "number_of_entries": 0,
        "number_of_exits": 0,
        "total_trades": 0,
        "average_holding_days": 0.0,
        "total_fee_paid": 0.0,
    }


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = set(columns).difference(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}")


def _validate_binary_position(value: int | float, *, name: str) -> int:
    numeric = float(value)
    if numeric not in {0.0, 1.0}:
        raise ValueError(f"{name} must be 0 or 1")
    return int(numeric)


def _as_float(row: pd.Series | dict[str, Any], key: str) -> float:
    value = row[key] if isinstance(row, dict) else row.loc[key]
    return float(value)

"""Multi-interval validation for v1.2, v1.4, v1.final and simple baselines."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtester import _calculate_metrics, _classify_layer4_intervention, _should_apply_cooldown, load_ohlcv_csv
from src.layer1_market_model import MarketStateV1, build_market_state_frame
from src.layer2_state_estimator import EstimatedMarketStateV1, estimate_market_state_frame
from src.layer3_strategy_controller import PortfolioStateV1, compute_control_action
from src.layer4_risk_filter import RiskConfigV1, apply_risk_filter


PERIODS_PER_YEAR_1H = 365 * 24
SUMMARY_COLUMNS = [
    "interval",
    "version",
    "total_return",
    "max_drawdown",
    "sharpe",
    "trade_count",
    "turnover",
    "average_exposure",
    "target_clip_rate",
    "turnover_clip_rate",
    "kill_switch_count",
    "reduce_only_count",
]


@dataclass(frozen=True)
class VersionConfig:
    name: str
    use_target_smoothing: bool
    target_deadband: float
    beta_increase_risk: float
    beta_decrease_risk: float
    min_trade_threshold: float
    minimum_rebalance_interval: int


VERSION_CONFIGS = [
    VersionConfig("v1.2", False, 0.0, 0.30, 0.80, 0.02, 0),
    VersionConfig("v1.4", True, 0.03, 0.30, 0.80, 0.02, 0),
    VersionConfig("v1.final", True, 0.05, 0.30, 0.60, 0.03, 3),
]


def run_validation(csv_path: str | Path, output_dir: str | Path = "reports/multi_interval_validation") -> dict[str, pd.DataFrame]:
    data = prepare_ohlcv(load_ohlcv_csv(csv_path))
    intervals = build_intervals(data)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []
    for interval_name, frame in intervals.items():
        for config in VERSION_CONFIGS:
            result = run_control_system_backtest(frame, config)
            summary_rows.append({"interval": interval_name, "version": config.name, **result["metrics"]})
            for regime, count in result["regime_counts"].items():
                regime_rows.append({"interval": interval_name, "version": config.name, "dominant_regime": regime, "count": count})

        for baseline_name, baseline_result in {
            "buy_and_hold": run_buy_and_hold(frame),
            "ma20_ma60": run_ma_crossover(frame),
        }.items():
            summary_rows.append({"interval": interval_name, "version": baseline_name, **baseline_result})

    summary = pd.DataFrame(summary_rows)
    summary = summary.loc[:, [column for column in SUMMARY_COLUMNS if column in summary.columns]]
    rankings = build_average_rankings(summary)
    regime_counts = pd.DataFrame(regime_rows)

    summary.to_csv(output_path / "summary_table.csv", index=False)
    rankings.to_csv(output_path / "average_rankings.csv", index=False)
    regime_counts.to_csv(output_path / "dominant_regime_counts.csv", index=False)
    return {"summary": summary, "average_rankings": rankings, "dominant_regime_counts": regime_counts}


def prepare_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data = data.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return data


def build_intervals(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    end = data["timestamp"].max()
    ranges = {
        "recent_90d": (end - pd.Timedelta(days=90), end),
        "previous_90d": (end - pd.Timedelta(days=180), end - pd.Timedelta(days=90)),
        "third_90d": (end - pd.Timedelta(days=270), end - pd.Timedelta(days=180)),
        "recent_180d": (end - pd.Timedelta(days=180), end),
        "recent_365d": (end - pd.Timedelta(days=365), end),
    }
    intervals = {}
    for name, (start, stop) in ranges.items():
        subset = data[(data["timestamp"] > start) & (data["timestamp"] <= stop)].reset_index(drop=True)
        if len(subset) < 80:
            raise ValueError(f"interval {name} has only {len(subset)} rows; download more data first")
        intervals[name] = subset
    return intervals


def run_control_system_backtest(frame: pd.DataFrame, config: VersionConfig, fee_rate: float = 0.0005) -> dict[str, Any]:
    market_states = build_market_state_frame(frame)
    estimated_states = estimate_market_state_frame(market_states)
    closes = pd.to_numeric(frame["close"], errors="coerce").astype(float).to_numpy()
    timestamps = frame["timestamp"].astype(str).tolist()
    risk_config = RiskConfigV1(min_turnover=0.02, min_trade_threshold=config.min_trade_threshold)

    equity = 1.0
    peak_equity = 1.0
    current_exposure = 0.0
    previous_target_exposure: float | None = None
    last_nonzero_trade_idx: int | None = None
    trade_count = 0
    turnover = 0.0
    minor_count = 0
    target_clip_count = 0
    turnover_clip_count = 0
    hard_count = 0
    any_count = 0
    kill_switch_count = 0
    reduce_only_count = 0
    cooldown_blocked_count = 0

    equity_rows = [{"timestamp": timestamps[0], "equity": equity, "period_return": 0.0, "drawdown": 0.0}]
    exposure_rows: list[dict[str, Any]] = []

    for idx in range(len(frame) - 1):
        market = market_state_from_row(market_states.iloc[idx])
        estimated = estimated_state_from_row(estimated_states.iloc[idx])
        close_t = closes[idx]
        close_next = closes[idx + 1]
        portfolio = PortfolioStateV1(
            current_exposure=current_exposure,
            current_position=current_exposure * equity / close_t if close_t > 0 else 0.0,
            equity=equity,
            cash=equity * max(0.0, 1.0 - abs(current_exposure)),
            unrealized_pnl=0.0,
            portfolio_drawdown=equity / peak_equity - 1.0,
            leverage=abs(current_exposure),
            available_margin=max(0.0, 1.0 - abs(current_exposure)),
        )
        action = compute_control_action(
            market,
            estimated,
            portfolio,
            previous_target_exposure=previous_target_exposure,
            use_target_smoothing=config.use_target_smoothing,
            target_deadband=config.target_deadband,
            beta_increase_risk=config.beta_increase_risk,
            beta_decrease_risk=config.beta_decrease_risk,
        )
        previous_target_exposure = action.target_exposure
        safe = apply_risk_filter(market, estimated, portfolio, action, config=risk_config)
        requested_change = safe.safe_exposure_change if safe.trade_allowed else 0.0
        if requested_change != 0.0 and _should_apply_cooldown(
            idx=idx,
            last_nonzero_trade_idx=last_nonzero_trade_idx,
            minimum_rebalance_interval=config.minimum_rebalance_interval,
            current_exposure=current_exposure,
            safe_action=safe,
        ):
            requested_change = 0.0
            cooldown_blocked_count += 1

        equity_before = equity
        asset_return = close_next / close_t - 1.0 if close_t > 0 else 0.0
        equity *= 1.0 + current_exposure * asset_return
        equity -= abs(requested_change) * fee_rate * equity
        equity = max(equity, 1e-12)
        peak_equity = max(peak_equity, equity)
        current_exposure = float(np.clip(current_exposure + requested_change, -1.0, 1.0))

        if abs(requested_change) > 0.0:
            trade_count += 1
            turnover += abs(requested_change)
            last_nonzero_trade_idx = idx
        intervention = _classify_layer4_intervention(action, safe)
        minor_count += int(intervention["minor"])
        target_clip_count += int(intervention["target_clip"])
        turnover_clip_count += int(intervention["turnover_clip"])
        hard_count += int(intervention["hard"])
        any_count += int(any(intervention.values()))
        kill_switch_count += int(safe.kill_switch)
        reduce_only_count += int(safe.reduce_only)

        exposure_rows.append(
            {
                "current_exposure": current_exposure,
                "allowed_turnover": safe.allowed_turnover,
                "raw_target_exposure": action.raw_target_exposure,
                "final_target_exposure": action.target_exposure,
                "safe_target_exposure": safe.safe_target_exposure,
                "cooldown_blocked": cooldown_blocked_count,
            }
        )
        equity_rows.append(
            {
                "timestamp": timestamps[idx + 1],
                "equity": equity,
                "period_return": equity / equity_before - 1.0,
                "drawdown": equity / peak_equity - 1.0,
            }
        )

    metrics = _calculate_metrics(
        pd.DataFrame(equity_rows),
        initial_equity=1.0,
        periods_per_year=PERIODS_PER_YEAR_1H,
        trade_count=trade_count,
        turnover=turnover,
        exposure_history=pd.DataFrame(exposure_rows),
        minor_intervention_count=minor_count,
        target_clip_count=target_clip_count,
        turnover_clip_count=turnover_clip_count,
        hard_intervention_count=hard_count,
        any_intervention_count=any_count,
        decision_count=max(len(frame) - 1, 1),
        kill_switch_count=kill_switch_count,
        reduce_only_count=reduce_only_count,
        cooldown_blocked_trade_count=cooldown_blocked_count,
    )
    regime_counts = estimated_states["dominant_regime"].value_counts().to_dict()
    return {"metrics": metrics, "regime_counts": regime_counts}


def market_state_from_row(row: pd.Series) -> MarketStateV1:
    return MarketStateV1(
        timestamp=0.0,
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


def estimated_state_from_row(row: pd.Series) -> EstimatedMarketStateV1:
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


def run_buy_and_hold(frame: pd.DataFrame, fee_rate: float = 0.0005) -> dict[str, float | int]:
    closes = pd.to_numeric(frame["close"], errors="coerce").astype(float)
    equity_values = [1.0 - fee_rate]
    for idx in range(len(frame) - 1):
        asset_return = closes.iloc[idx + 1] / closes.iloc[idx] - 1.0
        equity_values.append(equity_values[-1] * (1.0 + asset_return))
    equity = pd.Series(equity_values)
    drawdown = equity / equity.cummax() - 1.0
    period_return = equity.pct_change().fillna(0.0)
    return baseline_metrics(equity, period_return, drawdown, trade_count=1, turnover=1.0, average_exposure=1.0)


def run_ma_crossover(frame: pd.DataFrame, fee_rate: float = 0.0005) -> dict[str, float | int]:
    closes = pd.to_numeric(frame["close"], errors="coerce").astype(float)
    ma20 = closes.rolling(20, min_periods=1).mean()
    ma60 = closes.rolling(60, min_periods=1).mean()
    target = (ma20 > ma60).astype(float)
    exposure = 0.0
    equity_values = [1.0]
    period_returns = [0.0]
    exposures = []
    trade_count = 0
    turnover = 0.0
    for idx in range(len(frame) - 1):
        asset_return = closes.iloc[idx + 1] / closes.iloc[idx] - 1.0
        equity = equity_values[-1] * (1.0 + exposure * asset_return)
        change = float(target.iloc[idx] - exposure)
        if change != 0.0:
            trade_count += 1
            turnover += abs(change)
            equity -= abs(change) * fee_rate * equity
        exposure = float(target.iloc[idx])
        equity = max(equity, 1e-12)
        equity_values.append(equity)
        period_returns.append(equity / equity_values[-2] - 1.0)
        exposures.append(exposure)
    equity = pd.Series(equity_values)
    drawdown = equity / equity.cummax() - 1.0
    return baseline_metrics(
        equity,
        pd.Series(period_returns),
        drawdown,
        trade_count=trade_count,
        turnover=turnover,
        average_exposure=float(np.mean(np.abs(exposures))) if exposures else 0.0,
    )


def baseline_metrics(
    equity: pd.Series,
    period_return: pd.Series,
    drawdown: pd.Series,
    *,
    trade_count: int,
    turnover: float,
    average_exposure: float,
) -> dict[str, float | int]:
    returns = period_return.iloc[1:].astype(float)
    std = float(returns.std(ddof=0))
    return {
        "total_return": float(equity.iloc[-1] - 1.0),
        "max_drawdown": float(drawdown.min()),
        "sharpe": 0.0 if std == 0.0 else float(returns.mean() / std * np.sqrt(PERIODS_PER_YEAR_1H)),
        "trade_count": int(trade_count),
        "turnover": float(turnover),
        "average_exposure": float(average_exposure),
        "target_clip_rate": 0.0,
        "turnover_clip_rate": 0.0,
        "kill_switch_count": 0,
        "reduce_only_count": 0,
    }


def build_average_rankings(summary: pd.DataFrame) -> pd.DataFrame:
    ranked = summary.copy()
    ranked["abs_drawdown"] = ranked["max_drawdown"].abs()
    ranked["stability_value"] = (
        ranked["abs_drawdown"]
        + ranked["turnover_clip_rate"].fillna(0.0)
        + ranked["target_clip_rate"].fillna(0.0)
        + ranked["kill_switch_count"].fillna(0.0) * 0.1
    )
    rank_rows = []
    for interval, group in ranked.groupby("interval"):
        local = group.copy()
        local["return_rank"] = local["total_return"].rank(ascending=False, method="average")
        local["drawdown_rank"] = local["abs_drawdown"].rank(ascending=True, method="average")
        local["sharpe_rank"] = local["sharpe"].rank(ascending=False, method="average")
        local["turnover_rank"] = local["turnover"].rank(ascending=True, method="average")
        local["stability_rank"] = local["stability_value"].rank(ascending=True, method="average")
        rank_rows.append(local[["version", "return_rank", "drawdown_rank", "sharpe_rank", "turnover_rank", "stability_rank"]])
    all_ranks = pd.concat(rank_rows, ignore_index=True)
    return all_ranks.groupby("version", as_index=False).mean().sort_values("sharpe_rank")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-interval validation for BTCUSDT 1h.")
    parser.add_argument("--csv", type=Path, default=Path("data/btcusdt_1h_365d.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/multi_interval_validation"))
    args = parser.parse_args()
    outputs = run_validation(args.csv, args.output_dir)
    print("Summary")
    print(outputs["summary"].to_string(index=False))
    print("\nAverage rankings")
    print(outputs["average_rankings"].to_string(index=False))
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()

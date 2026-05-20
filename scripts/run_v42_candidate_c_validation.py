from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v4 import (  # noqa: E402
    BacktestConfig,
    BacktestEngine,
    ControllerConfig,
    FilteredSignals,
    MinimalContinuousController,
    MinimalFilterLayer,
    MinimalStateEstimator,
    Observation,
    StateVector,
    evaluate_metrics,
    make_v42_candidate_a_config,
    make_v42_candidate_c_config,
    run_fixed_exposure_benchmark,
    run_zero_position_benchmark,
)


PERIODS_PER_YEAR = 365
FEE_RATES = (0.0, 0.001, 0.002)
OUT_DIR = Path("reports") / "v42_candidate_c_validation"
DATASETS = (
    ("BTC", "365d", Path("data") / "btcusdt_1h_365d.csv"),
    ("BTC", "2y", Path("data") / "btcusdt_1h_2y.csv"),
    ("BTC", "3y", Path("data") / "btcusdt_1h_3y.csv"),
    ("BTC", "5y", Path("data") / "btcusdt_1h_5y.csv"),
    ("ETH", "365d", Path("data") / "ethusdt_1h_365d.csv"),
    ("ETH", "2y", Path("data") / "ethusdt_1h_2y.csv"),
    ("ETH", "3y", Path("data") / "ethusdt_1h_3y.csv"),
    ("ETH", "5y", Path("data") / "ethusdt_1h_5y.csv"),
)
OUTPUT_COLUMNS = [
    "asset",
    "window",
    "fee_rate",
    "strategy_name",
    "total_return",
    "annualized_return",
    "max_drawdown",
    "sharpe_ratio",
    "total_turnover",
    "average_turnover",
    "average_exposure",
    "total_fee_cost",
    "trade_count",
    "final_equity",
    "min_position",
    "max_position",
    "position_std",
    "average_drawdown",
    "pct_position_gt_0",
    "pct_position_gt_0_05",
    "pct_position_gt_0_10",
    "pct_position_gt_0_20",
    "pct_tau_gt_0",
    "pct_tau_gt_0_1",
    "pct_tau_gt_0_25",
    "pct_tau_gt_0_5",
    "pct_nu_gt_0_5",
    "pct_rho_gt_0_5",
    "base_exposure_mean",
    "base_exposure_median",
    "base_exposure_p25",
    "base_exposure_p75",
    "unsmoothed_target_mean",
    "unsmoothed_target_median",
    "market_risk_multiplier_mean",
    "portfolio_risk_multiplier_mean",
]


class RecordingStateEstimator:
    def __init__(self, config) -> None:
        self.filter_layer = MinimalFilterLayer(config.filter)
        self.mapper = MinimalStateEstimator(config=config.state_estimator)
        self.filtered_history: list[FilteredSignals] = []
        self.state_history: list[StateVector] = []

    def update(self, observation: Observation) -> StateVector:
        filtered = self.filter_layer.update(observation)
        state = self.mapper.estimate_from_filtered(filtered)
        self.filtered_history.append(filtered)
        self.state_history.append(state)
        return state


class RecordingController:
    def __init__(self, config: ControllerConfig) -> None:
        self.controller = MinimalContinuousController(config)
        self.trace_history: list[dict[str, float]] = []

    def decide(self, state: StateVector) -> float:
        trace = self.controller.explain(state)
        self.trace_history.append(trace)
        return trace["raw_target_position"]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for asset, window, path in DATASETS:
        prices = load_daily_close(path)
        for fee_rate in FEE_RATES:
            rows.extend(validate_dataset(asset=asset, window=window, prices=prices, fee_rate=fee_rate))

    comparison = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    csv_path = OUT_DIR / "v42_candidate_c_validation_comparison.csv"
    md_path = OUT_DIR / "v42_candidate_c_validation_comparison.md"
    comparison.to_csv(csv_path, index=False)
    md_path.write_text(frame_to_markdown(comparison), encoding="utf-8")
    print(f"comparison_csv: {csv_path}")
    print(f"comparison_markdown: {md_path}")
    print(comparison.to_string(index=False))


def validate_dataset(*, asset: str, window: str, prices: pd.Series, fee_rate: float) -> list[dict[str, Any]]:
    benchmark_config = BacktestConfig(fee_rate=fee_rate)
    return [
        run_control_strategy(
            asset=asset,
            window=window,
            fee_rate=fee_rate,
            prices=prices,
            strategy_name="v4.1_default",
            config=make_v41_like_config(fee_rate=fee_rate),
        ),
        run_control_strategy(
            asset=asset,
            window=window,
            fee_rate=fee_rate,
            prices=prices,
            strategy_name="v4.2_candidate_A",
            config=make_v42_candidate_a_config(fee_rate=fee_rate),
        ),
        run_control_strategy(
            asset=asset,
            window=window,
            fee_rate=fee_rate,
            prices=prices,
            strategy_name="v4.2_candidate_C",
            config=make_v42_candidate_c_config(fee_rate=fee_rate),
        ),
        summarize_true_buy_and_hold(asset=asset, window=window, fee_rate=fee_rate, prices=prices),
        summarize_backtest_result(
            asset=asset,
            window=window,
            fee_rate=fee_rate,
            strategy_name="fixed_0_5_exposure",
            result=run_fixed_exposure_benchmark(prices, exposure=0.5, config=benchmark_config),
            state_frame=None,
            trace_frame=None,
        ),
        summarize_backtest_result(
            asset=asset,
            window=window,
            fee_rate=fee_rate,
            strategy_name="zero_position",
            result=run_zero_position_benchmark(prices, config=benchmark_config),
            state_frame=None,
            trace_frame=None,
        ),
    ]


def make_v41_like_config(*, fee_rate: float):
    config = make_v42_candidate_a_config(fee_rate=fee_rate)
    return type(config)(
        version_name="v4.1-minimal-control-strategy",
        periods_per_year=config.periods_per_year,
        backtest=config.backtest,
        filter=config.filter,
        state_estimator=type(config.state_estimator)(
            k_tau=1.0,
            k_epsilon=config.state_estimator.k_epsilon,
            vol_ref=config.state_estimator.vol_ref,
            drawdown_ref=config.state_estimator.drawdown_ref,
            epsilon=config.state_estimator.epsilon,
        ),
        controller=config.controller,
    )


def run_control_strategy(
    *,
    asset: str,
    window: str,
    fee_rate: float,
    prices: pd.Series,
    strategy_name: str,
    config,
) -> dict[str, Any]:
    estimator = RecordingStateEstimator(config)
    controller = RecordingController(config.controller)
    result = BacktestEngine(
        controller=controller,
        state_estimator=estimator,
        config=config.backtest,
    ).run(prices)
    state_frame = pd.DataFrame(
        {
            "tau": [state.tau for state in estimator.state_history],
            "nu": [state.nu for state in estimator.state_history],
            "rho": [state.rho for state in estimator.state_history],
        }
    )
    trace_frame = pd.DataFrame(controller.trace_history)
    return summarize_backtest_result(
        asset=asset,
        window=window,
        fee_rate=fee_rate,
        strategy_name=strategy_name,
        result=result,
        state_frame=state_frame,
        trace_frame=trace_frame,
    )


def summarize_backtest_result(
    *,
    asset: str,
    window: str,
    fee_rate: float,
    strategy_name: str,
    result: pd.DataFrame,
    state_frame: pd.DataFrame | None,
    trace_frame: pd.DataFrame | None,
) -> dict[str, Any]:
    metrics = evaluate_metrics(result, periods_per_year=PERIODS_PER_YEAR)
    row = {
        "asset": asset,
        "window": window,
        "fee_rate": fee_rate,
        "strategy_name": strategy_name,
        **metrics,
        "final_equity": float(result["equity"].iloc[-1]),
        "min_position": float(result["position"].min()),
        "max_position": float(result["position"].max()),
        "position_std": float(result["position"].std(ddof=0)),
        "average_drawdown": float(result["drawdown"].mean()),
        "pct_position_gt_0": pct_gt(result["position"], 0.0),
        "pct_position_gt_0_05": pct_gt(result["position"], 0.05),
        "pct_position_gt_0_10": pct_gt(result["position"], 0.10),
        "pct_position_gt_0_20": pct_gt(result["position"], 0.20),
    }
    row.update(state_percentages(state_frame))
    row.update(controller_diagnostics(trace_frame))
    return row


def state_percentages(state_frame: pd.DataFrame | None) -> dict[str, float]:
    if state_frame is None:
        return {
            "pct_tau_gt_0": math.nan,
            "pct_tau_gt_0_1": math.nan,
            "pct_tau_gt_0_25": math.nan,
            "pct_tau_gt_0_5": math.nan,
            "pct_nu_gt_0_5": math.nan,
            "pct_rho_gt_0_5": math.nan,
        }
    return {
        "pct_tau_gt_0": pct_gt(state_frame["tau"], 0.0),
        "pct_tau_gt_0_1": pct_gt(state_frame["tau"], 0.1),
        "pct_tau_gt_0_25": pct_gt(state_frame["tau"], 0.25),
        "pct_tau_gt_0_5": pct_gt(state_frame["tau"], 0.5),
        "pct_nu_gt_0_5": pct_gt(state_frame["nu"], 0.5),
        "pct_rho_gt_0_5": pct_gt(state_frame["rho"], 0.5),
    }


def controller_diagnostics(trace_frame: pd.DataFrame | None) -> dict[str, float]:
    if trace_frame is None or trace_frame.empty:
        return {
            "base_exposure_mean": math.nan,
            "base_exposure_median": math.nan,
            "base_exposure_p25": math.nan,
            "base_exposure_p75": math.nan,
            "unsmoothed_target_mean": math.nan,
            "unsmoothed_target_median": math.nan,
            "market_risk_multiplier_mean": math.nan,
            "portfolio_risk_multiplier_mean": math.nan,
        }
    return {
        "base_exposure_mean": stat(trace_frame["base_exposure"], "mean"),
        "base_exposure_median": stat(trace_frame["base_exposure"], "median"),
        "base_exposure_p25": stat(trace_frame["base_exposure"], "p25"),
        "base_exposure_p75": stat(trace_frame["base_exposure"], "p75"),
        "unsmoothed_target_mean": stat(trace_frame["unsmoothed_target"], "mean"),
        "unsmoothed_target_median": stat(trace_frame["unsmoothed_target"], "median"),
        "market_risk_multiplier_mean": stat(trace_frame["market_risk_multiplier"], "mean"),
        "portfolio_risk_multiplier_mean": stat(trace_frame["portfolio_risk_multiplier"], "mean"),
    }


def summarize_true_buy_and_hold(*, asset: str, window: str, fee_rate: float, prices: pd.Series) -> dict[str, Any]:
    price_values = prices.astype(float).to_numpy()
    period_count = len(price_values) - 1
    equity = (1.0 - fee_rate) * price_values[1:] / price_values[0]
    equity_with_initial = np.concatenate([[1.0], equity])
    returns = equity_with_initial[1:] / equity_with_initial[:-1] - 1.0
    high_watermark = np.maximum.accumulate(equity_with_initial)
    drawdown = 1.0 - equity_with_initial[1:] / high_watermark[1:]
    std_return = float(np.std(returns))
    sharpe = math.nan if std_return == 0.0 else float(np.mean(returns)) / std_return * math.sqrt(PERIODS_PER_YEAR)
    final_equity = float(equity[-1])
    row = {
        "asset": asset,
        "window": window,
        "fee_rate": fee_rate,
        "strategy_name": "true_buy_and_hold",
        "total_return": final_equity - 1.0,
        "annualized_return": final_equity ** (PERIODS_PER_YEAR / period_count) - 1.0,
        "max_drawdown": float(np.max(drawdown)),
        "sharpe_ratio": sharpe,
        "total_turnover": 1.0 if fee_rate > 0.0 else 0.0,
        "average_turnover": (1.0 if fee_rate > 0.0 else 0.0) / period_count,
        "average_exposure": 1.0,
        "total_fee_cost": fee_rate,
        "trade_count": 1 if fee_rate > 0.0 else 0,
        "final_equity": final_equity,
        "min_position": 1.0,
        "max_position": 1.0,
        "position_std": 0.0,
        "average_drawdown": float(np.mean(drawdown)),
        "pct_position_gt_0": 1.0,
        "pct_position_gt_0_05": 1.0,
        "pct_position_gt_0_10": 1.0,
        "pct_position_gt_0_20": 1.0,
    }
    row.update(state_percentages(None))
    row.update(controller_diagnostics(None))
    return row


def load_daily_close(path: Path) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if "timestamp" not in frame.columns or "close" not in frame.columns:
        raise ValueError(f"{path} must contain timestamp and close columns")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("timestamp").set_index("timestamp")
    daily = frame["close"].astype(float).resample("1D").last().dropna()
    if len(daily) < 2:
        raise ValueError(f"{path} produced fewer than two daily closes")
    if (daily <= 0.0).any():
        raise ValueError(f"{path} contains non-positive daily closes")
    return daily


def pct_gt(series: pd.Series, threshold: float) -> float:
    return float((series.astype(float) > threshold).mean())


def stat(series: pd.Series, name: str) -> float:
    values = series.astype(float)
    if name == "mean":
        return float(values.mean())
    if name == "median":
        return float(values.median())
    if name == "p25":
        return float(values.quantile(0.25))
    if name == "p75":
        return float(values.quantile(0.75))
    raise ValueError(f"unknown stat {name}")


def frame_to_markdown(frame: pd.DataFrame) -> str:
    rendered = frame.copy()
    for column in rendered.columns:
        if pd.api.types.is_float_dtype(rendered[column]):
            rendered[column] = rendered[column].map(lambda value: "nan" if pd.isna(value) else f"{value:.6f}")
        else:
            rendered[column] = rendered[column].astype(str)
    headers = list(rendered.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rendered.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()

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
    StateEstimatorConfig,
    StateVector,
    V41DefaultConfig,
    create_v41_default_config,
    evaluate_metrics,
    make_v42_candidate_a_config,
    make_v42_candidate_c_config,
    make_v42_candidate_d_config,
    run_fixed_exposure_benchmark,
    run_zero_position_benchmark,
)


PERIODS_PER_YEAR = 365
FEE_RATES = (0.0, 0.001, 0.002)
MAIN_DIAGNOSTIC_FEE_RATE = 0.001
OUT_DIR = Path("reports") / "v42_candidate_d_validation"
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
VALIDATION_COLUMNS = [
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
    "deadband_skip_count",
    "deadband_skip_rate",
    "total_turnover_reduction_vs_C",
    "total_fee_reduction_vs_C",
    "trade_count_reduction_vs_C",
]
CONTROL_STRATEGIES = ("v4.1_default", "v4.2_candidate_A", "v4.2_candidate_C", "v4.2_candidate_D")
ATTRIBUTION_STRATEGIES = (
    "v4.1_default",
    "v4.2_candidate_A",
    "v4.2_candidate_C",
    "v4.2_candidate_D",
    "true_buy_and_hold",
    "fixed_0_5_exposure",
)


class RecordingStateEstimator:
    def __init__(self, config: V41DefaultConfig) -> None:
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
    validation = run_validation()
    validation_csv = OUT_DIR / "v42_candidate_d_validation_comparison.csv"
    validation_md = OUT_DIR / "v42_candidate_d_validation_comparison.md"
    validation.to_csv(validation_csv, index=False)
    validation_md.write_text(frame_to_markdown(validation), encoding="utf-8")

    segment, false_positive, whipsaw, drawdown = run_eth_attribution()
    segment_csv = OUT_DIR / "eth_5y_vs_3y_segment_attribution.csv"
    segment_md = OUT_DIR / "eth_5y_vs_3y_segment_attribution.md"
    false_csv = OUT_DIR / "eth_false_positive_trend_diagnostic.csv"
    whipsaw_csv = OUT_DIR / "eth_whipsaw_fee_diagnostic.csv"
    drawdown_csv = OUT_DIR / "eth_drawdown_lag_diagnostic.csv"

    segment.to_csv(segment_csv, index=False)
    false_positive.to_csv(false_csv, index=False)
    whipsaw.to_csv(whipsaw_csv, index=False)
    drawdown.to_csv(drawdown_csv, index=False)
    segment_md.write_text(attribution_markdown(segment), encoding="utf-8")

    print(f"validation_csv: {validation_csv}")
    print(f"validation_markdown: {validation_md}")
    print(f"segment_attribution_csv: {segment_csv}")
    print(f"segment_attribution_markdown: {segment_md}")
    print(f"false_positive_csv: {false_csv}")
    print(f"whipsaw_csv: {whipsaw_csv}")
    print(f"drawdown_lag_csv: {drawdown_csv}")
    print(validation.to_string(index=False))


def run_validation() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for asset, window, path in DATASETS:
        prices = load_daily_close(path)
        for fee_rate in FEE_RATES:
            dataset_rows = validate_dataset(asset=asset, window=window, prices=prices, fee_rate=fee_rate)
            rows.extend(add_reductions_vs_c(dataset_rows))
    return pd.DataFrame(rows, columns=VALIDATION_COLUMNS)


def validate_dataset(*, asset: str, window: str, prices: pd.Series, fee_rate: float) -> list[dict[str, Any]]:
    benchmark_config = BacktestConfig(fee_rate=fee_rate)
    return [
        summarize_run_bundle(
            asset=asset,
            window=window,
            fee_rate=fee_rate,
            strategy_name="v4.1_default",
            bundle=run_strategy_bundle(prices, make_v41_config(fee_rate=fee_rate), "v4.1_default"),
        ),
        summarize_run_bundle(
            asset=asset,
            window=window,
            fee_rate=fee_rate,
            strategy_name="v4.2_candidate_A",
            bundle=run_strategy_bundle(prices, make_v42_candidate_a_config(fee_rate=fee_rate), "v4.2_candidate_A"),
        ),
        summarize_run_bundle(
            asset=asset,
            window=window,
            fee_rate=fee_rate,
            strategy_name="v4.2_candidate_C",
            bundle=run_strategy_bundle(prices, make_v42_candidate_c_config(fee_rate=fee_rate), "v4.2_candidate_C"),
        ),
        summarize_run_bundle(
            asset=asset,
            window=window,
            fee_rate=fee_rate,
            strategy_name="v4.2_candidate_D",
            bundle=run_strategy_bundle(prices, make_v42_candidate_d_config(fee_rate=fee_rate), "v4.2_candidate_D"),
        ),
        summarize_true_buy_and_hold(asset=asset, window=window, fee_rate=fee_rate, prices=prices),
        summarize_benchmark_result(
            asset=asset,
            window=window,
            fee_rate=fee_rate,
            strategy_name="fixed_0_5_exposure",
            result=run_fixed_exposure_benchmark(prices, exposure=0.5, config=benchmark_config),
        ),
        summarize_benchmark_result(
            asset=asset,
            window=window,
            fee_rate=fee_rate,
            strategy_name="zero_position",
            result=run_zero_position_benchmark(prices, config=benchmark_config),
        ),
    ]


def add_reductions_vs_c(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    c_row = next(row for row in rows if row["strategy_name"] == "v4.2_candidate_C")
    for row in rows:
        if row["strategy_name"] == "v4.2_candidate_C":
            row["total_turnover_reduction_vs_C"] = 0.0
            row["total_fee_reduction_vs_C"] = 0.0
            row["trade_count_reduction_vs_C"] = 0
        elif row["strategy_name"] == "v4.2_candidate_D":
            row["total_turnover_reduction_vs_C"] = c_row["total_turnover"] - row["total_turnover"]
            row["total_fee_reduction_vs_C"] = c_row["total_fee_cost"] - row["total_fee_cost"]
            row["trade_count_reduction_vs_C"] = c_row["trade_count"] - row["trade_count"]
        else:
            row["total_turnover_reduction_vs_C"] = math.nan
            row["total_fee_reduction_vs_C"] = math.nan
            row["trade_count_reduction_vs_C"] = math.nan
    return rows


def run_eth_attribution() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eth5 = load_daily_close(Path("data") / "ethusdt_1h_5y.csv")
    eth3 = load_daily_close(Path("data") / "ethusdt_1h_3y.csv")
    recent_start = eth3.index[1]
    segments = {
        "ETH_extra_early_2y": lambda frame: frame["timestamp"] < recent_start,
        "ETH_3y_recent": lambda frame: frame["timestamp"] >= recent_start,
        "ETH_5y_full": lambda frame: pd.Series(True, index=frame.index),
    }

    segment_rows: list[dict[str, Any]] = []
    false_rows: list[dict[str, Any]] = []
    whipsaw_rows: list[dict[str, Any]] = []
    drawdown_rows: list[dict[str, Any]] = []

    for fee_rate in FEE_RATES:
        continuous_bundles = build_attribution_bundles(eth5, fee_rate)
        reset_bundles = build_reset_segment_bundles(eth5, eth3, recent_start, fee_rate)
        for strategy_name, bundle in continuous_bundles.items():
            full_frame = bundle["frame"]
            for segment_name, mask_fn in segments.items():
                mask = mask_fn(full_frame)
                segment_frame = full_frame.loc[mask].copy()
                if segment_frame.empty:
                    continue
                segment_rows.append(
                    segment_metrics(
                        frame=full_frame,
                        segment_frame=segment_frame,
                        fee_rate=fee_rate,
                        strategy_name=strategy_name,
                        segment_name=segment_name,
                        state_mode="continuous_state_segment_metrics",
                    )
                )
                segment_rows.append(
                    pnl_attribution(
                        frame=full_frame,
                        segment_frame=segment_frame,
                        fee_rate=fee_rate,
                        strategy_name=strategy_name,
                        segment_name=segment_name,
                        state_mode="continuous_state_segment_metrics",
                    )
                )
                whipsaw_rows.append(
                    whipsaw_diagnostic(
                        segment_frame=segment_frame,
                        fee_rate=fee_rate,
                        strategy_name=strategy_name,
                        segment_name=segment_name,
                    )
                )
                if strategy_name in CONTROL_STRATEGIES:
                    false_rows.extend(
                        false_positive_diagnostics(
                            full_prices=eth5,
                            segment_frame=segment_frame,
                            fee_rate=fee_rate,
                            strategy_name=strategy_name,
                            segment_name=segment_name,
                        )
                    )
                    drawdown_rows.extend(
                        drawdown_lag_diagnostics(
                            segment_frame=segment_frame,
                            fee_rate=fee_rate,
                            strategy_name=strategy_name,
                            segment_name=segment_name,
                        )
                    )
        for strategy_name, segment_bundle_map in reset_bundles.items():
            for segment_name, bundle in segment_bundle_map.items():
                segment_rows.append(
                    segment_metrics(
                        frame=bundle["frame"],
                        segment_frame=bundle["frame"],
                        fee_rate=fee_rate,
                        strategy_name=strategy_name,
                        segment_name=segment_name,
                        state_mode="reset_state_segment_metrics",
                    )
                )
                segment_rows.append(
                    pnl_attribution(
                        frame=bundle["frame"],
                        segment_frame=bundle["frame"],
                        fee_rate=fee_rate,
                        strategy_name=strategy_name,
                        segment_name=segment_name,
                        state_mode="reset_state_segment_metrics",
                    )
                )

    return (
        pd.DataFrame(segment_rows),
        pd.DataFrame(false_rows),
        pd.DataFrame(whipsaw_rows),
        pd.DataFrame(drawdown_rows),
    )


def build_attribution_bundles(prices: pd.Series, fee_rate: float) -> dict[str, dict[str, Any]]:
    return {
        "v4.1_default": run_strategy_bundle(prices, make_v41_config(fee_rate=fee_rate), "v4.1_default"),
        "v4.2_candidate_A": run_strategy_bundle(prices, make_v42_candidate_a_config(fee_rate=fee_rate), "v4.2_candidate_A"),
        "v4.2_candidate_C": run_strategy_bundle(prices, make_v42_candidate_c_config(fee_rate=fee_rate), "v4.2_candidate_C"),
        "v4.2_candidate_D": run_strategy_bundle(prices, make_v42_candidate_d_config(fee_rate=fee_rate), "v4.2_candidate_D"),
        "true_buy_and_hold": {"frame": true_buy_and_hold_frame(prices, fee_rate)},
        "fixed_0_5_exposure": {
            "frame": run_fixed_exposure_benchmark(prices, exposure=0.5, config=BacktestConfig(fee_rate=fee_rate))
        },
    }


def build_reset_segment_bundles(
    eth5: pd.Series,
    eth3: pd.Series,
    recent_start: pd.Timestamp,
    fee_rate: float,
) -> dict[str, dict[str, dict[str, Any]]]:
    early_prices = eth5.loc[eth5.index < recent_start]
    recent_prices = eth3
    segment_prices = {
        "ETH_extra_early_2y": early_prices,
        "ETH_3y_recent": recent_prices,
        "ETH_5y_full": eth5,
    }
    bundles: dict[str, dict[str, dict[str, Any]]] = {}
    for strategy_name, config_factory in (
        ("v4.1_default", make_v41_config),
        ("v4.2_candidate_A", make_v42_candidate_a_config),
        ("v4.2_candidate_C", make_v42_candidate_c_config),
        ("v4.2_candidate_D", make_v42_candidate_d_config),
    ):
        bundles[strategy_name] = {
            segment_name: run_strategy_bundle(prices, config_factory(fee_rate=fee_rate), strategy_name)
            for segment_name, prices in segment_prices.items()
        }
    bundles["true_buy_and_hold"] = {
        segment_name: {"frame": true_buy_and_hold_frame(prices, fee_rate)}
        for segment_name, prices in segment_prices.items()
    }
    bundles["fixed_0_5_exposure"] = {
        segment_name: {"frame": run_fixed_exposure_benchmark(prices, exposure=0.5, config=BacktestConfig(fee_rate=fee_rate))}
        for segment_name, prices in segment_prices.items()
    }
    return bundles


def make_v41_config(*, fee_rate: float) -> V41DefaultConfig:
    v41 = create_v41_default_config()
    return V41DefaultConfig(
        version_name=v41.version_name,
        periods_per_year=v41.periods_per_year,
        backtest=BacktestConfig(
            fee_rate=fee_rate,
            initial_equity=v41.backtest.initial_equity,
            initial_high_watermark=v41.backtest.initial_high_watermark,
            initial_position=v41.backtest.initial_position,
        ),
        filter=v41.filter,
        state_estimator=StateEstimatorConfig(
            k_tau=v41.state_estimator.k_tau,
            k_epsilon=v41.state_estimator.k_epsilon,
            vol_ref=v41.state_estimator.vol_ref,
            drawdown_ref=v41.state_estimator.drawdown_ref,
            epsilon=v41.state_estimator.epsilon,
        ),
        controller=v41.controller,
    )


def run_strategy_bundle(prices: pd.Series, config: V41DefaultConfig, strategy_name: str) -> dict[str, Any]:
    estimator = RecordingStateEstimator(config)
    controller = RecordingController(config.controller)
    result = BacktestEngine(controller=controller, state_estimator=estimator, config=config.backtest).run(prices)
    frame = result.copy()
    frame["strategy_name"] = strategy_name
    frame["tau"] = [state.tau for state in estimator.state_history]
    frame["nu"] = [state.nu for state in estimator.state_history]
    frame["epsilon"] = [state.epsilon for state in estimator.state_history]
    frame["rho"] = [state.rho for state in estimator.state_history]
    for column in (
        "base_exposure",
        "timing_multiplier",
        "market_risk_multiplier",
        "portfolio_risk_multiplier",
        "unsmoothed_target",
        "clipped_delta",
        "pre_deadband_target",
        "deadband_skip",
    ):
        frame[column] = [trace[column] for trace in controller.trace_history]
    return {
        "frame": frame,
        "result": result,
        "state_history": estimator.state_history,
        "trace_history": controller.trace_history,
    }


def summarize_run_bundle(
    *,
    asset: str,
    window: str,
    fee_rate: float,
    strategy_name: str,
    bundle: dict[str, Any],
) -> dict[str, Any]:
    frame = bundle["frame"]
    metrics = evaluate_metrics(frame, periods_per_year=PERIODS_PER_YEAR)
    row = base_summary(asset, window, fee_rate, strategy_name, frame, metrics)
    row.update(
        {
            "pct_tau_gt_0": pct_gt(frame["tau"], 0.0),
            "pct_tau_gt_0_1": pct_gt(frame["tau"], 0.1),
            "pct_tau_gt_0_25": pct_gt(frame["tau"], 0.25),
            "pct_tau_gt_0_5": pct_gt(frame["tau"], 0.5),
            "pct_nu_gt_0_5": pct_gt(frame["nu"], 0.5),
            "pct_rho_gt_0_5": pct_gt(frame["rho"], 0.5),
            "deadband_skip_count": int(frame["deadband_skip"].sum()),
            "deadband_skip_rate": float(frame["deadband_skip"].mean()),
            "total_turnover_reduction_vs_C": math.nan,
            "total_fee_reduction_vs_C": math.nan,
            "trade_count_reduction_vs_C": math.nan,
        }
    )
    return row


def summarize_benchmark_result(
    *,
    asset: str,
    window: str,
    fee_rate: float,
    strategy_name: str,
    result: pd.DataFrame,
) -> dict[str, Any]:
    metrics = evaluate_metrics(result, periods_per_year=PERIODS_PER_YEAR)
    row = base_summary(asset, window, fee_rate, strategy_name, result, metrics)
    row.update(empty_state_and_deadband_metrics())
    return row


def summarize_true_buy_and_hold(*, asset: str, window: str, fee_rate: float, prices: pd.Series) -> dict[str, Any]:
    frame = true_buy_and_hold_frame(prices, fee_rate)
    metrics = manual_metrics(frame)
    row = base_summary(asset, window, fee_rate, "true_buy_and_hold", frame, metrics)
    row.update(empty_state_and_deadband_metrics())
    return row


def base_summary(
    asset: str,
    window: str,
    fee_rate: float,
    strategy_name: str,
    result: pd.DataFrame,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
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


def empty_state_and_deadband_metrics() -> dict[str, Any]:
    return {
        "pct_tau_gt_0": math.nan,
        "pct_tau_gt_0_1": math.nan,
        "pct_tau_gt_0_25": math.nan,
        "pct_tau_gt_0_5": math.nan,
        "pct_nu_gt_0_5": math.nan,
        "pct_rho_gt_0_5": math.nan,
        "deadband_skip_count": math.nan,
        "deadband_skip_rate": math.nan,
        "total_turnover_reduction_vs_C": math.nan,
        "total_fee_reduction_vs_C": math.nan,
        "trade_count_reduction_vs_C": math.nan,
    }


def true_buy_and_hold_frame(prices: pd.Series, fee_rate: float) -> pd.DataFrame:
    values = prices.astype(float)
    rows = []
    equity_values = []
    for i in range(1, len(values)):
        equity = (1.0 - fee_rate) * float(values.iloc[i]) / float(values.iloc[0])
        equity_values.append(equity)
    high_watermarks = np.maximum.accumulate(np.concatenate([[1.0], np.array(equity_values)]))[1:]
    for i in range(1, len(values)):
        simple_return = float(values.iloc[i]) / float(values.iloc[i - 1]) - 1.0
        equity = equity_values[i - 1]
        rows.append(
            {
                "timestamp": values.index[i],
                "price": float(values.iloc[i]),
                "simple_return": simple_return,
                "log_return": math.log(float(values.iloc[i]) / float(values.iloc[i - 1])),
                "pre_trade_equity": equity,
                "pre_trade_high_watermark": float(high_watermarks[i - 1]),
                "pre_trade_drawdown": 1.0 - equity / float(high_watermarks[i - 1]),
                "previous_position": 1.0,
                "raw_target_position": 1.0,
                "position": 1.0,
                "turnover": 1.0 if i == 1 and fee_rate > 0.0 else 0.0,
                "transaction_cost": fee_rate if i == 1 else 0.0,
                "equity": equity,
                "high_watermark": float(high_watermarks[i - 1]),
                "drawdown": 1.0 - equity / float(high_watermarks[i - 1]),
            }
        )
    frame = pd.DataFrame(rows)
    frame.attrs["initial_equity"] = 1.0
    return frame


def manual_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    return segment_metric_values(frame, starting_equity=1.0)


def segment_metrics(
    *,
    frame: pd.DataFrame,
    segment_frame: pd.DataFrame,
    fee_rate: float,
    strategy_name: str,
    segment_name: str,
    state_mode: str,
) -> dict[str, Any]:
    starting_equity = starting_equity_for_segment(frame, segment_frame)
    metrics = segment_metric_values(segment_frame, starting_equity=starting_equity)
    return {
        "diagnostic_type": "segment_metrics",
        "state_mode": state_mode,
        "fee_rate": fee_rate,
        "strategy_name": strategy_name,
        "segment_name": segment_name,
        "segment_start_date": str(segment_frame["timestamp"].iloc[0].date()),
        "segment_end_date": str(segment_frame["timestamp"].iloc[-1].date()),
        "segment_days": len(segment_frame),
        "segment_total_return": metrics["total_return"],
        "segment_annualized_return": metrics["annualized_return"],
        "segment_max_drawdown": metrics["max_drawdown"],
        "segment_sharpe": metrics["sharpe_ratio"],
        "segment_total_turnover": metrics["total_turnover"],
        "segment_average_exposure": metrics["average_exposure"],
        "segment_total_fee_cost": metrics["total_fee_cost"],
        "segment_trade_count": metrics["trade_count"],
        "segment_average_drawdown": float(segment_frame["drawdown"].mean()),
        "segment_pct_position_gt_0": pct_gt(segment_frame["position"], 0.0),
        "segment_pct_position_gt_0_05": pct_gt(segment_frame["position"], 0.05),
        "segment_pct_position_gt_0_10": pct_gt(segment_frame["position"], 0.10),
        "segment_pct_position_gt_0_20": pct_gt(segment_frame["position"], 0.20),
        "segment_pct_rho_gt_0_5": pct_gt(segment_frame["rho"], 0.5) if "rho" in segment_frame else math.nan,
    }


def pnl_attribution(
    *,
    frame: pd.DataFrame,
    segment_frame: pd.DataFrame,
    fee_rate: float,
    strategy_name: str,
    segment_name: str,
    state_mode: str,
) -> dict[str, Any]:
    returns = segment_frame["simple_return"].astype(float)
    positions = segment_frame["previous_position"].astype(float)
    gross = float((positions * returns).sum())
    equity_return = compounded_segment_return(frame, segment_frame)
    negative = returns < 0.0
    positive = returns > 0.0
    return {
        "diagnostic_type": "pnl_attribution",
        "state_mode": state_mode,
        "fee_rate": fee_rate,
        "strategy_name": strategy_name,
        "segment_name": segment_name,
        "segment_start_date": str(segment_frame["timestamp"].iloc[0].date()),
        "segment_end_date": str(segment_frame["timestamp"].iloc[-1].date()),
        "segment_days": len(segment_frame),
        "gross_position_return": gross,
        "net_strategy_return": equity_return,
        "total_fee_cost": float(segment_frame["transaction_cost"].sum()),
        "fee_drag": gross - equity_return,
        "average_position_when_market_return_negative": safe_mean(positions[negative]),
        "average_position_when_market_return_positive": safe_mean(positions[positive]),
        "total_pnl_from_negative_return_days": float((positions[negative] * returns[negative]).sum()),
        "total_pnl_from_positive_return_days": float((positions[positive] * returns[positive]).sum()),
    }


def false_positive_diagnostics(
    *,
    full_prices: pd.Series,
    segment_frame: pd.DataFrame,
    fee_rate: float,
    strategy_name: str,
    segment_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    indexed = segment_frame.set_index("timestamp").copy()
    for horizon in (1, 5, 20):
        forward = full_prices.shift(-horizon) / full_prices - 1.0
        indexed[f"forward_{horizon}d"] = forward.reindex(indexed.index)
        valid = indexed.dropna(subset=[f"forward_{horizon}d"])
        fwd = valid[f"forward_{horizon}d"]
        rows.extend(
            pct_condition_rows(
                valid,
                fwd,
                fee_rate=fee_rate,
                strategy_name=strategy_name,
                segment_name=segment_name,
                horizon=f"{horizon}d",
            )
        )
        rows.extend(
            conditional_forward_rows(
                valid,
                fwd,
                fee_rate=fee_rate,
                strategy_name=strategy_name,
                segment_name=segment_name,
                horizon=f"{horizon}d",
            )
        )
    return rows


def pct_condition_rows(
    frame: pd.DataFrame,
    fwd: pd.Series,
    *,
    fee_rate: float,
    strategy_name: str,
    segment_name: str,
    horizon: str,
) -> list[dict[str, Any]]:
    conditions = {
        "pct_days_tau_gt_0_and_forward_negative": frame["tau"] > 0.0,
        "pct_days_tau_gt_0_1_and_forward_negative": frame["tau"] > 0.1,
        "pct_days_tau_gt_0_25_and_forward_negative": frame["tau"] > 0.25,
        "pct_days_position_gt_0_05_and_forward_negative": frame["position"] > 0.05,
        "pct_days_position_gt_0_10_and_forward_negative": frame["position"] > 0.10,
    }
    return [
        {
            "fee_rate": fee_rate,
            "strategy_name": strategy_name,
            "segment_name": segment_name,
            "horizon": horizon,
            "diagnostic": name,
            "value": float((condition & (fwd < 0.0)).mean()) if len(frame) else math.nan,
            "count": int((condition & (fwd < 0.0)).sum()) if len(frame) else 0,
        }
        for name, condition in conditions.items()
    ]


def conditional_forward_rows(
    frame: pd.DataFrame,
    fwd: pd.Series,
    *,
    fee_rate: float,
    strategy_name: str,
    segment_name: str,
    horizon: str,
) -> list[dict[str, Any]]:
    tau_bins = {
        "tau_le_0": frame["tau"] <= 0.0,
        "tau_0_to_0_1": (frame["tau"] > 0.0) & (frame["tau"] <= 0.1),
        "tau_0_1_to_0_25": (frame["tau"] > 0.1) & (frame["tau"] <= 0.25),
        "tau_0_25_to_0_5": (frame["tau"] > 0.25) & (frame["tau"] <= 0.5),
        "tau_gt_0_5": frame["tau"] > 0.5,
    }
    position_bins = {
        "position_eq_0": frame["position"] == 0.0,
        "position_0_to_0_05": (frame["position"] > 0.0) & (frame["position"] <= 0.05),
        "position_0_05_to_0_10": (frame["position"] > 0.05) & (frame["position"] <= 0.10),
        "position_0_10_to_0_20": (frame["position"] > 0.10) & (frame["position"] <= 0.20),
        "position_gt_0_20": frame["position"] > 0.20,
    }
    rows = []
    for prefix, bins in (("avg_forward_return_conditional_tau", tau_bins), ("avg_forward_return_conditional_position", position_bins)):
        for label, mask in bins.items():
            rows.append(
                {
                    "fee_rate": fee_rate,
                    "strategy_name": strategy_name,
                    "segment_name": segment_name,
                    "horizon": horizon,
                    "diagnostic": f"{prefix}:{label}",
                    "value": safe_mean(fwd[mask]),
                    "count": int(mask.sum()),
                }
            )
    return rows


def whipsaw_diagnostic(
    *,
    segment_frame: pd.DataFrame,
    fee_rate: float,
    strategy_name: str,
    segment_name: str,
) -> dict[str, Any]:
    changes = segment_frame["turnover"].astype(float)
    positive_changes = changes[changes > 0.0]
    return {
        "fee_rate": fee_rate,
        "strategy_name": strategy_name,
        "segment_name": segment_name,
        "number_of_position_changes": int((changes > 1e-6).sum()),
        "average_absolute_position_change": safe_mean(positive_changes),
        "median_absolute_position_change": safe_median(positive_changes),
        "pct_position_changes_below_0_005": pct_change_below(positive_changes, 0.005),
        "pct_position_changes_below_0_01": pct_change_below(positive_changes, 0.01),
        "pct_position_changes_below_0_02": pct_change_below(positive_changes, 0.02),
        "cumulative_fee_from_position_changes_below_0_005": fee_below(segment_frame, 0.005),
        "cumulative_fee_from_position_changes_below_0_01": fee_below(segment_frame, 0.01),
        "cumulative_fee_from_position_changes_below_0_02": fee_below(segment_frame, 0.02),
    }


def drawdown_lag_diagnostics(
    *,
    segment_frame: pd.DataFrame,
    fee_rate: float,
    strategy_name: str,
    segment_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for threshold in (0.05, 0.10, 0.15):
        in_episode = segment_frame["drawdown"].astype(float) > threshold
        starts = in_episode & ~in_episode.shift(fill_value=False)
        for start_idx in segment_frame.index[starts]:
            episode_indices = []
            for idx in segment_frame.loc[start_idx:].index:
                if not in_episode.loc[idx]:
                    break
                episode_indices.append(idx)
            episode = segment_frame.loc[episode_indices]
            start_position = float(episode["position"].iloc[0])
            rows.append(
                {
                    "fee_rate": fee_rate,
                    "strategy_name": strategy_name,
                    "segment_name": segment_name,
                    "drawdown_threshold": threshold,
                    "episode_start_date": str(episode["timestamp"].iloc[0].date()),
                    "episode_end_date": str(episode["timestamp"].iloc[-1].date()),
                    "max_drawdown_in_episode": float(episode["drawdown"].max()),
                    "average_tau_during_episode": float(episode["tau"].mean()),
                    "average_position_during_episode": float(episode["position"].mean()),
                    "average_base_exposure_during_episode": float(episode["base_exposure"].mean()),
                    "average_unsmoothed_target_during_episode": float(episode["unsmoothed_target"].mean()),
                    "average_market_risk_multiplier_during_episode": float(episode["market_risk_multiplier"].mean()),
                    "average_portfolio_risk_multiplier_during_episode": float(episode["portfolio_risk_multiplier"].mean()),
                    "days_until_position_halved_from_episode_start": days_until(episode, start_position / 2.0),
                    "days_until_position_below_0_05_from_episode_start": days_until(episode, 0.05),
                }
            )
    return rows


def segment_metric_values(frame: pd.DataFrame, *, starting_equity: float) -> dict[str, Any]:
    equity = frame["equity"].astype(float).to_numpy()
    equity_with_initial = np.concatenate([[starting_equity], equity])
    returns = equity_with_initial[1:] / equity_with_initial[:-1] - 1.0
    period_count = len(returns)
    std_return = float(np.std(returns))
    final_equity = float(equity_with_initial[-1])
    return {
        "total_return": final_equity / starting_equity - 1.0,
        "annualized_return": (final_equity / starting_equity) ** (PERIODS_PER_YEAR / period_count) - 1.0,
        "max_drawdown": float(frame["drawdown"].astype(float).max()),
        "sharpe_ratio": math.nan if std_return == 0.0 else float(np.mean(returns)) / std_return * math.sqrt(PERIODS_PER_YEAR),
        "total_turnover": float(frame["turnover"].astype(float).sum()),
        "average_turnover": float(frame["turnover"].astype(float).mean()),
        "average_exposure": float(frame["previous_position"].astype(float).mean()),
        "total_fee_cost": float(frame["transaction_cost"].astype(float).sum()),
        "trade_count": int((frame["turnover"].astype(float) > 1e-6).sum()),
    }


def starting_equity_for_segment(frame: pd.DataFrame, segment_frame: pd.DataFrame) -> float:
    first_index = segment_frame.index[0]
    previous_rows = frame.loc[:first_index].iloc[:-1]
    if previous_rows.empty:
        return float(frame.attrs.get("initial_equity", 1.0))
    return float(previous_rows["equity"].iloc[-1])


def compounded_segment_return(frame: pd.DataFrame, segment_frame: pd.DataFrame) -> float:
    start_equity = starting_equity_for_segment(frame, segment_frame)
    return float(segment_frame["equity"].iloc[-1]) / start_equity - 1.0


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


def pct_change_below(changes: pd.Series, threshold: float) -> float:
    if changes.empty:
        return math.nan
    return float((changes < threshold).mean())


def fee_below(frame: pd.DataFrame, threshold: float) -> float:
    turnover = frame["turnover"].astype(float)
    return float(frame.loc[(turnover > 0.0) & (turnover < threshold), "transaction_cost"].sum())


def safe_mean(series: pd.Series) -> float:
    return math.nan if series.empty else float(series.mean())


def safe_median(series: pd.Series) -> float:
    return math.nan if series.empty else float(series.median())


def days_until(frame: pd.DataFrame, position_threshold: float) -> int | float:
    for offset, value in enumerate(frame["position"].astype(float)):
        if value <= position_threshold:
            return offset
    return math.nan


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


def attribution_markdown(segment: pd.DataFrame) -> str:
    notes = (
        "# ETH 5y vs 3y Segment Attribution\n\n"
        "Diagnostic notes: strategy segment metrics are computed from the same continuous ETH 5y run "
        "for `continuous_state_segment_metrics`. Reset-based segment rows are also included and clearly "
        "labeled as `reset_state_segment_metrics`. Forward returns are used only for diagnostics and are "
        "not used by the strategy. Segment split uses the first return date of the ETH 3y daily window as "
        "the recent-window boundary.\n\n"
    )
    return notes + frame_to_markdown(segment)


if __name__ == "__main__":
    main()

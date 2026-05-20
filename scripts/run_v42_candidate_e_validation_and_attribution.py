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

from scripts.run_v42_candidate_d_validation_and_attribution import (  # noqa: E402
    PERIODS_PER_YEAR,
    RecordingController,
    RecordingStateEstimator,
    attribution_markdown,
    base_summary,
    frame_to_markdown,
    load_daily_close,
    make_v41_config,
    pnl_attribution,
    segment_metric_values,
    starting_equity_for_segment,
    true_buy_and_hold_frame,
)
from src.v4 import (  # noqa: E402
    BacktestConfig,
    BacktestEngine,
    V41DefaultConfig,
    evaluate_metrics,
    make_v42_candidate_a_config,
    make_v42_candidate_c_config,
    make_v42_candidate_d_config,
    make_v42_candidate_e_config,
    run_fixed_exposure_benchmark,
    run_zero_position_benchmark,
)


FEE_RATES = (0.0, 0.001, 0.002)
OUT_DIR = Path("reports") / "v42_candidate_e_validation"
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
    "trend_persistence_state_mean",
    "trend_persistence_state_median",
    "trend_persistence_state_p25",
    "trend_persistence_state_p75",
    "trend_persistence_state_max",
    "trend_persistence_gate_mean",
    "trend_persistence_gate_median",
    "trend_persistence_gate_p25",
    "trend_persistence_gate_p75",
    "trend_persistence_gate_max",
    "pct_trend_persistence_gate_eq_0",
    "pct_trend_persistence_gate_gt_0",
    "pct_trend_persistence_gate_gt_0_5",
    "pct_tau_gt_0_25_but_gate_eq_0",
    "base_exposure_C_mean",
    "base_exposure_E_mean",
    "exposure_reduction_from_gate_mean",
    "unsmoothed_target_mean",
    "final_position_mean",
]
CONTROL_STRATEGIES = ("v4.1_default", "v4.2_candidate_A", "v4.2_candidate_C", "v4.2_candidate_D", "v4.2_candidate_E")
ATTRIBUTION_STRATEGIES = ("v4.2_candidate_C", "v4.2_candidate_D", "v4.2_candidate_E", "true_buy_and_hold", "fixed_0_5_exposure")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    validation = run_validation()
    segment, false_positive, drawdown, gate = run_eth_attribution()

    validation_csv = OUT_DIR / "v42_candidate_e_validation_comparison.csv"
    validation_md = OUT_DIR / "v42_candidate_e_validation_comparison.md"
    segment_csv = OUT_DIR / "eth_candidate_e_segment_attribution.csv"
    segment_md = OUT_DIR / "eth_candidate_e_segment_attribution.md"
    false_csv = OUT_DIR / "eth_candidate_e_false_positive_trend_diagnostic.csv"
    drawdown_csv = OUT_DIR / "eth_candidate_e_drawdown_lag_diagnostic.csv"
    gate_csv = OUT_DIR / "eth_candidate_e_gate_diagnostic.csv"

    validation.to_csv(validation_csv, index=False)
    validation_md.write_text(frame_to_markdown(validation), encoding="utf-8")
    segment.to_csv(segment_csv, index=False)
    segment_md.write_text(attribution_markdown(segment), encoding="utf-8")
    false_positive.to_csv(false_csv, index=False)
    drawdown.to_csv(drawdown_csv, index=False)
    gate.to_csv(gate_csv, index=False)

    print(f"validation_csv: {validation_csv}")
    print(f"validation_markdown: {validation_md}")
    print(f"segment_attribution_csv: {segment_csv}")
    print(f"segment_attribution_markdown: {segment_md}")
    print(f"false_positive_csv: {false_csv}")
    print(f"drawdown_lag_csv: {drawdown_csv}")
    print(f"gate_diagnostic_csv: {gate_csv}")
    print(validation.to_string(index=False))


def run_validation() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for asset, window, path in DATASETS:
        prices = load_daily_close(path)
        for fee_rate in FEE_RATES:
            rows.extend(validate_dataset(asset=asset, window=window, prices=prices, fee_rate=fee_rate))
    return pd.DataFrame(rows, columns=VALIDATION_COLUMNS)


def validate_dataset(*, asset: str, window: str, prices: pd.Series, fee_rate: float) -> list[dict[str, Any]]:
    benchmark_config = BacktestConfig(fee_rate=fee_rate)
    return [
        summarize_control(asset, window, fee_rate, "v4.1_default", run_strategy_bundle(prices, make_v41_config(fee_rate=fee_rate), "v4.1_default")),
        summarize_control(asset, window, fee_rate, "v4.2_candidate_A", run_strategy_bundle(prices, make_v42_candidate_a_config(fee_rate=fee_rate), "v4.2_candidate_A")),
        summarize_control(asset, window, fee_rate, "v4.2_candidate_C", run_strategy_bundle(prices, make_v42_candidate_c_config(fee_rate=fee_rate), "v4.2_candidate_C")),
        summarize_control(asset, window, fee_rate, "v4.2_candidate_D", run_strategy_bundle(prices, make_v42_candidate_d_config(fee_rate=fee_rate), "v4.2_candidate_D")),
        summarize_control(asset, window, fee_rate, "v4.2_candidate_E", run_strategy_bundle(prices, make_v42_candidate_e_config(fee_rate=fee_rate), "v4.2_candidate_E")),
        summarize_true_buy_and_hold(asset, window, fee_rate, prices),
        summarize_benchmark(asset, window, fee_rate, "fixed_0_5_exposure", run_fixed_exposure_benchmark(prices, exposure=0.5, config=benchmark_config)),
        summarize_benchmark(asset, window, fee_rate, "zero_position", run_zero_position_benchmark(prices, config=benchmark_config)),
    ]


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
        "base_exposure_C",
        "base_exposure",
        "base_exposure_E",
        "trend_persistence_state",
        "trend_persistence_gate",
        "exposure_reduction_from_gate",
        "unsmoothed_target",
        "market_risk_multiplier",
        "portfolio_risk_multiplier",
        "deadband_skip",
    ):
        frame[column] = [trace[column] for trace in controller.trace_history]
    return {"frame": frame}


def summarize_control(asset: str, window: str, fee_rate: float, strategy_name: str, bundle: dict[str, Any]) -> dict[str, Any]:
    frame = bundle["frame"]
    metrics = evaluate_metrics(frame, periods_per_year=PERIODS_PER_YEAR)
    row = base_summary(asset, window, fee_rate, strategy_name, frame, metrics)
    row.update(state_percentages(frame))
    row.update(e_diagnostics(frame if strategy_name == "v4.2_candidate_E" else None))
    return row


def summarize_benchmark(asset: str, window: str, fee_rate: float, strategy_name: str, frame: pd.DataFrame) -> dict[str, Any]:
    metrics = evaluate_metrics(frame, periods_per_year=PERIODS_PER_YEAR)
    row = base_summary(asset, window, fee_rate, strategy_name, frame, metrics)
    row.update(empty_state_percentages())
    row.update(e_diagnostics(None))
    return row


def summarize_true_buy_and_hold(asset: str, window: str, fee_rate: float, prices: pd.Series) -> dict[str, Any]:
    frame = true_buy_and_hold_frame(prices, fee_rate)
    metrics = segment_metric_values(frame, starting_equity=1.0)
    row = base_summary(asset, window, fee_rate, "true_buy_and_hold", frame, metrics)
    row.update(empty_state_percentages())
    row.update(e_diagnostics(None))
    return row


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
    drawdown_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []

    for fee_rate in FEE_RATES:
        bundles = build_attribution_bundles(eth5, fee_rate)
        for strategy_name, bundle in bundles.items():
            full_frame = bundle["frame"]
            for segment_name, mask_fn in segments.items():
                segment_frame = full_frame.loc[mask_fn(full_frame)].copy()
                segment_rows.append(segment_metrics(full_frame, segment_frame, fee_rate, strategy_name, segment_name))
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
                if strategy_name in ("v4.2_candidate_C", "v4.2_candidate_D", "v4.2_candidate_E"):
                    false_rows.extend(false_positive_diagnostics(eth5, segment_frame, fee_rate, strategy_name, segment_name))
                    drawdown_rows.extend(drawdown_lag_diagnostics(segment_frame, fee_rate, strategy_name, segment_name))
                if strategy_name == "v4.2_candidate_E":
                    gate_rows.append(gate_segment_diagnostic(segment_frame, fee_rate, segment_name))
        reset_bundles = build_reset_segment_bundles(eth5, eth3, recent_start, fee_rate)
        for strategy_name in ATTRIBUTION_STRATEGIES:
            for segment_name, bundle in reset_bundles[strategy_name].items():
                frame = bundle["frame"]
                segment_rows.append(segment_metrics(frame, frame, fee_rate, strategy_name, segment_name, "reset_state_segment_metrics"))
                segment_rows.append(
                    pnl_attribution(
                        frame=frame,
                        segment_frame=frame,
                        fee_rate=fee_rate,
                        strategy_name=strategy_name,
                        segment_name=segment_name,
                        state_mode="reset_state_segment_metrics",
                    )
                )
    return pd.DataFrame(segment_rows), pd.DataFrame(false_rows), pd.DataFrame(drawdown_rows), pd.DataFrame(gate_rows)


def build_attribution_bundles(prices: pd.Series, fee_rate: float) -> dict[str, dict[str, Any]]:
    return {
        "v4.2_candidate_C": run_strategy_bundle(prices, make_v42_candidate_c_config(fee_rate=fee_rate), "v4.2_candidate_C"),
        "v4.2_candidate_D": run_strategy_bundle(prices, make_v42_candidate_d_config(fee_rate=fee_rate), "v4.2_candidate_D"),
        "v4.2_candidate_E": run_strategy_bundle(prices, make_v42_candidate_e_config(fee_rate=fee_rate), "v4.2_candidate_E"),
        "true_buy_and_hold": {"frame": true_buy_and_hold_frame(prices, fee_rate)},
        "fixed_0_5_exposure": {"frame": run_fixed_exposure_benchmark(prices, exposure=0.5, config=BacktestConfig(fee_rate=fee_rate))},
    }


def build_reset_segment_bundles(
    eth5: pd.Series,
    eth3: pd.Series,
    recent_start: pd.Timestamp,
    fee_rate: float,
) -> dict[str, dict[str, dict[str, Any]]]:
    early_prices = eth5.loc[eth5.index < recent_start]
    segment_prices = {
        "ETH_extra_early_2y": early_prices,
        "ETH_3y_recent": eth3,
        "ETH_5y_full": eth5,
    }
    bundles: dict[str, dict[str, dict[str, Any]]] = {}
    for strategy_name, config_factory in (
        ("v4.2_candidate_C", make_v42_candidate_c_config),
        ("v4.2_candidate_D", make_v42_candidate_d_config),
        ("v4.2_candidate_E", make_v42_candidate_e_config),
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


def segment_metrics(
    full_frame: pd.DataFrame,
    segment_frame: pd.DataFrame,
    fee_rate: float,
    strategy_name: str,
    segment_name: str,
    state_mode: str = "continuous_state_segment_metrics",
) -> dict[str, Any]:
    metrics = segment_metric_values(segment_frame, starting_equity=starting_equity_for_segment(full_frame, segment_frame))
    row = {
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
    row.update(segment_e_diagnostics(segment_frame if strategy_name == "v4.2_candidate_E" else None))
    return row


def false_positive_diagnostics(
    prices: pd.Series,
    segment_frame: pd.DataFrame,
    fee_rate: float,
    strategy_name: str,
    segment_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    indexed = segment_frame.set_index("timestamp").copy()
    for horizon in (1, 5, 20):
        fwd = (prices.shift(-horizon) / prices - 1.0).reindex(indexed.index)
        valid = indexed.assign(forward_return=fwd).dropna(subset=["forward_return"])
        rows.extend(false_positive_pct_rows(valid, fee_rate, strategy_name, segment_name, f"{horizon}d"))
        rows.extend(conditional_rows(valid, fee_rate, strategy_name, segment_name, f"{horizon}d"))
    return rows


def false_positive_pct_rows(frame: pd.DataFrame, fee_rate: float, strategy_name: str, segment_name: str, horizon: str) -> list[dict[str, Any]]:
    fwd = frame["forward_return"]
    conditions = {
        "pct_tau_gt_0_25_and_forward_negative": frame["tau"] > 0.25,
        "pct_position_gt_0_10_and_forward_negative": frame["position"] > 0.10,
        "pct_gate_gt_0_5_and_forward_negative": frame["trend_persistence_gate"] > 0.5,
        "pct_gate_eq_0_and_forward_negative": frame["trend_persistence_gate"] == 0.0,
    }
    return [
        {
            "fee_rate": fee_rate,
            "strategy_name": strategy_name,
            "segment_name": segment_name,
            "horizon": horizon,
            "diagnostic": name,
            "value": float((condition & (fwd < 0.0)).mean()),
            "count": int((condition & (fwd < 0.0)).sum()),
        }
        for name, condition in conditions.items()
    ]


def conditional_rows(frame: pd.DataFrame, fee_rate: float, strategy_name: str, segment_name: str, horizon: str) -> list[dict[str, Any]]:
    bins = {
        "tau:tau_le_0": frame["tau"] <= 0.0,
        "tau:tau_0_to_0_1": (frame["tau"] > 0.0) & (frame["tau"] <= 0.1),
        "tau:tau_0_1_to_0_25": (frame["tau"] > 0.1) & (frame["tau"] <= 0.25),
        "tau:tau_0_25_to_0_5": (frame["tau"] > 0.25) & (frame["tau"] <= 0.5),
        "tau:tau_gt_0_5": frame["tau"] > 0.5,
        "position:position_eq_0": frame["position"] == 0.0,
        "position:position_0_to_0_05": (frame["position"] > 0.0) & (frame["position"] <= 0.05),
        "position:position_0_05_to_0_10": (frame["position"] > 0.05) & (frame["position"] <= 0.10),
        "position:position_0_10_to_0_20": (frame["position"] > 0.10) & (frame["position"] <= 0.20),
        "position:position_gt_0_20": frame["position"] > 0.20,
        "gate:gate_eq_0": frame["trend_persistence_gate"] == 0.0,
        "gate:gate_0_to_0_25": (frame["trend_persistence_gate"] > 0.0) & (frame["trend_persistence_gate"] <= 0.25),
        "gate:gate_0_25_to_0_50": (frame["trend_persistence_gate"] > 0.25) & (frame["trend_persistence_gate"] <= 0.50),
        "gate:gate_0_50_to_0_75": (frame["trend_persistence_gate"] > 0.50) & (frame["trend_persistence_gate"] <= 0.75),
        "gate:gate_gt_0_75": frame["trend_persistence_gate"] > 0.75,
    }
    return [
        {
            "fee_rate": fee_rate,
            "strategy_name": strategy_name,
            "segment_name": segment_name,
            "horizon": horizon,
            "diagnostic": f"avg_forward_return_conditional_{label}",
            "value": safe_mean(frame.loc[mask, "forward_return"]),
            "count": int(mask.sum()),
        }
        for label, mask in bins.items()
    ]


def drawdown_lag_diagnostics(segment_frame: pd.DataFrame, fee_rate: float, strategy_name: str, segment_name: str) -> list[dict[str, Any]]:
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
                    "average_gate_during_episode": float(episode["trend_persistence_gate"].mean()),
                    "average_base_exposure_C_during_episode": float(episode["base_exposure_C"].mean()),
                    "average_base_exposure_E_during_episode": float(episode["base_exposure_E"].mean()),
                    "average_unsmoothed_target_during_episode": float(episode["unsmoothed_target"].mean()),
                }
            )
    return rows


def gate_segment_diagnostic(frame: pd.DataFrame, fee_rate: float, segment_name: str) -> dict[str, Any]:
    return {
        "fee_rate": fee_rate,
        "strategy_name": "v4.2_candidate_E",
        "segment_name": segment_name,
        **segment_e_diagnostics(frame),
    }


def e_diagnostics(frame: pd.DataFrame | None) -> dict[str, float]:
    if frame is None:
        return {key: math.nan for key in VALIDATION_COLUMNS[28:]}
    return {
        "trend_persistence_state_mean": stat(frame["trend_persistence_state"], "mean"),
        "trend_persistence_state_median": stat(frame["trend_persistence_state"], "median"),
        "trend_persistence_state_p25": stat(frame["trend_persistence_state"], "p25"),
        "trend_persistence_state_p75": stat(frame["trend_persistence_state"], "p75"),
        "trend_persistence_state_max": float(frame["trend_persistence_state"].max()),
        "trend_persistence_gate_mean": stat(frame["trend_persistence_gate"], "mean"),
        "trend_persistence_gate_median": stat(frame["trend_persistence_gate"], "median"),
        "trend_persistence_gate_p25": stat(frame["trend_persistence_gate"], "p25"),
        "trend_persistence_gate_p75": stat(frame["trend_persistence_gate"], "p75"),
        "trend_persistence_gate_max": float(frame["trend_persistence_gate"].max()),
        "pct_trend_persistence_gate_eq_0": float((frame["trend_persistence_gate"] == 0.0).mean()),
        "pct_trend_persistence_gate_gt_0": pct_gt(frame["trend_persistence_gate"], 0.0),
        "pct_trend_persistence_gate_gt_0_5": pct_gt(frame["trend_persistence_gate"], 0.5),
        "pct_tau_gt_0_25_but_gate_eq_0": float(((frame["tau"] > 0.25) & (frame["trend_persistence_gate"] == 0.0)).mean()),
        "base_exposure_C_mean": stat(frame["base_exposure_C"], "mean"),
        "base_exposure_E_mean": stat(frame["base_exposure_E"], "mean"),
        "exposure_reduction_from_gate_mean": stat(frame["exposure_reduction_from_gate"], "mean"),
        "unsmoothed_target_mean": stat(frame["unsmoothed_target"], "mean"),
        "final_position_mean": stat(frame["position"], "mean"),
    }


def segment_e_diagnostics(frame: pd.DataFrame | None) -> dict[str, float]:
    if frame is None:
        return {
            "segment_trend_persistence_gate_mean": math.nan,
            "segment_trend_persistence_gate_median": math.nan,
            "segment_pct_gate_eq_0": math.nan,
            "segment_pct_gate_gt_0": math.nan,
            "segment_pct_gate_gt_0_5": math.nan,
            "segment_base_exposure_C_mean": math.nan,
            "segment_base_exposure_E_mean": math.nan,
            "segment_exposure_reduction_from_gate_mean": math.nan,
        }
    return {
        "segment_trend_persistence_gate_mean": stat(frame["trend_persistence_gate"], "mean"),
        "segment_trend_persistence_gate_median": stat(frame["trend_persistence_gate"], "median"),
        "segment_pct_gate_eq_0": float((frame["trend_persistence_gate"] == 0.0).mean()),
        "segment_pct_gate_gt_0": pct_gt(frame["trend_persistence_gate"], 0.0),
        "segment_pct_gate_gt_0_5": pct_gt(frame["trend_persistence_gate"], 0.5),
        "segment_base_exposure_C_mean": stat(frame["base_exposure_C"], "mean"),
        "segment_base_exposure_E_mean": stat(frame["base_exposure_E"], "mean"),
        "segment_exposure_reduction_from_gate_mean": stat(frame["exposure_reduction_from_gate"], "mean"),
    }


def state_percentages(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "pct_tau_gt_0": pct_gt(frame["tau"], 0.0),
        "pct_tau_gt_0_1": pct_gt(frame["tau"], 0.1),
        "pct_tau_gt_0_25": pct_gt(frame["tau"], 0.25),
        "pct_tau_gt_0_5": pct_gt(frame["tau"], 0.5),
        "pct_nu_gt_0_5": pct_gt(frame["nu"], 0.5),
        "pct_rho_gt_0_5": pct_gt(frame["rho"], 0.5),
    }


def empty_state_percentages() -> dict[str, float]:
    return {key: math.nan for key in ("pct_tau_gt_0", "pct_tau_gt_0_1", "pct_tau_gt_0_25", "pct_tau_gt_0_5", "pct_nu_gt_0_5", "pct_rho_gt_0_5")}


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
    raise ValueError(name)


def safe_mean(series: pd.Series) -> float:
    return math.nan if series.empty else float(series.mean())


if __name__ == "__main__":
    main()

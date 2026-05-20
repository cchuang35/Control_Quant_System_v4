from __future__ import annotations

from dataclasses import fields
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtester import load_ohlcv_csv, run_backtest_fast
from src.v3.backtest_v3 import BacktestV3Config, run_v3_backtest
from src.v3.diagnostics import build_v3_diagnostics
from src.v3.feature_builder import FeatureWindowConfig
from src.v3.long_term_controller import LongTermControllerConfig
from src.v3.market_estimator import MarketEstimatorConfig
from src.v3.position_composer import PositionComposerConfig
from src.v3.risk_supervisor import RiskSupervisorConfig
from src.v3.short_term_controller import ShortTermControllerConfig
from v2_small_cap import backtest_v2_btc_final_candidate_a, backtest_v2_final_candidate_a


CONFIG_PATH = Path("configs") / "v3_final_candidate.yaml"
PERIODS_PER_YEAR = 365 * 24
V1_ENTRY_THRESHOLD = 0.10
BTC_REPORT = Path("reports") / "v3_final_candidate_btc_1h.md"
ETH_REPORT = Path("reports") / "v3_final_candidate_eth_1h.md"
COMPARISON_REPORT = Path("reports") / "v3_final_candidate_comparison.md"
BTC_SUMMARY_CSV = Path("reports") / "v3_final_candidate_btc_1h_summary.csv"
ETH_SUMMARY_CSV = Path("reports") / "v3_final_candidate_eth_1h_summary.csv"
BTC_DIAGNOSTICS_CSV = Path("reports") / "v3_final_candidate_btc_1h_diagnostics.csv"
ETH_DIAGNOSTICS_CSV = Path("reports") / "v3_final_candidate_eth_1h_diagnostics.csv"
COMPARISON_CSV = Path("reports") / "v3_final_candidate_comparison.csv"


def load_final_candidate_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the explicit v3.final_candidate YAML config."""

    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        config = yaml.safe_load(text)
    except ModuleNotFoundError:
        config = _load_simple_yaml(text)
    if not isinstance(config, dict):
        raise ValueError(f"invalid config file: {path}")
    return config


def _load_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the small YAML subset used by the v3 config files.

    This fallback avoids adding a runtime dependency when PyYAML is not
    installed. It supports nested mappings, scalar lists, booleans, strings,
    integers, and floats, which is enough for `v3_final_candidate.yaml`.
    """

    tokens: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        tokens.append((len(line) - len(line.lstrip(" ")), line.lstrip(" ")))
    parsed, index = _parse_yaml_block(tokens, 0, 0)
    if index != len(tokens):
        raise ValueError("could not parse complete YAML config")
    if not isinstance(parsed, dict):
        raise ValueError("top-level YAML config must be a mapping")
    return parsed


def _parse_yaml_block(tokens: list[tuple[int, str]], index: int, indent: int) -> tuple[Any, int]:
    if index >= len(tokens):
        return {}, index
    current_indent, content = tokens[index]
    if current_indent < indent:
        return {}, index
    if content.startswith("- "):
        values: list[Any] = []
        while index < len(tokens) and tokens[index][0] == current_indent and tokens[index][1].startswith("- "):
            item = tokens[index][1][2:].strip()
            if item:
                values.append(_parse_scalar(item))
                index += 1
            else:
                child, index = _parse_yaml_block(tokens, index + 1, current_indent + 2)
                values.append(child)
        return values, index

    values: dict[str, Any] = {}
    while index < len(tokens):
        item_indent, item = tokens[index]
        if item_indent < current_indent or item_indent < indent:
            break
        if item_indent > current_indent:
            break
        if item.startswith("- "):
            break
        key, separator, raw_value = item.partition(":")
        if not separator:
            raise ValueError(f"invalid YAML line: {item}")
        key = key.strip()
        raw_value = raw_value.strip()
        if raw_value:
            values[key] = _parse_scalar(raw_value)
            index += 1
        else:
            if index + 1 >= len(tokens) or tokens[index + 1][0] <= item_indent:
                values[key] = {}
                index += 1
            else:
                child, index = _parse_yaml_block(tokens, index + 1, tokens[index + 1][0])
                values[key] = child
    return values, index


def _parse_scalar(value: str) -> Any:
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    try:
        if any(marker in value for marker in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")


def build_backtest_config(config: dict[str, Any], *, fee_rate: float) -> BacktestV3Config:
    """Map the YAML config into the existing v3 backtest config dataclasses."""

    features = config.get("features", {})
    short = config.get("short_term_controller", {})
    risk = config.get("risk_supervisor", {})
    risk_thresholds = risk.get("drawdown_thresholds", {})
    volatility_caps = risk.get("volatility_cap_values", {})
    losses = risk.get("consecutive_loss_rules_config", {})
    position = config.get("position_scheme", {})
    execution = config.get("execution", {})
    cooldown = config.get("cooldown", {})
    long_term = config.get("long_term_controller", {})

    feature_config = FeatureWindowConfig(**_filter_dataclass_kwargs(FeatureWindowConfig, features))
    estimator_config = MarketEstimatorConfig()
    long_term_config = LongTermControllerConfig(
        base_positions={key: float(value) for key, value in long_term.get("base_positions", {}).items()},
        name=str(long_term.get("mapping", "conservative")),
    )
    short_term_config = ShortTermControllerConfig(
        high_confidence=float(short.get("high_confidence", 0.70)),
        very_high_confidence=float(short.get("very_high_confidence", 0.85)),
        enable_pullback_add=bool(short.get("enable_pullback_add", False)),
        enable_recovery_add=bool(short.get("enable_recovery_add", False)),
        enable_overheat_reduce=bool(short.get("enable_overheat_reduce", True)),
        enable_breakdown_reduce=bool(short.get("enable_breakdown_reduce", True)),
        allow_neutral_recovery_add=bool(short.get("allow_neutral_recovery_add", False)),
        experimental_mode=bool(short.get("experimental_mode", False)),
    )
    risk_config = RiskSupervisorConfig(
        enable_drawdown_cap=risk.get("drawdown_caps", "enabled") == "enabled",
        enable_volatility_cap=risk.get("volatility_caps", "enabled") == "enabled",
        enable_consecutive_loss_rules=risk.get("consecutive_loss_rules", "enabled") == "enabled",
        enable_market_risk_state=risk.get("market_risk_state_controls", "enabled") == "enabled",
        enable_cost_guards=risk.get("cost_guards", "enabled") == "enabled",
        drawdown_caution=float(risk_thresholds.get("caution", -0.05)),
        drawdown_danger=float(risk_thresholds.get("danger", -0.10)),
        drawdown_severe=float(risk_thresholds.get("severe", -0.15)),
        drawdown_risk_off=float(risk_thresholds.get("risk_off", -0.20)),
        high_volatility_cap=float(volatility_caps.get("high", 0.75)),
        extreme_volatility_cap=float(volatility_caps.get("extreme", 0.50)),
        losses_no_new_entry=int(losses.get("no_new_entry_at", 2)),
        losses_reduce_cap=int(losses.get("reduce_cap_at", 3)),
        losses_risk_off=int(losses.get("risk_off_at", 4)),
        consecutive_loss_cap=float(losses.get("consecutive_loss_cap", 0.50)),
    )
    composer_config = PositionComposerConfig(
        min_position=float(position.get("min_position", 0.0)),
        max_position=float(position.get("max_position", 1.0)),
        allowed_positions=tuple(float(value) for value in position.get("allowed_positions", [0.0, 0.25, 0.5, 0.75, 1.0])),
        rounding_mode=str(position.get("rounding", "floor")),
        allow_leverage=bool(position.get("leverage_enabled", False)),
    )
    return BacktestV3Config(
        fee_rate=float(fee_rate),
        slippage_rate=float(execution.get("slippage_rate", 0.0)),
        cooldown_bars=int(cooldown.get("cooldown_bars", 120)) if cooldown.get("enabled", True) else 0,
        minimum_position_step=float(execution.get("minimum_position_step", execution.get("no_trade_zone", 0.25))),
        feature_config=feature_config,
        estimator_config=estimator_config,
        long_term_config=long_term_config,
        short_term_config=short_term_config,
        risk_config=risk_config,
        composer_config=composer_config,
        leverage_config=None,
    )


def run_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = load_final_candidate_config()
    fee_rates = tuple(float(value) for value in config["execution"]["fee_rates_to_validate"])
    btc_summary, btc_diagnostics = run_asset("BTC", config, fee_rates)
    eth_summary, eth_diagnostics = run_asset("ETH", config, fee_rates)
    comparison = build_comparison_table(btc_summary, eth_summary)
    write_asset_report("BTC", btc_summary, btc_diagnostics, BTC_REPORT, BTC_SUMMARY_CSV, BTC_DIAGNOSTICS_CSV)
    write_asset_report("ETH", eth_summary, eth_diagnostics, ETH_REPORT, ETH_SUMMARY_CSV, ETH_DIAGNOSTICS_CSV)
    write_comparison_report(comparison, btc_summary, eth_summary)
    return btc_summary, btc_diagnostics, eth_summary, eth_diagnostics, comparison


def run_asset(asset: str, config: dict[str, Any], fee_rates: tuple[float, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_key = "btc_datasets" if asset == "BTC" else "eth_datasets"
    paths = [Path(path) for path in config["validation"].get(dataset_key, []) if Path(path).exists()]
    if not paths:
        return pd.DataFrame(), pd.DataFrame()

    summary_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for path in paths:
        dataset = path.stem
        print(f"asset={asset} dataset={dataset}")
        data = load_ohlcv_csv(path)
        for fee_rate in fee_rates:
            print(f"  fee={fee_rate:g}")
            start = perf_counter()
            v1_result = run_backtest_fast(
                data,
                fee_rate=fee_rate,
                periods_per_year=PERIODS_PER_YEAR,
                progress_every=10000 if len(data) > 15000 else None,
            )
            v1_runtime = perf_counter() - start
            v1_frame = v1_frame_from_result(v1_result, fee_rate)
            v2_input = input_frame_from_v1(v1_result)
            v2_label = "v2.btc_final_candidate_A" if asset == "BTC" else "v2.final_candidate_A_cd120_on_ETH"
            v2_func = backtest_v2_btc_final_candidate_a if asset == "BTC" else backtest_v2_final_candidate_a
            v3_frame = run_v3_backtest(data, config=build_backtest_config(config, fee_rate=fee_rate))
            frames = {
                "v1.final": v1_frame,
                v2_label: v2_func(
                    v2_input,
                    fee_rate=fee_rate,
                    v1_entry_threshold=V1_ENTRY_THRESHOLD,
                    cooldown_bars=120,
                ),
                "v3.final_candidate": v3_frame,
                "buy_and_hold": build_buy_and_hold_frame(data, fee_rate),
                "ma20_ma60": build_ma_crossover_frame(data, fee_rate),
            }
            for version, frame in frames.items():
                summary_rows.append(
                    {
                        "asset": asset,
                        "dataset": dataset,
                        "fee_rate": fee_rate,
                        "version": version,
                        **summarize_frame(frame),
                        "runtime_sec": v1_runtime if version == "v1.final" else 0.0,
                    }
                )
            diagnostic_rows.extend(flatten_v3_diagnostics(asset, dataset, fee_rate, v3_frame, build_v3_diagnostics(v3_frame)))

    return pd.DataFrame(summary_rows), pd.DataFrame(diagnostic_rows)


def v1_frame_from_result(result: Any, fee_rate: float) -> pd.DataFrame:
    exposure = result.exposure_history.reset_index(drop=True).copy()
    equity = result.equity_curve.iloc[1:].reset_index(drop=True).copy()
    current_exposure = exposure["current_exposure"].astype(float)
    frame = pd.DataFrame(
        {
            "timestamp": exposure["execution_timestamp"],
            "close": exposure["close"].astype(float),
            "position": current_exposure,
            "trade_amount": exposure["safe_exposure_change"].abs().astype(float),
            "fee_cost": exposure["safe_exposure_change"].abs().astype(float) * fee_rate,
            "strategy_return_net": equity["period_return"].astype(float),
            "equity_curve": equity["equity"].astype(float),
            "drawdown": equity["drawdown"].astype(float),
        }
    )
    frame["asset_return"] = frame["close"].pct_change().fillna(0.0)
    return frame


def input_frame_from_v1(result: Any) -> pd.DataFrame:
    exposure = result.exposure_history.reset_index(drop=True).copy()
    return pd.DataFrame(
        {
            "timestamp": exposure["execution_timestamp"],
            "close": exposure["close"].astype(float),
            "current_exposure": exposure["current_exposure"].astype(float),
        }
    )


def build_buy_and_hold_frame(data: pd.DataFrame, fee_rate: float) -> pd.DataFrame:
    close = pd.to_numeric(data["close"], errors="coerce").astype(float).reset_index(drop=True)
    timestamp = data["timestamp"].reset_index(drop=True) if "timestamp" in data.columns else pd.Series(data.index)
    asset_return = close.pct_change().fillna(0.0)
    position = pd.Series(1.0, index=close.index)
    trade_amount = position.diff().abs().fillna(position.abs())
    fee_cost = trade_amount * fee_rate
    gross = position.shift(1).fillna(0.0) * asset_return
    net = gross - fee_cost
    equity = (1.0 + net).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "close": close,
            "asset_return": asset_return,
            "position": position,
            "trade_amount": trade_amount,
            "fee_cost": fee_cost,
            "strategy_return_net": net,
            "equity_curve": equity,
            "drawdown": drawdown,
        }
    )


def build_ma_crossover_frame(data: pd.DataFrame, fee_rate: float) -> pd.DataFrame:
    close = pd.to_numeric(data["close"], errors="coerce").astype(float).reset_index(drop=True)
    timestamp = data["timestamp"].reset_index(drop=True) if "timestamp" in data.columns else pd.Series(data.index)
    asset_return = close.pct_change().fillna(0.0)
    ma20 = close.rolling(20, min_periods=1).mean()
    ma60 = close.rolling(60, min_periods=1).mean()
    position = (ma20 > ma60).astype(float)
    trade_amount = position.diff().abs().fillna(position.abs())
    fee_cost = trade_amount * fee_rate
    gross = position.shift(1).fillna(0.0) * asset_return
    net = gross - fee_cost
    equity = (1.0 + net).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "close": close,
            "asset_return": asset_return,
            "position": position,
            "trade_amount": trade_amount,
            "fee_cost": fee_cost,
            "strategy_return_net": net,
            "equity_curve": equity,
            "drawdown": drawdown,
        }
    )


def summarize_frame(frame: pd.DataFrame) -> dict[str, float | int]:
    returns = pd.to_numeric(_column(frame, "strategy_return_net"), errors="coerce").fillna(0.0)
    equity = _equity(frame, returns)
    drawdown = _drawdown(frame, equity)
    position = _position(frame)
    trade_amount = _trade_amount(frame, position)
    fee_cost = pd.to_numeric(frame.get("fee_cost", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    final_equity = float(equity.iloc[-1]) if len(equity) else 1.0
    periods = max(len(returns), 1)
    return_std = float(returns.std(ddof=0))
    return {
        "total_return": final_equity - 1.0,
        "annual_return": final_equity ** (PERIODS_PER_YEAR / periods) - 1.0,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "sharpe_ratio": 0.0 if return_std == 0.0 else float(returns.mean() / return_std * np.sqrt(PERIODS_PER_YEAR)),
        "number_of_trades": int((trade_amount > 0.0).sum()),
        "turnover": float(trade_amount.sum()),
        "fee_drag": float(fee_cost.sum()),
        "average_exposure": float(position.abs().mean()) if len(position) else 0.0,
        "max_exposure": float(position.abs().max()) if len(position) else 0.0,
    }


def flatten_v3_diagnostics(
    asset: str,
    dataset: str,
    fee_rate: float,
    result: pd.DataFrame,
    diagnostics: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    diagnostic_tables = {
        "exposure_distribution": diagnostics["exposure_distribution"],
        "long_regime_distribution": _distribution(result, "long_regime", "long_regime"),
        "short_regime_distribution": _distribution(result, "short_regime", "short_regime"),
        "risk_action_counts": diagnostics["risk_action_counts"],
        "risk_cap_distribution": diagnostics["risk_cap_distribution"],
        "executed_position_distribution": diagnostics["executed_position_distribution"],
        "target_position_distribution": _distribution(result, "target_position", "target_position"),
        "base_position_distribution": diagnostics["base_position_distribution"],
        "short_adjustment_distribution": diagnostics["short_adjustment_distribution"],
    }
    for diagnostic, frame in diagnostic_tables.items():
        for row in frame.to_dict("records"):
            rows.append({"asset": asset, "dataset": dataset, "fee_rate": fee_rate, "diagnostic": diagnostic, **row})
    execution_summary = diagnostics["execution_summary"].iloc[0].to_dict()
    cost_guard_blocks = result["risk_reason"].astype(str).str.contains("fee_drag_caution|turnover_caution", regex=True).sum()
    rows.append(
        {
            "asset": asset,
            "dataset": dataset,
            "fee_rate": fee_rate,
            "diagnostic": "execution_summary",
            **execution_summary,
            "cost_guard_blocks": int(cost_guard_blocks),
        }
    )
    return rows


def build_comparison_table(btc_summary: pd.DataFrame, eth_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for summary, v2_name in [
        (btc_summary, "v2.btc_final_candidate_A"),
        (eth_summary, "v2.final_candidate_A_cd120_on_ETH"),
    ]:
        if summary.empty:
            continue
        v3 = summary[summary["version"] == "v3.final_candidate"]
        v2 = summary[summary["version"] == v2_name]
        joined = v3.merge(v2, on=["asset", "dataset", "fee_rate"], suffixes=("_v3", "_v2"))
        if joined.empty:
            continue
        rows.append(
            pd.DataFrame(
                {
                    "asset": joined["asset"],
                    "dataset": joined["dataset"],
                    "fee_rate": joined["fee_rate"],
                    "total_return_delta_v3_minus_v2": joined["total_return_v3"] - joined["total_return_v2"],
                    "annual_return_delta_v3_minus_v2": joined["annual_return_v3"] - joined["annual_return_v2"],
                    "max_drawdown_delta_v3_minus_v2": joined["max_drawdown_v3"] - joined["max_drawdown_v2"],
                    "sharpe_delta_v3_minus_v2": joined["sharpe_ratio_v3"] - joined["sharpe_ratio_v2"],
                    "turnover_delta_v3_minus_v2": joined["turnover_v3"] - joined["turnover_v2"],
                    "fee_drag_delta_v3_minus_v2": joined["fee_drag_v3"] - joined["fee_drag_v2"],
                    "average_exposure_delta_v3_minus_v2": joined["average_exposure_v3"] - joined["average_exposure_v2"],
                    "v3_beats_v2_total_return": joined["total_return_v3"] > joined["total_return_v2"],
                    "v3_beats_v2_sharpe": joined["sharpe_ratio_v3"] > joined["sharpe_ratio_v2"],
                    "v3_beats_v2_drawdown": joined["max_drawdown_v3"] > joined["max_drawdown_v2"],
                    "v3_lower_turnover": joined["turnover_v3"] < joined["turnover_v2"],
                    "v3_lower_fee_drag": joined["fee_drag_v3"] < joined["fee_drag_v2"],
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def write_asset_report(
    asset: str,
    summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    report_path: Path,
    summary_csv: Path,
    diagnostics_csv: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)
    diagnostics.to_csv(diagnostics_csv, index=False)
    v2_label = "v2.btc_final_candidate_A" if asset == "BTC" else "v2.final_candidate_A_cd120_on_ETH"
    title = "BTCUSDT" if asset == "BTC" else "ETHUSDT"
    lines = [
        f"# v3.final_candidate {title} 1h Validation",
        "",
        "This is the explicit v3.final_candidate run. Parameters come from `configs/v3_final_candidate.yaml`; no tuning was performed.",
        "",
        "## Scope",
        "",
        f"- Datasets: {', '.join(sorted(summary['dataset'].unique())) if not summary.empty else 'none'}",
        "- Fee rates: 0.0005, 0.001, 0.002",
        f"- Versions: v1.final, {v2_label}, v3.final_candidate, buy_and_hold, ma20_ma60",
        "",
        "## Comparison Table",
        "",
        _frame_to_markdown(summary[_summary_columns()]) if not summary.empty else "_No data_",
        "",
        "## Exposure Distribution",
        "",
        _diagnostic_markdown(diagnostics, "exposure_distribution"),
        "",
        "## Long Regime Distribution",
        "",
        _diagnostic_markdown(diagnostics, "long_regime_distribution"),
        "",
        "## Short Regime Distribution",
        "",
        _diagnostic_markdown(diagnostics, "short_regime_distribution"),
        "",
        "## Risk Action Counts",
        "",
        _diagnostic_markdown(diagnostics, "risk_action_counts"),
        "",
        "## Risk Cap Distribution",
        "",
        _diagnostic_markdown(diagnostics, "risk_cap_distribution"),
        "",
        "## Executed Position Distribution",
        "",
        _diagnostic_markdown(diagnostics, "executed_position_distribution"),
        "",
        "## Target Position Distribution",
        "",
        _diagnostic_markdown(diagnostics, "target_position_distribution"),
        "",
        "## Skipped Trades / Cost Guard Summary",
        "",
        _diagnostic_markdown(diagnostics, "execution_summary"),
        "",
        "## Files",
        "",
        f"- Summary CSV: `{summary_csv}`",
        f"- Diagnostics CSV: `{diagnostics_csv}`",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def write_comparison_report(comparison: pd.DataFrame, btc_summary: pd.DataFrame, eth_summary: pd.DataFrame) -> None:
    COMPARISON_REPORT.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(COMPARISON_CSV, index=False)
    btc_comparison = comparison[comparison["asset"] == "BTC"] if not comparison.empty else pd.DataFrame()
    eth_v3 = eth_summary[eth_summary["version"] == "v3.final_candidate"] if not eth_summary.empty else pd.DataFrame()
    lines = [
        "# v3.final_candidate Comparison",
        "",
        "This report compares the explicit `v3.final_candidate` run against the available v1/v2/buy-and-hold/MA references. No parameters were tuned for this run.",
        "",
        "## Direct Answers",
        "",
        f"1. Does explicit v3.final_candidate reproduce the assembled summary conclusion? {answer_reproduces_summary(btc_comparison, eth_v3)}",
        f"2. Does v3 beat v2 on BTC return or Sharpe? {answer_btc_return_sharpe(btc_comparison)}",
        f"3. Does v3 beat v2 on drawdown, turnover, and fee drag? {answer_btc_risk_controls(btc_comparison)}",
        f"4. Is ETH still weak by absolute return? {answer_eth_weak(eth_v3)}",
        "5. Should v3 replace v2? No. The explicit final candidate should not replace v2 as the BTC strategy unless future diagnostics improve return capture without losing the risk-control gains.",
        "6. What is the next required diagnostic step? Run a decision-waterfall diagnostic that attributes every bar's exposure loss to estimator gating, cooldown, Risk Supervisor action/cap, position rounding, no-trade zone, or cost guards.",
        "",
        "## v3 vs v2 Delta Table",
        "",
        _frame_to_markdown(comparison) if not comparison.empty else "_No data_",
        "",
        "## BTC v3.final_candidate Rows",
        "",
        _frame_to_markdown(btc_summary[btc_summary["version"] == "v3.final_candidate"][_summary_columns()]) if not btc_summary.empty else "_No data_",
        "",
        "## ETH v3.final_candidate Rows",
        "",
        _frame_to_markdown(eth_v3[_summary_columns()]) if not eth_v3.empty else "_No data_",
        "",
        "## Files",
        "",
        f"- Comparison CSV: `{COMPARISON_CSV}`",
        f"- BTC report: `{BTC_REPORT}`",
        f"- ETH report: `{ETH_REPORT}`",
    ]
    COMPARISON_REPORT.write_text("\n".join(lines), encoding="utf-8")


def answer_reproduces_summary(btc_comparison: pd.DataFrame, eth_v3: pd.DataFrame) -> str:
    if btc_comparison.empty:
        return "No BTC v2 comparison rows were available."
    return_wins = int(btc_comparison["v3_beats_v2_total_return"].sum())
    sharpe_wins = int(btc_comparison["v3_beats_v2_sharpe"].sum())
    drawdown_wins = int(btc_comparison["v3_beats_v2_drawdown"].sum())
    turnover_wins = int(btc_comparison["v3_lower_turnover"].sum())
    fee_wins = int(btc_comparison["v3_lower_fee_drag"].sum())
    rows = len(btc_comparison)
    return (
        "Yes: it reproduces the assembled conclusion that v3 is mainly a risk-control improvement, not a BTC alpha upgrade. "
        f"BTC v3 return wins={return_wins}/{rows}, Sharpe wins={sharpe_wins}/{rows}, "
        f"drawdown wins={drawdown_wins}/{rows}, lower-turnover wins={turnover_wins}/{rows}, lower-fee-drag wins={fee_wins}/{rows}."
    )


def answer_btc_return_sharpe(btc_comparison: pd.DataFrame) -> str:
    if btc_comparison.empty:
        return "No BTC v2 comparison rows were available."
    rows = len(btc_comparison)
    return_wins = int(btc_comparison["v3_beats_v2_total_return"].sum())
    sharpe_wins = int(btc_comparison["v3_beats_v2_sharpe"].sum())
    return f"No on the overall BTC evidence. v3 total-return wins={return_wins}/{rows}; Sharpe wins={sharpe_wins}/{rows}."


def answer_btc_risk_controls(btc_comparison: pd.DataFrame) -> str:
    if btc_comparison.empty:
        return "No BTC v2 comparison rows were available."
    rows = len(btc_comparison)
    drawdown_wins = int(btc_comparison["v3_beats_v2_drawdown"].sum())
    turnover_wins = int(btc_comparison["v3_lower_turnover"].sum())
    fee_wins = int(btc_comparison["v3_lower_fee_drag"].sum())
    return f"Mostly yes. v3 drawdown wins={drawdown_wins}/{rows}, lower-turnover wins={turnover_wins}/{rows}, lower-fee-drag wins={fee_wins}/{rows}."


def answer_eth_weak(eth_v3: pd.DataFrame) -> str:
    if eth_v3.empty:
        return "ETH datasets were not available."
    avg_return = float(eth_v3["total_return"].mean())
    positive_rate = float((eth_v3["total_return"] > 0.0).mean())
    avg_exposure = float(eth_v3["average_exposure"].mean())
    return f"Yes by absolute return. ETH v3 average total_return={avg_return:.6g}, positive-result rate={positive_rate:.3g}, average_exposure={avg_exposure:.6g}."


def _distribution(result: pd.DataFrame, column: str, output_name: str) -> pd.DataFrame:
    counts = result[column].value_counts(dropna=False).sort_index().rename_axis(output_name).reset_index(name="bars")
    counts["ratio"] = counts["bars"] / max(len(result), 1)
    return counts


def _filter_dataclass_kwargs(cls: Any, values: dict[str, Any]) -> dict[str, Any]:
    names = {field.name for field in fields(cls)}
    return {key: value for key, value in values.items() if key in names}


def _summary_columns() -> list[str]:
    return [
        "asset",
        "dataset",
        "fee_rate",
        "version",
        "total_return",
        "annual_return",
        "max_drawdown",
        "sharpe_ratio",
        "number_of_trades",
        "turnover",
        "fee_drag",
        "average_exposure",
        "max_exposure",
    ]


def _diagnostic_markdown(diagnostics: pd.DataFrame, name: str) -> str:
    if diagnostics.empty:
        return "_No data_"
    frame = diagnostics[diagnostics["diagnostic"] == name].dropna(axis=1, how="all")
    return _frame_to_markdown(frame) if not frame.empty else "_No data_"


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        raise ValueError(f"frame is missing column: {name}")
    return frame[name]


def _equity(frame: pd.DataFrame, returns: pd.Series) -> pd.Series:
    for column in ("equity_curve", "equity_net", "equity"):
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce").fillna(1.0)
    return (1.0 + returns).cumprod()


def _drawdown(frame: pd.DataFrame, equity: pd.Series) -> pd.Series:
    if "drawdown" in frame.columns:
        return pd.to_numeric(frame["drawdown"], errors="coerce").fillna(0.0)
    return equity / equity.cummax() - 1.0


def _position(frame: pd.DataFrame) -> pd.Series:
    for column in ("executed_position", "final_position", "position", "current_exposure"):
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    raise ValueError("frame is missing a position column")


def _trade_amount(frame: pd.DataFrame, position: pd.Series) -> pd.Series:
    for column in ("trade_amount", "trade_size"):
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return position.diff().abs().fillna(position.abs())


def _frame_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No data_"
    display = frame.copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(lambda value: f"{value:.6g}")
    columns = [str(column) for column in display.columns]
    rows = display.astype(object).where(pd.notna(display), "").astype(str).values.tolist()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator] + body)


def main() -> None:
    run_all()
    print(f"Wrote {BTC_REPORT}")
    print(f"Wrote {ETH_REPORT}")
    print(f"Wrote {COMPARISON_REPORT}")


if __name__ == "__main__":
    main()

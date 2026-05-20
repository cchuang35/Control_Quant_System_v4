# Control Quant System v2

Research code for validating a v2 wrapper around the original five-layer `v1.final` control-style trading system.

The current v2 work keeps the v1 core strategy intact and adds outer validation layers for BTCUSDT 1h:

- regime detection
- regime trade gates
- weak-bull risk controls
- fee-aware backtests
- validation diagnostics
- out-of-sample data download and robustness reports

This repository is still research/validation code. `v2.btc_final_candidate_A` is not promoted to a general cross-asset final strategy.

## Current Versions

### `v1.final`

Conservative baseline strategy.

Used as benchmark for all later versions.

### `v2.btc_final_candidate_A`

BTCUSDT 1h specific candidate.

Based on sideways-hold-only regime gate plus weak_bull losing-trade cooldown.

Validated on BTCUSDT 1h datasets up to 5y relative to v2.4.

Not validated for ETHUSDT.

Not a general cross-asset final strategy.

## Strategy Summary

`v2.btc_final_candidate_A` uses the existing v1 output as its input signal and applies an outer binary-position wrapper.

Base regime gate:

```text
strong_bull:
    allow_entry = true
    allow_hold = true

weak_bull:
    allow_entry = true by default
    allow_hold = true
    but weak_bull new entries are blocked during weak_bull_loss_cooldown

sideways:
    allow_entry = false
    allow_hold = true

bear:
    allow_entry = false
    allow_hold = false
```

Weak-bull losing-trade cooldown:

```text
If a completed trade has:
    entry_regime == weak_bull
    and net_trade_return < 0
then block new weak_bull entries for cooldown_bars.
```

Defaults:

```text
cooldown_bars = 120
observed robust range = 120 / 144 / 168 bars
interpretation = roughly 5-7 days on 1h bars
```

During cooldown:

- only weak-bull new entries are blocked
- strong-bull entries remain allowed
- existing positions are not force-exited
- sideways and bear behavior is unchanged

Backtest return rule:

```text
strategy_return_net[t] =
    position[t-1] * asset_return[t]
    - abs(position[t] - position[t-1]) * fee_rate
```

The signal at bar `t` determines the position for bar `t+1`, avoiding look-ahead bias.

## Project Layout

```text
src/
  layer1_market_model.py
  layer2_state_estimator.py
  layer3_strategy_controller.py
  layer4_risk_filter.py
  layer5_adaptive_supervisor.py
  backtest.py

backtester.py
  v1.final sequential backtester and cached fast runner

v2_small_cap.py
  v2 wrapper implementations, diagnostics, and frozen BTC candidate

v2_eth_exploration.py
  placeholder for future ETH-specific research

scripts/
  download_oos_datasets.py
  run_v211_oos_validation.py

docs/
  v2_btc_final_candidate_A.md

tests/
  test_layers.py
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Tests

```bash
pytest
```

Current expected result:

```text
41 passed
```

## Data

Backtests expect OHLCV CSV files under `data/` with this schema:

```text
timestamp,open,high,low,close,volume
```

Large downloaded datasets are intentionally not committed. The `.gitignore` excludes:

```text
data/*.csv
reports/
backtest_outputs/
```

Keep `data/.gitkeep` so the directory exists in fresh clones.

## Download Validation Data

Download the BTC/ETH 1h datasets used for expanded validation:

```bash
python scripts/download_oos_datasets.py --overwrite
```

This writes:

```text
data/btcusdt_1h_2y.csv
data/btcusdt_1h_3y.csv
data/btcusdt_1h_5y.csv
data/ethusdt_1h_365d.csv
data/ethusdt_1h_2y.csv
data/ethusdt_1h_3y.csv
data/ethusdt_1h_5y.csv
```

It also writes a data quality report:

```text
reports/data_quality/oos_data_quality.csv
```

Quality checks include:

- duplicate timestamps
- timestamp sort order
- expected 1h spacing
- missing bar count
- invalid or non-positive close prices

## Run v2.11 Validation

After data is available:

```bash
python scripts/run_v211_oos_validation.py
```

The script compares:

- `v1.final`
- `v2.4.pause_4_24_exit_7`
- `v2.final_candidate_A_cd120`
- `v2.final_candidate_A_cd144`
- `v2.final_candidate_A_cd168`

Fee rates:

```text
0.0005
0.0010
0.0020
```

Reports are written to:

```text
reports/v211_oos_validation/
```

Key report files:

```text
expanded_oos_summary.csv
rolling_win_rates.csv
period_contribution.csv
cooldown_similarity.csv
available_datasets.csv
missing_requested_datasets.csv
```

## Run v1 Backtest Directly

Download or prepare a CSV:

```bash
python download_data.py --symbol BTCUSDT --interval 1h --output data/btcusdt_1h.csv
```

Run v1:

```bash
python backtester.py --csv data/btcusdt_1h.csv --fee-rate 0.0005 --output-dir backtest_outputs/btc_v1
```

The Layer 1-4 backtester avoids look-ahead bias by computing the decision for bar `t` using only data through bar `t`, then executing at the next bar close.

## Validation Status

BTC:

- Candidate beats v2.4 full-period on BTCUSDT 1h 365d / 2y / 3y.
- On BTCUSDT 1h 5y, both candidate and v2.4 are negative, but candidate is materially less negative.
- Rolling BTC win rates versus v2.4 are strong across Sharpe, return, and drawdown.
- Candidate remains better than v2.4 under `fee_rate = 0.002`.

ETH:

- ETHUSDT 1h 365d / 2y / 3y did not validate the BTC candidate.
- ETHUSDT 1h 5y improved versus v2.4, but both versions were deeply negative.
- ETH needs separate research under `v2_eth_exploration.py`.

## Important Caveats

- `v2.btc_final_candidate_A` is BTC-specific.
- It is not `v2.final`.
- It is not validated as a cross-asset strategy.
- Historical validation is not live trading evidence.
- Future production work should include walk-forward validation, paper trading, execution modeling, and monitoring.

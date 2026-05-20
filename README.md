# Control Quant System v4

Control Quant System v4 is a research implementation of a closed-loop trading
control system. It frames systematic trading as a control problem:

```text
observation -> filter -> state estimation -> controller -> portfolio accounting -> feedback
```

The current main candidate is:

```text
v4.2-candidate-C
```

This repository is for research, simulation, diagnostics, and validation record
keeping. It is not financial advice.

## Architecture

```text
Raw price data
-> Observation Vector
-> Filter Layer
-> State Estimator
-> Controller
-> Position Constraint
-> Portfolio Accounting
-> Metrics / Validation
```

Observation vector:

```text
y_t = [r_t, d_t^-, p_{t-1}]^T
```

Where:

- `r_t` is close-to-close log return.
- `d_t^-` is pre-trade drawdown.
- `p_{t-1}` is the previous position.

State vector:

```text
z_t = [tau_t, nu_t, epsilon_t, rho_t, p_{t-1}]^T
```

Where:

- `tau_t` is long-term trend state in `[-1, 1]`.
- `nu_t` is market risk / volatility state in `[0, 1]`.
- `epsilon_t` is short-term timing state in `[-1, 1]`.
- `rho_t` is portfolio risk state in `[0, 1]`.
- `p_{t-1}` is previous position in `[0, 1]`.

## Current Candidate

`v4.2-candidate-C` is the current main candidate. It is based on
`v4.2-candidate-A` and adds a weak-trend floor to the base exposure mapping:

```text
base_exposure = clip((tau - tau_floor) / (1 - tau_floor), 0, 1)
tau_floor = 0.10
```

This keeps the `k_tau = 5.0` trend sensitivity from candidate A, while reducing
exposure when trend is only weakly positive.

## Candidate Summary

`v4.1-default`

First minimal closed-loop control strategy. It was conservative with strong risk
control, but average exposure was too low.

`v4.2-candidate-A`

Changed only `k_tau` from `1.0` to `5.0`. This solved the low-exposure problem
and improved BTC validation, but ETH long-window stress increased.

`v4.2-candidate-B`

Based on A. Changed only `w_portfolio_risk` from `0.75` to `0.90`. It was not
promoted because it did not solve the ETH 5y issue.

`v4.2-candidate-C`

Current main candidate. Based on A and adds `tau_floor = 0.10` to avoid taking
exposure in weak positive trend states. It keeps most BTC improvement while
reducing weak-trend exposure.

`v4.2-candidate-D`

Based on C. Added `rebalance_threshold = 0.01`. Useful engineering experiment,
but not promoted. It reduced trade count but did not materially improve
performance.

`v4.2-candidate-E`

Based on C. Added a hard trend persistence gate. The direction was useful, but
this version was too conservative. It is preserved as an experiment record and
not promoted.

## Project Layout

```text
src/v4/
  data_types.py
  observation.py
  filters.py
  state_estimator.py
  controllers.py
  portfolio.py
  backtest.py
  metrics.py
  benchmarks.py
  configs.py

tests/
  test_v4_minimal_simulation_framework.py
  test_v4_filter_layer.py
  test_v4_minimal_state_estimator.py
  test_v4_minimal_controller.py
  test_v41_integration.py
  test_v42_candidate_a.py
  test_v42_candidate_b.py
  test_v42_candidate_c.py
  test_v42_candidate_d.py
  test_v42_candidate_e.py

examples/
  run_v41_minimal.py
  run_v42_candidate_c.py

docs/
  architecture.md
  v42_candidate_summary.md
  future_work_v43.md

results/
  validation/
  diagnostics/
```

The package path remains `src.v4` to preserve working imports and regression
tests.

## Install

```bash
pip install -r requirements.txt
```

## Run Tests

```bash
pytest
```

## Run Example

```bash
python examples/run_v42_candidate_c.py
```

## Run Validation

```bash
python scripts/run_validation.py
```

The validation scripts expect local daily or resampled crypto data. Raw market
CSV files under `data/` are intentionally not committed.

## Data Frequency

Default examples use daily crypto data:

```text
daily crypto data: periods_per_year = 365
4h crypto data:    periods_per_year = 2190
1h crypto data:    periods_per_year = 8760
```

## Disclaimer

This is a research project. It is not financial advice. Backtest results do not
guarantee future performance. Live trading requires additional validation,
execution modeling, monitoring, and risk controls.

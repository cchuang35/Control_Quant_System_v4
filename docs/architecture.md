# v4 Architecture

Control Quant System v4 is organized as a closed-loop trading control system.
Each period observes the latest market and portfolio state, filters the raw
observation, maps filtered signals into bounded state variables, chooses a
target position, applies constraints, and feeds the resulting portfolio state
back into the next observation.

## Layer 1: Observation Vector

The observation vector is:

```text
y_t = [r_t, d_t^-, p_{t-1}]^T
```

Where `r_t` is the close-to-close log return, `d_t^-` is pre-trade drawdown, and
`p_{t-1}` is the previous position. The return from `P_{t-1}` to `P_t` is earned
by `p_{t-1}`.

## Layer 2: Filter Layer

The minimal filter layer converts observations into causal filtered signals:

```text
phi_t = [L_t, V_t, S_t, D_t, p_{t-1}]^T
```

It uses recursive EWMA updates for long-term return, short-term return, and
variance proxy. It does not use centered windows, full-sample normalization, or
future data.

## Layer 3: State Estimator

The minimal state estimator maps filtered signals into the bounded state vector:

```text
z_t = [tau_t, nu_t, epsilon_t, rho_t, p_{t-1}]^T
```

Trend and timing are volatility-normalized and passed through `tanh`. Volatility
and drawdown are clipped against reference levels.

## Layer 4: Controller

The controller maps the state vector to a raw target position. The production
candidate for this repository is `v4.2-candidate-C`, a continuous long-only,
no-leverage controller with weak-trend suppression:

```text
base_exposure = clip((tau - tau_floor) / (1 - tau_floor), 0, 1)
tau_floor = 0.10
```

Timing, market risk, and portfolio risk are multiplicative modifiers. The final
target is smoothed by `max_position_change`.

## Layer 5: Portfolio Accounting

The accounting layer applies close-to-close returns, transaction costs, equity,
high watermark, and drawdown. Rebalancing occurs after observing `P_t`, so the
new position can only affect the next period.

## Layer 6: Metrics / Validation

Metrics include total return, annualized return, max drawdown, Sharpe ratio,
turnover, exposure, fee cost, and trade count. Validation compares strategy
candidates against zero position, true buy-and-hold, controller buy-and-hold,
and fixed exposure benchmarks.

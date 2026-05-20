# v4 Minimal Simulation Framework Specification

## 0. Purpose

This document defines the minimal simulation framework for the v4 closed-loop trading control system.

The goal of this stage is **not** to implement the final trading strategy yet. The goal is to build a clean, reusable, mathematically consistent simulation framework that future modules can plug into.

The framework currently includes:

1. State Vector
2. Observation Vector
3. Portfolio Accounting
4. Backtest Loop
5. Metrics

Future modules to be added later:

1. Filter Layer
2. State Estimator
3. Controller
4. Risk Feedback Extensions
5. Validation Layer

---

## 1. Core Design Principle

The v4 system is designed as a closed-loop trading control system.

The loop is:

```text
previous position
→ market return
→ portfolio equity update
→ drawdown update
→ observation vector
→ state estimation
→ controller decision
→ new position
→ transaction cost
→ next period
```

The most important anti-leakage principle is:

```text
At time t, position p_t cannot earn the return from P_{t-1} to P_t.
```

The return from `P_{t-1}` to `P_t` must be earned by the previous position:

```text
p_{t-1}
```

The newly decided position:

```text
p_t
```

only affects the next period:

```text
P_t → P_{t+1}
```

---

## 2. State Vector

### 2.1 Minimal State Vector

The v4 minimal state vector is defined as:

```text
z_t = [τ_t, ν_t, ε_t, ρ_t, p_{t-1}]^T
```

Where:

| Symbol | Name | Meaning | Suggested Range |
|---|---|---|---|
| `τ_t` | Long-term trend state | Main market direction / long-term bias | `[-1, 1]` |
| `ν_t` | Market risk / volatility state | Market risk or volatility level | `[0, 1]` |
| `ε_t` | Short-term timing state | Short-term entry / exit timing condition | `[-1, 1]` |
| `ρ_t` | Portfolio risk state | Internal portfolio stress / drawdown risk | `[0, 1]` |
| `p_{t-1}` | Previous position | Position held during the most recent return interval | `[0, 1]` |

### 2.2 State Meanings

#### 2.2.1 Long-Term Trend State `τ_t`

Purpose:

```text
Determine the main direction and base exposure.
```

Interpretation:

```text
τ_t > 0  → long-term bullish
τ_t = 0  → no clear long-term trend
τ_t < 0  → long-term bearish / risk-off
```

For the first version, the system is long-only, so a negative `τ_t` does not mean shorting. It means reducing or removing long exposure.

---

#### 2.2.2 Market Risk / Volatility State `ν_t`

Purpose:

```text
Scale exposure based on market risk.
```

Interpretation:

```text
ν_t = 0  → low market risk
ν_t = 1  → high market risk
```

Higher `ν_t` should generally reduce position size.

---

#### 2.2.3 Short-Term Timing State `ε_t`

Purpose:

```text
Fine-tune entries and exits.
```

Interpretation:

```text
ε_t > 0  → short-term condition supports increasing exposure
ε_t = 0  → neutral short-term timing
ε_t < 0  → short-term condition discourages increasing exposure
```

This state is auxiliary. The project direction is:

```text
Long-term controller as the main driver, short-term controller as auxiliary.
```

Therefore, `ε_t` should not fully override `τ_t` in the first version.

---

#### 2.2.4 Portfolio Risk State `ρ_t`

Purpose:

```text
Represent internal portfolio stress and enable closed-loop risk feedback.
```

Interpretation:

```text
ρ_t = 0  → portfolio condition normal
ρ_t = 1  → portfolio under high stress
```

This state should be estimated from portfolio drawdown or other internal portfolio risk measures.

---

#### 2.2.5 Previous Position `p_{t-1}`

Purpose:

```text
Control turnover, transaction cost, and position smoothness.
```

The previous position is required because the controller must know the current exposure before deciding how much to adjust.

---

## 3. Observation Vector

### 3.1 Minimal Observation Vector

The v4 minimal observation vector is defined as:

```text
y_t = [r_t, d_t^-, p_{t-1}]^T
```

Where:

| Symbol | Name | Meaning |
|---|---|---|
| `r_t` | Log market return | Market return observation from `P_{t-1}` to `P_t` |
| `d_t^-` | Pre-trade drawdown | Portfolio drawdown after market movement but before rebalancing cost |
| `p_{t-1}` | Previous position | Position that earned the return from `P_{t-1}` to `P_t` |

### 3.2 Market Return Observation

The observation return is defined as log return:

```text
r_t = log(P_t / P_{t-1})
```

This is used for filtering, state estimation, and signal construction.

### 3.3 Portfolio Drawdown Observation

The observation uses **pre-trade drawdown**:

```text
d_t^-
```

not post-trade drawdown:

```text
d_t
```

Reason:

If `d_t` were used in `y_t`, it would create a circular dependency:

```text
p_t → cost_t → E_t → d_t → y_t → z_t → p_t
```

Therefore, the observation at time `t` must use the portfolio state before the new trade is applied.

---

## 4. Portfolio Accounting

## 4.1 Minimal Assumptions

The first version uses the simplest possible accounting model:

```text
1. Long-only
2. No leverage
3. Position range: p_t ∈ [0, 1]
4. Cash return = 0
5. Close-to-close return
6. Constant proportional transaction fee
7. Rebalancing happens after observing P_t
8. New position p_t affects the next period only
```

Not included in the first version:

```text
slippage
bid-ask spread
funding rate
borrow cost
margin liquidation
shorting
partial fills
dynamic fees
tax
multi-asset allocation
```

---

## 4.2 Variables

| Symbol | Meaning |
|---|---|
| `P_t` | Price at time `t` |
| `R_t` | Simple return from `P_{t-1}` to `P_t` |
| `r_t` | Log return from `P_{t-1}` to `P_t` |
| `p_{t-1}` | Previous position |
| `p_t` | New position after decision at time `t` |
| `E_t^-` | Pre-trade equity at time `t` |
| `E_t` | Final equity at time `t` after trading cost |
| `H_t^-` | Pre-trade high watermark |
| `H_t` | Final high watermark |
| `d_t^-` | Pre-trade drawdown |
| `d_t` | Final drawdown |
| `u_t` | Turnover at time `t` |
| `f` | Fee rate |
| `cost_t` | Transaction cost at time `t` |

---

## 4.3 Return Definitions

Simple return:

```text
R_t = P_t / P_{t-1} - 1
```

Log return:

```text
r_t = log(P_t / P_{t-1})
```

Use `R_t` for equity accounting.

Use `r_t` for observation and state estimation.

---

## 4.4 Equity Update

The position held during the interval `P_{t-1} → P_t` is `p_{t-1}`.

Pre-trade equity:

```text
E_t^- = E_{t-1}(1 + p_{t-1}R_t)
```

Turnover:

```text
u_t = |p_t - p_{t-1}|
```

Transaction cost:

```text
cost_t = f u_t E_t^-
```

Final equity:

```text
E_t = E_t^- - cost_t
```

Equivalent form:

```text
E_t = E_t^- (1 - f u_t)
```

Combined formula:

```text
E_t = E_{t-1}(1 + p_{t-1}R_t)(1 - f|p_t - p_{t-1}|)
```

---

## 4.5 Drawdown Update

Pre-trade high watermark:

```text
H_t^- = max(H_{t-1}, E_t^-)
```

Pre-trade drawdown:

```text
d_t^- = 1 - E_t^- / H_t^-
```

Final high watermark:

```text
H_t = max(H_t^-, E_t)
```

Final drawdown:

```text
d_t = 1 - E_t / H_t
```

---

## 4.6 Initial Conditions

Use normalized initial capital:

```text
E_0 = 1.0
H_0 = 1.0
p_0 = 0.0
```

---

## 5. Backtest Loop

## 5.1 Purpose

The backtest loop defines the time order of the closed-loop simulation.

It is not only a performance-testing tool. It is the runtime skeleton for the control system.

---

## 5.2 Minimal Inputs

```text
price series: P_0, P_1, ..., P_T
initial equity: E_0 = 1.0
initial high watermark: H_0 = 1.0
initial position: p_0 = 0.0
fee rate: f
```

---

## 5.3 Required Module Interfaces

The framework should support the following module interfaces.

### Observation Builder

Input:

```text
r_t, d_t^-, p_{t-1}
```

Output:

```text
y_t = [r_t, d_t^-, p_{t-1}]^T
```

---

### State Estimator

Input:

```text
history of observations y_1, ..., y_t
```

Output:

```text
z_t = [τ_t, ν_t, ε_t, ρ_t, p_{t-1}]^T
```

For now, this can be a placeholder or dummy estimator.

---

### Controller

Input:

```text
z_t
```

Output:

```text
q_t
```

Where `q_t` is the raw target position before clipping.

For now, this can be a dummy controller such as:

```text
q_t = 0.0
q_t = 0.5
q_t = 1.0
```

---

### Position Constraint

Input:

```text
q_t
```

Output:

```text
p_t
```

Minimal version:

```text
p_t = clip(q_t, 0, 1)
```

---

## 5.4 Formal Backtest Loop

Initialize:

```text
E_0 = 1.0
H_0 = 1.0
p_0 = 0.0
```

For `t = 1` to `T`:

```text
1. Compute returns:

   R_t = P_t / P_{t-1} - 1
   r_t = log(P_t / P_{t-1})

2. Update pre-trade equity using previous position:

   E_t^- = E_{t-1}(1 + p_{t-1}R_t)

3. Update pre-trade high watermark:

   H_t^- = max(H_{t-1}, E_t^-)

4. Compute pre-trade drawdown:

   d_t^- = 1 - E_t^- / H_t^-

5. Build observation vector:

   y_t = [r_t, d_t^-, p_{t-1}]^T

6. Estimate state:

   z_t = StateEstimator(y_1, ..., y_t)

7. Controller produces raw target position:

   q_t = Controller(z_t)

8. Apply position constraint:

   p_t = clip(q_t, 0, 1)

9. Compute turnover:

   u_t = |p_t - p_{t-1}|

10. Compute transaction cost:

   cost_t = f u_t E_t^-

11. Update final equity:

   E_t = E_t^- - cost_t

   equivalent:

   E_t = E_t^- (1 - f u_t)

12. Update final high watermark:

   H_t = max(H_t^-, E_t)

13. Update final drawdown:

   d_t = 1 - E_t / H_t

14. Record all relevant variables.
```

---

## 5.5 Variables to Record Each Period

For each time step `t`, record:

```text
P_t
R_t
r_t
E_t^-
H_t^-
d_t^-
y_t
z_t
q_t
p_t
u_t
cost_t
E_t
H_t
d_t
```

Also record:

```text
p_{t-1}
```

because it is the position that actually earned `R_t`.

---

## 5.6 Sanity-Check Controllers

Before implementing real filters, state estimators, or controllers, the framework should be tested with dummy controllers.

### Zero Position Controller

```text
q_t = 0
```

Expected behavior:

```text
equity should remain near 1.0
turnover should be zero after initialization
fee should be zero
max drawdown should be zero
```

---

### Buy and Hold Controller

```text
q_t = 1
```

Expected behavior:

```text
system should approximate buy-and-hold after initial entry cost
average exposure should be near 1.0
turnover should mainly come from the first transition 0 → 1
```

---

### Fixed Half-Exposure Controller

```text
q_t = 0.5
```

Expected behavior:

```text
average exposure should be near 0.5
returns and drawdowns should generally be between zero exposure and full exposure
```

---

## 6. Metrics

## 6.1 Purpose

Metrics evaluate the behavior of the closed-loop system.

The first version should evaluate:

```text
return
risk
drawdown
risk-adjusted return
turnover
exposure
fee impact
trade activity
```

---

## 6.2 Inputs

Metrics use the records generated by the backtest loop:

```text
E_0, E_1, ..., E_T
d_1, d_2, ..., d_T
p_0, p_1, ..., p_T
u_1, u_2, ..., u_T
cost_1, cost_2, ..., cost_T
```

Also required:

```text
A = periods per year
```

For crypto:

```text
daily data: A = 365
4h data:    A = 365 × 6 = 2190
1h data:    A = 365 × 24 = 8760
```

---

## 6.3 Strategy Net Return Series

Define strategy net return:

```text
g_t = E_t / E_{t-1} - 1
```

This already includes:

```text
market return
position size
transaction cost
```

---

## 6.4 Core Metrics

### 1. Total Return

```text
Total Return = E_T / E_0 - 1
```

Since `E_0 = 1.0`:

```text
Total Return = E_T - 1
```

---

### 2. Annualized Return

```text
Annualized Return = (E_T / E_0)^(A / T) - 1
```

Where:

```text
A = periods per year
T = number of return periods
```

---

### 3. Max Drawdown

```text
Max Drawdown = max(d_t)
```

---

### 4. Sharpe Ratio

Assume risk-free rate is zero in the first version.

```text
Sharpe = mean(g_t) / std(g_t) × sqrt(A)
```

If `std(g_t) = 0`, Sharpe should be returned as `NaN`, not forced to zero.

---

### 5. Total Turnover

```text
Total Turnover = Σ u_t
```

---

### 6. Average Turnover

```text
Average Turnover = Σ u_t / T
```

Optional later extension:

```text
Annualized Turnover = Average Turnover × A
```

---

### 7. Average Exposure

Use the position that actually earned each period's return:

```text
Average Exposure = mean(p_{t-1})
```

Do not use `p_t` for exposure over period `t`, because `p_t` only affects the next period.

---

### 8. Total Fee Cost

```text
Total Fee Cost = Σ cost_t
```

Because `E_0 = 1.0`, this is directly interpretable relative to initial capital.

---

### 9. Trade Count

Use a small epsilon threshold to avoid floating-point noise:

```text
Trade Count = count(u_t > ε)
```

Suggested:

```text
ε = 1e-6
```

---

## 6.5 Minimal Metrics Output

The metrics evaluator should return a dictionary or structured object containing at least:

```text
total_return
annualized_return
max_drawdown
sharpe_ratio
total_turnover
average_turnover
average_exposure
total_fee_cost
trade_count
```

---

## 7. Implementation Requirements

## 7.1 Required Behavior

The implementation should:

```text
1. Accept a time-ordered price series.
2. Compute simple returns and log returns.
3. Run the minimal backtest loop without future leakage.
4. Support dummy controllers.
5. Record all period-level values.
6. Compute minimal metrics.
7. Keep modules separated.
```

---

## 7.2 Suggested Module Structure

Suggested files or classes:

```text
observation.py
    ObservationBuilder

portfolio.py
    PortfolioAccounting

backtest.py
    BacktestEngine

metrics.py
    MetricsEvaluator

interfaces.py
    StateEstimator interface
    Controller interface
    PositionConstraint interface

controllers.py
    ZeroController
    BuyAndHoldController
    FixedExposureController
```

A simpler first implementation may keep everything in one file, but the code should still be logically modular.

---

## 7.3 Suggested Data Structures

Each step's output can be stored as a row in a table-like structure.

Recommended columns:

```text
timestamp
price
simple_return
log_return
pre_trade_equity
pre_trade_high_watermark
pre_trade_drawdown
previous_position
observation
state
raw_target_position
position
turnover
transaction_cost
equity
high_watermark
drawdown
```

If using Python, a pandas DataFrame is acceptable.

---

## 7.4 Important Edge Cases

The implementation should handle:

```text
1. Missing or non-positive prices should raise an error or be cleaned before simulation.
2. The first row has no return and should be used only for initialization.
3. Sharpe should be NaN if return standard deviation is zero.
4. Position should always be clipped to [0, 1] in the minimal version.
5. Transaction cost should be based on pre-trade equity E_t^-.
6. Average exposure should use p_{t-1}, not p_t.
```

---

## 8. Definition of Done

The minimal framework is complete when the following are true:

```text
1. Zero-position controller keeps equity at 1.0, with no fees and no drawdown.
2. Buy-and-hold controller behaves like buy-and-hold after initial entry fee.
3. Fixed 0.5 exposure controller has average exposure near 0.5.
4. No future leakage exists in the loop.
5. Metrics are computed from recorded portfolio results.
6. The framework allows future replacement of dummy estimator/controller with real modules.
```

---


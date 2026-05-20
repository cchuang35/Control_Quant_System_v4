# Future Work: v4.3 Trend Quality

Do not implement v4.3 yet. This document records the next research direction.

The v4.2 diagnostics suggest that the next step should focus on trend quality
estimation rather than additional turnover controls or hard gates.

## Direction

Split the trend concept into two separate state ideas:

```text
tau_t = trend direction / strength
q_trend_t = trend quality / confidence
```

Then the controller can use:

```text
base_exposure = base_exposure_C * q_trend
```

This keeps `tau_t` responsible for direction and strength, while `q_trend_t`
captures persistence, reliability, or confidence.

## Why This Matters

`v4.2-candidate-C` improved over A by suppressing weak positive trend. However,
ETH long-window diagnostics still suggest that false-positive trend exposure can
occur during bear-market rallies or non-persistent trend signals.

`v4.2-candidate-E` tested a hard persistence gate. It helped explore the idea,
but was too conservative. v4.3 should use a softer and better-calibrated trend
quality state rather than a hard controller-only gate.

## Constraints

Future v4.3 work should preserve the anti-leakage rule:

```text
The return from P_{t-1} to P_t is earned by p_{t-1}.
The newly computed p_t can only affect the next period.
```

No future prices, future returns, centered windows, or full-sample statistics
should be used inside the strategy.

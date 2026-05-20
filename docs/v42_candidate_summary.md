# v4.2 Candidate Summary

The current main candidate is `v4.2-candidate-C`.

## Why v4.1 Was Too Conservative

`v4.1-default` was the first full runnable minimal closed-loop strategy. It had
strong risk control and low drawdown, but diagnostics showed very low average
exposure. The trend state and base exposure were often too small to participate
meaningfully in positive regimes.

## Why `k_tau = 5.0` Was Tested

`v4.2-candidate-A` changed only `k_tau` from `1.0` to `5.0`. The goal was to
increase long-term trend sensitivity without changing the filter, controller
architecture, accounting, or risk multipliers.

## Why A Improved BTC but Created ETH Long-Window Risk

Candidate A solved the low-exposure issue and improved BTC validation. However,
ETH long-window diagnostics showed higher portfolio stress and weaker behavior
in the extra early 2-year segment of the ETH 5y window. The system sometimes
held exposure in weak or false-positive trend conditions.

## Why v4.2-C Was Created

`v4.2-candidate-C` keeps candidate A's `k_tau = 5.0`, but changes the base
exposure mapping:

```text
base_exposure = clip((tau - tau_floor) / (1 - tau_floor), 0, 1)
tau_floor = 0.10
```

This makes weak positive trend insufficient to create exposure. It targets the
specific issue of weak-trend participation without adding a new estimator,
optimization step, or regime model.

## Why v4.2-C Is Current Main Candidate

Candidate C is the most balanced v4.2 candidate so far. It preserves most of
the BTC improvement from candidate A while reducing weak-trend exposure and ETH
stress. It is therefore the current main candidate for repository publication.

## Why D and E Are Not Promoted

`v4.2-candidate-D` added a rebalance deadband of `0.01`. It reduced trade count
and turnover, but did not materially improve the ETH long-window issue.

`v4.2-candidate-E` added a hard trend persistence gate. The direction was useful
for trend-quality research, but this specific implementation was too
conservative. It is preserved as an experiment record, not promoted.

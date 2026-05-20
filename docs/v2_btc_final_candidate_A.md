# v2.btc_final_candidate_A

## Purpose

`v2.btc_final_candidate_A` freezes the current best BTCUSDT 1h v2 wrapper candidate for validation and future comparison. It is not a general cross-asset final strategy.

## Scope

- Intended market: BTCUSDT 1h.
- Status: BTC-specific final candidate, not `v2.final`.
- Not validated for ETH. Expanded validation showed ETH 365d / 2y / 3y performance was generally worse than `v2.4.pause_4_24_exit_7`.

## Strategy Rules

The base regime gate is sideways-hold-only:

- `strong_bull`: allow entry, allow hold.
- `weak_bull`: allow entry and hold by default, except weak-bull entries are blocked during weak-bull loss cooldown.
- `sideways`: block new entries, allow hold.
- `bear`: block entry and block hold.

Weak-bull losing-trade cooldown:

- Track completed trades.
- If a completed trade has `entry_regime == weak_bull` and `net_trade_return < 0`, block new weak-bull entries for `cooldown_bars`.
- Default `cooldown_bars = 120`.
- Observed robust range: `120 / 144 / 168` one-hour bars, interpreted as roughly 5-7 days.
- During cooldown, only new weak-bull entries are blocked.
- Strong-bull entries remain allowed.
- Existing positions are not force-exited by the cooldown.
- Sideways and bear behavior is unchanged.

Position and return rules:

- Position is binary: `0` or `1`.
- Fee-aware return:

```text
strategy_return_net[t] =
    position[t-1] * asset_return[t]
    - abs(position[t] - position[t-1]) * fee_rate
```

- Signal at bar `t` determines position for bar `t+1`; row `t` return is earned by the prior position.

## Validation Summary

Available BTC validation:

- BTCUSDT 1h 365d, 2y, 3y: candidate beat `v2.4.pause_4_24_exit_7` full-period.
- BTCUSDT 1h 5y: both candidate and `v2.4` were negative, but candidate was materially less negative.
- Rolling BTC win rates versus `v2.4` were mostly near-perfect across Sharpe, total return, and max drawdown.
- Candidate remained better than `v2.4` under `fee_rate = 0.002`.

ETH caveat:

- ETHUSDT 1h 365d / 2y / 3y generally did not validate the candidate.
- ETHUSDT 1h 5y improved versus `v2.4`, but both were deeply negative.
- Do not treat this as a cross-asset final strategy.

## Fee Sensitivity

Fee rates tested:

- `0.0005`
- `0.0010`
- `0.0020`

On BTC datasets, the candidate remained viable under the highest tested fee. Lower turnover versus `v2.4` helped preserve performance.

## Cooldown Robustness

Cooldown values `120 / 144 / 168` were identical on many BTC validation slices and broadly similar on longer datasets. This supports interpreting the parameter as a coarse 5-7 day weak-bull pause after losing weak-bull trades, not a finely tuned exact value.

## Known Limitations

- Not promoted to `v2.final`.
- Cross-asset ETH validation failed.
- Some BTC performance remains concentrated in active trend windows.
- BTC/ETH 5y downloaded datasets contain a small number of missing 1h bars; see `reports/data_quality/oos_data_quality.csv`.
- Validation remains historical and should be expanded before production use.

## Future Work

- Build separate ETH-specific research under `v2.eth_exploration`.
- Add additional exchanges or alternate market regimes for BTC.
- Validate on walk-forward splits that were not used during candidate selection.
- Add live-paper diagnostics before any capital allocation.

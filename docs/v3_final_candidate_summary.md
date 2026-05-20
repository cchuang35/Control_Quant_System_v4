# v3 Final Candidate Summary

This document summarizes the v3 final-candidate decision based on the baseline reports, rolling validation, BTC/ETH validation, and v3.2-v3.5 experiments. The conclusion is deliberately conservative: v3 is a useful risk-controlled exposure-controller architecture, but it is not yet a proven BTC alpha upgrade over v2 and it is not yet proven cross-asset.

## Evidence Reviewed

- `reports/v3_baseline_btc_1h.md`
- `reports/v3_baseline_eth_1h.md`
- `reports/v3_rolling_validation_btc_1h.md`
- `reports/v3_baseline_small_improvements.md`
- `reports/v3_2_discrete_position_experiment.md`
- `reports/v3_3_short_term_aux_experiment.md`
- `reports/v3_4_risk_supervisor_experiment.md`
- `reports/v3_5_conditional_leverage_experiment.md`

## Final Selected v3 Configuration

The selected v3 final candidate should be treated as a named configuration assembled from the best-supported experiment results:

- Feature windows: BTCUSDT/ETHUSDT 1h defaults with `ma_short=24`, `ma_long=168`, `ma_long_term=720`, short/long momentum `24/168`, short/long volatility `24/168`, and short/long drawdown `168/720`.
- Market estimator: rule-based v3 estimator only. No particle filter.
- Long-term controller: primary controller, using the conservative base-position mapping:
  - `strong_bull -> 0.75`
  - `bull -> 0.50`
  - `neutral -> 0.25`
  - `bear -> 0.00`
  - `strong_bear -> 0.00`
- Short-term auxiliary controller: keep only defensive reductions:
  - `enable_pullback_add = false`
  - `enable_recovery_add = false`
  - `enable_overheat_reduce = true`
  - `enable_breakdown_reduce = true`
  - `allow_neutral_recovery_add = false`
  - `experimental_mode = false`
- Cooldown manager: preserve bull-like losing-trade cooldown with `cooldown_bars=120` by default. Cooldown blocks repeated bull-like additions or entries; it does not force immediate exits.
- Risk supervisor: keep the full supervisor enabled, including drawdown caps, volatility caps, market-risk-state controls, consecutive-loss rules, and cost guards.
- Position scheme: conservative discrete exposure `[0.0, 0.25, 0.50, 0.75, 1.0]`.
- Rounding: conservative floor rounding to the nearest allowed exposure not exceeding the risk-limited target.
- Leverage: disabled by default. `max_position=1.0`.
- Execution: fee-aware, no-lookahead execution. Signal at bar `t` sets position for bar `t+1`; return at bar `t` is earned by the position from `t-1`.
- Fee validation: continue validating at `0.0005`, `0.0010`, and `0.0020`.

This final candidate should be re-run as an explicit `v3.final_candidate` report before being promoted beyond candidate status. The current decision is assembled from component experiments, not from one standalone final-candidate backtest artifact.

## Why This Configuration Was Selected

The v3.2 discrete-position experiment selected conservative discrete exposure as the best BTC drawdown/Sharpe tradeoff. Binary exposure was too restrictive with floor rounding, and coarse exposure did not improve the main tradeoff. Conservative discrete exposure had the best average Sharpe and average max drawdown among the tested schemes.

The v3.3 short-term experiment rejected bullish pullback additions. Pullback adds worsened average annual return and drawdown. The best short-term rule set was `C_overheat_breakdown_reduce_only`, meaning the short-term controller should remain auxiliary and defensive.

The v3.4 risk-supervisor experiment selected the full Risk Supervisor. Drawdown cap was the most useful single rule, while the full rule stack had the best average drawdown, Sharpe, annual return, and turnover among tested risk variants.

The v3.5 leverage experiment did not justify adding leverage. Conditional leverage produced zero leverage entries under the strict test conditions, so it should remain experimental and disabled by default.

## v2 Ideas Preserved

v3 preserves several useful v2 ideas:

- Regime-gated behavior: v3 still uses market regime to decide whether entry and hold are allowed.
- Sideways/neutral caution: neutral conditions generally block new aggressive entry but can allow controlled hold exposure.
- Weak-bull losing-trade cooldown: v3 keeps the concept as a configurable cooldown manager for bull-like regimes.
- Fee-aware return calculation: v3 explicitly subtracts transaction costs from position changes.
- Multi-fee validation: v3 reports continue to test `0.0005`, `0.0010`, and `0.0020`.
- BTC/ETH separated validation: v3 keeps BTC as the main validation path and ETH as a separate generalization check.

## v2 Ideas Changed

v3 changes the strategy shape in important ways:

- v2 is a BTCUSDT 1h wrapper around `v1.final`; v3 is a modular exposure controller.
- v2 is effectively binary-position oriented; v3 uses discrete fractional exposure.
- v2 uses short-term and regime gates around a wrapped strategy; v3 makes the long-term controller primary and short-term behavior auxiliary.
- v3 gives the Risk Supervisor highest authority over final exposure.
- v3 rejects short-term pullback/recovery additions for the final candidate because the experiment did not support them.
- v3 does not introduce particle filters, complex ML, or high leverage.

## BTC Conclusion Versus v2

v3 does not beat `v2.btc_final_candidate_A` on BTC return or Sharpe in the current evidence. The BTC baseline report shows v2 still captures more upside on the main BTC validation sets. For example, on `btcusdt_1h_365d` at `fee_rate=0.001`, v2 returned `0.0834362` with Sharpe `1.21928`, while v3 baseline returned `-0.0135638` with Sharpe `-0.294511`.

v3 is better than v2 on risk containment, turnover, and fee drag. Rolling BTC validation averaged `max_drawdown=-0.000859559`, `turnover=0.186538`, and `fee_drag=0.000217628` for v3, versus `max_drawdown=-0.0360264`, `turnover=5.65722`, and `fee_drag=0.00660009` for v2. That is a real risk-control improvement, but it comes with weak return capture: rolling average total return was `-0.00021812` for v3 versus `0.00266448` for v2.

The honest BTC conclusion: v3 is not yet a better BTC strategy than v2. It is a better-controlled exposure framework that needs more work before it can replace v2 as the main BTC candidate.

## ETH Generalization Conclusion

v3 works mechanically on ETH: all tested ETH datasets produced fee-aware positions, returns, and diagnostics. It also compares favorably to applying the BTC-tuned v2 candidate to ETH, mostly because v2 performs poorly on ETH and v3 stays very defensive.

This is not enough to claim cross-asset validation. ETH remains weak by absolute return. The ETH baseline average v3 total return was only `0.00317603`, the positive-result rate was `0.25`, and average exposure was `0.00333691`. The 2-year ETH window was positive, but the 365d, 3y, and 5y windows remained weak or negative.

The honest ETH conclusion: v3 is potentially cross-asset in architecture only. It should not be marketed or treated as cross-asset validated yet.

## Fee Sensitivity

v3 is less fee-sensitive than v2 because it trades far less and carries lower average exposure. In rolling BTC validation, v3 average fee drag was `0.000217628` versus v2 average fee drag of `0.00660009`.

The caveat is that v3's return edge is also small. Low fee drag helps prevent damage, but it does not by itself create enough return. Fee robustness is therefore a strength of the architecture, not proof of alpha quality.

## Rolling Validation Conclusion

Rolling validation supports v3 as a robust risk controller. It repeatedly reduces drawdown, turnover, and fee drag versus v2.

Rolling validation does not support v3 as a stronger return engine. Average rolling total return and Sharpe were weaker than v2. The system is often underexposed or fully blocked during windows where v2 still captures gains.

## Known Weaknesses

- v3 is too conservative in its current form and often misses upside.
- Average exposure is extremely low in many BTC and ETH runs.
- BTC return and Sharpe do not beat v2.
- ETH does not validate strongly enough to claim cross-asset robustness.
- Pullback/recovery additions currently add risk without enough return benefit.
- Some strong-bull opportunities are blocked by confidence, risk-state, cooldown, or consecutive-loss rules.
- The final selected configuration is assembled from experiments and still needs one explicit `v3.final_candidate` run.
- Leverage conditions did not trigger in v3.5, so leverage remains unproven.
- The rule-based estimator may be too blunt for long-term regime transitions.

## Deferred To v4

- Particle filter or probabilistic market-state estimation.
- More advanced uncertainty-aware confidence scoring.
- Any short-term-primary strategy branch.
- Cross-asset calibration beyond BTC/ETH validation.
- Portfolio-level multi-asset exposure allocation.
- High leverage or routine `max_position > 1.0`.
- Complex ML models or large indicator expansions.
- Formal walk-forward parameter selection and anti-overfit tooling.

## Final Recommendation

Use v3 final candidate as a conservative architecture checkpoint, not as a replacement for v2. The selected candidate should preserve v3's strongest evidence-backed traits: discrete exposure, long-term-primary control, defensive short-term reductions, strict risk supervision, fee-aware execution, and no leverage.

The next implementation step should be to create and run an explicit `v3.final_candidate` configuration using the selected settings above, then compare it directly against v2 on BTC and against the untuned ETH validation set.

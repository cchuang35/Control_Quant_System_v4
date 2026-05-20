# Validation Protocol

Default validation uses daily crypto data and:

```text
fee_rate = 0.001
periods_per_year = 365
```

Candidate comparison scripts also evaluate fee rates:

```text
0
0.001
0.002
```

Standard datasets:

```text
BTC daily 365d / 2y / 3y / 5y
ETH daily 365d / 2y / 3y / 5y
```

Standard benchmarks:

```text
zero_position
true_buy_and_hold
controller_buy_and_hold
fixed_0_5_exposure
```

Main strategy candidate:

```text
v4.2-candidate-C
```

Raw market data is not committed to the repository. Validation scripts expect
local CSV files under `data/`.

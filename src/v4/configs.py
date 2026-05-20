"""Central defaults for v4 minimal-control strategy versions."""

from __future__ import annotations

from dataclasses import dataclass, field

from .backtest import run_backtest
from .controllers import ControllerConfig, MinimalContinuousController
from .data_types import BacktestConfig
from .filters import FilterConfig, MinimalFilterLayer
from .state_estimator import MinimalStateEstimator, StateEstimatorConfig


V41_VERSION_NAME = "v4.1-minimal-control-strategy"
V42_CANDIDATE_A_VERSION_NAME = "v4.2-candidate-A"
V42_CANDIDATE_A_DESCRIPTION = (
    "v4.2-candidate-A is v4.1-minimal-control-strategy with stronger long-term "
    "trend sensitivity. It changes only k_tau from 1.0 to 5.0. This addresses "
    "the v4.1 diagnostic finding that tau and base exposure were too small."
)
V42_CANDIDATE_B_VERSION_NAME = "v4.2-candidate-B"
V42_CANDIDATE_B_DESCRIPTION = (
    "v4.2-candidate-B is v4.2-candidate-A with stronger portfolio risk feedback. "
    "It is designed to reduce high portfolio stress seen in ETH long-window validation, "
    "especially ETH 5y, while preserving the improved exposure from k_tau = 5.0."
)
V42_CANDIDATE_C_VERSION_NAME = "v4.2-candidate-C"
V42_CANDIDATE_C_DESCRIPTION = (
    "v4.2-candidate-C is based on v4.2-candidate-A. It keeps k_tau = 5.0 "
    "but adds a tau_floor base-exposure mapping. The purpose is to avoid taking "
    "exposure in weak positive trend states, especially to reduce ETH long-window "
    "portfolio stress observed in v4.2-candidate-A."
)
V42_CANDIDATE_D_VERSION_NAME = "v4.2-candidate-D"
V42_CANDIDATE_D_DESCRIPTION = (
    "v4.2-candidate-D is based on v4.2-candidate-C. It keeps k_tau = 5.0 "
    "and tau_floor = 0.10. It adds a rebalance deadband of 0.01 to reduce "
    "small, low-value trades. The purpose is to reduce turnover and fee drag, "
    "especially in ETH long-window validation."
)
V42_CANDIDATE_E_VERSION_NAME = "v4.2-candidate-E"
V42_CANDIDATE_E_DESCRIPTION = (
    "v4.2-candidate-E is based on v4.2-candidate-C. It keeps k_tau = 5.0 "
    "and tau_floor = 0.10. It adds a causal trend persistence gate based on "
    "the recent persistence of tau > 0.25. The goal is to reduce false-positive "
    "trend exposure during bear-market rallies, especially in the ETH extra "
    "early 2-year segment that drags down ETH 5y performance."
)
DAILY_CRYPTO_PERIODS_PER_YEAR = 365
FOUR_HOUR_CRYPTO_PERIODS_PER_YEAR = 2190
HOURLY_CRYPTO_PERIODS_PER_YEAR = 8760


@dataclass(frozen=True)
class V41DefaultConfig:
    """Grouped default config for the first runnable v4.1 strategy."""

    version_name: str = V41_VERSION_NAME
    periods_per_year: int = DAILY_CRYPTO_PERIODS_PER_YEAR
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    state_estimator: StateEstimatorConfig = field(default_factory=StateEstimatorConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)

    @property
    def warmup_period(self) -> int:
        return self.filter.long_window

    @property
    def position_range(self) -> tuple[float, float]:
        return (0.0, 1.0)


def create_v41_default_config() -> V41DefaultConfig:
    """Return the official v4.1 minimal-control default config."""

    return V41DefaultConfig()


def build_v41_estimator(config: V41DefaultConfig | None = None) -> MinimalStateEstimator:
    """Build the default v4.1 filter plus state estimator."""

    resolved = config or create_v41_default_config()
    return MinimalStateEstimator(
        filter_layer=MinimalFilterLayer(resolved.filter),
        config=resolved.state_estimator,
    )


def build_v41_controller(config: V41DefaultConfig | None = None) -> MinimalContinuousController:
    """Build the default v4.1 continuous controller."""

    resolved = config or create_v41_default_config()
    return MinimalContinuousController(resolved.controller)


def run_v41_backtest(prices, config: V41DefaultConfig | None = None, *, price_column: str = "close"):
    """Run v4.1-minimal-control-strategy with the grouped defaults."""

    resolved = config or create_v41_default_config()
    return run_backtest(
        prices,
        controller=build_v41_controller(resolved),
        state_estimator=build_v41_estimator(resolved),
        config=resolved.backtest,
        price_column=price_column,
    )


def make_v42_candidate_a_config(*, fee_rate: float | None = None) -> V41DefaultConfig:
    """Return v4.2-candidate-A config.

    v4.2-candidate-A changes only ``k_tau`` from the v4.1 default 1.0 to 5.0.
    All filter, controller, and accounting defaults remain unchanged.
    """

    v41 = create_v41_default_config()
    backtest = v41.backtest
    if fee_rate is not None:
        backtest = BacktestConfig(
            fee_rate=fee_rate,
            initial_equity=v41.backtest.initial_equity,
            initial_high_watermark=v41.backtest.initial_high_watermark,
            initial_position=v41.backtest.initial_position,
        )
    return V41DefaultConfig(
        version_name=V42_CANDIDATE_A_VERSION_NAME,
        periods_per_year=v41.periods_per_year,
        backtest=backtest,
        filter=v41.filter,
        state_estimator=StateEstimatorConfig(
            k_tau=5.0,
            k_epsilon=v41.state_estimator.k_epsilon,
            vol_ref=v41.state_estimator.vol_ref,
            drawdown_ref=v41.state_estimator.drawdown_ref,
            epsilon=v41.state_estimator.epsilon,
        ),
        controller=v41.controller,
    )


def build_v42_candidate_a_strategy(
    config: V41DefaultConfig | None = None,
) -> tuple[MinimalContinuousController, MinimalStateEstimator]:
    """Build controller and estimator for v4.2-candidate-A."""

    resolved = config or make_v42_candidate_a_config()
    return build_v41_controller(resolved), build_v41_estimator(resolved)


def run_v42_candidate_a_backtest(prices, config: V41DefaultConfig | None = None, *, price_column: str = "close"):
    """Run v4.2-candidate-A."""

    resolved = config or make_v42_candidate_a_config()
    controller, estimator = build_v42_candidate_a_strategy(resolved)
    return run_backtest(
        prices,
        controller=controller,
        state_estimator=estimator,
        config=resolved.backtest,
        price_column=price_column,
    )


def make_v42_candidate_b_config(*, fee_rate: float | None = None) -> V41DefaultConfig:
    """Return v4.2-candidate-B config.

    v4.2-candidate-B starts from v4.2-candidate-A and changes only
    ``w_portfolio_risk`` from 0.75 to 0.90.
    """

    candidate_a = make_v42_candidate_a_config(fee_rate=fee_rate)
    return V41DefaultConfig(
        version_name=V42_CANDIDATE_B_VERSION_NAME,
        periods_per_year=candidate_a.periods_per_year,
        backtest=candidate_a.backtest,
        filter=candidate_a.filter,
        state_estimator=candidate_a.state_estimator,
        controller=ControllerConfig(
            w_epsilon=candidate_a.controller.w_epsilon,
            w_volatility=candidate_a.controller.w_volatility,
            w_portfolio_risk=0.90,
            max_position_change=candidate_a.controller.max_position_change,
            tau_floor=candidate_a.controller.tau_floor,
            rebalance_threshold=candidate_a.controller.rebalance_threshold,
            use_trend_persistence_gate=candidate_a.controller.use_trend_persistence_gate,
            tau_confirm_threshold=candidate_a.controller.tau_confirm_threshold,
            trend_persistence_window=candidate_a.controller.trend_persistence_window,
            persistence_floor=candidate_a.controller.persistence_floor,
        ),
    )


def build_v42_candidate_b_strategy(
    config: V41DefaultConfig | None = None,
) -> tuple[MinimalContinuousController, MinimalStateEstimator]:
    """Build controller and estimator for v4.2-candidate-B."""

    resolved = config or make_v42_candidate_b_config()
    return build_v41_controller(resolved), build_v41_estimator(resolved)


def run_v42_candidate_b_backtest(prices, config: V41DefaultConfig | None = None, *, price_column: str = "close"):
    """Run v4.2-candidate-B."""

    resolved = config or make_v42_candidate_b_config()
    controller, estimator = build_v42_candidate_b_strategy(resolved)
    return run_backtest(
        prices,
        controller=controller,
        state_estimator=estimator,
        config=resolved.backtest,
        price_column=price_column,
    )


def make_v42_candidate_c_config(*, fee_rate: float | None = None) -> V41DefaultConfig:
    """Return v4.2-candidate-C config.

    v4.2-candidate-C starts from v4.2-candidate-A and changes only the controller
    base-exposure mapping by setting ``tau_floor`` to 0.10.
    """

    candidate_a = make_v42_candidate_a_config(fee_rate=fee_rate)
    return V41DefaultConfig(
        version_name=V42_CANDIDATE_C_VERSION_NAME,
        periods_per_year=candidate_a.periods_per_year,
        backtest=candidate_a.backtest,
        filter=candidate_a.filter,
        state_estimator=candidate_a.state_estimator,
        controller=ControllerConfig(
            w_epsilon=candidate_a.controller.w_epsilon,
            w_volatility=candidate_a.controller.w_volatility,
            w_portfolio_risk=candidate_a.controller.w_portfolio_risk,
            max_position_change=candidate_a.controller.max_position_change,
            tau_floor=0.10,
            rebalance_threshold=candidate_a.controller.rebalance_threshold,
            use_trend_persistence_gate=candidate_a.controller.use_trend_persistence_gate,
            tau_confirm_threshold=candidate_a.controller.tau_confirm_threshold,
            trend_persistence_window=candidate_a.controller.trend_persistence_window,
            persistence_floor=candidate_a.controller.persistence_floor,
        ),
    )


def build_v42_candidate_c_strategy(
    config: V41DefaultConfig | None = None,
) -> tuple[MinimalContinuousController, MinimalStateEstimator]:
    """Build controller and estimator for v4.2-candidate-C."""

    resolved = config or make_v42_candidate_c_config()
    return build_v41_controller(resolved), build_v41_estimator(resolved)


def run_v42_candidate_c_backtest(prices, config: V41DefaultConfig | None = None, *, price_column: str = "close"):
    """Run v4.2-candidate-C."""

    resolved = config or make_v42_candidate_c_config()
    controller, estimator = build_v42_candidate_c_strategy(resolved)
    return run_backtest(
        prices,
        controller=controller,
        state_estimator=estimator,
        config=resolved.backtest,
        price_column=price_column,
    )


def make_v42_candidate_d_config(*, fee_rate: float | None = None) -> V41DefaultConfig:
    """Return v4.2-candidate-D config.

    v4.2-candidate-D starts from v4.2-candidate-C and changes only the
    controller rebalance deadband by setting ``rebalance_threshold`` to 0.01.
    """

    candidate_c = make_v42_candidate_c_config(fee_rate=fee_rate)
    return V41DefaultConfig(
        version_name=V42_CANDIDATE_D_VERSION_NAME,
        periods_per_year=candidate_c.periods_per_year,
        backtest=candidate_c.backtest,
        filter=candidate_c.filter,
        state_estimator=candidate_c.state_estimator,
        controller=ControllerConfig(
            w_epsilon=candidate_c.controller.w_epsilon,
            w_volatility=candidate_c.controller.w_volatility,
            w_portfolio_risk=candidate_c.controller.w_portfolio_risk,
            max_position_change=candidate_c.controller.max_position_change,
            tau_floor=candidate_c.controller.tau_floor,
            rebalance_threshold=0.01,
            use_trend_persistence_gate=candidate_c.controller.use_trend_persistence_gate,
            tau_confirm_threshold=candidate_c.controller.tau_confirm_threshold,
            trend_persistence_window=candidate_c.controller.trend_persistence_window,
            persistence_floor=candidate_c.controller.persistence_floor,
        ),
    )


def build_v42_candidate_d_strategy(
    config: V41DefaultConfig | None = None,
) -> tuple[MinimalContinuousController, MinimalStateEstimator]:
    """Build controller and estimator for v4.2-candidate-D."""

    resolved = config or make_v42_candidate_d_config()
    return build_v41_controller(resolved), build_v41_estimator(resolved)


def run_v42_candidate_d_backtest(prices, config: V41DefaultConfig | None = None, *, price_column: str = "close"):
    """Run v4.2-candidate-D."""

    resolved = config or make_v42_candidate_d_config()
    controller, estimator = build_v42_candidate_d_strategy(resolved)
    return run_backtest(
        prices,
        controller=controller,
        state_estimator=estimator,
        config=resolved.backtest,
        price_column=price_column,
    )


def make_v42_candidate_e_config(*, fee_rate: float | None = None) -> V41DefaultConfig:
    """Return v4.2-candidate-E config.

    v4.2-candidate-E starts from v4.2-candidate-C and changes only the
    controller by enabling the causal trend persistence gate.
    """

    candidate_c = make_v42_candidate_c_config(fee_rate=fee_rate)
    return V41DefaultConfig(
        version_name=V42_CANDIDATE_E_VERSION_NAME,
        periods_per_year=candidate_c.periods_per_year,
        backtest=candidate_c.backtest,
        filter=candidate_c.filter,
        state_estimator=candidate_c.state_estimator,
        controller=ControllerConfig(
            w_epsilon=candidate_c.controller.w_epsilon,
            w_volatility=candidate_c.controller.w_volatility,
            w_portfolio_risk=candidate_c.controller.w_portfolio_risk,
            max_position_change=candidate_c.controller.max_position_change,
            tau_floor=candidate_c.controller.tau_floor,
            rebalance_threshold=candidate_c.controller.rebalance_threshold,
            use_trend_persistence_gate=True,
            tau_confirm_threshold=0.25,
            trend_persistence_window=10,
            persistence_floor=0.50,
        ),
    )


def build_v42_candidate_e_strategy(
    config: V41DefaultConfig | None = None,
) -> tuple[MinimalContinuousController, MinimalStateEstimator]:
    """Build controller and estimator for v4.2-candidate-E."""

    resolved = config or make_v42_candidate_e_config()
    return build_v41_controller(resolved), build_v41_estimator(resolved)


def run_v42_candidate_e_backtest(prices, config: V41DefaultConfig | None = None, *, price_column: str = "close"):
    """Run v4.2-candidate-E."""

    resolved = config or make_v42_candidate_e_config()
    controller, estimator = build_v42_candidate_e_strategy(resolved)
    return run_backtest(
        prices,
        controller=controller,
        state_estimator=estimator,
        config=resolved.backtest,
        price_column=price_column,
    )

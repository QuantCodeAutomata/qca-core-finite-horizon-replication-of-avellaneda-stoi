"""Avellaneda-Stoikov analytical verification package."""

from .frozen_inventory import v_frozen, terminal_value
from .reservation_prices import (
    reservation_ask,
    reservation_bid,
    reservation_price,
)
from .hjb_ansatz import theta_from_v, reservation_from_theta
from .optimal_quotes import (
    foc_residual_ask,
    foc_residual_bid,
    exponential_intensity_simplification,
    liquidity_adjustment,
)
from .asymptotic import (
    asymptotic_spread,
    asymptotic_ask_quote,
    asymptotic_bid_quote,
    asymptotic_delta_a,
    asymptotic_delta_b,
)
from .infinite_horizon import (
    inf_horizon_ask,
    inf_horizon_bid,
    admissibility_holds,
    omega_from_qmax,
)

__all__ = [
    "v_frozen",
    "terminal_value",
    "reservation_ask",
    "reservation_bid",
    "reservation_price",
    "theta_from_v",
    "reservation_from_theta",
    "foc_residual_ask",
    "foc_residual_bid",
    "exponential_intensity_simplification",
    "liquidity_adjustment",
    "asymptotic_spread",
    "asymptotic_ask_quote",
    "asymptotic_bid_quote",
    "asymptotic_delta_a",
    "asymptotic_delta_b",
    "inf_horizon_ask",
    "inf_horizon_bid",
    "admissibility_holds",
    "omega_from_qmax",
]

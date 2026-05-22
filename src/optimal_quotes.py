"""Optimal quote first-order conditions (FOC).

Paper's implicit equations for the optimal half-spreads delta^a, delta^b:

    s - r^b = delta^b - (1/gamma) * ln(1 - gamma * lambda^b(delta^b) / lambda^{b'}(delta^b))
    r^a - s = delta^a - (1/gamma) * ln(1 - gamma * lambda^a(delta^a) / lambda^{a'}(delta^a))

For exponential intensities lambda(delta) = A * exp(-k * delta),
    lambda'(delta) = -k * lambda(delta)
    lambda / lambda' = -1/k
so
    ln(1 - gamma * (-1/k)) = ln(1 + gamma/k)

The optimal quote distance from the reservation price equals
    (1/gamma) * ln(1 + gamma/k)   (per side)

# Custom — Context7 found no library equivalent (paper Section 3, FOC).
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np


# -----------------------------------------------------------------------------
# Generic FOC residual: should be zero at the optimum.
# -----------------------------------------------------------------------------
def foc_residual_ask(
    delta_a: float,
    r_a: float,
    s: float,
    gamma: float,
    lam: Callable[[float], float],
    lam_prime: Callable[[float], float],
) -> float:
    """Residual of the ask FOC:
        (r^a - s) - delta_a + (1/gamma) * ln(1 - gamma * lam(delta)/lam'(delta))
    Should equal zero at the optimum.
    """
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    arg = 1.0 - gamma * lam(delta_a) / lam_prime(delta_a)
    if arg <= 0:
        raise ValueError("Log argument is non-positive — FOC not well-defined.")
    return (r_a - s) - delta_a + (1.0 / gamma) * math.log(arg)


def foc_residual_bid(
    delta_b: float,
    r_b: float,
    s: float,
    gamma: float,
    lam: Callable[[float], float],
    lam_prime: Callable[[float], float],
) -> float:
    """Residual of the bid FOC:
        (s - r^b) - delta_b + (1/gamma) * ln(1 - gamma * lam(delta)/lam'(delta))
    """
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    arg = 1.0 - gamma * lam(delta_b) / lam_prime(delta_b)
    if arg <= 0:
        raise ValueError("Log argument is non-positive — FOC not well-defined.")
    return (s - r_b) - delta_b + (1.0 / gamma) * math.log(arg)


# -----------------------------------------------------------------------------
# Exponential intensity simplification.
# -----------------------------------------------------------------------------
def liquidity_adjustment(gamma: float, k: float) -> float:
    """The (1/gamma) * ln(1 + gamma/k) term per side."""
    if gamma <= 0 or k <= 0:
        raise ValueError("gamma and k must be > 0")
    return (1.0 / gamma) * math.log1p(gamma / k)


def exponential_intensity_simplification(
    gamma: float, k: float
) -> dict:
    """For lambda(delta) = A*exp(-k*delta), return the constants entering the FOC.

    Returns
    -------
    dict with keys:
        lam_over_lamprime : lambda/lambda' = -1/k
        log_term          : ln(1 + gamma/k)
        adjustment        : (1/gamma) * ln(1 + gamma/k)
    """
    if gamma <= 0 or k <= 0:
        raise ValueError("gamma and k must be > 0")
    return {
        "lam_over_lamprime": -1.0 / k,
        "log_term": math.log1p(gamma / k),
        "adjustment": liquidity_adjustment(gamma, k),
    }


def exponential_intensity(delta: float, A: float, k: float) -> float:
    """lambda(delta) = A * exp(-k * delta)."""
    return A * math.exp(-k * delta)


def exponential_intensity_prime(delta: float, A: float, k: float) -> float:
    """lambda'(delta) = -k * A * exp(-k * delta)."""
    return -k * A * math.exp(-k * delta)


# -----------------------------------------------------------------------------
# Closed-form optimal quote distances under the asymptotic approximation.
# -----------------------------------------------------------------------------
def optimal_quote_distance(
    side: str,
    q: float,
    t: float,
    gamma: float,
    sigma: float,
    T: float,
    k: float,
) -> float:
    """Asymptotic optimal half-spread distance from mid:

        delta^a = (0.5 - q) * gamma * sigma^2 * (T - t) + (1/gamma) * ln(1 + gamma/k)
        delta^b = (0.5 + q) * gamma * sigma^2 * (T - t) + (1/gamma) * ln(1 + gamma/k)

    Parameters
    ----------
    side : 'a' for ask, 'b' for bid.
    """
    if side not in ("a", "b"):
        raise ValueError("side must be 'a' or 'b'")
    tau = T - t
    sign = -1.0 if side == "a" else 1.0
    return (0.5 + sign * q) * gamma * sigma**2 * tau + liquidity_adjustment(
        gamma, k
    )

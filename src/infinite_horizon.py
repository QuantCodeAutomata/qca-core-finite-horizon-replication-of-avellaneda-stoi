"""Infinite-horizon reservation prices (paper Section 4).

Discounted objective:
    \\bar v(x, s, q) = E[ \\int_0^\\infty -exp(-omega t) exp(-gamma (x + q S_t)) dt ]

Closed-form reservation prices:

    \\bar r^a(s, q) = s + (1/gamma) * ln(1 + ((1 - 2q) * gamma^2 * sigma^2) / (2 omega - gamma^2 q^2 sigma^2))
    \\bar r^b(s, q) = s + (1/gamma) * ln(1 + ((-1 - 2q) * gamma^2 * sigma^2) / (2 omega - gamma^2 q^2 sigma^2))

Admissibility condition:
    omega > 0.5 * gamma^2 * sigma^2 * q^2.

Paper suggests choosing omega = 0.5 * gamma^2 * sigma^2 * (q_max + 1)^2 so the
condition holds strictly for |q| <= q_max.

# Custom — Context7 found no library equivalent (paper Section 4 closed forms).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def admissibility_holds(
    q: ArrayLike, omega: float, gamma: float, sigma: float
) -> np.ndarray:
    """Boolean array: True where omega > 0.5 * gamma^2 * sigma^2 * q^2."""
    if gamma <= 0 or sigma <= 0:
        raise ValueError("gamma, sigma must be > 0")
    q_a = np.asarray(q, dtype=float)
    return omega > 0.5 * gamma**2 * sigma**2 * q_a**2


def omega_from_qmax(q_max: int, gamma: float, sigma: float) -> float:
    """omega = 0.5 * gamma^2 * sigma^2 * (q_max + 1)^2."""
    if q_max < 0:
        raise ValueError("q_max must be >= 0")
    if gamma <= 0 or sigma <= 0:
        raise ValueError("gamma, sigma must be > 0")
    return 0.5 * gamma**2 * sigma**2 * (q_max + 1) ** 2


def _log_argument(
    coeff_sign: float, q: np.ndarray, omega: float, gamma: float, sigma: float
) -> np.ndarray:
    """Helper computing 1 + (coeff_sign*1 - 2q) gamma^2 sigma^2 / (2 omega - gamma^2 q^2 sigma^2).

    coeff_sign = +1 for ask, -1 for bid.
    """
    num = (coeff_sign - 2.0 * q) * gamma**2 * sigma**2
    den = 2.0 * omega - gamma**2 * q**2 * sigma**2
    return 1.0 + num / den, den


def inf_horizon_ask(
    s: ArrayLike,
    q: ArrayLike,
    omega: float,
    gamma: float,
    sigma: float,
    *,
    strict: bool = True,
) -> np.ndarray:
    """Infinite-horizon reservation ask price ``r̄^a(s, q)``.

    If ``strict=False`` and the log argument is non-positive (post-trade
    inventory ``q-1`` violates admissibility), returns ``+inf`` for those
    points instead of raising. This corresponds to "refuse to sell".
    """
    if gamma <= 0 or sigma <= 0 or omega <= 0:
        raise ValueError("gamma, sigma, omega must be > 0")
    s_a = np.asarray(s, dtype=float)
    q_a = np.asarray(q, dtype=float)

    log_arg, den = _log_argument(+1.0, q_a, omega, gamma, sigma)
    if np.any(den <= 0):
        raise ValueError(
            "Admissibility violated: 2 omega - gamma^2 q^2 sigma^2 must be > 0."
        )
    bad = log_arg <= 0
    if np.any(bad):
        if strict:
            raise ValueError("Log argument non-positive in infinite-horizon ask.")
        out = np.full_like(s_a + q_a * 0.0, np.inf, dtype=float)
        safe = ~bad
        out[safe] = s_a + (1.0 / gamma) * np.log(log_arg[safe])
        return out
    return s_a + (1.0 / gamma) * np.log(log_arg)


def inf_horizon_bid(
    s: ArrayLike,
    q: ArrayLike,
    omega: float,
    gamma: float,
    sigma: float,
    *,
    strict: bool = True,
) -> np.ndarray:
    """Infinite-horizon reservation bid price ``r̄^b(s, q)``.

    If ``strict=False`` and the log argument is non-positive (post-trade
    inventory ``q+1`` violates admissibility), returns ``-inf`` for those
    points instead of raising. This corresponds to "refuse to buy".
    """
    if gamma <= 0 or sigma <= 0 or omega <= 0:
        raise ValueError("gamma, sigma, omega must be > 0")
    s_a = np.asarray(s, dtype=float)
    q_a = np.asarray(q, dtype=float)

    log_arg, den = _log_argument(-1.0, q_a, omega, gamma, sigma)
    if np.any(den <= 0):
        raise ValueError(
            "Admissibility violated: 2 omega - gamma^2 q^2 sigma^2 must be > 0."
        )
    bad = log_arg <= 0
    if np.any(bad):
        if strict:
            raise ValueError("Log argument non-positive in infinite-horizon bid.")
        out = np.full_like(s_a + q_a * 0.0, -np.inf, dtype=float)
        safe = ~bad
        out[safe] = s_a + (1.0 / gamma) * np.log(log_arg[safe])
        return out
    return s_a + (1.0 / gamma) * np.log(log_arg)


def inf_horizon_reservation_price(
    s: ArrayLike, q: ArrayLike, omega: float, gamma: float, sigma: float
) -> np.ndarray:
    """Average infinite-horizon reservation price = (r^a + r^b) / 2."""
    ra = inf_horizon_ask(s, q, omega, gamma, sigma)
    rb = inf_horizon_bid(s, q, omega, gamma, sigma)
    return 0.5 * (ra + rb)

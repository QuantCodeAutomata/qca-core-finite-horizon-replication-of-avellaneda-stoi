"""Finite-horizon reservation bid, ask, and average prices.

Derived from utility indifference under the frozen-inventory value function:

    v(x - r^b, s, q+1, t) = v(x, s, q, t)   ->  r^b(s, q, t) = s + ((-1 - 2q)/2) * gamma * sigma^2 * (T-t)
    v(x + r^a, s, q-1, t) = v(x, s, q, t)   ->  r^a(s, q, t) = s + (( 1 - 2q)/2) * gamma * sigma^2 * (T-t)
    r(s, q, t)            = (r^a + r^b)/2   =   s - q * gamma * sigma^2 * (T-t)

# Custom — Context7 found no library equivalent (paper Section 2.2).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _check(gamma: float, sigma: float, T: float) -> None:
    if gamma <= 0 or sigma <= 0 or T <= 0:
        raise ValueError("gamma, sigma, T must be positive")


def reservation_ask(
    s: ArrayLike,
    q: ArrayLike,
    t: ArrayLike,
    gamma: float,
    sigma: float,
    T: float,
) -> np.ndarray:
    """r^a(s, q, t) = s + ((1 - 2q)/2) * gamma * sigma^2 * (T-t)."""
    _check(gamma, sigma, T)
    s_a = np.asarray(s, dtype=float)
    q_a = np.asarray(q, dtype=float)
    t_a = np.asarray(t, dtype=float)
    tau = T - t_a
    return s_a + ((1.0 - 2.0 * q_a) / 2.0) * gamma * sigma**2 * tau


def reservation_bid(
    s: ArrayLike,
    q: ArrayLike,
    t: ArrayLike,
    gamma: float,
    sigma: float,
    T: float,
) -> np.ndarray:
    """r^b(s, q, t) = s + ((-1 - 2q)/2) * gamma * sigma^2 * (T-t)."""
    _check(gamma, sigma, T)
    s_a = np.asarray(s, dtype=float)
    q_a = np.asarray(q, dtype=float)
    t_a = np.asarray(t, dtype=float)
    tau = T - t_a
    return s_a + ((-1.0 - 2.0 * q_a) / 2.0) * gamma * sigma**2 * tau


def reservation_price(
    s: ArrayLike,
    q: ArrayLike,
    t: ArrayLike,
    gamma: float,
    sigma: float,
    T: float,
) -> np.ndarray:
    """Average reservation price r(s, q, t) = s - q * gamma * sigma^2 * (T-t)."""
    _check(gamma, sigma, T)
    s_a = np.asarray(s, dtype=float)
    q_a = np.asarray(q, dtype=float)
    t_a = np.asarray(t, dtype=float)
    tau = T - t_a
    return s_a - q_a * gamma * sigma**2 * tau


def reservation_spread(
    t: ArrayLike, gamma: float, sigma: float, T: float
) -> np.ndarray:
    """r^a - r^b = gamma * sigma^2 * (T-t) (independent of q and s)."""
    _check(gamma, sigma, T)
    t_a = np.asarray(t, dtype=float)
    return gamma * sigma**2 * (T - t_a)

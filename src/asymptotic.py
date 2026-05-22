"""Asymptotic expansion in inventory q (paper Section 3.2).

Using the ansatz theta(s, q, t) = theta^0 + q*theta^1 + 0.5 q^2 theta^2 + ...
and the exponential-intensity FOC, the paper obtains the closed-form approximation:

    delta^a + delta^b = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/k)

    p^a_t = r_t + 0.5 * (gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/k))
    p^b_t = r_t - 0.5 * (gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/k))

    delta^a_t = (0.5 - q_t) * gamma * sigma^2 * (T - t) + (1/gamma) * ln(1 + gamma/k)
    delta^b_t = (0.5 + q_t) * gamma * sigma^2 * (T - t) + (1/gamma) * ln(1 + gamma/k)

# Custom — Context7 found no library equivalent (paper Section 3.2 closed forms).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from .reservation_prices import reservation_price


def asymptotic_spread(
    t: ArrayLike, gamma: float, sigma: float, T: float, k: float
) -> np.ndarray:
    """delta^a + delta^b = gamma sigma^2 (T-t) + (2/gamma) ln(1+gamma/k)."""
    if gamma <= 0 or sigma <= 0 or T <= 0 or k <= 0:
        raise ValueError("gamma, sigma, T, k must be > 0")
    t_a = np.asarray(t, dtype=float)
    tau = T - t_a
    return gamma * sigma**2 * tau + (2.0 / gamma) * np.log1p(gamma / k)


def asymptotic_ask_quote(
    s: ArrayLike,
    q: ArrayLike,
    t: ArrayLike,
    gamma: float,
    sigma: float,
    T: float,
    k: float,
) -> np.ndarray:
    """p^a_t = r_t + 0.5 * (gamma sigma^2 (T-t) + (2/gamma) ln(1+gamma/k))."""
    r = reservation_price(s, q, t, gamma, sigma, T)
    spread = asymptotic_spread(t, gamma, sigma, T, k)
    return r + 0.5 * spread


def asymptotic_bid_quote(
    s: ArrayLike,
    q: ArrayLike,
    t: ArrayLike,
    gamma: float,
    sigma: float,
    T: float,
    k: float,
) -> np.ndarray:
    """p^b_t = r_t - 0.5 * (gamma sigma^2 (T-t) + (2/gamma) ln(1+gamma/k))."""
    r = reservation_price(s, q, t, gamma, sigma, T)
    spread = asymptotic_spread(t, gamma, sigma, T, k)
    return r - 0.5 * spread


def asymptotic_delta_a(
    q: ArrayLike,
    t: ArrayLike,
    gamma: float,
    sigma: float,
    T: float,
    k: float,
) -> np.ndarray:
    """delta^a_t = (0.5 - q) * gamma * sigma^2 * (T-t) + (1/gamma) * ln(1 + gamma/k)."""
    q_a = np.asarray(q, dtype=float)
    t_a = np.asarray(t, dtype=float)
    tau = T - t_a
    return (0.5 - q_a) * gamma * sigma**2 * tau + (1.0 / gamma) * np.log1p(
        gamma / k
    )


def asymptotic_delta_b(
    q: ArrayLike,
    t: ArrayLike,
    gamma: float,
    sigma: float,
    T: float,
    k: float,
) -> np.ndarray:
    """delta^b_t = (0.5 + q) * gamma * sigma^2 * (T-t) + (1/gamma) * ln(1 + gamma/k)."""
    q_a = np.asarray(q, dtype=float)
    t_a = np.asarray(t, dtype=float)
    tau = T - t_a
    return (0.5 + q_a) * gamma * sigma**2 * tau + (1.0 / gamma) * np.log1p(
        gamma / k
    )


def small_gamma_spread_limit(k: float) -> float:
    """Limit of (2/gamma) ln(1 + gamma/k) as gamma -> 0 equals 2/k."""
    if k <= 0:
        raise ValueError("k must be > 0")
    return 2.0 / k

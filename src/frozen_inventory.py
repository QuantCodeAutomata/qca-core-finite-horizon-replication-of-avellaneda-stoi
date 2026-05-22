"""Frozen-inventory exponential-utility value function.

Avellaneda & Stoikov (2008), Section 2 — frozen inventory problem.

Under dS_u = sigma * dW_u, S_t = s, the value function
    v(x, s, q, t) = E_t[-exp(-gamma * (x + q * S_T))]
admits the closed form

    v(x, s, q, t) = -exp(-gamma * x) * exp(-gamma * q * s)
                    * exp(0.5 * gamma^2 * q^2 * sigma^2 * (T - t)).

# Custom — Context7 found no library equivalent (paper Section 2 eq. for v).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _validate_inputs(sigma: float, T: float, gamma: float) -> None:
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if T <= 0:
        raise ValueError("T must be > 0")
    if gamma <= 0:
        raise ValueError("gamma must be > 0")


def v_frozen(
    x: ArrayLike,
    s: ArrayLike,
    q: ArrayLike,
    t: ArrayLike,
    gamma: float,
    sigma: float,
    T: float,
) -> np.ndarray:
    """Closed-form frozen-inventory value function v(x, s, q, t).

    Parameters
    ----------
    x, s, q, t : array-like
        Wealth, mid-price, inventory, time. Broadcast together.
    gamma : float
        Risk aversion (>0).
    sigma : float
        Mid-price volatility (>0).
    T : float
        Terminal time.

    Returns
    -------
    np.ndarray
        Value of v(x, s, q, t).
    """
    _validate_inputs(sigma, T, gamma)
    x_a = np.asarray(x, dtype=float)
    s_a = np.asarray(s, dtype=float)
    q_a = np.asarray(q, dtype=float)
    t_a = np.asarray(t, dtype=float)

    tau = T - t_a
    if np.any(tau < -1e-15):
        raise ValueError("t must satisfy t <= T")
    # Clamp tiny negative due to FP
    tau = np.maximum(tau, 0.0)

    exponent = (
        -gamma * x_a
        - gamma * q_a * s_a
        + 0.5 * gamma**2 * q_a**2 * sigma**2 * tau
    )
    return -np.exp(exponent)


def terminal_value(
    x: ArrayLike, s: ArrayLike, q: ArrayLike, gamma: float
) -> np.ndarray:
    """Terminal utility -exp(-gamma * (x + q*s))."""
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    x_a = np.asarray(x, dtype=float)
    s_a = np.asarray(s, dtype=float)
    q_a = np.asarray(q, dtype=float)
    return -np.exp(-gamma * (x_a + q_a * s_a))

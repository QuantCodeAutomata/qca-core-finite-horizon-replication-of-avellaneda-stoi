"""HJB ansatz u(s, x, q, t) = -exp(-gamma * x) exp(-gamma * theta(s, q, t)).

Under this ansatz, the indifference equations for reservation prices reduce to

    r^b(s, q, t) = theta(s, q+1, t) - theta(s, q, t)
    r^a(s, q, t) = theta(s, q, t) - theta(s, q-1, t)

For the frozen-inventory problem,
    u(s, x, q, t) = v(x, s, q, t)
                  = -exp(-gamma * x) * exp(-gamma * q * s + 0.5 * gamma^2 q^2 sigma^2 (T-t))
so
    theta(s, q, t) = q * s - 0.5 * gamma * q^2 * sigma^2 * (T - t).

# Custom — Context7 found no library equivalent (paper Section 3, ansatz).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def theta_from_v(
    s: ArrayLike,
    q: ArrayLike,
    t: ArrayLike,
    gamma: float,
    sigma: float,
    T: float,
) -> np.ndarray:
    """Closed-form theta for the frozen-inventory value function.

    theta(s, q, t) = q*s - 0.5 * gamma * q^2 * sigma^2 * (T - t)
    """
    if gamma <= 0 or sigma <= 0 or T <= 0:
        raise ValueError("gamma, sigma, T must be positive")
    s_a = np.asarray(s, dtype=float)
    q_a = np.asarray(q, dtype=float)
    t_a = np.asarray(t, dtype=float)
    tau = T - t_a
    return q_a * s_a - 0.5 * gamma * q_a**2 * sigma**2 * tau


def reservation_from_theta(
    theta_fn,
    s: float,
    q: int,
    t: float,
) -> tuple[float, float]:
    """Compute (r^a, r^b) from a theta(s, q, t) callable via the ansatz.

    Parameters
    ----------
    theta_fn : callable(s, q, t) -> float
    s, q, t  : evaluation point.

    Returns
    -------
    (r_a, r_b)
        r^a = theta(s, q, t)   - theta(s, q-1, t)
        r^b = theta(s, q+1, t) - theta(s, q, t)
    """
    th_q = float(theta_fn(s, q, t))
    th_qp1 = float(theta_fn(s, q + 1, t))
    th_qm1 = float(theta_fn(s, q - 1, t))
    r_a = th_q - th_qm1
    r_b = th_qp1 - th_q
    return r_a, r_b

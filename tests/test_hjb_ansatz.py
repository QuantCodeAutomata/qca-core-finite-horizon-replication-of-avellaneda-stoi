"""Tests for the HJB exponential-utility ansatz."""

import numpy as np
import pytest

from src.hjb_ansatz import reservation_from_theta, theta_from_v
from src.reservation_prices import reservation_ask, reservation_bid

SIGMA = 2.0
T = 1.0


@pytest.mark.parametrize("gamma", [0.01, 0.1, 0.5])
def test_theta_matches_reservation_via_ansatz(gamma: float) -> None:
    theta = lambda s, q, t, g=gamma: float(theta_from_v(s, q, t, g, SIGMA, T))
    for q in range(-5, 6):
        for t in np.linspace(0, T, 6):
            r_a_th, r_b_th = reservation_from_theta(theta, 100.0, int(q), float(t))
            r_a_direct = float(reservation_ask(100.0, q, t, gamma, SIGMA, T))
            r_b_direct = float(reservation_bid(100.0, q, t, gamma, SIGMA, T))
            assert np.isclose(r_a_th, r_a_direct, atol=1e-12)
            assert np.isclose(r_b_th, r_b_direct, atol=1e-12)


def test_theta_zero_at_q_zero() -> None:
    val = theta_from_v(100.0, 0, 0.5, 0.1, SIGMA, T)
    assert np.isclose(val, 0.0, atol=1e-12)


def test_theta_quadratic_in_q() -> None:
    """theta - q*s should be quadratic in q (parabola)."""
    q = np.arange(-10, 11)
    th = theta_from_v(0.0, q, 0.0, 0.1, SIGMA, T)
    # theta(s=0, q, t=0) = -0.5 * gamma * q^2 * sigma^2 * T
    expected = -0.5 * 0.1 * q**2 * SIGMA**2 * T
    assert np.allclose(th, expected, atol=1e-12)

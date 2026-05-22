"""Tests for the infinite-horizon reservation-price formulas."""

import math

import numpy as np
import pytest

from src.infinite_horizon import (
    admissibility_holds,
    inf_horizon_ask,
    inf_horizon_bid,
    omega_from_qmax,
)

S = 100.0
SIGMA = 2.0


@pytest.mark.parametrize("gamma", [0.01, 0.1, 0.5])
def test_omega_admissibility(gamma: float) -> None:
    q_max = 10
    omega = omega_from_qmax(q_max, gamma, SIGMA)
    q_grid = np.arange(-q_max, q_max + 1)
    assert admissibility_holds(q_grid, omega, gamma, SIGMA).all()


@pytest.mark.parametrize("gamma", [0.01, 0.1, 0.5])
def test_q_zero_ordering(gamma: float) -> None:
    """At q = 0: r^a_inf > s > r^b_inf."""
    omega = omega_from_qmax(10, gamma, SIGMA)
    ra = inf_horizon_ask(S, 0, omega, gamma, SIGMA)
    rb = inf_horizon_bid(S, 0, omega, gamma, SIGMA)
    assert ra > S > rb


@pytest.mark.parametrize("gamma", [0.01, 0.1, 0.5])
def test_ask_decreases_with_q(gamma: float) -> None:
    """r^a should be monotone decreasing in q (more long inventory -> lower quote)."""
    omega = omega_from_qmax(10, gamma, SIGMA)
    q_grid = np.arange(-10, 11)
    ra = inf_horizon_ask(S, q_grid, omega, gamma, SIGMA)
    assert np.all(np.diff(ra) < 0)


def test_admissibility_violation_raises() -> None:
    """Choosing omega below the admissibility bound must raise."""
    gamma, q = 0.1, 5
    bound = 0.5 * gamma**2 * SIGMA**2 * q**2
    bad_omega = 0.99 * bound
    with pytest.raises(ValueError):
        inf_horizon_ask(S, q, bad_omega, gamma, SIGMA)
    with pytest.raises(ValueError):
        inf_horizon_bid(S, q, bad_omega, gamma, SIGMA)


@pytest.mark.parametrize("gamma", [0.01, 0.1, 0.5])
def test_inf_horizon_explicit_formula(gamma: float) -> None:
    """Verify formula directly: r^a = s + (1/gamma) ln(1 + (1-2q) gamma^2 sigma^2 / (2 omega - gamma^2 q^2 sigma^2))."""
    omega = omega_from_qmax(10, gamma, SIGMA)
    for q in (-3, 0, 2, 5):
        num_a = (1 - 2 * q) * gamma**2 * SIGMA**2
        num_b = (-1 - 2 * q) * gamma**2 * SIGMA**2
        den = 2 * omega - gamma**2 * q**2 * SIGMA**2
        expected_a = S + (1 / gamma) * math.log1p(num_a / den)
        expected_b = S + (1 / gamma) * math.log1p(num_b / den)
        ra = inf_horizon_ask(S, q, omega, gamma, SIGMA)
        rb = inf_horizon_bid(S, q, omega, gamma, SIGMA)
        assert math.isclose(float(ra), expected_a, rel_tol=1e-13, abs_tol=1e-13)
        assert math.isclose(float(rb), expected_b, rel_tol=1e-13, abs_tol=1e-13)

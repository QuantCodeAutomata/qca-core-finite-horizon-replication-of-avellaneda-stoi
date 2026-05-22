"""Tests for the finite-horizon reservation prices."""

import numpy as np
import pytest

from src.frozen_inventory import v_frozen
from src.reservation_prices import (
    reservation_ask,
    reservation_bid,
    reservation_price,
    reservation_spread,
)

S = 100.0
SIGMA = 2.0
T = 1.0


@pytest.mark.parametrize("gamma", [0.01, 0.1, 0.5])
def test_zero_inventory_collapses_to_mid(gamma: float) -> None:
    t = np.linspace(0, T, 21)
    r = reservation_price(S, 0, t, gamma, SIGMA, T)
    assert np.allclose(r, S, atol=1e-12)


@pytest.mark.parametrize("gamma", [0.01, 0.1, 0.5])
def test_linear_in_q(gamma: float) -> None:
    q = np.arange(-10, 11)
    r = reservation_price(S, q, 0.0, gamma, SIGMA, T)
    slope = np.diff(r)
    expected = -gamma * SIGMA**2 * T
    assert np.allclose(slope, expected, atol=1e-12)


@pytest.mark.parametrize("gamma", [0.01, 0.1, 0.5])
def test_terminal_collapse(gamma: float) -> None:
    """As t -> T, all reservation prices converge to s."""
    for q in (-5, -1, 0, 1, 5):
        for name, fn in [
            ("ask", reservation_ask),
            ("bid", reservation_bid),
            ("mid", reservation_price),
        ]:
            val = fn(S, q, T, gamma, SIGMA, T)
            assert np.isclose(val, S, atol=1e-14), f"{name} failed for q={q}"


@pytest.mark.parametrize("gamma", [0.01, 0.1, 0.5])
def test_bid_ask_gap(gamma: float) -> None:
    t = np.linspace(0, T, 11)
    ra = reservation_ask(S, 2, t, gamma, SIGMA, T)
    rb = reservation_bid(S, 2, t, gamma, SIGMA, T)
    expected = gamma * SIGMA**2 * (T - t)
    assert np.allclose(ra - rb, expected, atol=1e-12)
    assert np.allclose(reservation_spread(t, gamma, SIGMA, T), expected, atol=1e-12)


@pytest.mark.parametrize("gamma", [0.01, 0.1, 0.5])
@pytest.mark.parametrize("q", [-5, -1, 0, 2, 7])
def test_indifference_ask(gamma: float, q: int) -> None:
    """v(x + r^a, s, q-1, t) == v(x, s, q, t)."""
    x = 0.42
    for t in np.linspace(0, T, 6):
        ra = reservation_ask(S, q, t, gamma, SIGMA, T)
        lhs = v_frozen(x + ra, S, q - 1, t, gamma, SIGMA, T)
        rhs = v_frozen(x, S, q, t, gamma, SIGMA, T)
        assert np.isclose(lhs, rhs, atol=1e-8, rtol=1e-10)


@pytest.mark.parametrize("gamma", [0.01, 0.1, 0.5])
@pytest.mark.parametrize("q", [-5, -1, 0, 2, 7])
def test_indifference_bid(gamma: float, q: int) -> None:
    """v(x - r^b, s, q+1, t) == v(x, s, q, t)."""
    x = 0.42
    for t in np.linspace(0, T, 6):
        rb = reservation_bid(S, q, t, gamma, SIGMA, T)
        lhs = v_frozen(x - rb, S, q + 1, t, gamma, SIGMA, T)
        rhs = v_frozen(x, S, q, t, gamma, SIGMA, T)
        assert np.isclose(lhs, rhs, atol=1e-8, rtol=1e-10)


def test_average_equals_mean_of_ask_bid() -> None:
    q = np.arange(-5, 6)
    t = np.linspace(0, T, 7)[:, None]
    ra = reservation_ask(S, q, t, 0.1, SIGMA, T)
    rb = reservation_bid(S, q, t, 0.1, SIGMA, T)
    r = reservation_price(S, q, t, 0.1, SIGMA, T)
    assert np.allclose(0.5 * (ra + rb), r, atol=1e-12)

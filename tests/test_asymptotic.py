"""Tests for the asymptotic (q-expansion) closed-form spreads and quotes."""

import math

import numpy as np
import pytest

from src.asymptotic import (
    asymptotic_ask_quote,
    asymptotic_bid_quote,
    asymptotic_delta_a,
    asymptotic_delta_b,
    asymptotic_spread,
    small_gamma_spread_limit,
)
from src.reservation_prices import reservation_price

SIGMA = 2.0
T = 1.0
K = 1.5
S = 100.0


@pytest.mark.parametrize("gamma", [0.01, 0.1, 0.5])
def test_spread_formula(gamma: float) -> None:
    t = np.linspace(0, T, 11)
    sp = asymptotic_spread(t, gamma, SIGMA, T, K)
    expected = gamma * SIGMA**2 * (T - t) + (2.0 / gamma) * np.log1p(gamma / K)
    assert np.allclose(sp, expected, atol=1e-12)


@pytest.mark.parametrize("gamma", [0.01, 0.1, 0.5])
def test_delta_sum_equals_spread(gamma: float) -> None:
    """delta^a + delta^b = spread (and inventory term cancels)."""
    t = 0.3
    for q in (-5, -1, 0, 1, 5):
        da = asymptotic_delta_a(q, t, gamma, SIGMA, T, K)
        db = asymptotic_delta_b(q, t, gamma, SIGMA, T, K)
        total = da + db
        expected = asymptotic_spread(t, gamma, SIGMA, T, K)
        assert math.isclose(float(total), float(expected), rel_tol=1e-12)


@pytest.mark.parametrize("gamma", [0.01, 0.1, 0.5])
def test_quotes_centered_on_reservation(gamma: float) -> None:
    """(p^a + p^b)/2 == r and p^a - p^b == spread."""
    for q in (-3, 0, 3):
        for t in (0.0, 0.5, 0.9):
            pa = asymptotic_ask_quote(S, q, t, gamma, SIGMA, T, K)
            pb = asymptotic_bid_quote(S, q, t, gamma, SIGMA, T, K)
            r = reservation_price(S, q, t, gamma, SIGMA, T)
            assert math.isclose(float(0.5 * (pa + pb)), float(r), abs_tol=1e-12)
            assert math.isclose(float(pa - pb), float(asymptotic_spread(t, gamma, SIGMA, T, K)), abs_tol=1e-12)


def test_small_gamma_limit() -> None:
    """spread term -> 2/k as gamma -> 0."""
    target = small_gamma_spread_limit(K)
    assert math.isclose(target, 2.0 / K)
    prev = float("inf")
    for g in (0.5, 0.1, 0.01, 1e-3, 1e-4):
        val = (2.0 / g) * math.log1p(g / K)
        gap = abs(val - target)
        assert gap < prev  # monotonically shrinking
        prev = gap
    assert prev < 1e-4


@pytest.mark.parametrize("gamma", [0.01, 0.1, 0.5])
def test_delta_inventory_structure(gamma: float) -> None:
    """delta^a = (1/2 - q) gamma sigma^2 (T-t) + (1/gamma) ln(1+gamma/k)."""
    t = 0.4
    q = 2
    da = asymptotic_delta_a(q, t, gamma, SIGMA, T, K)
    db = asymptotic_delta_b(q, t, gamma, SIGMA, T, K)
    adj = (1.0 / gamma) * math.log1p(gamma / K)
    expected_a = (0.5 - q) * gamma * SIGMA**2 * (T - t) + adj
    expected_b = (0.5 + q) * gamma * SIGMA**2 * (T - t) + adj
    assert math.isclose(float(da), expected_a, abs_tol=1e-12)
    assert math.isclose(float(db), expected_b, abs_tol=1e-12)

"""Tests for optimal-quote FOC under exponential intensities."""

import math

import pytest

from src.optimal_quotes import (
    exponential_intensity,
    exponential_intensity_prime,
    exponential_intensity_simplification,
    foc_residual_ask,
    foc_residual_bid,
    liquidity_adjustment,
    optimal_quote_distance,
)
from src.reservation_prices import reservation_ask, reservation_bid

S = 100.0
SIGMA = 2.0
T = 1.0
A = 140.0
K = 1.5


@pytest.mark.parametrize("gamma", [0.001, 0.01, 0.1, 0.5, 1.0])
def test_lam_over_lamprime_equals_minus_1_over_k(gamma: float) -> None:
    info = exponential_intensity_simplification(gamma, K)
    assert math.isclose(info["lam_over_lamprime"], -1.0 / K, rel_tol=1e-15)


@pytest.mark.parametrize("gamma", [0.001, 0.01, 0.1, 0.5, 1.0])
def test_log_term_is_log1p_gamma_over_k(gamma: float) -> None:
    info = exponential_intensity_simplification(gamma, K)
    assert math.isclose(info["log_term"], math.log1p(gamma / K), rel_tol=1e-15)


@pytest.mark.parametrize("gamma", [0.01, 0.1, 0.5])
def test_foc_residual_zero_on_exponential_intensity(gamma: float) -> None:
    """At the asymptotic-optimal delta (q=0, t=0) the FOC residual vanishes."""
    delta_a = optimal_quote_distance("a", 0, 0.0, gamma, SIGMA, T, K)
    delta_b = optimal_quote_distance("b", 0, 0.0, gamma, SIGMA, T, K)
    r_a = float(reservation_ask(S, 0, 0.0, gamma, SIGMA, T))
    r_b = float(reservation_bid(S, 0, 0.0, gamma, SIGMA, T))
    lam = lambda d: exponential_intensity(d, A, K)
    lam_p = lambda d: exponential_intensity_prime(d, A, K)
    assert abs(foc_residual_ask(delta_a, r_a, S, gamma, lam, lam_p)) < 1e-12
    assert abs(foc_residual_bid(delta_b, r_b, S, gamma, lam, lam_p)) < 1e-12


@pytest.mark.parametrize(
    "gamma,expected",
    [(0.1, 1.29), (0.01, 1.33), (0.5, 1.15)],
)
def test_paper_table_spreads(gamma: float, expected: float) -> None:
    """Verify the paper-table spread values to two decimal places."""
    val = (2.0 / gamma) * math.log1p(gamma / K)
    assert round(val, 2) == expected, f"gamma={gamma}: got {val}"


def test_liquidity_adjustment_invalid() -> None:
    with pytest.raises(ValueError):
        liquidity_adjustment(-0.1, K)
    with pytest.raises(ValueError):
        liquidity_adjustment(0.1, 0.0)

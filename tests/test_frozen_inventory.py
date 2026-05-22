"""Tests for the frozen-inventory value function."""

import math

import numpy as np
import pytest

from src.frozen_inventory import terminal_value, v_frozen


SIGMA = 2.0
T = 1.0


def test_terminal_condition_exact() -> None:
    """At t = T, v collapses to -exp(-gamma(x+qs))."""
    for gamma in (0.01, 0.1, 0.5):
        for q in range(-10, 11):
            for x in (-2.5, 0.0, 1.234, 7.0):
                for s in (90.0, 100.0, 110.0):
                    v = v_frozen(x, s, q, T, gamma, SIGMA, T)
                    expected = terminal_value(x, s, q, gamma)
                    assert math.isclose(float(v), float(expected), rel_tol=1e-12, abs_tol=1e-12)


def test_value_function_negative() -> None:
    """v is the expectation of a negative utility => v < 0."""
    v = v_frozen(0.0, 100.0, 3, 0.5, 0.1, SIGMA, T)
    assert v < 0


def test_value_function_increases_with_gamma_squared_term() -> None:
    """For q != 0 and t < T, v < terminal value (more risk)."""
    x, s, q, t = 0.0, 100.0, 5, 0.0
    gamma = 0.1
    v_now = v_frozen(x, s, q, t, gamma, SIGMA, T)
    v_term = terminal_value(x, s, q, gamma)
    assert v_now < v_term  # more negative (less utility)


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        v_frozen(0, 100, 0, 0, gamma=-0.1, sigma=SIGMA, T=T)
    with pytest.raises(ValueError):
        v_frozen(0, 100, 0, 0, gamma=0.1, sigma=0.0, T=T)
    with pytest.raises(ValueError):
        v_frozen(0, 100, 0, 0, gamma=0.1, sigma=SIGMA, T=0.0)
    with pytest.raises(ValueError):
        v_frozen(0, 100, 0, 2.0, gamma=0.1, sigma=SIGMA, T=T)  # t > T


def test_broadcasting() -> None:
    q = np.arange(-3, 4)
    t = np.linspace(0, 1, 5).reshape(-1, 1)
    v = v_frozen(0.0, 100.0, q, t, 0.1, SIGMA, T)
    assert v.shape == (5, 7)

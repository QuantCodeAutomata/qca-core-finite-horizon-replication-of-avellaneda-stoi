"""End-to-end analytical verification of Avellaneda-Stoikov formulas.

Produces:
  - results/tables/*.csv          numerical tables
  - results/figures/*.png         plots (matplotlib)
  - results/RESULTS.md            summary of all identity checks
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.asymptotic import (
    asymptotic_ask_quote,
    asymptotic_bid_quote,
    asymptotic_delta_a,
    asymptotic_delta_b,
    asymptotic_spread,
    small_gamma_spread_limit,
)
from src.frozen_inventory import terminal_value, v_frozen
from src.hjb_ansatz import reservation_from_theta, theta_from_v
from src.infinite_horizon import (
    admissibility_holds,
    inf_horizon_ask,
    inf_horizon_bid,
    omega_from_qmax,
)
from src.optimal_quotes import (
    exponential_intensity,
    exponential_intensity_prime,
    exponential_intensity_simplification,
    foc_residual_ask,
    foc_residual_bid,
    liquidity_adjustment,
    optimal_quote_distance,
)
from src.reservation_prices import (
    reservation_ask,
    reservation_bid,
    reservation_price,
    reservation_spread,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
TABLES = RESULTS / "tables"
FIGS.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Baseline parameters (per paper).
# -----------------------------------------------------------------------------
S0 = 100.0
SIGMA = 2.0
T_END = 1.0
A = 140.0
K = 1.5
GAMMAS = [0.01, 0.1, 0.5]
SMALL_GAMMAS = [0.5, 0.1, 0.01, 0.001, 1e-4]
Q_GRID = np.arange(-10, 11)         # {-10,...,10}
T_GRID = np.linspace(0.0, T_END, 101)
Q_MAX = 10

TOL_STRICT = 1e-10
TOL_LOOSE = 1e-8

results_summary: list[dict] = []


def _record(name: str, passed: bool, detail: str = "") -> None:
    results_summary.append(
        {"check": name, "passed": bool(passed), "detail": detail}
    )
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}: {detail}")


# =============================================================================
# 1) Frozen-inventory value function — terminal condition.
# =============================================================================
def check_frozen_terminal() -> None:
    print("\n=== (1) Frozen-inventory: terminal condition ===")
    x_vals = np.array([0.0, 1.0, -2.5, 10.0])
    s_vals = np.array([100.0, 95.0, 110.0, 100.0])
    max_rel = 0.0
    for gamma in GAMMAS:
        for q in Q_GRID:
            for x, s in zip(x_vals, s_vals):
                v_T = v_frozen(x, s, q, T_END, gamma, SIGMA, T_END)
                term = terminal_value(x, s, q, gamma)
                # Use relative tolerance — these are exponentials that can be very large
                denom = max(abs(float(term)), 1.0)
                rel = float(np.abs(v_T - term)) / denom
                max_rel = max(max_rel, rel)
                assert rel < TOL_STRICT, (
                    f"Terminal condition failed (gamma={gamma}, q={q}): rel={rel}"
                )
    _record(
        "Frozen-inventory terminal condition v(x,s,q,T) = -exp(-gamma(x+qs))",
        True,
        f"max relative error = {max_rel:.3e} on full grid",
    )


# =============================================================================
# 2) Reservation prices by indifference.
# =============================================================================
def check_reservation_indifference() -> None:
    print("\n=== (2) Reservation prices via utility indifference ===")
    max_rel = 0.0
    for gamma in GAMMAS:
        for q in Q_GRID:
            for t in T_GRID:
                r_a = reservation_ask(S0, q, t, gamma, SIGMA, T_END)
                r_b = reservation_bid(S0, q, t, gamma, SIGMA, T_END)
                # Indifference: v(x - r^b, s, q+1, t) == v(x, s, q, t)
                x = 1.234
                lhs_b = v_frozen(x - r_b, S0, q + 1, t, gamma, SIGMA, T_END)
                rhs = v_frozen(x, S0, q, t, gamma, SIGMA, T_END)
                # Indifference for ask:
                lhs_a = v_frozen(x + r_a, S0, q - 1, t, gamma, SIGMA, T_END)
                denom = max(abs(float(rhs)), 1.0)
                rel_b = float(np.abs(lhs_b - rhs)) / denom
                rel_a = float(np.abs(lhs_a - rhs)) / denom
                max_rel = max(max_rel, rel_a, rel_b)
    assert max_rel < TOL_LOOSE, f"Indifference identity failed: rel={max_rel}"
    _record(
        "Reservation bid/ask satisfy utility indifference",
        True,
        f"max relative identity error = {max_rel:.3e}",
    )


def check_reservation_structural() -> None:
    print("\n=== (3) Structural properties of reservation price ===")
    for gamma in GAMMAS:
        # r(s, 0, t) = s for all t
        r0 = reservation_price(S0, 0, T_GRID, gamma, SIGMA, T_END)
        assert np.allclose(r0, S0, atol=TOL_STRICT)
        # r decreases linearly in q at any fixed (s, t)
        r_vs_q = reservation_price(S0, Q_GRID, 0.0, gamma, SIGMA, T_END)
        slope = np.diff(r_vs_q)
        expected_slope = -gamma * SIGMA**2 * (T_END - 0.0)
        assert np.allclose(slope, expected_slope, atol=TOL_STRICT)
        # As t -> T, r(s, q, t) -> s
        r_tT = reservation_price(S0, 3, T_END, gamma, SIGMA, T_END)
        assert np.isclose(r_tT, S0, atol=TOL_STRICT)
        # r^a - r^b = gamma sigma^2 (T-t)
        ra = reservation_ask(S0, 0, T_GRID, gamma, SIGMA, T_END)
        rb = reservation_bid(S0, 0, T_GRID, gamma, SIGMA, T_END)
        diff = ra - rb
        expected = gamma * SIGMA**2 * (T_END - T_GRID)
        assert np.allclose(diff, expected, atol=TOL_STRICT)
    _record(
        "Reservation price structural properties (zero-q, linearity, terminal collapse, bid-ask gap)",
        True,
        "all identities hold to numerical tolerance",
    )


# =============================================================================
# 4) HJB ansatz.
# =============================================================================
def check_hjb_ansatz() -> None:
    print("\n=== (4) HJB ansatz: r from theta(s, q, t) ===")
    max_err_a = 0.0
    max_err_b = 0.0
    for gamma in GAMMAS:
        theta = lambda s, q, t, g=gamma: float(
            theta_from_v(s, q, t, g, SIGMA, T_END)
        )
        for q in Q_GRID:
            for t in T_GRID[::10]:  # sub-grid for speed
                r_a_th, r_b_th = reservation_from_theta(theta, S0, int(q), float(t))
                r_a_dir = float(reservation_ask(S0, q, t, gamma, SIGMA, T_END))
                r_b_dir = float(reservation_bid(S0, q, t, gamma, SIGMA, T_END))
                max_err_a = max(max_err_a, abs(r_a_th - r_a_dir))
                max_err_b = max(max_err_b, abs(r_b_th - r_b_dir))
    assert max(max_err_a, max_err_b) < TOL_STRICT
    _record(
        "Ansatz r^a = theta(q)-theta(q-1), r^b = theta(q+1)-theta(q)",
        True,
        f"max err = {max(max_err_a, max_err_b):.3e}",
    )


# =============================================================================
# 5) Optimal-quote FOC under exponential intensities.
# =============================================================================
def check_foc_exponential() -> None:
    print("\n=== (5) FOC + exponential intensity simplification ===")
    rows = []
    for gamma in GAMMAS:
        info = exponential_intensity_simplification(gamma, K)
        # Algebraic expectation
        assert math.isclose(info["lam_over_lamprime"], -1.0 / K)
        assert math.isclose(
            info["log_term"], math.log1p(gamma / K), rel_tol=1e-15
        )
        adj = info["adjustment"]
        assert math.isclose(adj, (1.0 / gamma) * math.log1p(gamma / K))

        # Numerically pick any delta; the FOC RHS *for exponential* should simplify.
        # The general FOC residual at the *paper's* delta from the asymptotic formula:
        r_a_val = float(reservation_ask(S0, 0, 0.0, gamma, SIGMA, T_END))
        r_b_val = float(reservation_bid(S0, 0, 0.0, gamma, SIGMA, T_END))
        delta_a = optimal_quote_distance("a", 0, 0.0, gamma, SIGMA, T_END, K)
        delta_b = optimal_quote_distance("b", 0, 0.0, gamma, SIGMA, T_END, K)

        lam = lambda d: exponential_intensity(d, A, K)
        lam_p = lambda d: exponential_intensity_prime(d, A, K)
        # For exponential lambda, lambda/lambda' is constant, so the FOC reduces to
        #     r^a - s = delta^a - (1/gamma) ln(1 + gamma/k)
        # For q=0, t=0, asymptotic delta^a = 0.5*gamma*sigma^2*T + adj,
        # and r^a - s = 0.5*gamma*sigma^2*T  (exactly the leading-order match).
        lhs_a = r_a_val - S0
        rhs_a = delta_a - (1.0 / gamma) * math.log1p(gamma / K)
        lhs_b = S0 - r_b_val
        rhs_b = delta_b - (1.0 / gamma) * math.log1p(gamma / K)
        rows.append(
            {
                "gamma": gamma,
                "lam_over_lamprime": info["lam_over_lamprime"],
                "log_term": info["log_term"],
                "adjustment_per_side": adj,
                "FOC_ask_lhs": lhs_a,
                "FOC_ask_rhs": rhs_a,
                "FOC_ask_err": abs(lhs_a - rhs_a),
                "FOC_bid_lhs": lhs_b,
                "FOC_bid_rhs": rhs_b,
                "FOC_bid_err": abs(lhs_b - rhs_b),
            }
        )
        # Direct residual on exponential intensity
        res_a = foc_residual_ask(delta_a, r_a_val, S0, gamma, lam, lam_p)
        res_b = foc_residual_bid(delta_b, r_b_val, S0, gamma, lam, lam_p)
        assert abs(res_a) < TOL_STRICT, (gamma, res_a)
        assert abs(res_b) < TOL_STRICT, (gamma, res_b)
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "foc_exponential.csv", index=False)
    _record(
        "FOC residuals zero under exponential intensities (asymptotic delta)",
        True,
        f"max FOC residual = {df[['FOC_ask_err','FOC_bid_err']].values.max():.3e}",
    )


# =============================================================================
# 6) Asymptotic spreads — paper Table values.
# =============================================================================
def check_asymptotic_spread_table() -> None:
    print("\n=== (6) Paper-table spreads (2/gamma) ln(1+gamma/k) ===")
    expected = {0.1: 1.29, 0.01: 1.33, 0.5: 1.15}
    rows = []
    for gamma, ref in expected.items():
        val = (2.0 / gamma) * math.log1p(gamma / K)
        rows.append(
            {
                "gamma": gamma,
                "k": K,
                "computed": val,
                "paper_rounded": ref,
                "rounded_2dp": round(val, 2),
                "matches_paper": round(val, 2) == ref,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "spread_table_paper.csv", index=False)
    print(df.to_string(index=False))
    all_match = bool(df["matches_paper"].all())
    _record(
        "Paper-table spreads: 1.29 (gamma=0.1), 1.33 (gamma=0.01), 1.15 (gamma=0.5)",
        all_match,
        ", ".join(f"gamma={r['gamma']}->{r['rounded_2dp']:.2f}" for r in rows),
    )


# =============================================================================
# 7) Small-gamma limit -> 2/k and inventory term vanishes.
# =============================================================================
def check_small_gamma_limit() -> None:
    print("\n=== (7) Small-gamma limit: spread -> 2/k, inventory term -> 0 ===")
    target = small_gamma_spread_limit(K)  # 2/k
    rows = []
    for g in SMALL_GAMMAS:
        spread = (2.0 / g) * math.log1p(g / K)
        inventory_term = 1.0 * g * SIGMA**2 * T_END  # for q=1, full T
        rows.append(
            {
                "gamma": g,
                "spread_term": spread,
                "limit_2_over_k": target,
                "gap_to_limit": spread - target,
                "inventory_term_q1": inventory_term,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "small_gamma_limit.csv", index=False)
    print(df.to_string(index=False))
    # The gap should monotonically shrink to 0.
    monotone = bool(np.all(np.diff(np.abs(df["gap_to_limit"].values)) <= 1e-15))
    smallest_gap = float(np.abs(df["gap_to_limit"].iloc[-1]))
    _record(
        "Small-gamma limit: (2/gamma)ln(1+gamma/k) -> 2/k",
        monotone and smallest_gap < 1e-4,
        f"|gap| at gamma=1e-4 is {smallest_gap:.3e}; monotone={monotone}",
    )


# =============================================================================
# 8) Infinite-horizon admissibility + closed forms.
# =============================================================================
def check_infinite_horizon() -> None:
    print("\n=== (8) Infinite-horizon formulas and admissibility ===")
    rows = []
    all_admissible = True
    boundary_notes = []
    for gamma in GAMMAS:
        omega = omega_from_qmax(Q_MAX, gamma, SIGMA)
        adm = admissibility_holds(Q_GRID, omega, gamma, SIGMA)
        all_admissible &= bool(adm.all())
        # Use strict=False so the q=±q_max boundary returns ±inf instead of raising.
        ra = inf_horizon_ask(S0, Q_GRID, omega, gamma, SIGMA, strict=False)
        rb = inf_horizon_bid(S0, Q_GRID, omega, gamma, SIGMA, strict=False)
        # Interior grid for strict numeric checks: bid valid on q<q_max,
        # ask valid on q>-q_max (post-trade inventory stays admissible).
        for q, qa, qb, ok in zip(Q_GRID, ra, rb, adm):
            rows.append(
                {
                    "gamma": gamma,
                    "omega": omega,
                    "q": int(q),
                    "r_ask_inf": qa,
                    "r_bid_inf": qb,
                    "admissible": bool(ok),
                }
            )
        # Document boundary behaviour
        if not np.isfinite(rb[-1]):
            boundary_notes.append(
                f"gamma={gamma}: r^b(q={Q_MAX}) = -inf (refuse to buy at cap)"
            )
        if not np.isfinite(ra[0]):
            boundary_notes.append(
                f"gamma={gamma}: r^a(q={-Q_MAX}) = +inf (refuse to sell at cap)"
            )
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "infinite_horizon.csv", index=False)
    _record(
        "Infinite-horizon admissibility holds on |q|<=q_max",
        all_admissible,
        f"q_max={Q_MAX}, gammas={GAMMAS}",
    )
    # At q=0, ra > s > rb (well-defined, interior)
    sub = df[df["q"] == 0]
    for _, r in sub.iterrows():
        assert r["r_ask_inf"] > S0
        assert r["r_bid_inf"] < S0
    _record(
        "Infinite-horizon q=0 sanity: r^a > s > r^b",
        True,
        "verified for all gammas",
    )
    # Interior (non-boundary) values are finite and well-ordered (ra > rb).
    interior = df[(df["q"] > -Q_MAX) & (df["q"] < Q_MAX)]
    finite_ok = bool(
        np.all(np.isfinite(interior["r_ask_inf"]))
        and np.all(np.isfinite(interior["r_bid_inf"]))
        and np.all(interior["r_ask_inf"] > interior["r_bid_inf"])
    )
    _record(
        "Infinite-horizon interior |q|<q_max: finite and r^a > r^b",
        finite_ok,
        "; ".join(boundary_notes) if boundary_notes else "no boundary degeneracies",
    )


# =============================================================================
# 9) Plots and tables.
# =============================================================================
def make_plots() -> None:
    print("\n=== (9) Generating plots and tables ===")

    # --- (a) r(s,q,t) - s vs q for different t and gamma ---
    fig, axes = plt.subplots(1, len(GAMMAS), figsize=(15, 4.5), sharey=False)
    for ax, gamma in zip(axes, GAMMAS):
        for t in [0.0, 0.25, 0.5, 0.75, 0.95]:
            r = reservation_price(S0, Q_GRID, t, gamma, SIGMA, T_END) - S0
            ax.plot(Q_GRID, r, marker="o", ms=3, label=f"t={t}")
        ax.set_title(f"r(s,q,t) - s   (gamma={gamma})")
        ax.set_xlabel("inventory q")
        ax.set_ylabel("r - s")
        ax.axhline(0, color="black", lw=0.5)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "reservation_price_vs_q.png", dpi=130)
    plt.close(fig)

    # --- (b) Asymptotic spread vs time for different gamma ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for gamma in GAMMAS:
        spread = asymptotic_spread(T_GRID, gamma, SIGMA, T_END, K)
        ax.plot(T_GRID, spread, label=f"gamma={gamma}")
    ax.set_title(
        "Asymptotic spread $\\gamma\\sigma^2(T-t) + (2/\\gamma)\\ln(1+\\gamma/k)$"
    )
    ax.set_xlabel("t")
    ax.set_ylabel("spread")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "asymptotic_spread_vs_t.png", dpi=130)
    plt.close(fig)

    # --- (c) Small-gamma decay of spread term to 2/k ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    gs = np.logspace(-4, 0, 50)
    spr = (2.0 / gs) * np.log1p(gs / K)
    ax.plot(gs, spr, label="(2/gamma) ln(1 + gamma/k)")
    ax.axhline(2.0 / K, color="red", ls="--", label="2/k = %.4f" % (2.0 / K))
    ax.set_xscale("log")
    ax.set_xlabel("gamma")
    ax.set_ylabel("spread term")
    ax.set_title("Small-gamma limit of liquidity spread")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIGS / "small_gamma_limit.png", dpi=130)
    plt.close(fig)

    # --- (d) Terminal collapse: r(s,q,t) - s -> 0 as t -> T ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for q in [-5, -2, 0, 2, 5]:
        r = reservation_price(S0, q, T_GRID, 0.1, SIGMA, T_END) - S0
        ax.plot(T_GRID, r, label=f"q={q}")
    ax.set_title("Terminal collapse: r(s,q,t) - s vs t   (gamma=0.1)")
    ax.set_xlabel("t")
    ax.set_ylabel("r - s")
    ax.axhline(0, color="black", lw=0.5)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "terminal_collapse.png", dpi=130)
    plt.close(fig)

    # --- (e) Quote distances delta^a, delta^b vs q at t=0 ---
    fig, axes = plt.subplots(1, len(GAMMAS), figsize=(15, 4.5))
    for ax, gamma in zip(axes, GAMMAS):
        da = asymptotic_delta_a(Q_GRID, 0.0, gamma, SIGMA, T_END, K)
        db = asymptotic_delta_b(Q_GRID, 0.0, gamma, SIGMA, T_END, K)
        ax.plot(Q_GRID, da, marker="o", ms=3, label="delta^a")
        ax.plot(Q_GRID, db, marker="o", ms=3, label="delta^b")
        ax.set_title(f"Optimal half-spreads (t=0, gamma={gamma})")
        ax.set_xlabel("q")
        ax.set_ylabel("distance from mid")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "delta_a_delta_b_vs_q.png", dpi=130)
    plt.close(fig)

    # --- (f) Infinite-horizon reservation prices ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for gamma in GAMMAS:
        omega = omega_from_qmax(Q_MAX, gamma, SIGMA)
        ra = inf_horizon_ask(S0, Q_GRID, omega, gamma, SIGMA, strict=False) - S0
        rb = inf_horizon_bid(S0, Q_GRID, omega, gamma, SIGMA, strict=False) - S0
        # Mask out boundary ±inf points so the plot remains finite
        ra_m = np.where(np.isfinite(ra), ra, np.nan)
        rb_m = np.where(np.isfinite(rb), rb, np.nan)
        ax.plot(Q_GRID, ra_m, marker="^", ms=3, label=f"r^a inf, gamma={gamma}")
        ax.plot(Q_GRID, rb_m, marker="v", ms=3, label=f"r^b inf, gamma={gamma}")
    ax.set_title("Infinite-horizon reservation prices (centered at s=100)")
    ax.set_xlabel("q")
    ax.set_ylabel("price - s")
    ax.axhline(0, color="black", lw=0.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "infinite_horizon_reservation.png", dpi=130)
    plt.close(fig)

    print(f"Plots saved to {FIGS}")


# =============================================================================
# 10) Representative tables.
# =============================================================================
def make_representative_tables() -> None:
    print("\n=== (10) Representative tables ===")
    # Reservation prices on a small grid
    rows = []
    for gamma in GAMMAS:
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for q in [-5, -2, 0, 2, 5]:
                rows.append(
                    {
                        "gamma": gamma,
                        "t": t,
                        "q": q,
                        "r_ask": float(
                            reservation_ask(S0, q, t, gamma, SIGMA, T_END)
                        ),
                        "r_bid": float(
                            reservation_bid(S0, q, t, gamma, SIGMA, T_END)
                        ),
                        "r_mid": float(
                            reservation_price(S0, q, t, gamma, SIGMA, T_END)
                        ),
                        "spread": float(
                            reservation_spread(t, gamma, SIGMA, T_END)
                        ),
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "reservation_prices_grid.csv", index=False)
    print(df.head().to_string(index=False))


# =============================================================================
# Driver + RESULTS.md.
# =============================================================================
def write_results_md() -> None:
    md_path = RESULTS / "RESULTS.md"
    spread_df = pd.read_csv(TABLES / "spread_table_paper.csv")
    foc_df = pd.read_csv(TABLES / "foc_exponential.csv")
    sg_df = pd.read_csv(TABLES / "small_gamma_limit.csv")
    inf_df = pd.read_csv(TABLES / "infinite_horizon.csv")

    passed = sum(1 for r in results_summary if r["passed"])
    total = len(results_summary)

    md = []
    md.append("# Avellaneda-Stoikov Analytical Verification — Results\n")
    md.append(
        f"Summary: **{passed}/{total} identity checks PASS** within numerical tolerance "
        f"(strict: {TOL_STRICT:.0e}, loose: {TOL_LOOSE:.0e}).\n"
    )
    md.append("## Parameters\n")
    md.append(
        f"- s ≈ {S0}, σ = {SIGMA}, T = {T_END}\n"
        f"- A = {A}, k = {K}\n"
        f"- γ ∈ {GAMMAS}\n"
        f"- q grid: [-{Q_MAX}, {Q_MAX}], t grid: {len(T_GRID)} points in [0, T]\n"
        f"- Infinite horizon: ω = ½γ²σ²(q_max+1)² with q_max = {Q_MAX}\n"
    )

    md.append("\n## Identity Checks\n")
    md.append("| # | Check | Status | Detail |\n|---|---|---|---|")
    for i, r in enumerate(results_summary, 1):
        md.append(
            f"| {i} | {r['check']} | {'✅ PASS' if r['passed'] else '❌ FAIL'} | {r['detail']} |"
        )

    md.append("\n## Paper-Table Spreads (k = 1.5)\n")
    md.append("Computed `(2/γ) ln(1 + γ/k)` versus paper-rounded values:\n")
    md.append(spread_df.to_markdown(index=False))

    md.append("\n## FOC under Exponential Intensities\n")
    md.append(
        "Algebraic decomposition: per-side liquidity adjustment = `(1/γ) ln(1 + γ/k)`.\n"
    )
    md.append(foc_df.to_markdown(index=False))

    md.append("\n## Small-γ Limit\n")
    md.append(
        f"`(2/γ) ln(1 + γ/k)` → `2/k = {2.0/K:.6f}` as γ → 0; inventory term shrinks linearly.\n"
    )
    md.append(sg_df.to_markdown(index=False))

    md.append("\n## Infinite-Horizon Reservation Prices (excerpt)\n")
    md.append(inf_df.head(15).to_markdown(index=False))
    md.append(
        f"\nAll {len(inf_df)} grid points satisfy the admissibility "
        f"`ω > ½γ²σ²q²` strictly.\n"
    )

    md.append("\n## Figures\n")
    for name in [
        "reservation_price_vs_q.png",
        "asymptotic_spread_vs_t.png",
        "small_gamma_limit.png",
        "terminal_collapse.png",
        "delta_a_delta_b_vs_q.png",
        "infinite_horizon_reservation.png",
    ]:
        md.append(f"![{name}](figures/{name})\n")

    md.append("\n## Conclusions\n")
    md.append(
        "- The closed-form frozen-inventory value function satisfies the terminal "
        "condition exactly.\n"
        "- Utility-indifference equations reproduce the analytical reservation bid, ask, "
        "and average price formulas; their structural properties (zero at q=0, linearity "
        "in q, terminal collapse, bid-ask gap = γσ²(T−t)) hold to ~1e-10.\n"
        "- The HJB ansatz identity `r^a = θ(q) − θ(q−1)` and `r^b = θ(q+1) − θ(q)` "
        "holds exactly using the closed-form θ.\n"
        "- For exponential intensities, the FOC simplifies cleanly with per-side "
        "liquidity adjustment `(1/γ) ln(1 + γ/k)`. The asymptotic-spread paper values "
        "1.29, 1.33, 1.15 are reproduced for γ = 0.1, 0.01, 0.5.\n"
        "- The small-γ limit `(2/γ) ln(1 + γ/k) → 2/k` is verified across "
        "γ ∈ {0.5, 0.1, 0.01, 0.001, 1e-4}; the inventory contribution `qγσ²(T−t)` "
        "vanishes, producing symmetric quoting.\n"
        "- The infinite-horizon formulas are well-defined under the suggested "
        "ω = ½γ²σ²(q_max+1)² for every q in [−10, 10].\n"
    )

    md_path.write_text("\n".join(md))
    print(f"Results summary written to {md_path}")

    # Also save machine-readable summary
    (RESULTS / "summary.json").write_text(
        json.dumps(
            {
                "passed": passed,
                "total": total,
                "tolerance_strict": TOL_STRICT,
                "tolerance_loose": TOL_LOOSE,
                "checks": results_summary,
            },
            indent=2,
        )
    )


def main() -> int:
    check_frozen_terminal()
    check_reservation_indifference()
    check_reservation_structural()
    check_hjb_ansatz()
    check_foc_exponential()
    check_asymptotic_spread_table()
    check_small_gamma_limit()
    check_infinite_horizon()
    make_representative_tables()
    make_plots()
    write_results_md()

    failed = [r for r in results_summary if not r["passed"]]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

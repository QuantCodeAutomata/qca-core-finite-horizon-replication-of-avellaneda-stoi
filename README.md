# qca-core-finite-horizon-replication-of-avellaneda-stoi

Analytical verification suite for the closed-form formulas of
**Avellaneda & Stoikov (2008), "High-frequency trading in a limit order book."**

This repository is a **deterministic** formula-checker — no Monte Carlo,
no datasets. It evaluates the paper's analytical identities on dense numerical
grids and reports whether each identity holds within floating-point tolerance.

## Identities verified
1. Frozen-inventory value function and terminal condition.
2. Reservation bid/ask/average prices from utility indifference.
3. Structural properties: zero-q collapse, linearity in q, terminal collapse,
   bid–ask gap `γσ²(T−t)`.
4. HJB exponential-utility ansatz: `r^a = θ(q) − θ(q−1)`, `r^b = θ(q+1) − θ(q)`.
5. FOC under exponential intensities → liquidity adjustment `(1/γ) ln(1+γ/k)`.
6. Asymptotic closed-form spread `γσ²(T−t) + (2/γ) ln(1+γ/k)`.
7. Small-γ limit `(2/γ) ln(1+γ/k) → 2/k` and vanishing inventory term.
8. Infinite-horizon `r̄^a`, `r̄^b` with admissibility `ω > ½γ²σ²q²`.

## Layout
```
src/                analytical modules (frozen, reservation, HJB ansatz,
                    optimal quotes, asymptotic, infinite horizon)
exp/run_verification.py
                    runs all identity checks, writes tables and figures
tests/              pytest unit tests for every module
results/RESULTS.md  human-readable summary
results/tables/     CSV tables (FOC, spreads, inf-horizon, etc.)
results/figures/    PNG plots
```

## Run
```bash
pip install -r requirements.txt
pytest -q
python -m exp.run_verification
```

## Baseline parameters (per paper)
`s ≈ 100, σ = 2, T = 1, A = 140, k = 1.5, γ ∈ {0.01, 0.1, 0.5}`,
`q ∈ [-10, 10]`, `t` on 101 points in `[0, T]`,
infinite horizon `ω = ½γ²σ²(q_max + 1)²` with `q_max = 10`.

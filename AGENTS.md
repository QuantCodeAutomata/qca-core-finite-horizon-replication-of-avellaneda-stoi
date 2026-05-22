# Repository Context: Avellaneda-Stoikov Analytical Verification

## Purpose
Self-contained analytical verification of the Avellaneda & Stoikov (2008)
"High-frequency trading in a limit order book" closed-form formulas.
No Monte Carlo, no data download — pure formula verification on numerical grids.

## Key Conventions
- Arithmetic Brownian mid-price: dS = sigma dW (zero drift).
- Exponential utility: U(x) = -exp(-gamma x).
- Symmetric exponential intensities: lambda(delta) = A * exp(-k * delta).
- Unit trade size, no transaction costs.

## Baseline Parameters (per paper)
- s ~ 100, sigma = 2, T = 1
- gamma in {0.01, 0.1, 0.5}
- A = 140, k = 1.5
- q grid: [-10, 10], t grid: 101 points in [0, T]
- Infinite horizon: omega = 0.5 * gamma^2 * sigma^2 * (q_max+1)^2

## Module Layout
- src/frozen_inventory.py    — v(x,s,q,t) closed form
- src/reservation_prices.py  — r^a, r^b, r finite-horizon
- src/hjb_ansatz.py          — exponential-utility ansatz checks
- src/optimal_quotes.py      — FOC + exponential intensity simplification
- src/asymptotic.py          — inventory-asymptotic spread/quote formulas
- src/infinite_horizon.py    — discounted reservation prices + admissibility

## Expected Spread Values (k=1.5)
(2/gamma) ln(1 + gamma/k) — paper Table:
- gamma=0.1  -> ~1.29
- gamma=0.01 -> ~1.33
- gamma=0.5  -> ~1.15

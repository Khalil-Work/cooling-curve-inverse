# Verification evidence

Reproduced by `ctest` (`tests/test_lm_fit.cpp` — the CI gate) and
`python examples/run_verification.py` (builds/runs the C++ demo, renders
figures/GIF). Verified with GCC 16.2.0 `-std=c++17 -O2 -Wall -Wextra
-Wpedantic`; CI repeats on Ubuntu + Windows.

## What each check establishes

| Check | Property established |
| --- | --- |
| Noise-free parameter recovery | Three distinct parameter sets recovered to < 1e-7 (relative) with cost at machine noise (< 1e-18) and tolerance-terminated status. |
| Jacobian validation | The analytic Jacobian matches central finite differences to < 5e-5 relative — catches sign/chain-rule errors. |
| Independent evaluation | The fitted curve matches a separately-coded model expression to 1e-9 at every sample. |
| Noisy recovery | Seeded σ = 0.3 K noise, n = 120: k within 5%, temperatures within the noise band, residual mean ~ 0 (unbiased), cost in [0.3, 0.9]·σ²n — consistent with the statistical expectation ½σ²n. |
| Failure handling | k0 ≤ 0 rejected (model domain); NaN data rejected; a 1-iteration budget returns MAX_ITER/ILL_CONDITIONED — never a fake success. |
| Poor/flat starts | All-three-wrong non-flat start (40, 60, 0.01) converges exactly; the flat start T_amb0 = T_00 (collapsed Jacobian) reports ILL_CONDITIONED honestly. |

## Recorded run (`docs/results/`)

- `cooling_fit.png/.svg` + demo CSVs: seeded noisy run (σ = 0.3 K, 240
  samples, truth Tamb = 25 K, T0 = 95 K, k = 0.08 1/s): fitted
  (25.05, 95.08, 0.08028) — k error **0.35%**, cost 11.53 vs noise floor
  ~10.8, residual panel shows the expected noise-band structure.
- `fit_convergence.gif`: 41 frames of the LM parameter path interpolated
  between recorded budget-refit states (no frame extrapolates past the
  last recorded iteration).
- `run_summary.json`: metrics + runtime context.

## Limits of this evidence

- Single-exponential model: multi-exponential data are outside scope.
- No weighting matrix (identity weights) and no confidence intervals —
  point estimates with honest termination codes only.
- The noisy-recovery band is a statistical expectation at the tested
  noise/size, not a guarantee for arbitrary noise realizations.

# Model and method

## Inverse problem

Recover θ = (T_amb, T_0, k) from noisy measurements of a Newton-cooling
curve

    T(t; θ) = T_amb + (T_0 − T_amb) e^(−k t)   [K],

with t [s]. The residual for data (t_i, T_i) is r_i(θ) = T(t_i; θ) − T_i
and the objective is ½‖r‖².

## Levenberg-Marquardt (textbook form, self-implemented)

At each iterate θ:

1. Assemble the analytic Jacobian J (n×3) —

       ∂T/∂T_amb = 1 − e^(−kt),  ∂T/∂T_0 = e^(−kt),
       ∂T/∂k = −(T_0 − T_amb) t e^(−kt)

   (validated against central differences in the tests);

2. Solve the damped normal equations

       (JᵀJ + μ diag(max(JᵀJ)_dd, ε)) δ = −Jᵀ r

   by Gaussian elimination with partial pivoting; a pivot-ratio test
   against machine ε rejects numerically singular systems
   (LM_ILL_CONDITIONED);

3. Accept δ when it lowers the cost — updating μ by the standard
   gain-ratio rule (ρ > 0.75 → μ/3) — otherwise increase μ (×10, capped at
   1e12 → ILL_CONDITIONED). Trial steps that would leave the k > 0 model
   domain are damped, never clipped.

Termination: gradient ‖Jᵀr‖ < gtol; step ‖δ‖ < stol(‖θ‖ + stol); budget
exhaustion (LM_MAX_ITER — an honest non-success, never a fake
convergence); non-finite residuals (LM_NONFINITE).

## Identifiability note (tested)

A flat start T_amb0 = T_00 collapses the exponential to a constant: the
k-column of J vanishes and the fit reports ILL_CONDITIONED immediately —
an honest failure. Non-flat starts — even with all three parameters wrong
— converge for this model (documented basin).

## Statistical expectations

For Gaussian noise σ the cost at the optimum approaches ½σ²n; the k
estimate is unbiased with variance determined by the Jacobian conditioning
(noisy-band assertions in the tests use these facts, with the tested
σ = 0.3 K, n = 120 giving k within 5%).

## Limits

Single-exponential model only; no multi-exponential/mixed dynamics, no
weighting matrix (identity weights), no confidence intervals (point
estimates + honest failure codes).

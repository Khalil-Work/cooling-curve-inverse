# API reference (C++)

Header `src/lm_fit.hpp`, namespace `cooling_curve`.

```cpp
enum lm_status { LM_OK, LM_MAX_ITER, LM_GRAD_TOL, LM_STEP_TOL,
                 LM_ILL_CONDITIONED, LM_NONFINITE, LM_BAD_INPUT };

struct lm_result {
    double theta[3];   // fitted (T_amb, T_0, k)
    double cost;       // 0.5*||r||^2 at exit
    int iterations;
    lm_status status;
};
```

- `fit_newton_cooling(t, T, n, theta0, gtol, stol, max_iter) -> lm_result`
  — requires n ≥ 4 finite data points, finite θ0 with k0 > 0 (a
  non-decaying guess is outside the model domain), positive tolerances.
- `model_temperature(t, theta)` — one evaluation.
- `model_jacobian(t, n, theta, J)` — analytic n×3 row-major Jacobian.
- `status_message(status)` — human-readable termination reason.

## Build

```bash
cmake -S . -B build
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

Direct gcc path (used by the results script):
`g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic src/cooling_demo.cpp src/lm_fit.cpp -o build/cooling_demo`.

## Example

```cpp
const double guess[3] = {20.0, 80.0, 0.05};
const auto res = cooling_curve::fit_newton_cooling(t, T, n, guess,
                                                   1e-12, 1e-14, 200);
std::printf("%s: k = %.6f\n", cooling_curve::status_message(res.status),
            res.theta[2]);
```

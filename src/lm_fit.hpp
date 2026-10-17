// cooling-curve-inverse: nonlinear least-squares parameter estimation for
// a Newton-cooling model, with validated Jacobian, conditioning handling,
// and clear failure signals.
//
// Model: T(t) = T_amb + (T_0 - T_amb) * exp(-k t)   [K]
// Parameters: theta = (T_amb, T_0, k). The residual vector for data
// (t_i, T_i) is r_i(theta) = T(t_i; theta) - T_i. Levenberg-Marquardt
// minimizes 0.5 * ||r||^2 with the damping-modified normal equations
// (J^T J + mu diag(J^T J)) delta = -J^T r, implemented from the standard
// textbook algorithm (write-up in docs/model.md). Convergence: gradient
// norm and step-size tolerances; iteration budget; all exits carry a
// termination reason.
//
// Failure handling: a condition check on (J^T J + mu D) rejects steps
// when the linear solve is ill-conditioned (ratio test against machine
// epsilon); non-finite residuals abort with a clear flag. Initial guesses
// with k <= 0 are rejected (non-decaying model outside scope).

#ifndef LM_FIT_HPP
#define LM_FIT_HPP

#include <cstddef>

namespace cooling_curve {

enum lm_status {
    LM_OK = 0,
    LM_MAX_ITER = 1,
    LM_GRAD_TOL = 2,
    LM_STEP_TOL = 3,
    LM_ILL_CONDITIONED = 4,
    LM_NONFINITE = 5,
    LM_BAD_INPUT = 6
};

struct lm_result {
    double theta[3];        // fitted (T_amb, T_0, k)
    double cost;            // 0.5 * ||r||^2 at exit
    int iterations;
    lm_status status;
};

// Fit the Newton-cooling model to n data points (t[i], T[i]).
// theta0: initial guess. tolerances: gtol (gradient), stol (step),
// max_iter budget. Returns termination info.
lm_result fit_newton_cooling(const double* t, const double* T, std::size_t n,
                             const double theta0[3], double gtol,
                             double stol, int max_iter);

// Model evaluation at one time.
double model_temperature(double t, const double theta[3]);

// Analytic Jacobian dr_i/dtheta at theta for the data (3 columns),
// written into J (n x 3, row-major).
void model_jacobian(const double* t, std::size_t n, const double theta[3],
                    double* J);

const char* status_message(lm_status s);

}  // namespace cooling_curve

#endif  // LM_FIT_HPP

// Implementation of the Levenberg-Marquardt Newton-cooling fit.
#include "lm_fit.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace cooling_curve {

double model_temperature(double t, const double theta[3]) {
    return theta[0] + (theta[1] - theta[0]) * std::exp(-theta[2] * t);
}

void model_jacobian(const double* t, std::size_t n, const double theta[3],
                    double* J) {
    const double Tamb = theta[0], T0 = theta[1], k = theta[2];
    for (std::size_t i = 0; i < n; ++i) {
        const double e = std::exp(-k * t[i]);
        J[3 * i + 0] = 1.0 - e;               // dT/dTamb
        J[3 * i + 1] = e;                     // dT/dT0
        J[3 * i + 2] = -(T0 - Tamb) * t[i] * e;  // dT/dk
    }
}

namespace {

void residuals(const double* t, const double* T, std::size_t n,
               const double theta[3], double* r) {
    for (std::size_t i = 0; i < n; ++i) {
        r[i] = model_temperature(t[i], theta) - T[i];
    }
}

double cost_of(const double* r, std::size_t n) {
    double c = 0.0;
    for (std::size_t i = 0; i < n; ++i) c += 0.5 * r[i] * r[i];
    return c;
}

// Solve the 3x3 damped normal system (A + mu*D) delta = g by Gaussian
// elimination with partial pivoting. Returns false when the pivot ratio
// indicates numerical singularity.
bool solve3(const double A[3][3], const double D[3], double mu,
            const double g[3], double delta[3]) {
    double M[3][4];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) M[i][j] = A[i][j] + (i == j ? mu * D[i] : 0.0);
        M[i][3] = -g[i];
    }
    const double eps = 2.220446049250313e-16;
    double scale = 0.0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) scale = std::max(scale, std::fabs(M[i][j]));
    for (int col = 0; col < 3; ++col) {
        int piv = col;
        for (int row = col + 1; row < 3; ++row)
            if (std::fabs(M[row][col]) > std::fabs(M[piv][col])) piv = row;
        if (std::fabs(M[piv][col]) <= 1e3 * eps * scale) return false;
        if (piv != col)
            for (int j = 0; j < 4; ++j) std::swap(M[piv][j], M[col][j]);
        for (int row = col + 1; row < 3; ++row) {
            const double f = M[row][col] / M[col][col];
            for (int j = col; j < 4; ++j) M[row][j] -= f * M[col][j];
        }
    }
    for (int row = 2; row >= 0; --row) {
        double s = M[row][3];
        for (int j = row + 1; j < 3; ++j) s -= M[row][j] * delta[j];
        delta[row] = s / M[row][row];
    }
    return true;
}

}  // namespace

lm_result fit_newton_cooling(const double* t, const double* T, std::size_t n,
                             const double theta0[3], double gtol,
                             double stol, int max_iter) {
    lm_result out;
    std::memset(&out, 0, sizeof(out));
    if (t == nullptr || T == nullptr || n < 4) {
        out.status = LM_BAD_INPUT;
        return out;
    }
    if (!(gtol > 0.0) || !(stol > 0.0) || max_iter <= 0) {
        out.status = LM_BAD_INPUT;
        return out;
    }
    for (std::size_t i = 0; i < n; ++i)
        if (!std::isfinite(t[i]) || !std::isfinite(T[i])) {
            out.status = LM_BAD_INPUT;
            return out;
        }
    double theta[3] = {theta0[0], theta0[1], theta0[2]};
    if (!std::isfinite(theta[0]) || !std::isfinite(theta[1]) ||
        !std::isfinite(theta[2]) || theta[2] <= 0.0) {
        out.status = LM_BAD_INPUT;   // non-decaying or invalid guess
        return out;
    }

    std::vector<double> r(n), J(3 * n), r_trial(n);
    residuals(t, T, n, theta, r.data());
    double cost = cost_of(r.data(), n);
    if (!std::isfinite(cost)) {
        out.status = LM_NONFINITE;
        return out;
    }
    double mu = 1e-3;

    for (int it = 0; it < max_iter; ++it) {
        model_jacobian(t, n, theta, J.data());
        double A[3][3] = {{0}}, g[3] = {0}, D[3] = {0};
        for (std::size_t i = 0; i < n; ++i)
            for (int a = 0; a < 3; ++a) {
                g[a] += J[3 * i + a] * r[i];
                for (int b = 0; b < 3; ++b) A[a][b] += J[3 * i + a] * J[3 * i + b];
            }
        for (int a = 0; a < 3; ++a) D[a] = std::max(A[a][a], 1e-12);

        double gnorm = std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
        if (gnorm < gtol) {
            out.status = LM_GRAD_TOL;
            break;
        }

        double delta[3];
        if (!solve3(A, D, mu, g, delta)) {
            out.status = LM_ILL_CONDITIONED;
            break;
        }
        double step_norm = std::sqrt(delta[0] * delta[0] + delta[1] * delta[1] +
                                     delta[2] * delta[2]);
        double theta_norm = std::sqrt(theta[0] * theta[0] + theta[1] * theta[1] +
                                      theta[2] * theta[2]);
        if (step_norm < stol * (theta_norm + stol)) {
            out.status = LM_STEP_TOL;
            break;
        }

        double theta_trial[3] = {theta[0] + delta[0], theta[1] + delta[1],
                                 theta[2] + delta[2]};
        if (!std::isfinite(theta_trial[0]) || !std::isfinite(theta_trial[1]) ||
            !std::isfinite(theta_trial[2]) || theta_trial[2] <= 0.0) {
            mu *= 10.0;   // damping guards the k > 0 domain
            continue;
        }
        residuals(t, T, n, theta_trial, r_trial.data());
        const double cost_trial = cost_of(r_trial.data(), n);
        if (!std::isfinite(cost_trial)) {
            out.status = LM_NONFINITE;
            break;
        }
        if (cost_trial < cost) {
            const double actual = cost - cost_trial;
            // Gauss-Newton predicted reduction: delta . (mu*D*delta - g)/... (standard)
            double pred = 0.0;
            for (int a = 0; a < 3; ++a) pred += -delta[a] * (g[a] - mu * D[a] * delta[a]);
            const double rho = pred > 0.0 ? actual / pred : 1.0;
            std::memcpy(theta, theta_trial, sizeof(theta));
            std::memcpy(r.data(), r_trial.data(), n * sizeof(double));
            cost = cost_trial;
            mu = rho > 0.75 ? std::max(mu / 3.0, 1e-12) : mu;
            out.iterations = it + 1;
        } else {
            mu *= 10.0;
            if (mu > 1e12) {
                out.status = LM_ILL_CONDITIONED;
                break;
            }
        }
    }
    if (out.status == LM_OK && out.iterations >= max_iter) out.status = LM_MAX_ITER;
    if (out.status == LM_OK && max_iter > 0 && out.iterations == 0)
        out.status = LM_MAX_ITER;  // budget exhausted without acceptance
    out.theta[0] = theta[0];
    out.theta[1] = theta[1];
    out.theta[2] = theta[2];
    out.cost = cost;
    return out;
}

const char* status_message(lm_status s) {
    switch (s) {
        case LM_OK: return "OK";
        case LM_MAX_ITER: return "MAX_ITER (iteration budget exhausted)";
        case LM_GRAD_TOL: return "GRAD_TOL (gradient below tolerance)";
        case LM_STEP_TOL: return "STEP_TOL (step below tolerance)";
        case LM_ILL_CONDITIONED: return "ILL_CONDITIONED (damped system singular / mu overflow)";
        case LM_NONFINITE: return "NONFINITE (residual became non-finite)";
        case LM_BAD_INPUT: return "BAD_INPUT (invalid data or guess)";
    }
    return "UNKNOWN";
}

}  // namespace cooling_curve

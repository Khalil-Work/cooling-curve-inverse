// Verification tests for cooling-curve-inverse.
//
// Properties pinned:
// 1. Parameter recovery: synthetic noise-free cooling data reproduces the
//    generating (Tamb, T0, k) to ~1e-7 across several parameter sets.
// 2. Jacobian validation: the analytic Jacobian matches central finite
//    differences at a non-trivial theta (catches sign/chain errors).
// 3. Independent evaluation: at the recovered theta the fitted curve
//    matches a separately-coded model to 1e-9.
// 4. Noisy recovery: with seeded Gaussian noise (documented sigma), the
//    fit lands within a stated statistical band of the truth, residuals
//    look unbiased, and the cost matches the noise floor.
// 5. Failure handling: k0 <= 0 rejected; NaN data rejected; a tiny budget
//    returns an honest non-success code, never a fake success.
// 6. Convergence from a poor start within the documented basin.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../src/lm_fit.hpp"

using cooling_curve::fit_newton_cooling;
using cooling_curve::lm_status;
using cooling_curve::model_jacobian;
using cooling_curve::model_temperature;

static int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAILED: %s\n", what); ++failures; }
}

// Deterministic Gaussian from a fixed LCG (self-contained).
static unsigned long long lcg = 20261019ULL;
double gauss() {
    lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL;
    const double u1 = ((lcg >> 11) + 1) / 9007199254740993.0;
    lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL;
    const double u2 = ((lcg >> 11) + 1) / 9007199254740993.0;
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
}

int main() {
    const int n = 120;
    std::vector<double> t(n), T(n);
    for (int i = 0; i < n; ++i) t[i] = 0.25 * i;  // 0..30 s

    const double guess[3] = {20.0, 80.0, 0.05};

    // 1. Parameter recovery, noise-free.
    {
        const double sets[3][3] = {{25.0, 95.0, 0.08}, {18.0, 120.0, 0.02},
                                   {40.0, 41.0, 0.5}};
        for (int s = 0; s < 3; ++s) {
            for (int i = 0; i < n; ++i)
                T[i] = model_temperature(t[i], sets[s]);
            const auto res = fit_newton_cooling(t.data(), T.data(), n, guess,
                                                1e-12, 1e-14, 200);
            check(res.status == lm_status::LM_GRAD_TOL ||
                      res.status == lm_status::LM_STEP_TOL,
                  "noise-free fit converges on a tolerance");
            for (int a = 0; a < 3; ++a)
                check(std::fabs(res.theta[a] - sets[s][a]) <
                          1e-7 * std::max(1.0, std::fabs(sets[s][a])),
                      "noise-free parameter recovered");
            check(res.cost < 1e-18, "noise-free cost at machine noise");
        }
    }

    // 2. Jacobian vs central differences.
    {
        const double theta[3] = {22.0, 90.0, 0.03};
        std::vector<double> J(3 * n), Jn(3 * n);
        model_jacobian(t.data(), n, theta, J.data());
        for (int a = 0; a < 3; ++a) {
            double tp[3] = {theta[0], theta[1], theta[2]};
            double tm[3] = {theta[0], theta[1], theta[2]};
            const double h = (a == 2 ? 1e-7 : 1e-6);
            tp[a] += h;
            tm[a] -= h;
            for (int i = 0; i < n; ++i)
                Jn[3 * i + a] =
                    (model_temperature(t[i], tp) - model_temperature(t[i], tm)) / (2 * h);
        }
        double max_rel = 0.0;
        for (std::size_t i = 0; i < 3 * n; ++i) {
            const double scale = std::max(1e-8, std::fabs(Jn[i]));
            max_rel = std::max(max_rel, std::fabs(J[i] - Jn[i]) / scale);
        }
        check(max_rel < 5e-5, "analytic Jacobian matches central differences");
    }

    // 3. Recovered curve vs independently coded model.
    {
        const double truth[3] = {25.0, 95.0, 0.08};
        for (int i = 0; i < n; ++i)
            T[i] = truth[0] + (truth[1] - truth[0]) * std::exp(-truth[2] * t[i]);
        const auto res = fit_newton_cooling(t.data(), T.data(), n, guess,
                                            1e-12, 1e-14, 200);
        for (int i = 0; i < n; ++i) {
            const double direct =
                truth[0] + (truth[1] - truth[0]) * std::exp(-truth[2] * t[i]);
            check(std::fabs(model_temperature(t[i], res.theta) - direct) < 1e-9,
                  "recovered curve matches independent evaluation");
        }
    }

    // 4. Noisy recovery with statistical honesty.
    {
        const double truth[3] = {25.0, 95.0, 0.08};
        const double sigma = 0.3;  // K
        for (int i = 0; i < n; ++i)
            T[i] = model_temperature(t[i], truth) + sigma * gauss();
        const auto res = fit_newton_cooling(t.data(), T.data(), n, guess,
                                            1e-12, 1e-14, 500);
        check(std::fabs(res.theta[2] - truth[2]) / truth[2] < 0.05,
              "noisy k recovered within 5%");
        check(std::fabs(res.theta[0] - truth[0]) < 1.0 &&
                  std::fabs(res.theta[1] - truth[1]) < 2.0,
              "noisy temperatures recovered within the noise band");
        double mean_r = 0.0;
        for (int i = 0; i < n; ++i)
            mean_r += model_temperature(t[i], res.theta) - T[i];
        mean_r /= n;
        check(std::fabs(mean_r) < 0.1, "residual mean near zero");
        check(res.cost > 0.3 * sigma * sigma * n && res.cost < 0.9 * sigma * sigma * n,
              "cost consistent with the noise level");
    }

    // 5. Failure handling.
    {
        const double truth[3] = {25.0, 95.0, 0.08};
        for (int i = 0; i < n; ++i) T[i] = model_temperature(t[i], truth);
        const double bad_k[3] = {20.0, 80.0, -0.05};
        check(fit_newton_cooling(t.data(), T.data(), n, bad_k, 1e-12, 1e-14, 100)
                  .status == lm_status::LM_BAD_INPUT,
              "k0 <= 0 rejected");
        T[5] = std::nan("");
        check(fit_newton_cooling(t.data(), T.data(), n, guess, 1e-12, 1e-14, 100)
                  .status == lm_status::LM_BAD_INPUT,
              "NaN data rejected");
        T[5] = 60.0;
        const auto tiny = fit_newton_cooling(t.data(), T.data(), n, guess,
                                             1e-12, 1e-14, 1);
        check(tiny.status == lm_status::LM_MAX_ITER ||
                  tiny.status == lm_status::LM_ILL_CONDITIONED,
              "tiny budget returns an honest non-success");
    }

    // 6. Poor-but-informative starts converge; the flat start (Tamb0 ==
    // T00: no decay signal, collapsed Jacobian) fails honestly.
    {
        const double truth[3] = {25.0, 95.0, 0.08};
        for (int i = 0; i < n; ++i) T[i] = model_temperature(t[i], truth);
        const double poor[3] = {40.0, 60.0, 0.01};  // all three wrong
        const auto res = fit_newton_cooling(t.data(), T.data(), n, poor,
                                            1e-12, 1e-14, 2000);
        check(std::fabs(res.theta[2] - 0.08) < 1e-6, "poor start converges (k)");
        check(std::fabs(res.theta[0] - 25.0) < 1e-4, "poor start converges (Tamb)");
        check(std::fabs(res.theta[1] - 95.0) < 1e-4, "poor start converges (T0)");

        const double flat[3] = {50.0, 50.0, 0.01};  // model is constant
        const auto flat_res = fit_newton_cooling(t.data(), T.data(), n, flat,
                                                 1e-12, 1e-14, 200);
        check(flat_res.status == lm_status::LM_ILL_CONDITIONED,
              "flat start reports ILL_CONDITIONED, not a fake success");
    }

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return EXIT_FAILURE;
    }
    std::printf("cooling-curve-inverse: all checks passed\n");
    return EXIT_SUCCESS;
}

// Demo driver: generates a seeded noisy cooling curve from known
// parameters, fits it, and writes CSVs for plotting.
#define _CRT_SECURE_NO_WARNINGS 1
#include <cmath>
#include <cstdio>
#include <vector>

#include "lm_fit.hpp"

static unsigned long long lcg = 424242ULL;
double gauss() {
    lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL;
    const double u1 = ((lcg >> 11) + 1) / 9007199254740993.0;
    lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL;
    const double u2 = ((lcg >> 11) + 1) / 9007199254740993.0;
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
}

int main(int argc, char** argv) {
    const char* out = argc > 1 ? argv[1] : "build";
    const double truth[3] = {25.0, 95.0, 0.08};
    const double sigma = 0.3;
    const int n = 240;
    std::vector<double> t(n), T(n);
    for (int i = 0; i < n; ++i) {
        t[i] = 0.2 * i;
        T[i] = cooling_curve::model_temperature(t[i], truth) + sigma * gauss();
    }
    const double guess[3] = {20.0, 80.0, 0.05};
    const auto res = cooling_curve::fit_newton_cooling(t.data(), T.data(), n, guess,
                                                       1e-12, 1e-14, 1000);
    std::printf("status=%s iters=%d\n", cooling_curve::status_message(res.status),
                res.iterations);
    std::printf("truth:  Tamb=%.4f T0=%.4f k=%.6f\n", truth[0], truth[1], truth[2]);
    std::printf("fitted: Tamb=%.4f T0=%.4f k=%.6f (k err %.2f%%)\n",
                res.theta[0], res.theta[1], res.theta[2],
                100.0 * std::fabs(res.theta[2] - truth[2]) / truth[2]);
    std::printf("cost=%.6e (noise floor ~ %.6e)\n", res.cost, 0.5 * sigma * sigma * n);

    char path[512];
    std::snprintf(path, sizeof(path), "%s/cooling_fit.csv", out);
    std::FILE* f = std::fopen(path, "w");
    std::fprintf(f, "time_s,T_data_K,T_fit_K,residual_K\n");
    for (int i = 0; i < n; ++i)
        std::fprintf(f, "%.6f,%.9f,%.9f,%.9f\n", t[i], T[i],
                     cooling_curve::model_temperature(t[i], res.theta),
                     cooling_curve::model_temperature(t[i], res.theta) - T[i]);
    std::fclose(f);
    std::snprintf(path, sizeof(path), "%s/convergence.csv", out);
    f = std::fopen(path, "w");
    std::fprintf(f, "iteration,k,Tamb,T0\n");
    // Re-run with intermediate prints by stepping the budget.
    for (int budget = 1; budget <= 12; ++budget) {
        const auto r = cooling_curve::fit_newton_cooling(t.data(), T.data(), n, guess,
                                                         1e-12, 1e-14, budget);
        std::fprintf(f, "%d,%.9f,%.9f,%.9f\n", budget, r.theta[2], r.theta[0], r.theta[1]);
    }
    std::fclose(f);
    return 0;
}

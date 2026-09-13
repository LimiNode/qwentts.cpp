#include "sampling.h"

#include <cstdio>
#include <cmath>
#include <limits>

namespace {

bool check(bool condition, const char *description) {
    if (!condition) {
        std::fprintf(stderr, "sampling contract failed: %s\\n", description);
    }
    return condition;
}

} // namespace

int main() {
    float first_step[] = {1.0F, 100.0F, 2.0F};
    suppress_initial_eos(first_step, 3, 1, 0);
    if (!check(std::isinf(first_step[1]) && first_step[1] < 0.0F,
               "EOS is masked at the first step") ||
        !check(first_step[0] == 1.0F, "non-EOS score is preserved") ||
        !check(first_step[2] == 2.0F, "scores after EOS are preserved")) {
        return 1;
    }

    float later_step[] = {1.0F, 100.0F, 2.0F};
    suppress_initial_eos(later_step, 3, 1, 1);
    if (!check(later_step[1] == 100.0F,
               "EOS remains eligible after the first step")) {
        return 1;
    }

    float invalid_id[] = {1.0F, 2.0F};
    suppress_initial_eos(invalid_id, 2, std::numeric_limits<int>::max(), 0);
    if (!check(invalid_id[0] == 1.0F && invalid_id[1] == 2.0F,
               "an invalid EOS id leaves scores unchanged")) {
        return 1;
    }

    const float weights[] = {1.0F, 4.0F, 2.0F, 0.0F};
    const SamplingDiagnostics diagnostics =
        sampling_diagnostics_from_weights(weights, 4, 2, 0, 3);
    if (!check(diagnostics.candidate_count == 3, "diagnostics count eligible candidates") ||
        !check(diagnostics.eos_rank == 3, "diagnostics rank EOS") ||
        !check(std::fabs(diagnostics.eos_probability - 1.0F / 7.0F) < 1e-6F,
               "diagnostics report EOS probability") ||
        !check(std::fabs(diagnostics.selected_probability - 2.0F / 7.0F) < 1e-6F,
               "diagnostics report selected probability") ||
        !check(diagnostics.top_tokens.size() == 3 &&
                   diagnostics.top_tokens[0].id == 1 &&
                   diagnostics.top_tokens[1].id == 2 &&
                   diagnostics.top_tokens[2].id == 0,
               "diagnostics retain deterministic top tokens")) {
        return 1;
    }
    return 0;
}

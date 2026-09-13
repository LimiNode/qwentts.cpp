#include "sampling.h"

#include <cstdio>
#include <cmath>
#include <cstring>
#include <limits>

namespace {

bool check(bool condition, const char *description) {
    if (!condition) {
        std::fprintf(stderr, "sampling contract failed: %s\\n", description);
    }
    return condition;
}

} // namespace

struct ParityVector {
    const char *name;
    float logits[4];
    int vocab;
    float temperature;
    int top_k;
    float top_p;
    float repetition_penalty;
    const int32_t *history;
    int history_size;
    float uniform_u;
};

int dump_parity_vectors() {
    const int32_t positive_history[] = { 0 };
    const int32_t negative_history[] = { 0 };
    const int32_t *no_history = nullptr;
    const ParityVector vectors[] = {
        { "positive_repetition", { 4.0F, 3.0F, 3.5F, 0.0F }, 3, 1.0F, 2, 1.0F, 2.0F, positive_history, 1, 0.1F },
        { "negative_repetition", { -4.0F, -3.0F, -3.5F, 0.0F }, 3, 1.0F, 2, 1.0F, 2.0F, negative_history, 1, 0.1F },
        { "top_k_one", { 1.0F, 9.0F, 8.0F, 0.0F }, 3, 1.0F, 1, 1.0F, 1.0F, no_history, 0, 0.99F },
        { "top_k_zero", { 1.0F, 2.0F, 3.0F, 0.0F }, 3, 1.0F, 0, 1.0F, 1.0F, no_history, 0, 0.01F },
        { "top_p_one", { 4.0F, 3.0F, 2.0F, 0.0F }, 3, 1.0F, 0, 1.0F, 1.0F, no_history, 0, 0.90F },
        { "top_p_crossing", { 4.0F, 3.0F, 2.0F, 1.0F }, 4, 1.0F, 0, 0.70F, 1.0F, no_history, 0, 0.95F },
        { "uniform_low", { 1.0F, 0.0F, -1.0F, 0.0F }, 3, 1.0F, 0, 1.0F, 1.0F, no_history, 0, 0.0000001F },
        { "uniform_high", { 1.0F, 0.0F, -1.0F, 0.0F }, 3, 1.0F, 0, 1.0F, 1.0F, no_history, 0, 0.9999999F },
        { "ties", { 1.0F, 1.0F, 1.0F, 0.0F }, 3, 1.0F, 0, 1.0F, 1.0F, no_history, 0, 0.50F },
        { "temperature", { 4.0F, 3.0F, 2.0F, 0.0F }, 3, 2.0F, 0, 1.0F, 1.0F, no_history, 0, 0.75F },
    };

    std::printf("name\tselected\tu\n");
    for (const ParityVector &vector : vectors) {
        float logits[4];
        std::memcpy(logits, vector.logits, sizeof(logits));
        const int selected = sample_top_k_p_with_uniform(
            logits, vector.vocab, vector.temperature, vector.top_k, vector.top_p,
            vector.repetition_penalty, vector.history, vector.history_size, vector.uniform_u);
        std::printf("%s\t%d\t%.9g\n", vector.name, selected, vector.uniform_u);
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 2 && std::strcmp(argv[1], "--dump-parity-vectors") == 0) {
        return dump_parity_vectors();
    }

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

    // Model-free deterministic policy checks. The explicit uniform draw keeps
    // these assertions independent of Philox/RNG scheduling.
    const int32_t history[] = { 0 };
    float policy_logits[]    = { 4.0F, 3.0F, 3.5F };
    const int selected = sample_top_k_p_with_uniform(policy_logits, 3, 1.0F, 2, 1.0F, 2.0F, history, 1, 0.1F);
    if (!check(selected == 1, "repetition penalty is applied before top-k selection")) {
        return 1;
    }

    // The production wrapper must remain equivalent to the explicit-u seam:
    // it owns only the Philox draw, not sampling policy.
    const int64_t test_seed = 1006;
    const int64_t test_subseq = 17;
    float expected_logits[] = { 4.0F, 3.0F, 3.5F };
    float actual_logits[]   = { 4.0F, 3.0F, 3.5F };
    float expected_u = 0.0F;
    philox_uniform_fill(test_seed, test_subseq, 0u, &expected_u, 1);
    const int expected = sample_top_k_p_with_uniform(
        expected_logits, 3, 1.0F, 2, 1.0F, 2.0F, history, 1, expected_u);
    float actual_u = -2.0F;
    const int actual = sample_top_k_p(
        actual_logits, 3, 1.0F, 2, 1.0F, 2.0F, history, 1,
        test_seed, test_subseq, &actual_u);
    if (!check(actual == expected, "production wrapper matches explicit-u policy") ||
        !check(actual_u == expected_u, "production wrapper exposes its Philox draw")) {
        return 1;
    }

    float greedy_logits[] = { 2.0F, 7.0F, 3.0F };
    if (!check(sample_top_k_p_with_uniform(greedy_logits, 3, 0.0F, 50, 1.0F, 2.0F, history, 1, -1.0F) == 1,
               "greedy policy bypasses repetition penalty and RNG")) {
        return 1;
    }
    float greedy_wrapper_u = 123.0F;
    float greedy_wrapper_logits[] = { 2.0F, 7.0F, 3.0F };
    if (!check(sample_top_k_p(greedy_wrapper_logits, 3, 0.0F, 50, 1.0F, 2.0F, history, 1,
                              test_seed, test_subseq, &greedy_wrapper_u) == 1,
               "greedy wrapper bypasses Philox") ||
        !check(greedy_wrapper_u == -1.0F, "greedy wrapper reports no RNG draw")) {
        return 1;
    }

    float top_p_logits[] = { 4.0F, 3.0F, 2.0F, 1.0F };
    const int top_p_selected =
        sample_top_k_p_with_uniform(top_p_logits, 4, 1.0F, 0, 0.70F, 1.0F, nullptr, 0, 0.95F);
    if (!check(top_p_selected == 1, "top-p retains the boundary-crossing candidate")) {
        return 1;
    }
    return 0;
}

#include "sampling-graph.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <array>
#include <cstdio>
#include <cstddef>
#include <vector>

namespace {

struct Lane {
    std::array<float, 4> logits;
    float               uniform;
    int32_t             expected;
};

bool run_case(const char * name,
              float       temperature,
              const std::vector<Lane> & lanes,
              bool        diagnostics = false,
              const std::vector<int32_t> & forced_tokens = {},
              int         step_idx = 0) {
    const int n_lanes = static_cast<int>(lanes.size());
    const int n_steps = step_idx + 1;
    if (!forced_tokens.empty() && forced_tokens.size() != lanes.size()) {
        return false;
    }
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) {
        std::fprintf(stderr, "sampling graph contract: CPU backend unavailable\n");
        return false;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 64 + 256 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(backend);
        return false;
    }

    SamplerInputs sampler;
    sampler_inputs_build(ctx, &sampler, n_lanes, n_steps, 2);
    sampler.diagnostics.enabled     = diagnostics;
    sampler.diagnostics.target_frame = 0;
    sampler.diagnostics.target_step = step_idx;
    sampler.forced_enabled           = !forced_tokens.empty();
    ggml_tensor * logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, n_lanes);
    ggml_tensor * output = sampler_tail_build(ctx, logits, &sampler, step_idx);
    const bool shape_ok = output->ne[0] == n_lanes && output->ne[1] == 1;
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 256, false);
    ggml_build_forward_expand(graph, output);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    std::vector<float> logit_values;
    std::vector<float> state_values;
    // Earlier predictor rows must not leak into the requested step.
    std::vector<int32_t> forced_values(static_cast<size_t>(n_steps * n_lanes), 3);
    for (const Lane & lane : lanes) {
        logit_values.insert(logit_values.end(), lane.logits.begin(), lane.logits.end());
    }
    for (int step = 0; step < n_steps; ++step) {
        for (const Lane & lane : lanes) {
            state_values.insert(state_values.end(), {temperature, step == step_idx ? lane.uniform : 0.95F});
        }
    }
    if (!forced_tokens.empty()) {
        for (int lane = 0; lane < n_lanes; ++lane) {
            forced_values[static_cast<size_t>(step_idx * n_lanes + lane)] = forced_tokens[static_cast<size_t>(lane)];
        }
    }
    ggml_backend_tensor_set(logits, logit_values.data(), 0, logit_values.size() * sizeof(float));
    ggml_backend_tensor_set(sampler.state, state_values.data(), 0, state_values.size() * sizeof(float));
    sampler_forced_codes_upload(&sampler, forced_values.data(), n_lanes);
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "sampling graph contract: graph compute failed for %s\n", name);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    std::vector<int32_t> actual(lanes.size(), -1);
    ggml_backend_tensor_get(sampler.codes, actual.data(), static_cast<size_t>(step_idx + 1) * sampler.codes->nb[1],
                            actual.size() * sizeof(int32_t));
    bool passed = shape_ok;
    if (!shape_ok) {
        std::fprintf(stderr, "sampling graph contract: %s expected [%d] output\n", name, n_lanes);
    }
    for (size_t lane = 0; lane < lanes.size(); ++lane) {
        if (actual[lane] != lanes[lane].expected) {
            std::fprintf(stderr, "sampling graph contract: %s lane %zu expected %d, got %d\n", name, lane,
                         lanes[lane].expected, actual[lane]);
            passed = false;
        }
    }
    if (diagnostics) {
        passed = passed && sampler.diagnostics.top_ids && sampler.diagnostics.top_values
                 && sampler.diagnostics.masked_logits && sampler.diagnostics.probabilities
                 && sampler.diagnostics.cumsum && sampler.diagnostics.uniform && sampler.diagnostics.selected;
        if (passed) {
            int32_t top_ids[2] = { -1, -1 };
            float   masked[4]  = { 0.0F, 0.0F, 0.0F, 0.0F };
            float   top_values[2] = { 0.0F, 0.0F };
            float   probabilities[4] = { 0.0F, 0.0F, 0.0F, 0.0F };
            float   cdf[4]     = { 0.0F, 0.0F, 0.0F, 0.0F };
            float   actual_uniform = 0.0F;
            int32_t selected = -1;
            ggml_backend_tensor_get(sampler.diagnostics.top_ids, top_ids, 0, sizeof(top_ids));
            ggml_backend_tensor_get(sampler.diagnostics.masked_logits, masked, 0, sizeof(masked));
            ggml_backend_tensor_get(sampler.diagnostics.top_values, top_values, 0, sizeof(top_values));
            ggml_backend_tensor_get(sampler.diagnostics.probabilities, probabilities, 0, sizeof(probabilities));
            ggml_backend_tensor_get(sampler.diagnostics.cumsum, cdf, 0, sizeof(cdf));
            ggml_backend_tensor_get(sampler.diagnostics.uniform, &actual_uniform, 0, sizeof(actual_uniform));
            ggml_backend_tensor_get(sampler.diagnostics.selected, &selected, 0, sizeof(selected));
            passed = top_ids[0] == 3 && top_ids[1] == 1 && top_values[0] == 4.0F && top_values[1] == 3.0F
                     && masked[0] < -1.0e20F && masked[1] == 3.0F && masked[3] == 4.0F
                     && probabilities[1] > 0.26F && probabilities[1] < 0.28F && probabilities[3] > 0.72F
                     && probabilities[3] < 0.74F && cdf[0] < cdf[1] && cdf[1] < cdf[3]
                     && actual_uniform == 0.20F && selected == lanes[0].expected;
        }
        if (!passed) {
            std::fprintf(stderr, "sampling graph contract: diagnostic tensors missing or incorrect for %s\n", name);
        }
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return passed;
}

} // namespace

int main() {
    // The surviving top-k set is {1, 3}. Vocabulary-order CDF with u=.2
    // selects token 1, while probability-order CDF would select token 3.
    const bool stochastic = run_case("stochastic-vocabulary-order", 1.0F, {{{0.0F, 3.0F, 0.0F, 4.0F}, 0.20F, 1}}, true);
    // Greedy must remain argmax (3), even though the first surviving vocab id
    // is 1. This catches accidental reuse of the stochastic CDF ordering.
    const bool greedy = run_case("greedy-argmax", 1.0F, {{{0.0F, 3.0F, 0.0F, 4.0F}, -1.0F, 3}});
    // A forced predictor prefix must override the selected id inside the graph
    // so subsequent codebook steps consume the requested history.
    const bool forced = run_case("forced-prefix-token", 1.0F, {{{0.0F, 3.0F, 0.0F, 4.0F}, 0.20F, 2}}, false, {2});
    bool passed = stochastic && greedy && forced;
    for (int n_lanes : {1, 2, 3}) {
        std::vector<Lane> sampled = {
            {{0.0F, 3.0F, 0.0F, 4.0F}, 0.20F, 1},
            {{4.0F, 0.0F, 3.0F, 0.0F}, 0.95F, 2},
            {{0.0F, 4.0F, 3.0F, 0.0F}, 0.80F, 2},
        };
        sampled.resize(static_cast<size_t>(n_lanes));
        passed = run_case("all-stochastic", 1.0F, sampled, false, {}, 1) && passed;
        std::vector<Lane> greedy_lanes = sampled;
        const int32_t argmax[] = {3, 0, 1};
        for (int lane = 0; lane < n_lanes; ++lane) {
            greedy_lanes[static_cast<size_t>(lane)].uniform = -1.0F;
            greedy_lanes[static_cast<size_t>(lane)].expected = argmax[lane];
        }
        passed = run_case("all-greedy", 1.0F, greedy_lanes, false, {}, 1) && passed;
        std::vector<Lane> mixed = sampled;
        if (n_lanes > 1) {
            mixed[1] = greedy_lanes[1];
        }
        passed = run_case("mixed-stochastic-greedy", 1.0F, mixed, false, {}, 1) && passed;
        // Zero is a real token. -1 preserves the selected token for that slot.
        // Different rows/slots catch both the step stride and [1,N]/[N] shape.
        for (int step_idx : {0, 1, 2}) {
            std::vector<int32_t> forced_row = step_idx == 1 ? std::vector<int32_t>{0, 2, -1}
                                                          : std::vector<int32_t>{-1, 0, 3};
            forced_row.resize(static_cast<size_t>(n_lanes));
            std::vector<Lane> forced_lanes = mixed;
            for (int lane = 0; lane < n_lanes; ++lane) {
                if (forced_row[static_cast<size_t>(lane)] >= 0) {
                    forced_lanes[static_cast<size_t>(lane)].expected = forced_row[static_cast<size_t>(lane)];
                }
            }
            passed = run_case("forced-mixed-zero-and-sentinel", 1.0F, forced_lanes, false, forced_row, step_idx) && passed;
        }
    }
    if (passed) {
        std::puts("sampling graph contract: 21 cases passed (N=1/2/3, stochastic/greedy/forced)");
    }
    return passed ? 0 : 1;
}

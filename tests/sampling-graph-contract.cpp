#include "sampling-graph.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cstdio>
#include <cstddef>
#include <vector>

namespace {

bool run_case(const char * name,
              float       temperature,
              float       uniform,
              int         expected,
              bool        diagnostics = false,
              int         forced_token = -1,
              int         lanes = 1) {
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
    sampler_inputs_build(ctx, &sampler, lanes, 2, 2);
    sampler.diagnostics.enabled     = diagnostics;
    sampler.diagnostics.target_frame = 0;
    sampler.diagnostics.target_step = 1;
    sampler.forced_enabled           = forced_token >= 0;
    ggml_tensor * logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, lanes);
    ggml_tensor * output = sampler_tail_build(ctx, logits, &sampler, 1);
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
    std::vector<int32_t> forced_values;
    for (int lane = 0; lane < lanes; ++lane) {
        logit_values.insert(logit_values.end(), {0.0F, 3.0F, 0.0F, 4.0F});
        forced_values.push_back(0); // The unused first step must not leak into step 1.
    }
    for (int step = 0; step < 2; ++step) {
        for (int lane = 0; lane < lanes; ++lane) {
            state_values.insert(state_values.end(), {temperature, lane % 2 ? -1.0F : uniform});
            if (step == 1) forced_values.push_back(lane % 2 ? -1 : forced_token);
        }
    }
    ggml_backend_tensor_set(logits, logit_values.data(), 0, logit_values.size() * sizeof(float));
    ggml_backend_tensor_set(sampler.state, state_values.data(), 0, state_values.size() * sizeof(float));
    sampler_forced_codes_upload(&sampler, forced_token >= 0 ? forced_values.data() : nullptr, lanes);
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "sampling graph contract: graph compute failed for %s\n", name);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    std::vector<int32_t> actuals(static_cast<size_t>(lanes), -1);
    ggml_backend_tensor_get(sampler.codes, actuals.data(), 2 * sampler.codes->nb[1], actuals.size() * sizeof(int32_t));
    const int32_t actual = actuals[0];
    bool passed = true;
    for (int lane = 0; lane < lanes; ++lane) {
        passed = passed && actuals[static_cast<size_t>(lane)] == (lane % 2 ? 3 : expected);
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
                     && actual_uniform == 0.20F && selected == expected;
        }
        if (!passed) {
            std::fprintf(stderr, "sampling graph contract: diagnostic tensors missing or incorrect for %s\n", name);
        }
    }
    if (!passed) {
        std::fprintf(stderr, "sampling graph contract: %s expected %d, got %d\n", name, expected, actual);
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
    const bool stochastic = run_case("stochastic-vocabulary-order", 1.0F, 0.20F, 1, true);
    // Greedy must remain argmax (3), even though the first surviving vocab id
    // is 1. This catches accidental reuse of the stochastic CDF ordering.
    const bool greedy = run_case("greedy-argmax", 1.0F, -1.0F, 3);
    // A forced predictor prefix must override the selected id inside the graph
    // so subsequent codebook steps consume the requested history.
    const bool forced = run_case("forced-prefix-token", 1.0F, 0.20F, 2, false, 2);
    const bool batch_two = run_case("batched-mixed-two", 1.0F, 0.20F, 1, false, -1, 2);
    const bool batch_three = run_case("batched-mixed-three", 1.0F, 0.20F, 1, false, -1, 3);
    const bool batch_forced = run_case("batched-forced-mixed", 1.0F, 0.20F, 2, false, 2, 3);
    return (stochastic && greedy && forced && batch_two && batch_three && batch_forced) ? 0 : 1;
}

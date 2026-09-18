#include "sampling-graph.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cstdio>
#include <cstddef>
#include <vector>

namespace {

bool run_case(const char * name, float temperature, float uniform, int expected) {
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
    sampler_inputs_build(ctx, &sampler, 1, 1, 2);
    ggml_tensor * logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 1);
    ggml_tensor * output = sampler_tail_build(ctx, logits, &sampler, 0);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 256, false);
    ggml_build_forward_expand(graph, output);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    const float logit_values[] = { 0.0F, 3.0F, 0.0F, 4.0F };
    const float state_values[] = { temperature, uniform };
    ggml_backend_tensor_set(logits, logit_values, 0, sizeof(logit_values));
    ggml_backend_tensor_set(sampler.state, state_values, 0, sizeof(state_values));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "sampling graph contract: graph compute failed for %s\n", name);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    int32_t actual = -1;
    ggml_backend_tensor_get(sampler.codes, &actual, sampler.codes->nb[1], sizeof(actual));
    const bool passed = actual == expected;
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
    const bool stochastic = run_case("stochastic-vocabulary-order", 1.0F, 0.20F, 1);
    // Greedy must remain argmax (3), even though the first surviving vocab id
    // is 1. This catches accidental reuse of the stochastic CDF ordering.
    const bool greedy = run_case("greedy-argmax", 1.0F, -1.0F, 3);
    return (stochastic && greedy) ? 0 : 1;
}

#pragma once
// sampling-graph.h: the predictor sampling tail in standard ops, so
// the whole frame decodes on the backend without per step logits
// readbacks. Each tail applies the per step temperature, keeps the
// top_k candidates, then draws one token where the cdf crosses the
// per step uniform u. top_k bakes from the generation defaults at
// build; nucleus filtering is not applied. Greedy slots upload u = -1,
// a private sentinel that is blended with a direct argmax after the CDF walk.
//
// Sampler inputs and the codes accumulator live in a caller owned
// persistent context, never in gallocr input buffers. The uniform
// draws stay on the host (philox depends only on seed and subsequence)
// and upload once per frame inside the state tensor.

#include "ggml-backend.h"
#include "ggml.h"
#include "philox.h"

#include <cmath>
#include <vector>

struct SamplerInputs {
    struct Diagnostics {
        bool                 enabled      = false;
        int                  target_step  = 9;
        struct ggml_tensor * scaled_logits = nullptr;
        struct ggml_tensor * top_order    = nullptr;
        struct ggml_tensor * top_ids      = nullptr;
        struct ggml_tensor * top_values   = nullptr;
        struct ggml_tensor * masked_logits = nullptr;
        struct ggml_tensor * probabilities = nullptr;
        struct ggml_tensor * cumsum       = nullptr;
        struct ggml_tensor * uniform      = nullptr;
        struct ggml_tensor * selected     = nullptr;
    } diagnostics;

    struct ggml_tensor * state   = nullptr;  // [2, N, n_steps] f32, per slot (temperature, u)
    struct ggml_tensor * codes   = nullptr;  // [N, n_codes] i32, row g holds code g of every slot
    struct ggml_tensor * forced  = nullptr;  // [N, n_steps] f32, -1 means use sampled id
    int                  n_steps = 0;        // sampled codes per frame (semantic + acoustic)
    int                  N       = 0;
    int                  top_k   = 0;        // candidate count baked into every tail
};

// Create the sampler tensors inside pctx. The caller allocates pctx
// into a persistent backend buffer afterwards.
static inline void sampler_inputs_build(struct ggml_context * pctx, SamplerInputs * sp, int N, int n_steps, int top_k) {
    sp->n_steps = n_steps;
    sp->N       = N;
    sp->top_k   = top_k;
    sp->state   = ggml_new_tensor_3d(pctx, GGML_TYPE_F32, 2, N, n_steps);
    sp->codes   = ggml_new_tensor_2d(pctx, GGML_TYPE_I32, N, n_steps + 1);
    sp->forced  = ggml_new_tensor_2d(pctx, GGML_TYPE_F32, N, n_steps);
    ggml_set_name(sp->state, "sampler.state");
    ggml_set_name(sp->codes, "sampler.codes");
    ggml_set_name(sp->forced, "sampler.forced");
}

// Upload optional diagnostic predictor-prefix overrides. Values are arranged
// row-major as [step][slot], with -1 meaning that the graph keeps its sampled
// token. The override is applied inside each sampling tail, so later
// predictor steps consume the forced history rather than a host-side result
// that arrives after the frame graph has already run.
static inline void sampler_forced_codes_upload(SamplerInputs * sp, const int32_t * forced_codes, int N) {
    std::vector<float> values((size_t) sp->n_steps * (size_t) N, -1.0f);
    if (forced_codes) {
        for (int g = 0; g < sp->n_steps; g++) {
            for (int i = 0; i < N; i++) {
                values[(size_t) g * (size_t) N + (size_t) i] =
                    (float) forced_codes[(size_t) g * (size_t) N + (size_t) i];
            }
        }
    }
    ggml_backend_tensor_set(sp->forced, values.data(), 0, values.size() * sizeof(float));
}

// Upload the per frame sampler state. Greedy slots (temperature <= 0)
// carry temperature 1 and u -1. The negative sentinel is reserved for greedy
// because philox_uniform_fill always returns a value in (0, 1).
// subseq_base[i] indexes slot i's philox
// stream: draw g uses subsequence subseq_base[i] + 1 + g.
static inline void sampler_inputs_upload(SamplerInputs * sp,
                                         const float *   temperature,
                                         const int64_t * seed,
                                         const int64_t * subseq_base,
                                         int             N) {
    std::vector<float> st((size_t) 2 * (size_t) N * (size_t) sp->n_steps);

    for (int g = 0; g < sp->n_steps; g++) {
        for (int i = 0; i < N; i++) {
            const bool greedy = temperature[i] <= 0.0f;
            float      u      = -1.0f; // greedy sentinel: direct argmax
            if (!greedy) {
                philox_uniform_fill(seed[i], subseq_base[i] + 1 + g, 0u, &u, 1);
            }
            float * row = st.data() + ((size_t) g * (size_t) N + (size_t) i) * 2;
            row[0]      = greedy ? 1.0f : temperature[i];
            row[1]      = u;
        }
    }
    ggml_backend_tensor_set(sp->state, st.data(), 0, st.size() * sizeof(float));
}

// One sampling tail: reads this step's per slot (temperature, u) from
// the state, draws one token id per slot from logits [n_vocab, N] and
// writes the N ids to row step_idx + 1 of the codes accumulator. Every
// gather batches over the slot dim through 3D get_rows.
static inline struct ggml_tensor * sampler_tail_build(struct ggml_context * gctx,
                                                      struct ggml_tensor *  logits,
                                                      SamplerInputs *       sp,
                                                      int                   step_idx) {
    const int64_t n_vocab = logits->ne[0];
    const int64_t N       = logits->ne[1];

    struct ggml_tensor * temp =
        ggml_view_2d(gctx, sp->state, 1, N, sp->state->nb[1], (size_t) step_idx * sp->state->nb[2]);
    struct ggml_tensor * u =
        ggml_view_2d(gctx, sp->state, 1, N, sp->state->nb[1], sp->state->nb[0] + (size_t) step_idx * sp->state->nb[2]);

    struct ggml_tensor * cur = ggml_div(gctx, logits, temp);
    const bool capture_diagnostics = sp->diagnostics.enabled && step_idx == sp->diagnostics.target_step && N == 1;
    if (capture_diagnostics) {
        sp->diagnostics.scaled_logits = cur;
        ggml_set_output(cur);
    }

    // Keep the top-k set, but restore the original vocabulary order before
    // the CDF walk. torch.multinomial (the FasterQwen reference) consumes
    // probabilities in vocabulary order; walking a probability-sorted list
    // changes deterministic Philox replay for the same uniform u.
    if (sp->top_k > 0 && sp->top_k < n_vocab) {
        struct ggml_tensor * order = ggml_argsort(gctx, cur, GGML_SORT_ORDER_DESC);
        struct ggml_tensor * idx =
            ggml_cont(gctx, ggml_view_2d(gctx, order, sp->top_k, N, order->nb[1], 0));  // [top_k, N] i32
        struct ggml_tensor * a3d = ggml_reshape_3d(gctx, cur, 1, n_vocab, N);
        struct ggml_tensor * top_values = ggml_get_rows(gctx, a3d, idx); // [1, top_k, N]

        if (capture_diagnostics) {
            sp->diagnostics.top_order  = order;
            sp->diagnostics.top_ids    = idx;
            sp->diagnostics.top_values = top_values;
            ggml_set_output(order);
            ggml_set_output(idx);
            ggml_set_output(top_values);
        }

        // Scatter the selected logits into a full-vocabulary tensor filled
        // with -inf. Softmax/cumsum now follow token-id order while keeping
        // exactly the same top-k mask.
        struct ggml_tensor * full = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, 1, n_vocab, N);
        full = ggml_fill(gctx, full, -INFINITY);
        full = ggml_set_rows(gctx, full, top_values, idx);
        cur  = ggml_reshape_2d(gctx, full, n_vocab, N);
    }

    if (capture_diagnostics) {
        // Materialize an owning graph tensor before softmax. Retaining only
        // the reshape/set_rows view is insufficient on CUDA because the
        // allocator may recycle the view source after its last consumer.
        cur = ggml_dup(gctx, cur);
        sp->diagnostics.masked_logits = cur;
        ggml_set_output(cur);
    }

    // draw one token per slot: find where the cdf crosses u
    struct ggml_tensor * probs  = ggml_soft_max(gctx, cur);
    struct ggml_tensor * cumsum = ggml_cumsum(gctx, probs);

    if (capture_diagnostics) {
        sp->diagnostics.probabilities = probs;
        sp->diagnostics.cumsum        = cumsum;
        sp->diagnostics.uniform       = u;
        ggml_set_output(probs);
        ggml_set_output(cumsum);
        ggml_set_output(u);
    }

    struct ggml_tensor * diff       = ggml_sub(gctx, cumsum, u);
    struct ggml_tensor * cross_mask = ggml_step(gctx, diff);
    struct ggml_tensor * idxf       = ggml_sum_rows(gctx, cross_mask);  // [1, N]
    struct ggml_tensor * idx =
        ggml_cast(gctx, ggml_scale_bias(gctx, idxf, -1.0f, (float) cross_mask->ne[0]), GGML_TYPE_I32);

    // Vocabulary-order CDF is required for stochastic parity. Preserve the
    // separate greedy contract explicitly instead of relying on candidate
    // ordering: u < 0 is the private greedy sentinel.
    struct ggml_tensor * greedy_mask = ggml_step(gctx, ggml_scale(gctx, u, -1.0f));
    struct ggml_tensor * idx_1d      = ggml_reshape_1d(gctx, idx, N);
    struct ggml_tensor * idx_f       = ggml_cast(gctx, idx_1d, GGML_TYPE_F32);
    struct ggml_tensor * argmax_f    = ggml_cast(gctx, ggml_argmax(gctx, logits), GGML_TYPE_F32);
    struct ggml_tensor * selected_f  = ggml_add(
        gctx, idx_f, ggml_mul(gctx, greedy_mask, ggml_sub(gctx, argmax_f, idx_f)));
    struct ggml_tensor * forced =
        ggml_view_2d(gctx, sp->forced, 1, N, sp->forced->nb[1], (size_t) step_idx * sp->forced->nb[1]);
    struct ggml_tensor * forced_mask = ggml_step(gctx, forced);
    selected_f = ggml_add(gctx, selected_f, ggml_mul(gctx, forced_mask, ggml_sub(gctx, forced, selected_f)));
    struct ggml_tensor * ids = ggml_cast(gctx, selected_f, GGML_TYPE_I32);
    if (capture_diagnostics) {
        sp->diagnostics.selected = ids;
        ggml_set_output(ids);
    }
    struct ggml_tensor * dst = ggml_view_1d(gctx, sp->codes, N, (size_t) (step_idx + 1) * sp->codes->nb[1]);
    return ggml_cpy(gctx, ids, dst);
}

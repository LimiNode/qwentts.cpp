#pragma once
// sampling.h: token sampling for the Talker LM and the CodePredictor MTP
// head. Pipeline matches the HuggingFace generate() chain in F32 :
//   repetition_penalty -> temperature -> top_k -> top_p -> softmax -> multinomial
// The multinomial uniform draw comes from philox_uniform_fill so the
// sequence stays byte for byte aligned with the patched torch.multinomial
// in tests/debug-tts-cossim.py.
//
// apply_suppress is exposed separately so callers can mask the codec
// reserved range before invoking the sampler. The Talker masks
// [vocab - 1024, vocab) except codec_eos before calling sample_top_k_p,
// the CodePredictor does not need any suppression.

#include "philox.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

struct TokenProb {
    int   id;
    float prob;
};

// Bounded diagnostics derived from the final multinomial weights after
// suppression, repetition penalty, temperature, top-k and top-p. The helper
// observes the sampler output without changing it or consuming RNG state.
struct SamplingDiagnostics {
    float                  selected_probability = 0.0f;
    float                  eos_probability      = 0.0f;
    int                    eos_rank             = 0; // 0 means filtered/ineligible
    int                    candidate_count      = 0;
    std::vector<TokenProb> top_tokens;
};

// Exact replay of the host sampler's final F32 accumulator. The input must be
// the post-softmax unnormalized weights left in-place by
// sample_top_k_p_with_uniform(). This helper is diagnostic-only and does not
// consume RNG state or mutate the weights.
struct SamplingAccumulatorDiagnostics {
    float              sum      = 0.0f;
    float              target   = 0.0f;
    int                selected = -1;
    std::vector<float> cdf;
};

static SamplingAccumulatorDiagnostics sampling_accumulator_diagnostics_from_weights(const float * weights,
                                                                                      int           V,
                                                                                      float         uniform_u) {
    SamplingAccumulatorDiagnostics result;
    if (!weights || V <= 0) {
        return result;
    }

    result.cdf.resize((size_t) V);
    for (int i = 0; i < V; i++) {
        result.sum += weights[i];
    }
    result.target = uniform_u * result.sum;

    float acc = 0.0f;
    for (int i = 0; i < V; i++) {
        acc += weights[i];
        result.cdf[(size_t) i] = acc;
        if (result.selected < 0 && acc >= result.target) {
            result.selected = i;
        }
    }
    if (result.selected < 0) {
        result.selected = V - 1;
    }
    return result;
}

static SamplingDiagnostics sampling_diagnostics_from_weights(const float * weights,
                                                              int           V,
                                                              int           selected_id,
                                                              int           eos_id,
                                                              int           top_n) {
    SamplingDiagnostics result;
    if (!weights || V <= 0) {
        return result;
    }

    double sum = 0.0;
    for (int i = 0; i < V; i++) {
        const float weight = weights[i];
        if (std::isfinite(weight) && weight > 0.0f) {
            sum += weight;
            result.candidate_count++;
        }
    }
    if (!(sum > 0.0)) {
        return result;
    }

    if (selected_id >= 0 && selected_id < V) {
        result.selected_probability = static_cast<float>(weights[selected_id] / sum);
    }
    if (eos_id >= 0 && eos_id < V && std::isfinite(weights[eos_id]) && weights[eos_id] > 0.0f) {
        result.eos_probability = static_cast<float>(weights[eos_id] / sum);
        result.eos_rank        = 1;
        for (int i = 0; i < V; i++) {
            if (weights[i] > weights[eos_id]) {
                result.eos_rank++;
            }
        }
    }

    if (top_n > 0) {
        std::vector<TokenProb> candidates;
        candidates.reserve(static_cast<size_t>(result.candidate_count));
        for (int i = 0; i < V; i++) {
            if (std::isfinite(weights[i]) && weights[i] > 0.0f) {
                candidates.push_back({ i, static_cast<float>(weights[i] / sum) });
            }
        }
        const size_t keep = std::min(static_cast<size_t>(top_n), candidates.size());
        std::partial_sort(candidates.begin(), candidates.begin() + keep, candidates.end(),
                          [](const TokenProb & a, const TokenProb & b) {
                              return a.prob > b.prob || (a.prob == b.prob && a.id < b.id);
                          });
        candidates.resize(keep);
        result.top_tokens = std::move(candidates);
    }
    return result;
}

// Mask logits in [lo, hi) to -inf, except keep is left untouched.
static inline void apply_suppress(float * logits, int V, int lo, int hi, int keep) {
    if (lo < 0) {
        lo = 0;
    }
    if (hi > V) {
        hi = V;
    }
    for (int i = lo; i < hi; i++) {
        if (i == keep) {
            continue;
        }
        logits[i] = -INFINITY;
    }
}

// A successful synthesis must emit at least one codec frame. Keep EOS
// available for later steps, but make the first-frame contract explicit so
// callers never receive a successful empty stream.
static inline void suppress_initial_eos(float * logits, int V, int eos_id, int step) {
    if (step == 0 && eos_id >= 0 && eos_id < V) {
        logits[(size_t) eos_id] = -INFINITY;
    }
}

// Repetition penalty over unique tokens in history (HF rule):
//   if score >= 0 -> score / penalty
//   if score <  0 -> score * penalty
// Each token is touched at most once per call.
static inline void apply_repetition_penalty(float *         logits,
                                            int             V,
                                            const int32_t * history,
                                            int             n_history,
                                            float           penalty) {
    if (penalty == 1.0f || n_history <= 0) {
        return;
    }
    static thread_local std::vector<uint8_t> seen_buf;
    seen_buf.assign((size_t) V, 0);
    for (int h = 0; h < n_history; h++) {
        int32_t tok = history[h];
        if (tok < 0 || tok >= V) {
            continue;
        }
        if (seen_buf[(size_t) tok]) {
            continue;
        }
        seen_buf[(size_t) tok] = 1;
        float s                = logits[tok];
        logits[tok]            = (s < 0.0f) ? s * penalty : s / penalty;
    }
}

// Stochastic sampler. Pipeline mirrors HF generate() in F32 :
//   1. repetition_penalty(history)
//   2. temperature divide
//   3. top_k mask (skipped when k <= 0 or k >= V)
//   4. top_p nucleus mask (skipped when p >= 1.0)
//   5. softmax
//   6. multinomial via philox_uniform_fill(seed, philox_subseq, 0)
//
// Greedy path: temperature <= 0 returns argmax over the suppressed
// logits, no rep_pen, no philox draw.
//
// Buffers are thread_local to avoid alloc per token.
// Deterministic sampler seam used by model-free parity tests. A uniform value
// in [0, 1) is supplied explicitly, so tests can compare native and Python
// categorical selection without coupling the policy test to an RNG.
static int sample_top_k_p_with_uniform(float *         logits,
                                       int             V,
                                       float           temperature,
                                       int             top_k,
                                       float           top_p,
                                       float           rep_pen,
                                       const int32_t * history,
                                       int             n_history,
                                       float           uniform_u) {
    if (temperature <= 0.0f) {
        return (int) (std::max_element(logits, logits + V) - logits);
    }

    apply_repetition_penalty(logits, V, history, n_history, rep_pen);

    float inv_temp = 1.0f / temperature;
    for (int i = 0; i < V; i++) {
        logits[i] *= inv_temp;
    }

    static thread_local std::vector<float>     tmp_buf;
    static thread_local std::vector<TokenProb> sorted_buf;

    if (top_k > 0 && top_k < V) {
        tmp_buf.resize((size_t) V);
        std::memcpy(tmp_buf.data(), logits, (size_t) V * sizeof(float));
        std::nth_element(tmp_buf.begin(), tmp_buf.begin() + (top_k - 1), tmp_buf.end(), std::greater<float>());
        float threshold = tmp_buf[(size_t) (top_k - 1)];
        for (int i = 0; i < V; i++) {
            if (logits[i] < threshold) {
                logits[i] = -INFINITY;
            }
        }
    }

    if (top_p > 0.0f && top_p < 1.0f) {
        // Full vocab softmax for an exact cumsum boundary (matches HF
        // TopPLogitsWarper which softmaxes the sorted tensor in place).
        float max_logit = -INFINITY;
        for (int i = 0; i < V; i++) {
            if (logits[i] > max_logit) {
                max_logit = logits[i];
            }
        }
        float sum_exp = 0.0f;
        for (int i = 0; i < V; i++) {
            sum_exp += expf(logits[i] - max_logit);
        }
        float inv_sum = 1.0f / sum_exp;

        // Compact only tokens above a relative cutoff. exp(-16) is ~1e-7
        // so the dropped mass stays well below any reachable nucleus
        // boundary, and the sort runs on a small set (typically less
        // than top_k entries).
        float cutoff = max_logit - 16.0f;
        sorted_buf.clear();
        for (int i = 0; i < V; i++) {
            if (logits[i] >= cutoff) {
                float prob = expf(logits[i] - max_logit) * inv_sum;
                sorted_buf.push_back({ i, prob });
            } else {
                logits[i] = -INFINITY;
            }
        }

        int K = (int) sorted_buf.size();
        if (K > 0) {
            std::sort(sorted_buf.begin(), sorted_buf.end(),
                      [](const TokenProb & a, const TokenProb & b) { return a.prob > b.prob; });
            // HF convention: keep tokens until the cumulative probability
            // crosses top_p, drop the rest. Test before accumulate so the
            // first crossing entry is kept.
            float cum = 0.0f;
            for (int i = 0; i < K; i++) {
                if (i > 0 && cum >= top_p) {
                    logits[sorted_buf[(size_t) i].id] = -INFINITY;
                }
                cum += sorted_buf[(size_t) i].prob;
            }
        }
    }

    // Final softmax then philox driven multinomial. We keep the
    // unnormalized exponentials and draw u in [0, sum) which is
    // mathematically identical to softmax + u in [0, 1) but spares a
    // pass.
    float max_val = -INFINITY;
    for (int i = 0; i < V; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
        }
    }
    float sum = 0.0f;
    for (int i = 0; i < V; i++) {
        logits[i] = expf(logits[i] - max_val);
        sum += logits[i];
    }

    float r   = uniform_u * sum;
    float acc = 0.0f;
    for (int i = 0; i < V; i++) {
        acc += logits[i];
        if (acc >= r) {
            return i;
        }
    }
    return V - 1;
}

// Stochastic production entry point. It owns the Philox draw and delegates
// the policy itself to the deterministic seam above.
static int sample_top_k_p(float *         logits,
                          int             V,
                          float           temperature,
                          int             top_k,
                          float           top_p,
                          float           rep_pen,
                          const int32_t * history,
                          int             n_history,
                          int64_t         seed,
                          int64_t         philox_subseq,
                          float *         dump_u_out) {
    float uniform_u = -1.0f;
    if (temperature > 0.0f) {
        philox_uniform_fill(seed, philox_subseq, 0u, &uniform_u, 1);
    }
    if (dump_u_out) {
        *dump_u_out = uniform_u;
    }
    return sample_top_k_p_with_uniform(logits, V, temperature, top_k, top_p, rep_pen, history, n_history,
                                       uniform_u);
}

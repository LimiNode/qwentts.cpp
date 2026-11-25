#pragma once

// Research-only Talker policy. Not part of qwen.h or the production build.
#include "sampling.h"

enum class ResearchEosMode { Baseline, Penalty, AlternativeOnce };

struct ResearchEosControl {
    ResearchEosMode mode = ResearchEosMode::Baseline;
    float penalty = 0.0f;
    bool alternative_used = false;
};

static inline bool research_eos_parse(const char * text, ResearchEosControl * out) {
    ResearchEosControl parsed;
    if (!text || !*text || std::strcmp(text, "baseline") == 0) {
        *out = parsed;
        return true;
    }
    if (std::strcmp(text, "penalty:2") == 0 || std::strcmp(text, "penalty:4") == 0 ||
        std::strcmp(text, "penalty:6") == 0) {
        parsed.mode = ResearchEosMode::Penalty;
        parsed.penalty = static_cast<float>(text[8] - '0');
    } else if (std::strcmp(text, "alternative-once:8") == 0) {
        parsed.mode = ResearchEosMode::AlternativeOnce;
    } else {
        return false;
    }
    *out = parsed;
    return true;
}

struct ResearchEosSelection {
    int initial = -1;
    int effective = -1;
    bool intervened = false;
};

// Inputs already have reserved/minimum masks and the existing guard applied.
// The raw-logit subtraction precedes repetition, temperature, top-k and top-p.
// A one-shot alternative reuses the same uniform/subsequence and the existing
// prefill/KV state; it reruns the sampler, not the model. The fixed <8-frame
// window is a discovery intervention, not a premature-EOS detector.
static ResearchEosSelection research_eos_sample(
    ResearchEosControl & control, float * logits, int vocab, int eos, int frame,
    float temperature, int top_k, float top_p, float repetition,
    const int32_t * history, int history_size, int64_t seed, int64_t subsequence,
    float * uniform_out, std::vector<float> * effective_raw = nullptr) {
    const bool alternative_eligible = control.mode == ResearchEosMode::AlternativeOnce &&
        !control.alternative_used && frame >= 2 && frame < 8;
    std::vector<float> original;
    if (alternative_eligible) original.assign(logits, logits + vocab);
    if (control.mode == ResearchEosMode::Penalty) logits[eos] -= control.penalty;
    if (effective_raw) effective_raw->assign(logits, logits + vocab);

    float uniform = -1.0f;
    ResearchEosSelection result;
    result.initial = sample_top_k_p(logits, vocab, temperature, top_k, top_p,
        repetition, history, history_size, seed, subsequence, &uniform);
    result.effective = result.initial;
    if (alternative_eligible && result.initial == eos) {
        // Do not invoke the existing sampler on an empty/all-masked support.
        bool has_alternative = false;
        for (int i = 0; i < vocab; ++i) {
            if (i != eos && std::isfinite(original[static_cast<size_t>(i)])) {
                has_alternative = true;
                break;
            }
        }
        if (has_alternative) {
            std::copy(original.begin(), original.end(), logits);
            logits[eos] = -INFINITY;
            if (effective_raw) effective_raw->assign(logits, logits + vocab);
            result.effective = sample_top_k_p_with_uniform(logits, vocab, temperature,
                top_k, top_p, repetition, history, history_size, uniform);
            result.intervened = true;
            control.alternative_used = true;
        }
    }
    if (uniform_out) *uniform_out = uniform;
    return result;
}

#pragma once

#include <algorithm>
#include <cmath>

// A bounded, opt-in termination policy for stochastic autoregressive audio
// generation.  The policy does not alter the categorical sampler below the
// soft threshold; it only adds a finite EOS bias in the upper tail and exposes
// a hard terminal condition to the caller.
struct EosGuardConfig {
    bool  enabled          = false;
    float start_ratio      = 0.6F;
    float max_ratio        = 1.2F;
    float force_ratio      = 1.5F;
    float max_boost        = 25.0F;
    float voice_multiplier = 1.5F;
    int   min_expected     = 24;
    int   frames_per_text  = 4;
};

struct EosGuardPlan {
    int   expected_frames = 0;
    int   soft_start      = 0;
    int   max_boost_step  = 0;
    int   force_step      = 0;
    float max_boost       = 0.0F;
};

static inline bool eos_guard_config_valid(const EosGuardConfig & cfg) {
    return std::isfinite(cfg.start_ratio) && std::isfinite(cfg.max_ratio) &&
           std::isfinite(cfg.force_ratio) && std::isfinite(cfg.max_boost) &&
           std::isfinite(cfg.voice_multiplier) && cfg.start_ratio >= 0.0F &&
           cfg.start_ratio < cfg.max_ratio && cfg.max_ratio < cfg.force_ratio &&
           cfg.max_boost >= 0.0F && cfg.voice_multiplier > 0.0F &&
           cfg.min_expected > 0 && cfg.frames_per_text > 0;
}

static inline int eos_guard_expected_frames(int text_tokens, bool has_voice, const EosGuardConfig & cfg) {
    const int base = std::max(cfg.min_expected, std::max(1, text_tokens) * cfg.frames_per_text);
    const float scaled = has_voice ? (float) base * cfg.voice_multiplier : (float) base;
    return std::max(1, (int) std::ceil(scaled));
}

static inline EosGuardPlan eos_guard_plan(int expected_frames, const EosGuardConfig & cfg) {
    EosGuardPlan plan;
    plan.expected_frames = std::max(1, expected_frames);
    plan.soft_start = std::max(1, (int) std::lround((float) plan.expected_frames * cfg.start_ratio));
    plan.max_boost_step = std::max(plan.soft_start + 1,
                                   (int) std::lround((float) plan.expected_frames * cfg.max_ratio));
    plan.force_step = std::max(plan.max_boost_step + 1,
                               (int) std::lround((float) plan.expected_frames * cfg.force_ratio));
    plan.max_boost = std::max(0.0F, cfg.max_boost);
    return plan;
}

static inline float eos_guard_boost(const EosGuardPlan & plan, int step) {
    if (step < plan.soft_start || plan.max_boost <= 0.0F) {
        return 0.0F;
    }
    const float progress = std::min(
        1.0F,
        (float) (step - plan.soft_start) / (float) (plan.max_boost_step - plan.soft_start));
    return plan.max_boost * progress;
}

static inline bool eos_guard_force(const EosGuardPlan & plan, int step) {
    return step >= plan.force_step;
}

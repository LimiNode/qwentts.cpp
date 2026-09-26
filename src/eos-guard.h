#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

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
    std::int64_t expected_frames = 0;
    std::int64_t soft_start      = 0;
    std::int64_t max_boost_step  = 0;
    std::int64_t force_step      = 0;
    float max_boost       = 0.0F;
};

static inline bool eos_guard_config_valid(const EosGuardConfig & cfg) {
    return std::isfinite(cfg.start_ratio) && std::isfinite(cfg.max_ratio) &&
           std::isfinite(cfg.force_ratio) && std::isfinite(cfg.max_boost) &&
           std::isfinite(cfg.voice_multiplier) && cfg.start_ratio >= 0.0F &&
           cfg.start_ratio < cfg.max_ratio && cfg.max_ratio < cfg.force_ratio &&
           cfg.max_boost >= 0.0F && cfg.max_boost <= 10000.0F &&
           cfg.voice_multiplier > 0.0F && cfg.voice_multiplier <= 1000.0F &&
           cfg.min_expected > 0 && cfg.min_expected <= 1000000000 &&
           cfg.frames_per_text > 0 && cfg.frames_per_text <= 1000000;
}

static inline std::int64_t eos_guard_expected_frames(int text_tokens, bool has_voice,
                                                     const EosGuardConfig & cfg) {
    const std::int64_t tokens = std::max<std::int64_t>(1, text_tokens);
    const std::int64_t base = std::max<std::int64_t>(
        cfg.min_expected, tokens * (std::int64_t) cfg.frames_per_text);
    const long double scaled = (long double) base *
                               (has_voice ? (long double) cfg.voice_multiplier : 1.0L);
    if (scaled >= (long double) std::numeric_limits<std::int64_t>::max()) {
        return std::numeric_limits<std::int64_t>::max();
    }
    return std::max<std::int64_t>(1, (std::int64_t) std::ceil(scaled));
}

static inline std::int64_t eos_guard_ratio_step(std::int64_t expected_frames, float ratio) {
    const long double scaled = (long double) expected_frames * (long double) ratio;
    if (scaled >= (long double) std::numeric_limits<std::int64_t>::max()) {
        return std::numeric_limits<std::int64_t>::max();
    }
    return std::max<std::int64_t>(1, (std::int64_t) std::llround(scaled));
}

static inline EosGuardPlan eos_guard_plan(std::int64_t expected_frames, const EosGuardConfig & cfg,
                                           std::int64_t max_steps = std::numeric_limits<int>::max()) {
    EosGuardPlan plan;
    max_steps = std::max<std::int64_t>(1, max_steps);
    plan.expected_frames = std::clamp<std::int64_t>(expected_frames, 1, max_steps);
    plan.soft_start = std::clamp<std::int64_t>(eos_guard_ratio_step(plan.expected_frames, cfg.start_ratio), 1,
                                               max_steps);
    const std::int64_t next_after_soft = plan.soft_start < max_steps ? plan.soft_start + 1 : max_steps;
    plan.max_boost_step = std::clamp<std::int64_t>(
        std::max(next_after_soft, eos_guard_ratio_step(plan.expected_frames, cfg.max_ratio)), 1, max_steps);
    const std::int64_t next_after_max = plan.max_boost_step < max_steps ? plan.max_boost_step + 1 : max_steps;
    plan.force_step = std::clamp<std::int64_t>(
        std::max(next_after_max, eos_guard_ratio_step(plan.expected_frames, cfg.force_ratio)), 1, max_steps);
    plan.max_boost = std::max(0.0F, cfg.max_boost);
    return plan;
}

static inline float eos_guard_boost(const EosGuardPlan & plan, std::int64_t step) {
    if (step < plan.soft_start || plan.max_boost <= 0.0F) {
        return 0.0F;
    }
    if (plan.max_boost_step <= plan.soft_start) {
        return plan.max_boost;
    }
    const float progress = std::min(
        1.0F,
        (float) (step - plan.soft_start) / (float) (plan.max_boost_step - plan.soft_start));
    return plan.max_boost * progress;
}

static inline bool eos_guard_force(const EosGuardPlan & plan, std::int64_t step) {
    return step >= plan.force_step;
}

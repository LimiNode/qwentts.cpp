#include "eos-guard.h"

#include <cmath>
#include <cstdio>

namespace {
bool check(bool value, const char *message) {
    if (!value) {
        std::fprintf(stderr, "eos guard contract failed: %s\n", message);
    }
    return value;
}
} // namespace

int main() {
    EosGuardConfig cfg;
    if (!check(eos_guard_expected_frames(2, false, cfg) == 24, "minimum expected length") ||
        !check(eos_guard_expected_frames(10, true, cfg) == 60, "voice multiplier") ) {
        return 1;
    }

    const EosGuardPlan plan = eos_guard_plan(100, cfg);
    if (!check(plan.soft_start == 60, "soft threshold") ||
        !check(plan.max_boost_step == 120, "maximum boost threshold") ||
        !check(plan.force_step == 150, "force threshold") ||
        !check(eos_guard_boost(plan, 59) == 0.0F, "no boost before threshold") ||
        !check(std::fabs(eos_guard_boost(plan, 120) - 25.0F) < 1e-6F, "boost reaches cap") ||
        !check(eos_guard_boost(plan, 149) <= 25.0F, "boost is capped") ||
        !check(!eos_guard_force(plan, 149) && eos_guard_force(plan, 150), "hard threshold") ) {
        return 1;
    }

    cfg.max_boost = -1.0F;
    if (!check(!eos_guard_config_valid(cfg), "invalid negative boost is rejected")) {
        return 1;
    }
    const EosGuardPlan disabled_boost = eos_guard_plan(10, cfg);
    if (!check(eos_guard_boost(disabled_boost, disabled_boost.soft_start) == 0.0F,
               "negative boost clamps to zero")) {
        return 1;
    }
    return 0;
}

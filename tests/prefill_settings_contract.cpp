#include "research_prefill.h"

#include <cstdio>

static int check(bool condition, const char * name) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", name);
        return 1;
    }
    return 0;
}

int main() {
    int                     failures = 0;
    ResearchPrefillSettings settings;
    std::string             error;
    failures +=
        check(research_prefill_parse(nullptr, nullptr, nullptr, nullptr, nullptr, settings, error), "defaults parse");
    failures += check(!settings.profile && !settings.last_only && !settings.kv_f16 && settings.window == 256 &&
                          !settings.retain && error.empty(),
                      "baseline defaults");
    failures += check(research_prefill_parse("1", "1", "f16", "64", "1", settings, error), "all options parse");
    failures +=
        check(settings.profile && settings.last_only && settings.kv_f16 && settings.window == 64 && settings.retain,
              "all options applied");
    const char * bad[][5] = {
        { "yes", "0",    "f32",  "256", "0"    },
        { "0",   "true", "f32",  "256", "0"    },
        { "0",   "0",    "bf16", "256", "0"    },
        { "0",   "0",    "f32",  "128", "0"    },
        { "0",   "0",    "f32",  "256", "true" },
        { "0",   "0",    "F16",  "256", "0"    },
    };
    for (const auto & values : bad) {
        failures +=
            check(!research_prefill_parse(values[0], values[1], values[2], values[3], values[4], settings, error),
                  "reject invalid setting");
        failures += check(settings.profile && settings.last_only && settings.kv_f16 && settings.window == 64 &&
                              settings.retain && !error.empty(),
                          "failure is atomic");
    }
    failures += check(research_prefill_parse("0", "0", "f32", "256", "0", settings, error), "explicit baseline");
    failures += check(!settings.profile && !settings.last_only && !settings.kv_f16 && settings.window == 256 &&
                          !settings.retain && error.empty(),
                      "baseline reset");
    return failures ? 1 : 0;
}

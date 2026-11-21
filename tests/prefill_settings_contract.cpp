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
    failures +=
        check(research_prefill_parse("1", "0", "f32", "256", "0", settings, error, "86") && settings.prefix_rows == 86,
              "explicit prefix experiment");
    const char * invalid_prefix[] = { "0", "-1", "4096", "999999999999999999999999", "2.5", " 2", "+2" };
    for (const char * value : invalid_prefix) {
        failures += check(!research_prefill_parse("0", "0", "f32", "256", "0", settings, error, value) &&
                              settings.prefix_rows == 86 && settings.profile,
                          "prefix failure is atomic");
    }
    failures += check(research_prefill_parse(nullptr, nullptr, nullptr, nullptr, nullptr, settings, error) &&
                          settings.prefix_rows == 0 && settings.prefix_source == ResearchPrefixSource::Split,
                      "prefix is opt-in only");
    failures += check(research_prefill_parse("1", "0", "f32", "256", "0", settings, error, "86", "full") &&
                          settings.prefix_source == ResearchPrefixSource::Full && settings.prefix_rows == 86,
                      "full-prefill causal source selected");
    const char * invalid_source[] = { "auto", "FULL", "full ", "prefix", "0" };
    for (const char * value : invalid_source) {
        failures += check(!research_prefill_parse("0", "0", "f32", "256", "0", settings, error, "86", value) &&
                              settings.prefix_source == ResearchPrefixSource::Full && settings.prefix_rows == 86 &&
                              settings.profile,
                          "invalid source rejected atomically");
    }
    for (const char * source : { "split", "full" }) {
        failures += check(!research_prefill_parse("0", "0", "f32", "256", "0", settings, error, nullptr, source) &&
                              settings.prefix_source == ResearchPrefixSource::Full && settings.prefix_rows == 86 &&
                              settings.profile,
                          "explicit source requires prefix rows atomically");
    }
    failures += check(research_prefill_parse("0", "0", "f32", "256", "0", settings, error, "9", "split") &&
                          settings.prefix_source == ResearchPrefixSource::Split && settings.prefix_rows == 9,
                      "explicit split source selected");
    failures += check(std::strcmp(research_prefix_source_name(ResearchPrefixSource::Full), "full") == 0 &&
                          std::strcmp(research_prefix_source_name(ResearchPrefixSource::Split), "split") == 0,
                      "source observation names stable");
    const auto parse_diagnostic = [&](const char * profile, const char * last, const char * retain, const char * dir,
                                      const char * layer, const char * attention) {
        return research_prefill_parse(profile, last, "f32", "256", retain, settings, error, nullptr, nullptr, dir,
                                      layer, attention);
    };
    failures +=
        check(parse_diagnostic("1", "0", "0", ".tmp/diagnostic", "all", "manual") && settings.diagnostic_layer == -1 &&
                  settings.diagnostic_dir == ".tmp/diagnostic" && settings.manual_attention,
              "bounded diagnostic selection");
    const char * invalid_diagnostic[][6] = {
        { "0", "0", "0", ".tmp/diagnostic", "0",   "default" },
        { "1", "1", "0", ".tmp/diagnostic", "0",   "default" },
        { "1", "0", "1", ".tmp/diagnostic", "0",   "default" },
        { "1", "0", "0", nullptr,           "0",   "default" },
        { "1", "0", "0", ".tmp/diagnostic", "-1",  "default" },
        { "1", "0", "0", ".tmp/diagnostic", "128", "default" },
        { "1", "0", "0", ".tmp/diagnostic", "ALL", "default" },
        { "1", "0", "0", ".tmp/diagnostic", "0",   "f32"     },
    };
    for (const auto & values : invalid_diagnostic) {
        failures += check(!parse_diagnostic(values[0], values[1], values[2], values[3], values[4], values[5]) &&
                              settings.diagnostic_layer == -1 && settings.manual_attention,
                          "invalid diagnostic rejected atomically");
    }
    const std::string long_dir(351, 'x');
    failures += check(!parse_diagnostic("1", "0", "0", long_dir.c_str(), "0", "default"), "diagnostic path bounded");
    failures += check(research_prefill_diag_shape_valid(settings, 512, 28, 7) &&
                          !research_prefill_diag_shape_valid(settings, 513, 28, 7) &&
                          !research_prefill_diag_shape_valid(settings, 512, 28, 8) &&
                          !research_prefill_diag_shape_valid(settings, 0, 28, 0),
                      "diagnostic rows/calls bounded");
    failures += check(parse_diagnostic("1", "0", "0", ".tmp/diagnostic", "28", "default") &&
                          !research_prefill_diag_shape_valid(settings, 236, 28, 0),
                      "loaded layer count checked");
    failures += check(research_prefill_parse(nullptr, nullptr, nullptr, nullptr, nullptr, settings, error) &&
                          settings.diagnostic_dir.empty() && !settings.manual_attention,
                      "ordinary defaults do not enable diagnostics or manual attention");
    const float first[]        = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
    const float different[]    = { 1.0f, 2.0f, 3.0f, 4.0f, 9.0f, 6.0f };
    const float changed_head[] = { 1.0f, 0.0f, 3.0f, 4.0f, 5.0f, 6.0f };
    failures += check(research_prefill_common_rows(first, 3, first, 3, 2) == 3, "identical prompt rows");
    failures += check(research_prefill_common_rows(first, 3, different, 3, 2) == 2, "actual changing row detected");
    failures += check(research_prefill_common_rows(first, 3, changed_head, 3, 2) == 0, "changed conditioning detected");
    failures += check(research_prefill_common_rows(first, 3, first, 2, 2) == 2, "different prompt lengths bounded");
    failures += check(research_prefill_common_rows(nullptr, 3, first, 3, 2) == 0 &&
                          research_prefill_common_rows(first, 0, first, 3, 2) == 0 &&
                          research_prefill_common_rows(first, 3, first, 3, 0) == 0,
                      "invalid row comparisons bounded");
    const float zero[]          = { 0.0f };
    const float negative_zero[] = { -0.0f };
    failures += check(research_prefill_common_rows(zero, 1, negative_zero, 1, 1) == 0, "byte equality is conservative");
    return failures ? 1 : 0;
}

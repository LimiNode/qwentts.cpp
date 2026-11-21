#pragma once
// Private research configuration. Ordinary builds never read these switches.

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct ResearchPrefillSettings {
    bool profile   = false;
    bool last_only = false;
    bool kv_f16    = false;
    int  window    = 256;
    bool retain    = false;
};

// Private snapshot made from the normal final-logits readback. It must only be
// read between requests; inference remains serialized by the pipeline owner.
struct ResearchPrefillObservation {
    bool               available         = false;
    bool               last_only         = false;
    bool               retained_hit      = false;
    bool               diagnostic        = false;
    int                T                 = 0;
    int                n_past            = 0;
    int                slot              = 0;
    int                n_kv_pad          = 0;
    int                graph_nodes       = 0;
    int                scheduler_splits  = 0;
    double             setup_ms          = 0.0;
    double             graph_build_ms    = 0.0;
    double             schedule_alloc_ms = 0.0;
    double             upload_ms         = 0.0;
    double             compute_wait_ms   = 0.0;
    double             readback_ms       = 0.0;
    double             total_ms          = 0.0;
    std::vector<float> logits_last;
};

static inline bool research_prefill_parse_bool(const char * value, bool & out) {
    if (!value || !*value || std::strcmp(value, "0") == 0) {
        out = false;
        return true;
    }
    if (std::strcmp(value, "1") == 0) {
        out = true;
        return true;
    }
    return false;
}

// Fail closed, without publishing a partially parsed configuration.
static inline bool research_prefill_parse(const char *              profile,
                                          const char *              last_only,
                                          const char *              kv,
                                          const char *              window,
                                          const char *              retain,
                                          ResearchPrefillSettings & out,
                                          std::string &             error) {
    ResearchPrefillSettings candidate;
    if (!research_prefill_parse_bool(profile, candidate.profile)) {
        error = "QWEN_PREFILL_PROFILE must be 0 or 1";
        return false;
    }
    if (!research_prefill_parse_bool(last_only, candidate.last_only)) {
        error = "QWEN_PREFILL_LAST_ONLY must be 0 or 1";
        return false;
    }
    if (!research_prefill_parse_bool(retain, candidate.retain)) {
        error = "QWEN_PREFILL_RETAIN must be 0 or 1";
        return false;
    }
    if (kv && *kv && std::strcmp(kv, "f32") != 0) {
        if (std::strcmp(kv, "f16") != 0) {
            error = "QWEN_PREFILL_KV must be f32 or f16";
            return false;
        }
        candidate.kv_f16 = true;
    }
    if (window && *window && std::strcmp(window, "256") != 0) {
        if (std::strcmp(window, "64") != 0) {
            error = "QWEN_PREFILL_WINDOW must be 256 or 64";
            return false;
        }
        candidate.window = 64;
    }
    out = candidate;
    error.clear();
    return true;
}

static inline bool research_prefill_from_env(ResearchPrefillSettings & out, std::string & error) {
    return research_prefill_parse(std::getenv("QWEN_PREFILL_PROFILE"), std::getenv("QWEN_PREFILL_LAST_ONLY"),
                                  std::getenv("QWEN_PREFILL_KV"), std::getenv("QWEN_PREFILL_WINDOW"),
                                  std::getenv("QWEN_PREFILL_RETAIN"), out, error);
}

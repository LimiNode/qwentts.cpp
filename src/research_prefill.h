#pragma once
// Private research configuration. Ordinary builds never read these switches.

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

enum class ResearchPrefixSource { Split, Full };

static inline const char * research_prefix_source_name(ResearchPrefixSource source) {
    return source == ResearchPrefixSource::Full ? "full" : "split";
}

constexpr int RESEARCH_PREFILL_DIAG_MAX_ROWS  = 512;
constexpr int RESEARCH_PREFILL_DIAG_MAX_CALLS = 8;

struct ResearchPrefillSettings {
    bool                 profile       = false;
    bool                 last_only     = false;
    bool                 kv_f16        = false;
    int                  window        = 256;
    bool                 retain        = false;
    int                  prefix_rows   = 0;
    ResearchPrefixSource prefix_source = ResearchPrefixSource::Split;
    std::string          diagnostic_dir;
    int                  diagnostic_layer = 0;  // -1 selects every layer.
    bool                 manual_attention = false;
    bool                 icl_full_text    = false;
};

// Private snapshot made from the normal final-logits readback. It must only be
// read between requests; inference remains serialized by the pipeline owner.
struct ResearchPrefillObservation {
    bool                 available              = false;
    bool                 last_only              = false;
    bool                 retained_hit           = false;
    bool                 diagnostic             = false;
    int                  T                      = 0;
    int                  n_past                 = 0;
    int                  slot                   = 0;
    int                  n_kv_pad               = 0;
    int                  graph_nodes            = 0;
    int                  scheduler_splits       = 0;
    double               setup_ms               = 0.0;
    double               graph_build_ms         = 0.0;
    double               schedule_alloc_ms      = 0.0;
    double               upload_ms              = 0.0;
    double               compute_wait_ms        = 0.0;
    double               readback_ms            = 0.0;
    double               total_ms               = 0.0;
    int                  prefix_rows            = 0;
    int                  prompt_rows            = 0;
    int                  common_rows_with_first = 0;
    bool                 prefix_hit             = false;
    ResearchPrefixSource prefix_source          = ResearchPrefixSource::Split;
    double               prefix_setup_ms        = 0.0;
    double               prefix_snapshot_ms     = 0.0;
    double               prefix_restore_ms      = 0.0;
    double               prefix_total_ms        = 0.0;
    bool                 prefill_use_flash_attn = false;
    int                  diagnostic_serial      = 0;
    int                  diagnostic_layer       = 0;
    std::string          diagnostic_path;
    std::vector<float>   logits_last;
};

static inline bool research_prefill_diag_shape_valid(const ResearchPrefillSettings & settings,
                                                     int                             rows,
                                                     int                             layers,
                                                     int                             serial) {
    return rows > 0 && rows <= RESEARCH_PREFILL_DIAG_MAX_ROWS && serial >= 0 &&
           serial < RESEARCH_PREFILL_DIAG_MAX_CALLS && layers > 0 && settings.diagnostic_layer >= -1 &&
           settings.diagnostic_layer < layers;
}

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

static inline int research_prefill_common_rows(const float * first,
                                               int           first_rows,
                                               const float * current,
                                               int           current_rows,
                                               int           hidden) {
    if (!first || !current || first_rows <= 0 || current_rows <= 0 || hidden <= 0) {
        return 0;
    }
    const int    rows   = first_rows < current_rows ? first_rows : current_rows;
    const size_t bytes  = static_cast<size_t>(hidden) * sizeof(float);
    int          common = 0;
    while (common < rows &&
           std::memcmp(first + static_cast<size_t>(common) * static_cast<size_t>(hidden),
                       current + static_cast<size_t>(common) * static_cast<size_t>(hidden), bytes) == 0) {
        ++common;
    }
    return common;
}

// Fail closed, without publishing a partially parsed configuration.
static inline bool research_prefill_parse(const char *              profile,
                                          const char *              last_only,
                                          const char *              kv,
                                          const char *              window,
                                          const char *              retain,
                                          ResearchPrefillSettings & out,
                                          std::string &             error,
                                          const char *              prefix_rows      = nullptr,
                                          const char *              prefix_source    = nullptr,
                                          const char *              diagnostic_dir   = nullptr,
                                          const char *              diagnostic_layer = nullptr,
                                          const char *              attention        = nullptr,
                                          const char *              icl_full_text    = nullptr) {
    ResearchPrefillSettings candidate;
    if (!research_prefill_parse_bool(icl_full_text, candidate.icl_full_text)) {
        error = "QWEN_ICL_FULL_TEXT must be 0 or 1";
        return false;
    }
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
    if (prefix_rows && *prefix_rows) {
        int rows = 0;
        for (const char * digit = prefix_rows; *digit; ++digit) {
            if (*digit < '0' || *digit > '9' || rows > 4095 / 10) {
                error = "QWEN_PREFILL_PREFIX_ROWS must be an integer in 1..4095";
                return false;
            }
            rows = rows * 10 + (*digit - '0');
        }
        if (rows < 1 || rows > 4095) {
            error = "QWEN_PREFILL_PREFIX_ROWS must be an integer in 1..4095";
            return false;
        }
        candidate.prefix_rows = rows;
    }
    if (prefix_source && *prefix_source) {
        if (std::strcmp(prefix_source, "full") == 0) {
            candidate.prefix_source = ResearchPrefixSource::Full;
        } else if (std::strcmp(prefix_source, "split") != 0) {
            error = "QWEN_PREFILL_PREFIX_SOURCE must be split or full";
            return false;
        }
        if (!candidate.prefix_rows) {
            error = "QWEN_PREFILL_PREFIX_SOURCE requires QWEN_PREFILL_PREFIX_ROWS";
            return false;
        }
    }
    if (attention && *attention) {
        if (std::strcmp(attention, "manual") == 0) {
            candidate.manual_attention = true;
        } else if (std::strcmp(attention, "default") != 0) {
            error = "QWEN_PREFILL_ATTN must be default or manual";
            return false;
        }
    }
    if (diagnostic_dir && *diagnostic_dir) {
        if (std::strlen(diagnostic_dir) > 350 || !candidate.profile || candidate.last_only || candidate.retain) {
            error = "QWEN_PREFILL_DIAG_DIR needs PROFILE=1, LAST_ONLY=0, RETAIN=0 and a path <=350 bytes";
            return false;
        }
        candidate.diagnostic_dir = diagnostic_dir;
    }
    if (diagnostic_layer && *diagnostic_layer) {
        if (candidate.diagnostic_dir.empty()) {
            error = "QWEN_PREFILL_DIAG_LAYER requires QWEN_PREFILL_DIAG_DIR";
            return false;
        }
        if (std::strcmp(diagnostic_layer, "all") == 0) {
            candidate.diagnostic_layer = -1;
        } else {
            int layer = 0;
            for (const char * digit = diagnostic_layer; *digit; ++digit) {
                if (*digit < '0' || *digit > '9' || layer > 127 / 10) {
                    error = "QWEN_PREFILL_DIAG_LAYER must be all or an integer in 0..127";
                    return false;
                }
                layer = layer * 10 + (*digit - '0');
            }
            if (layer > 127) {
                error = "QWEN_PREFILL_DIAG_LAYER must be all or an integer in 0..127";
                return false;
            }
            candidate.diagnostic_layer = layer;
        }
    }
    out = candidate;
    error.clear();
    return true;
}

static inline bool research_prefill_from_env(ResearchPrefillSettings & out, std::string & error) {
    return research_prefill_parse(
        std::getenv("QWEN_PREFILL_PROFILE"), std::getenv("QWEN_PREFILL_LAST_ONLY"), std::getenv("QWEN_PREFILL_KV"),
        std::getenv("QWEN_PREFILL_WINDOW"), std::getenv("QWEN_PREFILL_RETAIN"), out, error,
        std::getenv("QWEN_PREFILL_PREFIX_ROWS"), std::getenv("QWEN_PREFILL_PREFIX_SOURCE"),
        std::getenv("QWEN_PREFILL_DIAG_DIR"), std::getenv("QWEN_PREFILL_DIAG_LAYER"), std::getenv("QWEN_PREFILL_ATTN"),
        std::getenv("QWEN_ICL_FULL_TEXT"));
}

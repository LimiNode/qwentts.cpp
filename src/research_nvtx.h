#pragma once
// Optional labels for a separate profiler run; no additional synchronization.
#include <cstdlib>
#include <cstring>
#ifdef QWEN_RESEARCH_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

class ResearchPrefillNvtx {
public:
    explicit ResearchPrefillNvtx(const char * name) {
#ifdef QWEN_RESEARCH_NVTX
        const char * value = std::getenv("QWEN_PREFILL_NVTX");
        active_ = value && std::strcmp(value, "1") == 0;
        if (active_) nvtxRangePushA(name);
#else
        (void) name;
#endif
    }
    ~ResearchPrefillNvtx() { end(); }
    ResearchPrefillNvtx(const ResearchPrefillNvtx &) = delete;
    ResearchPrefillNvtx & operator=(const ResearchPrefillNvtx &) = delete;
    void next(const char * name) {
#ifdef QWEN_RESEARCH_NVTX
        if (active_) { nvtxRangePop(); nvtxRangePushA(name); }
#else
        (void) name;
#endif
    }
    void end() {
#ifdef QWEN_RESEARCH_NVTX
        if (active_) { nvtxRangePop(); active_ = false; }
#endif
    }
private:
#ifdef QWEN_RESEARCH_NVTX
    bool active_ = false;
#endif
};

#pragma once

// Research-only Philox key schedule v1. This does not replace Philox or
// consume/rewind its subsequence counter. See docs/research-frame-seeds.md.
#include <cstdint>
#include <cstring>

enum class ResearchFrameSeedMode { Baseline, PerFrame, Early, SwitchAfter };

struct ResearchFrameSeedSchedule {
    ResearchFrameSeedMode mode = ResearchFrameSeedMode::Baseline;
    std::uint32_t frames = 0;
};

static inline bool research_frame_seed_parse(const char * text, ResearchFrameSeedSchedule * out) {
    ResearchFrameSeedSchedule parsed;
    if (text == nullptr || *text == '\0' || std::strcmp(text, "baseline") == 0) {
        *out = parsed;
        return true;
    }
    if (std::strcmp(text, "per-frame") == 0) {
        parsed.mode = ResearchFrameSeedMode::PerFrame;
    } else if (std::strcmp(text, "early-2") == 0 || std::strcmp(text, "early-4") == 0) {
        parsed.mode = ResearchFrameSeedMode::Early;
        parsed.frames = text[6] == '2' ? 2U : 4U;
    } else {
        constexpr char prefix[] = "switch-after:";
        if (std::strncmp(text, prefix, sizeof(prefix) - 1) != 0) {
            return false;
        }
        const char * digits = text + sizeof(prefix) - 1;
        if (*digits == '\0' || (*digits == '0' && digits[1] != '\0')) {
            return false;
        }
        for (; *digits != '\0'; ++digits) {
            if (*digits < '0' || *digits > '9') {
                return false;
            }
            parsed.frames = parsed.frames * 10U + static_cast<std::uint32_t>(*digits - '0');
            if (parsed.frames > 128U) {
                return false;
            }
        }
        parsed.mode = ResearchFrameSeedMode::SwitchAfter;
    }
    *out = parsed;
    return true;
}

static inline const char * research_frame_seed_mode_name(ResearchFrameSeedMode mode) {
    switch (mode) {
        case ResearchFrameSeedMode::Baseline: return "baseline";
        case ResearchFrameSeedMode::PerFrame: return "per-frame";
        case ResearchFrameSeedMode::Early: return "early";
        case ResearchFrameSeedMode::SwitchAfter: return "switch-after";
    }
    return "invalid";
}

static inline std::uint64_t research_frame_seed_mix_v1(std::uint64_t base, std::uint32_t index) {
    // SplitMix64 finalizer after its standard Weyl increment; unsigned
    // overflow is intentional and makes the mapping modulo 2^64 explicit.
    std::uint64_t value = base + UINT64_C(0x9e3779b97f4a7c15) * (static_cast<std::uint64_t>(index) + 1U);
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

static inline std::int64_t research_frame_seed_for(
    const ResearchFrameSeedSchedule & policy, std::int64_t base, std::uint32_t frame) {
    if (policy.mode == ResearchFrameSeedMode::Baseline ||
        (policy.mode == ResearchFrameSeedMode::SwitchAfter && frame < policy.frames)) {
        return base;
    }
    std::uint32_t index = frame;
    if (policy.mode == ResearchFrameSeedMode::Early && frame >= policy.frames) {
        index = policy.frames - 1U;
    } else if (policy.mode == ResearchFrameSeedMode::SwitchAfter) {
        index = policy.frames;
    }
    const std::uint64_t bits = research_frame_seed_mix_v1(static_cast<std::uint64_t>(base), index);
    std::int64_t key;
    static_assert(sizeof(key) == sizeof(bits), "Philox seed must retain all 64 bits");
    std::memcpy(&key, &bits, sizeof(key));
    return key;
}

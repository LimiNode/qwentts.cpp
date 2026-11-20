#include "research_frame_seed.h"
#include "research_prefix.h"
#include "philox.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {
bool check(bool value, const char * message) {
    if (!value) {
        std::fprintf(stderr, "frame seed contract failed: %s\n", message);
    }
    return value;
}

std::vector<float> draws_by_chunks(const ResearchFrameSeedSchedule & policy,
                                  const std::vector<unsigned int> & chunks) {
    std::vector<float> draws;
    unsigned int frame = 0;
    for (unsigned int chunk : chunks) {
        for (unsigned int local = 0; local < chunk; ++local, ++frame) {
            const std::int64_t key = research_frame_seed_for(policy, 2, frame);
            for (unsigned int codebook = 0; codebook < 16; ++codebook) {
                float value;
                philox_uniform_fill(key, static_cast<std::int64_t>(16U * frame + codebook), 0U, &value, 1);
                draws.push_back(value);
            }
        }
    }
    return draws;
}
}  // namespace

int main() {
    std::vector<std::vector<int32_t>> prefix;
    std::istringstream valid_prefix("0 1 2 3\n4 5 6 7\n");
    if (!check(research_prefix_parse(valid_prefix, 4, 8, &prefix) &&
               prefix == std::vector<std::vector<int32_t>>{{0, 1, 2, 3}, {4, 5, 6, 7}},
               "complete prefix frames")) return 1;
    for (const char * invalid : {"", "\n", "0 1 2\n", "0 1 2 3 4\n", "0 1 -2 3\n",
                                 "0 1 2 8\n", "0 1 2 bad\n"}) {
        std::istringstream input(invalid);
        if (!check(!research_prefix_parse(input, 4, 8, &prefix) && prefix.size() == 2,
                   "invalid prefix fails without partial assignment")) return 1;
    }
    std::string oversized;
    for (int frame = 0; frame < 129; ++frame) oversized += "0 1 2 3\n";
    std::istringstream too_many(oversized);
    if (!check(!research_prefix_parse(too_many, 4, 8, &prefix), "bounded prefix frames")) return 1;
    ResearchFrameSeedSchedule policy;
    for (const char * text : { static_cast<const char *>(nullptr), "", "baseline" }) {
        if (!check(research_frame_seed_parse(text, &policy) &&
                   policy.mode == ResearchFrameSeedMode::Baseline, "neutral policy")) {
            return 1;
        }
    }
    for (const char * text : { "per-frame", "early-2", "early-4", "switch-after:0",
                              "switch-after:1", "switch-after:2", "switch-after:128" }) {
        if (!check(research_frame_seed_parse(text, &policy) &&
                   std::strcmp(research_frame_seed_mode_name(policy.mode), "invalid") != 0,
                   "valid policy")) {
            return 1;
        }
    }
    for (const char * text : { "unknown", "BASELINE", "baseline ", "per-frame:1", "early-0", "early-3",
                              "switch-after:", "switch-after:-1", "switch-after:+1", "switch-after:01",
                              "switch-after:129", "switch-after:999999999999999999999999", "switch-after:1 " }) {
        policy = { ResearchFrameSeedMode::PerFrame, 17U };
        if (!check(!research_frame_seed_parse(text, &policy) &&
                   policy.mode == ResearchFrameSeedMode::PerFrame && policy.frames == 17U,
                   "invalid policy fails without partial assignment")) {
            return 1;
        }
    }

    const std::array<std::uint64_t, 4> splitmix_zero = {
        UINT64_C(0xe220a8397b1dcdaf), UINT64_C(0x6e789e6aa1b965f4),
        UINT64_C(0x06c45d188009454f), UINT64_C(0xf88bb8a8724c81ec)
    };
    for (std::uint32_t frame = 0; frame < splitmix_zero.size(); ++frame) {
        if (!check(research_frame_seed_mix_v1(0, frame) == splitmix_zero[frame], "SplitMix64 v1 fixed vector")) {
            return 1;
        }
    }
    if (!check(research_frame_seed_mix_v1(2, 0) == UINT64_C(0x975835de1c9756ce), "nonzero-base fixed vector")) {
        return 1;
    }

    const ResearchFrameSeedSchedule baseline;
    for (std::int64_t base : { INT64_C(0), INT64_C(2), INT64_C(14), INT64_C(-1),
                               std::numeric_limits<std::int64_t>::min(),
                               std::numeric_limits<std::int64_t>::max() }) {
        for (std::uint32_t frame = 0; frame < 128; ++frame) {
            if (!check(research_frame_seed_for(baseline, base, frame) == base, "baseline key invariant")) {
                return 1;
            }
            for (std::int64_t codebook = 0; codebook < 16; ++codebook) {
                float old_draw, selected_draw;
                const std::int64_t subseq = static_cast<std::int64_t>(frame) * 16 + codebook;
                philox_uniform_fill(base, subseq, 0U, &old_draw, 1);
                philox_uniform_fill(research_frame_seed_for(baseline, base, frame), subseq, 0U, &selected_draw, 1);
                if (!check(std::memcmp(&old_draw, &selected_draw, sizeof(float)) == 0, "baseline Philox bits")) {
                    return 1;
                }
            }
        }
    }

    const ResearchFrameSeedSchedule per_frame{ ResearchFrameSeedMode::PerFrame, 0U };
    if (!check(static_cast<std::uint64_t>(research_frame_seed_for(per_frame, 0, 0)) == splitmix_zero[0],
               "high key bit retained") ||
        !check(research_frame_seed_for(per_frame, 2, 0) != research_frame_seed_for(per_frame, 2, 1),
               "codec frame changes key")) {
        return 1;
    }
    for (std::uint32_t window : { 2U, 4U }) {
        const ResearchFrameSeedSchedule early{ ResearchFrameSeedMode::Early, window };
        for (std::uint32_t frame = 0; frame < 16; ++frame) {
            const std::uint32_t index = frame < window ? frame : window - 1U;
            if (!check(research_frame_seed_for(early, 2, frame) == research_frame_seed_for(per_frame, 2, index),
                       "early schedule holds its final key")) {
                return 1;
            }
        }
    }
    for (std::uint32_t boundary : { 0U, 1U, 2U, 3U, 128U }) {
        const ResearchFrameSeedSchedule switched{ ResearchFrameSeedMode::SwitchAfter, boundary };
        for (std::uint32_t frame = 0; frame < 130; ++frame) {
            const std::int64_t expected = frame < boundary ? INT64_C(2) :
                research_frame_seed_for(per_frame, 2, boundary);
            if (!check(research_frame_seed_for(switched, 2, frame) == expected, "switch after complete frames")) {
                return 1;
            }
        }
    }
    for (const ResearchFrameSeedSchedule & schedule : {
             baseline, per_frame, ResearchFrameSeedSchedule{ ResearchFrameSeedMode::Early, 2U },
             ResearchFrameSeedSchedule{ ResearchFrameSeedMode::Early, 4U },
             ResearchFrameSeedSchedule{ ResearchFrameSeedMode::SwitchAfter, 1U } }) {
        const auto ramp = draws_by_chunks(schedule, { 1U, 2U, 4U, 8U });
        const auto alternate = draws_by_chunks(schedule, { 1U, 1U, 2U, 4U, 7U });
        const auto repeat = draws_by_chunks(schedule, { 1U, 2U, 4U, 8U });
        if (!check(ramp == alternate && ramp == repeat, "chunk-independent deterministic draw stream")) {
            return 1;
        }
    }
    return 0;
}

#pragma once

// Research-only complete-frame prefix replay without tensor dumps.
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static inline bool research_prefix_parse(std::istream & stream, int groups, int vocab,
                                         std::vector<std::vector<int32_t>> * out) {
    std::vector<std::vector<int32_t>> frames;
    std::string line;
    while (std::getline(stream, line)) {
        std::istringstream row(line);
        std::vector<int32_t> frame;
        int64_t code;
        while (true) {
            row >> std::ws;
            if (row.eof()) break;
            if (!(row >> code)) return false;
            if (code < 0 || code >= vocab || frame.size() >= static_cast<size_t>(groups)) {
                return false;
            }
            frame.push_back(static_cast<int32_t>(code));
        }
        if (!row.eof()) {
            return false;
        }
        if (!frame.empty()) {
            if (frame.size() != static_cast<size_t>(groups) || frames.size() >= 128) {
                return false;
            }
            frames.push_back(std::move(frame));
        }
    }
    if (!stream.eof() || frames.empty()) {
        return false;
    }
    *out = std::move(frames);
    return true;
}

static inline bool research_prefix_load(const char * path, int groups, int vocab,
                                        std::vector<std::vector<int32_t>> * out) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream || stream.tellg() <= 0 || stream.tellg() > 65536) return false;
    stream.seekg(0);
    return research_prefix_parse(stream, groups, vocab, out);
}

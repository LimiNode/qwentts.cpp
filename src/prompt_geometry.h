#pragma once
// Model-free layout operations shared by prompt construction and Talker upload.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

static inline int icl_prompt_rows(int text_rows, int codec_rows, bool full_text) {
    return full_text ? text_rows + codec_rows : codec_rows;
}

// The caller supplies projected [reference text; request text; tts_eos] and
// [codec_bos; every reference frame]. Full-text changes only their layout:
// upstream generate_icl_prompt(non_streaming_mode=True), not PCM streaming.
static inline void compose_icl_prompt(float * dst, const float * text, int text_rows,
                                      const float * codec, int codec_rows, int hidden,
                                      const float * tts_pad, const float * codec_pad, bool full_text,
                                      std::vector<float> & trailing, int & trailing_rows) {
    const size_t width = static_cast<size_t>(hidden);
    const size_t bytes = width * sizeof(float);
    const int rows = icl_prompt_rows(text_rows, codec_rows, full_text);
    for (int row = 0; row < rows; ++row) {
        const float * text_row = row < text_rows ? text + static_cast<size_t>(row) * width : tts_pad;
        const float * codec_row = full_text && row < text_rows ? codec_pad :
            codec + static_cast<size_t>(full_text ? row - text_rows : row) * width;
        float * out = dst + static_cast<size_t>(row) * width;
        for (int col = 0; col < hidden; ++col) out[col] = text_row[col] + codec_row[col];
    }
    const int remainder = full_text ? 0 : std::max(0, text_rows - codec_rows);
    trailing_rows = remainder > 0 ? remainder : 1;
    trailing.resize(static_cast<size_t>(trailing_rows) * width);
    std::memcpy(trailing.data(), remainder > 0 ? text + static_cast<size_t>(codec_rows) * width : tts_pad,
                remainder > 0 ? static_cast<size_t>(remainder) * bytes : bytes);
}

static inline std::vector<int32_t> prompt_positions(int rows, int n_past) {
    std::vector<int32_t> positions(static_cast<size_t>(rows));
    for (int row = 0; row < rows; ++row) positions[static_cast<size_t>(row)] = n_past + row;
    return positions;
}

template <typename T>
static inline std::vector<T> prompt_causal_mask(int rows, int n_past, int window, T zero, T negative_infinity) {
    std::vector<T> mask(static_cast<size_t>(rows) * static_cast<size_t>(window), negative_infinity);
    for (int query = 0; query < rows; ++query) {
        const auto first = mask.begin() + static_cast<size_t>(query) * static_cast<size_t>(window);
        std::fill(first, first + n_past + query + 1, zero);
    }
    return mask;
}

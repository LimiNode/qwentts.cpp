#include "prompt_geometry.h"

#include <cmath>
#include <cstdio>
#include <limits>

static int check(bool condition, const char * name) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", name);
        return 1;
    }
    return 0;
}

static int check_geometry(int reference_text_rows, int request_rows, int reference_frames, bool full_text) {
    int failures = 0;
    constexpr int hidden = 2;
    constexpr int prefix_rows = 9; // instruction/role/language/speaker prefix stays before the ICL block.
    const int text_rows = reference_text_rows + request_rows + 1;
    const int codec_rows = reference_frames + 1;
    const int rows = icl_prompt_rows(text_rows, codec_rows, full_text);
    failures += check(rows == (full_text ? text_rows + codec_rows : codec_rows), "prompt length");
    std::vector<float> text(static_cast<size_t>(text_rows) * hidden);
    std::vector<float> codec(static_cast<size_t>(codec_rows) * hidden);
    for (int row = 0; row < text_rows; ++row) {
        text[static_cast<size_t>(row) * hidden] = static_cast<float>(10 + row);
        text[static_cast<size_t>(row) * hidden + 1] = static_cast<float>(20 + row);
    }
    for (int row = 0; row < codec_rows; ++row) {
        codec[static_cast<size_t>(row) * hidden] = static_cast<float>(100 + row);
        codec[static_cast<size_t>(row) * hidden + 1] = static_cast<float>(200 + row);
    }
    const float tts_pad[] = {1000.0f, 2000.0f};
    const float codec_pad[] = {3000.0f, 4000.0f};
    std::vector<float> output(static_cast<size_t>(prefix_rows + rows + 1) * hidden, -99.0f);
    std::vector<float> trailing;
    int trailing_rows = 0;
    compose_icl_prompt(output.data() + prefix_rows * hidden, text.data(), text_rows, codec.data(), codec_rows,
                       hidden, tts_pad, codec_pad, full_text, trailing, trailing_rows);
    for (int i = 0; i < prefix_rows * hidden; ++i) failures += check(output[static_cast<size_t>(i)] == -99.0f, "prefix intact");
    failures += check(output[static_cast<size_t>(prefix_rows + rows) * hidden] == -99.0f &&
                      output.back() == -99.0f, "no write past prompt");
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < hidden; ++col) {
            float expected;
            if (full_text) {
                expected = row < text_rows ? text[static_cast<size_t>(row) * hidden + col] + codec_pad[col] :
                    tts_pad[col] + codec[static_cast<size_t>(row - text_rows) * hidden + col];
            } else {
                expected = (row < text_rows ? text[static_cast<size_t>(row) * hidden + col] : tts_pad[col]) +
                    codec[static_cast<size_t>(row) * hidden + col];
            }
            failures += check(output[static_cast<size_t>(prefix_rows + row) * hidden + col] == expected,
                              "every text/reference row and boundary preserved");
        }
    }
    const int expected_trailing = !full_text && text_rows > codec_rows ? text_rows - codec_rows : 1;
    failures += check(trailing_rows == expected_trailing && trailing.size() == static_cast<size_t>(expected_trailing) * hidden,
                      "trailing length");
    for (size_t i = 0; i < trailing.size(); ++i) {
        const float expected = !full_text && text_rows > codec_rows ? text[static_cast<size_t>(codec_rows) * hidden + i] : tts_pad[i];
        failures += check(trailing[i] == expected, "trailing exact content");
    }

    // These are the same helpers used for the actual Talker pos/mask uploads.
    // Full-text remains causal: all text precedes BOS/reference; neither a
    // future reference frame nor padded KV slots become visible early.
    const int context_rows = prefix_rows + rows;
    const int window = ((context_rows + 256) / 256) * 256;
    const auto positions = prompt_positions(context_rows, 0);
    const auto mask = prompt_causal_mask(context_rows, 0, window, 0.0f, -std::numeric_limits<float>::infinity());
    for (int query = 0; query < context_rows; ++query) {
        failures += check(positions[static_cast<size_t>(query)] == query, "prefill position");
        for (int key = 0; key < window; ++key) {
            const float value = mask[static_cast<size_t>(query) * window + key];
            failures += check(key <= query ? value == 0.0f : std::isinf(value) && value < 0.0f, "prefill causal mask");
        }
    }
    const auto decode_positions = prompt_positions(1, context_rows);
    const auto decode_mask = prompt_causal_mask(1, context_rows, window, 0.0f, -std::numeric_limits<float>::infinity());
    failures += check(decode_positions[0] == context_rows, "first generated position follows full prompt");
    for (int key = 0; key < window; ++key) {
        failures += check(key <= context_rows ? decode_mask[static_cast<size_t>(key)] == 0.0f :
                          std::isinf(decode_mask[static_cast<size_t>(key)]) && decode_mask[static_cast<size_t>(key)] < 0.0f,
                          "first generated query sees entire prompt and not padding");
    }
    return failures;
}

int main() {
    int failures = 0;
    for (bool full_text : {false, true}) {
        failures += check_geometry(2, 1, 6, full_text);  // text shorter than codec
        failures += check_geometry(2, 3, 5, full_text);  // equal
        failures += check_geometry(4, 3, 2, full_text);  // text longer than codec
        failures += check_geometry(1, 1, 1, full_text);  // minimum reference
        failures += check_geometry(88, 3, 226, full_text); // retained Kraftwerk reference shape
    }
    return failures ? 1 : 0;
}

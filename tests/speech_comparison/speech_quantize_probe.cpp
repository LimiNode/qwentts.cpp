// Runs the pinned upstream Qwen layout and quantize_model_file without a GPU.
// Usage: quantize_probe input_f32.gguf output_q8_0.gguf

#include <cstdio>
#include <stdexcept>

#include "args.h"
#include "quantize.h"
#include "qwen3-tts/layout.h"

int main(int argc, char** argv) {
    const auto args = utf8_args(argc, argv);
    if (args.size() != 3) {
        std::fprintf(stderr, "usage: quantize_probe input_f32.gguf output_q8_0.gguf\n");
        return 2;
    }
    try {
        for (const auto& type : weight_types()) {
            if (type.type == GGML_TYPE_Q8_0) {
                quantize_model_file(args[1], args[2], type, qwen3_tts_layout);
                return 0;
            }
        }
        throw std::runtime_error("upstream weight types did not advertise Q8_0");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}

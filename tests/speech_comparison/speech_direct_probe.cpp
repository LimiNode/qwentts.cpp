// Measures each direct synthesis and retains every WAV and float payload.
// Usage: probe model.gguf device output_dir speaker language seed runs text_file...

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "args.h"
#include "backend.h"
#include "json.h"
#include "qwen3-tts/synthesizer.h"
#include "wav.h"

namespace {

using Clock = std::chrono::steady_clock;

double milliseconds(Clock::time_point from, Clock::time_point to) {
    return std::chrono::duration<double, std::milli>(to - from).count();
}

std::string read_text(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read text file " + path.u8string());
    std::string text{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    if (text.compare(0, 3, "\xEF\xBB\xBF") == 0) text.erase(0, 3);
    return text;
}

std::string phases(const SynthesisStats& stats) {
    return "{\"prompt_ms\":" + json_number(stats.prompt * 1000) +
        ",\"prefill_ms\":" + json_number(stats.prefill * 1000) +
        ",\"talker_decode_ms\":" + json_number(stats.talker * 1000) +
        ",\"code_predictor_ms\":" + json_number(stats.code_predictor * 1000) +
        ",\"codec_ms\":" + json_number(stats.codec * 1000) + "}";
}

std::string sampling(const SamplingParams& params) {
    return "{\"greedy\":" + std::string(params.greedy ? "true" : "false") +
        ",\"temperature\":" + json_number(params.temperature) +
        ",\"top_k\":" + std::to_string(params.top_k) +
        ",\"top_p\":" + json_number(params.top_p) +
        ",\"repetition_penalty\":" + json_number(params.repetition_penalty) + "}";
}

} // namespace

int main(int argc, char** argv) {
    const auto args = utf8_args(argc, argv);
    if (args.size() < 9) {
        std::fprintf(stderr, "usage: probe model.gguf device output_dir speaker language seed runs text_file...\n");
        return 2;
    }
    try {
        const auto output_dir = std::filesystem::u8path(args[3]);
        const auto seed = std::stoull(args[6]);
        const int runs = std::stoi(args[7]);
        if (runs < 1) throw std::runtime_error("runs must be positive");
        std::filesystem::create_directories(output_dir);
        const auto result_path = output_dir / "results.jsonl";
        if (std::filesystem::exists(result_path)) throw std::runtime_error("output results.jsonl already exists");
        std::ofstream results(result_path, std::ios::binary);
        if (!results) throw std::runtime_error("cannot create results.jsonl");
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(init_backend(args[2]), ggml_backend_free);
        const auto load_start = Clock::now();
        Synthesizer synth(args[1], backend.get());
        const double model_load_ms = milliseconds(load_start, Clock::now());
        Qwen2Tokenizer tokenizer(synth.model(), kTextTokenizer);
        for (size_t case_index = 8; case_index < args.size(); ++case_index) {
            SynthesisRequest request;
            request.text = read_text(std::filesystem::u8path(args[case_index]));
            request.speaker = args[4];
            request.language = args[5];
            request.seed = seed;
            request.max_frames = 256;
            const size_t text_tokens = tokenizer.encode(request.text).size();
            const int language_id = synth.ids().language_id(request.language, synth.ids().voice(request.speaker));
            const size_t prompt_rows = text_tokens + kPromptRows - (language_id == -1 ? 1 : 0);
            for (int run = 0; run < runs; ++run) {
                const auto stem = "case-" + std::to_string(case_index - 8) + "-run-" + std::to_string(run);
                const auto wav_path = output_dir / (stem + ".wav");
                const auto raw_path = output_dir / (stem + ".f32");
                if (std::filesystem::exists(wav_path) || std::filesystem::exists(raw_path)) {
                    throw std::runtime_error("refusing to overwrite a retained artifact");
                }
                SynthesisStats stats, first_stats;
                std::vector<float> audio;
                bool invalid_pcm = false, capacity_exceeded = false;
                size_t chunks = 0, first_samples = 0;
                Clock::time_point first;
                const auto start = Clock::now();
                const auto outcome = synth.synthesize(request, [&](const float* samples, size_t count) {
                    if (count == 0 || count > 256 * 1920 || audio.size() + count > 256 * 1920) {
                        capacity_exceeded = true;
                        return false;
                    }
                    for (size_t at = 0; at < count; ++at) {
                        if (!std::isfinite(samples[at])) { invalid_pcm = true; return false; }
                    }
                    if (chunks == 0) {
                        first = Clock::now();
                        first_stats = stats;
                        first_samples = count;
                    }
                    ++chunks;
                    audio.insert(audio.end(), samples, samples + count);
                    return true;
                }, &stats);
                const auto finish = Clock::now();
                std::ofstream raw(raw_path, std::ios::binary);
                raw.write(reinterpret_cast<const char*>(audio.data()), static_cast<std::streamsize>(audio.size() * sizeof(float)));
                if (!raw) throw std::runtime_error("cannot retain float payload");
                raw.close();
                write_wav(wav_path.u8string(), audio, synth.sample_rate());
                const std::string row = "{\"case\":" + std::to_string(case_index - 8) +
                    ",\"run\":" + std::to_string(run) +
                    ",\"warmth\":" + json_string(case_index == 8 && run == 0 ? "first_after_model_load" : "persistent_model") +
                    ",\"text\":" + json_string(request.text) +
                    ",\"text_file\":" + json_string(args[case_index]) +
                    ",\"text_tokens\":" + std::to_string(text_tokens) +
                    ",\"prompt_rows\":" + std::to_string(prompt_rows) +
                    ",\"model\":" + json_string(args[1]) +
                    ",\"speaker\":" + json_string(request.speaker) +
                    ",\"language\":" + json_string(request.language) +
                    ",\"seed\":" + std::to_string(seed) +
                    ",\"device\":" + json_string(ggml_backend_name(backend.get())) +
                    ",\"model_load_ms\":" + json_number(model_load_ms) +
                    ",\"first_pcm_ms\":" + (chunks ? json_number(milliseconds(start, first)) : "null") +
                    ",\"total_ms\":" + json_number(milliseconds(start, finish)) +
                    ",\"first_chunk_samples\":" + std::to_string(first_samples) +
                    ",\"chunks\":" + std::to_string(chunks) +
                    ",\"frames\":" + std::to_string(outcome.frames) +
                    ",\"natural_eos\":" + (outcome.ended ? "true" : "false") +
                    ",\"invalid_pcm\":" + (invalid_pcm ? "true" : "false") +
                    ",\"capacity_exceeded\":" + (capacity_exceeded ? "true" : "false") +
                    ",\"talker_sampling\":" + sampling(synth.generation().talker) +
                    ",\"predictor_sampling\":" + sampling(synth.generation().code_predictor) +
                    ",\"min_frames\":" + std::to_string(synth.generation().min_frames) +
                    ",\"sample_rate\":" + std::to_string(synth.sample_rate()) +
                    ",\"samples\":" + std::to_string(audio.size()) +
                    ",\"seconds\":" + json_number(static_cast<double>(audio.size()) / synth.sample_rate()) +
                    ",\"phases\":" + phases(stats) +
                    ",\"first_pcm_phases\":" + phases(first_stats) +
                    ",\"wav\":" + json_string(wav_path.u8string()) +
                    ",\"raw_f32\":" + json_string(raw_path.u8string()) +
                    ",\"listening\":\"unassessed\"}";
                results << row << '\n';
                results.flush();
                if (!results) throw std::runtime_error("cannot retain result row");
                std::printf("%s\n", row.c_str());
                std::fflush(stdout);
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}

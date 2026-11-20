// Bounded engine-internal experiment, not the QTB scheduler or a player.
#include "pipeline-tts.h"
#include "bpe.h"
#include "yyjson.h"

#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>

namespace {
using Clock = std::chrono::steady_clock;
using JsonDoc = std::unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)>;
constexpr size_t kMaxSamples = 2048 * 1920;

std::string read_bytes(const std::string & path, size_t limit) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() < 0 || static_cast<size_t>(file.tellg()) > limit) {
        throw std::runtime_error("invalid/bounded input file: " + path);
    }
    std::string result(static_cast<size_t>(file.tellg()), '\0');
    file.seekg(0);
    if (!file.read(result.data(), static_cast<std::streamsize>(result.size()))) {
        throw std::runtime_error("cannot read input file: " + path);
    }
    return result;
}
std::string string_field(yyjson_val * object, const char * key) {
    yyjson_val * value = yyjson_obj_get(object, key);
    if (!yyjson_is_str(value) || yyjson_get_len(value) == 0) {
        throw std::runtime_error(std::string("missing string: ") + key);
    }
    std::string result(yyjson_get_str(value), yyjson_get_len(value));
    if (result.find('\0') != std::string::npos) throw std::runtime_error("embedded NUL");
    return result;
}
int integer_field(yyjson_val * object, const char * key, int low, int high) {
    yyjson_val * value = yyjson_obj_get(object, key);
    if (!yyjson_is_int(value)) throw std::runtime_error(std::string("missing integer: ") + key);
    const int64_t number = yyjson_get_sint(value);
    if (number < low || number > high) throw std::runtime_error("integer out of bounds");
    return static_cast<int>(number);
}
template<class T> std::vector<T> raw_values(const std::string & path, size_t max_count) {
    const auto bytes = read_bytes(path, max_count * sizeof(T));
    if (bytes.empty() || bytes.size() % sizeof(T)) throw std::runtime_error("invalid raw tensor");
    std::vector<T> values(bytes.size() / sizeof(T));
    std::memcpy(values.data(), bytes.data(), bytes.size());
    return values;
}
struct Owner {
    BackendPair bp{};
    PipelineTTS pt{};
    BPETokenizer tokenizer;
    TtsEngine * engine = nullptr;
    ~Owner() {
        if (engine) tts_engine_free(engine);
        pipeline_tts_free(&pt);
        backend_release(bp.backend, bp.cpu_backend);
    }
};
struct Chunk { double ms; size_t samples; };
struct Step { double ms; int active; size_t retired; };
struct Lane {
    Clock::time_point start, deadline;
    std::vector<float> pcm;
    std::vector<Chunk> chunks;
    bool invalid_pcm = false;
    bool capacity_exceeded = false;
};
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
bool cancelled(void * pointer) {
    return Clock::now() >= static_cast<Lane *>(pointer)->deadline;
}
bool on_audio(const float * samples, int count, void * pointer) {
    auto & lane = *static_cast<Lane *>(pointer);
    const auto now = elapsed(lane.start);
    if (count <= 0 || lane.pcm.size() + static_cast<size_t>(count) > kMaxSamples) {
        lane.capacity_exceeded = true;
        return false;
    }
    for (int index = 0; index < count; ++index) {
        if (!std::isfinite(samples[index])) {
            lane.invalid_pcm = true;
            return false;
        }
    }
    lane.chunks.push_back({now, static_cast<size_t>(count)});
    lane.pcm.insert(lane.pcm.end(), samples, samples + count);
    return !cancelled(pointer);
}
}

int main(int argc, char ** argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: eos-research-probe manifest.json NEW-output-dir");
        const auto manifest = read_bytes(argv[1], 1024 * 1024);
        JsonDoc doc(yyjson_read(manifest.data(), manifest.size(), 0), yyjson_doc_free);
        if (!doc) throw std::runtime_error("invalid JSON manifest");
        yyjson_val * root = yyjson_doc_get_root(doc.get());
        const auto talker = string_field(root, "talker");
        const auto codec = string_field(root, "codec");
        const auto reference_text = read_bytes(string_field(root, "reference_text_file"), 65536);
        if (reference_text.empty() || reference_text.find('\0') != std::string::npos) {
            throw std::runtime_error("empty/invalid reference text");
        }
        const auto speaker = raw_values<float>(string_field(root, "speaker_f32"), 4096);
        const auto codes = raw_values<int32_t>(string_field(root, "reference_i32"), 16 * 4096);
        for (float value : speaker) if (!std::isfinite(value)) throw std::runtime_error("nonfinite reference");
        for (int32_t value : codes) if (value < 0 || value >= 2048) throw std::runtime_error("invalid reference code");
        const int ref_frames = integer_field(root, "reference_frames", 1, 4096);
        if (codes.size() != static_cast<size_t>(ref_frames) * 16) throw std::runtime_error("reference shape mismatch");
        const int max_batch = integer_field(root, "max_batch", 1, 2);
        const int max_tokens = integer_field(root, "max_tokens", 4, 2048);
        const int deadline_ms = integer_field(root, "deadline_ms", 1, 120000);
        const int max_chunk = integer_field(root, "stream_max_chunk_frames", 1, 8);
        if (max_chunk != 1 && max_chunk != 2 && max_chunk != 4 && max_chunk != 8) throw std::runtime_error("invalid chunk cap");
        yyjson_val * groups = yyjson_obj_get(root, "groups");
        if (!yyjson_is_arr(groups) || yyjson_arr_size(groups) == 0 || yyjson_arr_size(groups) > 256) {
            throw std::runtime_error("expected 1..256 groups");
        }
        // Exclusive destination: preserve failed or partial experiments.
        const std::filesystem::path destination(argv[2]);
        if (!std::filesystem::create_directory(destination)) throw std::runtime_error("destination exists");
        std::ofstream results(destination / "results.jsonl", std::ios::binary);
        if (!results) throw std::runtime_error("cannot open results");
        Owner owner;
        owner.bp = backend_init("Research");
        if (!owner.bp.backend || !owner.bp.has_gpu) throw std::runtime_error("research run requires selected GPU");
        if (!pipeline_tts_load(&owner.pt, talker.c_str(), codec.c_str(), owner.bp,
                              true, false, max_batch, 24.0f, max_chunk) ||
            !load_bpe_from_gguf(&owner.tokenizer, talker.c_str())) {
            throw std::runtime_error("model/tokenizer load failed");
        }
        const char * specials[] = {"qwen3-tts.text.im_start_id", "qwen3-tts.text.im_end_id",
            "qwen3-tts.text.tts_pad_id", "qwen3-tts.text.tts_bos_id", "qwen3-tts.text.tts_eos_id"};
        bpe_load_specials_from_keys(&owner.tokenizer, talker.c_str(), specials, 5);
        if (speaker.size() != static_cast<size_t>(owner.pt.talker.hidden_size) || owner.pt.num_code_groups != 16) {
            throw std::runtime_error("conditioning/model mismatch");
        }
        owner.engine = tts_engine_new(&owner.pt, &owner.tokenizer);
        if (!owner.engine) throw std::runtime_error("cannot create engine");
        for (size_t group_index = 0; group_index < yyjson_arr_size(groups); ++group_index) {
            yyjson_val * group = yyjson_arr_get(groups, group_index);
            const auto text = read_bytes(string_field(group, "text_file"), 65536);
            const auto language = string_field(group, "language");
            if (text.empty() || text.find('\0') != std::string::npos) throw std::runtime_error("invalid spoken text");
            yyjson_val * seeds = yyjson_obj_get(group, "seeds");
            const size_t lanes = yyjson_arr_size(seeds);
            if (!yyjson_is_arr(seeds) || lanes < 1 || lanes > static_cast<size_t>(max_batch)) {
                throw std::runtime_error("invalid lane count");
            }
            std::array<Lane, 2> observations;
            std::array<qt_tts_params, 2> params{};
            std::array<qt_audio, 2> audio{};
            std::array<TtsJob, 2> jobs{};
            const auto start = Clock::now();
            for (size_t lane = 0; lane < lanes; ++lane) {
                yyjson_val * seed = yyjson_arr_get(seeds, lane);
                if (!yyjson_is_int(seed) || yyjson_get_sint(seed) < 0) throw std::runtime_error("explicit nonnegative seed required");
                observations[lane].start = start;
                observations[lane].deadline = start + std::chrono::milliseconds(deadline_ms);
                observations[lane].pcm.reserve(static_cast<size_t>(max_tokens) * 1920);
                auto & p = params[lane];
                qt_tts_default_params_ex(&p, sizeof(p));
                p.text = text.c_str(); p.lang = language.c_str(); p.ref_text = reference_text.c_str();
                p.ref_spk_emb = speaker.data(); p.ref_spk_dim = static_cast<int>(speaker.size());
                p.ref_codes = codes.data(); p.ref_T = ref_frames;
                p.seed = yyjson_get_sint(seed); p.max_new_tokens = max_tokens;
                p.on_chunk = on_audio; p.on_chunk_user_data = &observations[lane];
                p.cancel = cancelled; p.cancel_user_data = &observations[lane];
                jobs[lane].params = &p; jobs[lane].resolved_seed = p.seed; jobs[lane].out = &audio[lane];
                if (!tts_engine_admit(owner.engine, &jobs[lane])) throw std::runtime_error("admission failed: " + jobs[lane].error);
            }
            if (tts_engine_active(owner.engine) != static_cast<int>(lanes)) throw std::runtime_error("not co-admitted");
            const double admitted_ms = elapsed(start);
            std::vector<TtsJob *> retired;
            std::vector<Step> steps;
            while (tts_engine_active(owner.engine)) {
                tts_engine_step(owner.engine, &retired);
                steps.push_back({elapsed(start), tts_engine_active(owner.engine), retired.size()});
            }
            const double group_total_ms = elapsed(start);
            if (retired.size() != lanes) throw std::runtime_error("terminal count mismatch");
            for (size_t lane = 0; lane < lanes; ++lane) {
                const auto & observation = observations[lane];
                const auto & job = jobs[lane];
                const auto filename = "group-" + std::to_string(group_index) + "-lane-" + std::to_string(lane) + ".f32";
                std::ofstream pcm(destination / filename, std::ios::binary);
                pcm.write(reinterpret_cast<const char *>(observation.pcm.data()),
                          static_cast<std::streamsize>(observation.pcm.size() * sizeof(float)));
                if (!pcm) throw std::runtime_error("PCM write failure");
                results << "{\"group\":" << group_index << ",\"lane\":" << lane << ",\"seed\":" << params[lane].seed
                    << ",\"status\":" << job.status << ",\"finish_reason\":" << job.finish_reason
                    << ",\"frames\":" << job.n_frames << ",\"samples\":" << observation.pcm.size()
                    << ",\"admitted_together\":" << lanes << ",\"admitted_ms\":" << admitted_ms
                    << ",\"prefill_ms\":" << job.prefill_ms << ",\"total_ms\":" << job.total_ms
                    << ",\"group_total_ms\":" << group_total_ms
                    << ",\"invalid_pcm\":" << (observation.invalid_pcm ? "true" : "false")
                    << ",\"capacity_exceeded\":" << (observation.capacity_exceeded ? "true" : "false")
                    << ",\"chunks\":[";
                for (size_t index = 0; index < observation.chunks.size(); ++index) {
                    if (index) results << ',';
                    results << "{\"ms\":" << observation.chunks[index].ms << ",\"samples\":" << observation.chunks[index].samples << '}';
                }
                results << "],\"steps\":[";
                for (size_t index = 0; index < steps.size(); ++index) {
                    if (index) results << ',';
                    results << "{\"ms\":" << steps[index].ms << ",\"active\":" << steps[index].active
                            << ",\"retired\":" << steps[index].retired << '}';
                }
                results << "]}\n";
                results.flush();
                if (!results) throw std::runtime_error("result write failure");
                qt_audio_free(&audio[lane]);
            }
        }
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "research probe: %s\n", error.what());
        return 1;
    }
}

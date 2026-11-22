// Teacher-forced codec state experiment, not a live AR rollback policy.
#include "pipeline-codec.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
constexpr int kCodebooks = TOKENIZER_NUM_CODEBOOKS;
constexpr int kMaxFrames = 2048;
constexpr int kMaxReferenceFrames = 8192;
constexpr uint64_t kCheckpointKey = 0xc0dec000;
constexpr uint64_t kInspectionKey = 0xc0decfff;

struct Codes {
    int frames = 0;
    std::vector<int32_t> values;
};

struct Chunk {
    int start = 0;
    int frames = 0;
    int position_before = 0;
    int position_after = 0;
};

struct Audio {
    std::vector<float> samples;
    std::vector<Chunk> chunks;
};

struct State {
    int position = 0;
    std::vector<std::vector<unsigned char>> tensors;
};

struct CodecOwner {
    BackendPair backend{};
    PipelineCodec codec{};
    bool loaded = false;
    ~CodecOwner() {
        if (loaded) pipeline_codec_free(&codec);
        backend_release(backend.backend, backend.cpu_backend);
    }
};

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

void check_deadline(Clock::time_point started) {
    require(Clock::now() - started < std::chrono::seconds(600), "600 second probe deadline exceeded");
}

std::string json_string(const std::string & value) {
    std::string result = "\"";
    const char hex[] = "0123456789abcdef";
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') {
            result += '\\';
            result += static_cast<char>(c);
        } else if (c < 32) {
            result += "\\u00";
            result += hex[c >> 4];
            result += hex[c & 15];
        } else {
            result += static_cast<char>(c);
        }
    }
    return result + '"';
}

Codes read_codes(const fs::path & path, int limit) {
    require(fs::is_regular_file(path), "code input is not a regular file");
    const uintmax_t bytes = fs::file_size(path);
    require(bytes > 0 && bytes % (kCodebooks * sizeof(int32_t)) == 0 &&
            bytes <= static_cast<uintmax_t>(limit) * kCodebooks * sizeof(int32_t),
            "invalid or oversized complete16 code input");
    std::vector<unsigned char> raw(static_cast<size_t>(bytes));
    std::ifstream input(path, std::ios::binary);
    require(static_cast<bool>(input.read(reinterpret_cast<char *>(raw.data()),
                                        static_cast<std::streamsize>(raw.size()))), "cannot read code input");
    Codes codes;
    codes.frames = static_cast<int>(bytes / (kCodebooks * sizeof(int32_t)));
    codes.values.reserve(raw.size() / sizeof(int32_t));
    for (size_t i = 0; i < raw.size(); i += 4) {
        const uint32_t value = static_cast<uint32_t>(raw[i]) |
            (static_cast<uint32_t>(raw[i + 1]) << 8) |
            (static_cast<uint32_t>(raw[i + 2]) << 16) |
            (static_cast<uint32_t>(raw[i + 3]) << 24);
        require(value < (1u << TOKENIZER_CODE_BITS), "code outside tokenizer vocabulary");
        codes.values.push_back(static_cast<int32_t>(value));
    }
    return codes;
}

std::vector<int32_t> slice(const Codes & codes, int start, int frames) {
    require(start >= 0 && frames > 0 && start <= codes.frames - frames, "invalid code slice");
    std::vector<int32_t> result(static_cast<size_t>(frames) * kCodebooks);
    for (int k = 0; k < kCodebooks; ++k) {
        std::copy_n(codes.values.begin() + static_cast<size_t>(k) * codes.frames + start,
                    frames, result.begin() + static_cast<size_t>(k) * frames);
    }
    return result;
}

int common_frames(const Codes & a, const Codes & b) {
    int frame = 0;
    for (; frame < std::min(a.frames, b.frames); ++frame) {
        for (int k = 0; k < kCodebooks; ++k) {
            if (a.values[static_cast<size_t>(k) * a.frames + frame] !=
                b.values[static_cast<size_t>(k) * b.frames + frame]) return frame;
        }
    }
    return frame;
}

std::vector<int> schedule(int frames, bool ramp) {
    std::vector<int> result;
    int target = 1;
    while (frames > 0) {
        int width = ramp ? target : 1;
        while (width > frames) width /= 2;
        result.push_back(width);
        frames -= width;
        if (ramp) target = std::min(target * 2, 8);
    }
    return result;
}

bool is_boundary(const std::vector<int> & widths, int boundary) {
    int cursor = 0;
    for (int width : widths) {
        if (cursor == boundary) return true;
        cursor += width;
    }
    return cursor == boundary;
}

template <typename T>
void write_raw(const fs::path & path, const std::vector<T> & values) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    require(static_cast<bool>(output), "cannot create output artifact");
    if (!values.empty()) output.write(reinterpret_cast<const char *>(values.data()),
                                     static_cast<std::streamsize>(values.size() * sizeof(T)));
    output.close();
    require(static_cast<bool>(output), "cannot flush output artifact");
}

void prime(PipelineCodec & codec, const Codes & reference, Clock::time_point started) {
    const int staging = codec.stream_sets - 1;
    require(pipeline_codec_stream_reset(&codec, staging), "staging reset failed");
    int offset = 0;
    while (offset < reference.frames) {
        check_deadline(started);
        int width = 8;
        while (width > reference.frames - offset) width /= 2;
        const auto chunk = slice(reference, offset, width);
        require(pipeline_codec_decode_stream(&codec, chunk.data(), width, nullptr), "reference priming failed");
        offset += width;
    }
    require(pipeline_codec_stream_copy_set(&codec, staging, 0), "primed staging handoff failed");
}

void decode_chunk(PipelineCodec & codec, const Codes & codes, int offset, int width,
                  Audio & audio, Clock::time_point started) {
    check_deadline(started);
    auto chunk = slice(codes, offset, width);
    std::vector<float> samples(static_cast<size_t>(width) * TOKENIZER_HOP_LENGTH);
    float * output = samples.data();
    const int before = codec.stream_pos.at(0);
    require(pipeline_codec_decode_stream_batch(&codec, chunk.data(), width, 1, &output), "lane decode failed");
    require(codec.stream_pos.at(0) == before + width, "unexpected codec cursor increment");
    require(std::all_of(samples.begin(), samples.end(), [](float value) { return std::isfinite(value); }),
            "non-finite decoded PCM");
    audio.samples.insert(audio.samples.end(), samples.begin(), samples.end());
    audio.chunks.push_back({offset, width, before, codec.stream_pos.at(0)});
}

void decode_range(PipelineCodec & codec, const Codes & codes, const std::vector<int> & widths,
                  int begin, int end, Audio & audio, Clock::time_point started) {
    require(begin <= end && is_boundary(widths, begin) && is_boundary(widths, end),
            "requested position is not a completed decode boundary");
    int offset = 0;
    for (int width : widths) {
        if (offset >= begin && offset < end) decode_chunk(codec, codes, offset, width, audio, started);
        offset += width;
    }
}

CodecStateSnap & find_snapshot(PipelineCodec & codec, uint64_t key) {
    for (auto & item : codec.snaps) {
        if (item.stamp && item.key == key) return item;
    }
    throw std::runtime_error("expected private codec checkpoint was evicted");
}

// Inspect the existing snapshot mirror instead of duplicating the codec's state walker.
State inspect(PipelineCodec & codec) {
    require(pipeline_codec_stream_snapshot(&codec, kInspectionKey, 0), "state inspection snapshot failed");
    auto & mirror = find_snapshot(codec, kInspectionKey);
    State result;
    result.position = mirror.pos;
    for (auto * tensor = ggml_get_first_tensor(mirror.ctx); tensor;
         tensor = ggml_get_next_tensor(mirror.ctx, tensor)) {
        result.tensors.emplace_back(ggml_nbytes(tensor));
        auto & bytes = result.tensors.back();
        ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
    }
    pipeline_codec_snap_free(&mirror);
    return result;
}

bool same_state(const State & a, const State & b) {
    return a.position == b.position && a.tensors == b.tensors;
}

bool same_pcm(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

Audio retained_prefix(const Audio & source, int boundary) {
    Audio result;
    const size_t samples = static_cast<size_t>(boundary) * TOKENIZER_HOP_LENGTH;
    require(samples <= source.samples.size(), "PCM prefix exceeds source");
    result.samples.assign(source.samples.begin(), source.samples.begin() + samples);
    for (const auto & chunk : source.chunks) {
        if (chunk.start + chunk.frames <= boundary) result.chunks.push_back(chunk);
    }
    return result;
}

void audio_json(std::ostream & json, const char * name, const Audio & audio) {
    json << json_string(name) << ":{\"samples\":" << audio.samples.size() << ",\"chunks\":[";
    bool first = true;
    for (const auto & chunk : audio.chunks) {
        if (!first) json << ',';
        first = false;
        json << "{\"start_frame\":" << chunk.start << ",\"frames\":" << chunk.frames
             << ",\"samples\":" << chunk.frames * TOKENIZER_HOP_LENGTH
             << ",\"position_before\":" << chunk.position_before
             << ",\"position_after\":" << chunk.position_after << '}';
    }
    json << "]}";
}

bool run_case(PipelineCodec & codec, const Codes & reference, const Codes & original_codes,
              const Codes & changed_codes, bool ramp, int boundary, const fs::path & output,
              std::ostream & json, Clock::time_point started) {
    const auto original_widths = schedule(original_codes.frames, ramp);
    const auto changed_widths = schedule(changed_codes.frames, ramp);
    require(is_boundary(original_widths, boundary) && is_boundary(changed_widths, boundary),
            "case boundary does not exist in both schedules");
    require(common_frames(original_codes, changed_codes) >= boundary, "candidate does not retain code prefix");
    require(fs::create_directory(output), "case output already exists");

    prime(codec, reference, started);
    Audio original;
    decode_range(codec, original_codes, original_widths, 0, boundary, original, started);
    require(pipeline_codec_stream_snapshot(&codec, kCheckpointKey, 0), "checkpoint save failed");
    const State saved = inspect(codec);
    decode_range(codec, original_codes, original_widths, boundary, original_codes.frames, original, started);
    const State original_final = inspect(codec);
    write_raw(output / "original.f32", original.samples);

    require(pipeline_codec_stream_restore(&codec, kCheckpointKey, 0), "checkpoint restore failed");
    const bool restored_state_exact = same_state(saved, inspect(codec));
    Audio replay = retained_prefix(original, boundary);
    decode_range(codec, original_codes, original_widths, boundary, original_codes.frames, replay, started);
    const bool replay_state_exact = same_state(original_final, inspect(codec));
    write_raw(output / "replay.f32", replay.samples);

    prime(codec, reference, started);
    Audio reconstruct;
    decode_range(codec, original_codes, original_widths, 0, boundary, reconstruct, started);
    const bool reconstructed_state_exact = same_state(saved, inspect(codec));
    decode_range(codec, original_codes, original_widths, boundary, original_codes.frames, reconstruct, started);
    const bool reconstruct_final_exact = same_state(original_final, inspect(codec));
    write_raw(output / "reconstruct.f32", reconstruct.samples);

    require(pipeline_codec_stream_restore(&codec, kCheckpointKey, 0), "changed continuation restore failed");
    Audio changed = retained_prefix(original, boundary);
    decode_range(codec, changed_codes, changed_widths, boundary, changed_codes.frames, changed, started);
    const int changed_final_position = codec.stream_pos.at(0);
    const State changed_final = inspect(codec);
    write_raw(output / "changed.f32", changed.samples);

    prime(codec, reference, started);
    Audio candidate_fresh;
    decode_range(codec, changed_codes, changed_widths, 0, changed_codes.frames, candidate_fresh, started);
    const bool changed_vs_fresh_state_exact = same_state(changed_final, inspect(codec));
    write_raw(output / "candidate-fresh.f32", candidate_fresh.samples);
    const bool changed_vs_fresh_pcm_exact = same_pcm(changed.samples, candidate_fresh.samples);
    const bool candidate_fresh_prefix_exact = same_pcm(retained_prefix(original, boundary).samples,
                                                       retained_prefix(candidate_fresh, boundary).samples);
    const bool prefix_exact = same_pcm(retained_prefix(original, boundary).samples,
                                       retained_prefix(changed, boundary).samples);
    const bool replay_exact = same_pcm(original.samples, replay.samples);
    const bool reconstruct_exact = same_pcm(original.samples, reconstruct.samples);
    const bool passed = restored_state_exact && replay_state_exact && reconstructed_state_exact &&
        reconstruct_final_exact && replay_exact && reconstruct_exact && prefix_exact &&
        changed_vs_fresh_pcm_exact && changed_vs_fresh_state_exact && candidate_fresh_prefix_exact;
    json << "{\"directory\":" << json_string(output.filename().string())
         << ",\"schedule\":" << json_string(ramp ? "production-ramp" : "fixed-t1")
         << ",\"boundary_frames\":" << boundary
         << ",\"retained_prefix_samples\":" << boundary * TOKENIZER_HOP_LENGTH
         << ",\"prefix_identity_scope\":\"owned_retained_samples_not_redecoded_candidate_prefix\""
         << ",\"restored_state_exact\":" << restored_state_exact
         << ",\"replay_final_state_exact\":" << replay_state_exact
         << ",\"reconstructed_boundary_state_exact\":" << reconstructed_state_exact
         << ",\"reconstruct_final_state_exact\":" << reconstruct_final_exact
         << ",\"replay_pcm_exact\":" << replay_exact
         << ",\"reconstruct_pcm_exact\":" << reconstruct_exact
         << ",\"retained_pcm_prefix_exact\":" << prefix_exact
         << ",\"changed_vs_fresh_pcm_exact\":" << changed_vs_fresh_pcm_exact
         << ",\"changed_vs_fresh_state_exact\":" << changed_vs_fresh_state_exact
         << ",\"candidate_fresh_prefix_exact\":" << candidate_fresh_prefix_exact
         << ",\"changed_final_position\":" << changed_final_position
         << ",\"checks_passed\":" << passed << ',';
    audio_json(json, "original", original); json << ',';
    audio_json(json, "replay", replay); json << ',';
    audio_json(json, "reconstruct", reconstruct); json << ',';
    audio_json(json, "changed", changed); json << ',';
    audio_json(json, "candidate_fresh", candidate_fresh); json << '}';
    pipeline_codec_snap_free(&find_snapshot(codec, kCheckpointKey));
    return passed;
}

void self_test() {
    require(schedule(14, true) == std::vector<int>({1, 2, 4, 4, 2, 1}), "ramp tail schedule test failed");
    require(schedule(6, true) == std::vector<int>({1, 2, 2, 1}), "ramp short tail test failed");
    require(schedule(3, false) == std::vector<int>({1, 1, 1}), "T1 schedule test failed");
    require(is_boundary(schedule(7, true), 3) && !is_boundary(schedule(7, true), 2),
            "production boundary test failed");
    Codes a{3, std::vector<int32_t>(3 * kCodebooks)};
    for (int k = 0; k < kCodebooks; ++k) {
        for (int t = 0; t < 3; ++t) a.values[static_cast<size_t>(k) * 3 + t] = k * 3 + t;
    }
    Codes b = a;
    b.values[2] += 1;
    require(common_frames(a, b) == 2, "complete16 common prefix test failed");
    require(slice(a, 1, 2)[2] == 4, "codebook-major slice test failed");
    require(json_string("a\n\"\\") == "\"a\\u000a\\\"\\\\\"", "JSON escaping test failed");
}
}  // namespace

int main(int argc, char ** argv) {
    fs::path output;
    bool owns_output = false;
    try {
        static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "requires IEEE754 f32");
        if (argc == 2 && std::string(argv[1]) == "--self-test") {
            self_test();
            std::cout << "codec stream replay host contracts passed\n";
            return 0;
        }
        require(argc == 6, "usage: codec-stream-replay-probe CODEC.gguf REFERENCE.i32|- ORIGINAL.i32 CANDIDATE.i32 NEW_OUTPUT_DIR");
        const uint32_t endian = 1;
        require(*reinterpret_cast<const unsigned char *>(&endian) == 1, "raw output requires a little-endian host");
        const auto started = Clock::now();
        const fs::path model = fs::canonical(fs::path(argv[1]));
        require(fs::is_regular_file(model), "codec GGUF is not a regular file");
        const Codes reference = std::string(argv[2]) == "-" ? Codes{} : read_codes(fs::path(argv[2]), kMaxReferenceFrames);
        const Codes original = read_codes(fs::path(argv[3]), kMaxFrames);
        const Codes candidate = read_codes(fs::path(argv[4]), kMaxFrames);
        const int common = common_frames(original, candidate);
        const fs::path requested = fs::absolute(fs::path(argv[5])).lexically_normal();
        require(requested.has_filename() && requested.filename() != "." && requested.filename() != "..",
                "output must name a new child directory");
        output = fs::canonical(requested.parent_path()) / requested.filename();
        require(!fs::exists(output), "output directory must not exist");
        require(fs::create_directory(output), "cannot create output directory");
        owns_output = true;
        write_raw(output / "reference.i32", reference.values);
        write_raw(output / "original.i32", original.values);
        write_raw(output / "candidate.i32", candidate.values);

        CodecOwner owner;
        owner.backend = backend_init("codec-stream-replay");
        require(owner.backend.backend != nullptr, "backend initialization failed");
        require(pipeline_codec_load(&owner.codec, model.string().c_str(), owner.backend), "codec model load failed");
        owner.loaded = true;
        owner.codec.stream_sets = 2;
        std::ofstream json(output / "result.json");
        require(static_cast<bool>(json), "cannot create result metadata");
        json << std::boolalpha << "{\"scope\":\"teacher-forced-private-codec-state-only\","
             << "\"streaming_qualified\":false,\"live_ar_recovery\":false,\"committed_samples\":0,"
             << "\"timing_qualified\":false,\"state_scope\":\"existing_private_snapshot_tensors_and_position\","
             << "\"input_layout\":\"codebook-major16-int32-le\",\"pcm_format\":\"float32-le-mono-24000\","
             << "\"codec_model\":" << json_string(model.string())
             << ",\"reference_frames\":" << reference.frames
             << ",\"original_frames\":" << original.frames
             << ",\"candidate_frames\":" << candidate.frames
             << ",\"common_prefix_frames\":" << common << ",\"cases\":[";
        bool first = true;
        bool passed = true;
        for (bool ramp : {false, true}) {
            const auto widths = schedule(original.frames, ramp);
            const auto candidate_widths = schedule(candidate.frames, ramp);
            const std::vector<int> boundaries = ramp ? std::vector<int>{0, 1, 3} : std::vector<int>{0, 1, 2};
            for (int boundary : boundaries) {
                if (boundary > common || !is_boundary(widths, boundary) || !is_boundary(candidate_widths, boundary)) continue;
                if (!first) json << ',';
                first = false;
                const std::string name = std::string(ramp ? "ramp-" : "t1-") + std::to_string(boundary);
                passed = run_case(owner.codec, reference, original, candidate, ramp, boundary,
                                  output / name, json, started) && passed;
                json.flush();
                require(static_cast<bool>(json), "cannot flush case metadata");
            }
        }
        json << "],\"checks_passed\":" << passed << "}\n";
        json.close();
        require(static_cast<bool>(json), "cannot flush result metadata");
        return passed ? 0 : 2;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        if (owns_output) {
            std::ofstream failure(output / "error.json");
            failure << "{\"status\":\"failed\",\"error\":" << json_string(error.what()) << "}\n";
        }
        return 1;
    }
}

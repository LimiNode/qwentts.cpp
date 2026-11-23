// Single-lane research experiment. No public ABI, QTB or playback changes.
#include "pipeline-tts.h"
#include "bpe.h"
#include "yyjson.h"
#include "research_pcm_holdback.h"

#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <stdexcept>

namespace {
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using Json = std::unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)>;
using Snapshot = std::unique_ptr<TtsResearchSnapshot, decltype(&tts_engine_research_snapshot_free)>;
constexpr size_t kHop = TOKENIZER_HOP_LENGTH;
constexpr size_t kMaxSamples = 2048 * kHop;

void require(bool value, const char * message) { if (!value) throw std::runtime_error(message); }
double ms(Clock::time_point start) { return std::chrono::duration<double, std::milli>(Clock::now() - start).count(); }
std::string read(const std::string & path, size_t limit) {
    std::ifstream file(fs::u8path(path), std::ios::binary | std::ios::ate);
    require(file && file.tellg() >= 0 && static_cast<size_t>(file.tellg()) <= limit, "invalid bounded input");
    std::string bytes(static_cast<size_t>(file.tellg()), '\0');
    file.seekg(0);
    require(static_cast<bool>(file.read(bytes.data(), static_cast<std::streamsize>(bytes.size()))), "input read failed");
    return bytes;
}
std::string string_field(yyjson_val * object, const char * name) {
    auto * field = yyjson_obj_get(object, name);
    require(yyjson_is_str(field) && yyjson_get_len(field), "missing nonempty string");
    std::string result(yyjson_get_str(field), yyjson_get_len(field));
    require(result.find('\0') == std::string::npos, "embedded NUL");
    return result;
}
int64_t integer(yyjson_val * object, const char * name, int64_t low, int64_t high) {
    auto * field = yyjson_obj_get(object, name);
    require(yyjson_is_int(field), "missing integer");
    const auto result = yyjson_get_sint(field);
    require(result >= low && result <= high, "integer outside bounded range");
    return result;
}
void fields(yyjson_val * object, std::initializer_list<const char *> allowed) {
    require(yyjson_is_obj(object), "expected object");
    size_t index, count;
    yyjson_val * key;
    yyjson_val * value;
    yyjson_obj_foreach(object, index, count, key, value) {
        (void) value;
        const std::string name(yyjson_get_str(key), yyjson_get_len(key));
        bool found = false;
        for (const char * item : allowed) found = found || name == item;
        require(found, "unknown manifest field");
    }
}
template<class T> std::vector<T> raw(const std::string & path, size_t count) {
    const auto bytes = read(path, count * sizeof(T));
    require(!bytes.empty() && bytes.size() % sizeof(T) == 0, "invalid raw tensor");
    std::vector<T> result(bytes.size() / sizeof(T));
    std::memcpy(result.data(), bytes.data(), bytes.size());
    return result;
}
template<class T> void write(const fs::path & path, const std::vector<T> & values) {
    std::ofstream file(path, std::ios::binary);
    if (!values.empty()) file.write(reinterpret_cast<const char *>(values.data()),
                                    static_cast<std::streamsize>(values.size() * sizeof(T)));
    file.close();
    require(static_cast<bool>(file), "artifact write failed");
}
bool equal(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}
struct Group {
    std::string text, language, mode;
    int64_t seed = 0;
    int64_t continuation = -1;
    int boundary = 1;
    bool negative() const { return mode != "baseline" && mode != "replay" && mode != "rollback1"; }
};
Group group_control(yyjson_val * object) {
    fields(object, {"text_file", "language", "mode", "seed", "snapshot_frame", "continuation_seed"});
    Group group;
    group.language = string_field(object, "language");
    group.mode = string_field(object, "mode");
    require(group.mode == "baseline" || group.mode == "replay" || group.mode == "rollback1" ||
            group.mode == "cancel" || group.mode == "deadline" || group.mode == "cap" ||
            group.mode == "sink-fail" || group.mode == "stale" || group.mode == "newer-admission", "unknown mode");
    group.seed = integer(object, "seed", 0, INT64_MAX);
    if (group.mode == "replay") group.boundary = static_cast<int>(integer(object, "snapshot_frame", 1, 2));
    else require(!yyjson_obj_get(object, "snapshot_frame"), "snapshot_frame applies to replay only");
    if (group.mode == "rollback1") group.continuation = integer(object, "continuation_seed", 0, INT64_MAX);
    else require(!yyjson_obj_get(object, "continuation_seed"), "continuation_seed applies to rollback1 only");
    return group;
}
std::vector<Group> groups(yyjson_val * root) {
    auto * list = yyjson_obj_get(root, "groups");
    require(yyjson_is_arr(list) && yyjson_arr_size(list) && yyjson_arr_size(list) <= 256, "invalid groups");
    std::vector<Group> result;
    for (size_t index = 0; index < yyjson_arr_size(list); ++index) {
        auto * object = yyjson_arr_get(list, index);
        auto group = group_control(object);
        group.text = read(string_field(object, "text_file"), 65536);
        require(!group.text.empty() && group.text.find('\0') == std::string::npos, "empty/invalid effective text");
        result.push_back(std::move(group));
    }
    return result;
}

struct Owner {
    BackendPair backend{};
    PipelineTTS pipeline{};
    BPETokenizer tokenizer;
    TtsEngine * engine = nullptr;
    ~Owner() {
        if (engine) tts_engine_free(engine);
        pipeline_tts_free(&pipeline);
        backend_release(backend.backend, backend.cpu_backend);
    }
};
struct Chunk { double ms; size_t samples; size_t committed; bool terminal; int attempt; };
struct Lane {
    Clock::time_point start, deadline;
    ResearchPcmHoldback holdback;
    std::vector<float> seen, committed;
    std::vector<Chunk> chunks;
    const TtsJob * job = nullptr;
    bool cancelled = false;
    bool eos = false;
    bool unknown_job = false;
    bool sink_fail = false;
    int attempt = 0;
    explicit Lane(size_t capacity) : holdback(capacity) {}
    bool alive() const { return !cancelled && Clock::now() < deadline && !holdback.failed(); }
    bool sink(const float * p, size_t n) {
        if (!alive() || sink_fail || committed.size() > kMaxSamples - n) return false;
        committed.insert(committed.end(), p, p + n);
        return true;
    }
};
bool cancel(void * pointer) { return !static_cast<Lane *>(pointer)->alive(); }
bool audio(const float * samples, int count, void * pointer) {
    auto & lane = *static_cast<Lane *>(pointer);
    if (!lane.alive() || count <= 0 || static_cast<size_t>(count) > kMaxSamples - lane.seen.size()) return false;
    auto sink = [&](const float * p, size_t n) { return lane.sink(p, n); };
    if (!lane.holdback.append(samples, static_cast<size_t>(count), lane.eos, sink)) return false;
    lane.seen.insert(lane.seen.end(), samples, samples + count);
    lane.chunks.push_back({ms(lane.start), static_cast<size_t>(count), lane.holdback.committed(), lane.eos, lane.attempt});
    return lane.alive();
}
void decision(const TtsResearchDecision & item, void * pointer) noexcept {
    auto & lane = *static_cast<Lane *>(pointer);
    if (item.job != lane.job) { lane.unknown_job = true; return; }
    if (item.eos) lane.eos = true;
}
struct Checkpoint {
    Snapshot engine{nullptr, tts_engine_research_snapshot_free};
    ResearchPcmHoldback::Snapshot audio;
    TtsResearchSnapshotInfo info;
};
Checkpoint capture(Owner & owner, const Lane & lane) {
    Checkpoint checkpoint;
    checkpoint.engine.reset(tts_engine_research_snapshot(owner.engine));
    require(static_cast<bool>(checkpoint.engine), "live checkpoint unavailable");
    checkpoint.info = tts_engine_research_snapshot_info(checkpoint.engine.get());
    checkpoint.audio = lane.holdback.snapshot();
    require(checkpoint.info.streaming && checkpoint.info.decoded_frames * kHop == checkpoint.audio.decoded &&
            checkpoint.info.frame == checkpoint.info.decoded_frames + checkpoint.info.pending_frames,
            "AR/codec/PCM checkpoint is not coherent");
    if (checkpoint.info.frame == 2) require(checkpoint.info.decoded_frames == 1 &&
        checkpoint.info.pending_frames == 1 && checkpoint.info.codec_target == 2, "frame2 pending/ramp contract mismatch");
    return checkpoint;
}
bool restore(Owner & owner, Lane & lane, const Checkpoint & saved, int64_t seed, double * cost) {
    if (!lane.alive() || !lane.holdback.can_restore(saved.audio)) return false;
    if (!tts_engine_research_stream_restore(owner.engine, saved.engine.get(), seed,
                                           lane.holdback.committed(), cost)) return false;
    require(lane.holdback.restore(saved.audio), "validated PCM restore failed");
    lane.seen.resize(saved.audio.decoded);
    lane.eos = false;
    lane.attempt = 1;
    return true;
}

// Independent fresh reconstruction uses the production decoder, not a copied
// graph or a restored state. AR codes are frame-major; codec expects [K,T].
std::vector<float> fresh_codec(Owner & owner, const std::vector<int32_t> & reference,
                                int reference_frames, const std::vector<int32_t> & frames,
                                const Lane & lane) {
    auto & codec = owner.pipeline.codec;
    const int staging = codec.stream_sets - 1;
    require(pipeline_codec_stream_reset(&codec, staging), "fresh codec staging reset failed");
    for (int offset = 0; offset < reference_frames;) {
        require(lane.alive(), "deadline expired during fresh reconstruction");
        int width = 8;
        while (width > reference_frames - offset) width /= 2;
        std::vector<int32_t> chunk(static_cast<size_t>(width) * 16);
        for (int k = 0; k < 16; ++k) for (int t = 0; t < width; ++t)
            chunk[static_cast<size_t>(k) * width + t] = reference[static_cast<size_t>(k) * reference_frames + offset + t];
        require(pipeline_codec_decode_stream(&codec, chunk.data(), width, nullptr), "fresh reference priming failed");
        offset += width;
    }
    require(pipeline_codec_stream_copy_set(&codec, staging, 0), "fresh reference handoff failed");
    const int count = static_cast<int>(frames.size() / 16);
    std::vector<float> pcm;
    int target = 1;
    for (int offset = 0; offset < count;) {
        require(lane.alive(), "deadline expired during fresh decoding");
        int width = target;
        while (width > count - offset) width /= 2;
        std::vector<int32_t> chunk(static_cast<size_t>(width) * 16);
        for (int k = 0; k < 16; ++k) for (int t = 0; t < width; ++t)
            chunk[static_cast<size_t>(k) * width + t] = frames[static_cast<size_t>(offset + t) * 16 + k];
        std::vector<float> samples(static_cast<size_t>(width) * kHop);
        float * output = samples.data();
        require(pipeline_codec_decode_stream_batch(&codec, chunk.data(), width, 1, &output), "fresh candidate decode failed");
        pcm.insert(pcm.end(), samples.begin(), samples.end());
        offset += width;
        target = std::min(target * 2, 8);
    }
    return pcm;
}

void self_test() {
    const auto parse = [](const char * source) {
        Json doc(yyjson_read(source, std::strlen(source), 0), yyjson_doc_free);
        require(static_cast<bool>(doc), "self-test JSON invalid");
        return group_control(yyjson_doc_get_root(doc.get()));
    };
    require(parse("{\"mode\":\"replay\",\"seed\":64,\"snapshot_frame\":2,\"language\":\"Russian\"}").boundary == 2, "replay parse failed");
    require(parse("{\"mode\":\"rollback1\",\"seed\":2,\"continuation_seed\":64,\"language\":\"Russian\"}").continuation == 64, "rollback parse failed");
    const char * invalid[] = {
        "{\"mode\":\"replay\",\"seed\":1,\"snapshot_frame\":3,\"language\":\"Russian\"}",
        "{\"mode\":\"baseline\",\"seed\":1,\"snapshot_frame\":1,\"language\":\"Russian\"}",
        "{\"mode\":\"rollback1\",\"seed\":1,\"continuation_seed\":-1,\"language\":\"Russian\"}",
        "{\"mode\":\"unknown\",\"seed\":1,\"language\":\"Russian\"}",
        "{\"mode\":\"cancel\",\"seed\":1,\"force\":true,\"language\":\"Russian\"}"};
    for (const char * source : invalid) {
        bool rejected = false;
        try { (void) parse(source); } catch (const std::runtime_error &) { rejected = true; }
        require(rejected, "invalid group accepted");
    }
    require(!tts_engine_research_stream_restore(nullptr, nullptr, -1, 0, nullptr), "null stream restore accepted");
    std::fprintf(stderr, "live stream probe basic contract passed; holdback has separate model-free tests\n");
}
}

int main(int argc, char ** argv) {
    try {
        if (argc == 2 && std::strcmp(argv[1], "--self-test") == 0) { self_test(); return 0; }
        require(argc == 3, "usage: live-stream-recovery-probe manifest.json NEW-output-dir");
        const auto bytes = read(argv[1], 1024 * 1024);
        Json document(yyjson_read(bytes.data(), bytes.size(), 0), yyjson_doc_free);
        require(static_cast<bool>(document), "invalid JSON manifest");
        auto * root = yyjson_doc_get_root(document.get());
        fields(root, {"talker", "codec", "speaker_f32", "reference_i32", "reference_text_file", "reference_frames", "max_tokens", "deadline_ms", "groups"});
        const auto talker = string_field(root, "talker"), codec = string_field(root, "codec");
        const auto reference_text = read(string_field(root, "reference_text_file"), 65536);
        require(!reference_text.empty() && reference_text.find('\0') == std::string::npos, "invalid reference text");
        const auto speaker = raw<float>(string_field(root, "speaker_f32"), 4096);
        const auto reference = raw<int32_t>(string_field(root, "reference_i32"), 16 * 4096);
        const int reference_frames = static_cast<int>(integer(root, "reference_frames", 1, 4096));
        require(reference.size() == static_cast<size_t>(reference_frames) * 16, "invalid reference shape");
        for (float value : speaker) require(std::isfinite(value), "nonfinite speaker embedding");
        for (int32_t value : reference) require(value >= 0 && value < 2048, "reference outside vocabulary");
        const int max_tokens = static_cast<int>(integer(root, "max_tokens", 8, 2048));
        const int deadline_ms = static_cast<int>(integer(root, "deadline_ms", 1, 120000));
        const auto plan = groups(root);
        const char * prohibited[] = {"QWEN_RESEARCH_PREFIX_FILE", "QWEN_PREFILL_PREFIX_ROWS", "QWEN_RESEARCH_FORCE_TALKER_FILE", "QWEN_RESEARCH_FORCE_PREDICTOR_FILE", "QWEN_RESEARCH_SAMPLER_TARGET_FRAME"};
        for (const char * name : prohibited) { const char * value = std::getenv(name); require(!value || !*value, "incompatible research environment"); }
        const char * schedule = std::getenv("QWEN_RESEARCH_FRAME_SEED_POLICY");
        require(!schedule || !*schedule || std::strcmp(schedule, "baseline") == 0, "nonbaseline seed schedule");
        const fs::path destination = fs::u8path(argv[2]);
        require(fs::create_directory(destination), "output exists");
        std::ofstream results(destination / "results.jsonl", std::ios::binary);
        require(static_cast<bool>(results), "results cannot be created");
        results << std::setprecision(12);
        Owner owner;
        owner.backend = backend_init("LiveResearch");
        require(owner.backend.backend && owner.backend.has_gpu, "requires selected GPU");
        require(pipeline_tts_load(&owner.pipeline, talker.c_str(), codec.c_str(), owner.backend, true, false, 1, 24.0f, 8) &&
                load_bpe_from_gguf(&owner.tokenizer, talker.c_str()), "model/tokenizer load failed");
        const char * specials[] = {"qwen3-tts.text.im_start_id", "qwen3-tts.text.im_end_id", "qwen3-tts.text.tts_pad_id", "qwen3-tts.text.tts_bos_id", "qwen3-tts.text.tts_eos_id"};
        bpe_load_specials_from_keys(&owner.tokenizer, talker.c_str(), specials, 5);
        require(owner.pipeline.model_type == "base" && owner.pipeline.num_code_groups == 16 &&
                speaker.size() == static_cast<size_t>(owner.pipeline.talker.hidden_size), "requires Base ICL complete16 model");
        owner.engine = tts_engine_new(&owner.pipeline, &owner.tokenizer);
        require(owner.engine != nullptr, "engine allocation failed");
        for (size_t group_index = 0; group_index < plan.size(); ++group_index) {
            const auto & group = plan[group_index];
            const auto stem = "group-" + std::to_string(group_index);
            Lane lane((group.mode == "cap" ? 1 : 16) * kHop);
            lane.start = Clock::now(); lane.deadline = lane.start + std::chrono::milliseconds(deadline_ms);
            lane.sink_fail = group.mode == "sink-fail";
            qt_tts_params params{};
            require(qt_tts_default_params_ex(&params, sizeof(params)) == QT_STATUS_OK, "default params failed");
            params.text = group.text.c_str(); params.lang = group.language.c_str(); params.seed = group.seed;
            params.max_new_tokens = max_tokens; params.ref_spk_emb = speaker.data(); params.ref_spk_dim = static_cast<int>(speaker.size());
            params.ref_codes = reference.data(); params.ref_T = reference_frames; params.ref_text = reference_text.c_str();
            params.on_chunk = audio; params.on_chunk_user_data = &lane; params.cancel = cancel; params.cancel_user_data = &lane;
            qt_audio output{};
            TtsJob job{};
            job.params = &params; job.resolved_seed = params.seed; job.out = &output; lane.job = &job;
            require(tts_engine_research_capture_codes(owner.engine, true), "capture configure failed");
            tts_engine_set_research_observer(owner.engine, decision, &lane);
            require(tts_engine_admit(owner.engine, &job), "admission failed");
            const double admission_ms = ms(lane.start);
            std::array<Checkpoint, 2> ring;
            Checkpoint replay_checkpoint;
            int frame = 0;
            int restore_frame = -1;
            TtsResearchSnapshotInfo restored_info{};
            size_t committed_at_restore = 0;
            bool triggered = false, replay_step_equal = true, negative_pass = false, stale_guard = true;
            double capture_ms = 0.0, restore_ms = 0.0;
            int prefill_count = 1;
            size_t peak_snapshot_bytes = 0;
            std::vector<float> committed_prefix;
            std::vector<float> original_pcm;
            std::vector<int32_t> original_codes;
            int original_frames = 0, original_status = QT_STATUS_OK, original_finish = QT_FINISH_UNKNOWN;
            std::vector<TtsJob *> retired;
            try {
            while (tts_engine_active(owner.engine)) {
                if (group.mode == "rollback1" && frame <= 8) {
                    auto saved = capture(owner, lane);
                    capture_ms += saved.info.capture_ms;
                    ring[static_cast<size_t>(frame) % 2] = std::move(saved);
                    peak_snapshot_bytes = std::max(peak_snapshot_bytes, ring[0].info.bytes + ring[1].info.bytes);
                }
                if (!replay_checkpoint.engine && (group.mode == "replay" || group.negative()) && frame == group.boundary) {
                    replay_checkpoint = capture(owner, lane);
                    capture_ms += replay_checkpoint.info.capture_ms;
                    peak_snapshot_bytes = replay_checkpoint.info.bytes;
                    const size_t stale = replay_checkpoint.info.decoded_frames * kHop + 1;
                    stale_guard = !tts_engine_research_stream_restore(owner.engine, replay_checkpoint.engine.get(), -1, stale, nullptr);
                    require(stale_guard, "stale watermark accepted");
                    if (group.mode == "cancel") lane.cancelled = true;
                    if (group.mode == "deadline") lane.deadline = Clock::now() - std::chrono::milliseconds(1);
                }
                tts_engine_step(owner.engine, &retired);
                ++frame;
                if (group.mode == "replay" && !triggered && replay_checkpoint.engine && frame == group.boundary + 1) {
                    const auto before = lane.seen;
                    write(destination / (stem + "-replay-step-original.f32"), before);
                    committed_prefix = lane.committed;
                    committed_at_restore = lane.holdback.committed();
                    restored_info = replay_checkpoint.info;
                    restore_frame = restored_info.frame;
                    require(restore(owner, lane, replay_checkpoint, -1, &restore_ms), "unchanged live restore failed");
                    triggered = true;
                    // EOS can retire the original step (seed75 at checkpoint2).
                    // Its internal notification is superseded by this replay.
                    retired.clear();
                    tts_engine_step(owner.engine, &retired);
                    write(destination / (stem + "-replay-step-restored.f32"), lane.seen);
                    replay_step_equal = equal(before, lane.seen) && equal(committed_prefix, lane.committed);
                    write(destination / (stem + "-replay-before.f32"), before);
                    write(destination / (stem + "-replay-after.f32"), lane.seen);
                    require(replay_step_equal, "unchanged live one-step PCM mismatch");
                }
                if (group.mode == "stale" && replay_checkpoint.engine && lane.holdback.committed() > replay_checkpoint.audio.decoded) {
                    negative_pass = !lane.holdback.can_restore(replay_checkpoint.audio) &&
                        !tts_engine_research_stream_restore(owner.engine, replay_checkpoint.engine.get(), -1, lane.holdback.committed(), nullptr);
                    lane.cancelled = true;
                }
            }
            require(retired.size() == 1 && retired[0] == &job && !lane.unknown_job, "unexpected original retirement");
            const auto * captured = tts_engine_research_codes(owner.engine, &job);
            require(captured != nullptr, "missing original complete codes");
            original_codes = *captured; original_pcm = lane.seen;
            original_frames = static_cast<int>(original_codes.size() / 16);
            original_status = job.status; original_finish = job.finish_reason;
            if (group.mode == "rollback1" && job.status == QT_STATUS_OK && job.finish_reason == QT_FINISH_EOS && job.n_frames <= 8) {
                const int target = job.n_frames - 1;
                require(target >= 0, "rollback before first frame");
                auto & saved = ring[static_cast<size_t>(target) % 2];
                require(saved.info.frame == target, "rollback ring checkpoint unavailable");
                committed_prefix = lane.committed; committed_at_restore = lane.holdback.committed();
                restored_info = saved.info; restore_frame = saved.info.frame;
                write(destination / (stem + "-original.f32"), original_pcm);
                write(destination / (stem + "-original.i32"), original_codes);
                require(restore(owner, lane, saved, group.continuation, &restore_ms), "natural-EOS live rollback failed");
                triggered = true; retired.clear();
                while (tts_engine_active(owner.engine)) tts_engine_step(owner.engine, &retired);
                require(retired.size() == 1 && retired[0] == &job, "unexpected restored retirement");
            }
            if (group.negative()) {
                committed_prefix = lane.committed;
                restored_info = replay_checkpoint.info;
                if (group.mode == "newer-admission") {
                    require(replay_checkpoint.engine && job.status == QT_STATUS_OK && job.finish_reason == QT_FINISH_EOS, "newer admission requires originalEOS");
                    lane.cancelled = false; lane.eos = false;
                    require(lane.alive() && tts_engine_admit(owner.engine, &job), "newer admission failed");
                    ++prefill_count;
                    const bool active_rejected = !tts_engine_research_stream_restore(owner.engine,
                        replay_checkpoint.engine.get(), -1, 0, nullptr);
                    lane.cancelled = true; retired.clear();
                    while (tts_engine_active(owner.engine)) tts_engine_step(owner.engine, &retired);
                    // Isolate admission identity from current cancellation and
                    // retirement status, which are independently tested above.
                    const auto actual_status = job.status;
                    const auto actual_finish = job.finish_reason;
                    lane.cancelled = false;
                    job.status = QT_STATUS_OK; job.finish_reason = QT_FINISH_EOS;
                    negative_pass = active_rejected && !tts_engine_research_stream_restore(owner.engine,
                        replay_checkpoint.engine.get(), -1, 0, nullptr);
                    job.status = actual_status; job.finish_reason = actual_finish;
                } else if (group.mode != "stale") {
                    negative_pass = replay_checkpoint.engine && job.status == QT_STATUS_CANCELLED &&
                        !tts_engine_research_stream_restore(owner.engine, replay_checkpoint.engine.get(), -1, lane.holdback.committed(), nullptr);
                    lane.holdback.abort();
                    negative_pass = negative_pass && !restore(owner, lane, replay_checkpoint, -1, nullptr);
                }
                require(negative_pass, "negative recovery guard failed");
            }
            const auto * final = tts_engine_research_codes(owner.engine, &job);
            require(final != nullptr, "missing final complete codes");
            const auto final_codes = *final;
            const bool prefix_equal = lane.committed.size() >= committed_prefix.size() &&
                (committed_prefix.empty() || std::memcmp(lane.committed.data(), committed_prefix.data(), committed_prefix.size() * sizeof(float)) == 0);
            require(prefix_equal, "committed prefix changed");
            bool fresh_equal = false;
            std::vector<float> fresh;
            if (!group.negative()) {
                require(job.status == QT_STATUS_OK && job.finish_reason == QT_FINISH_EOS && lane.alive(), "final outcome not naturalEOS");
                require(lane.seen.size() == final_codes.size() / 16 * kHop, "codec/AR sample count mismatch");
                fresh = fresh_codec(owner, reference, reference_frames, final_codes, lane);
                fresh_equal = equal(fresh, lane.seen);
                write(destination / (stem + ".f32"), lane.seen);
                write(destination / (stem + "-codes.i32"), final_codes);
                write(destination / (stem + "-fresh-codec.f32"), fresh);
                require(fresh_equal, "fresh candidate streaming decode mismatch");
                auto sink = [&](const float * p, size_t n) { return lane.sink(p, n); };
                require(lane.holdback.finish(sink) && equal(lane.committed, lane.seen), "final commit mismatch");
            }
            if ((group.mode == "rollback1" && !triggered) || group.negative()) {
                write(destination / (stem + "-original.f32"), original_pcm);
                write(destination / (stem + "-original.i32"), original_codes);
            }
            write(destination / (stem + ".f32"), lane.seen);
            write(destination / (stem + "-committed.f32"), lane.committed);
            write(destination / (stem + "-codes.i32"), final_codes);
            write(destination / (stem + "-committed-prefix.f32"), committed_prefix);
            write(destination / (stem + "-fresh-codec.f32"), fresh);
            results << "{\"group\":" << group_index << ",\"seed\":" << group.seed << ",\"mode\":\"" << group.mode
                << "\",\"status\":" << job.status << ",\"finish_reason\":" << job.finish_reason
                << ",\"frames\":" << final_codes.size() / 16 << ",\"engine_reported_frames\":" << job.n_frames
                << ",\"samples\":" << lane.seen.size() << ",\"original_frames\":" << original_frames
                << ",\"original_status\":" << original_status << ",\"original_finish_reason\":" << original_finish
                << ",\"triggered\":" << (triggered ? "true" : "false") << ",\"restore_frame\":" << restore_frame
                << ",\"continuation_seed\":" << (triggered && group.mode == "rollback1" ? group.continuation : -1)
                << ",\"decoded_frames\":" << restored_info.decoded_frames << ",\"pending_frames\":" << restored_info.pending_frames
                << ",\"codec_target\":" << restored_info.codec_target << ",\"committed_samples_at_restore\":" << committed_at_restore
                << ",\"held_peak_samples\":" << lane.holdback.peak() << ",\"holdback_cap_samples\":" << (group.mode == "cap" ? 1 : 16) * kHop
                << ",\"committed_samples\":" << lane.holdback.committed() << ",\"committed_prefix_equal\":" << (prefix_equal ? "true" : "false")
                << ",\"unchanged_step_pcm_equal\":" << (replay_step_equal ? "true" : "false")
                << ",\"fresh_codec_pcm_equal\":" << (fresh_equal ? "true" : "false")
                << ",\"stale_restore_rejected\":" << (stale_guard ? "true" : "false")
                << ",\"negative_guard_pass\":" << (negative_pass ? "true" : "false")
                << ",\"restore_rejected\":" << (negative_pass ? "true" : "false")
                << ",\"checks_passed\":true"
                << ",\"capture_ms\":" << capture_ms << ",\"restore_ms\":" << restore_ms
                << ",\"snapshot_payload_bytes\":" << peak_snapshot_bytes << ",\"admission_ms\":" << admission_ms
                << ",\"group_total_ms\":" << ms(lane.start) << ",\"prefill_count\":" << prefill_count << ",\"physical_playback\":false,\"chunks\":[";
            for (size_t i = 0; i < lane.chunks.size(); ++i) {
                if (i) results << ',';
                const auto & chunk = lane.chunks[i];
                results << "{\"ms\":" << chunk.ms << ",\"samples\":" << chunk.samples << ",\"committed\":" << chunk.committed
                    << ",\"terminal\":" << (chunk.terminal ? "true" : "false") << ",\"attempt\":" << chunk.attempt << '}';
            }
            results << "]}\n"; results.flush();
            require(static_cast<bool>(results), "results flush failed");
            tts_engine_set_research_observer(owner.engine, nullptr, nullptr);
            qt_audio_free(&output);
            } catch (...) {
                // Retain the failed trajectory, not just preceding good rows.
                // Cancellation retires an active slot without another AR frame.
                try {
                    lane.cancelled = true;
                    if (tts_engine_active(owner.engine)) tts_engine_step(owner.engine, &retired);
                    write(destination / (stem + "-failed-observed.f32"), lane.seen);
                    write(destination / (stem + "-failed-committed.f32"), lane.committed);
                    write(destination / (stem + "-failed-original.f32"), original_pcm);
                    write(destination / (stem + "-failed-original.i32"), original_codes);
                    const auto * codes = tts_engine_research_codes(owner.engine, &job);
                    if (codes) write(destination / (stem + "-failed-codes.i32"), *codes);
                    results << "{\"group\":" << group_index << ",\"seed\":" << group.seed
                        << ",\"mode\":\"" << group.mode << "\",\"checks_passed\":false,\"status\":"
                        << job.status << ",\"finish_reason\":" << job.finish_reason << "}\n";
                    results.flush();
                } catch (...) {
                    std::fprintf(stderr, "failed-attempt evidence finalization also failed\n");
                }
                tts_engine_set_research_observer(owner.engine, nullptr, nullptr);
                qt_audio_free(&output);
                throw;
            }
        }
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "live stream recovery probe: %s\n", error.what());
        return 1;
    }
}

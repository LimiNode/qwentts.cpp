// Bounded engine-internal experiment, not the QTB scheduler or a player.
#include "pipeline-tts.h"
#include "bpe.h"
#include "yyjson.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
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
bool optional_boolean(yyjson_val * object, const char * key, bool fallback) {
    yyjson_val * value = yyjson_obj_get(object, key);
    if (value && !yyjson_is_bool(value)) throw std::runtime_error(std::string(key) + " must be boolean");
    return value ? yyjson_get_bool(value) : fallback;
}

enum class RecoveryPolicy { None, Replay, EosResample, Rollback1, Rollback2 };
struct RecoveryConfig {
    RecoveryPolicy policy = RecoveryPolicy::None;
    std::string name = "none";
    int trigger_max_frames = 8;
    int snapshot_frame = 1;
    int64_t continuation_seed = 0;
};

RecoveryConfig recovery_config(yyjson_val * group) {
    RecoveryConfig result;
    auto * value = yyjson_obj_get(group, "recovery");
    if (!value) return result;
    if (!yyjson_is_obj(value)) throw std::runtime_error("recovery must be an object");
    size_t index, count;
    yyjson_val * key;
    yyjson_val * field;
    yyjson_obj_foreach(value, index, count, key, field) {
        const std::string name(yyjson_get_str(key), yyjson_get_len(key));
        if (name != "policy" && name != "trigger_max_frames" && name != "snapshot_frame" &&
            name != "continuation_seed") throw std::runtime_error("unknown recovery field: " + name);
    }
    result.name = string_field(value, "policy");
    if (result.name == "replay") result.policy = RecoveryPolicy::Replay;
    else if (result.name == "eos-resample") result.policy = RecoveryPolicy::EosResample;
    else if (result.name == "rollback-1") result.policy = RecoveryPolicy::Rollback1;
    else if (result.name == "rollback-2") result.policy = RecoveryPolicy::Rollback2;
    else throw std::runtime_error("unknown recovery policy");
    if (yyjson_obj_get(value, "trigger_max_frames")) {
        result.trigger_max_frames = integer_field(value, "trigger_max_frames", 1, 8);
    }
    if (result.policy == RecoveryPolicy::Replay) {
        if (yyjson_obj_get(value, "continuation_seed")) throw std::runtime_error("replay preserves original seed");
        result.snapshot_frame = integer_field(value, "snapshot_frame", 0, 8);
    } else {
        if (yyjson_obj_get(value, "snapshot_frame")) throw std::runtime_error("intervention chooses rollback boundary");
        auto * seed = yyjson_obj_get(value, "continuation_seed");
        if (!yyjson_is_int(seed) || yyjson_get_sint(seed) < 0) {
            throw std::runtime_error("recovery requires explicit nonnegative continuation_seed");
        }
        result.continuation_seed = yyjson_get_sint(seed);
    }
    return result;
}

std::vector<RecoveryConfig> group_recovery_configs(yyjson_val * groups, int max_batch) {
    std::vector<RecoveryConfig> result;
    for (size_t index = 0; index < yyjson_arr_size(groups); ++index) {
        auto * group = yyjson_arr_get(groups, index);
        auto config = recovery_config(group);
        if (config.policy != RecoveryPolicy::None) {
            if (max_batch != 1 || yyjson_arr_size(yyjson_obj_get(group, "seeds")) != 1 ||
                optional_boolean(group, "diagnostic_dump", false)) {
                throw std::runtime_error("recovery stage requires one buffered lane and no dumps");
            }
            const char * prefix = std::getenv("QWEN_RESEARCH_PREFIX_FILE");
            const char * prefix_rows = std::getenv("QWEN_PREFILL_PREFIX_ROWS");
            const char * schedule = std::getenv("QWEN_RESEARCH_FRAME_SEED_POLICY");
            if ((prefix && *prefix) || (prefix_rows && *prefix_rows) ||
                (schedule && *schedule && std::strcmp(schedule, "baseline"))) {
                throw std::runtime_error("recovery stage requires prefix OFF and baseline seed schedule");
            }
        }
        result.push_back(std::move(config));
    }
    return result;
}

void recovery_self_test() {
    const auto parse = [](const char * json) {
        JsonDoc doc(yyjson_read(json, std::strlen(json), 0), yyjson_doc_free);
        if (!doc) throw std::runtime_error("invalid recovery self-test JSON");
        return recovery_config(yyjson_doc_get_root(doc.get()));
    };
    if (parse("{}").policy != RecoveryPolicy::None ||
        parse("{\"recovery\":{\"policy\":\"replay\",\"snapshot_frame\":0}}").snapshot_frame != 0 ||
        parse("{\"recovery\":{\"policy\":\"rollback-2\",\"continuation_seed\":64}}").policy != RecoveryPolicy::Rollback2 ||
        parse("{\"recovery\":{\"policy\":\"eos-resample\",\"continuation_seed\":0}}").continuation_seed != 0) {
        throw std::runtime_error("recovery policy parsing failed");
    }
    const char * invalid[] = {
        "{\"recovery\":null}", "{\"recovery\":{}}", "{\"recovery\":{\"policy\":\"none\"}}",
        "{\"recovery\":{\"policy\":\"rollback-1\"}}",
        "{\"recovery\":{\"policy\":\"rollback-1\",\"continuation_seed\":-1}}",
        "{\"recovery\":{\"policy\":\"rollback-1\",\"continuation_seed\":1,\"snapshot_frame\":0}}",
        "{\"recovery\":{\"policy\":\"replay\",\"snapshot_frame\":9}}",
        "{\"recovery\":{\"policy\":\"replay\",\"snapshot_frame\":1,\"continuation_seed\":1}}",
        "{\"recovery\":{\"policy\":\"replay\",\"snapshot_frame\":1,\"trigger_max_frames\":0}}",
        "{\"recovery\":{\"policy\":\"replay\",\"snapshot_frame\":1,\"trigger_max_frames\":9}}",
        "{\"recovery\":{\"policy\":\"replay\",\"snapshot_frame\":1,\"force\":true}}"};
    for (const char * json : invalid) {
        bool rejected = false;
        try { (void) parse(json); } catch (const std::runtime_error &) { rejected = true; }
        if (!rejected) throw std::runtime_error("invalid recovery policy accepted");
    }
    if (tts_engine_research_snapshot(nullptr) ||
        tts_engine_research_restore(nullptr, nullptr, 0, nullptr) ||
        tts_engine_research_capture_codes(nullptr, true) ||
        tts_engine_research_codes(nullptr, nullptr)) {
        throw std::runtime_error("null recovery guard failed");
    }
}

enum class SamplerControl { Default, TalkerTop1, TalkerTop2, TalkerTop4, PredictorGreedy };

SamplerControl sampler_control(yyjson_val * group) {
    if (!yyjson_is_obj(group)) throw std::runtime_error("group must be an object");
    if (!yyjson_obj_get(group, "sampler_control")) return SamplerControl::Default;
    const auto name = string_field(group, "sampler_control");
    if (name == "default") return SamplerControl::Default;
    if (name == "talker-top1") return SamplerControl::TalkerTop1;
    if (name == "talker-top2") return SamplerControl::TalkerTop2;
    if (name == "talker-top4") return SamplerControl::TalkerTop4;
    if (name == "predictor-greedy") return SamplerControl::PredictorGreedy;
    throw std::runtime_error("sampler_control must be default, talker-top1/top2/top4 or predictor-greedy");
}

const char * sampler_control_name(SamplerControl control) {
    switch (control) {
        case SamplerControl::Default: return "default";
        case SamplerControl::TalkerTop1: return "talker-top1";
        case SamplerControl::TalkerTop2: return "talker-top2";
        case SamplerControl::TalkerTop4: return "talker-top4";
        case SamplerControl::PredictorGreedy: return "predictor-greedy";
    }
    throw std::runtime_error("invalid internal sampler control");
}

std::vector<SamplerControl> group_sampler_controls(yyjson_val * groups) {
    if (!yyjson_is_arr(groups) || yyjson_arr_size(groups) == 0 || yyjson_arr_size(groups) > 256) {
        throw std::runtime_error("expected 1..256 groups");
    }
    std::vector<SamplerControl> result;
    for (size_t index = 0; index < yyjson_arr_size(groups); ++index) {
        result.push_back(sampler_control(yyjson_arr_get(groups, index)));
    }
    return result;
}

void apply_sampler_control(qt_tts_params & params, SamplerControl control) {
    if (control == SamplerControl::TalkerTop1 || control == SamplerControl::TalkerTop2 ||
        control == SamplerControl::TalkerTop4) {
        // Keep the stochastic path's repetition processing, unlike native greedy.
        params.do_sample = true;
        params.top_k = control == SamplerControl::TalkerTop1 ? 1 :
                       control == SamplerControl::TalkerTop2 ? 2 : 4;
    } else if (control == SamplerControl::PredictorGreedy) {
        params.subtalker_do_sample = false;
    }
}

void write_effective_sampler(std::ostream & output, const qt_tts_params & params, int predictor_top_k) {
    output << "{\"do_sample\":" << (params.do_sample ? "true" : "false")
        << ",\"temperature\":" << (params.do_sample ? params.temperature : 0.0f)
        << ",\"top_k\":" << params.top_k << ",\"top_p\":" << params.top_p
        << ",\"repetition_penalty\":" << params.repetition_penalty
        << ",\"subtalker_do_sample\":" << (params.subtalker_do_sample ? "true" : "false")
        << ",\"subtalker_temperature\":" << (params.subtalker_do_sample ? params.subtalker_temperature : 0.0f)
        // The graph bakes model top-k; it has no per-request nucleus filter.
        << ",\"subtalker_top_k\":" << predictor_top_k << ",\"subtalker_top_p\":1.0"
        << ",\"subtalker_top_k_source\":\"model_default\",\"subtalker_repetition_penalty\":1.0"
        << ",\"min_new_tokens\":2,\"eos_guard_enabled\":" << (params.eos_guard_enabled ? "true" : "false")
        << ",\"eos_guard_start_ratio\":" << params.eos_guard_start_ratio
        << ",\"eos_guard_max_ratio\":" << params.eos_guard_max_ratio
        << ",\"eos_guard_force_ratio\":" << params.eos_guard_force_ratio
        << ",\"eos_guard_max_boost\":" << params.eos_guard_max_boost
        << ",\"eos_guard_voice_multiplier\":" << params.eos_guard_voice_multiplier
        << ",\"eos_guard_min_expected_frames\":" << params.eos_guard_min_expected_frames
        << ",\"eos_guard_frames_per_text_token\":" << params.eos_guard_frames_per_text_token << '}';
}

void sampler_self_test() {
    const auto parse = [](const char * json) {
        JsonDoc doc(yyjson_read(json, std::strlen(json), 0), yyjson_doc_free);
        if (!doc) throw std::runtime_error("invalid sampler self-test JSON");
        return group_sampler_controls(yyjson_doc_get_root(doc.get()));
    };
    const auto controls = parse("[{},{\"sampler_control\":\"default\"},{\"sampler_control\":\"talker-top1\"},"
                                "{\"sampler_control\":\"predictor-greedy\"},"
                                "{\"sampler_control\":\"talker-top2\"},{\"sampler_control\":\"talker-top4\"}]");
    if (controls != std::vector<SamplerControl>{SamplerControl::Default, SamplerControl::Default,
                                               SamplerControl::TalkerTop1, SamplerControl::PredictorGreedy,
                                               SamplerControl::TalkerTop2, SamplerControl::TalkerTop4}) {
        throw std::runtime_error("sampler control parsing failed");
    }
    const char * invalid[] = {"[]", "{}", "[null]", "[false]", "[{\"sampler_control\":null}]",
        "[{\"sampler_control\":false}]", "[{\"sampler_control\":1}]", "[{\"sampler_control\":\"\"}]",
        "[{\"sampler_control\":\"DEFAULT\"}]", "[{\"sampler_control\":\"talker_top1\"}]",
        "[{\"sampler_control\":\"default\\u0000\"}]", "[{}, {\"sampler_control\":\"greedy\"}]"};
    for (const char * json : invalid) {
        bool rejected = false;
        try { (void) parse(json); } catch (const std::runtime_error &) { rejected = true; }
        if (!rejected) throw std::runtime_error("invalid sampler control was accepted");
    }
    for (SamplerControl control : controls) {
        qt_tts_params params;
        if (qt_tts_default_params_ex(&params, sizeof(params)) != QT_STATUS_OK) {
            throw std::runtime_error("sampler defaults unavailable");
        }
        apply_sampler_control(params, control);
        const int expected_top_k = control == SamplerControl::TalkerTop1 ? 1 :
                                   control == SamplerControl::TalkerTop2 ? 2 :
                                   control == SamplerControl::TalkerTop4 ? 4 : 50;
        if (!params.do_sample || params.top_k != expected_top_k
                || params.subtalker_do_sample != (control != SamplerControl::PredictorGreedy)
                || std::fabs(params.repetition_penalty - 1.05f) > 1e-6f || params.eos_guard_enabled) {
            throw std::runtime_error("sampler control changed an unrelated default");
        }
        std::ostringstream output;
        write_effective_sampler(output, params, 37);
        const auto json = output.str();
        JsonDoc doc(yyjson_read(json.data(), json.size(), 0), yyjson_doc_free);
        if (!doc) throw std::runtime_error("invalid effective sampler JSON");
        auto * root = yyjson_doc_get_root(doc.get());
        if (integer_field(root, "top_k", 1, 50) != params.top_k
                || integer_field(root, "subtalker_top_k", 1, 50) != 37
                || integer_field(root, "min_new_tokens", 1, 2) != 2
                || optional_boolean(root, "subtalker_do_sample", true) != params.subtalker_do_sample
                || optional_boolean(root, "eos_guard_enabled", true)
                || std::fabs(yyjson_get_num(yyjson_obj_get(root, "subtalker_temperature"))
                             - (params.subtalker_do_sample ? 0.9 : 0.0)) > 1e-6) {
            throw std::runtime_error("effective sampler echo differs from applied settings");
        }
        if (control == SamplerControl::TalkerTop1 && std::strcmp(sampler_control_name(control), "talker-top1")) {
            throw std::runtime_error("sampler control name differs");
        }
    }
}

// Existing manifests omit speaker and keep their Base ICL contract. A preset
// speaker explicitly selects CustomVoice and cannot silently consume ICL files.
bool custom_voice_conditioning(yyjson_val * object) {
    if (!yyjson_obj_get(object, "speaker")) return false;
    (void) string_field(object, "speaker");
    const char * reference_keys[] = {"reference_text_file", "speaker_f32", "reference_i32", "reference_frames"};
    for (const char * key : reference_keys) {
        if (yyjson_obj_get(object, key)) throw std::runtime_error("speaker and ICL reference fields are mutually exclusive");
    }
    return true;
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
struct Step { double ms; int active; size_t retired; int attempt = 0; };
struct RecoveryObservation {
    bool enabled = false;
    bool triggered = false;
    bool replay_codes_equal = false;
    bool replay_pcm_equal = false;
    int original_frames = 0;
    int original_status = 0;
    int original_finish_reason = 0;
    int restore_frame = -1;
    int64_t continuation_seed = 0;
    double snapshot_ms = 0.0;
    double restore_ms = 0.0;
    double original_completed_ms = 0.0;
    size_t peak_snapshot_bytes = 0;
    std::vector<TtsResearchDecision> original_decisions;
};
using SnapshotOwner = std::unique_ptr<TtsResearchSnapshot, decltype(&tts_engine_research_snapshot_free)>;

template<class T>
void write_raw(const std::filesystem::path & path, const std::vector<T> & values) {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char *>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(T)));
    if (!file) throw std::runtime_error("bounded recovery artifact write failed");
}

std::vector<float> buffered_pcm(const qt_audio & audio) {
    if (audio.n_samples < 0 || static_cast<size_t>(audio.n_samples) > kMaxSamples ||
        (audio.n_samples && !audio.samples)) throw std::runtime_error("invalid buffered PCM size");
    std::vector<float> pcm;
    if (audio.n_samples) pcm.assign(audio.samples, audio.samples + audio.n_samples);
    for (float sample : pcm) {
        if (!std::isfinite(sample)) throw std::runtime_error("nonfinite buffered PCM");
    }
    return pcm;
}

void buffered_self_test() {
    qt_audio audio{};
    if (!buffered_pcm(audio).empty()) throw std::runtime_error("empty buffered PCM changed");
    float sample = 0.25f;
    audio.samples = &sample;
    audio.n_samples = 1;
    if (buffered_pcm(audio) != std::vector<float>{sample}) throw std::runtime_error("buffered PCM copy changed");
    const auto must_reject = [](const qt_audio & value) {
        bool rejected = false;
        try { (void) buffered_pcm(value); } catch (const std::runtime_error &) { rejected = true; }
        if (!rejected) throw std::runtime_error("invalid buffered PCM accepted");
    };
    audio.n_samples = -1;
    must_reject(audio);
    audio.n_samples = static_cast<int>(kMaxSamples + 1);
    must_reject(audio);
    audio.n_samples = 1;
    audio.samples = nullptr;
    must_reject(audio);
    audio.samples = &sample;
    sample = INFINITY;
    must_reject(audio);
}

void write_recovery_observation(std::ostream & results, const RecoveryConfig & config,
                                 const RecoveryObservation & observation, Clock::time_point start) {
    if (!observation.enabled) { results << "null"; return; }
    results << "{\"policy\":\"" << config.name << "\",\"buffered_uncommitted\":true"
        << ",\"committed_samples\":0,\"prefill_count\":1,\"streaming_qualified\":false"
        << ",\"trigger_max_frames\":" << config.trigger_max_frames
        << ",\"triggered\":" << (observation.triggered ? "true" : "false")
        << ",\"attempt_count\":" << (observation.triggered ? 2 : 1)
        << ",\"original_frames\":" << observation.original_frames
        << ",\"original_status\":" << observation.original_status
        << ",\"original_finish_reason\":" << observation.original_finish_reason
        << ",\"original_completed_ms\":" << observation.original_completed_ms
        << ",\"restore_frame\":" << observation.restore_frame
        << ",\"continuation_seed\":" << observation.continuation_seed
        << ",\"snapshot_ms\":" << observation.snapshot_ms
        << ",\"restore_ms\":" << observation.restore_ms
        << ",\"peak_snapshot_bytes\":" << observation.peak_snapshot_bytes
        << ",\"snapshot_byte_scope\":\"host_copies_of_device_payload_only\""
        << ",\"replay_codes_equal\":";
    if (config.policy == RecoveryPolicy::Replay && observation.triggered) {
        results << (observation.replay_codes_equal ? "true" : "false");
    } else results << "null";
    results << ",\"replay_pcm_equal\":";
    if (config.policy == RecoveryPolicy::Replay && observation.triggered) {
        results << (observation.replay_pcm_equal ? "true" : "false");
    } else results << "null";
    results << ",\"original_decisions\":[";
    for (size_t index = 0; index < observation.original_decisions.size(); ++index) {
        if (index) results << ',';
        const auto & decision = observation.original_decisions[index];
        const double ms = std::chrono::duration<double, std::milli>(decision.selected_at - start).count();
        results << "{\"ms\":" << ms << ",\"frame\":" << decision.frame << ",\"token\":" << decision.token
            << ",\"sampled_token\":" << decision.sampled_token << ",\"eos\":" << (decision.eos ? "true" : "false") << '}';
    }
    results << "]}";
}
struct Lane {
    Clock::time_point start, deadline;
    double admission_started_ms = 0.0;
    double admission_completed_ms = 0.0;
    std::vector<float> pcm;
    std::vector<Chunk> chunks;
    std::array<TtsResearchDecision, TTS_RESEARCH_MAX_DECISIONS> decisions{};
    size_t decision_count = 0;
    bool decision_overflow = false;
    bool invalid_pcm = false;
    bool capacity_exceeded = false;
#ifdef QWEN_RESEARCH_PREFILL
    ResearchPrefillObservation prefill;
#endif
};
struct DecisionCapture {
    std::array<Lane, 2> * observations;
    const std::array<TtsJob, 2> * jobs;
    size_t lanes;
    bool unknown_job = false;
};

void on_decision(const TtsResearchDecision & decision, void * pointer) noexcept {
    auto & capture = *static_cast<DecisionCapture *>(pointer);
    for (size_t index = 0; index < capture.lanes; ++index) {
        if (decision.job != &(*capture.jobs)[index]) continue;
        auto & lane = (*capture.observations)[index];
        if (lane.decision_count == lane.decisions.size()) {
            lane.decision_overflow = true;
            return;
        }
        lane.decisions[lane.decision_count++] = decision;
        return;
    }
    capture.unknown_job = true;
}

void observer_self_test() {
    sampler_self_test();
    recovery_self_test();
    buffered_self_test();
    const auto parse_mode = [](const char * json) {
        JsonDoc doc(yyjson_read(json, std::strlen(json), 0), yyjson_doc_free);
        if (!doc) throw std::runtime_error("invalid self-test JSON");
        return custom_voice_conditioning(yyjson_doc_get_root(doc.get()));
    };
    if (parse_mode("{\"reference_text_file\":\"r.txt\",\"speaker_f32\":\"s.f32\",\"reference_i32\":\"r.i32\",\"reference_frames\":1}") ||
        !parse_mode("{\"speaker\":\"ryan\"}")) {
        throw std::runtime_error("conditioning mode compatibility failed");
    }
    const char * invalid_modes[] = {"{\"speaker\":\"\"}", "{\"speaker\":null}", "{\"speaker\":\"ryan\",\"speaker_f32\":\"s.f32\"}",
        "{\"speaker\":\"ryan\",\"reference_text_file\":\"r.txt\"}", "{\"speaker\":\"ryan\",\"reference_i32\":\"r.i32\"}",
        "{\"speaker\":\"ryan\",\"reference_frames\":1}"};
    for (const char * json : invalid_modes) {
        bool rejected = false;
        try { (void) parse_mode(json); } catch (const std::runtime_error &) { rejected = true; }
        if (!rejected) throw std::runtime_error("mixed/invalid conditioning was accepted");
    }
    std::array<Lane, 2> observations;
    std::array<TtsJob, 2> jobs{};
    DecisionCapture capture{&observations, &jobs, 2};
    const auto start = Clock::time_point{};
    for (int frame = 0; frame <= 30; ++frame) {
        const bool eos = frame == 30;
        if (tts_research_wants_decision(frame, eos)) {
            on_decision({start + std::chrono::milliseconds(frame), &jobs[1],
                         7, 1, frame, eos ? 2048 : 10, 10, false, eos}, &capture);
        }
    }
    const auto & late = observations[1];
    if (late.decision_count != TTS_RESEARCH_MAX_DECISIONS || late.decision_overflow ||
        late.decisions[7].frame != 7 || late.decisions[8].frame != 30 ||
        !late.decisions[8].eos || late.decisions[8].job != &jobs[1] ||
        late.decisions[8].selected_at - start != std::chrono::milliseconds(30) ||
        observations[0].decision_count != 0 || !late.pcm.empty() || !late.chunks.empty()) {
        throw std::runtime_error("bounded observer/routing contract failed");
    }
    for (int frame = 0; frame <= 3; ++frame) {
        on_decision({start, &jobs[0], 9, 0, frame, frame == 3 ? 2048 : 5,
                     5, frame < 2, frame == 3}, &capture);
    }
    if (observations[0].decision_count != 4 || !observations[0].decisions[3].eos ||
        !observations[0].decisions[0].forced || observations[0].decision_overflow ||
        capture.unknown_job || tts_research_wants_decision(-1, true) ||
        tts_research_wants_decision(8, false)) {
        throw std::runtime_error("early EOS/forced-token observer contract failed");
    }
    on_decision(late.decisions[8], &capture);
    TtsJob unknown{};
    on_decision({start, &unknown, 0, 0, 0, 0, 0, false, false}, &capture);
    if (!observations[1].decision_overflow || !capture.unknown_job ||
        observations[1].decision_count != TTS_RESEARCH_MAX_DECISIONS) {
        throw std::runtime_error("observer overflow/unknown-job detection failed");
    }
    std::fprintf(stderr, "research observer/sampler contracts passed (no model/backend loaded)\n");
}
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

#ifdef QWEN_RESEARCH_PREFILL
void write_prefill_observation(std::ostream & results, const ResearchPrefillObservation & observation,
                               const ResearchPrefillSettings & settings, const std::filesystem::path & destination,
                               const std::string & stem) {
    const auto filename = stem + "-prefill-logits.f32";
    std::ofstream logits(destination / filename, std::ios::binary);
    logits.write(reinterpret_cast<const char *>(observation.logits_last.data()),
                 static_cast<std::streamsize>(observation.logits_last.size() * sizeof(float)));
    if (!logits) throw std::runtime_error("prefill logits write failure");
    size_t nonfinite = 0;
    for (float value : observation.logits_last) if (!std::isfinite(value)) ++nonfinite;
    results << "{\"T\":" << observation.T << ",\"n_past\":" << observation.n_past
        << ",\"slot\":" << observation.slot << ",\"window\":" << observation.n_kv_pad
        << ",\"alignment\":" << settings.window << ",\"kv_f16\":" << (settings.kv_f16 ? "true" : "false")
        << ",\"last_only\":" << (observation.last_only ? "true" : "false")
        << ",\"retain\":" << (settings.retain ? "true" : "false")
        << ",\"retained_hit\":" << (observation.retained_hit ? "true" : "false")
        << ",\"diagnostic\":" << (observation.diagnostic ? "true" : "false")
        << ",\"diagnostic_serial\":" << observation.diagnostic_serial
        << ",\"diagnostic_layer\":" << observation.diagnostic_layer
        << ",\"prefill_use_flash_attn\":" << (observation.prefill_use_flash_attn ? "true" : "false")
        << ",\"graph_nodes\":" << observation.graph_nodes << ",\"scheduler_splits\":" << observation.scheduler_splits
        << ",\"setup_ms\":" << observation.setup_ms << ",\"graph_build_ms\":" << observation.graph_build_ms
        << ",\"schedule_alloc_ms\":" << observation.schedule_alloc_ms << ",\"upload_ms\":" << observation.upload_ms
        << ",\"compute_wait_ms\":" << observation.compute_wait_ms << ",\"readback_ms\":" << observation.readback_ms
        << ",\"total_ms\":" << observation.total_ms << ",\"logits_f32\":\"" << filename
        << "\",\"logits_count\":" << observation.logits_last.size() << ",\"nonfinite_logits\":" << nonfinite
        << ",\"prefix_rows\":" << observation.prefix_rows << ",\"prompt_rows\":" << observation.prompt_rows
        << ",\"common_rows_with_first\":" << observation.common_rows_with_first
        << ",\"prefix_hit\":" << (observation.prefix_hit ? "true" : "false")
        << ",\"prefix_source\":\"" << research_prefix_source_name(observation.prefix_source) << '"'
        << ",\"prefix_setup_ms\":" << observation.prefix_setup_ms
        << ",\"prefix_snapshot_ms\":" << observation.prefix_snapshot_ms
        << ",\"prefix_restore_ms\":" << observation.prefix_restore_ms
        << ",\"prefix_total_ms\":" << observation.prefix_total_ms << '}';
}
#endif
}

int main(int argc, char ** argv) {
    try {
        if (argc == 2 && std::strcmp(argv[1], "--self-test") == 0) {
            observer_self_test();
            return 0;
        }
        if (argc != 3) throw std::runtime_error("usage: eos-research-probe manifest.json NEW-output-dir");
        const auto manifest = read_bytes(argv[1], 1024 * 1024);
        JsonDoc doc(yyjson_read(manifest.data(), manifest.size(), 0), yyjson_doc_free);
        if (!doc) throw std::runtime_error("invalid JSON manifest");
        yyjson_val * root = yyjson_doc_get_root(doc.get());
        const auto talker = string_field(root, "talker");
        const auto codec = string_field(root, "codec");
        const bool custom_voice = custom_voice_conditioning(root);
        const bool icl_reference = optional_boolean(root, "icl_reference", true);
        if (custom_voice && !icl_reference) throw std::runtime_error("icl_reference applies only to Base conditioning");
        const auto speaker_name = custom_voice ? string_field(root, "speaker") : std::string{};
        std::string reference_text;
        std::vector<float> speaker;
        std::vector<int32_t> codes;
        int ref_frames = 0;
        if (!custom_voice) {
            reference_text = read_bytes(string_field(root, "reference_text_file"), 65536);
            if (reference_text.empty() || reference_text.find('\0') != std::string::npos) {
                throw std::runtime_error("empty/invalid reference text");
            }
            speaker = raw_values<float>(string_field(root, "speaker_f32"), 4096);
            codes = raw_values<int32_t>(string_field(root, "reference_i32"), 16 * 4096);
            for (float value : speaker) if (!std::isfinite(value)) throw std::runtime_error("nonfinite reference");
            for (int32_t value : codes) if (value < 0 || value >= 2048) throw std::runtime_error("invalid reference code");
            ref_frames = integer_field(root, "reference_frames", 1, 4096);
            if (codes.size() != static_cast<size_t>(ref_frames) * 16) throw std::runtime_error("reference shape mismatch");
        }
        const int max_batch = integer_field(root, "max_batch", 1, 2);
        const int max_tokens = integer_field(root, "max_tokens", 4, 2048);
        const int deadline_ms = integer_field(root, "deadline_ms", 1, 120000);
        const int max_chunk = integer_field(root, "stream_max_chunk_frames", 1, 8);
        if (max_chunk != 1 && max_chunk != 2 && max_chunk != 4 && max_chunk != 8) throw std::runtime_error("invalid chunk cap");
        const bool observe_decisions = optional_boolean(root, "observe_decisions", true);
        const bool observe_prefill = optional_boolean(root, "observe_prefill", false);
        if (observe_prefill) {
#ifdef QWEN_RESEARCH_PREFILL
            ResearchPrefillSettings settings;
            std::string settings_error;
            if (!research_prefill_from_env(settings, settings_error)) throw std::runtime_error(settings_error);
            if (!settings.profile) throw std::runtime_error("observe_prefill requires QWEN_PREFILL_PROFILE=1");
#else
            throw std::runtime_error("observe_prefill requires QWEN_RESEARCH_PREFILL build");
#endif
        }
        yyjson_val * groups = yyjson_obj_get(root, "groups");
        if (yyjson_obj_get(root, "sampler_control")) throw std::runtime_error("sampler_control belongs to each group");
        // Parse every group's control before any backend/model initialization.
        const auto sampler_controls = group_sampler_controls(groups);
        const auto recovery_configs = group_recovery_configs(groups, max_batch);
        // Exclusive destination: preserve failed or partial experiments.
        const std::filesystem::path destination(argv[2]);
        if (!std::filesystem::create_directory(destination)) throw std::runtime_error("destination exists");
        std::ofstream results(destination / "results.jsonl", std::ios::binary);
        std::ofstream errors(destination / "errors.log", std::ios::binary);
        if (!results || !errors) throw std::runtime_error("cannot open results/errors");
        results << std::setprecision(12);
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
        if (owner.pt.num_code_groups != 16 ||
            (custom_voice ? owner.pt.model_type != "custom_voice" :
             owner.pt.model_type != "base" || speaker.size() != static_cast<size_t>(owner.pt.talker.hidden_size))) {
            throw std::runtime_error("conditioning/model mismatch");
        }
        owner.engine = tts_engine_new(&owner.pt, &owner.tokenizer);
        if (!owner.engine) throw std::runtime_error("cannot create engine");
        bool failed_replay = false;
        for (size_t group_index = 0; group_index < yyjson_arr_size(groups); ++group_index) {
            yyjson_val * group = yyjson_arr_get(groups, group_index);
            const auto & recovery = recovery_configs[group_index];
            RecoveryObservation recovery_observation;
            recovery_observation.enabled = recovery.policy != RecoveryPolicy::None;
            if (!tts_engine_research_capture_codes(owner.engine, recovery_observation.enabled)) {
                throw std::runtime_error("cannot configure research code capture while active");
            }
            const bool diagnostic_dump = optional_boolean(group, "diagnostic_dump", false);
            const auto diagnostic_path = diagnostic_dump ?
                (destination / ("group-" + std::to_string(group_index) + "-dump")).u8string() : std::string{};
            if (diagnostic_dump && !std::filesystem::create_directory(std::filesystem::u8path(diagnostic_path))) {
                throw std::runtime_error("diagnostic destination exists");
            }
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
            DecisionCapture capture{&observations, &jobs, lanes};
            tts_engine_set_research_observer(owner.engine, observe_decisions ? on_decision : nullptr, &capture);
            const auto start = Clock::now();
            for (size_t lane = 0; lane < lanes; ++lane) {
                yyjson_val * seed = yyjson_arr_get(seeds, lane);
                if (!yyjson_is_int(seed) || yyjson_get_sint(seed) < 0) throw std::runtime_error("explicit nonnegative seed required");
                observations[lane].start = start;
                observations[lane].deadline = start + std::chrono::milliseconds(deadline_ms);
                observations[lane].pcm.reserve(static_cast<size_t>(max_tokens) * 1920);
                auto & p = params[lane];
                qt_tts_default_params_ex(&p, sizeof(p));
                apply_sampler_control(p, sampler_controls[group_index]);
                p.text = text.c_str(); p.lang = language.c_str();
                if (custom_voice) {
                    p.speaker = speaker_name.c_str();
                } else {
                    p.ref_spk_emb = speaker.data(); p.ref_spk_dim = static_cast<int>(speaker.size());
                    if (icl_reference) {
                        p.ref_text = reference_text.c_str();
                        p.ref_codes = codes.data(); p.ref_T = ref_frames;
                    }
                }
                if (diagnostic_dump) p.dump_dir = diagnostic_path.c_str();
                p.seed = yyjson_get_sint(seed); p.max_new_tokens = max_tokens;
                p.on_chunk = recovery_observation.enabled ? nullptr : on_audio;
                p.on_chunk_user_data = &observations[lane];
                p.cancel = cancelled; p.cancel_user_data = &observations[lane];
                jobs[lane].params = &p; jobs[lane].resolved_seed = p.seed; jobs[lane].out = &audio[lane];
                observations[lane].admission_started_ms = elapsed(start);
                if (!tts_engine_admit(owner.engine, &jobs[lane])) throw std::runtime_error("admission failed: " + jobs[lane].error);
                observations[lane].admission_completed_ms = elapsed(start);
#ifdef QWEN_RESEARCH_PREFILL
                if (observe_prefill && !pipeline_tts_research_prefill_observe(&owner.pt, &observations[lane].prefill)) {
                    throw std::runtime_error("requested prefill observation is unavailable");
                }
#endif
            }
            if (tts_engine_active(owner.engine) != static_cast<int>(lanes)) throw std::runtime_error("not co-admitted");
            const double admitted_ms = elapsed(start);
            std::vector<TtsJob *> retired;
            std::vector<Step> steps;
            std::vector<SnapshotOwner> snapshots;
            if (recovery_observation.enabled) {
                for (int frame = 0; frame <= 8; ++frame) {
                    snapshots.emplace_back(nullptr, tts_engine_research_snapshot_free);
                }
            }
            while (tts_engine_active(owner.engine)) {
                if (recovery_observation.enabled && steps.size() <= 8) {
                    SnapshotOwner snapshot(tts_engine_research_snapshot(owner.engine), tts_engine_research_snapshot_free);
                    if (!snapshot) throw std::runtime_error("bounded AR snapshot unavailable");
                    const auto info = tts_engine_research_snapshot_info(snapshot.get());
                    if (info.frame != static_cast<int>(steps.size())) throw std::runtime_error("snapshot frame mismatch");
                    recovery_observation.snapshot_ms += info.capture_ms;
                    recovery_observation.peak_snapshot_bytes += info.bytes;
                    snapshots[static_cast<size_t>(info.frame)] = std::move(snapshot);
                }
                tts_engine_step(owner.engine, &retired);
                steps.push_back({elapsed(start), tts_engine_active(owner.engine), retired.size()});
            }
            if (recovery_observation.enabled) {
                if (retired.size() != 1) throw std::runtime_error("original buffered terminal mismatch");
                const auto stem = "group-" + std::to_string(group_index) + "-lane-0";
                auto original_pcm = buffered_pcm(audio[0]);
                auto * captured_codes = tts_engine_research_codes(owner.engine, &jobs[0]);
                if (!captured_codes) throw std::runtime_error("missing original complete codes");
                const auto original_codes = *captured_codes;
                write_raw(destination / (stem + "-original.f32"), original_pcm);
                write_raw(destination / (stem + "-original.i32"), original_codes);
                recovery_observation.original_status = jobs[0].status;
                recovery_observation.original_frames = jobs[0].n_frames;
                recovery_observation.original_finish_reason = jobs[0].finish_reason;
                recovery_observation.original_completed_ms = elapsed(start);
                recovery_observation.original_decisions.assign(observations[0].decisions.begin(),
                    observations[0].decisions.begin() + observations[0].decision_count);
                if (jobs[0].status == QT_STATUS_OK &&
                    (recovery.policy == RecoveryPolicy::Replay ||
                     (jobs[0].finish_reason == QT_FINISH_EOS && jobs[0].n_frames <= recovery.trigger_max_frames))) {
                    const int rewind = recovery.policy == RecoveryPolicy::Rollback1 ? 1 :
                                       recovery.policy == RecoveryPolicy::Rollback2 ? 2 : 0;
                    const int target = recovery.policy == RecoveryPolicy::Replay ? recovery.snapshot_frame :
                                       jobs[0].n_frames - rewind;
                    if (target < 0 || target > 8 || !snapshots[static_cast<size_t>(target)]) {
                        throw std::runtime_error("requested rollback/replay boundary unavailable");
                    }
                    const int64_t seed = recovery.policy == RecoveryPolicy::Replay ? params[0].seed : recovery.continuation_seed;
                    qt_audio_free(&audio[0]);
                    if (!tts_engine_research_restore(owner.engine, snapshots[static_cast<size_t>(target)].get(),
                                                      seed, &recovery_observation.restore_ms)) {
                        throw std::runtime_error("bounded AR restoration failed");
                    }
                    recovery_observation.triggered = true;
                    recovery_observation.restore_frame = target;
                    recovery_observation.continuation_seed = seed;
                    observations[0].decision_count = 0;
                    observations[0].decision_overflow = false;
                    retired.clear();
                    while (tts_engine_active(owner.engine)) {
                        tts_engine_step(owner.engine, &retired);
                        steps.push_back({elapsed(start), tts_engine_active(owner.engine), retired.size(), 1});
                    }
                    auto * final_codes = tts_engine_research_codes(owner.engine, &jobs[0]);
                    if (!final_codes) throw std::runtime_error("missing restored complete codes");
                    const auto final_pcm = buffered_pcm(audio[0]);
                    recovery_observation.replay_codes_equal = *final_codes == original_codes;
                    recovery_observation.replay_pcm_equal = final_pcm.size() == original_pcm.size() &&
                        (final_pcm.empty() || std::memcmp(final_pcm.data(), original_pcm.data(), final_pcm.size() * sizeof(float)) == 0);
                    if (recovery.policy == RecoveryPolicy::Replay &&
                        (!recovery_observation.replay_codes_equal || !recovery_observation.replay_pcm_equal)) {
                        errors << "group=" << group_index << " unchanged replay diverged\n";
                        errors.flush();
                        failed_replay = true;
                    }
                }
                observations[0].pcm = buffered_pcm(audio[0]);
                const auto * final_codes = tts_engine_research_codes(owner.engine, &jobs[0]);
                if (!final_codes) throw std::runtime_error("missing final complete codes");
                write_raw(destination / (stem + "-codes.i32"), *final_codes);
            }
            const double group_total_ms = elapsed(start);
            tts_engine_set_research_observer(owner.engine, nullptr, nullptr);
            if (retired.size() != lanes) throw std::runtime_error("terminal count mismatch");
            for (size_t lane = 0; lane < lanes; ++lane) {
                const auto & observation = observations[lane];
                const auto & job = jobs[lane];
                if (!job.error.empty()) {
                    errors << "group=" << group_index << " lane=" << lane << " error=" << job.error << '\n';
                    errors.flush();
                }
                const auto stem = "group-" + std::to_string(group_index) + "-lane-" + std::to_string(lane);
                const auto filename = stem + ".f32";
                std::ofstream pcm(destination / filename, std::ios::binary);
                pcm.write(reinterpret_cast<const char *>(observation.pcm.data()),
                          static_cast<std::streamsize>(observation.pcm.size() * sizeof(float)));
                if (!pcm) throw std::runtime_error("PCM write failure");
                results << "{\"group\":" << group_index << ",\"lane\":" << lane << ",\"seed\":" << params[lane].seed
                    << ",\"status\":" << job.status << ",\"finish_reason\":" << job.finish_reason
                    << ",\"frames\":" << job.n_frames << ",\"samples\":" << observation.pcm.size()
                    << ",\"admitted_together\":" << lanes << ",\"admitted_ms\":" << admitted_ms
                    << ",\"admission_started_ms\":" << observation.admission_started_ms
                    << ",\"admission_completed_ms\":" << observation.admission_completed_ms
                    << ",\"prefill_ms\":" << job.prefill_ms << ",\"total_ms\":" << job.total_ms
                    << ",\"group_total_ms\":" << group_total_ms
                    << ",\"decision_observation_enabled\":" << (observe_decisions ? "true" : "false")
                    << ",\"decision_capture_invalid\":" << (capture.unknown_job || observation.decision_overflow ? "true" : "false")
                    << ",\"invalid_pcm\":" << (observation.invalid_pcm ? "true" : "false")
                    << ",\"capacity_exceeded\":" << (observation.capacity_exceeded ? "true" : "false")
                    << ",\"conditioning\":\""
                    << (custom_voice ? "custom_voice" : icl_reference ? "base_icl" : "base_speaker_only") << '"'
                    << ",\"execution_mode\":\"" << (recovery_observation.enabled ? "buffered-uncommitted" : "streaming") << '"'
                    << ",\"research_recovery\":";
                write_recovery_observation(results, recovery, recovery_observation, start);
                results << ",\"prefill_observation_enabled\":" << (observe_prefill ? "true" : "false")
                    << ",\"prefill_observation\":";
#ifdef QWEN_RESEARCH_PREFILL
                if (observe_prefill) {
                    write_prefill_observation(results, observation.prefill, owner.pt.research_prefill_settings,
                                              destination, stem);
                } else
#endif
                {
                    results << "null";
                }
                results << ",\"sampler_control\":\"" << sampler_control_name(sampler_controls[group_index])
                        << "\",\"effective_sampler\":";
                write_effective_sampler(results, params[lane], owner.pt.gen_defaults.subtalker_top_k);
                results << ",\"decisions\":[";
                for (size_t index = 0; index < observation.decision_count; ++index) {
                    if (index) results << ',';
                    const auto & decision = observation.decisions[index];
                    const double decision_ms =
                        std::chrono::duration<double, std::milli>(decision.selected_at - start).count();
                    results << "{\"ms\":" << decision_ms << ",\"frame\":" << decision.frame
                            << ",\"token\":" << decision.token << ",\"sampled_token\":" << decision.sampled_token
                            << ",\"forced\":" << (decision.forced ? "true" : "false")
                            << ",\"eos\":" << (decision.eos ? "true" : "false")
                            << ",\"serial\":" << decision.serial << ",\"slot\":" << decision.slot << '}';
                }
                results << "],\"chunks\":[";
                for (size_t index = 0; index < observation.chunks.size(); ++index) {
                    if (index) results << ',';
                    results << "{\"ms\":" << observation.chunks[index].ms << ",\"samples\":" << observation.chunks[index].samples << '}';
                }
                results << "],\"steps\":[";
                for (size_t index = 0; index < steps.size(); ++index) {
                    if (index) results << ',';
                    results << "{\"ms\":" << steps[index].ms << ",\"active\":" << steps[index].active
                            << ",\"retired\":" << steps[index].retired << ",\"attempt\":" << steps[index].attempt << '}';
                }
                results << "]}\n";
                results.flush();
                if (!results) throw std::runtime_error("result write failure");
                qt_audio_free(&audio[lane]);
            }
        }
        return failed_replay ? 2 : 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "research probe: %s\n", error.what());
        return 1;
    }
}

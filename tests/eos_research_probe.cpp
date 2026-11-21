// Bounded engine-internal experiment, not the QTB scheduler or a player.
#include "pipeline-tts.h"
#include "bpe.h"
#include "yyjson.h"

#include <array>
#include <chrono>
#include <cmath>
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

enum class SamplerControl { Default, TalkerTop1, PredictorGreedy };

SamplerControl sampler_control(yyjson_val * group) {
    if (!yyjson_is_obj(group)) throw std::runtime_error("group must be an object");
    if (!yyjson_obj_get(group, "sampler_control")) return SamplerControl::Default;
    const auto name = string_field(group, "sampler_control");
    if (name == "default") return SamplerControl::Default;
    if (name == "talker-top1") return SamplerControl::TalkerTop1;
    if (name == "predictor-greedy") return SamplerControl::PredictorGreedy;
    throw std::runtime_error("sampler_control must be default, talker-top1 or predictor-greedy");
}

const char * sampler_control_name(SamplerControl control) {
    switch (control) {
        case SamplerControl::Default: return "default";
        case SamplerControl::TalkerTop1: return "talker-top1";
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
    if (control == SamplerControl::TalkerTop1) {
        // Keep the stochastic path's repetition processing, unlike native greedy.
        params.do_sample = true;
        params.top_k = 1;
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
                                "{\"sampler_control\":\"predictor-greedy\"}]");
    if (controls != std::vector<SamplerControl>{SamplerControl::Default, SamplerControl::Default,
                                               SamplerControl::TalkerTop1, SamplerControl::PredictorGreedy}) {
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
        if (!params.do_sample || params.top_k != (control == SamplerControl::TalkerTop1 ? 1 : 50)
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
struct Step { double ms; int active; size_t retired; };
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
        for (size_t group_index = 0; group_index < yyjson_arr_size(groups); ++group_index) {
            yyjson_val * group = yyjson_arr_get(groups, group_index);
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
                p.on_chunk = on_audio; p.on_chunk_user_data = &observations[lane];
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
            while (tts_engine_active(owner.engine)) {
                tts_engine_step(owner.engine, &retired);
                steps.push_back({elapsed(start), tts_engine_active(owner.engine), retired.size()});
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
                    << ",\"prefill_observation_enabled\":" << (observe_prefill ? "true" : "false")
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

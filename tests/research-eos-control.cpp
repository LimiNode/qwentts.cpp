#include "research_eos_control.h"

#include <cstdio>
#include <cstdlib>

static int checks = 0;
static void check(bool condition, const char * name) {
    ++checks;
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", name);
        std::exit(1);
    }
}

int main() {
    ResearchEosControl control;
    for (const char * valid : {"", "baseline", "penalty:2", "penalty:4", "penalty:6", "alternative-once:8"}) {
        check(research_eos_parse(valid, &control), "valid policy");
    }
    for (const char * invalid : {"penalty:0", "penalty:3", "penalty:nan", "penalty:inf",
                                "penalty:-2", "penalty:2x", "alternative-once:9", " baseline"}) {
        check(!research_eos_parse(invalid, &control), "reject unsupported policy");
    }
    check(research_eos_parse(nullptr, &control), "unset policy is baseline");
    for (float temperature : {0.0f, 0.9f}) {
        for (int top_k : {1, 3}) {
            for (float top_p : {0.7f, 1.0f}) {
                const int32_t history[] = {0, 0};
                float baseline[] = {1.0f, -0.5f, 2.0f};
                float observed[] = {1.0f, -0.5f, 2.0f};
                float expected_u, actual_u;
                const int expected = sample_top_k_p(baseline, 3, temperature, top_k,
                    top_p, 1.05f, history, 2, 64, 17, &expected_u);
                control = {};
                const auto result = research_eos_sample(control, observed, 3, 2, 2,
                    temperature, top_k, top_p, 1.05f, history, 2, 64, 17, &actual_u);
                check(result.initial == expected && result.effective == expected && !result.intervened,
                    "baseline selected token unchanged");
                check(std::memcmp(baseline, observed, sizeof(baseline)) == 0 && expected_u == actual_u,
                    "baseline weights and uniform unchanged");
            }
        }
    }
    for (const char * policy : {"penalty:2", "penalty:4", "penalty:6"}) {
        research_eos_parse(policy, &control);
        const int32_t history[] = {0};
        float logits[] = {1.0f, -2.0f, 3.0f};
        float expected[] = {1.0f, -2.0f, 3.0f - control.penalty};
        float u;
        const int expected_token = sample_top_k_p(expected, 3, 0.5f, 1, 0.8f,
            2.0f, history, 1, 64, 0, &u);
        const auto result = research_eos_sample(control, logits, 3, 2, 2, 0.5f,
            1, 0.8f, 2.0f, history, 1, 64, 0, &u);
        check(result.effective == expected_token && std::memcmp(logits, expected, sizeof(logits)) == 0,
            "raw penalty precedes temperature/top-k/top-p");
    }
    research_eos_parse("penalty:6", &control);
    float masked[] = {1.0f, -INFINITY};
    std::vector<float> raw;
    research_eos_sample(control, masked, 2, 1, 0, 0.9f, 2, 1.0f, 1.0f,
        nullptr, 0, 64, 0, nullptr, &raw);
    check(std::isinf(raw[1]) && raw[1] < 0, "minimum/reserved mask survives penalty");

    for (float temperature : {0.0f, 0.9f}) {
        research_eos_parse("alternative-once:8", &control);
        float logits[] = {1.0f, 2.0f, 100.0f};
        float u;
        auto result = research_eos_sample(control, logits, 3, 2, 2, temperature, 1,
            0.8f, 1.0f, nullptr, 0, 64, 4, &u, &raw);
        check(result.initial == 2 && result.effective == 1 && result.intervened && control.alternative_used,
            "alternative reruns top-k after removing EOS");
        check(std::isinf(raw[2]) && raw[2] < 0, "effective raw reflects one-shot mask");
        float next[] = {1.0f, 2.0f, 100.0f};
        result = research_eos_sample(control, next, 3, 2, 3, temperature, 1, 1.0f,
            1.0f, nullptr, 0, 64, 5, &u);
        check(result.effective == 2 && !result.intervened, "EOS immediately available after intervention");
    }
    for (int frame : {0, 1, 8, 20}) {
        research_eos_parse("alternative-once:8", &control);
        float logits[] = {0.0f, 100.0f};
        const auto result = research_eos_sample(control, logits, 2, 1, frame, 0.9f,
            1, 1.0f, 1.0f, nullptr, 0, 64, 0, nullptr);
        check(result.effective == 1 && !result.intervened, "outside window unchanged");
    }
    research_eos_parse("alternative-once:8", &control);
    float empty_alternative[] = {-INFINITY, 1.0f};
    const auto no_support = research_eos_sample(control, empty_alternative, 2, 1, 2,
        0.9f, 1, 1.0f, 1.0f, nullptr, 0, 64, 0, nullptr);
    check(no_support.effective == 1 && !control.alternative_used, "no finite alternative preserves EOS");

    research_eos_parse("alternative-once:8", &control);
    ResearchEosControl second = control;
    float first_logits[] = {0.0f, 100.0f};
    research_eos_sample(control, first_logits, 2, 1, 2, 0.9f, 1, 1.0f, 1.0f,
        nullptr, 0, 64, 0, nullptr);
    check(!second.alternative_used, "one-shot state belongs to request");

    research_eos_parse("alternative-once:8", &control);
    const int32_t history[] = {0, 1, 0};
    float alternative[] = {1.2f, -0.3f, 0.4f, 100.0f};
    float expected[] = {1.2f, -0.3f, 0.4f, -INFINITY};
    float expected_u, actual_u;
    philox_uniform_fill(2027, 17, 0u, &expected_u, 1);
    const int expected_token = sample_top_k_p_with_uniform(expected, 4, 0.9f, 3,
        0.8f, 1.5f, history, 3, expected_u);
    const auto replay = research_eos_sample(control, alternative, 4, 3, 2, 0.9f,
        3, 0.8f, 1.5f, history, 3, 2027, 17, &actual_u);
    check(replay.intervened && replay.effective == expected_token && expected_u == actual_u,
        "alternative uses identical draw/subsequence");
    check(std::memcmp(alternative, expected, sizeof(expected)) == 0,
        "alternative restores raw logits before applying repetition/nucleus once");

    research_eos_parse("alternative-once:8", &control);
    float healthy[] = {100.0f, 0.0f};
    const auto unchanged = research_eos_sample(control, healthy, 2, 1, 2, 0.9f,
        1, 1.0f, 1.0f, nullptr, 0, 64, 0, nullptr);
    check(!unchanged.intervened && !control.alternative_used && unchanged.effective == 0,
        "non-EOS decision does not spend alternative");

    research_eos_parse("penalty:4", &control);
    float penalty_input[] = {2.0f, -1.0f, 3.0f};
    research_eos_sample(control, penalty_input, 3, 2, 7, 0.5f, 3, 0.8f, 1.05f,
        history, 3, 2027, 0, nullptr, &raw);
    check(raw[0] == 2.0f && raw[1] == -1.0f && raw[2] == -1.0f,
        "captured penalty input is raw, before repetition and temperature");
    std::printf("PASS: %d EOS research checks\n", checks);
}

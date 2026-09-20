#pragma once
// code-predictor-graph.h: one static predictor graph per flavor and
// batch width, built lazily, allocated once, and replayed with an N
// code id upload per call. Positions, kv rows, and the causal mask
// bake in at build time because the frame cache layout repeats
// identically for every slot: the prefill always writes rows 0..1 and
// step g always appends at row g + 1.

#include "ggml-alloc.h"
#include "ggml.h"

#include <vector>

struct CodePredGraph {
    struct ggml_context * ctx    = nullptr;
    struct ggml_cgraph *  gf     = nullptr;
    ggml_gallocr_t        galloc = nullptr;
    struct ggml_tensor *  logits = nullptr;  // [Vg, N] f32
    // All per-codebook logits, retained only as graph metadata so the
    // opt-in diagnostic path can read them back after one frame replay.
    std::vector<struct ggml_tensor *> logits_steps;
    // Post-norm predictor hidden states, retained only by diagnostic graphs.
    std::vector<struct ggml_tensor *> hidden_steps;
    int                   N      = 0;        // batch width this build covers
};

static void code_predictor_graph_free(CodePredGraph * cp) {
    if (cp->galloc) {
        ggml_gallocr_free(cp->galloc);
        cp->galloc = nullptr;
    }
    if (cp->ctx) {
        ggml_free(cp->ctx);
        cp->ctx = nullptr;
    }
    cp->gf     = nullptr;
    cp->logits = nullptr;
    cp->logits_steps.clear();
    cp->hidden_steps.clear();
    cp->N      = 0;
}

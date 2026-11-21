#pragma once
// Instrumented, timing-ineligible prefill diagnostics. Not a production path.

#include "backend.h"
#include "debug.h"
#include "research_prefill.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

struct ResearchPrefillTap {
    int                  layer;
    const char *         stage;
    struct ggml_tensor * tensor;
};

struct ResearchPrefillLayerCapture {
    int                               layer;
    std::vector<ResearchPrefillTap> * taps;
};

static bool research_prefill_diag_layer(const ResearchPrefillSettings & settings, int layer) {
    return settings.diagnostic_layer < 0 || settings.diagnostic_layer == layer;
}

static void research_prefill_tap(ResearchPrefillLayerCapture * capture,
                                 const char *                  stage,
                                 struct ggml_tensor *          tensor) {
    if (!capture) {
        return;
    }
    // A view's output flag does not protect its backing allocation when the
    // allocator releases the last view. Retain the view and its owner chain
    // until readback. These flags may affect allocation/fusion, so these
    // observations must not be used as latency measurements.
    for (auto * owner = tensor; owner; owner = owner->view_src) {
        ggml_set_output(owner);
    }
    capture->taps->push_back({ capture->layer, stage, tensor });
}

static std::string research_prefill_diag_begin(const ResearchPrefillSettings & settings,
                                               int                             rows,
                                               int                             layers,
                                               int *                           serial) {
    if (!research_prefill_diag_shape_valid(settings, rows, layers, *serial)) {
        throw std::runtime_error("prefill diagnostic requires T_full<=512, <=8 calls and an existing layer");
    }
    const auto root = std::filesystem::u8path(settings.diagnostic_dir);
    std::filesystem::create_directories(root);
    char name[32];
    std::snprintf(name, sizeof(name), "prefill-%04d", *serial + 1);
    const auto path = root / name;
    if (!std::filesystem::create_directory(path)) {
        throw std::runtime_error("prefill diagnostic destination already exists");
    }
    ++*serial;
    return path.u8string();
}

static void research_prefill_diag_dump(const std::string & directory,
                                       const char *        name,
                                       const float *       data,
                                       const int *         shape,
                                       int                 dimensions) {
    size_t count = 1;
    for (int index = 0; index < dimensions; ++index) {
        count *= static_cast<size_t>(shape[index]);
    }
    DebugDumper dumper;
    debug_init(&dumper, directory.c_str());
    debug_dump(&dumper, name, data, shape, dimensions);
    const auto path = std::filesystem::u8path(directory) / (std::string(name) + ".bin");
    if (std::filesystem::file_size(path) !=
        (static_cast<size_t>(dimensions) + 1) * sizeof(int32_t) + count * sizeof(float)) {
        throw std::runtime_error("prefill diagnostic tensor write was incomplete");
    }
}

static void research_prefill_diag_tensor_shape(std::ostream & file, const struct ggml_tensor * tensor) {
    file << "{\"type\":\"" << ggml_type_name(tensor->type) << "\",\"ne\":[";
    for (int dim = 0; dim < 4; ++dim) {
        file << (dim ? "," : "") << tensor->ne[dim];
    }
    file << "],\"nb\":[";
    for (int dim = 0; dim < 4; ++dim) {
        file << (dim ? "," : "") << tensor->nb[dim];
    }
    file << "]}";
}

static void research_prefill_diag_taps(const std::string &                     directory,
                                       int                                     rows,
                                       const std::vector<ResearchPrefillTap> & taps) {
    std::ofstream index(std::filesystem::u8path(directory) / "taps.jsonl", std::ios::binary);
    if (!index) {
        throw std::runtime_error("cannot open prefill diagnostic tap index");
    }
    for (const auto & tap : taps) {
        const auto * tensor = tap.tensor;
        if (tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor) || ggml_nelements(tensor) % rows) {
            throw std::runtime_error("prefill diagnostic tap is not contiguous row-major F32");
        }
        std::vector<float> values(static_cast<size_t>(ggml_nelements(tensor)));
        ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
        char name[96];
        std::snprintf(name, sizeof(name), "layer-%02d-%s", tap.layer, tap.stage);
        const int shape[] = { rows, static_cast<int>(ggml_nelements(tensor) / rows) };
        research_prefill_diag_dump(directory, name, values.data(), shape, 2);
        index << "{\"layer\":" << tap.layer << ",\"stage\":\"" << tap.stage << "\",\"file\":\"" << name
              << ".bin\",\"op\":\"" << ggml_op_name(tensor->op)
              << "\",\"output_retention\":\"view_and_backing_chain\",\"tensor\":";
        research_prefill_diag_tensor_shape(index, tensor);
        for (int source = 0; source < 2; ++source) {
            if (tensor->src[source]) {
                index << ",\"src" << source << "\":";
                research_prefill_diag_tensor_shape(index, tensor->src[source]);
            }
        }
        index << "}\n";
    }
    index.flush();
    if (!index) {
        throw std::runtime_error("prefill diagnostic tap index write failed");
    }
}

// Pack only live prefix rows from the strided [hd, max_seq_len, heads]
// tensor. Keep native bytes as well as the existing Python-readable f32 dump:
// raw files establish whether snapshot/restore is byte-exact, including F16.
static void research_prefill_diag_kv(const std::string &        directory,
                                     const char *               phase,
                                     int                        layer,
                                     const char *               part,
                                     const struct ggml_tensor * tensor,
                                     int                        rows) {
    const size_t element_bytes = ggml_type_size(tensor->type);
    if ((tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16) || rows < 1 || rows > tensor->ne[1] ||
        tensor->nb[0] != element_bytes || tensor->nb[1] != static_cast<size_t>(tensor->ne[0]) * element_bytes) {
        throw std::runtime_error("unsupported prefill diagnostic KV layout");
    }
    const size_t         head_values = static_cast<size_t>(rows) * static_cast<size_t>(tensor->ne[0]);
    std::vector<uint8_t> raw(head_values * static_cast<size_t>(tensor->ne[2]) * element_bytes);
    for (int64_t head = 0; head < tensor->ne[2]; ++head) {
        ggml_backend_tensor_get(tensor, raw.data() + static_cast<size_t>(head) * head_values * element_bytes,
                                static_cast<size_t>(head) * tensor->nb[2], head_values * element_bytes);
    }
    char name[96];
    std::snprintf(name, sizeof(name), "layer-%02d-%s-%s", layer, phase, part);
    const auto    path = std::filesystem::u8path(directory);
    std::ofstream native(path / (std::string(name) + ".raw"), std::ios::binary);
    native.write(reinterpret_cast<const char *>(raw.data()), static_cast<std::streamsize>(raw.size()));
    native.flush();
    if (!native) {
        throw std::runtime_error("prefill diagnostic native KV write failed");
    }
    std::vector<float> values(raw.size() / element_bytes);
    if (tensor->type == GGML_TYPE_F32) {
        std::memcpy(values.data(), raw.data(), raw.size());
    } else {
        for (size_t index = 0; index < values.size(); ++index) {
            ggml_fp16_t value;
            std::memcpy(&value, raw.data() + index * sizeof(value), sizeof(value));
            values[index] = ggml_fp16_to_fp32(value);
        }
    }
    const int shape[] = { static_cast<int>(tensor->ne[2]), rows, static_cast<int>(tensor->ne[0]) };
    research_prefill_diag_dump(directory, name, values.data(), shape, 3);
    std::ofstream metadata(path / (std::string(name) + ".json"), std::ios::binary);
    metadata << "{\"type\":\"" << ggml_type_name(tensor->type) << "\",\"heads\":" << tensor->ne[2]
             << ",\"rows\":" << rows << ",\"head_dim\":" << tensor->ne[0] << ",\"raw_bytes\":" << raw.size()
             << ",\"packed_head_major\":true}\n";
    metadata.flush();
    if (!metadata) {
        throw std::runtime_error("prefill diagnostic KV metadata write failed");
    }
}

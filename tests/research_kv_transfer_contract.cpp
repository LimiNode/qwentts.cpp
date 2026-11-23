// Model-free packing check for the private checkpoint's pitched KV transfers.
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {
void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

struct Allocation {
    ggml_context * context = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ~Allocation() {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (context) ggml_free(context);
    }
};
struct Backend {
    ggml_backend_t value = nullptr;
    ~Backend() { if (value) ggml_backend_free(value); }
};

void check(ggml_backend_t backend, ggml_type type, int heads, int rows) {
    constexpr int dimension = 4;
    constexpr int max_rows = 12;
    Allocation owner;
    const ggml_init_params params{ggml_tensor_overhead() * 6, nullptr, true};
    owner.context = ggml_init(params);
    require(owner.context != nullptr, "tensor context allocation failed");
    auto * source_owner = ggml_new_tensor_4d(owner.context, type, dimension, max_rows, heads, 2);
    auto * old_owner = ggml_dup_tensor(owner.context, source_owner);
    auto * new_owner = ggml_dup_tensor(owner.context, source_owner);
    // Nonzero base-view offset mirrors the actual KV set-view contract;
    // get/set offsets remain zero relative to the selected view's data.
    auto view = [&](ggml_tensor * tensor) {
        return ggml_view_3d(owner.context, tensor, dimension, max_rows, heads,
                            tensor->nb[1], tensor->nb[2], tensor->nb[3]);
    };
    auto * source = view(source_owner);
    auto * old_target = view(old_owner);
    auto * new_target = view(new_owner);
    owner.buffer = ggml_backend_alloc_ctx_tensors(owner.context, backend);
    require(owner.buffer != nullptr, "tensor buffer allocation failed");

    std::vector<uint8_t> input(ggml_nbytes(source_owner));
    std::vector<uint8_t> poison(input.size(), 0xa5);
    for (size_t index = 0; index < input.size(); ++index) input[index] = static_cast<uint8_t>((index * 29 + 7) % 251);
    ggml_backend_tensor_set(source_owner, input.data(), 0, input.size());
    ggml_backend_tensor_set(old_owner, poison.data(), 0, poison.size());
    ggml_backend_tensor_set(new_owner, poison.data(), 0, poison.size());

    const size_t head_bytes = static_cast<size_t>(rows) * source->nb[1];
    std::vector<uint8_t> old_bytes(head_bytes * static_cast<size_t>(heads));
    std::vector<uint8_t> new_bytes(old_bytes.size());
    for (int head = 0; head < heads; ++head) {
        ggml_backend_tensor_get(source, old_bytes.data() + static_cast<size_t>(head) * head_bytes,
                                static_cast<size_t>(head) * source->nb[2], head_bytes);
    }
    ggml_backend_tensor_get_2d(source, new_bytes.data(), 0, head_bytes,
                               static_cast<size_t>(heads), source->nb[2], head_bytes);
    require(old_bytes == new_bytes, "pitched capture differs from per-head capture");

    for (int head = 0; head < heads; ++head) {
        ggml_backend_tensor_set(old_target, old_bytes.data() + static_cast<size_t>(head) * head_bytes,
                                static_cast<size_t>(head) * old_target->nb[2], head_bytes);
    }
    ggml_backend_tensor_set_2d(new_target, new_bytes.data(), 0, head_bytes,
                               static_cast<size_t>(heads), new_target->nb[2], head_bytes);
    std::vector<uint8_t> old_full(input.size()), new_full(input.size());
    ggml_backend_tensor_get(old_owner, old_full.data(), 0, old_full.size());
    ggml_backend_tensor_get(new_owner, new_full.data(), 0, new_full.size());
    require(old_full == new_full, "pitched restore differs from per-head restore");
    for (size_t index = 0; index < input.size(); ++index) {
        const size_t set_offset = new_target->nb[3];
        const bool in_set = index >= set_offset;
        const bool in_prefix = in_set && (index - set_offset) % new_target->nb[2] < head_bytes;
        require(new_full[index] == (in_prefix ? input[index] : poison[index]),
                "restore altered uncaptured tail or another set");
    }
}
}

int main() {
    try {
        Backend backend;
        backend.value = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        require(backend.value != nullptr, "CPU backend unavailable");
        for (const auto type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
            for (const int heads : {1, 3, 8}) {
                for (const int rows : {1, 5, 12}) check(backend.value, type, heads, rows);
            }
        }
        std::fprintf(stderr, "pitched KV capture/restore contracts passed on CPU (18 cases)\n");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "pitched KV contract: %s\n", error.what());
        return 1;
    }
}

#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "umd/device/types/arch.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct ggml_backend_metalium_device_context {
    std::shared_ptr<ttnn::MeshDevice> device = nullptr;
    int device_id = -1;
    std::string name;
    std::string description;
    tt::ARCH arch = tt::ARCH::BLACKHOLE;
};

struct ggml_backend_metalium_context {
    ggml_backend_metalium_device_context * dev_ctx = nullptr;
    std::unordered_map<std::string, tt::tt_metal::Tensor> transposed_weights;
    int device_id = 0;
    std::string name;
};

struct ggml_backend_metalium_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

struct TensorWithMetadata;

struct ggml_backend_metalium_buffer_context {
    size_t ggml_buffer_size_bytes = 0;
    std::string name;
    std::shared_ptr<ttnn::MeshDevice> device = nullptr;
    size_t base_offset = 0;

    // Tracking our own allocations because Metalium limitations and GGML assuming them
    std::vector<std::unique_ptr<TensorWithMetadata>> metadata_to_free;
};

struct TensorWithMetadata {
    std::shared_ptr<tt::tt_metal::Tensor> tensor;
    ggml_type ggtype = GGML_TYPE_COUNT;
    ggml_backend_metalium_buffer_context* bufctx = nullptr;
    std::vector<std::byte> host_shadow;
};

struct ggml_backend_metalium_buffer_type_context {
    ggml_backend_metalium_device_context * device_ctx = nullptr;
    std::string name;
};

struct ggml_backend_metalium_debug_flags {
    bool print_rejected_ops = false;
    bool print_view = false;
    bool cache_mm_transpose = false;
    bool disable_program_cache = false;
};

extern const ggml_backend_metalium_debug_flags ggml_metalium_debug_flags;

ttnn::MeshDevice * ggml_metalium_get_device(ggml_backend_metalium_device_context * dev_ctx);
ttnn::DeviceComputeKernelConfig ggml_metalium_make_compute_kernel_config(ttnn::MeshDevice* device);
tt::tt_metal::DataType ggml_metalium_ggml2tt_type(ggml_type ggtype, tt::ARCH arch);
bool ggml_metalium_is_ggml_type_supported(ggml_type ggtype, tt::ARCH arch);
bool ggml_metalium_numpy_broadcast_rule(const ggml_tensor* t, const ggml_tensor* q);
bool ggml_metalium_is_view(const ggml_tensor* tensor);
bool ggml_metalium_is_simple_unit_slice(const ggml_tensor* view);
std::vector<float> ggml_metalium_tensor_to_float(const ggml_tensor * tensor);
std::shared_ptr<tt::tt_metal::Tensor> ggml_metalium_tensor_from_float(const ggml_tensor * tensor, ggml_backend_metalium_buffer_context * bufctx, const float * data);
tt::tt_metal::Tensor ggml_metalium_reshape_tt_tensor_into_ggml(const tt::tt_metal::Tensor& tensor, const struct ggml_tensor * node);
std::shared_ptr<tt::tt_metal::Tensor> ggml_metalium_realize_ggml_view(const ggml_tensor* tensor);

const char * ggml_backend_metalium_buffer_type_name(ggml_backend_buffer_type_t buft);
ggml_backend_buffer_type_t ggml_backend_metalium_buffer_type(ggml_backend_dev_t dev, ggml_backend_metalium_device_context* dev_ctx);

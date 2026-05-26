#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"

#include "ggml-metalium-context.h"
#include "ggml-metalium-util.h"

#include "tt-metalium/bfloat16.hpp"
#include "tt-metalium/host_buffer.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "umd/device/types/arch.hpp"
#include <sys/types.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <string_view>
#include <type_traits>
#include <vector>

#include <ttnn/core.hpp>
#include <ttnn/device.hpp>
#include <ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/operations/moreh/moreh_matmul/moreh_matmul.hpp>
#include <ttnn/operations/kv_cache/kv_cache.hpp>
#include <ttnn/operations/data_movement/slice/slice.hpp>
#include <ttnn/operations/normalization/layernorm/layernorm.hpp>
#include <ttnn/operations/normalization/rmsnorm/rmsnorm.hpp>
#include <ttnn/operations/data_movement/untilize/untilize.hpp>
#include <ttnn/operations/experimental/transformer/nlp_kv_cache_load_slice/nlp_kv_cache_load_slice.hpp>
#include <ttnn/operations/creation/creation.hpp>
#include <ttnn/operations/eltwise/unary/unary_composite.hpp>
#include <ttnn/operations/data_movement/transpose/transpose.hpp>
#include <ttnn/operations/data_movement/permute/permute.hpp>
#include <ttnn/operations/data_movement/repeat/repeat.hpp>
#include <ttnn/operations/data_movement/concat/concat.hpp>
#include <ttnn/operations/copy/typecast/typecast.hpp>
#include <ttnn/operations/normalization/softmax/softmax.hpp>
#include <ttnn/operations/data_movement/reshape_view/reshape.hpp>
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>

ttnn::MeshDevice * ggml_metalium_get_device(ggml_backend_metalium_device_context * dev_ctx)
{
    if (dev_ctx->device == nullptr) {
        int device_id = dev_ctx->device_id;
        GGML_ASSERT(device_id >= 0 && (size_t)device_id < tt::tt_metal::GetNumAvailableDevices());

        dev_ctx->device = ttnn::open_mesh_device(device_id);
        if (dev_ctx->device == nullptr) {
            GGML_ABORT("failed to open Metalium mesh device %d", device_id);
        }
    }
    return dev_ctx->device.get();
}

ttnn::DeviceComputeKernelConfig ggml_metalium_make_compute_kernel_config(ttnn::MeshDevice* device)
{
    ttnn::DeviceComputeKernelConfig cfg;
    if (device->arch() == tt::ARCH::WORMHOLE_B0 || device->arch() == tt::ARCH::BLACKHOLE) {
        cfg = ttnn::WormholeComputeKernelConfig{
            .math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
            .math_approx_mode = false,
            .fp32_dest_acc_en = true,
            .packer_l1_acc = true
        };
    }
    else {
        fmt::println(stderr,"Unsupported device arch {} in ggml_metalium_make_compute_kernel_config", device->arch());
        abort();
    }
    return cfg;
}

// Debug flags that can be enabled at runtime. Because recompiling the backend takes forever
// this enables faster iteration on debugging. Eventually these should be removed
// NOTE: DO NOT invent more _hack flags. Else it devolves into a mess like what BUDA did
const ggml_backend_metalium_debug_flags ggml_metalium_debug_flags = []() {
    auto func = [](const char* env) -> bool {
        const char* val = std::getenv(env);
        if(val != nullptr) {
            std::string str(val);
            std::transform(str.begin(), str.end(), str.begin(), ::tolower);
            if(str != "0" && str != "false" && str != "no" && str != "off") {
                return true;
            }
        }
        return false;
    };

    return ggml_backend_metalium_debug_flags {
        .print_rejected_ops = func("GGML_METALIUM_PRINT_REJECTED_OPS"),
        .print_view = func("GGML_METALIUM_PRINT_VIEW"),
        .cache_mm_transpose = func("GGML_METALIUM_CACHE_MM_TRANSPOSE"), // GGML uses pre-transposed weights. Remove this flag when TT implements it
        .disable_program_cache = func("GGML_METALIUM_DISABLE_PROGRAM_CACHE")
    };
}();

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Backend internal state tracking because GGML API does not allow
///////////////////////////////////////////////////////////////////////////////////////////////////////

// Maintain all base addresses are unique
// TODO: Do we still need this since we already removed the virtual address mapping hack?
static size_t g_metalium_base_offset = 0;

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Actual backend code
///////////////////////////////////////////////////////////////////////////////////////////////////////

static tt::tt_metal::DataType ggml_metalium_ggml2tt_type_internal(ggml_type ggtype, tt::ARCH arch) {
    // This table is consulted to map GGML types to TT types dueing tensor creation
    if(arch == tt::ARCH::WORMHOLE_B0 || arch == tt::ARCH::BLACKHOLE) {
        static constexpr std::array<tt::tt_metal::DataType, GGML_TYPE_COUNT> table = {
            /*GGML_TYPE_F32 = */ tt::tt_metal::DataType::BFLOAT16,
            /*GGML_TYPE_F16 = */ tt::tt_metal::DataType::BFLOAT16,
            /*GGML_TYPE_Q4_0 = */ tt::tt_metal::DataType::BFLOAT8_B,    // Using BFLOAT8_B for now as BFLOAT4_B is not accurate enough
            /*GGML_TYPE_Q4_1 = */ tt::tt_metal::DataType::BFLOAT8_B,    // Does work but causes issues in unit tests
            tt::tt_metal::DataType::INVALID,
            tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_Q5_0 = */ tt::tt_metal::DataType::BFLOAT8_B,
            /*GGML_TYPE_Q5_1 = */ tt::tt_metal::DataType::BFLOAT8_B,    // Does work but causes issues in unit tests
            /*GGML_TYPE_Q8_0 = */ tt::tt_metal::DataType::BFLOAT8_B,
            /*GGML_TYPE_Q8_1 = */ tt::tt_metal::DataType::BFLOAT8_B,
            /*GGML_TYPE_Q2_K = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_Q3_K = */ tt::tt_metal::DataType::BFLOAT8_B,   // Using BFLOAT8_B for now as BFLOAT4_B is not accurate enough
            /*GGML_TYPE_Q4_K = */ tt::tt_metal::DataType::BFLOAT8_B,   // Using BFLOAT8_B for now as BFLOAT4_B is not accurate enough
            /*GGML_TYPE_Q5_K = */ tt::tt_metal::DataType::BFLOAT8_B,
            /*GGML_TYPE_Q6_K = */ tt::tt_metal::DataType::BFLOAT8_B,
            /*GGML_TYPE_Q8_K = */ tt::tt_metal::DataType::BFLOAT8_B,
            /*GGML_TYPE_IQ2_XXS = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_IQ2_XS = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_IQ3_XXS = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_IQ1_S = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_IQ4_NL = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_IQ3_S = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_IQ2_S = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_IQ4_XS = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_I8 = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_I16 = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_I32 = */ tt::tt_metal::DataType::UINT32, // Yeah not ideal. but don't have support for tilizing int32 on device
            /*GGML_TYPE_I64 = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_F64 = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_IQ1_M = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_BF16 = */ tt::tt_metal::DataType::BFLOAT16,
            /*GGML_TYPE_Q4_0_4_4 = */ tt::tt_metal::DataType::INVALID, // Untested from this point on
            /*GGML_TYPE_Q4_0_4_8 = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_Q4_0_8_8 = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_TQ1_0   = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_TQ2_0   = */ tt::tt_metal::DataType::INVALID,
        };
        tt::tt_metal::DataType type = table[ggtype];
        return type;
    }
    GGML_ASSERT(false && "Unsupported Tenstorrent card architecture");
}

bool ggml_metalium_numpy_broadcast_rule(const ggml_tensor* t, const ggml_tensor* q)
{
    int tdim = ggml_n_dims(t);
    int qdim = ggml_n_dims(q);

    int min_dim = tdim < qdim ? tdim : qdim;
    for(int i = 0; i < min_dim; i++) {
        if(t->ne[i] != q->ne[i] && t->ne[i] != 1 && q->ne[i] != 1) {
            return false;
        }
    }
    return true;
}

tt::tt_metal::DataType ggml_metalium_ggml2tt_type(ggml_type ggtype, tt::ARCH arch)
{
    tt::tt_metal::DataType type = ggml_metalium_ggml2tt_type_internal(ggtype, arch);
    if(type == tt::tt_metal::DataType::INVALID) {
        fmt::println(stderr, "Unsupported data type: {}", ggml_type_name(ggtype));
        GGML_ASSERT(false && "Unsupported data type");
    }
    return type;

}

bool ggml_metalium_is_ggml_type_supported(ggml_type ggtype, tt::ARCH arch) {
    return ggml_metalium_ggml2tt_type_internal(ggtype, arch) != tt::tt_metal::DataType::INVALID;
}

template <typename SrcType, typename DstType>
static tt::tt_metal::HostBuffer ggml_metalium_data_to_borrowed_storage(const SrcType* src, size_t size) {
    // Converts GGML floating point (FP32, FP16, BF16) to TT floating point (FP32, BF16)
    using Src = std::remove_cv_t<std::remove_reference_t<SrcType>>;
    using Dst = std::remove_cv_t<std::remove_reference_t<DstType>>;
    // Convert from  GGML types to TT types
    static_assert(std::is_same_v<Src, float> || std::is_same_v<Src, ggml_bf16_t> || std::is_same_v<Src, ggml_fp16_t> || std::is_same_v<Src, int>);
    static_assert(std::is_same_v<Dst, float> || std::is_same_v<Dst, bfloat16> || std::is_same_v<Dst, uint32_t>);

    auto src_adaptor = [](const SrcType& src) -> float {
        if constexpr(std::is_same_v<Src, ggml_fp16_t>) {
            return ggml_fp16_to_fp32(src);
        }
        else if constexpr(std::is_same_v<Src, ggml_bf16_t>) {
            return ggml_bf16_to_fp32(src);
        }
        else if constexpr(std::is_same_v<Src, float>) {
            return src;
        }
        else if constexpr(std::is_same_v<Src, int>) {
            return static_cast<float>(src);
        }
        GGML_UNREACHABLE();
    };

    auto dst_adaptor = [](DstType& dst, float val) {
        if constexpr(std::is_same_v<Dst, bfloat16>) {
            dst = bfloat16(val);
        }
        else if constexpr(std::is_same_v<Dst, float>) {
            dst = val;
        }
        else if constexpr(std::is_same_v<Dst, int>) {
            dst = static_cast<int>(val);
        }
        else if constexpr(std::is_same_v<Dst, uint32_t>) {
            dst = static_cast<uint32_t>(val);
        }
        else {
            GGML_UNREACHABLE();
        }
    };

    // Optimization: avoid unnecessary initialization and copying like vec<float>(size) as it tanks performance
    Dst* vec = new Dst[size];
    // special case if both GGML and TT types have the same underlying type (e.g. both FP32 or BF16)
    if constexpr(std::is_same_v<Src, Dst> || (std::is_same_v<Src, ggml_bf16_t> && std::is_same_v<Dst, bfloat16>)) {
        static_assert(sizeof(Src) == sizeof(Dst), "Src and Dst must have the same size");
        // Make GCC shut up about writing into a class like it's flat memory
        memcpy((void*)vec, src, size * sizeof(Src));
    }
    // special case for BFP16 (much faster then TTNN's implementation)
    else if constexpr(std::is_same_v<Src, float> && std::is_same_v<Dst, bfloat16>) {
        const auto* trait = ggml_get_type_traits_cpu(GGML_TYPE_BF16);
        assert(trait != nullptr);
        trait->from_float(src, vec, size);
    }
    else {
        for(size_t i = 0; i < size; i++) {
            dst_adaptor(vec[i], src_adaptor(src[i]));
        }
    }

    int* refcount = new int(0);
    tt::tt_metal::MemoryPin pin(
        [refcount]() mutable {
            (*refcount)++;
        },
        [refcount, vec]() mutable {
            assert(refcount != nullptr);
            (*refcount)--;
            if(*refcount == 0) {
                delete refcount;
                delete [] vec;
                refcount = nullptr;
            }
        }
    );
    auto storage = tt::tt_metal::HostBuffer(ttsl::Span<DstType>(vec, size), std::move(pin));

    return storage;
}

template <typename DstType>
static tt::tt_metal::HostBuffer ggml_metalium_quantized_to_owned_storage(const void* src, const ggml_tensor* tensor) {
    const ggml_type_traits* trait = ggml_get_type_traits(tensor->type);
    const size_t size = ggml_nelements(tensor);
    GGML_ASSERT(trait->to_float != NULL);

    std::shared_ptr<float[]> vec(new float[size]);
    trait->to_float(src, vec.get(), size);

    if constexpr(std::is_same_v<DstType, float>) {
        int* refcount = new int(0);
        float* vec_ptr = vec.get();
        tt::tt_metal::MemoryPin pin(
            [refcount]() mutable {
                (*refcount)++;
            },
            [refcount, vec=std::move(vec)]() mutable {
                assert(refcount != nullptr);
                (*refcount)--;
                if(*refcount == 0) {
                    delete refcount;
                    vec.reset();
                }
            }
        );
        return tt::tt_metal::HostBuffer(ttsl::Span<float>(vec_ptr, size), std::move(pin));
    }
    return ggml_metalium_data_to_borrowed_storage<float, DstType>(vec.get(), size);
}

template <typename SrcType>
static void ggml_metalium_tensor_to_ggml(const tt::tt_metal::Tensor& tensor, void* dst, ggml_type dst_ggtype) {
    //std::cout << "RLE: dst_ggtype=" << dst_ggtype << ", tensor=" << tensor.dtype() << std::endl;
    // Converts TT tensors to GGML types
    static_assert(std::is_same_v<SrcType, float> || std::is_same_v<SrcType, bfloat16> || std::is_same_v<SrcType, uint32_t>);

    tt::tt_metal::Tensor row_major_tensor = ttnn::untilize(tensor).cpu();
    ttnn::Shape shape = row_major_tensor.logical_shape();
    ttnn::Shape padded_shape = row_major_tensor.padded_shape();
    GGML_ASSERT(row_major_tensor.storage_type() == tt::tt_metal::StorageType::HOST);
    const tt::tt_metal::HostStorage& storage = row_major_tensor.host_storage();
    const auto buffer = storage.buffer().get_shard({0, 0}).value();
    auto view = buffer.view_as<SrcType>();
    const SrcType* buf = &view[0];
    size_t buf_size = view.size();

    GGML_ASSERT(buf != nullptr);
    void* intermid = nullptr;
    std::vector<std::byte> intermid_buf;
    bool need_quantized_conversion = false;
    bool src_dst_same = false;
    if(dst_ggtype == GGML_TYPE_F32 && !std::is_same_v<SrcType, float>) {
        intermid = dst;
        need_quantized_conversion = false;
        src_dst_same = false;
    }
    // Just putting the integer types here to remind me TT tensors can have integer types
    else if ((std::is_same_v<SrcType, float> && dst_ggtype == GGML_TYPE_F32) ||
             (std::is_same_v<SrcType, bfloat16> && dst_ggtype == GGML_TYPE_BF16) ||
             (std::is_same_v<SrcType, int32_t> && dst_ggtype == GGML_TYPE_I32) ||
             (std::is_same_v<SrcType, uint32_t> && dst_ggtype == GGML_TYPE_I32) ||
             (std::is_same_v<SrcType, int16_t> && dst_ggtype == GGML_TYPE_I16) ||
             (std::is_same_v<SrcType, int8_t> && dst_ggtype == GGML_TYPE_I8)) {
        intermid = dst;
        need_quantized_conversion = false;
        src_dst_same = true;
    }
    else {
        intermid_buf.resize(shape.volume() * sizeof(float));
        intermid = intermid_buf.data();
        need_quantized_conversion = true;
        src_dst_same = false;
    }

    auto src_adaptor = [](const SrcType& src) -> float {
        if constexpr(std::is_same_v<SrcType, bfloat16>) {
            return static_cast<float>(src);
        }
        else if constexpr(std::is_same_v<SrcType, float>) {
            return src;
        }
        else if constexpr(std::is_same_v<SrcType, uint32_t>) {
            return static_cast<float>(src);
        }
        GGML_UNREACHABLE();
    };

    // Tilize to ROW_MAJOR doesn't mean the tensor is contiguous. It still has the underlying 32x32 tiles
    // we need to view into the tensor to get the contiguous data
    std::array<size_t, 4> padded_nshape {1, 1, 1, 1};
    for(size_t i = 0; i < padded_shape.size(); i++) {
        padded_nshape[4 - padded_shape.size() + i] = padded_shape[i];
    }
    std::array<size_t, 4> stride = {
        padded_nshape[1] * padded_nshape[2] * padded_nshape[3],
        padded_nshape[2] * padded_nshape[3],
        padded_nshape[3],
        1
    };

    std::array<size_t, 4> nshape {1, 1, 1, 1};
    for(size_t i = 0; i < shape.size(); i++) {
        nshape[4 - shape.size() + i] = shape[i];
    }
    static_assert(GGML_MAX_DIMS == 4, "Looping depth is hardcoded to 4");

    // Sanity check: src_dst_same shuld indicate there is no need for quantized conversion
    GGML_ASSERT(((src_dst_same && !need_quantized_conversion) || !src_dst_same) && "src and dst should be the same type if src_dst_same is true");
    // Optimization: If the source shape indicates that the tensor is contiguous in memory - memcpy it directly or (since we are converting to float) abuse the pointer
    // NOTE: The following optimizations are not full and has some slow paths taken unoptimally. But good enough for now
    if(nshape[3] % 32 == 0 && ((nshape[0] == 1 && nshape[1] == 1) || nshape[2] % 32 == 0)) {
        const size_t buf_size = std::accumulate(nshape.begin(), nshape.end(), 1, std::multiplies<size_t>());
        if(src_dst_same && !need_quantized_conversion) {
            memcpy(dst, buf, sizeof(SrcType) * buf_size);
            return;
        }
        if(std::is_same_v<SrcType, float> && need_quantized_conversion) {
            // Pointer abuse
            intermid = const_cast<void*>(static_cast<const void*>(buf));
        }
        else {
            for(size_t i = 0; i < buf_size; i++) {
                ((float*)intermid)[i] = src_adaptor(buf[i]);
            }
        }
    }
    // If the 2nd dimension is not divisible by 32, we can still copy block by block
    else if(src_dst_same && !need_quantized_conversion && nshape[0] % 32 == 0 && nshape[1] % 32 != 0) {
        const size_t src_block_size = nshape[2] * nshape[3];
        const size_t src_block_stride = stride[1];
        for(size_t i=0;i<nshape[0]*nshape[1];i++) {
            memcpy((SrcType*)intermid + i * src_block_size, buf + i * src_block_stride, sizeof(SrcType) * src_block_size);
        }
    }
    // If we can do row-by-row copy
    // Only avoid small copies via memcpy if not copying into FP32 - we rely on raw copies for other types as the
    // fallback loop asserts FP32
    else if(src_dst_same && !need_quantized_conversion && (shape[3] >= 4 || !std::is_same_v<SrcType, float>)) {
        const size_t dst_stride = nshape[3];
        for(size_t i = 0; i < nshape[0] * nshape[1]; i++) {
            for(size_t j = 0; j < nshape[2]; j++) {
                // optimization: copy a row of memory at a time
                const size_t src_idx = i * stride[1] + j * stride[2];
                memcpy((SrcType*)intermid + (i * nshape[2] + j) * dst_stride, buf + src_idx, sizeof(SrcType) * nshape[3]);
            }
        }
    }
    // Slow path: src and dst are different types or the data is not contiguous in memory
    else {
        size_t idx = 0;
        for(size_t w = 0; w < nshape[0]; w++) {
            for(size_t z = 0; z < nshape[1]; z++) {
                for(size_t y = 0; y < nshape[2]; y++) {
                    for(size_t x = 0; x < nshape[3]; x++) {
                        const size_t src_idx = w * stride[0] + z * stride[1] + y * stride[2] + x * stride[3];
                        GGML_ASSERT(src_idx < buf_size);
                        ((float*)intermid)[idx] = src_adaptor(buf[src_idx]);
                        idx++;
                    }
                }
            }
        }
    }

    if (need_quantized_conversion) {
        GGML_ASSERT(intermid != nullptr);
        const float* intermid_f32 = (const float*)intermid;
        const int64_t nelements = shape.volume();

        switch (dst_ggtype) {
            case GGML_TYPE_I8:
                for (int64_t i = 0; i < nelements; i++) {
                    ((int8_t*)dst)[i] = (int8_t)intermid_f32[i];
                }
                break;
            case GGML_TYPE_I16:
                for (int64_t i = 0; i < nelements; i++) {
                    ((int16_t*)dst)[i] = (int16_t)intermid_f32[i];
                }
                break;
            case GGML_TYPE_I32:
                for (int64_t i = 0; i < nelements; i++) {
                    ((int32_t*)dst)[i] = (int32_t)intermid_f32[i];
                }
                break;
            case GGML_TYPE_I64:
                for (int64_t i = 0; i < nelements; i++) {
                    ((int64_t*)dst)[i] = (int64_t)intermid_f32[i];
                }
                break;
            case GGML_TYPE_F64:
                for (int64_t i = 0; i < nelements; i++) {
                    ((double*)dst)[i] = (double)intermid_f32[i];
                }
                break;
            case GGML_TYPE_BF16:
                ggml_fp32_to_bf16_row(intermid_f32, (ggml_bf16_t*)dst, nelements);
                break;
            case GGML_TYPE_F16:
            default:
                GGML_ASSERT((ggml_is_quantized(dst_ggtype) || dst_ggtype == GGML_TYPE_F16) && "This block should only reach for quantized data types, FP16, BF16, or numeric conversions");
                {
                    const ggml_type_traits_cpu* trait = ggml_get_type_traits_cpu(dst_ggtype);
                    GGML_ASSERT(trait->from_float != NULL);
                    trait->from_float(intermid_f32, dst, nelements);
                }
                break;
        }
    }
}

bool ggml_metalium_is_view(const ggml_tensor* tensor)
{
    return tensor->view_src != nullptr ||
        tensor->op == GGML_OP_VIEW ||
        tensor->op == GGML_OP_RESHAPE ||
        tensor->op == GGML_OP_TRANSPOSE ||
        tensor->op == GGML_OP_PERMUTE;
}

bool ggml_metalium_is_simple_unit_slice(const ggml_tensor* view)
{
    if(view == nullptr || view->op != GGML_OP_VIEW || view->view_src == nullptr) {
        return false;
    }

    const ggml_tensor* src = view->view_src;

    // TTNN slice(start, end, step=1) can only represent same-stride rectangular subviews.
    for(int i = 0; i < GGML_MAX_DIMS; ++i) {
        if(view->nb[i] != src->nb[i]) {
            return false;
        }
    }

    size_t remaining = view->view_offs;
    for(int i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        const size_t stride = src->nb[i];
        if(stride == 0) {
            return false;
        }

        const int64_t start = remaining / stride;
        remaining %= stride;

        if(start + view->ne[i] > src->ne[i]) {
            return false;
        }
    }

    return remaining == 0;
}

static bool ggml_metalium_tt_tensor_shape_compatible(const ggml_tensor * ggtensor, const tt::tt_metal::Tensor & ttensor)
{
    return ggml_tt_tensors_shape_equal(ggtensor, ttensor);
}

tt::tt_metal::Tensor ggml_metalium_reshape_tt_tensor_into_ggml(const tt::tt_metal::Tensor& tensor, const struct ggml_tensor * node)
{
    if(ggml_tt_tensors_shape_equal(node, tensor)) {
        return tensor;
    }

    std::array<uint32_t, GGML_MAX_DIMS> target_shape;
    for(int i = 0; i < GGML_MAX_DIMS; i++) {
        target_shape[i] = node->ne[GGML_MAX_DIMS - i - 1];
    }

    // std::cerr << "Reshaping tensor " << tensor.logical_shape() << " to " << target_shape << std::endl;
    return ttnn::reshape(tensor, ttnn::Shape(target_shape));
}

static std::shared_ptr<tt::tt_metal::Tensor> ggml_metalium_realize_ggml_view_impl(const ggml_tensor* tensor);
std::shared_ptr<tt::tt_metal::Tensor> ggml_metalium_realize_ggml_view(const ggml_tensor* tensor)
{
    auto res = ggml_metalium_realize_ggml_view_impl(tensor);
    if(!ggml_metalium_tt_tensor_shape_compatible(tensor, *res)) {
        std::cout << "FATAL ERROR: Shape mismatch between TTNN and GGML after view op " << ggml_op_name(tensor->op) << "\n"
            << "  Result: " << res->logical_shape() << "\n"
            << "  GGML expecting: " << tensor->ne[3] << " " << tensor->ne[2] << " " << tensor->ne[1] << " " << tensor->ne[0] << "\n";
        GGML_ASSERT(ggml_metalium_tt_tensor_shape_compatible(tensor, *res));
    }
    return res;
}


static std::shared_ptr<tt::tt_metal::Tensor> ggml_metalium_realize_ggml_view_impl(const ggml_tensor* tensor)
{
    // Since TTNN does not support the traditional view operation, we had to support it ourselves
    // This function, realize, extracts the data from the source tensor and creates a new tensor
    // that is separate from the source tensor. DO NOT eagerly call this function

    ggml_tensor* src0 = tensor->src[0];
    ggml_op op = tensor->op;


    // Do we really need to lazy evaluate this? Currently transpose is eagerly evaluated
    if(op == GGML_OP_TRANSPOSE) {
        auto parent = ggml_metalium_realize_ggml_view(src0);
        auto res = ttnn::transpose(*parent, -2, -1);
        return std::make_shared<tt::tt_metal::Tensor>(res);
    }
    if(op == GGML_OP_VIEW) {

        std::shared_ptr<tt::tt_metal::Tensor> parent = ggml_metalium_realize_ggml_view(tensor->view_src);
        std::array dst_size = std::to_array(tensor->ne);
        std::array dst_stride = std::to_array(tensor->nb);
        std::array src_size = std::to_array(src0->ne);
        std::array src_stride = std::to_array(src0->nb);
        size_t offset = tensor->view_offs;
        // ggml_backend_metalium_buffer_context* bufctx = ((TensorWithMetadata*)tensor->extra)->bufctx;

        // TODO: Generalize this to use permute instead of transpose
        // FIXME: This is failing views in test-backend-ops
        // std::optional<std::pair<uint32_t, uint32_t>> axisswap;
        // for (int i = 0; i < ggml_n_dims(tensor); ++i) {
        //     size_t expected_stride = tensor->nb[0];
        //     for (int j = 0; j < i; ++j) {
        //         expected_stride *= tensor->ne[j];
        //     }
        //     // std::cout << "  Axis " << i << " stride: " << tensor->nb[i] << " expected: " << expected_stride << std::endl;
        //     if (tensor->nb[i] != expected_stride) {
        //         if (!axisswap) {
        //             axisswap = std::make_pair(i, 1000);
        //         } else if (axisswap->second == 1000) {
        //             axisswap->second = i;
        //         } else {
        //             GGML_ASSERT(false && "More than one axis swap detected");
        //         }
        //     }
        // }
        // TODO: Do something with axisswap. I think some ops needs this but it haven't crashed yet

        // Fast path if we can just return the parent tensor (view is a no-op)
        if(dst_size == src_size && dst_stride == src_stride && offset == 0) {
            return parent;
        }
        //TODO: Handle strided views (seems to be unused in the current codebase)
        std::array<uint32_t, GGML_MAX_DIMS> start;
        std::array<uint32_t, GGML_MAX_DIMS> end;

        // FIXME: Does not work when we are viewing into a permuted tensor. Sucks
        size_t remaining_offset = offset;
        for(size_t i = GGML_MAX_DIMS - 1; i < GGML_MAX_DIMS; i--) {
            start[i] = remaining_offset / src_stride[i];
            end[i] = dst_size[i] + start[i];
            remaining_offset = remaining_offset % src_stride[i];
        }
        std::reverse(start.begin(), start.end());
        std::reverse(end.begin(), end.end());
        tt::tt_metal::Tensor res;

        if(ggml_metalium_debug_flags.print_view) {
            // Debug prints to help debug complicated view operations
            std::cout << "\nggml_metalium_realize_ggml_view() OP: " << ggml_op_desc(tensor) << "\n";
            std::cout << "  dst name: " << tensor->name << "\n";
            std::cout << "  dst shape: " << tensor->ne[0] << " " << tensor->ne[1] << " " << tensor->ne[2] << " " << tensor->ne[3] << "\n";
            std::cout << "  dst stride: " << tensor->nb[0] << " " << tensor->nb[1] << " " << tensor->nb[2] << " " << tensor->nb[3] << "\n";
            std::cout << "  dst extra: " << tensor->extra << "\n";
            if(tensor->extra != nullptr) {
                TensorWithMetadata* meta = (TensorWithMetadata*)tensor->extra;
                std::cout << "  dst tensor: " << meta->tensor << "\n";
                if(meta->tensor != nullptr) {
                    std::cout << "  dst tensor shape: " << meta->tensor->logical_shape() << "\n";
                }
            }
            std::cout << "  dst data: " << tensor->data << "\n";
            std::cout << "  dst view_src: " << tensor->view_src << "\n";
            std::cout << "  dst view_src shape: " << tensor->view_src->ne[0] << " " << tensor->view_src->ne[1] << " " << tensor->view_src->ne[2] << " " << tensor->view_src->ne[3] << "\n";
            std::cout << "  dst view_src stride: " << tensor->view_src->nb[0] << " " << tensor->view_src->nb[1] << " " << tensor->view_src->nb[2] << " " << tensor->view_src->nb[3] << "\n";
            std::cout << "  dst src0: " << src0 << "\n";
            std::cout << "  dst src1: " << tensor->src[1] << "\n";
            std::cout << "  src0 shape: " << src0->ne[0] << " " << src0->ne[1] << " " << src0->ne[2] << " " << src0->ne[3] << "\n";
            std::cout << "  src0 stride: " << src0->nb[0] << " " << src0->nb[1] << " " << src0->nb[2] << " " << src0->nb[3] << "\n";
            std::cout << "  src0 OP: " << ggml_op_desc(src0) << "\n";
            std::cout << "  TT parent shape: " << parent->logical_shape() << "\n";
            std::cout << "  TT slice start: " << start[0] << " " << start[1] << " " << start[2] << " " << start[3] << "\n";
            std::cout << "  TT slice end: " << end[0] << " " << end[1] << " " << end[2] << " " << end[3] << "\n";
            std::cout << std::flush;
        }

        // Actually a reshape written as a view
        if(offset == 0 && ggml_nelements(src0) == ggml_nelements(tensor)) {
            res = ggml_metalium_reshape_tt_tensor_into_ggml(*parent, tensor);
        }
        // Trying to convert a flat 1D tensor to N-D tensor (with an offset, else's it's the above case)
        else if(ggml_n_dims(src0) == 1 && ggml_n_dims(tensor) > 1) {
            // grab the source tensor, slice out the relevant part, and reshape it
            uint32_t offset_elements = offset / ggml_type_size(src0->type);
            uint32_t dst_volume = (uint32_t)ggml_nelements(tensor);
            std::array<uint32_t, GGML_MAX_DIMS> start{0, 0, 0, offset_elements};
            std::array<uint32_t, GGML_MAX_DIMS> end({1, 1, 1, dst_volume + offset_elements});
            std::array<uint32_t, GGML_MAX_DIMS> step = {1, 1, 1, 1};
            tt::tt_metal::Tensor tmp = ttnn::slice(*parent, start, end, step);
            res = ggml_metalium_reshape_tt_tensor_into_ggml(tmp, tensor);
        }
        // The fast path, this is what TTNN is designed for (direct slicing)
        else {
            std::array<uint32_t, GGML_MAX_DIMS> step = {1, 1, 1, 1};
            res = ttnn::slice(*parent, start, end, step);
        }

        return std::make_shared<tt::tt_metal::Tensor>(std::move(res));
    }
    if(op == GGML_OP_RESHAPE) {
        auto t = ggml_metalium_realize_ggml_view(src0);
        return std::make_shared<tt::tt_metal::Tensor>(ggml_metalium_reshape_tt_tensor_into_ggml(*t, tensor));
    }
    if(op == GGML_OP_PERMUTE) {
        std::array<int32_t, GGML_MAX_DIMS> permute;
        memcpy(permute.data(), tensor->op_params, sizeof(permute));

        int ndiff = 0;
        for(int i=0;i<GGML_MAX_DIMS;i++) {
            ndiff += permute[i] != i;
        }
        GGML_ASSERT(ndiff != 1); // Logically impossible

        auto t = ggml_metalium_realize_ggml_view(src0);
        if(ndiff == 0) {
            return t;
        }

        ttsl::SmallVector<int64_t> permute_tt(GGML_MAX_DIMS);
        for(int i=0;i<GGML_MAX_DIMS;i++) {
            permute_tt[i] = GGML_MAX_DIMS - permute[GGML_MAX_DIMS - i - 1] - 1;
        }

        auto res = ttnn::permute(*t, permute_tt);
        if(!ggml_tt_tensors_shape_equal(tensor, res)) {
            res = ggml_metalium_reshape_tt_tensor_into_ggml(res, tensor);
        }
        return std::make_shared<tt::tt_metal::Tensor>(std::move(res));
    }

    TensorWithMetadata* meta = (TensorWithMetadata*)tensor->extra;
    GGML_ASSERT(meta != nullptr);
    if(meta != nullptr && meta->tensor != nullptr) {
        return meta->tensor;
    }

    if(ggml_metalium_is_view(tensor) && tensor->view_src != nullptr) {
        // recursivly resolve the source tensor
        return ggml_metalium_realize_ggml_view(tensor->view_src);
    }

    // HACK: Fallback path: if somehow the framework does not set the real tensor, we can make our own
    auto tt_type = ggml_metalium_ggml2tt_type(tensor->type, meta->bufctx->device->arch());
    auto shape = ttnn::Shape({uint32_t(tensor->ne[3]), uint32_t(tensor->ne[2]), uint32_t(tensor->ne[1]), uint32_t(tensor->ne[0])});
    auto res = ttnn::tilize_with_zero_padding(ttnn::zeros(shape, tt::tt_metal::DataType::BFLOAT16).to_device(meta->bufctx->device.get()), std::nullopt, tt_type);
    meta->tensor = std::make_shared<tt::tt_metal::Tensor>(res);
    meta->ggtype = tensor->type;
    return meta->tensor;
}

const char * ggml_backend_metalium_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_metalium_buffer_type_context * ctx = (ggml_backend_metalium_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

static size_t ggml_backend_metalium_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    // Not using this. Metalium's allication model is not compatible with GGML's allocator
    return 128;
    GGML_UNUSED(buft);
}

// NOTE: I might need to add a metalium tensor wrapper to work around TT tensors have hardware-tagged data types
//       and GGML tensors does not specify the data type during tensor creation.
static size_t ggml_backend_metalium_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_metalium_buffer_type_context * ctx = (ggml_backend_metalium_buffer_type_context *)buft->context;
    auto device = ggml_metalium_get_device(ctx->device_ctx);
    return device->num_dram_channels() * (size_t)device->dram_size_per_channel();
}

static size_t ggml_backend_metalium_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    // Not using this. Metalium's allication model is not compatible with GGML's allocator
    return ggml_nbytes(tensor);
    GGML_UNUSED(buft);
}

static void
ggml_backend_metalium_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_metalium_buffer_context * ctx = ( ggml_backend_metalium_buffer_context *)buffer->context;
    delete ctx;
}

static void ggml_backend_metalium_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                ggml_tensor *tensor,
                                                const void *data, size_t offset,
                                                size_t size)
{
    // Here's the general logic of set_tensor
    // 1. Make a flat buffer and copy the data into it
    //    - If the data is quantized, convert it to BFLOAT16
    //    - Try to directly copy the data if it is already in the correct format
    // 2. Create a TT tensor from the flat buffer as ROW_MAJOR. Send it to the device and tile it
    // 3. If the data is quantized, cast down to BFLOAT8_B or BFLOAT4_B
    // There's a lot of things to do here.
    // TODO: Currently FP32 is hard coded to convert to BFLOAT16. Use FP32 when the hardware supports it
    // TODO: Make a scalable way to decide which GGML type casts to TT quantized types
    // TODO: Use the simpler tilize() when the final 2 dimensions are both multiples of 32
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(tensor->extra != NULL);

    ggml_backend_metalium_buffer_context * bufctx = (ggml_backend_metalium_buffer_context *)buffer->context;
    ggml_type ggtype = tensor->type;
    TensorWithMetadata * meta = (TensorWithMetadata *)tensor->extra;
    const tt::ARCH processor_class = bufctx->device->arch();

    // Make sure we are not writing to a view tensor
    if(size != ggml_nbytes(tensor) || (meta->tensor && ggml_tt_tensors_shape_equal(tensor, *meta->tensor) == false)
        || tensor->view_src != NULL) {
        // FIXME: Reenable this when got time
        // fprintf(stderr, "Warning: Metalium set_tensor() does not work with tensor views\n");
        return;
    }

    std::optional<tt::tt_metal::HostBuffer> storage;
    tt::tt_metal::DataType intermidiate_type = tt::tt_metal::DataType::BFLOAT16;
    if(ggtype == GGML_TYPE_F32) {
        // For now we cast F32 to BF16. Need a scalable way to handle this as WORMHOLD_B0 have native support for F32
        // TODO: Enable proper FP32 when all related bugs gets fixed for devices that support it
        storage = ggml_metalium_data_to_borrowed_storage<float, bfloat16>((const float*)data, size / sizeof(float));
    }
    else if (ggtype == GGML_TYPE_F16) {
        // TT hardware claims to support FP16 but the API does not expose it. For now we use BF16 as it is close enough
        storage = ggml_metalium_data_to_borrowed_storage<ggml_fp16_t, bfloat16>((const ggml_fp16_t*)data, size / sizeof(ggml_fp16_t));
    }
    else if (ggtype == GGML_TYPE_BF16) {
        storage = ggml_metalium_data_to_borrowed_storage<ggml_bf16_t, bfloat16>((const ggml_bf16_t*)data, size / sizeof(ggml_bf16_t));
    }
    else if (ggtype == GGML_TYPE_I32) {
        storage = ggml_metalium_data_to_borrowed_storage<int, uint32_t>((const int*)data, size / sizeof(int));
        intermidiate_type = tt::tt_metal::DataType::UINT32;
    }
    else if (ggml_is_quantized(ggtype)) {
        // Going to FP16 requires a cast to BFLOAT16 which is slower. Instead go to FP32. Even though it's larger
        // it's faster due to one less step.
        storage = ggml_metalium_quantized_to_owned_storage<float>(data, tensor);
        intermidiate_type = tt::tt_metal::DataType::FLOAT32;
    }
    else {
        fmt::println(stderr, "Unsupported data type while uploading to device: {}, name '{}', op type: {}\n", ggml_type_name(ggtype), tensor->name, ggml_op_name(tensor->op));
        GGML_ASSERT(false && "Unsupported data type while uploading to device");
    }
    GGML_ASSERT(storage.has_value() && "Failed to convert data to TT storage");

    ttsl::SmallVector<uint32_t> shape(GGML_MAX_DIMS, 1);
    for(int i = 0; i < GGML_MAX_DIMS; i++) {
        // GGML stores the shape in reverse order
        shape[i] = tensor->ne[GGML_MAX_DIMS - i - 1];
    }

    std::optional<ttsl::SmallVector<int64_t>> permute;
    // In case GGML sent us a non-contiguous tensor, we need to permute it to make it contiguous
    // We don't care about reshape as that doesn't make a difference in row-major layout
    // TODO: This code does not handle yucky cases like stries of [4, 8, 0, 0] but I assume GGML
    // is decent enough to not send us such tensors
    if(!ggml_is_contiguous(tensor)) {
        // Look at ne (aka strides) and figure out the real underlying shape
        std::array<std::pair<uint64_t, int>, GGML_MAX_DIMS> strides;
        for(int i = 0; i < GGML_MAX_DIMS; i++) {
            strides[i] = {tensor->nb[i], i};
        }
        std::sort(strides.begin(), strides.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });

        std::array<std::pair<uint64_t, int>, GGML_MAX_DIMS> s;
        for(int i = 0; i < GGML_MAX_DIMS; i++) {
            s[i] = {tensor->ne[i], strides[i].second};
        }
        std::sort(s.begin(), s.end(), [](const auto& a, const auto& b) {
            return a.second < b.second;
        });
        for(int i = 0; i < GGML_MAX_DIMS; i++) {
            shape[GGML_MAX_DIMS - i - 1] = s[i].first;
        }

        // Now we can figure out the permutation that we need to apply
        ttsl::SmallVector<int64_t> perm(GGML_MAX_DIMS, -1);
        for(int i = 0; i < GGML_MAX_DIMS; i++) {
            perm[strides[i].second] = i;
        }
        permute = perm;
    }

    tt::tt_metal::Tensor t(std::move(*storage), ttnn::Shape(shape)
        , intermidiate_type, tt::tt_metal::Layout::ROW_MAJOR);

    tt::tt_metal::DataType final_type = ggml_metalium_ggml2tt_type(ggtype, processor_class);
    t = ttnn::tilize_with_zero_padding(t.to_device(bufctx->device.get()), std::nullopt, final_type);
    if(permute.has_value()) {
        t = ttnn::permute(t, *permute);
    }
    GGML_ASSERT(t.storage_type() == tt::tt_metal::StorageType::DEVICE);
    GGML_ASSERT(t.dtype() == final_type);
    GGML_ASSERT(ggml_tt_tensors_shape_equal(tensor, t));
    *meta = TensorWithMetadata {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(std::move(t)),
        .ggtype = ggtype,
        .bufctx = bufctx
    };
}

static void ggml_backend_metalium_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                const ggml_tensor *tensor,
                                                void *data, size_t offset,
                                                size_t size)
{
    GGML_UNUSED(buffer);
    // Here's the general logic of get_tensor
    // 1. Get the TT tensor from the metadata
    // 2. If the TT tensor is quantized, cast it to BFLOAT16
    // 3. Call ggml_metalium_tensor_to_ggml to convert the TT tensor to GGML tensor
    //    - ggml_metalium_tensor_to_ggml internally handles the data type conversion
    GGML_ASSERT(size == ggml_nbytes(tensor));
    GGML_ASSERT(tensor->extra != NULL);
    GGML_UNUSED(offset);

    // ggml_backend_metalium_buffer_context * ctx = (ggml_backend_metalium_buffer_context *)buffer->context;

    ggml_type dst_ggtype = tensor->type;

    // auto *meta = (TensorWithMetadata*)tensor->extra;
    // auto shape = meta->tensor->logical_shape();
    // std::cout << "get_tensor():\n";
    // std::cout << "  GGML thinks shape: " << tensor->ne[0] << " " << tensor->ne[1] << " " << tensor->ne[2] << " " << tensor->ne[3] << std::endl;
    // std::cout << "  TTNN thinks shape: " << shape << std::endl;
    std::shared_ptr<tt::tt_metal::Tensor> t;
    if(tensor->op == GGML_OP_TRANSPOSE) {
        // std::cout << "Reading out to transpose tensor" << std::endl;
        // HACK: Yeah this one is stupid. GGML as a row-major framework uses lazy evaluation for transpose.
        //      Which means if we try to copy a transposed tensor. We should not transpose it. Else the other
        //      backend would transpose it again.
        ggml_tensor* src = tensor->src[0];
        bool do_transpose = false;
        while(src->op == GGML_OP_TRANSPOSE) {
            do_transpose = !do_transpose;
            src = src->src[0];
            GGML_ASSERT(src != NULL);
        }
        GGML_ASSERT(src != NULL);
        t = ggml_metalium_realize_ggml_view(src);
        if(do_transpose) {
            *t = ttnn::transpose(*t, -2, -1);
        }
    }
    else if (tensor->op == GGML_OP_PERMUTE) {
        // DITTO above.
        // XXX: This only handles the case where the permute is the only view class operation
        // May broke if there are multiple permutes
        ggml_tensor* src = tensor->src[0];
        t = ggml_metalium_realize_ggml_view(src);
    }
    else if (tensor->op == GGML_OP_RESHAPE) {
        // No reason to do actual reshaping as it doesn't make a difference in row-major layout
        ggml_tensor* src = tensor->src[0];
        while(src->op == GGML_OP_RESHAPE) {
            src = src->src[0];
            GGML_ASSERT(src != NULL);
        }
        GGML_ASSERT(src != NULL);
        t = ggml_metalium_realize_ggml_view(src);
    }
    else {
        t = ggml_metalium_realize_ggml_view(tensor);
        GGML_ASSERT(ggml_metalium_tt_tensor_shape_compatible(tensor, *t));
    }
    if(!ggml_metalium_tt_tensor_shape_compatible(tensor, *t)) {
        t = std::make_shared<tt::tt_metal::Tensor>(ggml_metalium_reshape_tt_tensor_into_ggml(*t, tensor));
    }
    GGML_ASSERT(t->layout() == tt::tt_metal::Layout::TILE);
    if(t->dtype() != tt::tt_metal::DataType::BFLOAT16 && t->dtype() != tt::tt_metal::DataType::FLOAT32 && t->dtype() != tt::tt_metal::DataType::UINT32) {
        t = std::make_shared<tt::tt_metal::Tensor>(ttnn::typecast(*t, tt::tt_metal::DataType::BFLOAT16));
    }

    // TODO: Proper handling of data types
    GGML_ASSERT(dst_ggtype != GGML_TYPE_F64 && dst_ggtype != GGML_TYPE_I16 && dst_ggtype != GGML_TYPE_I8);
    switch(t->dtype()) {
        case tt::tt_metal::DataType::BFLOAT16:
            ggml_metalium_tensor_to_ggml<bfloat16>(*t, (float*)data, dst_ggtype);
            break;
        case tt::tt_metal::DataType::FLOAT32:
            ggml_metalium_tensor_to_ggml<float>(*t, (float*)data, dst_ggtype);
            break;
        case tt::tt_metal::DataType::UINT32:
            ggml_metalium_tensor_to_ggml<uint32_t>(*t, (int*)data, dst_ggtype);
            break;
        default:
            GGML_ASSERT(false && "Unsupported data type in TT tensor when converting to GGML tensor");
            break;
    }
}

static void * ggml_backend_metalium_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_metalium_buffer_context * ctx = (ggml_backend_metalium_buffer_context *)buffer->context;
    return (uint8_t*)0xdeadbeef + ctx->base_offset;
}

static enum ggml_status
ggml_backend_metalium_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                     ggml_tensor *tensor)
{
    ggml_backend_metalium_buffer_context * bufctx = (ggml_backend_metalium_buffer_context *)buffer->context;

    bufctx->metadata_to_free.push_back(std::make_unique<TensorWithMetadata>());
    TensorWithMetadata* meta = bufctx->metadata_to_free.back().get();
    tensor->extra = meta;
    *meta = {
        .tensor = nullptr,
        .ggtype = GGML_TYPE_COUNT,
        .bufctx = bufctx
    };

    // HACK: Make KV cache work
    std::string_view name(tensor->name);
    if(strstr(std::string(name).c_str(), "cache") != NULL && tensor->op == GGML_OP_NONE) {
        std::vector<uint32_t> shape(tensor->ne, tensor->ne + GGML_MAX_DIMS);
        std::reverse(shape.begin(), shape.end());
        auto t = ttnn::zeros(ttnn::Shape(shape), ggml_metalium_ggml2tt_type(tensor->type, bufctx->device->arch()), tt::tt_metal::Layout::ROW_MAJOR);
        t = ttnn::tilize_with_zero_padding(t.to_device(bufctx->device.get()));
        meta->tensor = std::make_shared<tt::tt_metal::Tensor>(std::move(t));
    }
    // std::cout << "Creating tensor with address: " << tensor->data << ", shape = " << tensor->ne[0] << " " << tensor->ne[1] << " " << tensor->ne[2] << " " << tensor->ne[3] << ", name " << tensor->name << std::endl;
    GGML_UNUSED(buffer);
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_metalium_buffer_clear(ggml_backend_buffer_t buffer,
                                                        uint8_t value)
{
    // Not using this. Metalium's allication model is not compatible with GGML's allocator
    GGML_UNUSED(buffer);
    GGML_UNUSED(value);
}

static bool
ggml_backend_metalium_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                    const ggml_tensor *src,
                                    ggml_tensor *dst)
{
    GGML_UNUSED(buffer);

    GGML_ASSERT(src->extra != NULL);
    GGML_ASSERT(dst->extra != NULL);

    TensorWithMetadata * src_meta = (TensorWithMetadata *)src->extra;
    TensorWithMetadata * dst_meta = (TensorWithMetadata *)dst->extra;

    tt::tt_metal::Tensor& src_tensor = *src_meta->tensor;

    tt::tt_metal::Tensor ret = ttnn::identity(src_tensor);
    GGML_ASSERT(ret.storage_type() == tt::tt_metal::StorageType::DEVICE);
    dst_meta->tensor = std::make_shared<tt::tt_metal::Tensor>(std::move(ret));
    dst_meta->ggtype = dst->type;
    return true;
}

static void ggml_backend_metalium_buffer_reset(ggml_backend_buffer_t buffer) {
    ggml_backend_metalium_buffer_context * bufctx = (ggml_backend_metalium_buffer_context *)buffer->context;
    bufctx->metadata_to_free.clear();
}

static struct ggml_backend_buffer_i ggml_backend_metalium_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_metalium_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_metalium_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_metalium_buffer_init_tensor,
    /* .memset_tensor   = */ nullptr,
    /* .set_tensor      = */ ggml_backend_metalium_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_metalium_buffer_get_tensor,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ ggml_backend_metalium_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_metalium_buffer_clear,
    /* .reset           = */ ggml_backend_metalium_buffer_reset,
};


static ggml_backend_buffer_t
ggml_backend_metalium_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                           size_t size) {
    ggml_backend_metalium_buffer_type_context * buft_ctx = (ggml_backend_metalium_buffer_type_context *)buft->context;

    ggml_metalium_get_device(buft_ctx->device_ctx);

    // FIXME: GGML unit tests fails if I don't add some additional memory to the buffer beyond the requested size
    size_t alloc_size = size + 4096 * 1024;
    // real allocation is deferred until the first tensor is set because we don't know the underlying tensor type yet
    ggml_backend_metalium_buffer_context* ctx = new ggml_backend_metalium_buffer_context {
        .ggml_buffer_size_bytes = size,
        .name = buft_ctx->name,
        .device = buft_ctx->device_ctx->device,
        .base_offset = g_metalium_base_offset,

        .metadata_to_free = {}
    };
    g_metalium_base_offset += alloc_size;
    // std::cout << "Allocating buffer of size " << size << " bytes\n";
    return ggml_backend_buffer_init(buft, ggml_backend_metalium_buffer_interface, ctx, alloc_size);
}

static bool ggml_backend_metalium_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return false;
}

static ggml_backend_buffer_type_i ggml_backend_metalium_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_metalium_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_metalium_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_metalium_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_metalium_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_metalium_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_metalium_buffer_type_is_host,
};

ggml_backend_buffer_type_t ggml_backend_metalium_buffer_type(ggml_backend_dev_t dev, ggml_backend_metalium_device_context* dev_ctx) {
    auto device_id = dev_ctx->device_id;
    ggml_backend_metalium_reg_context* regctx = (ggml_backend_metalium_reg_context*)(dev->reg->context);

    GGML_ASSERT((size_t)device_id < tt::tt_metal::GetNumAvailableDevices());
    GGML_ASSERT((size_t)device_id < regctx->devices.size());

    static std::map<int, ggml_backend_buffer_type> buffer_type_map;
    static std::set<std::unique_ptr<ggml_backend_metalium_buffer_type_context>> buffer_type_context_deleter;
    auto it = buffer_type_map.find(device_id);
    if(it != buffer_type_map.end()) {
        return &it->second;
    }

    auto bufctx = std::make_unique<ggml_backend_metalium_buffer_type_context>(
        ggml_backend_metalium_buffer_type_context{
            .device_ctx = dev_ctx,
            .name = "Metalium " + std::to_string(device_id),
        });
    auto* bufctx_ptr = bufctx.get();
    buffer_type_context_deleter.insert(std::move(bufctx));

    buffer_type_map[device_id] = {
        /* .iface    = */ ggml_backend_metalium_buffer_type_interface,
        /* .device   = */ regctx->devices[device_id],
        /* .context  = */ bufctx_ptr,
    };
    return &buffer_type_map[device_id];
}

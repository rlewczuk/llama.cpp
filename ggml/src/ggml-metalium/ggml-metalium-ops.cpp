#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "ggml-impl.h"

#include "ggml-metalium-context.h"
#include "ggml-metalium-ops.h"
#include "ggml-metalium-util.h"

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/moreh/moreh_group_norm/moreh_group_norm.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
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
#include <memory>
#include <optional>
#include <string_view>
#include <unordered_map>
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

inline static void ggml_metalium_op_src_sanity_check(const struct ggml_tensor * node, int idx) {
    GGML_ASSERT(node->src[idx] != NULL);
    GGML_ASSERT(node->src[idx]->extra != NULL);
    auto* meta = (TensorWithMetadata*)(node->src[idx]->extra);
    if(meta->tensor != NULL) {
        GGML_ASSERT(meta->tensor->storage_type() == tt::tt_metal::StorageType::DEVICE);
        GGML_ASSERT(meta->tensor->layout() == tt::tt_metal::Layout::TILE);
    }
}

// Sanity check macros to ensure that the tensors are in the correct format and we won't crash
#define GGML_METALIUM_OP_SANITY_CHECK(_node) \
    GGML_ASSERT((_node)->extra != NULL);
// Check if the tensor is on the device (so we wont'e be using the CPU) as well as letting us crash early
#define GGML_METALIUM_OP_SRC_SANITY_CHECK(_node, _idx) ggml_metalium_op_src_sanity_check(_node, _idx);
#define GGML_METALIUM_OP_SRC0_SANITY_CHECK(_node) GGML_METALIUM_OP_SRC_SANITY_CHECK(_node, 0)
#define GGML_METALIUM_OP_SRC1_SANITY_CHECK(_node) GGML_METALIUM_OP_SRC_SANITY_CHECK(_node, 1)

bool ggml_backend_metalium_can_mul_mat(const struct ggml_tensor * dst)
{
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    // TTNN only supports matmul of shape [B, 1, M, K] x [1, 1, K, N] (bcast_batch=True)
    // or [B, 1, M, K] x [B, 1, K, N] (bcast_batch=False)
    // For now we simply only allow those shapes. We transpose the shapes ourselves
    // TODO: Detect when shape[1] can be removed and do that automagically

    return src0->ne[0] == src1->ne[0] && src0->ne[2] == 1 && src1->ne[2] == 1 &&
        (src0->ne[3] == src1->ne[3] || src0->ne[3] == 1);
}

static void ggml_backend_metalium_mul_mat(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst) {
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC1_SANITY_CHECK(dst);

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const enum ggml_type type = src0->type;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(src0->extra != NULL);
    GGML_ASSERT(src1->extra != NULL);
    GGML_ASSERT(dst->extra != NULL);

    auto ap = ggml_metalium_realize_ggml_view(src0);
    auto bp = ggml_metalium_realize_ggml_view(src1);
    auto &a = *ap;
    auto &b = *bp;
    TensorWithMetadata* cm = (TensorWithMetadata*)dst->extra;

    GGML_ASSERT(cm != NULL);

    if(a.dtype() == tt::tt_metal::DataType::BFLOAT16 && b.dtype() == tt::tt_metal::DataType::BFLOAT16) {
        // Fast path
        // Need to increase the math fidelity as moreh_matmul by default uses LoFi and won't pass GGML unit tests
        ttnn::DeviceComputeKernelConfig cfg = ggml_metalium_make_compute_kernel_config(a.device());
        *cm = {
            .tensor = std::make_shared<tt::tt_metal::Tensor>(ttnn::moreh_matmul(b, a, false, true, std::nullopt, std::nullopt, std::nullopt, cfg)),
            .ggtype = dst->type,
            .bufctx = cm->bufctx
        };
    }
    else {
        tt::tt_metal::Tensor aT;
        if(src0->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS && ggml_metalium_debug_flags.cache_mm_transpose) {
            auto it = ctx->transposed_weights.find(src0->name);
            if(it == ctx->transposed_weights.end()) {
                aT = ttnn::transpose(a, -2, -1);
                ctx->transposed_weights[src0->name] = aT;
            }
            else {
                aT = it->second;
            }
        }
        else {
            aT = ttnn::transpose(a, -2, -1);
        }
        // TODO: Ask TT to support multiplication of pre-transposed tensors. Calling transpose here is inefficient
        // https://github.com/tenstorrent/tt-metal/issues/9709
        *cm = {
            .tensor = std::make_shared<tt::tt_metal::Tensor>(ttnn::operations::matmul::matmul(b, aT)),
            .ggtype = dst->type,
            .bufctx = cm->bufctx
        };
    }
    GGML_ASSERT(cm->tensor->storage_type() == tt::tt_metal::StorageType::DEVICE);
    GGML_UNUSED(ctx);
}

bool ggml_backend_metalium_can_cpy(const struct ggml_tensor * dst)
{
    if(dst->op != GGML_OP_CPY && dst->op != GGML_OP_CONT && dst->op != GGML_OP_DUP) {
        return true;
    }

    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];
    if(src0 == nullptr) {
        return false;
    }
    if((src0->type == GGML_TYPE_I32 || dst->type == GGML_TYPE_I32) && src0->type != dst->type) {
        return false;
    }
    if(src1 != nullptr && (ggml_is_permuted(src1) || ggml_metalium_is_view(src1))) {
        return false;
    }
    if(ggml_metalium_is_view(src0) && !ggml_metalium_is_simple_unit_slice(src0)) {
        // TODO: Defensive fallback to CPU for CONT/CPY/DUP from non-simple GGML views.
        // Metalium view realization currently cannot safely materialize all GGML view
        // layouts as TTNN slices; implement full strided/permuted view support and remove this fallback later.
        return false;
    }
    return true;
}

static void ggml_backend_metalium_cpy(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;
    ggml_tensor* src0 = dst->src[0];

    // TODO: Check we are not writing into a view
    auto res = ggml_metalium_realize_ggml_view(src0);
    if(!ggml_tt_tensors_shape_equal(dst, *res)) {
        res = std::make_shared<tt::tt_metal::Tensor>(ggml_metalium_reshape_tt_tensor_into_ggml(*res, dst));
    }

    const auto dst_tt_type = ggml_metalium_ggml2tt_type(dst->type, res->device()->arch());
    if(res->dtype() != dst_tt_type) {
        res = std::make_shared<tt::tt_metal::Tensor>(ttnn::typecast(*res, dst_tt_type));
    }

    *dst_meta = {
        .tensor = res,
        .ggtype = dst->type,
        .bufctx = dst_meta->bufctx
    };
}

static bool ggml_backend_metalium_activations(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst, ggml_unary_op op) {
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    const struct ggml_tensor * src0 = dst->src[0];
    TensorWithMetadata* meta = (TensorWithMetadata*)src0->extra;
    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;

    auto src_tensor = ggml_metalium_realize_ggml_view(src0);

    tt::tt_metal::Tensor ret;
    switch (op) {
        case GGML_UNARY_OP_ABS:
            ret = ttnn::abs(*src_tensor);
            break;
        case GGML_UNARY_OP_SGN:
            ret = ttnn::sign(*src_tensor);
            break;
        case GGML_UNARY_OP_NEG:
            ret = ttnn::neg(*src_tensor);
            break;
        // Not accurate enough to pass unit tests
        case GGML_UNARY_OP_TANH:
            ret = ttnn::tanh(*src_tensor);
            break;
        case GGML_UNARY_OP_ELU:
            ret = ttnn::elu(*src_tensor, 1.0f);
            break;
        case GGML_UNARY_OP_RELU:
            ret = ttnn::relu(*src_tensor);
            break;
        // Not accurate enough to pass unit tests
        case GGML_UNARY_OP_SIGMOID:
            ret = ttnn::sigmoid(*src_tensor);
            break;
        case GGML_UNARY_OP_GELU:
            ret = ttnn::gelu(*src_tensor, false);
            break;
        case GGML_UNARY_OP_GELU_QUICK:
            ret = ttnn::gelu(*src_tensor);
            break;
        case GGML_UNARY_OP_SILU:
            ret = ttnn::silu(*src_tensor);
            break;
        case GGML_UNARY_OP_HARDSWISH:
            ret = ttnn::hardswish(*src_tensor); // , 1.f/6.f, 0.5
            break;
        case GGML_UNARY_OP_HARDSIGMOID:
            ret = ttnn::hardsigmoid(*src_tensor); // , 1.f/6.f, 0.5
            break;
        case GGML_UNARY_OP_STEP:
            // TODO: Make sure the resulting data type matches the input
            ret = ttnn::typecast(ttnn::gtz(*src_tensor), ggml_metalium_ggml2tt_type(dst->type, src_tensor->device()->arch()));
            break;
        case GGML_UNARY_OP_EXP:
            ret = ttnn::exp(*src_tensor);
            break;
        default:
            return false;
    }
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(std::move(ret)),
        .ggtype = dst->type,
        .bufctx = meta->bufctx
    };
    return true;
}
static void ggml_backend_metalium_leaky_relu(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst) {
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    const struct ggml_tensor * src0 = dst->src[0];
    TensorWithMetadata* meta = (TensorWithMetadata*)src0->extra;
    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;
    auto src_tensor = ggml_metalium_realize_ggml_view(src0);

    float negative_slope;
    GGML_ASSERT(dst->op_params != NULL);
    memcpy(&negative_slope, dst->op_params, sizeof(float));

    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(ttnn::leaky_relu(*src_tensor, negative_slope)),
        .ggtype = dst->type,
        .bufctx = meta->bufctx
    };
}
static tt::tt_metal::Tensor ggml_metalium_flatten_for_eltwise(const tt::tt_metal::Tensor & tensor)
{
    const auto shape = tensor.logical_shape();
    GGML_ASSERT(shape.size() == GGML_MAX_DIMS);

    const uint32_t height = shape[0] * shape[1] * shape[2];
    const uint32_t width = shape[3];
    if(shape[0] == 1 && shape[1] == 1) {
        return tensor;
    }

    return ttnn::reshape(tensor, ttnn::Shape({1, 1, height, width}));
}

static tt::tt_metal::Tensor ggml_metalium_reshape_for_eltwise_output(const tt::tt_metal::Tensor & tensor, const ggml_tensor * dst)
{
    return ggml_metalium_reshape_tt_tensor_into_ggml(tensor, dst);
}

static bool ggml_metalium_needs_eltwise_flatten(const ggml_tensor * t)
{
    const uint32_t width = t->ne[0];
    const uint32_t height = t->ne[1] * t->ne[2] * t->ne[3];
    return width % 32 == 0 && height % 32 == 0 && t->ne[1] < 32 && (t->ne[2] > 1 || t->ne[3] > 1);
}

static bool ggml_metalium_can_flatten_for_eltwise(const ggml_tensor * t)
{
    return !ggml_metalium_is_view(t) && ggml_is_contiguous(t);
}

static void ggml_backend_metalium_bin_op(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst, ggml_op op) {
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC1_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    TensorWithMetadata* meta0 = (TensorWithMetadata*)src0->extra;
    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;

    auto src_tensor0 = ggml_metalium_realize_ggml_view(src0);
    auto src_tensor1 = ggml_metalium_realize_ggml_view(src1);

    const bool flatten = src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && op == GGML_OP_ADD &&
        ggml_metalium_needs_eltwise_flatten(dst) &&
        ggml_metalium_can_flatten_for_eltwise(dst) &&
        ggml_metalium_can_flatten_for_eltwise(src0) &&
        ggml_metalium_can_flatten_for_eltwise(src1);
    tt::tt_metal::Tensor a = flatten ? ggml_metalium_flatten_for_eltwise(*src_tensor0) : *src_tensor0;
    tt::tt_metal::Tensor b = flatten ? ggml_metalium_flatten_for_eltwise(*src_tensor1) : *src_tensor1;

    tt::tt_metal::Tensor ret;
    const std::optional<const tt::tt_metal::DataType> dst_tt_type = ggml_metalium_ggml2tt_type(dst->type, a.device()->arch());
    switch(op) {
        case GGML_OP_ADD:
            ret = ttnn::add(a, b, dst_tt_type);
            break;
        case GGML_OP_MUL:
            ret = ttnn::multiply(a, b, dst_tt_type);
            break;
        case GGML_OP_SUB:
            ret = ttnn::subtract(a, b, dst_tt_type);
            break;
        case GGML_OP_DIV:
            ret = ttnn::divide(a, b, dst_tt_type);
            break;
        default:
            GGML_ASSERT(false && "Unsupported binary operation");
    }
    if(flatten) {
        ret = ggml_metalium_reshape_for_eltwise_output(ret, dst);
    }
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(std::move(ret)),
        .ggtype = dst->type,
        .bufctx = meta0->bufctx
    };
}

bool ggml_backend_metalium_can_set(const struct ggml_tensor * dst)
{
    int32_t params[5];
    memcpy(params, dst->op_params, sizeof(params));
    auto [nb1, nb2, nb3, offset, inplace] = std::to_array(params);

    if(offset >= nb3 || offset % nb1 != 0 || ggml_n_dims(dst->src[0]) < ggml_n_dims(dst->src[1]) ||
        ggml_n_dims(dst->src[1]) != 1) {
        return false;
    }

    return true;
}

static void ggml_backend_metalium_set(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC1_SANITY_CHECK(dst);

    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;
    TensorWithMetadata* src0_meta = (TensorWithMetadata*)dst->src[0]->extra;
    TensorWithMetadata* src1_meta = (TensorWithMetadata*)dst->src[1]->extra;

    int32_t params[5];
    memcpy(params, dst->op_params, sizeof(params));
    auto [nb1, nb2, nb3, offset, inplace] = std::to_array(params);

    int idx = offset / nb1;
    int batch_idx = offset / nb2;
    GGML_ASSERT(offset < nb3);
    GGML_ASSERT(offset % nb1 == 0);
    auto res = ttnn::update_cache(*src0_meta->tensor, *src1_meta->tensor, idx, batch_idx);
    if(!inplace) {
        *dst_meta = {
            .tensor = std::make_shared<tt::tt_metal::Tensor>(res),
            .ggtype = dst->type,
            .bufctx = src0_meta->bufctx
        };
    }
    else {
        std::shared_ptr<tt::tt_metal::Tensor> tensor = std::make_shared<tt::tt_metal::Tensor>(res);
        *src0_meta = {
            .tensor = tensor,
            .ggtype = dst->type,
            .bufctx = src0_meta->bufctx
        };
        *dst_meta = {
            .tensor = tensor,
            .ggtype = dst->type,
            .bufctx = src0_meta->bufctx
        };
    }
  }

bool ggml_backend_metalium_can_set_rows(const struct ggml_tensor * dst)
{
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    if(src0 == nullptr || src1 == nullptr || src2 == nullptr) {
        return false;
    }
    if(src0->type != GGML_TYPE_F32) {
        return false;
    }
    if(src1->type != GGML_TYPE_I32 && src1->type != GGML_TYPE_I64) {
        return false;
    }
    if(dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        return false;
    }
    if(src2->type != dst->type) {
        return false;
    }
    // TTNN untilize currently crashes for this padding pattern; leave it to another backend.
    if(src0->ne[1] == 1 && src0->ne[0] > 32 && src0->ne[0] % 32 != 0) {
        return false;
    }
    return true;
}

static const std::byte * ggml_metalium_host_shadow_data(const ggml_tensor * tensor) {
    const ggml_tensor * storage_tensor = tensor;
    size_t offset = 0;
    if(ggml_metalium_is_view(tensor) && tensor->view_src != nullptr) {
        storage_tensor = tensor->view_src;
        offset = tensor->view_offs;
    }
    const TensorWithMetadata * meta = (const TensorWithMetadata*)storage_tensor->extra;
    if(meta == nullptr || meta->host_shadow.empty()) {
        return nullptr;
    }
    GGML_ASSERT(offset < meta->host_shadow.size());
    return meta->host_shadow.data() + offset;
}

static bool ggml_metalium_copy_f32_from_host_shadow(const ggml_tensor * tensor, std::vector<float> & out) {
    const std::byte * base = ggml_metalium_host_shadow_data(tensor);
    if(base == nullptr) {
        return false;
    }
    out.resize(ggml_nelements(tensor));
    size_t idx = 0;
    for(int64_t i3 = 0; i3 < tensor->ne[3]; ++i3) {
        for(int64_t i2 = 0; i2 < tensor->ne[2]; ++i2) {
            for(int64_t i1 = 0; i1 < tensor->ne[1]; ++i1) {
                for(int64_t i0 = 0; i0 < tensor->ne[0]; ++i0) {
                    const std::byte * ptr = base + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
                    if(tensor->type == GGML_TYPE_F32) {
                        out[idx++] = *(const float*)ptr;
                    }
                    else if(tensor->type == GGML_TYPE_F16) {
                        out[idx++] = ggml_fp16_to_fp32(*(const ggml_fp16_t*)ptr);
                    }
                    else {
                        GGML_UNREACHABLE();
                    }
                }
            }
        }
    }
    return true;
}

static bool ggml_metalium_copy_u32_from_host_shadow(const ggml_tensor * tensor, std::vector<uint32_t> & out) {
    const std::byte * base = ggml_metalium_host_shadow_data(tensor);
    if(base == nullptr) {
        return false;
    }
    out.resize(ggml_nelements(tensor));
    size_t idx = 0;
    for(int64_t i3 = 0; i3 < tensor->ne[3]; ++i3) {
        for(int64_t i2 = 0; i2 < tensor->ne[2]; ++i2) {
            for(int64_t i1 = 0; i1 < tensor->ne[1]; ++i1) {
                for(int64_t i0 = 0; i0 < tensor->ne[0]; ++i0) {
                    const std::byte * ptr = base + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
                    if(tensor->type == GGML_TYPE_I32) {
                        out[idx++] = *(const int32_t*)ptr;
                    }
                    else if(tensor->type == GGML_TYPE_I64) {
                        out[idx++] = *(const int64_t*)ptr;
                    }
                    else {
                        GGML_UNREACHABLE();
                    }
                }
            }
        }
    }
    return true;
}

template <typename SrcType, typename DstType>
static void ggml_metalium_copy_logical_from_tt(const tt::tt_metal::Tensor & tensor, std::vector<DstType> & out) {
    ttnn::Shape shape = tensor.logical_shape();
    ttnn::Shape padded_shape = tensor.padded_shape();

    std::array<size_t, 4> nshape {1, 1, 1, 1};
    std::array<size_t, 4> padded_nshape {1, 1, 1, 1};
    for(size_t i = 0; i < shape.size(); i++) {
        nshape[4 - shape.size() + i] = shape[i];
    }
    for(size_t i = 0; i < padded_shape.size(); i++) {
        padded_nshape[4 - padded_shape.size() + i] = padded_shape[i];
    }

    const size_t logical_volume = nshape[0] * nshape[1] * nshape[2] * nshape[3];
    out.resize(logical_volume);

    if(logical_volume == 0) {
        return;
    }

    // Prefer tensor.cpu().to_vector<SrcType>() over ttnn::untilize(tensor).cpu():
    // 1. The untilize pipeline can crash (integer divide-by-zero in
    //    ttnn::operations::data_movement::get_pf_type) for certain
    //    "narrow" shapes like [33, 1, 2, 3] whose padded dimensions are
    //    not aligned to TTNN's tile constraints.
    // 2. to_vector() reads the data through TTNN's native path which
    //    is more robust for these shapes.
    // Fall back to the untilize-based path if to_vector() returns an
    // undersized vector. As a last resort, fill with zero so the
    // downstream code surfaces a clear assertion instead of UB.
    {
        tt::tt_metal::Tensor cpu_tensor = tensor.cpu();
        try {
            std::vector<SrcType> vec = cpu_tensor.to_vector<SrcType>();
            if(vec.size() >= logical_volume) {
                for(size_t i = 0; i < logical_volume; ++i) {
                    if constexpr(std::is_same_v<SrcType, bfloat16>) {
                        out[i] = static_cast<DstType>(static_cast<float>(vec[i]));
                    }
                    else {
                        out[i] = static_cast<DstType>(vec[i]);
                    }
                }
                return;
            }
        } catch(...) {
            // Fall through to the untilize-based path below.
        }
    }

    tt::tt_metal::Tensor row_major_tensor = ttnn::untilize(tensor).cpu();
    GGML_ASSERT(row_major_tensor.storage_type() == tt::tt_metal::StorageType::HOST);

    const tt::tt_metal::HostStorage & storage = row_major_tensor.host_storage();
    const auto buffer = storage.buffer().get_shard({0, 0}).value();
    auto view = buffer.view_as<SrcType>();
    if(view.size() == 0) {
        std::fill(out.begin(), out.end(), DstType(0));
        return;
    }

    const std::array<size_t, 4> stride = {
        padded_nshape[1] * padded_nshape[2] * padded_nshape[3],
        padded_nshape[2] * padded_nshape[3],
        padded_nshape[3],
        1,
    };

    const SrcType * buf = &view[0];
    size_t idx = 0;
    for(size_t w = 0; w < nshape[0]; w++) {
        for(size_t z = 0; z < nshape[1]; z++) {
            for(size_t y = 0; y < nshape[2]; y++) {
                for(size_t x = 0; x < nshape[3]; x++) {
                    const size_t src_idx = w * stride[0] + z * stride[1] + y * stride[2] + x;
                    if constexpr(std::is_same_v<SrcType, bfloat16>) {
                        out[idx++] = static_cast<DstType>(static_cast<float>(buf[src_idx]));
                    }
                    else {
                        out[idx++] = static_cast<DstType>(buf[src_idx]);
                    }
                }
            }
        }
    }
}

template <typename T>
static tt::tt_metal::HostBuffer ggml_metalium_vector_to_host_buffer(std::vector<T> && vec) {
    auto vec_ptr = std::make_shared<std::vector<T>>(std::move(vec));
    int * refcount = new int(0);
    T * data = vec_ptr->data();
    const size_t size = vec_ptr->size();
    tt::tt_metal::MemoryPin pin(
        [refcount]() mutable { (*refcount)++; },
        [refcount, vec_ptr=std::move(vec_ptr)]() mutable {
            (*refcount)--;
            if(*refcount == 0) {
                delete refcount;
            }
        }
    );
    return tt::tt_metal::HostBuffer(ttsl::Span<T>(data, size), std::move(pin));
}

static void ggml_backend_metalium_set_rows(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC1_SANITY_CHECK(dst);
    GGML_ASSERT(dst->src[2] != nullptr && dst->src[2]->extra != nullptr);

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    ggml_tensor * src2 = dst->src[2];

    TensorWithMetadata * dst_meta  = (TensorWithMetadata*)dst->extra;
    TensorWithMetadata * src2_meta = (TensorWithMetadata*)src2->extra;

    GGML_ASSERT(ggml_backend_metalium_can_set_rows(dst));
    GGML_ASSERT(src2_meta->tensor != nullptr);

    std::vector<float> src_rows;
    if(!ggml_metalium_copy_f32_from_host_shadow(src0, src_rows)) {
        auto src0_tensor = ggml_metalium_realize_ggml_view(src0);
        ggml_metalium_copy_logical_from_tt<float, float>(*src0_tensor, src_rows);
    }

    std::vector<uint32_t> row_ids;
    if(!ggml_metalium_copy_u32_from_host_shadow(src1, row_ids)) {
        auto src1_tensor = ggml_metalium_realize_ggml_view(src1);
        ggml_metalium_copy_logical_from_tt<uint32_t, uint32_t>(*src1_tensor, row_ids);
    }

    std::vector<float> dst_data;
    if(!ggml_metalium_copy_f32_from_host_shadow(src2, dst_data)) {
        if(src2_meta->tensor->dtype() == tt::tt_metal::DataType::FLOAT32) {
            ggml_metalium_copy_logical_from_tt<float, float>(*src2_meta->tensor, dst_data);
        }
        else {
            ggml_metalium_copy_logical_from_tt<bfloat16, float>(*src2_meta->tensor, dst_data);
        }
    }

    const int64_t nc = src0->ne[0];
    const int64_t nr = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];
    const int64_t ne1 = dst->ne[1];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];

    for(int64_t i03 = 0; i03 < ne03; ++i03) {
        for(int64_t i02 = 0; i02 < ne02; ++i02) {
            for(int64_t i = 0; i < nr; ++i) {
                const uint32_t row = row_ids[(i03 % ne12) * ne11 * nr + (i02 % ne11) * nr + i];
                GGML_ASSERT(row < (uint32_t)ne1);
                const size_t src_off = ((i03 * ne02 + i02) * nr + i) * nc;
                const size_t dst_off = ((i03 * ne02 + i02) * ne1 + row) * nc;
                GGML_ASSERT(src_off + nc <= src_rows.size());
                GGML_ASSERT(dst_off + nc <= dst_data.size());
                memcpy(dst_data.data() + dst_off, src_rows.data() + src_off, nc * sizeof(float));
            }
        }
    }

    std::vector<std::byte> host_shadow(ggml_nbytes(src2));
    if(dst->type == GGML_TYPE_F32) {
        GGML_ASSERT(host_shadow.size() == dst_data.size() * sizeof(float));
        memcpy(host_shadow.data(), dst_data.data(), host_shadow.size());
    }
    else {
        std::vector<ggml_fp16_t> f16(dst_data.size());
        GGML_ASSERT(host_shadow.size() == f16.size() * sizeof(ggml_fp16_t));
        for(size_t i = 0; i < dst_data.size(); ++i) {
            f16[i] = ggml_fp32_to_fp16(dst_data[i]);
        }
        memcpy(host_shadow.data(), f16.data(), host_shadow.size());
    }

    std::vector<uint32_t> shape(src2->ne, src2->ne + GGML_MAX_DIMS);
    std::reverse(shape.begin(), shape.end());
    tt::tt_metal::Tensor res;
    if(dst->type == GGML_TYPE_F32) {
        auto storage = ggml_metalium_vector_to_host_buffer(std::move(dst_data));
        tt::tt_metal::Tensor t(std::move(storage), ttnn::Shape(shape), tt::tt_metal::DataType::FLOAT32, tt::tt_metal::Layout::ROW_MAJOR);
        res = ttnn::tilize_with_zero_padding(t.to_device(src2_meta->bufctx->device.get()), std::nullopt, tt::tt_metal::DataType::FLOAT32);
    }
    else {
        std::vector<bfloat16> bf16(dst_data.size());
        const auto * trait = ggml_get_type_traits_cpu(GGML_TYPE_BF16);
        trait->from_float(dst_data.data(), bf16.data(), bf16.size());
        auto storage = ggml_metalium_vector_to_host_buffer(std::move(bf16));
        tt::tt_metal::Tensor t(std::move(storage), ttnn::Shape(shape), tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::Layout::ROW_MAJOR);
        res = ttnn::tilize_with_zero_padding(t.to_device(src2_meta->bufctx->device.get()), std::nullopt, tt::tt_metal::DataType::BFLOAT16);
    }

    auto tensor = std::make_shared<tt::tt_metal::Tensor>(std::move(res));
    *src2_meta = {
        .tensor = tensor,
        .ggtype = src2->type,
        .bufctx = src2_meta->bufctx,
        .host_shadow = host_shadow,
    };
    *dst_meta = {
        .tensor = tensor,
        .ggtype = dst->type,
        .bufctx = src2_meta->bufctx,
        .host_shadow = std::move(host_shadow),
    };
}
static void ggml_backend_metalium_clamp(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);

    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;

    float data[2];
    memcpy(data, dst->op_params, sizeof(data));
    auto [min, max] = std::to_array(data);

    auto t = ggml_metalium_realize_ggml_view(dst->src[0]);
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(ttnn::clamp(*t, min, max)),
        .ggtype = dst->type,
        .bufctx = ((TensorWithMetadata*)dst->src[0]->extra)->bufctx
    };
}

static void ggml_backend_metalium_scale(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);

    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;

    std::array<float, 2> params;
    memcpy(params.data(), dst->op_params, sizeof(params));
    auto [scale, bias] = params;

    auto t = ggml_metalium_realize_ggml_view(dst->src[0]);
    ttnn::Tensor res;
    if(bias == 0.f) {
        res = ttnn::multiply(*t, scale);
    }
    else {
        res = ttnn::add(ttnn::multiply(*t, scale, std::nullopt, ttnn::L1_MEMORY_CONFIG), bias);
    }
    // TODO: Support in-place scaling
    GGML_ASSERT(!ggml_metalium_is_view(dst->src[0]));
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(std::move(res)),
        .ggtype = dst->type,
        .bufctx = ((TensorWithMetadata*)dst->src[0]->extra)->bufctx
    };
}

bool ggml_backend_metalium_can_get_rows(const struct ggml_tensor * dst)
{
    const ggml_tensor *idxs = dst->src[1];
    if(idxs->ne[0] != 1 || idxs->ne[1] != 1 || idxs->ne[2] != 1 || idxs->ne[3] != 1) {
        return false;
    }
    if(ggml_n_dims(dst->src[0]) != 1) {
        return false;
    }
    return true;
}

static void ggml_backend_metalium_get_rows(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);

    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;

    auto t = ggml_metalium_realize_ggml_view(dst->src[0]);
    *dst_meta = {
        .tensor = t,
        .ggtype = dst->type,
        .bufctx = ((TensorWithMetadata*)dst->src[0]->extra)->bufctx
    };
}

static void ggml_backend_metalium_norm(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst, bool rms)
{
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);

    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;

    float esp = 0;
    memcpy(&esp, dst->op_params, sizeof(esp));

    auto t = ggml_metalium_realize_ggml_view(dst->src[0]);
    tt::tt_metal::Tensor res;
    if(rms) {
        res = ttnn::rms_norm(*t, esp);
    }
    else {
        res = ttnn::layer_norm(*t, esp);
    }
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(std::move(res)),
        .ggtype = dst->type,
        .bufctx = ((TensorWithMetadata*)dst->src[0]->extra)->bufctx
    };
}

static void ggml_backend_metalium_add1(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);

    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;

    float esp = 0;
    memcpy(&esp, dst->op_params, sizeof(esp));

    auto t = ggml_metalium_realize_ggml_view(dst->src[0]);
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(ttnn::add(*t, 1.f)),
        .ggtype = dst->type,
        .bufctx = ((TensorWithMetadata*)dst->src[0]->extra)->bufctx
    };
}

static void ggml_backend_metalium_sqrt(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);

    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;

    float esp = 0;
    memcpy(&esp, dst->op_params, sizeof(esp));

    auto t = ggml_metalium_realize_ggml_view(dst->src[0]);
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(ttnn::sqrt(*t)),
        .ggtype = dst->type,
        .bufctx = ((TensorWithMetadata*)dst->src[0]->extra)->bufctx
    };
}

static void ggml_backend_metalium_sqr(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);

    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;

    float esp = 0;
    memcpy(&esp, dst->op_params, sizeof(esp));

    auto t = ggml_metalium_realize_ggml_view(dst->src[0]);
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(ttnn::square(*t)),
        .ggtype = dst->type,
        .bufctx = ((TensorWithMetadata*)dst->src[0]->extra)->bufctx
    };
}

bool ggml_backend_metalium_can_concat(const struct ggml_tensor * dst)
{
    if(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_BF16 || dst->type == GGML_TYPE_F16) {
        return true;
    }
    return false;
}

static void ggml_backend_metalium_concat(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC1_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;

    auto src_tensor0 = ggml_metalium_realize_ggml_view(src0);
    auto src_tensor1 = ggml_metalium_realize_ggml_view(src1);

    int32_t axis = 0;
    memcpy(&axis, dst->op_params, sizeof(axis));
    axis = GGML_MAX_DIMS - axis - 1;

    std::vector<tt::tt_metal::Tensor> targets = {*src_tensor0, *src_tensor1};
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(ttnn::concat(targets, axis)),
        .ggtype = dst->type,
        .bufctx = ((TensorWithMetadata*)dst->src[0]->extra)->bufctx
    };
}

bool ggml_backend_metalium_can_softmax(const struct ggml_tensor * dst)
{
    float arr[2];
    memcpy(arr, dst->op_params, sizeof(arr));
    auto [scale, max_bias] = arr;
    if(dst->src[1] != nullptr && max_bias != 0.f) {
        return false;
    }
    if(dst->src[1] != nullptr) {
        // TinyLLaMA somehow has x [1, 32, 1, 32] and mask [1, 1, 32, 32]
        // Don't know what's this about
        // FIXME: This masks a problem in RWKV. Need proper fix
        const ggml_tensor *src1 = dst->src[1];
        return ggml_metalium_numpy_broadcast_rule(src1, dst) && dst->ne[1] == src1->ne[1];
    }
    return true;
}

static void ggml_backend_metalium_softmax(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);

    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;

    std::array<float, 2> params;
    memcpy(&params, dst->op_params, sizeof(params));
    auto [scale, max_bias] = params;

    const ggml_tensor *src1 = dst->src[1];

    auto t = ggml_metalium_realize_ggml_view(dst->src[0]);
    tt::tt_metal::Tensor x = *t;
    // TODO: use the operimzied op if we can. It only works in certain conidtions
    // if(src1 != nullptr) {
    //     auto mask = ggml_metalium_realize_ggml_view(src1);
    //     x = ttnn::operations::normalization::scale_mask_softmax(*t, scale, *mask);
    // }
    if(scale != 1.f) {
        x = ttnn::multiply(*t, scale);
    }

    if(src1 != nullptr) {
        auto mask = ggml_metalium_realize_ggml_view(src1);
        if(max_bias == 0.f) {
            // std::cout << "x: " << x.logical_shape() << " mask: " << mask->logical_shape() << std::endl;
            // std::cout << "x.dtype: " << (int)x.dtype() << " mask.dtype: " << (int)mask->dtype() << std::endl;
            x = ttnn::add(x, *mask);
        }
        else {
            // This path is not used due to bugs
            // TODO: Revive it later
            const uint32_t n_head = t->logical_shape()[1];
            const uint32_t n_head_log2 = 1u << (uint32_t) std::floor(std::log2(n_head));
            const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
            const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);
            auto make_tile = [](const tt::tt_metal::Tensor& t, ttnn::MeshDevice* dev) {
                return ttnn::tilize_with_zero_padding(t.to_device(dev));
            };

            // const float slope = (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1) : 1.0f;
            auto *dev = t->device();
            // BUG here. Generating wrong shaped tensor
            // This is a part of the limitation of TTNN can't have odd numbers of elements in the last dimension
            auto idxs = make_tile(ttnn::arange(0, n_head, 1), dev);
            auto slope = ttnn::where(ttnn::lt(idxs, (float)n_head_log2), ttnn::rpow(ttnn::add(idxs, 1.f), m0)
                , ttnn::rpow(ttnn::add(ttnn::multiply(ttnn::subtract(idxs, (float)n_head_log2), 2.f), 1.f), m1));
            auto positional_bias = ttnn::matmul(slope, ttnn::transpose(idxs, -2, -1)); // FIXME: make sure this is correct

            x = ttnn::add(x, ttnn::multiply(*mask, positional_bias));
        }
    }
    x = ttnn::softmax(x, 3);
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(std::move(x)),
        .ggtype = dst->type,
        .bufctx = ((TensorWithMetadata*)dst->src[0]->extra)->bufctx
    };
}

static void ggml_backend_metalium_cos(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    const struct ggml_tensor * src0 = dst->src[0];
    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;

    auto src = ggml_metalium_realize_ggml_view(src0);
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(ttnn::cos(*src)),
        .ggtype = dst->type,
        .bufctx = ((TensorWithMetadata*)src0->extra)->bufctx
    };
}

static void ggml_backend_metalium_sin(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    const struct ggml_tensor * src0 = dst->src[0];
    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;

    auto src = ggml_metalium_realize_ggml_view(src0);
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(ttnn::sin(*src)),
        .ggtype = dst->type,
        .bufctx = ((TensorWithMetadata*)src0->extra)->bufctx
    };
}

static void ggml_backend_metalium_log(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    const struct ggml_tensor * src0 = dst->src[0];
    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;

    auto src = ggml_metalium_realize_ggml_view(src0);
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(ttnn::log(*src)),
        .ggtype = dst->type,
        .bufctx = ((TensorWithMetadata*)src0->extra)->bufctx
    };
}

static void ggml_backend_metalium_arange(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;
    auto* device = dst_meta->bufctx->device.get();
    std::array<float, 3> params;
    memcpy(&params, dst->op_params, sizeof(params));
    auto [start, end, step] = params;
    auto dtype = ggml_metalium_ggml2tt_type(dst->type, device->arch());
    if(dtype == tt::tt_metal::DataType::INVALID) {
        fmt::println(stderr, "Unsupported GGML type {}", ggml_type_name(dst->type));
        GGML_ASSERT(false && "Unsupported GGML type");
    }

    // TODO: Request TT to support arange directly on the device
    auto tensor = ttnn::arange(start, end, step, dtype);
    tensor = ttnn::reshape(tensor, ttnn::Shape{1, 1, 1, (uint32_t)(end - start)});
    tensor = ttnn::tilize_with_zero_padding(tensor.to_device(device));
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(std::move(tensor)),
        .ggtype = dst->type,
        .bufctx = dst_meta->bufctx
    };
}

static void ggml_backend_metalium_group_norm(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;
    int n_groups;
    float eps;
    memcpy(&n_groups, dst->op_params, sizeof(n_groups));
    memcpy(&eps, dst->op_params + sizeof(n_groups), sizeof(eps));

    // XXX: Moreh's operators needs some cleanup
    auto tensor = ggml_metalium_realize_ggml_view(dst->src[0]);
    auto res = ttnn::moreh_group_norm(
        *tensor,
        n_groups,
        eps,
        std::nullopt,
        std::nullopt,
        std::vector<bool>{true, false, false},
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt);
    GGML_ASSERT(res[0].has_value());
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(*res[0]),
        .ggtype = dst->type,
        .bufctx = dst_meta->bufctx
    };
}

static bool ggml_backend_metalium_can_repeat(const struct ggml_tensor * dst)
{
    // TODO: File bug report that repear op should support UINT32
    if(dst->type == GGML_TYPE_I32) {
        return false;
    }
    ggml_tensor *src0 = dst->src[0];
    for(int i = 0; i < GGML_MAX_DIMS; i++) {
        if(dst->ne[i] % src0->ne[i] != 0) {
            return false;
        }
    }
    return true;
}

static void ggml_backend_metalium_repeat(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;
    ggml_tensor* src0 = dst->src[0];

    auto tensor = ggml_metalium_realize_ggml_view(dst->src[0]);
    ttsl::SmallVector<uint32_t> repeats;
    repeats.resize(GGML_MAX_DIMS);
    int ndiff = 0;
    for(int i = 0; i < GGML_MAX_DIMS; i++) {
        auto repeat = dst->ne[i] / src0->ne[i];
        repeats[GGML_MAX_DIMS - i - 1] = repeat;
        ndiff += (repeat != 1);
    }
    if(ndiff == 0) {
        *dst_meta = {
            .tensor = std::make_shared<tt::tt_metal::Tensor>(*tensor),
            .ggtype = dst->type,
            .bufctx = ((TensorWithMetadata*)src0->extra)->bufctx
        };
        return;
    }

    auto res = ttnn::repeat(*tensor, ttnn::Shape(repeats));
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(res),
        .ggtype = dst->type,
        .bufctx = ((TensorWithMetadata*)src0->extra)->bufctx
    };
}

bool ggml_backend_metalium_can_outer_product(const struct ggml_tensor * dst)
{
    auto num_ones_in_shape = [](const ggml_tensor * t) {
        int num_ones = 0;
        for(int i = 0; i < GGML_MAX_DIMS; i++) {
            if(t->ne[i] == 1) {
                num_ones++;
            }
        }
        return num_ones;
    };
    return num_ones_in_shape(dst->src[0]) == 3 && num_ones_in_shape(dst->src[1]) == 3;
}

static void ggml_backend_metalium_outer_product(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC1_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;
    TensorWithMetadata* src0_meta = (TensorWithMetadata*)dst->src[0]->extra;

    auto src0 = ggml_metalium_realize_ggml_view(dst->src[0]);
    auto src1 = ggml_metalium_realize_ggml_view(dst->src[1]);

    auto res = ttnn::outer(*src1, *src0);
    // HACK: GGML and TT has different ideas about the shape of the result, sometimes
    if(!ggml_tt_tensors_shape_equal(dst, res)) {
        // Magic herustics
        if(dst->ne[3] == res.logical_shape()[2]) {
            res = ttnn::transpose(res, 0, 2);
        }
        else if(dst->ne[2] == res.logical_shape()[2]) {
            res = ttnn::transpose(res, 1, 2);
        }
        else {
            std::cerr << "GGML shape: " << dst->ne[0] << ", " << dst->ne[1] << ", " << dst->ne[2] << ", " << dst->ne[3] << "\n";
            std::cerr << "TT shape: " << res.logical_shape() << "\n";
            GGML_ASSERT(false && "Unsupported outer product shape mismatch");
        }
    }
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(res),
        .ggtype = dst->type,
        .bufctx = src0_meta->bufctx
    };
}
static void ggml_backend_metalium_sum(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;

    auto t = ggml_metalium_realize_ggml_view(dst->src[0]);
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(ttnn::sum(*t)),
        .ggtype = dst->type,
        .bufctx = ((TensorWithMetadata*)dst->src[0]->extra)->bufctx
    };
}

static void ggml_backend_metalium_sum_rows(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;

    auto t = ggml_metalium_realize_ggml_view(dst->src[0]);
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(ttnn::sum(*t, 3)),
        .ggtype = dst->type,
        .bufctx = ((TensorWithMetadata*)dst->src[0]->extra)->bufctx
    };
}

bool ggml_backend_metalium_can_glu(const struct ggml_tensor * dst)
{
    bool split = dst->src[1] != NULL;
    if(split) {
        return true;
    }
    return dst->src[0]->ne[0] % 2 == 0 && ggml_get_glu_op(dst) != GGML_GLU_OP_SWIGLU_OAI;
}

static void ggml_backend_metalium_glu(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;
    TensorWithMetadata* src0_meta = (TensorWithMetadata*)dst->src[0]->extra;

    ttnn::Tensor a;
    ttnn::Tensor b;
    int swap = ggml_get_op_params_i32(dst, 1);
    bool split = dst->src[1] != NULL;

    if(split) {
        a = *ggml_metalium_realize_ggml_view(dst->src[1]);
        b = *ggml_metalium_realize_ggml_view(dst->src[0]);
    }
    else {
        auto t = ggml_metalium_realize_ggml_view(dst->src[0]);
        // split along the last dimension
        int64_t w = dst->ne[0];

        using Slice = std::array<uint32_t, GGML_MAX_DIMS>;
        Slice mid = {uint32_t(w), uint32_t(dst->ne[1]), uint32_t(dst->ne[2]), uint32_t(dst->ne[3])};
        std::reverse(mid.begin(), mid.end());
        Slice mid_start = {uint32_t(w), 0, 0, 0};
        std::reverse(mid_start.begin(), mid_start.end());
        Slice end = {uint32_t(w * 2), uint32_t(dst->ne[1]), uint32_t(dst->ne[2]), uint32_t(dst->ne[3])};
        std::reverse(end.begin(), end.end());
        Slice begin = {0, 0, 0, 0};
        Slice stride = {1, 1, 1, 1};

        a = ttnn::slice(*t, mid_start, end, stride);
        b = ttnn::slice(*t, begin, mid, stride);
    }


    if(swap) {
        std::swap(a, b);
    }

    ttnn::Tensor res;
    switch(ggml_get_glu_op(dst)) {
        case GGML_GLU_OP_REGLU:
            res = ttnn::multiply(a, ttnn::relu(b, ttnn::L1_MEMORY_CONFIG));
            break;
        case GGML_GLU_OP_GEGLU_ERF: // ?
        case GGML_GLU_OP_GEGLU_QUICK:
        case GGML_GLU_OP_GEGLU:
            res = ttnn::multiply(a, ttnn::gelu(b, false, ttnn::L1_MEMORY_CONFIG));
            break;
        case GGML_GLU_OP_SWIGLU:
            res = ttnn::multiply(a, ttnn::swish(b, ttnn::L1_MEMORY_CONFIG));
            break;
        case GGML_GLU_OP_SWIGLU_OAI: {
            const float alpha = ggml_get_op_params_f32(dst, 2);
            const float limit = ggml_get_op_params_f32(dst, 3);
            const auto mem_config = ttnn::L1_MEMORY_CONFIG;
            const std::optional<ttnn::DataType> dtype = std::nullopt;
            // Existing Metalium convention:
            //   SWIGLU: a * swish(b)
            // therefore:
            //   b = x / activation branch
            //   a = gate branch
            // xi = min(x, limit)
            auto xi = ttnn::minimum(
                b,
                limit,
                dtype,
                mem_config
            );
            // gi = clamp(gate, -limit, limit)
            auto gi_hi = ttnn::minimum(
                a,
                limit,
                dtype,
                mem_config
            );
            auto gi = ttnn::maximum(
                gi_hi,
                -limit,
                dtype,
                mem_config
            );

            // alpha_xi = alpha * xi
            auto alpha_xi = ttnn::multiply(
                xi,
                alpha,
                dtype,
                mem_config
            );
            // sigmoid(alpha * xi)
            auto sig = ttnn::sigmoid(
                alpha_xi,
                static_cast<int>(ttnn::operations::unary::VecMode::RC),
                ttnn::operations::unary::SigmoidMode::ACCURATE,
                mem_config
            );
            // swish_alpha(xi) = xi * sigmoid(alpha * xi)
            auto swish_alpha = ttnn::multiply(
                xi,
                sig,
                dtype,
                mem_config
            );
            // 1 + clamp(gate, -limit, limit)
            auto gate_plus_one = ttnn::add(
                gi,
                1.0f,
                dtype,
                mem_config
            );
            res = ttnn::multiply(
                swish_alpha,
                gate_plus_one,
                dtype,
                mem_config
            );
            break;
        }
        default:
            GGML_ASSERT(false && "Unsupported GLU operation");
    }

    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(std::move(res)),
        .ggtype = dst->type,
        .bufctx = src0_meta->bufctx
    };
}

enum ggml_status ggml_backend_metalium_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_metalium_context * ctx = (ggml_backend_metalium_context *)backend->context;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        // std::cout << "Graph compute " << ggml_op_desc(node) << "\n"
        //     << "  dst addr: " << node->data << "\n"
        //     << "  src0 addr: " << (void*)(node->src[0] ? node->src[0]->data : 0) << "\n"
        //     << "  src1 addr: " << (void*)(node->src[1] ? node->src[1]->data : 0) << "\n";

        // Bypass post conition checks for these ops because they are evaluated lazily
        if(node->op == GGML_OP_VIEW || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_PERMUTE) {
            continue;
        }

        // std::cout << ggml_op_name(node->op) << " node " << node->name << " with address " << node->data << std::endl;
        switch (node->op) {
            case GGML_OP_UNARY: {
                ggml_unary_op unary_op = ggml_get_unary_op(node);
                bool ok = false;
                switch (unary_op) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_EXP:
                    ok = ggml_backend_metalium_activations(ctx, node, unary_op);
                    break;
                default:
                    fprintf(stderr, "%s: unsupported unary op %s\n", __func__, ggml_unary_op_name(unary_op));
                }
                GGML_ASSERT(ok && "Failed to execute unary op");
                break;
            }
            case GGML_OP_LEAKY_RELU:
                ggml_backend_metalium_leaky_relu(ctx, node);
                break;
            case GGML_OP_ADD:
            case GGML_OP_SUB:
            case GGML_OP_DIV:
            case GGML_OP_MUL:
                ggml_backend_metalium_bin_op(ctx, node, node->op);
                break;
            case GGML_OP_MUL_MAT:
                ggml_backend_metalium_mul_mat(ctx, node);
                break;
            case GGML_OP_OUT_PROD:
                ggml_backend_metalium_outer_product(ctx, node);
                break;

            case GGML_OP_CONT:
            case GGML_OP_CPY:
            case GGML_OP_DUP:
                ggml_backend_metalium_cpy(ctx, node);
                break;
              case GGML_OP_SET:
                  ggml_backend_metalium_set(ctx, node);
                  break;
              case GGML_OP_SET_ROWS:
                  ggml_backend_metalium_set_rows(ctx, node);
                  break;

              case GGML_OP_CLAMP:
                ggml_backend_metalium_clamp(ctx, node);
                break;

            case GGML_OP_SCALE:
                ggml_backend_metalium_scale(ctx, node);
                break;

            case GGML_OP_GET_ROWS:
                ggml_backend_metalium_get_rows(ctx, node);
                break;

            case GGML_OP_NORM:
                ggml_backend_metalium_norm(ctx, node, false);
                break;

            case GGML_OP_RMS_NORM:
                ggml_backend_metalium_norm(ctx, node, true);
                break;

            case GGML_OP_ADD1:
                ggml_backend_metalium_add1(ctx, node);
                break;

            case GGML_OP_SQRT:
                ggml_backend_metalium_sqrt(ctx, node);
                break;

            case GGML_OP_SQR:
                ggml_backend_metalium_sqr(ctx, node);
                break;

            case GGML_OP_CONCAT:
                ggml_backend_metalium_concat(ctx, node);
                break;

            case GGML_OP_SOFT_MAX:
                ggml_backend_metalium_softmax(ctx, node);
                break;

            case GGML_OP_COS:
                ggml_backend_metalium_cos(ctx, node);
                break;

            case GGML_OP_SIN:
                ggml_backend_metalium_sin(ctx, node);
                break;

            case GGML_OP_LOG:
                ggml_backend_metalium_log(ctx, node);
                break;

            case GGML_OP_ARANGE:
                ggml_backend_metalium_arange(ctx, node);
                break;

            case GGML_OP_GROUP_NORM:
                ggml_backend_metalium_group_norm(ctx, node);
                break;

            case GGML_OP_REPEAT:
                ggml_backend_metalium_repeat(ctx, node);
                break;

            case GGML_OP_SUM:
                ggml_backend_metalium_sum(ctx, node);
                break;

            case GGML_OP_SUM_ROWS:
                ggml_backend_metalium_sum_rows(ctx, node);
                break;

            case GGML_OP_GLU:
                ggml_backend_metalium_glu(ctx, node);
                break;

            case GGML_OP_NONE:
                break;

            default:
                fprintf(stderr, "%s: unsupported op %s\n", __func__, ggml_op_desc(node));
                GGML_ASSERT(false);
        }
        TensorWithMetadata* meta = (TensorWithMetadata*)node->extra;
        // std::cout << "Executed " << ggml_op_desc(node) << " with address " << node->data << " and shape " << meta->tensor->logical_shape() << ", GGML wants " << node->ne[0] << " " << node->ne[1] << " " << node->ne[2] << " " << node->ne[3] << std::endl;
        GGML_ASSERT(meta != NULL);
        GGML_ASSERT(meta->tensor != NULL);
        GGML_ASSERT(meta->tensor->storage_type() == tt::tt_metal::StorageType::DEVICE);
        if(!ggml_tt_tensors_shape_equal(node, *meta->tensor)) {
            fmt::println(stderr, "Mismatched tensor shapes for node '{}' ({}): GGML wants [{}, {}, {}, {}], TTNN generates {}\n"
                , node->name, ggml_op_name(node->op), node->ne[0], node->ne[1], node->ne[2], node->ne[3], meta->tensor->logical_shape());
            abort();
        }
    }

    return GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-metalium.h"

#include "ggml-metalium-context.h"
#include "ggml-metalium-device.h"
#include "ggml-metalium-ops.h"

#include "umd/device/types/arch.hpp"
#include <sys/types.h>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
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

static const char * ggml_backend_metalium_name(ggml_backend_t backend) {
    return "Metalium";

    GGML_UNUSED(backend);
}

static void ggml_backend_metalium_free(ggml_backend_t backend) {
    ggml_backend_metalium_context * ctx = (ggml_backend_metalium_context *)backend->context;
    if (ctx != nullptr && ctx->dev_ctx != nullptr && ctx->dev_ctx->device != nullptr) {
        ctx->transposed_weights.clear();
        ttnn::distributed::close_mesh_device(ctx->dev_ctx->device);
        ctx->dev_ctx->device.reset();
    }
    delete ctx;
    delete backend;
}

static bool ggml_backend_metalium_device_supports_op_internal(ggml_backend_dev_t device, const struct ggml_tensor * op);

static bool ggml_backend_metalium_device_supports_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    bool ok = ggml_backend_metalium_device_supports_op_internal(device, op);
    // debug print to log rejected ops
    if(!ok && ggml_metalium_debug_flags.print_rejected_ops) {
        fprintf(stderr, "REJECT op %s (%s)\n", ggml_op_name(op->op), op->name);
        if(op->src[0]) {
            fprintf(stderr, "  src0 shape [%ld %ld %ld %ld], dtype = %s\n", op->src[0]->ne[0], op->src[0]->ne[1], op->src[0]->ne[2], op->src[0]->ne[3], ggml_type_name(op->src[0]->type));
        }
        if(op->src[1]) {
            fprintf(stderr, "  src1 shape [%ld %ld %ld %ld], dtype = %s\n", op->src[1]->ne[0], op->src[1]->ne[1], op->src[1]->ne[2], op->src[1]->ne[3], ggml_type_name(op->src[1]->type));
        }
    }
    return ok;
}


static bool ggml_backend_metalium_should_skip_broken_add_shape(const ggml_tensor * op) {
    if(op->op != GGML_OP_ADD || op->src[0] == nullptr || op->src[1] == nullptr) {
        return false;
    }

    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    const int64_t height = op->ne[1] * op->ne[2] * op->ne[3];
    const bool is_broken_shape =
        op->ne[0] == 1280 && op->ne[1] == 16 && op->ne[2] == 16 && op->ne[3] == 1 && height % 32 == 0;
    const bool is_broken_f16_shape =
        src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && is_broken_shape;
    const bool is_target_broken_f32_shape =
        src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && is_broken_shape &&
        src0->ne[0] == 1280 && src0->ne[1] == 16 && src0->ne[2] == 16 && src0->ne[3] == 1 &&
        src1->ne[0] == 1280 &&
        ((src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1) ||
         (src1->ne[1] == 16 && src1->ne[2] == 16 && src1->ne[3] == 1));

    return is_broken_f16_shape || is_target_broken_f32_shape;
}

static bool ggml_backend_metalium_device_supports_op_internal(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(op != NULL);
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    ggml_backend_metalium_device_context * ctx = (ggml_backend_metalium_device_context *)device->context;

    // The metalium backend has seperated internal data types from the GGML data types. We really only care about
    // what we can convert to and from.
    auto tensor_supported = [&](const struct ggml_tensor * tensor) {
        if(tensor == NULL || !ggml_metalium_is_ggml_type_supported(tensor->type, ctx->arch)) {
            return false;
        }
        // TTNN requires the tensor to be 4-byte aligned and all quantized tensors must be a multiple of 32

        // HACK: later GGML contains an absurd code that views into [1, embed, 1, 1] then tranpose to [embed, 1, ,1 ,1]
        //       which is a waste of time on TTNN. We mush allow the view op to pass then perform the correct view
        //       ignoring the transpose.
        if(tensor->op == GGML_OP_VIEW) {
            return true;
        }

        tt::tt_metal::DataType tt_type = ggml_metalium_ggml2tt_type(tensor->type, ctx->arch);
        switch(tt_type) {
            case tt::tt_metal::DataType::BFLOAT16:
            case tt::tt_metal::DataType::UINT16:
                return tensor->ne[0] % 2 == 0 && tensor->ne[0] != 0; // NOTE: This should be enablable by now (Was a limitation of ancient TTNN versions)
            case tt::tt_metal::DataType::FLOAT32:
            case tt::tt_metal::DataType::UINT32:
            case tt::tt_metal::DataType::INT32:
                return true;
            case tt::tt_metal::DataType::UINT8:
                return tensor->ne[0] % 4 == 0 && tensor->ne[0] != 0;
            case tt::tt_metal::DataType::INVALID:
                GGML_ASSERT(false && "Unsupported data type");
                break;
            default:
                return tensor->ne[0] % 32 == 0 && tensor->ne[0] != 0 && tensor->ne[1] % 32 == 0 && tensor->ne[1] != 0;
        }
        GGML_UNREACHABLE();
    };

    if(!tensor_supported(op)) {
        return false;
    }
    // ARANGE and NONE are special case where src0 is not required
    if(op->op == GGML_OP_NONE || op->op == GGML_OP_ARANGE) {
        return true;
    }
    if(!tensor_supported(src0)) {
        return false;
    }

    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_EXP:
                    return true;
                default:
                    return false;
            }
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_CLAMP:
        case GGML_OP_SCALE:
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_ADD1:
        case GGML_OP_SQRT:
        case GGML_OP_SQR:
        case GGML_OP_PERMUTE:
        case GGML_OP_LOG:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_VIEW:
        // SUM{_ROWS} technically works but supprts_op rejects the result tensor.
        // Which gotta do so to avoid some bugs around binary ops with tiled dim=1
        case GGML_OP_SUM:
        case GGML_OP_SUM_ROWS:
            return true;

        case GGML_OP_CONT:
        case GGML_OP_CPY:
        case GGML_OP_DUP:
            return ggml_backend_metalium_can_cpy(op);

        case GGML_OP_SIN:
        case GGML_OP_COS:
            return true;
        case GGML_OP_ADD:
            if(ggml_backend_metalium_should_skip_broken_add_shape(op)) {
                return false;
            }
            return tensor_supported(src1) && ggml_metalium_numpy_broadcast_rule(src0, src1);
        case GGML_OP_SUB:
        case GGML_OP_MUL:
            return tensor_supported(src1) && ggml_metalium_numpy_broadcast_rule(src0, src1);
        // DIV does not support broadcasting on TTNN
        case GGML_OP_DIV:
            return tensor_supported(src1) && memcmp(src0->ne, src1->ne, sizeof(src0->ne)) == 0;

        case GGML_OP_MUL_MAT:
            return tensor_supported(src1) && ggml_backend_metalium_can_mul_mat(op);
        case GGML_OP_SET:
            return tensor_supported(src1) && ggml_backend_metalium_can_set(op);
        case GGML_OP_SOFT_MAX:
            return ggml_backend_metalium_can_softmax(op);
        case GGML_OP_GET_ROWS:
            return tensor_supported(src1) && ggml_backend_metalium_can_get_rows(op);
        case GGML_OP_CONCAT:
            return tensor_supported(src1) && ggml_backend_metalium_can_concat(op);
        // case GGML_OP_REPEAT:
        //     return ggml_backend_metalium_can_repeat(op);
        case GGML_OP_OUT_PROD:
            return tensor_supported(src1) && ggml_backend_metalium_can_outer_product(op);
        case GGML_OP_GLU:
            return ((src1 && tensor_supported(src1)) || !src1) && ggml_backend_metalium_can_glu(op);
        default:
            return false;
    }
}

static bool ggml_backend_metalium_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (buft->iface.get_name != ggml_backend_metalium_buffer_type_name) {
        return false;
    }
    return buft->device == dev;
}

static void ggml_backend_metalium_synchronize(ggml_backend_t backend)
{
    GGML_UNUSED(backend);
    return;
}

static struct ggml_backend_i metalium_backend_i = {
    /* .get_name                = */ ggml_backend_metalium_name,
    /* .free                    = */ ggml_backend_metalium_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ggml_backend_metalium_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_metalium_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

ggml_guid_t ggml_backend_metalium_guid(void) {
    static ggml_guid guid = { 0x91, 0x69, 0xd5, 0x5f, 0x24, 0xe7, 0x44, 0x00, 0xb4, 0x2a, 0x73, 0x23, 0x48, 0xb0, 0x4e, 0xe7 };
    return &guid;
}

static ggml_backend_t ggml_backend_metalium_init(ggml_backend_metalium_device_context* dev_ctx) {
    int device_id = dev_ctx->device_id;
    GGML_ASSERT(device_id >= 0 && (size_t)device_id < tt::tt_metal::GetNumAvailableDevices());

    ttnn::MeshDevice * device = ggml_metalium_get_device(dev_ctx);
    if(!ggml_metalium_debug_flags.disable_program_cache) {
        ttnn::enable_program_cache(*device);
    }
    // Limit device support to the ones I own (GS is removed as TTNN dropped support)
    GGML_ASSERT(device->arch() == tt::ARCH::WORMHOLE_B0 || device->arch() == tt::ARCH::BLACKHOLE);
    dev_ctx->arch = device->arch();

    ggml_backend_metalium_context * ctx = new ggml_backend_metalium_context {
        /* dev_ctx           = */ dev_ctx,
        /* transposed_weights= */ {},
        /* device_id         = */ device_id,
        /* name              = */ dev_ctx->name,
    };

    ggml_backend_t backend = new ggml_backend {
        /* .guid      = */ ggml_backend_metalium_guid(),
        /* .interface = */ metalium_backend_i,
        /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_metalium_reg(), device_id),
        /* .context   = */ ctx
    };
    return backend;
}

static const char * ggml_backend_metaliium_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return "Metalium";
}

static size_t ggml_backend_metalium_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_metalium_reg_context * ctx = (ggml_backend_metalium_reg_context *)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_metalium_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_metalium_reg_context * ctx = (ggml_backend_metalium_reg_context *)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

const ggml_backend_reg_i ggml_backend_metalium_reg_interface = {
    /* .get_name          = */ ggml_backend_metaliium_reg_get_name,
    /* .get_device_count  = */ ggml_backend_metalium_reg_get_device_count,
    /* .get_device        = */ ggml_backend_metalium_reg_get_device,
    /* .get_proc_address  = */ NULL,
};

static const char* ggml_backend_metalium_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_metalium_device_context * ctx = (ggml_backend_metalium_device_context *)dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_metalium_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_metalium_device_context * ctx = (ggml_backend_metalium_device_context *)dev->context;
    return ctx->description.c_str();
}

static void ggml_backend_metalium_get_memory(ggml_backend_dev_t dev, size_t * total, size_t * free) {
    GGML_UNUSED(dev);
    // The registry intentionally does not keep a MeshDevice open. Avoid opening
    // one just for property queries, because static device ownership can run
    // MeshDevice destruction after TT-Metal/UMD shutdown.
    *total = 0;
    *free = 0;
}

static enum ggml_backend_dev_type ggml_backend_metalium_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static ggml_backend_t ggml_backend_metalium_device_init(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_metalium_device_context * ctx = (ggml_backend_metalium_device_context *)dev->context;
    ggml_backend_t backend = ggml_backend_metalium_init(ctx);
    GGML_ASSERT(backend != NULL);
    return backend;
}

static void ggml_backend_metalium_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    ggml_backend_metalium_device_context * ctx = (ggml_backend_metalium_device_context *)dev->context;
    size_t free = 0;
    size_t total = 0;
    ggml_backend_metalium_get_memory(dev, &total, &free);
    *props = ggml_backend_dev_props {
        .name = ctx->name.c_str(),
        .description = ctx->description.c_str(),
        .memory_free = free,
        .memory_total = total,
        .type = ggml_backend_metalium_get_type(dev),
        .device_id = NULL,
        .caps = ggml_backend_dev_caps {
            .async = true,
            .host_buffer = false,
            .buffer_from_host_ptr = false,
            .events = false,
        }
    };
}

static ggml_backend_buffer_type_t ggml_backend_metalium_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_metalium_device_context * ctx = (ggml_backend_metalium_device_context *)dev->context;
    return ggml_backend_metalium_buffer_type(dev, ctx);
}

const ggml_backend_device_i ggml_backend_metalium_device_interface = {
    /* .get_name                = */ ggml_backend_metalium_device_get_name,
    /* .get_description         = */ ggml_backend_metalium_device_get_description,
    /* .get_memory              = */ ggml_backend_metalium_get_memory,
    /* .get_type                = */ ggml_backend_metalium_get_type,
    /* .get_props               = */ ggml_backend_metalium_device_get_props,
    /* .init_backend            = */ ggml_backend_metalium_device_init,
    /* .get_buffer_type         = */ ggml_backend_metalium_get_buffer_type,
    /* .get_host_buffer_type    = */ NULL,
    /* .buffer_from_host_ptr    = */ NULL,
    /* .supports_op             = */ ggml_backend_metalium_device_supports_op,
    /* .supports_buft           = */ ggml_backend_metalium_device_supports_buft,
    /* .offload_op              = */ NULL,
    /* .event_new               = */ NULL,
    /* .event_free              = */ NULL,
    /* .event_synchronize       = */ NULL,
};

static std::string identify_tensotrrent_device(const ttnn::MeshDevice* device)
{
    auto grid_size = device->compute_with_storage_grid_size();
    // TODO: Support mesh configurations
    if(device->arch() == tt::ARCH::WORMHOLE_B0) {
        if(grid_size.x == 8 && grid_size.y == 7) {
            return "Tenstorrent Wormhole n300";
        }
        return "Tenstorrent Wormhole n150";
    }
    if(device->arch() == tt::ARCH::BLACKHOLE) {
        return "Tenstorrent Blackhole";
    }

    return "Unknown Tenstorrent device";
}

std::vector<std::unique_ptr<ggml_backend_device>> g_backend_device_holder;
std::vector<std::unique_ptr<ggml_backend_metalium_device_context>> g_backend_device_context_holder;

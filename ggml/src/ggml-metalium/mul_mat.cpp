#include "mul_mat.hpp"
#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/kernel_types.hpp"
#include "tt-metalium/tt_backend_api_types.hpp"
#include <ttnn/device_operation.hpp>
#include <ttnn/tensor/layout/layout.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "tt_stl/assert.hpp"
#include "ttnn/types.hpp"
#include "utils.hpp"

using namespace tt::tt_metal;

struct MulMatDeviceOperation {
    using operation_attributes_t = MulMatDeviceOperation;
    struct tensor_args_t {
        const Tensor & a;
        const Tensor & b;
    };
    using spec_return_value_t = std::vector<ttnn::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;

    const tt::tt_metal::MemoryConfig output_mem_config;
    const tt::tt_metal::DataType output_dtype{};
    const bool high_percision = false;

    struct ProgramFactory {
        struct shared_variables_t {
            KernelHandle reader;
            KernelHandle writer;
            CoreRangeSet all_cores;
        };
        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t & operation_attributes,
            const tensor_args_t & tensor_args,
            tensor_return_value_t & output_tensors);

        static void override_runtime_arguments(
            cached_program_t & cached_program,
            const operation_attributes_t & operation_attributes,
            const tensor_args_t & tensor_args,
            tensor_return_value_t & output_tensors);
    };
    using program_factory_t = std::variant<ProgramFactory>;

    static void validate_on_program_cache_miss(
        const operation_attributes_t & operation_attributes,
        const tensor_args_t & tensor_args);
    static spec_return_value_t compute_output_specs(
        const operation_attributes_t & operation_attributes,
        const tensor_args_t & tensor_args);
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t & operation_attributes,
        const tensor_args_t & tensor_args);
};

ttnn::Tensor ttggml::MulMatOperation::invoke(const Tensor & a, const Tensor & b, bool high_percision) {
    return ttnn::device_operation::launch<MulMatDeviceOperation>(
        MulMatDeviceOperation{
            b.memory_config(),
            b.dtype(),
            high_percision,
        },
        MulMatDeviceOperation::tensor_args_t{a, b})[0];
}

std::vector<ttnn::TensorSpec> MulMatDeviceOperation::compute_output_specs(
    const operation_attributes_t & operation_attributes, const tensor_args_t & tensor_args) {
    const auto & a = tensor_args.a;
    const auto & b = tensor_args.b;
    ttnn::Shape output_shape({
        std::max(a.logical_shape()[0], b.logical_shape()[0]),
        std::max(a.logical_shape()[1], b.logical_shape()[1]),
        b.logical_shape()[2],
        a.logical_shape()[2],
    });
    return {TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(
            operation_attributes.output_dtype,
            tt::tt_metal::PageConfig(ttnn::TILE_LAYOUT),
            operation_attributes.output_mem_config))};
}

std::vector<ttnn::Tensor> MulMatDeviceOperation::create_output_tensors(
    const operation_attributes_t & operation_attributes, const tensor_args_t & tensor_args) {
    const auto & input_tensor = tensor_args.a;
    auto spec = compute_output_specs(operation_attributes, tensor_args)[0];
    return {create_device_tensor(spec, input_tensor.device())};
}

void MulMatDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t &, const tensor_args_t & tensor_args) {
    const auto & a = tensor_args.a;
    const auto & b = tensor_args.b;
    const auto & a_shape = a.logical_shape();
    const auto & b_shape = b.logical_shape();
    const auto & a_shape4d = a.logical_shape().to_array_4D();
    const auto & b_shape4d = b.logical_shape().to_array_4D();

    TT_FATAL(a.layout() == ttnn::TILE_LAYOUT, "Expected layout TILE_LAYOUT for tensor a");
    TT_FATAL(b.layout() == ttnn::TILE_LAYOUT, "Expected layout TILE_LAYOUT for tensor b");

    TT_FATAL(
        a_shape4d[3] == b_shape4d[3] && b_shape4d[1] % a_shape4d[1] == 0 && b_shape4d[1] > 0 &&
            b_shape4d[0] % a_shape4d[0] == 0 && b_shape4d[0] > 0 && a_shape4d[2] > 0 && b_shape4d[2] > 0,
        "Expcted format a: [B, N, M, K], b: [B*x, C*x, N, K] but get a: {}, b: {}",
        a_shape,
        b_shape);
}

MulMatDeviceOperation::ProgramFactory::cached_program_t MulMatDeviceOperation::ProgramFactory::create(
    const operation_attributes_t & operation_attributes,
    const tensor_args_t & tensor_args,
    tensor_return_value_t & output_tensors) {
    tt::tt_metal::Program program{};
    const auto & a_tensor = tensor_args.a;
    const auto & b_tensor = tensor_args.b;
    const auto & o_tensor = output_tensors.at(0);

    const uint32_t K = a_tensor.logical_shape()[-1];
    const uint32_t N = b_tensor.logical_shape()[2];
    const uint32_t M = a_tensor.logical_shape()[2];
    const uint32_t C = a_tensor.logical_shape()[1];
    const uint32_t B = a_tensor.logical_shape()[0];
    const uint32_t x = b_tensor.logical_shape()[0] / B;
    const uint32_t y = b_tensor.logical_shape()[1] / C;
    TT_FATAL(x != 0 && y != 0, "Internal error: batch multipler cannot be 0");

    tt::tt_metal::IDevice * device = a_tensor.device();

    auto * a = a_tensor.buffer();
    auto * b = b_tensor.buffer();
    auto * o = o_tensor.buffer();

    const uint32_t Kt = K / 32 + (K % 32 != 0);
    const uint32_t Nt = N / 32 + (N % 32 != 0);
    const uint32_t Mt = M / 32 + (M % 32 != 0);

    auto core_grid = device->compute_with_storage_grid_size();

    auto [num_cores, all_cores, core_group_1, core_group_2, work_per_core1, work_per_core2] =
        tt::tt_metal::split_work_to_cores(core_grid, Mt * Nt * C * y * B * x);
    GGML_UNUSED(num_cores);

    MakeCircularBuffer(program, all_cores, tt::CBIndex::c_0, 4, a_tensor.dtype());
    MakeCircularBuffer(program, all_cores, tt::CBIndex::c_1, 4, b_tensor.dtype());
    MakeCircularBuffer(program, all_cores, tt::CBIndex::c_16, 2, o_tensor.dtype());

    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*a).append_to(reader_compile_time_args);
    TensorAccessorArgs(*b).append_to(reader_compile_time_args);
    KernelHandle reader = CreateMetaliumKernel(program, "mul_mat_reader", all_cores, DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
        .compile_args = reader_compile_time_args,
        .defines = {},
        .named_compile_args = {}});

    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(*o).append_to(writer_compile_time_args);
    KernelHandle writer = CreateMetaliumKernel(program, "mul_mat_writer", all_cores, DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::RISCV_1_default,
        .compile_args = writer_compile_time_args,
        .defines = {},
        .named_compile_args = {}});

    KernelHandle compute = CreateMetaliumKernel(program, "mul_mat_compute", all_cores, ComputeConfig{
        .fp32_dest_acc_en = operation_attributes.high_percision,
        .unpack_to_dest_mode = {},
        .compile_args = {},
        .defines = {},
        .named_compile_args = {}});

    uint32_t id = 0;
    auto work_groups = {std::make_pair(core_group_1, work_per_core1), std::make_pair(core_group_2, work_per_core2)};
    for (const auto & [group, work_per_item] : work_groups) {
        for (const auto & range : group.ranges()) {
            for (const auto & core : range) {
                SetRuntimeArgs(
                    program,
                    reader,
                    core,
                    std::vector<uint32_t>{a->address(), b->address(), Mt, Nt, Kt, B, C, x, y, id, work_per_item});
                SetRuntimeArgs(
                    program,
                    compute,
                    core,
                    std::vector<uint32_t>{Mt, Nt, Kt, B, C, x, y, id, work_per_item});
                SetRuntimeArgs(
                    program,
                    writer,
                    core,
                    std::vector<uint32_t>{o->address(), Mt, Nt, Kt, B, C, x, y, id, work_per_item});
                id += work_per_item;
            }
        }
    }

    return cached_program_t{
        std::move(program),
        shared_variables_t{reader, writer, all_cores}};
}

void MulMatDeviceOperation::ProgramFactory::override_runtime_arguments(
    cached_program_t & cached_program,
    const operation_attributes_t &,
    const tensor_args_t & tensor_args,
    tensor_return_value_t & output_tensors) {
    auto * a = tensor_args.a.buffer();
    auto * b = tensor_args.b.buffer();
    auto * o = output_tensors.at(0).buffer();

    for (const auto & range : cached_program.shared_variables.all_cores.ranges()) {
        for (const auto & core : range) {
            {
                auto & runtime_args =
                    GetRuntimeArgs(cached_program.program, cached_program.shared_variables.reader, core);
                runtime_args[0] = a->address();
                runtime_args[1] = b->address();
            }
            {
                auto & runtime_args =
                    GetRuntimeArgs(cached_program.program, cached_program.shared_variables.writer, core);
                runtime_args[0] = o->address();
            }
        }
    }
}

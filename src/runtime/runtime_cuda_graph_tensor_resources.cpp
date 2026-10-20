/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime_cuda_graph_tensor_resources.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/domain/dlpack.hpp>
#include <holoscan/core/domain/dlpack_utils.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/memory/tensor_storage.hpp>

#include "../compiler/execution_plan_lookup.hpp"
#include "../compiler/execution_plan_state.hpp"
#include "../core/operator_spec_compiler_access.hpp"
#include "../core/tensor_output_loan_internal.hpp"
#include "graph_exec.hpp"
#include "runtime_callback_context.hpp"

namespace holoscan::detail {
std::optional<std::size_t> cuda_graph_contiguous_byte_span(
    const TensorLoanRequest& request) noexcept {
  if (request.dtype.bits == 0U || request.dtype.lanes == 0U || request.byte_strides.has_value()) {
    return std::nullopt;
  }
  std::size_t elements = 1U;
  for (const std::int64_t extent : request.shape) {
    if (extent < 0 || (elements != 0U && static_cast<std::uint64_t>(extent) >
                                             std::numeric_limits<std::size_t>::max() / elements)) {
      return std::nullopt;
    }
    elements *= static_cast<std::size_t>(extent);
  }
  const std::uint64_t item_bytes = dlpack_dtype_item_size_bytes(request.dtype);
  if (item_bytes == 0U || item_bytes > std::numeric_limits<std::size_t>::max() ||
      elements > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(item_bytes)) {
    return std::nullopt;
  }
  return elements * static_cast<std::size_t>(item_bytes);
}

namespace {

[[nodiscard]] Error tensor_error(std::string_view message) noexcept {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    return Error{ErrorCode::kFailure, std::string{message}};
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (...) {
    return Error{ErrorCode::kFailure};
  }
#endif
}

[[nodiscard]] bool same_dtype(const DLDataType left, const DLDataType right) noexcept {
  return left.code == right.code && left.bits == right.bits && left.lanes == right.lanes;
}

struct ReplayDeclaration {
  void* declaration{};
  const RuntimePortTypeOps* type_ops{};
};

[[nodiscard]] ReplayDeclaration find_declaration(RuntimeOperatorInstance& instance,
                                                 std::string_view name,
                                                 PortDirection direction) noexcept {
  if (instance.replay_spec == nullptr) {
    return {};
  }
  for (std::size_t index = 0U;
       index < OperatorSpecCompilerAccess::port_count(*instance.replay_spec);
       ++index) {
    const auto replay = OperatorSpecCompilerAccess::port(*instance.replay_spec, index);
    if (replay.direction == direction && replay.name == name && replay.declaration != nullptr &&
        replay.runtime_type_ops != nullptr) {
      return {.declaration = const_cast<void*>(replay.declaration),
              .type_ops = replay.runtime_type_ops};
    }
  }
  return {};
}

[[nodiscard]] RuntimeOperatorInstance* find_operator(std::span<RuntimeOperatorInstance> operators,
                                                     VertexId vertex) noexcept {
  const auto found = std::ranges::find(operators, vertex, &RuntimeOperatorInstance::vertex);
  return found == operators.end() ? nullptr : &*found;
}

class FixedMemoryReference final : public TensorMemoryReference {
 public:
  void retain_tensor_handle() noexcept override {}
  void release_tensor_handle() noexcept override {}
  bool set_deallocation_stream(cudaStream_t stream) override { return stream != nullptr; }
  bool set_readiness_stream(cudaStream_t stream) override { return stream != nullptr; }

  [[nodiscard]] bool has_readiness_stream_contract() const noexcept override { return false; }

  [[nodiscard]] cudaStream_t readiness_stream() const noexcept override { return nullptr; }

  [[nodiscard]] expected<void, Error> synchronize_for_consumer(
      cudaStream_t consumer_stream) override {
    return consumer_stream != nullptr
               ? expected<void, Error>{}
               : expected<void, Error>{make_unexpected(Error{ErrorCode::kInvalidArgument})};
  }

  [[nodiscard]] expected<void, Error> synchronize_for_host() override { return {}; }
};

}  // namespace

class RuntimeCUDAGraphTensorResources::Impl final {
 public:
  struct Region;

  // One record binds a region edge to the address owned by its one-slot RuntimePool.
  struct Allocation final {
    Impl* owner{};
    CUDAGraphCaptureBufferPlanRecord plan;
    RuntimeOperatorInstance* producer{};
    ReplayDeclaration output;
    RuntimeOutputBinding capture_output_binding;
    RuntimeFixedTensorAddress address;
    std::shared_ptr<FixedMemoryReference> memory_reference;
    TensorShape shape;
    DLDataType dtype{};
    std::size_t logical_byte_size{};
    bool geometry_available{};
    bool capture_output_emitted{};
    bool loan_live{};
    bool writer_active{};
    bool writer_cuda{};
    bool write_committed{};
    cudaStream_t writer_stream{};
    EmitOptions capture_options;

    [[nodiscard]] bool accepts(const TensorLoanRequest& request) const noexcept {
      if (!plan.layout.dtype || !plan.layout.rank || !plan.layout.strides ||
          *plan.layout.strides != TensorStridePolicy::kContiguousOnly ||
          request.shape.size() != *plan.layout.rank ||
          request.shape.size() != plan.fixed_shape.size() ||
          !same_dtype(request.dtype, *plan.layout.dtype)) {
        return false;
      }
      const auto bytes = cuda_graph_contiguous_byte_span(request);
      if (!bytes || *bytes > address.capacity) {
        return false;
      }
      for (std::size_t index = 0U; index < request.shape.size(); ++index) {
        if (request.shape[index] < 0 ||
            static_cast<std::uint64_t>(request.shape[index]) != plan.fixed_shape[index]) {
          return false;
        }
      }
      return true;
    }

    void set_geometry(const TensorLoanRequest& request) {
      shape.assign(request.shape.begin(), request.shape.end());
      dtype = request.dtype;
      logical_byte_size = cuda_graph_contiguous_byte_span(request).value_or(0U);
      geometry_available = true;
    }

    [[nodiscard]] expected<Tensor, Error> tensor() const {
      const MemoryStorageType storage_type = plan.memory_kind == MemoryKind::kCudaDevice
                                                 ? MemoryStorageType::kDevice
                                                 : MemoryStorageType::kHost;
      return make_tensor(shape,
                         dtype,
                         std::nullopt,
                         storage_type,
                         address.data,
                         memory_reference,
                         plan.device.valid() ? plan.device.value : Allocator::kUnknownDeviceId);
    }
  };

  // Presents a fixed Tensor as an ordinary selected Input while a member compute() is captured.
  struct CaptureInput final {
    PortId port{};
    VertexId owner{};
    bool restore_runtime_binding{};
    ReplayDeclaration declaration;
    RuntimeInputBinding binding;
    std::optional<Sample<Tensor>> sample;

    static InputSelection selection(const void* state, std::uint64_t generation) noexcept {
      const auto& input = *static_cast<const CaptureInput*>(state);
      return generation == 1U && input.sample.has_value()
                 ? InputSelection{InputState::kSelected, SelectionReason::kNone}
                 : InputSelection{InputState::kAbsent, SelectionReason::kNeverSeen};
    }

    static expected<RuntimeScalarReadBinding, Error> scalar(const void* state,
                                                            std::uint64_t generation,
                                                            bool) noexcept {
      const auto& input = *static_cast<const CaptureInput*>(state);
      if (generation != 1U || !input.sample) {
        return make_unexpected(Error{ErrorCode::kInputEmpty});
      }
      return RuntimeScalarReadBinding{
          .sample = &*input.sample,
          .claim = PublishedReadClaimBinding{
              .state = const_cast<CaptureInput*>(&input),
              .retain = [](void*) noexcept {},
              .release = [](void*) noexcept {},
          },
      };
    }

    static expected<RuntimeBatchReadBinding, Error> batch(const void*, std::uint64_t) noexcept {
      return make_unexpected(Error{ErrorCode::kInputSelectionCardinalityMismatch});
    }

    inline static const RuntimeInputBindingOps kOps{
        .selection = &selection,
        .scalar = &scalar,
        .batch = &batch,
    };
  };

  struct Region final {
    struct Buffer {
      const CUDAGraphCaptureBufferPlanRecord* plan{};
      Allocation* allocation{};
    };

    const CUDAGraphExecutionRegionPlanRecord* execution{};
    std::vector<Buffer> buffers;
    std::vector<std::unique_ptr<CaptureInput>> inputs;
    Allocation* egress{};
    std::optional<TensorOutputLoan> launch_loan;
    std::optional<TensorOutputWriteGuard> launch_writer;
    cudaStream_t capture_stream{};
    bool capture_active{};

    [[nodiscard]] CaptureInput* input(PortId port) noexcept {
      const auto found = std::ranges::find_if(
          inputs, [&](const auto& candidate) { return candidate->port == port; });
      return found == inputs.end() ? nullptr : found->get();
    }
  };

  RuntimeCUDAGraphChannelBindings channel_bindings;
  RuntimeCUDAGraphTensorPoolBinding tensor_pools;
  std::vector<std::unique_ptr<Allocation>> allocations;
  std::vector<std::unique_ptr<Region>> regions;
  Region* active_capture{};

  ~Impl() {
    // A partial startup failure can destroy this object while capture bindings are still active.
    // Restore them before releasing the callbacks owned by the Tensor and Channel sessions.
    if (active_capture != nullptr) {
      static_cast<void>(restore_capture_bindings(*active_capture));
    }
    channel_bindings = {};
    tensor_pools = {};
  }

  [[nodiscard]] Region* find_region(GraphId graph, RegionId region) noexcept {
    const auto found = std::ranges::find_if(regions, [&](const auto& candidate) {
      return candidate->execution->graph == graph && candidate->execution->region == region;
    });
    return found == regions.end() ? nullptr : found->get();
  }

  [[nodiscard]] bool restore_capture_bindings(Region& region) noexcept {
    bool restored = true;
    for (auto& input : region.inputs) {
      input->sample.reset();
      input->declaration.type_ops->unbind_input(input->declaration.declaration, &input->binding);
      if (!input->restore_runtime_binding) {
        continue;
      }
      if (channel_bindings.state != nullptr && channel_bindings.restore_input != nullptr) {
        restored = channel_bindings.restore_input(channel_bindings.state, input->port) && restored;
      } else {
        restored = false;
      }
    }
    for (const Region::Buffer& buffer : region.buffers) {
      Allocation* const allocation = buffer.allocation;
      allocation->loan_live = false;
      allocation->writer_active = false;
      allocation->writer_cuda = false;
      allocation->write_committed = false;
      allocation->writer_stream = nullptr;
      if (buffer.plan->kind != CUDAGraphCaptureBufferKind::kIngress) {
        allocation->output.type_ops->unbind_output(allocation->output.declaration,
                                                   &allocation->capture_output_binding);
      }
      if (buffer.plan->kind == CUDAGraphCaptureBufferKind::kEgress) {
        if (channel_bindings.state != nullptr && channel_bindings.restore_output != nullptr) {
          restored = channel_bindings.restore_output(channel_bindings.state,
                                                     allocation->plan.source_port) &&
                     restored;
        } else {
          restored = false;
        }
      }
      if (tensor_pools.state != nullptr && tensor_pools.restore_output_binding != nullptr) {
        restored =
            tensor_pools.restore_output_binding(tensor_pools.state, allocation->plan.source_port) &&
            restored;
      } else {
        restored = false;
      }
    }
    region.capture_active = false;
    region.capture_stream = nullptr;
    active_capture = nullptr;
    return restored;
  }

  void detach_session_state() noexcept {
    // Finish any temporary capture binding while the Channel session is still available.
    if (active_capture != nullptr) {
      static_cast<void>(restore_capture_bindings(*active_capture));
    }
    if (channel_bindings.state != nullptr && channel_bindings.clear_fixed_tensor != nullptr) {
      for (const auto& region : regions) {
        for (const Region::Buffer& buffer : region->buffers) {
          if (buffer.plan->kind == CUDAGraphCaptureBufferKind::kIngress) {
            static_cast<void>(channel_bindings.clear_fixed_tensor(
                channel_bindings.state, buffer.plan->target_port, buffer.allocation->address.data));
          }
        }
      }
    }
    channel_bindings = {};
  }

  [[nodiscard]] expected<void, Error> capture_emit(Allocation& allocation, Tensor tensor,
                                                   EmitOptions options) {
    if (active_capture == nullptr || !active_capture->capture_active) {
      return make_unexpected(tensor_error("CUDA_GRAPH_CAPTURE_TENSOR_STATE_INVALID"));
    }
    const auto buffer = std::ranges::find_if(active_capture->buffers, [&](const auto& candidate) {
      return candidate.allocation == &allocation;
    });
    if (buffer == active_capture->buffers.end()) {
      return make_unexpected(tensor_error("CUDA_GRAPH_CAPTURE_TENSOR_BINDING_INVALID"));
    }
    if (allocation.capture_output_emitted) {
      return make_unexpected(tensor_error("CUDA_GRAPH_CAPTURE_OUTPUT_REPEATED"));
    }
    // Internal publications collapse into the captured graph, including their EmitOptions. Keep
    // only the region egress producer's options for each fused output publication.
    if (buffer->plan->kind == CUDAGraphCaptureBufferKind::kInternal) {
      CaptureInput* const input = active_capture->input(buffer->plan->target_port);
      if (input == nullptr || input->sample.has_value()) {
        return make_unexpected(tensor_error("CUDA_GRAPH_CAPTURE_TENSOR_BINDING_INVALID"));
      }
      input->sample.emplace(std::move(tensor), SampleMetadata{});
    } else if (buffer->plan->kind == CUDAGraphCaptureBufferKind::kEgress) {
      allocation.capture_options = options;
    }
    allocation.capture_output_emitted = true;
    return {};
  }

  static expected<TensorOutputLoan, Error> allocate(void* state,
                                                    TensorLoanRequest request) noexcept {
    auto& allocation = *static_cast<Allocation*>(state);
    const RuntimeCallbackIdentity identity = runtime_callback_identity();
    if (allocation.producer == nullptr || identity.context != RuntimeCallbackContext::kCompute ||
        identity.operator_id != allocation.producer->operator_id || !identity.tag.valid() ||
        allocation.owner->active_capture == nullptr || allocation.loan_live ||
        allocation.writer_active) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    if (!allocation.accepts(request)) {
      return make_unexpected(tensor_error("CUDA_GRAPH_CAPTURE_FIXED_SHAPE_MISMATCH"));
    }
    // Keep the Tensor descriptor's logical shape while reusing the fixed maximum-capacity address.
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      allocation.set_geometry(request);
      allocation.loan_live = true;
      allocation.write_committed = false;
      return TensorOutputLoanAccess::loan(&allocation, &kLoanOps);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (...) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
#endif
  }

  static expected<void, Error> emit(void* state, TensorOutputLoan loan, const void* descriptor,
                                    EmitOptions options) noexcept {
    auto* const allocation = static_cast<Allocation*>(state);
    if (allocation == nullptr || descriptor != nullptr ||
        TensorOutputLoanAccess::state_for_ops(loan, &kLoanOps) != allocation ||
        !allocation->loan_live || allocation->writer_active || !allocation->write_committed) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (allocation->owner->active_capture == nullptr) {
      return make_unexpected(tensor_error("CUDA_GRAPH_CAPTURE_TENSOR_BINDING_INVALID"));
    }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      auto tensor = allocation->tensor();
      allocation->loan_live = false;
      allocation->write_committed = false;
      if (!tensor) {
        return make_unexpected(std::move(tensor.error()));
      }
      return allocation->owner->capture_emit(*allocation, std::move(*tensor), options);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (...) {
      abandon_loan(allocation);
      return make_unexpected(Error{ErrorCode::kFailure});
    }
#endif
  }

  static void abandon_loan(void* state) noexcept {
    auto& allocation = *static_cast<Allocation*>(state);
    allocation.loan_live = false;
    allocation.writer_active = false;
    allocation.writer_cuda = false;
    allocation.write_committed = false;
    allocation.writer_stream = nullptr;
  }

  static TensorShapeView shape(const void* state) noexcept {
    return static_cast<const Allocation*>(state)->shape;
  }

  static TensorByteStridesView strides(const void*) noexcept { return std::nullopt; }

  static DLDataType dtype(const void* state) noexcept {
    return static_cast<const Allocation*>(state)->dtype;
  }

  static DeviceId device(const void* state) noexcept {
    return static_cast<const Allocation*>(state)->plan.device;
  }

  static MemoryKind memory_kind(const void* state) noexcept {
    return static_cast<const Allocation*>(state)->plan.memory_kind;
  }

  static std::size_t byte_size(const void* state) noexcept {
    return static_cast<const Allocation*>(state)->logical_byte_size;
  }

  static expected<TensorOutputWriteGuard, Error> begin_write(void* state,
                                                             ::cuda::stream_ref stream) noexcept {
    auto& allocation = *static_cast<Allocation*>(state);
    const Region* const capture = allocation.owner->active_capture;
    if (capture == nullptr || !allocation.loan_live || allocation.writer_active ||
        stream.get() == nullptr || stream.get() != capture->capture_stream) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    }
    allocation.writer_active = true;
    allocation.writer_cuda = true;
    allocation.writer_stream = stream.get();
    return TensorOutputLoanAccess::write_guard(&allocation, &kWriteOps);
  }

  static expected<TensorOutputWriteGuard, Error> begin_host_write(void*) noexcept {
    // A host write executes once during startup and cannot become a CUDA Graph replay operation.
    return make_unexpected(tensor_error("CUDA_GRAPH_CAPTURE_HOST_OUTPUT_WRITE_UNSUPPORTED"));
  }

  static expected<void, Error> cancel(void* state) noexcept {
    abandon_loan(state);
    return {};
  }

  static void abandon_write(void* state) noexcept { abandon_loan(state); }

  static void* write_data(const void* state) noexcept {
    return static_cast<const Allocation*>(state)->address.data;
  }

  static std::optional<::cuda::stream_ref> write_stream(const void* state) noexcept {
    const auto* allocation = static_cast<const Allocation*>(state);
    return allocation->writer_cuda && allocation->writer_stream != nullptr
               ? std::optional<::cuda::stream_ref>{::cuda::stream_ref{allocation->writer_stream}}
               : std::nullopt;
  }

  static expected<void, Error> commit_write(void* state) noexcept {
    auto& allocation = *static_cast<Allocation*>(state);
    if (!allocation.loan_live || !allocation.writer_active ||
        (allocation.writer_cuda && allocation.writer_stream == nullptr)) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    allocation.writer_active = false;
    allocation.writer_cuda = false;
    allocation.write_committed = true;
    return {};
  }

  static expected<void, Error> reject_direct_tensor_emit(void*, const void*) noexcept {
    // Existing Tensor values have arbitrary backing addresses. Capture accepts only loans from the
    // region's fixed RuntimePool slot.
    return make_unexpected(tensor_error("CUDA_GRAPH_CAPTURE_DIRECT_TENSOR_EMIT_UNSUPPORTED"));
  }

  inline static const TensorOutputBindingOps kBindingOps{
      .allocate = &allocate,
      .emit_loan = &emit,
  };

  inline static const RuntimeOutputBindingOps kCaptureOutputOps{
      .emit_sample = &reject_direct_tensor_emit,
  };

  inline static const TensorOutputLoanOps kLoanOps{
      .abandon = &abandon_loan,
      .shape = &shape,
      .byte_strides = &strides,
      .dtype = &dtype,
      .device_id = &device,
      .memory_kind = &memory_kind,
      .byte_size = &byte_size,
      .write = &begin_write,
      .write_host = &begin_host_write,
      .cancel = &cancel,
  };

  inline static const TensorOutputWriteGuardOps kWriteOps{
      .abandon = &abandon_write,
      .data = &write_data,
      .byte_size = &byte_size,
      .stream = &write_stream,
      .commit = &commit_write,
  };

  // Resolve shared inter-region storage independently of capture-input construction.
  [[nodiscard]] expected<Allocation*, Error> resolve_allocation(
      const ExecutionPlanState& plan, std::span<RuntimeOperatorInstance> operators,
      const CUDAGraphCaptureBufferPlanRecord& buffer) {
    if (buffer.active_slot_count != 1U ||
        buffer.aggregate_byte_capacity != buffer.bounds.max_byte_span) {
      return make_unexpected(tensor_error("CUDA_GRAPH_FIXED_TENSOR_PLAN_INVALID"));
    }
    auto existing = std::ranges::find_if(allocations, [&](const auto& allocation) {
      return allocation->plan.fixed_allocation_id == buffer.fixed_allocation_id;
    });
    Allocation* allocation{};
    if (existing != allocations.end()) {
      // Reuse one fixed allocation for an edge connecting two CUDA Graph regions. Compare all
      // physical properties after ignoring its region-local ingress or egress classification.
      auto comparable = buffer;
      comparable.kind = existing->get()->plan.kind;
      if (!(existing->get()->plan == comparable)) {
        return make_unexpected(tensor_error("CUDA_GRAPH_FIXED_TENSOR_PLAN_INVALID"));
      }
      allocation = existing->get();
    } else {
      const auto* output_port =
          find_port_by_id_and_direction(plan, buffer.source_port, PortDirection::kOutput);
      RuntimeOperatorInstance* producer =
          output_port != nullptr ? find_operator(operators, output_port->owner) : nullptr;
      const ReplayDeclaration declaration =
          producer != nullptr
              ? find_declaration(*producer, output_port->local_name, PortDirection::kOutput)
              : ReplayDeclaration{};
      if (producer == nullptr || declaration.declaration == nullptr ||
          declaration.type_ops != runtime_port_type_ops<Tensor>() ||
          declaration.type_ops->bind_output == nullptr ||
          declaration.type_ops->unbind_output == nullptr ||
          declaration.type_ops->bind_tensor_output == nullptr ||
          declaration.type_ops->unbind_tensor_output == nullptr ||
          declaration.type_ops->publish_tensor_loan == nullptr) {
        return make_unexpected(tensor_error("CUDA_GRAPH_FIXED_TENSOR_BINDING_INVALID"));
      }
      if (tensor_pools.state == nullptr || tensor_pools.fixed_output_address == nullptr ||
          tensor_pools.fixed_output_available == nullptr ||
          tensor_pools.restore_output_binding == nullptr) {
        return make_unexpected(tensor_error("CUDA_GRAPH_FIXED_TENSOR_BINDING_INVALID"));
      }
      auto fixed_address =
          tensor_pools.fixed_output_address(tensor_pools.state, buffer.source_port);
      if (!fixed_address) {
        return make_unexpected(std::move(fixed_address.error()));
      }
      if (fixed_address->data == nullptr || fixed_address->capacity < buffer.bounds.max_byte_span ||
          fixed_address->memory_kind != buffer.memory_kind ||
          fixed_address->device != buffer.device) {
        return make_unexpected(tensor_error("CUDA_GRAPH_FIXED_TENSOR_PLAN_INVALID"));
      }
      auto created = std::make_unique<Allocation>();
      created->owner = this;
      created->plan = buffer;
      created->producer = producer;
      created->output = declaration;
      created->capture_output_binding =
          RuntimeOutputBinding{.state = created.get(), .ops = &kCaptureOutputOps};
      created->address = *fixed_address;
      created->memory_reference = std::make_shared<FixedMemoryReference>();
      allocation = created.get();
      allocations.push_back(std::move(created));
    }
    return allocation;
  }
};

RuntimeCUDAGraphTensorResources::RuntimeCUDAGraphTensorResources(
    std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

RuntimeCUDAGraphTensorResources::~RuntimeCUDAGraphTensorResources() = default;

// Validate every capture input and shared allocation before registering fixed ingress addresses;
// keep registration and its rollback together so a failed region leaves no partial requirements.
expected<std::unique_ptr<RuntimeCUDAGraphTensorResources>, Error>
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
RuntimeCUDAGraphTensorResources::create(const ExecutionPlanState& plan,
                                        std::span<RuntimeOperatorInstance> operators,
                                        RuntimeCUDAGraphTensorPoolBinding tensor_pools,
                                        ProcessGroupId process_group,
                                        RuntimeCUDAGraphChannelBindings channel_bindings,
                                        CUDAGraphRuntimeRegionIdentity* failure_region) {
  if (failure_region != nullptr) {
    *failure_region = {};
  }
  auto impl = std::make_unique<Impl>();
  impl->channel_bindings = channel_bindings;
  impl->tensor_pools = tensor_pools;
  struct FixedInputRequirement {
    PortId port{};
    Impl::Allocation* allocation{};
  };
  std::vector<FixedInputRequirement> fixed_input_requirements;

  for (const auto& execution : plan.cuda_graph_execution_regions) {
    if (execution.process_group != process_group) {
      continue;
    }
    if (failure_region != nullptr) {
      *failure_region = CUDAGraphRuntimeRegionIdentity{
          .graph = execution.graph,
          .region = execution.region,
          .fused_operator = execution.fused_operator,
      };
    }
    const auto* capture =
        find_unique_cuda_graph_capture_plan_by_region(plan, execution.graph, execution.region);
    if (capture == nullptr) {
      return make_unexpected(tensor_error("CUDA_GRAPH_FIXED_TENSOR_PLAN_INVALID"));
    }
    auto region = std::make_unique<Impl::Region>();
    region->execution = &execution;
    for (const auto& buffer : execution.buffers) {
      auto resolved = impl->resolve_allocation(plan, operators, buffer);
      if (!resolved) {
        return make_unexpected(std::move(resolved).error());
      }
      Impl::Allocation* const allocation = *resolved;
      region->buffers.push_back(Impl::Region::Buffer{.plan = &buffer, .allocation = allocation});

      if (buffer.kind == CUDAGraphCaptureBufferKind::kEgress) {
        if (channel_bindings.state == nullptr || channel_bindings.restore_output == nullptr) {
          return make_unexpected(tensor_error("CUDA_GRAPH_FIXED_TENSOR_BINDING_INVALID"));
        }
        region->egress = allocation;
        continue;
      }
      if (buffer.kind == CUDAGraphCaptureBufferKind::kIngress &&
          (channel_bindings.state == nullptr || channel_bindings.restore_input == nullptr ||
           channel_bindings.require_fixed_tensor == nullptr ||
           channel_bindings.clear_fixed_tensor == nullptr)) {
        return make_unexpected(tensor_error("CUDA_GRAPH_FIXED_TENSOR_BINDING_INVALID"));
      }
      const auto* input_port =
          find_port_by_id_and_direction(plan, buffer.target_port, PortDirection::kInput);
      RuntimeOperatorInstance* consumer =
          input_port != nullptr ? find_operator(operators, input_port->owner) : nullptr;
      const ReplayDeclaration declaration =
          consumer != nullptr
              ? find_declaration(*consumer, input_port->local_name, PortDirection::kInput)
              : ReplayDeclaration{};
      if (consumer == nullptr || declaration.declaration == nullptr ||
          declaration.type_ops != runtime_port_type_ops<Tensor>() ||
          declaration.type_ops->bind_input == nullptr ||
          declaration.type_ops->unbind_input == nullptr) {
        return make_unexpected(tensor_error("CUDA_GRAPH_CAPTURE_INPUT_BINDING_INVALID"));
      }
      auto input = std::make_unique<Impl::CaptureInput>();
      input->port = buffer.target_port;
      input->owner = input_port->owner;
      // Only the fused source ingress remains in the steady-state channel plan. Internal inputs
      // belong to capture-only members and have no Runtime binding to restore.
      input->restore_runtime_binding = buffer.kind == CUDAGraphCaptureBufferKind::kIngress;
      input->declaration = declaration;
      input->binding = RuntimeInputBinding{
          .state = input.get(), .ops = &Impl::CaptureInput::kOps, .generation = 1U};
      region->inputs.push_back(std::move(input));
      if (buffer.kind == CUDAGraphCaptureBufferKind::kIngress) {
        fixed_input_requirements.push_back(
            FixedInputRequirement{.port = buffer.target_port, .allocation = allocation});
      }
    }
    impl->regions.push_back(std::move(region));
  }

  // Register fixed ingress addresses after all fallible plan and port validation. If one Channel
  // rejects its address, remove the requirements already installed by this construction attempt.
  std::size_t registered_requirement_count{};
  for (const FixedInputRequirement& requirement : fixed_input_requirements) {
    if (!channel_bindings.require_fixed_tensor(channel_bindings.state,
                                               requirement.port,
                                               requirement.allocation->address.data,
                                               requirement.allocation->plan.fixed_shape)) {
      for (std::size_t index = 0U; index < registered_requirement_count; ++index) {
        const FixedInputRequirement& registered = fixed_input_requirements[index];
        static_cast<void>(channel_bindings.clear_fixed_tensor(
            channel_bindings.state, registered.port, registered.allocation->address.data));
      }
      return make_unexpected(tensor_error("CUDA_GRAPH_FIXED_TENSOR_BINDING_INVALID"));
    }
    ++registered_requirement_count;
  }

  if (failure_region != nullptr) {
    *failure_region = {};
  }
  return std::unique_ptr<RuntimeCUDAGraphTensorResources>{
      new RuntimeCUDAGraphTensorResources{std::move(impl)}};
}

// Keep temporary output/input binding and ingress materialization beside the common restore path
// so any invalid shape, missing sample, or allocation failure leaves capture adapters detached.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<void, Error> RuntimeCUDAGraphTensorResources::begin_capture(GraphId graph,
                                                                     RegionId region_id,
                                                                     cudaStream_t stream) noexcept {
  Impl::Region* const region = impl_->find_region(graph, region_id);
  if (region == nullptr || stream == nullptr || impl_->active_capture != nullptr ||
      region->capture_active) {
    return make_unexpected(tensor_error("CUDA_GRAPH_CAPTURE_TENSOR_STATE_INVALID"));
  }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    region->capture_stream = stream;
    region->capture_active = true;
    impl_->active_capture = region;
    // Member compute() uses this adapter only while CUDA records the region. The adapter returns
    // the address owned by the corresponding one-slot RuntimePool and routes internal outputs
    // directly to the next captured member.
    for (const Impl::Region::Buffer& buffer : region->buffers) {
      Impl::Allocation& allocation = *buffer.allocation;
      allocation.capture_output_emitted = false;
      if (buffer.plan->kind != CUDAGraphCaptureBufferKind::kIngress) {
        // Reject publication of an externally backed Tensor before it can reach the ordinary
        // Channel. Captured outputs must come from the fixed-address loan bound below.
        allocation.output.type_ops->bind_output(allocation.output.declaration,
                                                &allocation.capture_output_binding);
      }
      allocation.output.type_ops->bind_tensor_output(
          allocation.output.declaration,
          TensorOutputBinding{
              .state = &allocation,
              .ops = &Impl::kBindingOps,
              .tensor_dtype = allocation.plan.layout.dtype,
          });
    }
    // Replace member input bindings with capture-only samples backed by the region's fixed storage.
    for (auto& input : region->inputs) {
      input->declaration.type_ops->bind_input(input->declaration.declaration, &input->binding);
    }

    const auto ingress = std::ranges::find_if(region->buffers, [](const auto& buffer) {
      return buffer.plan->kind == CUDAGraphCaptureBufferKind::kIngress;
    });
    if (ingress != region->buffers.end()) {
      // Capture and replay use the exact shape frozen by lowering. Runtime ingress validation and
      // output allocation enforce this same shape before a graph launch is admitted.
      Impl::Allocation& allocation = *ingress->allocation;
      if (!allocation.plan.layout.dtype || !allocation.plan.layout.rank ||
          allocation.plan.fixed_shape.size() != *allocation.plan.layout.rank) {
        static_cast<void>(impl_->restore_capture_bindings(*region));
        return make_unexpected(tensor_error("CUDA_GRAPH_FIXED_TENSOR_LAYOUT_INVALID"));
      }
      TensorShape shape;
      shape.reserve(allocation.plan.fixed_shape.size());
      for (const std::uint64_t extent : allocation.plan.fixed_shape) {
        if (extent > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
          static_cast<void>(impl_->restore_capture_bindings(*region));
          return make_unexpected(tensor_error("CUDA_GRAPH_FIXED_TENSOR_LAYOUT_INVALID"));
        }
        shape.push_back(static_cast<std::int64_t>(extent));
      }
      const TensorLoanRequest request{.shape = shape,
                                      .dtype = allocation.plan.layout.dtype.value_or(DLDataType{})};
      if (!allocation.accepts(request)) {
        static_cast<void>(impl_->restore_capture_bindings(*region));
        return make_unexpected(tensor_error("CUDA_GRAPH_FIXED_TENSOR_LAYOUT_INVALID"));
      }
      allocation.set_geometry(request);
      auto tensor = allocation.tensor();
      Impl::CaptureInput* const input = region->input(ingress->plan->target_port);
      if (!tensor || input == nullptr) {
        static_cast<void>(impl_->restore_capture_bindings(*region));
        return make_unexpected(tensor ? tensor_error("CUDA_GRAPH_CAPTURE_INPUT_BINDING_INVALID")
                                      : std::move(tensor.error()));
      }
      input->sample.emplace(std::move(*tensor), SampleMetadata{});
    }
    return {};
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (...) {
    if (region->capture_active) {
      static_cast<void>(impl_->restore_capture_bindings(*region));
    }
    return make_unexpected(Error{ErrorCode::kFailure});
  }
#endif
}

expected<void, Error> RuntimeCUDAGraphTensorResources::prepare_capture_member(
    VertexId member) noexcept {
  if (impl_->active_capture == nullptr || !impl_->active_capture->capture_active) {
    return make_unexpected(tensor_error("CUDA_GRAPH_CAPTURE_TENSOR_STATE_INVALID"));
  }
  for (const auto& input : impl_->active_capture->inputs) {
    if (input->owner == member && !input->sample) {
      return make_unexpected(tensor_error("CUDA_GRAPH_CAPTURE_INPUT_NOT_READY"));
    }
  }
  return {};
}

expected<CUDAGraphCaptureEndState, Error> RuntimeCUDAGraphTensorResources::end_capture(
    GraphId graph, RegionId region_id) noexcept {
  Impl::Region* const region = impl_->find_region(graph, region_id);
  if (region == nullptr || impl_->active_capture != region || !region->capture_active) {
    return make_unexpected(tensor_error("CUDA_GRAPH_CAPTURE_TENSOR_STATE_INVALID"));
  }
  const bool complete = std::ranges::all_of(region->buffers, [](const auto& buffer) {
    const bool member_output_produced = buffer.plan->kind == CUDAGraphCaptureBufferKind::kIngress ||
                                        buffer.allocation->capture_output_emitted;
    return member_output_produced && buffer.allocation->geometry_available &&
           !buffer.allocation->loan_live && !buffer.allocation->writer_active;
  });
  const bool restored = impl_->restore_capture_bindings(*region);
  if (!restored) {
    return make_unexpected(tensor_error("CUDA_GRAPH_CAPTURE_BINDING_RESTORE_FAILED"));
  }
  return CUDAGraphCaptureEndState{.outputs_complete = complete};
}

expected<void, Error> RuntimeCUDAGraphTensorResources::begin_launch(std::size_t region_index,
                                                                    cudaStream_t stream) noexcept {
  if (impl_ == nullptr || region_index >= impl_->regions.size() || stream == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  Impl::Region* const region = impl_->regions[region_index].get();
  if (region->launch_loan || region->launch_writer) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  if (region->egress == nullptr) {
    return {};
  }

  Impl::Allocation& egress = *region->egress;
  auto* const output = static_cast<Output<Tensor>*>(egress.output.declaration);
  if (output == nullptr || !egress.geometry_available) {
    return make_unexpected(tensor_error("CUDA_GRAPH_RUNTIME_OUTPUT_INVALID"));
  }
  RuntimeCallbackIdentity output_identity = runtime_callback_identity();
  if (egress.producer == nullptr || output_identity.context != RuntimeCallbackContext::kCompute ||
      !output_identity.tag.valid()) {
    return make_unexpected(tensor_error("CUDA_GRAPH_RUNTIME_OUTPUT_IDENTITY_INVALID"));
  }
  // The fused callback is scheduled with the region source identity. Use the sink member identity
  // while acquiring its output so the ordinary RuntimePool records the same activation tag.
  output_identity.operator_id = egress.producer->operator_id;
  const RuntimeCallbackContextScope output_scope{output_identity};
  auto loan =
      output->allocate_tensor(TensorLoanRequest{.shape = egress.shape, .dtype = egress.dtype});
  if (!loan) {
    return make_unexpected(std::move(loan.error()));
  }
  auto writer = loan->write(::cuda::stream_ref{stream});
  if (!writer) {
    return make_unexpected(std::move(writer.error()));
  }
  if (writer->data() != egress.address.data) {
    return make_unexpected(tensor_error("CUDA_GRAPH_FIXED_TENSOR_ADDRESS_MISMATCH"));
  }
  region->launch_loan.emplace(std::move(*loan));
  region->launch_writer.emplace(std::move(*writer));
  return {};
}

expected<bool, Error> RuntimeCUDAGraphTensorResources::output_available(
    const std::size_t region_index) noexcept {
  if (impl_ == nullptr || region_index >= impl_->regions.size()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const Impl::Region* const region = impl_->regions[region_index].get();
  if (region->egress == nullptr) {
    return true;
  }
  if (impl_->tensor_pools.state == nullptr ||
      impl_->tensor_pools.fixed_output_available == nullptr) {
    return make_unexpected(tensor_error("CUDA_GRAPH_FIXED_TENSOR_BINDING_INVALID"));
  }
  return impl_->tensor_pools.fixed_output_available(impl_->tensor_pools.state,
                                                    region->egress->plan.source_port);
}

expected<void, Error> RuntimeCUDAGraphTensorResources::finish_launch(
    std::size_t region_index) noexcept {
  if (impl_ == nullptr || region_index >= impl_->regions.size()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  Impl::Region* const region = impl_->regions[region_index].get();
  if (region->egress == nullptr) {
    return {};
  }
  if (!region->launch_loan || !region->launch_writer) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  RuntimeCallbackIdentity output_identity = runtime_callback_identity();
  if (region->egress->producer == nullptr ||
      output_identity.context != RuntimeCallbackContext::kCompute || !output_identity.tag.valid()) {
    return make_unexpected(tensor_error("CUDA_GRAPH_RUNTIME_OUTPUT_IDENTITY_INVALID"));
  }
  output_identity.operator_id = region->egress->producer->operator_id;
  const RuntimeCallbackContextScope output_scope{output_identity};
  auto committed = std::move(*region->launch_writer).commit();
  region->launch_writer.reset();
  if (!committed) {
    region->launch_loan.reset();
    return committed;
  }
  auto* const output = static_cast<Output<Tensor>*>(region->egress->output.declaration);
  auto published = output->emit(std::move(*region->launch_loan), region->egress->capture_options);
  region->launch_loan.reset();
  return published;
}

void RuntimeCUDAGraphTensorResources::cancel_launch(const std::size_t region_index) noexcept {
  if (impl_ == nullptr || region_index >= impl_->regions.size()) {
    return;
  }
  Impl::Region* const region = impl_->regions[region_index].get();
  region->launch_writer.reset();
  region->launch_loan.reset();
}

void RuntimeCUDAGraphTensorResources::detach_session_state() noexcept {
  if (impl_ != nullptr) {
    impl_->detach_session_state();
  }
}

bool RuntimeCUDAGraphTensorResources::matches(std::size_t region_index,
                                              VertexId fused_operator) const noexcept {
  return impl_ != nullptr && region_index < impl_->regions.size() &&
         impl_->regions[region_index]->execution->fused_operator == fused_operator;
}

bool RuntimeCUDAGraphTensorResources::has_egress(const std::size_t region_index) const noexcept {
  return impl_ != nullptr && region_index < impl_->regions.size() &&
         impl_->regions[region_index]->egress != nullptr;
}

std::size_t RuntimeCUDAGraphTensorResources::size() const noexcept {
  return impl_ != nullptr ? impl_->regions.size() : 0U;
}

}  // namespace holoscan::detail

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime_cuda_graph_region_resources.hpp"

#include <algorithm>
#include <cassert>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "../compiler/execution_plan_lookup.hpp"
#include "../compiler/execution_plan_state.hpp"
#include "../cuda_execution/cuda_green_context_driver.hpp"
#include "graph_exec.hpp"
#include "port_binding.hpp"
#include "runtime_cuda_execution_diagnostic.hpp"
#include "runtime_cuda_execution_resources.hpp"
#include "runtime_cuda_graph_tensor_resources.hpp"
#include "scoped_cuda_device.hpp"

namespace holoscan::detail {
namespace {

[[nodiscard]] Error cuda_graph_named_error(ErrorCode code, std::string_view message) noexcept {
  Error result{code};
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    result.message.assign(message);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (const std::bad_alloc&) {
    // A code-only Error keeps noexcept runtime failure paths recoverable under memory pressure.
  }
#endif
  return result;
}

[[nodiscard]] Error cuda_graph_error(std::string_view operation, cudaError_t status) noexcept {
  Error result{ErrorCode::kFailure};
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    result.message = std::string{"CUDA Graph "}
                         .append(operation)
                         .append(" failed: ")
                         .append(cudaGetErrorName(status));
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (const std::bad_alloc&) {
    result.message.clear();
  }
#endif
  return result;
}

[[nodiscard]] Error cuda_graph_shutdown_error(std::string_view operation,
                                              std::int32_t status) noexcept {
  Error result{ErrorCode::kFailure};
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    result.message = std::string{"CUDA_GRAPH_RUNTIME_SHUTDOWN_FAILED: "}
                         .append(operation)
                         .append(" failed with CUDA status ")
                         .append(std::to_string(status));
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (const std::bad_alloc&) {
    result.message.clear();
  }
#endif
  return result;
}

void assign_cuda_graph_diagnostic(RuntimeCudaExecutionDiagnostic* diagnostic,
                                  LifecycleStage lifecycle_stage, std::string_view operation,
                                  DeviceId device, std::string_view stream_id, GraphId graph,
                                  RegionId region, VertexId fused_operator,
                                  std::int32_t driver_result = -1) noexcept {
  if (diagnostic == nullptr) {
    return;
  }
  *diagnostic = RuntimeCudaExecutionDiagnostic{
      .lifecycle_stage = lifecycle_stage,
      .device = device,
      .graph = graph,
      .region = region,
      .fused_operator = fused_operator,
      .driver_result = driver_result,
  };
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    diagnostic->operation.assign(operation);
    diagnostic->stream_id.assign(stream_id);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (const std::bad_alloc&) {
    // Preserve allocation-free structural evidence even when descriptive strings cannot be kept.
    diagnostic->operation.clear();
    diagnostic->stream_id.clear();
  }
#endif
}

[[nodiscard]] RuntimeOperatorInstance* find_unique_operator(
    std::span<RuntimeOperatorInstance> operators, VertexId vertex) noexcept {
  RuntimeOperatorInstance* result{};
  for (RuntimeOperatorInstance& candidate : operators) {
    if (candidate.vertex != vertex) {
      continue;
    }
    if (result != nullptr) {
      return nullptr;
    }
    result = &candidate;
  }
  return result;
}

[[nodiscard]] bool lifecycle_restart_can_reconstruct(const ExecutionPlanState& plan,
                                                     OperatorRef member) {
  const auto admitted_target = [&](OperatorRef target_ref) {
    const auto target = std::ranges::find(
        plan.lifecycle_restart_targets, target_ref, &LifecycleRestartTargetPlanRecord::target);
    return target != plan.lifecycle_restart_targets.end() &&
           target->disposition == LifecycleRestartTargetDisposition::kAdmitted;
  };
  if (admitted_target(member)) {
    return true;
  }
  return std::ranges::any_of(plan.lifecycle_restart_closures, [&](const auto& closure) {
    return admitted_target(closure.target) &&
           std::ranges::find(closure.closure, member) != closure.closure.end();
  });
}

/// Retains an ordinary readiness witness until the fused region can acquire its fixed egress slot.
/// Polling, retirement, lifecycle generation, and trigger timing remain owned by the wrapped
/// binding.
class CUDAGraphOutputReadinessBinding final : public PortBinding {
 public:
  CUDAGraphOutputReadinessBinding(std::unique_ptr<PortBinding> binding,
                                  RuntimeCUDAGraphTensorResources& tensors,
                                  std::size_t region_index) noexcept
      : binding_(std::move(binding)), tensors_(&tensors), region_index_(region_index) {}

  [[nodiscard]] expected<void, Error> poll(TimePoint observed_at) noexcept override {
    return binding_->poll(observed_at);
  }
  [[nodiscard]] expected<void, Error> poll_blocking(TimePoint observed_at) noexcept override {
    return binding_->poll_blocking(observed_at);
  }
  [[nodiscard]] expected<void, Error> poll_after_contention(
      TimePoint observed_at) noexcept override {
    return binding_->poll_after_contention(observed_at);
  }
  [[nodiscard]] std::optional<PortBindingPollTarget> poll_target() const noexcept override {
    return binding_->poll_target();
  }
  [[nodiscard]] SourceId source() const noexcept override { return binding_->source(); }
  [[nodiscard]] bool last_poll_incomplete() const noexcept override {
    return binding_->last_poll_incomplete();
  }
  [[nodiscard]] bool last_poll_lock_contended() const noexcept override {
    return binding_->last_poll_lock_contended();
  }
  [[nodiscard]] expected<void, Error> enqueue_window_timeout(
      PortId port, TimePoint terminal_time) noexcept override {
    return binding_->enqueue_window_timeout(port, terminal_time);
  }
  [[nodiscard]] expected<void, Error> retire(std::uint64_t source_sequence) noexcept override {
    auto retired = binding_->retire(source_sequence);
    if (retired && retained_sequence_ == source_sequence) {
      retained_sequence_.reset();
    }
    return retired;
  }
  [[nodiscard]] expected<void, Error> retire_pending_events() noexcept override {
    // Restart must retire events from the current generation even while scheduling is held back by
    // the region's occupied output slot.
    auto retired = binding_->retire_pending_events();
    if (retired) {
      output_blocked_ = false;
    }
    return retired;
  }
  void cancel() noexcept override {
    binding_->cancel();
    retained_sequence_.reset();
  }
  [[nodiscard]] expected<std::uint64_t, Error> advance_source_generation() noexcept override {
    return binding_->advance_source_generation();
  }

  [[nodiscard]] expected<std::size_t, Error> drain(
      std::span<RuntimeEvent> destination) noexcept override {
    if (destination.empty()) {
      return 0U;
    }
    // Keep at most one region activation beyond the wrapped source. The fixed egress is not
    // claimed until compute() starts, so its availability alone cannot distinguish two trigger
    // witnesses drained before either activation runs.
    if (retained_sequence_) {
      return 0U;
    }
    auto available = tensors_->output_available(region_index_);
    if (!available) {
      return make_unexpected(std::move(available.error()));
    }
    if (!*available) {
      // Leave the trigger witness in its original binding. The controller's bounded source wait
      // will rescan after downstream Tensor retirement can return the fixed slot.
      output_blocked_ = true;
      return 0U;
    }
    output_blocked_ = false;
    auto drained = binding_->drain(destination.first(1U));
    if (drained && *drained == 1U) {
      retained_sequence_ = destination.front().source_sequence;
    }
    return drained;
  }

  [[nodiscard]] std::size_t capacity() const noexcept override { return binding_->capacity(); }
  [[nodiscard]] bool closed() const noexcept override { return binding_->closed(); }
  [[nodiscard]] std::optional<TimePoint> next_wake() const noexcept override {
    // A due clock remains due while its witness is retained. Use the controller's bounded wait
    // while output is blocked instead of immediately rescanning that same due coordinate.
    // Releasing a fixed slot does not directly wake a consumer in another partition, so retry can
    // wait until the bounded scheduler timeout. Revisit this general scheduling limitation if
    // measurements show a material latency impact.
    return output_blocked_ || retained_sequence_ ? std::nullopt : binding_->next_wake();
  }

 private:
  std::unique_ptr<PortBinding> binding_;
  RuntimeCUDAGraphTensorResources* tensors_{};
  std::size_t region_index_{};
  bool output_blocked_{};
  std::optional<std::uint64_t> retained_sequence_;
};

}  // namespace

class RuntimeCUDAGraphRegionResources::Impl final {
 public:
  // One record owns the captured graph, executable, and streams used by a fused region operator.
  struct Region final {
    GraphId graph{};
    RegionId region{};
    VertexId fused_operator{};
    DeviceId device{};
    std::string stream_id;
    cudaStream_t launch_stream{};
    cudaStream_t capture_stream{};
#if HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE
    std::shared_ptr<const cuda_execution_detail::CudaGreenContextDriver> capture_stream_driver;
#endif
    bool launch_uses_green_context{};
    std::vector<RuntimeOperatorInstance*> members;
    cudaGraph_t graph_handle{};
    cudaGraphExec_t executable{};
    RuntimeOperatorInstance* source_instance{};
    SourceId readiness_source{};
    bool has_egress{};
    bool output_readiness_bound{};
    std::unique_ptr<CUDAGraphRegionOperator> fused;
    bool captured{};
    bool submitted_work{};

    Region() = default;
    Region(const Region&) = delete;
    Region& operator=(const Region&) = delete;
    Region(Region&& other) noexcept
        : graph(std::exchange(other.graph, {})),
          region(std::exchange(other.region, {})),
          fused_operator(std::exchange(other.fused_operator, {})),
          device(std::exchange(other.device, {})),
          stream_id(std::move(other.stream_id)),
          launch_stream(std::exchange(other.launch_stream, nullptr)),
          capture_stream(std::exchange(other.capture_stream, nullptr)),
#if HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE
          capture_stream_driver(std::move(other.capture_stream_driver)),
#endif
          launch_uses_green_context(std::exchange(other.launch_uses_green_context, false)),
          members(std::move(other.members)),
          graph_handle(std::exchange(other.graph_handle, nullptr)),
          executable(std::exchange(other.executable, nullptr)),
          source_instance(std::exchange(other.source_instance, nullptr)),
          readiness_source(std::exchange(other.readiness_source, {})),
          has_egress(std::exchange(other.has_egress, false)),
          output_readiness_bound(std::exchange(other.output_readiness_bound, false)),
          fused(std::move(other.fused)),
          captured(std::exchange(other.captured, false)),
          submitted_work(std::exchange(other.submitted_work, false)) {
    }
    Region& operator=(Region&&) = delete;

    ~Region() {
      if (source_instance != nullptr && source_instance->fused_cuda_graph_operator == fused.get()) {
        source_instance->fused_cuda_graph_operator = nullptr;
      }
      const ScopedCudaDevice selected{device};
      if (selected.active() && submitted_work && launch_stream != nullptr) {
        static_cast<void>(cudaStreamSynchronize(launch_stream));
      }
      // Destruction can still succeed in the handle's current context when selecting the planned
      // device failed. Attempt every release instead of abandoning native handles unconditionally.
      if (executable != nullptr) {
        static_cast<void>(cudaGraphExecDestroy(executable));
      }
      if (graph_handle != nullptr) {
        static_cast<void>(cudaGraphDestroy(graph_handle));
      }
      if (capture_stream != nullptr) {
        bool destroyed{};
#if HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE
        if (capture_stream_driver != nullptr && capture_stream_driver->stream_destroy != nullptr) {
          static_cast<void>(
              capture_stream_driver->stream_destroy(reinterpret_cast<CUstream>(capture_stream)));
          destroyed = true;
        }
#endif
        if (!destroyed) {
          static_cast<void>(cudaStreamDestroy(capture_stream));
        }
      }
    }
  };

  std::unique_ptr<RuntimeCUDAGraphTensorResources> tensors;
  std::shared_ptr<RuntimeCudaExecutionTestControl> test_control;
  bool shutdown_complete{};
  // Regions are destroyed first, while their captured Tensor storage is still alive.
  std::vector<Region> regions;
  [[nodiscard]] Region* find_region(VertexId fused_operator) noexcept {
    const auto found = std::ranges::find(regions, fused_operator, &Region::fused_operator);
    return found == regions.end() ? nullptr : &*found;
  }
};

RuntimeCUDAGraphRegionResources::RuntimeCUDAGraphRegionResources(
    std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

RuntimeCUDAGraphRegionResources::~RuntimeCUDAGraphRegionResources() = default;

CUDAGraphRegionOperator::CUDAGraphRegionOperator(RuntimeCUDAGraphRegionResources* resources,
                                                 std::size_t region_index) noexcept
    : resources_(resources), region_index_(region_index) {}

expected<void, Error> CUDAGraphRegionOperator::compute(ExecutionContext& context) {
  // RunSession binds the resource and its fixed region index before scheduling starts.
  assert(resources_ != nullptr);
  return resources_->launch(region_index_, context.cuda_stream().get());
}

expected<std::unique_ptr<RuntimeCUDAGraphRegionResources>, Error>
RuntimeCUDAGraphRegionResources::create(
    const ExecutionPlanState& plan, std::span<RuntimeOperatorInstance> operators,
    RuntimeCUDAGraphTensorPoolBinding tensor_pools, ProcessGroupId process_group,
    RuntimeCUDAGraphChannelBindings channel_bindings,
    CUDAGraphRuntimeRegionIdentity* failure_region,
    std::shared_ptr<RuntimeCudaExecutionTestControl> test_control) {
  return create_impl(plan,
                     operators,
                     tensor_pools,
                     process_group,
                     channel_bindings,
                     failure_region,
                     std::move(test_control));
}

expected<std::unique_ptr<RuntimeCUDAGraphRegionResources>, Error>
RuntimeCUDAGraphRegionResources::create(const ExecutionPlanState& plan,
                                        std::span<RuntimeOperatorInstance> operators,
                                        ProcessGroupId process_group) {
  return create_impl(plan, operators, {}, process_group, {}, nullptr, {});
}

expected<std::unique_ptr<RuntimeCUDAGraphRegionResources>, Error>
RuntimeCUDAGraphRegionResources::create_impl(
    const ExecutionPlanState& plan, std::span<RuntimeOperatorInstance> operators,
    RuntimeCUDAGraphTensorPoolBinding tensor_pools, ProcessGroupId process_group,
    RuntimeCUDAGraphChannelBindings channel_bindings,
    CUDAGraphRuntimeRegionIdentity* failure_region,
    std::shared_ptr<RuntimeCudaExecutionTestControl> test_control) {
  if (failure_region != nullptr) {
    *failure_region = {};
  }
  auto impl = std::make_unique<Impl>();
  impl->test_control = std::move(test_control);
  impl->regions.reserve(plan.cuda_graph_execution_regions.size());

  for (const CUDAGraphExecutionRegionPlanRecord& execution : plan.cuda_graph_execution_regions) {
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
    if (std::ranges::any_of(impl->regions, [&](const auto& region) {
          return region.fused_operator == execution.fused_operator;
        })) {
      return make_unexpected(Error{ErrorCode::kIncompatible, "CUDA_GRAPH_RUNTIME_PLAN_INVALID"});
    }
    // Join this physical execution record to the validated member order captured in the same
    // graph and region.
    const auto* capture =
        find_unique_cuda_graph_capture_plan_by_region(plan, execution.graph, execution.region);
    if (capture == nullptr || capture->members.empty() ||
        capture->members.front().vertex != execution.fused_operator) {
      return make_unexpected(Error{ErrorCode::kIncompatible, "CUDA_GRAPH_RUNTIME_PLAN_INVALID"});
    }

    // Resolve the fused source instance and retain its compiler-selected stream for graph replay.
    RuntimeOperatorInstance* const source =
        find_unique_operator(operators, execution.fused_operator);
    if (!execution.device.valid() || source == nullptr ||
        source->partition != execution.partition ||
        source->execution_lane != execution.execution_lane || source->compute_stream == nullptr ||
        source->compute_device != execution.device) {
      return make_unexpected(Error{ErrorCode::kIncompatible, "CUDA_GRAPH_RUNTIME_STREAM_MISMATCH"});
    }
    cudaStream_t source_stream = source->compute_stream;

    Impl::Region runtime_region;
    runtime_region.graph = execution.graph;
    runtime_region.region = execution.region;
    runtime_region.fused_operator = execution.fused_operator;
    runtime_region.device = execution.device;
    runtime_region.stream_id = execution.stream_id;
    runtime_region.launch_stream = source_stream;
    runtime_region.has_egress = execution.egress.has_value();
    if (runtime_region.has_egress &&
        (source->temporal_trigger == nullptr || !source->temporal_trigger->readiness_source)) {
      return make_unexpected(Error{ErrorCode::kIncompatible, "CUDA_GRAPH_RUNTIME_PLAN_INVALID"});
    }
    if (runtime_region.has_egress) {
      runtime_region.readiness_source = source->temporal_trigger->readiness_source;
    }
    const auto stream_plan =
        std::ranges::find_if(plan.cuda_compute_streams, [&](const auto& stream) {
          return stream.process_group == execution.process_group &&
                 stream.device == execution.device && stream.stream_id == execution.stream_id;
        });
    if (stream_plan == plan.cuda_compute_streams.end() ||
        stream_plan->green_context_id != execution.green_context_id) {
      return make_unexpected(Error{ErrorCode::kIncompatible, "CUDA_GRAPH_RUNTIME_STREAM_MISMATCH"});
    }
    runtime_region.launch_uses_green_context = stream_plan->green_context_id.has_value();
    runtime_region.source_instance = source;
    runtime_region.fused = std::unique_ptr<CUDAGraphRegionOperator>{
        new CUDAGraphRegionOperator{nullptr, impl->regions.size()}};
    runtime_region.members.reserve(capture->members.size());
    // Bind every original member for startup capture and reject any runtime reconstruction that no
    // longer matches the planned placement or immutable state. Member compute streams are not
    // relevant here: capture invokes every member on the region-wide private capture stream.
    for (const CUDAGraphCaptureMemberRecord& member : capture->members) {
      RuntimeOperatorInstance* const instance = find_unique_operator(operators, member.vertex);
      if (instance == nullptr || instance->partition != execution.partition ||
          instance->execution_lane != execution.execution_lane) {
        return make_unexpected(Error{ErrorCode::kIncompatible, "CUDA_GRAPH_RUNTIME_PLAN_INVALID"});
      }
      if (instance->runtime_parameter_updates_enabled ||
          instance->lifecycle_parameter_updates_enabled ||
          lifecycle_restart_can_reconstruct(plan, member.operator_ref)) {
        return make_unexpected(Error{ErrorCode::kIncompatible, "CUDA_GRAPH_CAPTURE_STATE_MUTABLE"});
      }
      runtime_region.members.push_back(instance);
    }
    impl->regions.push_back(std::move(runtime_region));
  }

  // Construct fixed Tensor storage after all member bindings have been resolved.
  auto tensors = RuntimeCUDAGraphTensorResources::create(
      plan, operators, tensor_pools, process_group, channel_bindings, failure_region);
  if (!tensors) {
    return make_unexpected(std::move(tensors.error()));
  }
  impl->tensors = std::move(*tensors);
  if (impl->tensors->size() != impl->regions.size()) {
    return make_unexpected(Error{ErrorCode::kIncompatible, "CUDA_GRAPH_RUNTIME_PLAN_INVALID"});
  }
  for (std::size_t index = 0U; index < impl->regions.size(); ++index) {
    if (!impl->tensors->matches(index, impl->regions[index].fused_operator) ||
        impl->tensors->has_egress(index) != impl->regions[index].has_egress) {
      return make_unexpected(Error{ErrorCode::kIncompatible, "CUDA_GRAPH_RUNTIME_PLAN_INVALID"});
    }
  }

  auto result = std::unique_ptr<RuntimeCUDAGraphRegionResources>{
      new RuntimeCUDAGraphRegionResources{std::move(impl)}};
  // Replace each source instance's steady-state callback with its prebound fused operator.
  for (Impl::Region& region : result->impl_->regions) {
    region.fused->resources_ = result.get();
    region.source_instance->fused_cuda_graph_operator = region.fused.get();
  }
  if (failure_region != nullptr) {
    *failure_region = {};
  }
  return result;
}

expected<void, Error> RuntimeCUDAGraphRegionResources::capture_all(
    void* state, CaptureMemberInvoke invoke, CUDAGraphRuntimeRegionIdentity* failure_region) {
  if (failure_region != nullptr) {
    *failure_region = {};
  }
  if (invoke == nullptr) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "CUDA_GRAPH_CAPTURE_BINDING_INVALID"});
  }
  for (Impl::Region& region : impl_->regions) {
    if (failure_region != nullptr) {
      *failure_region = CUDAGraphRuntimeRegionIdentity{
          .graph = region.graph,
          .region = region.region,
          .fused_operator = region.fused_operator,
      };
    }
    if (region.captured || region.graph_handle != nullptr || region.executable != nullptr) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "CUDA_GRAPH_CAPTURE_STATE_INVALID"});
    }
    const ScopedCudaDevice selected{region.device};
    if (!selected.active()) {
      return make_unexpected(
          Error{ErrorCode::kNotSupported, "RUNTIME_COMPUTE_CUDA_DEVICE_UNAVAILABLE"});
    }

    // Create a private nonblocking capture stream in the source stream's CUDA context. The
    // captured executable will later be launched on the source operator's designated stream.
    if (region.capture_stream == nullptr) {
      if (region.launch_uses_green_context) {
#if HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE
        auto driver = cuda_execution_detail::cuda_green_context_driver();
        if (!driver || (*driver)->stream_get_green_context == nullptr ||
            (*driver)->green_context_stream_create == nullptr) {
          return make_unexpected(
              Error{ErrorCode::kFailure, "CUDA_GRAPH_CAPTURE_STREAM_CREATION_FAILED"});
        }
        CUgreenCtx green_context{};
        CUresult driver_status = (*driver)->stream_get_green_context(
            reinterpret_cast<CUstream>(region.launch_stream), &green_context);
        if (driver_status != CUDA_SUCCESS || green_context == nullptr) {
          return make_unexpected(
              Error{ErrorCode::kFailure, "CUDA_GRAPH_CAPTURE_STREAM_CONTEXT_MISMATCH"});
        }
        CUstream capture_stream{};
        driver_status = (*driver)->green_context_stream_create(
            &capture_stream, green_context, CU_STREAM_NON_BLOCKING, 0);
        if (driver_status != CUDA_SUCCESS || capture_stream == nullptr) {
          return make_unexpected(
              Error{ErrorCode::kFailure, "CUDA_GRAPH_CAPTURE_STREAM_CREATION_FAILED"});
        }
        region.capture_stream = reinterpret_cast<cudaStream_t>(capture_stream);
        region.capture_stream_driver = std::move(*driver);
#else
        return make_unexpected(
            Error{ErrorCode::kNotSupported, "CUDA_GRAPH_GREEN_CONTEXT_UNAVAILABLE"});
#endif
      } else {
        const cudaError_t stream_status =
            cudaStreamCreateWithFlags(&region.capture_stream, cudaStreamNonBlocking);
        if (stream_status != cudaSuccess) {
          return make_unexpected(cuda_graph_error("capture stream creation", stream_status));
        }
      }
    }
    cudaError_t status{};
    // Bind the region's fixed Tensors before CUDA starts recording member compute operations.
    auto tensor_capture =
        impl_->tensors->begin_capture(region.graph, region.region, region.capture_stream);
    if (!tensor_capture) {
      return make_unexpected(std::move(tensor_capture.error()));
    }
    // Capture begins, invokes every member, and ends on this RunSession worker thread. Thread-local
    // mode confines capture-safety checks to that thread while other application threads use CUDA.
    status = cudaStreamBeginCapture(region.capture_stream, cudaStreamCaptureModeThreadLocal);
    if (status != cudaSuccess) {
      static_cast<void>(impl_->tensors->end_capture(region.graph, region.region));
      return make_unexpected(cuda_graph_error("capture begin", status));
    }

    expected<void, Error> callback_result;
    // Invoke each member once in region order, passing the private capture stream through its
    // ExecutionContext so every enqueued CUDA operation enters this graph capture.
    for (RuntimeOperatorInstance* const member : region.members) {
      callback_result = impl_->tensors->prepare_capture_member(member->vertex);
      if (callback_result) {
        callback_result = invoke(state, *member, region.device, region.capture_stream);
      }
      if (!callback_result) {
        break;
      }
    }

    // End native capture even after a member failure, then restore the temporary Tensor input
    // bindings. This leaves neither CUDA nor the operator ports in capture state.
    status = cudaStreamEndCapture(region.capture_stream, &region.graph_handle);
    if (impl_->test_control != nullptr &&
        impl_->test_control->should_fail(RuntimeCudaExecutionFailurePoint::kCUDAGraphCaptureEnd)) {
      status = cudaErrorUnknown;
    }
    auto tensor_capture_end = impl_->tensors->end_capture(region.graph, region.region);
    // Keep any graph returned by an aborted capture for the common checked shutdown path. That
    // path preserves the handle and quarantines the device if native destruction fails.
    if (!callback_result) {
      // A failed restore leaves operator ports bound to capture-only state, so report it before the
      // member error that caused capture to stop.
      if (!tensor_capture_end) {
        return make_unexpected(std::move(tensor_capture_end.error()));
      }
      if (status != cudaSuccess) {
        return make_unexpected(cuda_graph_error("capture end", status));
      }
      return make_unexpected(std::move(callback_result).error());
    }
    if (!tensor_capture_end) {
      return make_unexpected(std::move(tensor_capture_end.error()));
    }
    if (status != cudaSuccess) {
      return make_unexpected(cuda_graph_error("capture end", status));
    }
    if (!tensor_capture_end->outputs_complete) {
      return make_unexpected(Error{ErrorCode::kFailure, "CUDA_GRAPH_CAPTURE_OUTPUT_MISSING"});
    }
    if (region.graph_handle == nullptr) {
      return make_unexpected(Error{ErrorCode::kFailure, "CUDA_GRAPH_CAPTURE_EMPTY"});
    }
    std::size_t captured_node_count{};
    status = cudaGraphGetNodes(region.graph_handle, nullptr, &captured_node_count);
    if (status != cudaSuccess) {
      return make_unexpected(cuda_graph_error("node inspection", status));
    }
    if (captured_node_count == 0U) {
      return make_unexpected(Error{ErrorCode::kFailure, "CUDA_GRAPH_CAPTURE_EMPTY"});
    }
    status = cudaGraphInstantiate(&region.executable, region.graph_handle, 0U);
    if (status != cudaSuccess) {
      return make_unexpected(cuda_graph_error("instantiation", status));
    }
    region.captured = true;
  }
  if (failure_region != nullptr) {
    *failure_region = {};
  }
  return {};
}

expected<void, Error> RuntimeCUDAGraphRegionResources::bind_output_readiness(
    std::vector<std::unique_ptr<PortBinding>>& bindings) noexcept {
  if (impl_ == nullptr || impl_->tensors == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, "CUDA_GRAPH_RUNTIME_PLAN_INVALID"});
  }
  for (std::size_t region_index = 0U; region_index < impl_->regions.size(); ++region_index) {
    Impl::Region& region = impl_->regions[region_index];
    if (!region.has_egress) {
      continue;
    }
    auto match = bindings.end();
    for (auto candidate = bindings.begin(); candidate != bindings.end(); ++candidate) {
      if (*candidate == nullptr || (*candidate)->source() != region.readiness_source) {
        continue;
      }
      if (match != bindings.end()) {
        return make_unexpected(Error{ErrorCode::kIncompatible, "CUDA_GRAPH_RUNTIME_PLAN_INVALID"});
      }
      match = candidate;
    }
    // This binding list may belong to another execution partition. The source is wrapped when
    // bind_ready_sources() processes the partition that owns the fused operator.
    if (match == bindings.end()) {
      continue;
    }
    // A region source belongs to one partition and must receive this scheduling check once.
    if (region.output_readiness_bound) {
      return make_unexpected(Error{ErrorCode::kIncompatible, "CUDA_GRAPH_RUNTIME_PLAN_INVALID"});
    }
    auto replacement = std::unique_ptr<CUDAGraphOutputReadinessBinding>{
        new (std::nothrow)
            CUDAGraphOutputReadinessBinding{std::move(*match), *impl_->tensors, region_index}};
    if (replacement == nullptr) {
      return make_unexpected(
          Error{ErrorCode::kResourceExhausted, "CUDA_GRAPH_RUNTIME_PLAN_INVALID"});
    }
    *match = std::move(replacement);
    region.output_readiness_bound = true;
  }
  return {};
}

expected<void, Error> RuntimeCUDAGraphRegionResources::validate_output_readiness_bindings()
    const noexcept {
  if (impl_ == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, "CUDA_GRAPH_RUNTIME_PLAN_INVALID"});
  }
  for (const Impl::Region& region : impl_->regions) {
    if (region.has_egress && !region.output_readiness_bound) {
      return make_unexpected(Error{ErrorCode::kIncompatible, "CUDA_GRAPH_RUNTIME_PLAN_INVALID"});
    }
  }
  return {};
}

expected<void, Error> RuntimeCUDAGraphRegionResources::launch(VertexId fused_operator) noexcept {
  const auto found =
      std::ranges::find(impl_->regions, fused_operator, &Impl::Region::fused_operator);
  if (found == impl_->regions.end()) {
    return make_unexpected(
        cuda_graph_named_error(ErrorCode::kNotReady, "CUDA_GRAPH_LAUNCH_STATE_INVALID"));
  }
  // Select the region's CUDA device for this identity-based diagnostic and test entry point.
  // Production compute enters launch(region_index) with the device and index already selected.
  const ScopedCudaDevice selected{found->device};
  if (!selected.active()) {
    return make_unexpected(cuda_graph_named_error(ErrorCode::kNotSupported,
                                                  "RUNTIME_COMPUTE_CUDA_DEVICE_UNAVAILABLE"));
  }
  const std::size_t region_index =
      static_cast<std::size_t>(std::distance(impl_->regions.begin(), found));
  return launch(region_index, found->launch_stream);
}

expected<void, Error> RuntimeCUDAGraphRegionResources::launch(std::size_t region_index,
                                                              cudaStream_t stream) noexcept {
  // Use the region index that create() bound into this fused operator before scheduling started.
  if (impl_ == nullptr || region_index >= impl_->regions.size()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  Impl::Region& region = impl_->regions[region_index];
  if (!region.captured || region.executable == nullptr) {
    return make_unexpected(
        cuda_graph_named_error(ErrorCode::kNotReady, "CUDA_GRAPH_LAUNCH_STATE_INVALID"));
  }
  if (stream == nullptr || stream != region.launch_stream) {
    return make_unexpected(
        cuda_graph_named_error(ErrorCode::kNotSupported, "CUDA_GRAPH_RUNTIME_STREAM_MISMATCH"));
  }

  // Acquire the same one-slot RuntimePool output used during capture. If downstream still owns
  // the preceding result, ordinary allocate_tensor() returns kResourceExhausted.
  auto prepared = impl_->tensors->begin_launch(region_index, stream);
  if (!prepared) {
    return make_unexpected(std::move(prepared.error()));
  }

  const cudaError_t status =
      impl_->test_control != nullptr &&
              impl_->test_control->should_fail(RuntimeCudaExecutionFailurePoint::kCUDAGraphLaunch)
          ? cudaErrorUnknown
          : cudaGraphLaunch(region.executable, stream);
  if (status != cudaSuccess) {
    impl_->tensors->cancel_launch(region_index);
    return make_unexpected(cuda_graph_error("launch", status));
  }
  region.submitted_work = true;
  return impl_->tensors->finish_launch(region_index);
}

expected<void, Error> RuntimeCUDAGraphRegionResources::quiesce(
    RuntimeCudaExecutionDiagnostic* diagnostic) noexcept {
  if (diagnostic != nullptr) {
    *diagnostic = {};
  }
  for (Impl::Region& region : impl_->regions) {
    if (!region.submitted_work) {
      continue;
    }
    const ScopedCudaDevice selected{region.device};
    if (!selected.active()) {
      assign_cuda_graph_diagnostic(diagnostic,
                                   LifecycleStage::kArm,
                                   "cuda_graph_device_selection",
                                   region.device,
                                   region.stream_id,
                                   region.graph,
                                   region.region,
                                   region.fused_operator);
      return make_unexpected(cuda_graph_named_error(ErrorCode::kNotSupported,
                                                    "RUNTIME_COMPUTE_CUDA_DEVICE_UNAVAILABLE"));
    }
    cudaError_t status = cudaStreamSynchronize(region.launch_stream);
    if (impl_->test_control != nullptr &&
        impl_->test_control->should_fail(RuntimeCudaExecutionFailurePoint::kCUDAGraphQuiesce)) {
      status = cudaErrorUnknown;
    }
    if (status != cudaSuccess) {
      assign_cuda_graph_diagnostic(diagnostic,
                                   LifecycleStage::kArm,
                                   "cuda_graph_stream_quiesce",
                                   region.device,
                                   region.stream_id,
                                   region.graph,
                                   region.region,
                                   region.fused_operator,
                                   static_cast<std::int32_t>(status));
      return make_unexpected(cuda_graph_error("stream quiesce", status));
    }
    region.submitted_work = false;
  }
  return {};
}

expected<void, Error> RuntimeCUDAGraphRegionResources::shutdown(
    RuntimeCudaExecutionDiagnostic* diagnostic) noexcept {
  if (diagnostic != nullptr) {
    *diagnostic = {};
  }
  if (impl_ == nullptr || impl_->shutdown_complete) {
    return {};
  }
  if (auto synchronized = quiesce(diagnostic); !synchronized) {
    return synchronized;
  }

  std::optional<Error> first_failure;
  std::optional<RuntimeCudaExecutionDiagnostic> first_diagnostic;
  const auto record_failure =
      [&](const Impl::Region& region, std::string_view operation, std::int32_t status) {
        if (first_failure) {
          return;
        }
        first_failure = cuda_graph_shutdown_error(operation, status);
        first_diagnostic.emplace();
        assign_cuda_graph_diagnostic(&*first_diagnostic,
                                     LifecycleStage::kRelease,
                                     operation,
                                     region.device,
                                     region.stream_id,
                                     region.graph,
                                     region.region,
                                     region.fused_operator,
                                     status);
      };
  const auto injects = [&](RuntimeCudaExecutionFailurePoint point) {
    return impl_->test_control != nullptr && impl_->test_control->should_fail(point);
  };

  for (Impl::Region& region : impl_->regions) {
    const ScopedCudaDevice selected{region.device};
    if (!selected.active()) {
      record_failure(region, "cuda_graph_device_selection", -1);
    }

    if (region.executable != nullptr) {
      const cudaError_t status = injects(RuntimeCudaExecutionFailurePoint::kCUDAGraphExecDestroy)
                                     ? cudaErrorUnknown
                                     : cudaGraphExecDestroy(region.executable);
      if (status == cudaSuccess) {
        region.executable = nullptr;
      } else {
        record_failure(region, "cuda_graph_exec_destroy", static_cast<std::int32_t>(status));
      }
    }
    if (region.graph_handle != nullptr) {
      const cudaError_t status = injects(RuntimeCudaExecutionFailurePoint::kCUDAGraphDestroy)
                                     ? cudaErrorUnknown
                                     : cudaGraphDestroy(region.graph_handle);
      if (status == cudaSuccess) {
        region.graph_handle = nullptr;
      } else {
        record_failure(region, "cuda_graph_destroy", static_cast<std::int32_t>(status));
      }
    }
    if (region.capture_stream != nullptr) {
      std::int32_t status = static_cast<std::int32_t>(cudaSuccess);
      if (injects(RuntimeCudaExecutionFailurePoint::kCUDAGraphCaptureStreamDestroy)) {
        status = static_cast<std::int32_t>(cudaErrorUnknown);
      } else {
        bool used_driver{};
#if HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE
        if (region.capture_stream_driver != nullptr &&
            region.capture_stream_driver->stream_destroy != nullptr) {
          status = static_cast<std::int32_t>(region.capture_stream_driver->stream_destroy(
              reinterpret_cast<CUstream>(region.capture_stream)));
          used_driver = true;
        }
#endif
        if (!used_driver) {
          status = static_cast<std::int32_t>(cudaStreamDestroy(region.capture_stream));
        }
      }
      if (status == static_cast<std::int32_t>(cudaSuccess)) {
        region.capture_stream = nullptr;
#if HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE
        region.capture_stream_driver.reset();
#endif
      } else {
        record_failure(region, "cuda_graph_capture_stream_destroy", status);
      }
    }
  }

  if (first_failure) {
    if (diagnostic != nullptr && first_diagnostic) {
      *diagnostic = std::move(*first_diagnostic);
    }
    return make_unexpected(std::move(*first_failure));
  }
  impl_->shutdown_complete = true;
  return {};
}

void RuntimeCUDAGraphRegionResources::detach_session_state() noexcept {
  if (impl_ == nullptr) {
    return;
  }
  if (impl_->tensors != nullptr) {
    impl_->tensors->detach_session_state();
  }
  // Operator storage can be released after detachment. Remove every back-reference before the
  // region destructor runs.
  for (Impl::Region& region : impl_->regions) {
    if (region.source_instance != nullptr &&
        region.source_instance->fused_cuda_graph_operator == region.fused.get()) {
      region.source_instance->fused_cuda_graph_operator = nullptr;
    }
    region.source_instance = nullptr;
  }
}

std::size_t RuntimeCUDAGraphRegionResources::size() const noexcept {
  return impl_ != nullptr ? impl_->regions.size() : 0U;
}

}  // namespace holoscan::detail

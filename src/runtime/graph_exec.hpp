/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime_api.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>

#include <holomq/cuda/stream_lifetime.hpp>
#include "../compiler/execution_plan_state.hpp"
#include "../core/diagnostic_context.hpp"
#include "../core/tag_internal.hpp"
#include "../time/clock_registry_internal.hpp"
#include "parameter_update_coordinator.hpp"
#include "runtime_diagnostic_sink.hpp"
#include "runtime_package_capability.hpp"
#include "runtime_rate_state.hpp"
#include "runtime_replay_mismatch.hpp"

#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvtx3.hpp>

namespace holoscan::detail {

class RuntimeLifecycleExecutor;
class RuntimeNotificationSourceRegistry;
class ExecutionLaneSet;
class CUDAGraphRegionOperator;
struct LifecycleRestartExecutionOutcome;
struct LifecycleRestartExecutionRequest;
struct RuntimeLifecycleFailure;
struct LifecycleExecutionResult;

/// @brief Own one freshly reconstructed Operator and its session-local tag allocator.
struct RuntimeOperatorInstance {
  VertexId vertex{};
  OperatorInstanceId operator_id{};
  ExecutionPartitionId partition{};
  std::uint32_t execution_lane{};
  std::unique_ptr<OperatorBase> op;
  // Retain the verified replay declaration because advanced lifecycle member
  // trampolines are process-local and cannot enter the immutable plan.
  std::unique_ptr<OperatorSpec> replay_spec;
  std::unique_ptr<TagAllocator> tags;
  // Cold-bound immutable plan records keep activation dispatch from rescanning plan vectors.
  const TemporalTriggerPlanRecord* temporal_trigger{};
  const PredicateTriggerPlanRecord* predicate_trigger{};
  // Cold-bound compute resources are indexed with the reconstructed Operator so activation
  // dispatch never scans the Tensor resource registry. Host-only Operators retain the null/invalid
  // defaults and therefore issue no CUDA query or device switch.
  cudaStream_t compute_stream{};
  std::shared_ptr<const holomq::cuda::StreamLifetime> compute_stream_lifetime;
  DeviceId compute_device{};
  // Bound only on the region source. Capture still invokes `op`; steady-state dispatch invokes
  // this fused operator, which launches the executable CUDA Graph.
  CUDAGraphRegionOperator* fused_cuda_graph_operator{};
  // The plan's canonical percent-encoded rendered path grammar.
  std::string_view canonical_operator_path{};
  PinnedParameterHeader static_parameter_header{};
  std::span<const ParameterDerivedTermPlanRecord> parameter_derived_terms;
  bool runtime_parameter_updates_enabled{};
  bool lifecycle_parameter_updates_enabled{};
  // Bound from verified replay declarations, independent of channel/direct-continuation binding.
  bool has_output_ports{};
  nvtx3::message compute_nvtx_message{nvtxStringHandle_t{}};
  DiagnosticCapture failure_capture;
  TensorContractMismatchCapture tensor_failure;
};

/// @brief Own immutable plan state plus every fresh object for one RunSession.
class GraphExec {
 public:
  GraphExec(const GraphExec&) = delete;
  GraphExec& operator=(const GraphExec&) = delete;
  GraphExec(GraphExec&&) = delete;
  GraphExec& operator=(GraphExec&&) = delete;
  ~GraphExec();

  /// @brief Reconstruct and verify process-local executable objects before lifecycle entry.
  /// @param plan Immutable accepted state retained by the RunSession.
  /// @param run_generation Fresh nonzero identity of the owning RunSession.
  /// @param process_group Compile-admitted group owned by this process-local worker.
  /// @return Fresh isolated runtime state, or a stable startup failure.
  [[nodiscard]] static expected<std::unique_ptr<GraphExec>, Error> reconstruct(
      std::shared_ptr<const ExecutionPlanState> plan, RunGenerationId run_generation,
      ProcessGroupId process_group = {}, RuntimeReplayMismatchDetail* replay_mismatch = nullptr,
      RuntimeDiagnosticDetail* failure_detail = nullptr);

  [[nodiscard]] std::span<RuntimeOperatorInstance> operators() noexcept { return operators_; }
  [[nodiscard]] std::span<const RuntimeOperatorInstance> operators() const noexcept {
    return operators_;
  }
  /// Return whether this process reconstructed any callback-owning Operators.
  [[nodiscard]] bool has_process_local_callback_owners() const noexcept {
    return !operators_.empty();
  }
  /// @brief Resolve a local Operator receiver from its accepted plan-local vertex identity.
  /// @param vertex Nonzero vertex identity from the retained accepted plan.
  /// @return Receiver index while this GraphExec remains alive, or no value when the vertex is not
  /// an Operator owned by this process group.
  [[nodiscard]] std::optional<std::size_t> receiver_index(VertexId vertex) const noexcept;
  /// @brief Resolve a local Operator receiver from its accepted plan-local Operator identity.
  /// @param operator_id Nonzero generation-zero Operator identity from the retained accepted plan.
  /// @return Receiver index while this GraphExec remains alive, or no value when the Operator is
  /// not owned by this process group.
  [[nodiscard]] std::optional<std::size_t> receiver_index(
      OperatorInstanceId operator_id) const noexcept;
  /// @brief Resolve a mutable reconstructed Operator by accepted plan-local vertex identity.
  /// @param vertex Nonzero vertex identity from the retained accepted plan.
  /// @return Borrowed instance valid while this GraphExec remains alive, or nullptr when absent.
  [[nodiscard]] RuntimeOperatorInstance* operator_by_vertex(VertexId vertex) noexcept;
  /// @brief Resolve a reconstructed Operator by accepted plan-local vertex identity.
  /// @param vertex Nonzero vertex identity from the retained accepted plan.
  /// @return Borrowed instance valid while this GraphExec remains alive, or nullptr when absent.
  [[nodiscard]] const RuntimeOperatorInstance* operator_by_vertex(VertexId vertex) const noexcept;
  /// @brief Resolve a mutable reconstructed Operator by accepted plan-local Operator identity.
  /// @param operator_id Nonzero generation-zero Operator identity from the retained accepted plan.
  /// @return Borrowed instance valid while this GraphExec remains alive, or nullptr when absent.
  [[nodiscard]] RuntimeOperatorInstance* operator_by_id(OperatorInstanceId operator_id) noexcept;
  /// @brief Resolve a reconstructed Operator by accepted plan-local Operator identity.
  /// @param operator_id Nonzero generation-zero Operator identity from the retained accepted plan.
  /// @return Borrowed instance valid while this GraphExec remains alive, or nullptr when absent.
  [[nodiscard]] const RuntimeOperatorInstance* operator_by_id(
      OperatorInstanceId operator_id) const noexcept;
  [[nodiscard]] const ExecutionPlanState& plan() const noexcept { return *plan_; }
  [[nodiscard]] ProcessGroupId process_group() const noexcept { return process_group_; }
  [[nodiscard]] const ClockRef& default_clock() const noexcept { return default_clock_; }
  [[nodiscard]] const ClockRegistry& clock_registry() const noexcept { return clock_registry_; }
  /// Resolve one plan-local clock against this session's retained catalog generation.
  [[nodiscard]] expected<ClockRef, Error> resolve_clock(ClockId id) const noexcept;
  /// Return whether this session's catalog generation contains one plan-local clock.
  [[nodiscard]] bool has_clock(ClockId id) const noexcept { return clock_registry_.contains(id); }
  /// Read one plan-local clock without minting a reference to it; see ClockRegistry::now().
  [[nodiscard]] expected<TimePoint, Error> now(ClockId id) const noexcept {
    return clock_registry_.now(id);
  }
  /// Wait on one plan-local clock through scheduler-only registry authority.
  [[nodiscard]] expected<void, Error> wait_until(ClockId id,
                                                 std::int64_t timestamp_ns) const noexcept;
  /// @brief Construct lifecycle execution state for this reconstructed graph.
  /// @param run_generation Nonzero identity of the owning run.
  /// @param cancellation Optional borrowed cancellation flag that may be null.
  /// @param lanes Borrowed started lane set used to execute lifecycle callbacks.
  /// @param notifications Optional borrowed notification registry; may be null when the plan has
  /// no notification sources. This function does not take ownership of the registry.
  /// @return Success, or a typed error when lifecycle state cannot be initialized.
  /// @throws std::bad_alloc if bounded lifecycle execution storage cannot be allocated.
  [[nodiscard]] expected<void, Error> initialize_lifecycle(
      RunGenerationId run_generation, const std::atomic_bool* cancellation, ExecutionLaneSet* lanes,
      RuntimeNotificationSourceRegistry* notifications = nullptr, RuntimeRateState* rates = nullptr,
      RuntimeRateMaterializationBinding rate_materialization = {});
  [[nodiscard]] expected<void, Error> set_lifecycle_notifications(
      RuntimeNotificationSourceRegistry* notifications) noexcept;
  [[nodiscard]] bool lifecycle_startup() noexcept;
  [[nodiscard]] LifecycleRestartExecutionOutcome lifecycle_restart(
      const LifecycleRestartExecutionRequest& request) noexcept;
  [[nodiscard]] bool lifecycle_shutdown() noexcept;
  [[nodiscard]] std::string_view lifecycle_failure_token() const noexcept;
  [[nodiscard]] const Error& lifecycle_failure_error() const noexcept;
  [[nodiscard]] VertexId lifecycle_failure_vertex() const noexcept;
  [[nodiscard]] std::span<const RuntimeLifecycleFailure> lifecycle_failures() const noexcept;
  [[nodiscard]] const LifecycleExecutionResult* lifecycle_result() const noexcept;
  void clear_lifecycle_failure_evidence() noexcept;
  [[nodiscard]] ParameterUpdateCoordinator& parameter_updates() noexcept {
    return *parameter_updates_;
  }
  [[nodiscard]] std::unique_ptr<ParameterUpdateCoordinator> take_parameter_updates() noexcept {
    return std::move(parameter_updates_);
  }

 private:
  static constexpr std::size_t kNoReceiver = std::numeric_limits<std::size_t>::max();

  GraphExec(std::shared_ptr<const ExecutionPlanState> plan, ProcessGroupId process_group);

  std::shared_ptr<const ExecutionPlanState> plan_;
  ProcessGroupId process_group_{};
  std::vector<RuntimeOperatorInstance> operators_;
  std::vector<std::size_t> receiver_by_vertex_;
  std::vector<std::size_t> receiver_by_operator_id_;
  ClockRegistry clock_registry_;
  ClockRef default_clock_;
  std::unique_ptr<RuntimeLifecycleExecutor> lifecycle_;
  std::optional<RuntimePackageCapabilityMatchReport> package_capability_report_;
  std::unique_ptr<ParameterUpdateCoordinator> parameter_updates_;
};

}  // namespace holoscan::detail

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime_api.h>

#include <atomic>
#include <cstdint>
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
#include <holoscan/core/tensor_output_loan.hpp>

#include "../compiler/execution_plan_state.hpp"
#include "../core/tag_internal.hpp"
#include "../time/clock_registry_internal.hpp"
#include "parameter_update_coordinator.hpp"
#include "runtime_package_capability.hpp"
#include "runtime_replay_mismatch.hpp"

namespace holoscan::detail {

class RuntimeLifecycleExecutor;
class RuntimeNotificationSourceRegistry;
class ExecutionLaneSet;
struct LifecycleRestartExecutionOutcome;
struct LifecycleRestartExecutionRequest;

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
  // Cold-bound compute resources are indexed with the reconstructed Operator so activation
  // dispatch never scans the Tensor resource registry. Host-only Operators retain the null/invalid
  // defaults and therefore issue no CUDA query or device switch.
  cudaStream_t compute_stream{};
  DeviceId compute_device{};
  PinnedParameterHeader static_parameter_header{};
  std::span<const ParameterDerivedTermPlanRecord> parameter_derived_terms;
  bool runtime_parameter_updates_enabled{};
  bool lifecycle_parameter_updates_enabled{};
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
      ProcessGroupId process_group = {}, RuntimeReplayMismatchDetail* replay_mismatch = nullptr);

  [[nodiscard]] std::span<RuntimeOperatorInstance> operators() noexcept { return operators_; }
  [[nodiscard]] std::span<const RuntimeOperatorInstance> operators() const noexcept {
    return operators_;
  }
  [[nodiscard]] const ExecutionPlanState& plan() const noexcept { return *plan_; }
  [[nodiscard]] ProcessGroupId process_group() const noexcept { return process_group_; }
  [[nodiscard]] ClockRef default_clock() const noexcept { return default_clock_; }
  [[nodiscard]] const ClockRegistry& clock_registry() const noexcept { return clock_registry_; }
  /// Resolve one plan-local clock against this session's retained catalog generation.
  [[nodiscard]] expected<ClockRef, Error> resolve_clock(ClockId id) const noexcept;
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
      RuntimeNotificationSourceRegistry* notifications = nullptr);
  [[nodiscard]] bool lifecycle_startup() noexcept;
  [[nodiscard]] LifecycleRestartExecutionOutcome lifecycle_restart(
      const LifecycleRestartExecutionRequest& request) noexcept;
  [[nodiscard]] bool lifecycle_shutdown() noexcept;
  [[nodiscard]] std::string_view lifecycle_failure_token() const noexcept;
  [[nodiscard]] ParameterUpdateCoordinator& parameter_updates() noexcept {
    return *parameter_updates_;
  }

 private:
  GraphExec(std::shared_ptr<const ExecutionPlanState> plan, ProcessGroupId process_group);

  std::shared_ptr<const ExecutionPlanState> plan_;
  ProcessGroupId process_group_{};
  std::vector<RuntimeOperatorInstance> operators_;
  ClockRegistry clock_registry_;
  ClockRef default_clock_;
  std::unique_ptr<RuntimeLifecycleExecutor> lifecycle_;
  std::optional<RuntimePackageCapabilityMatchReport> package_capability_report_;
  std::unique_ptr<ParameterUpdateCoordinator> parameter_updates_;
};

}  // namespace holoscan::detail

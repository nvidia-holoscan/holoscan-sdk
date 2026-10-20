/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/threading/thread_placement.hpp>

#include "runtime_callback_context.hpp"

namespace holoscan::detail {

class RuntimePhaseCounters;

/// @brief Describe only the operating-system state applied by one Runtime lane.
///
/// Wake behavior and IRQ-isolation evidence are deliberately absent. They are
/// Runtime and admission policy respectively, not operating-system placement.
struct ExecutionLaneOsPlacement {
  std::optional<CPUAffinity> affinity;
  ThreadScheduling scheduling;
};

/// @brief Preserve the operating-system placement read back inside one lane.
struct ExecutionLanePlacementSnapshot {
  CPUAffinity affinity;
  SchedulingPolicy policy{SchedulingPolicy::kOther};
  std::optional<int> priority;
  std::optional<DeadlineScheduling> deadline;
  bool reset_on_fork{};
};

/// @brief Classify a placement-adapter failure without relying on message text.
enum class ExecutionLanePlacementErrorKind : std::uint8_t {
  kUnavailable = 0,  ///< The accepted operation is unavailable on this host or privilege set.
  kFailed = 1,       ///< The host operation was available but failed.
};

/// @brief Return typed placement-adapter failure evidence.
struct ExecutionLanePlacementError {
  ExecutionLanePlacementErrorKind kind{ExecutionLanePlacementErrorKind::kFailed};
  Error error;
};

/// @brief Abstract the platform-specific placement operations used inside lane threads.
///
/// Implementations must be safe for concurrent calls from distinct lane
/// threads. Runtime invokes each operation at most once per started lane.
class ExecutionLanePlacementAdapter {
 public:
  virtual ~ExecutionLanePlacementAdapter() = default;

  [[nodiscard]] virtual expected<void, ExecutionLanePlacementError> apply(
      const ExecutionLaneOsPlacement& placement) = 0;
  [[nodiscard]] virtual expected<ExecutionLanePlacementSnapshot, ExecutionLanePlacementError>
  snapshot() = 0;
  [[nodiscard]] virtual expected<void, ExecutionLanePlacementError> restore(
      const ExecutionLanePlacementSnapshot& snapshot) = 0;
};

/// @brief Identify the startup operation that produced one lane failure.
enum class ExecutionLaneStartupPhase : std::uint8_t {
  kNone = 0,
  kWorkerStart = 1,
  kPlacementApply = 2,
  kPlacementReadback = 3,
  kPlacementVerify = 4,
  kAllLanesBarrier = 5,
};

/// @brief Classify the immutable startup disposition of one lane.
enum class ExecutionLaneStartupDisposition : std::uint8_t {
  kPending = 0,
  kReady = 1,
  kUnavailable = 2,
  kFailed = 3,
  kStopped = 4,
};

/// @brief Preserve bounded lane-local startup and placement evidence.
struct ExecutionLaneStartupEvidence {
  std::uint32_t execution_lane{};
  ExecutionPartitionId partition{};
  WakeMode wake_mode{WakeMode::kSleepingWaitset};
  IRQIsolation irq_isolation{IRQIsolation::kNotRequested};
  ExecutionLaneStartupDisposition disposition{ExecutionLaneStartupDisposition::kPending};
  ExecutionLaneStartupPhase phase{ExecutionLaneStartupPhase::kNone};
  std::optional<ExecutionLanePlacementSnapshot> applied;
  std::optional<Error> error;
};

/// @brief Return the structural diagnostic token for one lane-startup result.
///
/// A platform/backend capability failure is projected as the canonical
/// `PLATFORM_CAPABILITY_UNAVAILABLE` structural diagnostic. The lane-local
/// error remains unchanged in \ref ExecutionLaneStartupEvidence so terminal
/// evidence can retain the exact placement cause independently.
[[nodiscard]] std::string_view execution_lane_startup_diagnostic_token(
    const ExecutionLaneStartupEvidence& evidence) noexcept;

/// @brief Validate an OS-placement request against a platform capability fact.
///
/// This platform-neutral helper lets portability tests exercise the guarded
/// non-native path without claiming native Windows or macOS execution. Default
/// placement is a no-op on a platform without an OS placement controller;
/// affinity or scheduling requests fail with typed unavailable evidence.
[[nodiscard]] expected<void, ExecutionLanePlacementError>
validate_execution_lane_platform_placement(bool platform_supports_os_placement,
                                           const ExecutionLaneOsPlacement& placement);

/// @brief Describe one allocation-free invocation submitted to an execution lane.
struct ExecutionLaneTask {
  void* context{};
  expected<void, Error> (*invoke)(void*) = nullptr;
  RuntimeCallbackContext callback_context{RuntimeCallbackContext::kNone};
  RuntimeCallbackIdentity callback_identity{};
};

/// @brief Describe one long-running Runtime owner task bound to an execution lane.
///
/// Owner tasks are Runtime infrastructure, not Operator callbacks. The task
/// context is borrowed from \ref ExecutionLaneSet::launch_owner_tasks until
/// \ref ExecutionLaneSet::wait_owner_tasks or \ref ExecutionLaneSet::join
/// returns.
struct ExecutionLaneOwnerTask {
  std::uint32_t execution_lane{};
  void* context{};
  expected<void, Error> (*invoke)(void*) = nullptr;
};

/// @brief Own the partition execution lanes for one process-local RunSession.
///
/// Worker-backed lanes have distinct workers only to realize their accepted
/// placement. One optionally selected inline lane instead applies placement
/// and executes on the thread that starts the set. All dispatches share one
/// controller-owned serial token, so at most one legacy task can execute across
/// the complete local set. A parallel scheduler may instead launch exactly one
/// persistent Runtime owner task per non-inline lane. Owner tasks execute
/// concurrently without the legacy dispatch token or per-activation handoff.
/// No task is accepted until every local lane has applied and read back
/// placement successfully.
class ExecutionLaneSet {
 public:
  using StartupEvidenceView = std::span<const ExecutionLaneStartupEvidence>;

  ExecutionLaneSet(const ExecutionLaneSet&) = delete;
  ExecutionLaneSet& operator=(const ExecutionLaneSet&) = delete;
  ExecutionLaneSet(ExecutionLaneSet&&) = delete;
  ExecutionLaneSet& operator=(ExecutionLaneSet&&) = delete;
  ~ExecutionLaneSet();

  /// @brief Copy and validate accepted lane records before creating any worker.
  /// @param scheduler Frozen scheduler geometry governing the lane set.
  /// @param lanes Sorted local subset of accepted global lane records copied by the
  /// returned set. Global execution-lane identities are retained.
  /// @param placement_adapter Optional shared placement adapter; the system adapter is used when
  /// absent.
  /// @param phase_counters Optional borrowed execution counters. When provided, they must outlive
  /// the returned lane set through \ref join or destruction.
  /// @param inline_execution_lane Optional lane whose placement and callbacks execute on the
  /// thread that calls \ref start. The lane creates no worker, and later dispatches from another
  /// thread are rejected. Inline-owner mode requires the materialized local set to contain exactly
  /// this one lane.
  /// @param unbound_worker_affinity Optional allowed CPU set restored inside worker-backed lanes
  /// that have no authored affinity. Explicit lane affinity takes precedence, and an inline lane
  /// continues to use the calling thread's placement.
  /// @return A stopped lane set, or a typed validation, placement, or capacity error.
  /// @throws std::bad_alloc if cold lane or worker storage cannot be allocated.
  [[nodiscard]] static expected<std::unique_ptr<ExecutionLaneSet>, Error> create(
      const SchedulerPlanRecord& scheduler, std::span<const ExecutionLanePlanRecord> lanes,
      std::shared_ptr<ExecutionLanePlacementAdapter> placement_adapter = {},
      RuntimePhaseCounters* phase_counters = nullptr,
      std::optional<std::uint32_t> inline_execution_lane = std::nullopt,
      std::optional<CPUAffinity> unbound_worker_affinity = std::nullopt);

  /// @brief Start all lanes and open the all-lanes placement-ready barrier.
  /// @return Success only after every lane has applied and verified placement.
  /// @return Typed evidence for the first unavailable or failed lane otherwise.
  [[nodiscard]] expected<void, ExecutionLaneStartupEvidence> start();

  /// @brief Execute one complete callback/commit unit on its accepted lane.
  ///
  /// The legacy dispatch token remains held through task completion. The task
  /// function and context are borrowed until this call returns. An inline lane
  /// invokes directly on its owner; a busy-poll lane spins on its preallocated
  /// task state; a sleeping lane blocks.
  [[nodiscard]] expected<void, Error> dispatch(std::uint32_t execution_lane,
                                               ExecutionLaneTask task);

  /// @brief Launch one persistent Runtime owner task on every non-inline local lane.
  ///
  /// The input must be sorted by global execution-lane identity and form an
  /// exact cover of the materialized non-inline lanes. Each task and context is
  /// borrowed until \ref wait_owner_tasks or \ref join returns. This operation
  /// is accepted after \ref start when no prior owner round remains active and
  /// temporarily closes legacy dispatch admission for the lane set. A
  /// successful \ref wait_owner_tasks reopens legacy dispatch and permits a
  /// later owner round. Owner tasks execute directly on their stable lane
  /// workers without the global legacy dispatch token.
  /// @return Success after every task has been published to its lane.
  /// @return A typed validation, state, or lane-ownership error otherwise.
  [[nodiscard]] expected<void, Error> launch_owner_tasks(
      std::span<const ExecutionLaneOwnerTask> tasks);

  /// @brief Wait for every launched Runtime owner task to retire.
  ///
  /// This call does not request stop. Running owner tasks must terminate
  /// cooperatively through their borrowed Runtime context. The first task
  /// failure in execution-lane order is returned after all owner tasks retire.
  /// The completed round is then reset and ordinary dispatch is admitted again.
  /// @return Success when every task completed successfully.
  /// @return A typed task failure, invalid-state, or self-wait error otherwise.
  [[nodiscard]] expected<void, Error> wait_owner_tasks() noexcept;

  /// @brief Cooperatively reject new work and stop every idle lane.
  void request_stop() noexcept;

  /// @brief Stop and join every lane exactly once.
  [[nodiscard]] expected<void, Error> join() noexcept;

  /// @brief Return whether the all-lanes startup barrier is open.
  [[nodiscard]] bool ready() const noexcept;

  /// @brief Return the immutable startup evidence after start completes.
  [[nodiscard]] StartupEvidenceView startup_evidence() const noexcept;

  /// @brief Transfer startup evidence into its owning RunSession archive.
  ///
  /// Call only after `start()` completes. The returned vector retains one
  /// final record per materialized local lane in global lane order;
  /// `startup_evidence()` is empty after transfer. This cold-path operation
  /// performs no element copy.
  [[nodiscard]] std::vector<ExecutionLaneStartupEvidence> take_startup_evidence() noexcept;

 private:
  struct Impl;
  explicit ExecutionLaneSet(std::unique_ptr<Impl> impl) noexcept;

  std::unique_ptr<Impl> impl_;
};

/// @brief Create the guarded system placement adapter.
///
/// Linux uses LinuxThreadController. Other platforms admit the default no-op
/// placement and return typed unavailable evidence for OS-specific requests.
[[nodiscard]] std::shared_ptr<ExecutionLanePlacementAdapter>
make_system_execution_lane_placement_adapter();

}  // namespace holoscan::detail

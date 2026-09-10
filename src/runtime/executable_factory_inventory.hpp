/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <holoscan/core/graph.hpp>

namespace holoscan::detail {

class RuntimePortBindingFactory;
class RuntimeDirectContinuationTestControl;
class RuntimeManagedSignalTestControl;
class RuntimeClosedIslandTestControl;
class RuntimeChannelTeardownTestControl;
class RuntimeChannelIncompletePollTestControl;
class RuntimeChannelSessionBindTestControl;
class RuntimeCudaExecutionTestControl;
class RuntimeOperatorCompletionFinalEntryTestControl;
class RuntimeLateReadinessDrainTestControl;
class RuntimePreDispatchTestControl;
class RuntimePhaseCounters;
class RuntimeOperatorCompletionProgressSeedTestControl;
class RuntimeStartupProgressFailureTestControl;
class RuntimeSupportThreadControl;
class RuntimeTerminalSummarySink;
class RuntimeTensorOutputTestControl;
class RuntimeTerminationAdmissionTestControl;
class RuntimeWaitTestControl;
class RuntimeWindowTimeoutTestControl;
class ExecutionLanePlacementAdapter;

/// @brief Identify how one frozen lifecycle callback must be reconstructed.
enum class ExecutableLifecycleCallbackOrigin : std::uint8_t {
  kAdvancedMember,
  kLegacyStart,
  kLegacyStop,
};

/// @brief Retain one pointer-free lifecycle replay expectation.
struct ExecutableLifecycleCallbackRecord {
  LifecycleStage stage{LifecycleStage::kConfigure};
  bool required{};
  LifecycleMethodId method{};
  ExecutableLifecycleCallbackOrigin origin{ExecutableLifecycleCallbackOrigin::kAdvancedMember};
};

/// @brief Retain one executable Operator factory outside serialized plan state.
struct ExecutableOperatorFactoryRecord {
  VertexId vertex{};                             ///< Canonical plan vertex owning this factory.
  OperatorInstanceId operator_id{};              ///< Dense canonical Operator identity.
  std::shared_ptr<OperatorFactoryBase> factory;  ///< Cold clone used by compile and RunSession.
  /// Exact callback declarations captured with the canonical factory.
  std::vector<ExecutableLifecycleCallbackRecord> lifecycle_callbacks;
  /// Source root-Graph Operator index, absent for every embedded Operator.
  std::optional<std::size_t> root_operator_index;
};

/// @brief Retain one executable clock source outside serialized plan state.
struct ExecutableClockFactoryRecord {
  GraphId graph{};  ///< Canonical Graph scope of the declaration.
  ClockKey key;     ///< Exact owned declaration key within the Graph scope.
  std::shared_ptr<ClockFactoryBase> factory;  ///< Factory for ordinary session-local clocks.
  std::shared_ptr<Clock> shared;              ///< Explicitly shared clock when authored as such.
};

/// @brief Own every process-local factory needed to reconstruct one immutable plan.
///
/// This inventory is never serialized, hashed into portable plan bytes, exposed
/// through GraphView, or interpreted as semantic identity. The pointer-free
/// plan remains authoritative; fresh declarations must match it before any
/// user lifecycle callback is invoked.
struct ExecutableFactoryInventory {
  std::vector<ExecutableOperatorFactoryRecord> operators;
  std::vector<ExecutableClockFactoryRecord> clocks;
  /// Optional process-local readiness factory configured by the embedding Runtime.
  ///
  /// The baseline direct conformance implementation supplies this only through
  /// non-installed test support. Later physical adapters reuse the same
  /// interface without changing the immutable plan schema.
  std::shared_ptr<RuntimePortBindingFactory> port_binding_factory;
  /// Optional process-local lane-placement adapter supplied by the embedding Runtime.
  ///
  /// Production plans normally leave this empty so RunSession constructs the
  /// system adapter. Source-tree conformance tests may inject an adapter through
  /// the non-installed RuntimeTestPlanAccess seam without changing plan identity.
  std::shared_ptr<ExecutionLanePlacementAdapter> lane_placement_adapter;
  /// Optional source-tree measurement control for Runtime-owned support threads.
  ///
  /// This private process-local object is excluded from plan identity and serialized artifacts.
  std::shared_ptr<RuntimeSupportThreadControl> support_thread_control;
  /// Optional source-tree execution-phase counters excluded from production plan identity.
  std::shared_ptr<RuntimePhaseCounters> phase_counters;
  /// Optional source-tree terminal-summary sink excluded from production plan identity.
  std::shared_ptr<RuntimeTerminalSummarySink> terminal_summary_sink;
  std::shared_ptr<RuntimeWaitTestControl> test_wait_control;  ///< Source-tree wait interleaving.
  /// Source-tree-only deterministic managed-signal transaction fault control.
  std::shared_ptr<RuntimeManagedSignalTestControl> test_managed_signal_control;
  /// Source-tree-only one-shot barrier after candidate freeze and before Dispatcher selection.
  std::shared_ptr<RuntimePreDispatchTestControl> test_pre_dispatch_control;
  /// Source-tree-only one-shot barrier before the late readiness fallback drain.
  std::shared_ptr<RuntimeLateReadinessDrainTestControl> test_late_readiness_drain_control;
  /// Source-tree-only direct Tensor fault and claim observer; absent in production plans.
  std::shared_ptr<RuntimeTensorOutputTestControl> test_tensor_output_control;
  /// Source-tree-only CUDA execution materialization fault control; absent in production plans.
  std::shared_ptr<RuntimeCudaExecutionTestControl> test_cuda_execution_control;
  /// Source-tree-only Channel teardown observer; absent in production plans.
  std::shared_ptr<RuntimeChannelTeardownTestControl> test_channel_teardown_control;
  /// Source-tree-only one-shot incomplete blocking Channel poll; absent in production plans.
  std::shared_ptr<RuntimeChannelIncompletePollTestControl> test_channel_incomplete_poll_control;
  /// Source-tree-only Channel-session post-create hook; absent in production plans.
  std::shared_ptr<RuntimeChannelSessionBindTestControl> test_channel_session_bind_control;
  /// Source-tree-only barrier after a continuation receipt is staged and before its next scan.
  std::shared_ptr<RuntimeDirectContinuationTestControl> test_direct_continuation_control;
  /// Source-tree-only observer and interleaving control for closed-island transactions.
  std::shared_ptr<RuntimeClosedIslandTestControl> test_closed_island_control;
  /// Source-tree-only observer for partition-owned window-timeout evaluation.
  std::shared_ptr<RuntimeWindowTimeoutTestControl> test_window_timeout_control;
  /// Source-tree-only callback-admission linearization barrier.
  std::shared_ptr<RuntimeTerminationAdmissionTestControl> test_termination_admission_control;
  /// Source-tree-only stale-work observer for the Operator completion final-entry fence.
  std::shared_ptr<RuntimeOperatorCompletionFinalEntryTestControl>
      test_operator_completion_final_entry_control;
  /// Source-tree-only one-shot initial Operator compute-call progress.
  std::shared_ptr<RuntimeOperatorCompletionProgressSeedTestControl>
      test_operator_completion_progress_seed;
  /// Source-tree-only startup fault injected after completion progress materializes.
  std::shared_ptr<RuntimeStartupProgressFailureTestControl> test_startup_progress_failure_control;
  /// Source-tree-only one-shot Channel activation-retirement fault.
  bool test_fail_next_channel_activation_retirement{};
  /// Source-tree-only finite witness count for every direct Spin-backed Operator.
  ///
  /// Zero preserves the production fail-closed path. A positive value is set only through the
  /// non-installed Runtime test accessor and never enters plan identity or serialized artifacts.
  std::uint32_t test_direct_activation_count{};
  /// Source-tree-only control that retains direct delivery while disabling continuation elision.
  ///
  /// This provides an exact control arm for Runtime tests and benchmarks without changing the
  /// compiled delivery kind or routing the sample through HoloMQ.
  bool test_disable_direct_continuation_elision{};
  /// Source-tree-only ordinary-path control for compiler-admitted closed islands.
  bool test_disable_closed_island_composite{};
  /// Source-tree-only control that retains the background pump for an all-eventless inventory.
  ///
  /// This provides an exact benchmark and conformance control without changing event-wake
  /// attachment, transport selection, or immutable plan identity.
  bool test_force_process_local_channel_pump{};
  /// Source-tree-only initial Tag sequence for reconstructed Operators.
  ///
  /// One is the production value. Tests may select a later nonzero value through the non-installed
  /// Runtime accessor to exercise the terminal no-wrap boundary without changing plan identity.
  std::uint64_t test_initial_tag_sequence{1U};
};

}  // namespace holoscan::detail

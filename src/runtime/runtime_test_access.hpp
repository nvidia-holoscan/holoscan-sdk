/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/execution_plan.hpp>

#include "../compiler/execution_plan_state.hpp"
#include "../compiler/runtime_package_capabilities.hpp"
#include "../core/graph_compiler_access.hpp"
#include "managed_shutdown_signals_test_control.hpp"
#include "port_binding.hpp"
#include "runtime_channel_binding.hpp"
#include "runtime_cuda_execution_resources.hpp"
#include "runtime_diagnostic_sink.hpp"
#include "runtime_direct_composite.hpp"
#include "runtime_fanout_test_control.hpp"
#include "runtime_phase_counters.hpp"
#include "runtime_support_thread_control.hpp"
#include "runtime_tensor_resources.hpp"
#include "runtime_test_controls.hpp"

namespace holoscan::detail {

/// @brief Construct source-tree runtime plans with explicit Runtime test controls.
///
/// These non-installed seams preserve production plan and readiness invariants while allowing
/// focused tests to attach bounded witnesses, interleaving controls, and observers.
class RuntimeTestPlanAccess {
 public:
  enum class ClosedIslandMutation {
    kExecutionLane,
    kReadinessRoot,
    kReadinessSource,
    kReadinessClock,
    kEdge,
  };

  /// @brief Identify one deliberately malformed lifecycle record used by startup tests.
  enum class LifecycleMutation {
    kCheckpointMissing,
    kCheckpointCallback,
    kCheckpointRequired,
    kCheckpointLevel,
    kCheckpointOrder,
    kDependency,
    kRestartPermission,
    kMutableField,
    kTransactionCapacity,
    kRestartTargetMissing,
    kRestartTargetIdentity,
    kRestartTargetAuthority,
    kRestartClosureMissing,
    kRestartStructuralEventCount,
    kCapacityDerivationReason,
    kRequiredCapacityUnderflow,
    kRequiredCapacityOverflow,
  };

  /// @brief Identify one corrupted clock fact in an otherwise inert empty-Graph catalog.
  enum class InertClockMutation {
    kPlanClockKey,
    kMissingFactory,
    kFactoryMode,
    kRootDefault,
    kConversionGeneration,
    kConversionEndpoint,
  };

  /// @brief Identify one corrupted reusable-Graph clock import inventory fact.
  enum class ClockImportMutation {
    kMissingImport,
    kMissingBinding,
    kMissingImportAndBinding,
    kChangedParentClock,
    kNonImmediateParent,
    kFrozenResolvedClock,
    kDuplicateImportedDefault,
  };

  /// @brief Identify one package-compatibility fact corrupted for startup-order tests.
  enum class ProfileMutation {
    kPlanSchema,
    kProfilePolicy,
    kRuntimeAbi,
    kTarget,
    kCapability,
    kEvidence,
  };

  /// @brief Identify one malformed Parameter owner grouping used by startup tests.
  enum class ParameterInventoryMutation {
    kInterleavedFields,
    kInterleavedDerivedTerms,
    kZeroFieldOwner,
    kZeroDerivedTermOwner,
  };

  /// @brief Identify one corrupted direct-Tensor pool or publication contract.
  enum class TensorRuntimeMutation {
    kPublicationRetry,
    kPublicationPreSubmitError,
    kPublicationPostSubmitError,
    kPublicationPostSubmitPartialError,
    kPublicationAuthority,
    kContributorCapacity,
    kContributorReasonOrder,
    kContributorPool,
    kContributorRecordOrder,
    kMissingTransportRetention,
    kAllocationCapacity,
    kAllocationAggregate,
  };

  /// @brief Copy one accepted plan and enable the bounded direct-once test backend.
  [[nodiscard]] static ExecutionPlan enable_direct_once(const ExecutionPlan& plan) {
    return enable_direct_count(plan, 1U);
  }

  /// @brief Copy one accepted plan and enable a finite direct-witness test backend.
  /// @param plan Accepted plan whose non-portable executable inventory is copied.
  /// @param activation_count Positive callback count injected for each direct Spin source.
  /// @return Independent plan facade carrying only the source-tree test witness count.
  /// @throws std::invalid_argument when \p activation_count is zero.
  [[nodiscard]] static ExecutionPlan enable_direct_count(const ExecutionPlan& plan,
                                                         std::uint32_t activation_count) {
    if (activation_count == 0U) {
      throw std::invalid_argument{"direct runtime activation count must be positive"};
    }
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    state->executable_factories.test_direct_activation_count = activation_count;
    for (auto& readiness : state->readiness) {
      readiness.referenced = true;
      readiness.materialized = true;
    }
    return ExecutionPlanAccess::make(std::move(state));
  }

  /// @brief Copy one accepted plan and replace its Operator completion rules with one test rule.
  /// @param plan Accepted plan whose immutable records are copied.
  /// @param owner Plan-bound logical Operator that owns the completion rule.
  /// @param limit Complete unsigned successful-compute-call limit.
  /// @return Independent plan facade carrying one separate Operator completion record.
  [[nodiscard]] static ExecutionPlan with_operator_completion_rule(const ExecutionPlan& plan,
                                                                   OperatorRef owner,
                                                                   std::uint64_t limit) {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source || !owner.plan_bound() || owner.plan_generation != source->plan_generation) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    const auto trigger =
        std::find_if(state->temporal_triggers.begin(),
                     state->temporal_triggers.end(),
                     [owner](const auto& record) { return record.owner == owner; });
    if (trigger == state->temporal_triggers.end()) {
      return {};
    }
    state->operator_completion_rules = {OperatorCompletionPlanRecord{
        .owner = owner,
        .kind = OperatorCompletionRuleKind::kComputeCalls,
        .limit = limit,
    }};
    return ExecutionPlanAccess::make(std::move(state));
  }

  /// @brief Copy one accepted plan and remove every completion rule for replay-negative tests.
  [[nodiscard]] static ExecutionPlan without_operator_completion_rules(const ExecutionPlan& plan) {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    state->operator_completion_rules.clear();
    return ExecutionPlanAccess::make(std::move(state));
  }

  /// @brief Copy one plan and route bounded stale work through its final completion-entry fence.
  /// @param plan Accepted plan whose private executable inventory is copied.
  /// @param control Counter retained by the copied plan and the observing source-tree test.
  /// @return Independent plan facade, or an empty plan when either input is unavailable.
  [[nodiscard]] static ExecutionPlan with_operator_completion_final_entry_test_control(
      const ExecutionPlan& plan,
      std::shared_ptr<RuntimeOperatorCompletionFinalEntryTestControl> control) noexcept {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source || !control) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    state->executable_factories.test_operator_completion_final_entry_control = std::move(control);
    return ExecutionPlanAccess::make(std::move(state));
  }

  /// @brief Copy a plan with one initial successful-compute-call value for its next session.
  /// @param plan Accepted plan whose process-local test inventory is copied.
  /// @param successful_compute_calls Value consumed by the next session during local completion
  /// progress initialization.
  /// @return Independent plan facade carrying one process-local one-shot seed, or an empty plan
  /// when the source plan is invalid or the value cannot be a valid pre-limit seed.
  /// @note This non-installed seam exists only to exercise the exact unsigned accounting boundary.
  [[nodiscard]] static ExecutionPlan with_operator_compute_call_progress_seed_for_test(
      const ExecutionPlan& plan, const std::uint64_t successful_compute_calls) noexcept {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source || successful_compute_calls == std::numeric_limits<std::uint64_t>::max()) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    state->executable_factories.test_operator_completion_progress_seed =
        std::make_shared<RuntimeOperatorCompletionProgressSeedTestControl>(
            successful_compute_calls);
    return ExecutionPlanAccess::make(std::move(state));
  }

  /// @brief Copy one plan and fail the next session after local completion progress materializes.
  /// @param plan Accepted plan whose private executable inventory is copied.
  /// @return Independent plan facade carrying one source-tree-only startup failure.
  /// @note The fault is consumed before later startup initialization and never enters plan
  /// identity.
  [[nodiscard]] static ExecutionPlan
  fail_after_operator_completion_progress_materialization_for_test(
      const ExecutionPlan& plan) noexcept {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    state->executable_factories.test_startup_progress_failure_control =
        std::make_shared<RuntimeStartupProgressFailureTestControl>();
    return ExecutionPlanAccess::make(std::move(state));
  }

  /// @brief Copy one plan and fail its next post-callback Channel activation retirement.
  /// @param plan Accepted plan whose private executable inventory is copied.
  /// @param failed_callback_only Fail only the next retirement of an activation whose callback
  /// failed, whichever owner retires other activations first.
  /// @return Independent plan facade carrying one source-tree-only one-shot retirement fault.
  /// @note The fault is installed before Channel arm and does not enter plan identity or artifacts.
  [[nodiscard]] static ExecutionPlan fail_next_channel_activation_retirement_for_test(
      const ExecutionPlan& plan, const bool failed_callback_only = false) noexcept {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    state->executable_factories.test_fail_next_channel_activation_retirement = true;
    state->executable_factories.test_fail_only_failed_callback_retirement = failed_callback_only;
    return ExecutionPlanAccess::make(std::move(state));
  }

  /// @brief Copy one accepted plan and attach one non-installed Runtime binding factory.
  [[nodiscard]] static ExecutionPlan with_port_binding_factory(
      const ExecutionPlan& plan, std::shared_ptr<RuntimePortBindingFactory> factory) {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source || !factory) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    state->executable_factories.port_binding_factory = std::move(factory);
    return ExecutionPlanAccess::make(std::move(state));
  }

  /// @brief Copy one accepted plan and attach a source-tree-only Tensor output observer.
  [[nodiscard]] static ExecutionPlan with_tensor_output_test_control(
      const ExecutionPlan& plan, std::shared_ptr<RuntimeTensorOutputTestControl> control) {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source || !control) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    state->executable_factories.test_tensor_output_control = std::move(control);
    return ExecutionPlanAccess::make(std::move(state));
  }

  /// @brief Copy one accepted plan and attach a source-tree-only CUDA materialization fault.
  [[nodiscard]] static ExecutionPlan with_cuda_execution_test_control(
      const ExecutionPlan& plan, std::shared_ptr<RuntimeCudaExecutionTestControl> control) {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source || !control) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    state->executable_factories.test_cuda_execution_control = std::move(control);
    return ExecutionPlanAccess::make(std::move(state));
  }

  /// @brief Copy one accepted plan and observe RunSession-owned Channel teardown.
  [[nodiscard]] static ExecutionPlan with_channel_teardown_test_control(
      const ExecutionPlan& plan, std::shared_ptr<RuntimeChannelTeardownTestControl> control) {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source || !control) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    state->executable_factories.test_channel_teardown_control = std::move(control);
    return ExecutionPlanAccess::make(std::move(state));
  }

  /// @brief Copy one accepted plan and force bounded blocking Channel polls to report incomplete.
  [[nodiscard]] static ExecutionPlan with_channel_incomplete_poll_test_control(
      const ExecutionPlan& plan, std::shared_ptr<RuntimeChannelIncompletePollTestControl> control) {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source || !control) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    state->executable_factories.test_channel_incomplete_poll_control = std::move(control);
    return ExecutionPlanAccess::make(std::move(state));
  }

  /// @brief Copy one accepted plan and install a source-tree Channel-session post-create hook.
  [[nodiscard]] static ExecutionPlan with_channel_session_bind_test_control(
      const ExecutionPlan& plan, std::shared_ptr<RuntimeChannelSessionBindTestControl> control) {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source || !control) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    state->executable_factories.test_channel_session_bind_control = std::move(control);
    return ExecutionPlanAccess::make(std::move(state));
  }

  /// @brief Copy one plan and attach one non-installed lane-placement adapter.
  [[nodiscard]] static ExecutionPlan with_lane_placement_adapter(
      const ExecutionPlan& plan, std::shared_ptr<ExecutionLanePlacementAdapter> adapter) {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source || !adapter) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    state->executable_factories.lane_placement_adapter = std::move(adapter);
    return ExecutionPlanAccess::make(std::move(state));
  }

  /// @brief Copy one plan and attach a non-installed Runtime support-thread control.
  /// @param plan Source plan whose immutable state is copied.
  /// @param control Shared control retained by the copied plan.
  /// @return The copied plan, or an empty plan when \p plan or \p control is unavailable.
  [[nodiscard]] static ExecutionPlan with_support_thread_control(
      const ExecutionPlan& plan, std::shared_ptr<RuntimeSupportThreadControl> control) {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source || !control) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    state->executable_factories.support_thread_control = std::move(control);
    return ExecutionPlanAccess::make(std::move(state));
  }

  /// @brief Copy one accepted plan and replace one support-thread affinity for failure testing.
  [[nodiscard]] static ExecutionPlan with_support_thread_affinity_core_for_test(
      const ExecutionPlan& plan, SupportThreadRole role, std::uint32_t core) {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    const auto found = std::find_if(state->support_thread_affinities.begin(),
                                    state->support_thread_affinities.end(),
                                    [role](const auto& record) { return record.role == role; });
    if (found == state->support_thread_affinities.end()) {
      return {};
    }
    auto& accepted = found->accepted;
    if (!accepted.has_value()) {
      return {};
    }
    found->requested.affinity = CPUAffinity{core};
    accepted->affinity = CPUAffinity{core};
    return ExecutionPlanAccess::make(std::move(state));
  }

  /// @brief Replace one accepted affinity after its first role thread entered successfully.
  [[nodiscard]] static ExecutionPlan with_support_thread_affinity_core_after_first_entry_for_test(
      const ExecutionPlan& plan, SupportThreadRole role, std::uint32_t replacement_core) {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    const auto found = std::ranges::find(
        state->support_thread_affinities, role, &SupportThreadAffinityPlanRecord::role);
    if (found == state->support_thread_affinities.end() || !found->accepted) {
      return {};
    }
    state->executable_factories.support_thread_control =
        std::make_shared<RuntimeSupportThreadAffinityMutationControl>(
            std::weak_ptr<ExecutionPlanState>{state}, role, replacement_core);
    return ExecutionPlanAccess::make(std::move(state));
  }

  /// @brief Copy one plan and retain the background pump for an all-eventless inventory.
  /// @param plan Source plan whose immutable state is copied.
  /// @return The copied plan, or an empty plan when \p plan is unavailable.
  /// @note This source-tree-only control preserves eventless HoloMQ delivery and changes only the
  /// cold Runtime support-thread selection.
  [[nodiscard]] static ExecutionPlan force_process_local_channel_pump_for_test(
      const ExecutionPlan& plan) {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    state->executable_factories.test_force_process_local_channel_pump = true;
    return ExecutionPlanAccess::make(std::move(state));
  }

  /// @brief Copy one plan and attach source-tree-only Runtime execution-phase counters.
  /// @param plan Source plan whose immutable state is copied.
  /// @param counters Shared counters retained by the copied plan.
  /// @return The copied plan, or an empty plan when \p plan or \p counters is unavailable.
  [[nodiscard]] static ExecutionPlan with_phase_counters(
      const ExecutionPlan& plan, std::shared_ptr<RuntimePhaseCounters> counters) {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source || !counters) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    state->executable_factories.phase_counters = std::move(counters);
    return ExecutionPlanAccess::make(std::move(state));
  }

  /// @brief Copy one plan and disable direct-continuation elision without changing delivery.
  /// @param plan Source plan whose immutable state is copied.
  /// @return The copied plan, or an empty plan when \p plan is unavailable.
  /// @note This source-tree-only control preserves direct retention so tests can compare the
  /// optimized path with its exact ordinary-dispatch control.
  [[nodiscard]] static ExecutionPlan disable_direct_continuation_elision_for_test(
      const ExecutionPlan& plan) {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    state->executable_factories.test_disable_direct_continuation_elision = true;
    return ExecutionPlanAccess::make(std::move(state));
  }

  /// @brief Copy one plan and keep partitions with boundary edges off the causal chain.
  /// @param plan Source plan whose immutable state is copied.
  /// @return The copied plan, or an empty plan when \p plan is unavailable.
  /// @note This source-tree-only control compares boundary chains with the ordinary scan.
  [[nodiscard]] static ExecutionPlan disable_boundary_causal_chain_for_test(
      const ExecutionPlan& plan) {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    state->executable_factories.test_disable_boundary_causal_chain = true;
    return ExecutionPlanAccess::make(std::move(state));
  }

  /// @brief Copy one plan and replace the cap on an owner's long park.
  /// @param plan Source plan whose immutable state is copied.
  /// @param cap Positive cap; tests lower it to keep fallback scans, or raise it so a missed wake
  /// fails visibly.
  /// @return The copied plan, or an empty plan when \p plan is unavailable or \p cap is not
  /// positive.
  [[nodiscard]] static ExecutionPlan with_owner_park_cap(const ExecutionPlan& plan,
                                                         std::chrono::nanoseconds cap) {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source || cap.count() <= 0) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    state->executable_factories.test_owner_park_cap_ns = cap.count();
    return ExecutionPlanAccess::make(std::move(state));
  }

  /// @brief Copy one plan and keep owner data hints on side-only Channel inputs.
  /// @param plan Source plan whose immutable state is copied.
  /// @return The copied plan, or an empty plan when \p plan is unavailable.
  /// @note This source-tree-only control is the exact comparison arm for the side-only hint
  /// elision; selection and delivery are unchanged.
  [[nodiscard]] static ExecutionPlan keep_side_only_data_hints_for_test(const ExecutionPlan& plan) {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    state->executable_factories.test_keep_side_only_data_hints = true;
    return ExecutionPlanAccess::make(std::move(state));
  }

  /// @brief Copy one plan and keep chain hops into members with side inputs on the ordinary scan.
  /// @param plan Source plan whose immutable state is copied.
  /// @return The copied plan, or an empty plan when \p plan is unavailable.
  /// @note This source-tree-only control isolates continuation into side-input members and through
  /// direct fanouts.
  [[nodiscard]] static ExecutionPlan disable_side_input_continuation_for_test(
      const ExecutionPlan& plan) {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    state->executable_factories.test_disable_side_input_continuation = true;
    return ExecutionPlanAccess::make(std::move(state));
  }

  /// @brief Copy one plan and pause after a direct-continuation receipt is staged.
  /// @param plan Source plan whose immutable state is copied.
  /// @param control Shared source-tree barrier retained by the copied plan.
  /// @return The copied plan, or an empty plan when \p plan or \p control is unavailable.
  [[nodiscard]] static ExecutionPlan with_direct_continuation_test_control(
      const ExecutionPlan& plan, std::shared_ptr<RuntimeDirectContinuationTestControl> control) {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source || !control) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    state->executable_factories.test_direct_continuation_control = std::move(control);
    return ExecutionPlanAccess::make(std::move(state));
  }

  /// @brief Copy one plan and attach a source-tree managed-signal transaction fault.
  [[nodiscard]] static ExecutionPlan with_managed_signal_test_control(
      const ExecutionPlan& plan, std::shared_ptr<RuntimeManagedSignalTestControl> control) {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source || !control) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    state->executable_factories.test_managed_signal_control = std::move(control);
    return ExecutionPlanAccess::make(std::move(state));
  }

  /// @brief Copy one plan and attach a source-tree callback-admission barrier.
  [[nodiscard]] static ExecutionPlan with_termination_admission_test_control(
      const ExecutionPlan& plan, std::shared_ptr<RuntimeTerminationAdmissionTestControl> control) {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source || !control) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    state->executable_factories.test_termination_admission_control = std::move(control);
    return ExecutionPlanAccess::make(std::move(state));
  }

  /// @brief Copy one plan while retaining direct delivery and disabling closed-island execution.
  ///
  /// The ordinary control retains Checkpoint-1 continuation elision, so a differential test changes
  /// only whether the source and successor share one lane task.
  [[nodiscard]] static ExecutionPlan disable_closed_island_composite_for_test(
      const ExecutionPlan& plan) {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    state->executable_factories.test_disable_closed_island_composite = true;
    return ExecutionPlanAccess::make(std::move(state));
  }

  /// @brief Attach private per-recipient attribution without changing plan identity.
  [[nodiscard]] static ExecutionPlan with_fanout_test_control(
      const ExecutionPlan& plan, std::shared_ptr<RuntimeFanoutTestControl> control) {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source || !control) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    state->executable_factories.test_fanout_control = std::move(control);
    return ExecutionPlanAccess::make(std::move(state));
  }

  /// @brief Copy one plan and attach a non-installed closed-island transaction observer.
  [[nodiscard]] static ExecutionPlan with_closed_island_test_control(
      const ExecutionPlan& plan, std::shared_ptr<RuntimeClosedIslandTestControl> control) {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source || !control) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    state->executable_factories.test_closed_island_control = std::move(control);
    return ExecutionPlanAccess::make(std::move(state));
  }

  /// @brief Copy one plan and attach a source-tree-only window-timeout owner observer.
  [[nodiscard]] static ExecutionPlan with_window_timeout_test_control(
      const ExecutionPlan& plan, std::shared_ptr<RuntimeWindowTimeoutTestControl> control) {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source || !control) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    state->executable_factories.test_window_timeout_control = std::move(control);
    return ExecutionPlanAccess::make(std::move(state));
  }

  /// @brief Copy one plan and corrupt its private closed-island replay authority.
  ///
  /// This source-tree-only seam proves that Runtime reconstructs the compiler's exact record before
  /// any Operator callback or backend activation can begin.
  [[nodiscard]] static ExecutionPlan corrupt_closed_island_activation_for_test(
      const ExecutionPlan& plan,
      ClosedIslandMutation mutation = ClosedIslandMutation::kExecutionLane) {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source || source->closed_island_activations.empty()) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    auto& record = state->closed_island_activations.front();
    switch (mutation) {
      case ClosedIslandMutation::kExecutionLane:
        record.execution_lane = record.execution_lane == std::numeric_limits<std::uint32_t>::max()
                                    ? 0U
                                    : record.execution_lane + 1U;
        break;
      case ClosedIslandMutation::kReadinessRoot:
        record.readiness_root = record.readiness_root == ClosedIslandReadinessRootKind::kClock
                                    ? ClosedIslandReadinessRootKind::kPhysical
                                    : ClosedIslandReadinessRootKind::kClock;
        break;
      case ClosedIslandMutation::kReadinessSource:
        record.readiness_source = {};
        break;
      case ClosedIslandMutation::kReadinessClock:
        record.readiness_clock = {};
        break;
      case ClosedIslandMutation::kEdge:
        record.edge = {};
        break;
    }
    return ExecutionPlanAccess::make(std::move(state));
  }

  /// @brief Copy one plan and seed reconstructed Operator Tag allocators at a nonzero sequence.
  /// @param plan Source plan whose immutable state is copied.
  /// @param next_sequence First sequence reserved by every reconstructed Operator.
  /// @return The copied plan, or an empty plan when \p plan is unavailable or \p next_sequence is
  /// zero.
  /// @note This non-installed seam exists only to exercise the terminal Tag no-wrap boundary.
  [[nodiscard]] static ExecutionPlan with_initial_tag_sequence_for_test(
      const ExecutionPlan& plan, std::uint64_t next_sequence) {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source || next_sequence == 0U) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    state->executable_factories.test_initial_tag_sequence = next_sequence;
    return ExecutionPlanAccess::make(std::move(state));
  }

  /// @brief Copy one plan and retain a private terminal structural-diagnostic sink.
  [[nodiscard]] static ExecutionPlan with_diagnostic_sink(
      const ExecutionPlan& plan, std::shared_ptr<RuntimeStructuralDiagnosticSink> sink) {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    state->runtime_diagnostic_sink = std::move(sink);
    return ExecutionPlanAccess::make(std::move(state));
  }

  /// @brief Copy one plan and retain a private terminal-summary sink.
  /// @param plan Source plan whose immutable state is copied.
  /// @param sink Terminal-summary sink retained by the copied plan.
  /// @return The copied plan, or an empty plan when \p plan is unavailable.
  [[nodiscard]] static ExecutionPlan with_terminal_summary_sink(
      const ExecutionPlan& plan, std::shared_ptr<RuntimeTerminalSummarySink> sink) {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    state->executable_factories.terminal_summary_sink = std::move(sink);
    return ExecutionPlanAccess::make(std::move(state));
  }

  /// @brief Copy one plan and append a max_start_delay that has no owning temporal trigger.
  [[nodiscard]] static ExecutionPlan with_stray_max_start_delay(const ExecutionPlan& plan) {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    state->max_start_delays.emplace_back();
    return ExecutionPlanAccess::make(std::move(state));
  }

  /// @brief Copy one injected plan and install a deterministic wait interleaving barrier.
  [[nodiscard]] static ExecutionPlan enable_wait_interleaving(
      const ExecutionPlan& plan, std::shared_ptr<RuntimeWaitTestControl> control) {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source || !control) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    state->executable_factories.test_wait_control = std::move(control);
    return ExecutionPlanAccess::make(std::move(state));
  }

  /// @brief Copy one injected plan and hold a singleton partition before Dispatcher selection.
  [[nodiscard]] static ExecutionPlan enable_pre_dispatch_interleaving(
      const ExecutionPlan& plan, std::shared_ptr<RuntimePreDispatchTestControl> control) {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source || !control) {
      return {};
    }
    const OperatorInstanceId blocked = control->blocked_operator();
    const bool singleton_runner = std::any_of(
        source->partition_runners.begin(),
        source->partition_runners.end(),
        [blocked](const PartitionRunnerPlanRecord& runner) {
          return runner.operators.size() == 1U && runner.operators.front().operator_id == blocked;
        });
    if (!singleton_runner) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    state->executable_factories.test_pre_dispatch_control = std::move(control);
    return ExecutionPlanAccess::make(std::move(state));
  }

  /// @brief Copy one injected plan and pause before one late readiness fallback drain.
  [[nodiscard]] static ExecutionPlan with_late_readiness_drain_test_control(
      const ExecutionPlan& plan, std::shared_ptr<RuntimeLateReadinessDrainTestControl> control) {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source || !control) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    state->executable_factories.test_late_readiness_drain_control = std::move(control);
    return ExecutionPlanAccess::make(std::move(state));
  }

  /// @brief Copy one accepted plan and force Parameter reconstruction to retire before attach.
  /// @param plan Accepted Parameter plan whose public admitted state remains unchanged.
  /// @return Source-tree-only plan with an invalid zero audit capacity.
  /// @note This seam exists only to exercise startup attach/retire/control-call races; production
  /// compilation never emits this inconsistent record.
  [[nodiscard]] static ExecutionPlan invalidate_parameter_runtime_for_test(
      const ExecutionPlan& plan) {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    auto& runtime_capacity = state->parameter_runtime_capacity;
    if (!runtime_capacity.has_value()) {
      return {};
    }
    runtime_capacity->audit_capacity = 0U;
    return ExecutionPlanAccess::make(std::move(state));
  }

  /// @brief Copy one accepted plan and corrupt one Parameter owner inventory.
  /// @param plan Accepted plan with records suitable for the selected mutation.
  /// @param mutation Noncontiguous grouping or invalid zero owner to introduce.
  /// @return Source-tree-only malformed plan, or an empty plan when the fixture is unsuitable.
  [[nodiscard]] static ExecutionPlan mutate_parameter_inventory(
      const ExecutionPlan& plan, ParameterInventoryMutation mutation) {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    switch (mutation) {
      case ParameterInventoryMutation::kInterleavedFields:
        if (state->parameter_fields.size() < 2U ||
            state->parameter_fields.front().owner == state->parameter_fields.back().owner) {
          return {};
        }
        state->parameter_fields.push_back(state->parameter_fields.front());
        break;
      case ParameterInventoryMutation::kInterleavedDerivedTerms:
        if (state->parameter_derived_terms.size() < 2U ||
            state->parameter_derived_terms.front().owner ==
                state->parameter_derived_terms.back().owner) {
          return {};
        }
        state->parameter_derived_terms.push_back(state->parameter_derived_terms.front());
        break;
      case ParameterInventoryMutation::kZeroFieldOwner:
        if (state->parameter_fields.empty()) {
          return {};
        }
        state->parameter_fields.front().owner.id = 0U;
        break;
      case ParameterInventoryMutation::kZeroDerivedTermOwner:
        if (state->parameter_derived_terms.empty()) {
          return {};
        }
        state->parameter_derived_terms.front().owner.id = 0U;
        break;
    }
    return ExecutionPlanAccess::make(std::move(state));
  }

  /// @brief Copy one accepted plan and remove all endpoint-catalog authority.
  /// @param plan Accepted plan whose Graph still contains authored external topics.
  /// @return Source-tree-only plan with a default-empty endpoint catalog.
  [[nodiscard]] static ExecutionPlan remove_endpoint_catalog(const ExecutionPlan& plan) {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    state->endpoint_catalog = {};
    return ExecutionPlanAccess::make(std::move(state));
  }

  /// @brief Copy one accepted plan and append one runtime-mutable selector term with no consumer.
  /// @param plan Accepted plan containing at least one runtime-mutable LatestWithin term.
  /// @param input Exact input that deliberately has no LatestWithin selector consumer.
  /// @return Source-tree-only malformed plan, or an empty facade when the preconditions fail.
  /// @note This seam proves startup rejects a cold-derived binding that cannot reach a production
  /// selector. Production compilation never emits this inconsistent record.
  [[nodiscard]] static ExecutionPlan append_orphan_latest_within_term_for_test(
      const ExecutionPlan& plan, PortId input) {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source || !input) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    const auto prototype = std::find_if(
        state->parameter_derived_terms.begin(),
        state->parameter_derived_terms.end(),
        [](const auto& term) {
          return term.kind == ParameterDerivedTermKind::kLatestWithinMaxAge &&
                 term.update_disposition == ParameterDerivedTermUpdateDisposition::kRuntimeAtomic;
        });
    if (prototype == state->parameter_derived_terms.end()) {
      return {};
    }
    ParameterDerivedTermPlanRecord orphan = *prototype;
    orphan.input = input;
    orphan.term_id = "operator:" + std::to_string(orphan.owner.id) +
                     "/input:" + std::to_string(input.value()) + "/latest_within.max_age.orphan";
    state->parameter_derived_terms.push_back(std::move(orphan));
    std::sort(state->parameter_derived_terms.begin(),
              state->parameter_derived_terms.end(),
              [](const auto& left, const auto& right) { return left.term_id < right.term_id; });
    return ExecutionPlanAccess::make(std::move(state));
  }

  /// @brief Copy one accepted direct-Tensor plan and corrupt one cold Runtime invariant.
  // Mutation cases intentionally enumerate the complete validation seam used by negative tests.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] static ExecutionPlan mutate_tensor_runtime(const ExecutionPlan& plan,
                                                           TensorRuntimeMutation mutation) {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source || source->tensor_allocations.empty() || source->publication_outcomes.empty() ||
        source->pool_capacity_contributors.empty()) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    auto& allocation = state->tensor_allocations.front();
    const auto publication =
        std::find_if(state->publication_outcomes.begin(),
                     state->publication_outcomes.end(),
                     [&](const auto& row) { return row.output == allocation.output; });
    const auto contributor =
        std::find_if(state->pool_capacity_contributors.begin(),
                     state->pool_capacity_contributors.end(),
                     [&](const auto& row) { return row.pool_id == allocation.pool_id; });
    if (publication == state->publication_outcomes.end() ||
        contributor == state->pool_capacity_contributors.end()) {
      return {};
    }
    switch (mutation) {
      case TensorRuntimeMutation::kPublicationRetry:
        publication->runtime_retry_enabled = true;
        break;
      case TensorRuntimeMutation::kPublicationPreSubmitError:
        publication->pre_submit_error = ErrorCode::kFailure;
        break;
      case TensorRuntimeMutation::kPublicationPostSubmitError:
        publication->post_submit_error = ErrorCode::kFailure;
        break;
      case TensorRuntimeMutation::kPublicationPostSubmitPartialError:
        publication->post_submit_partial_error = ErrorCode::kFailure;
        break;
      case TensorRuntimeMutation::kPublicationAuthority:
        publication->pre_submit_authority = PreSubmitAuthorityDisposition::kRetainedSameActivation;
        break;
      case TensorRuntimeMutation::kContributorCapacity:
        ++contributor->maximum_distinct_slots;
        break;
      case TensorRuntimeMutation::kContributorReasonOrder:
        if (contributor->reasons.size() < 2U) {
          return {};
        }
        std::swap(contributor->reasons[0], contributor->reasons[1]);
        break;
      case TensorRuntimeMutation::kContributorPool:
        contributor->pool_id += ".wrong";
        break;
      case TensorRuntimeMutation::kContributorRecordOrder: {
        const auto producer =
            std::find_if(state->pool_capacity_contributors.begin(),
                         state->pool_capacity_contributors.end(),
                         [&](const auto& row) {
                           return row.contributor_id == allocation.pool_id + ".producer";
                         });
        const auto retirement =
            std::find_if(state->pool_capacity_contributors.begin(),
                         state->pool_capacity_contributors.end(),
                         [&](const auto& row) {
                           return row.contributor_id == allocation.pool_id + ".retirement";
                         });
        if (producer == state->pool_capacity_contributors.end() ||
            retirement == state->pool_capacity_contributors.end()) {
          return {};
        }
        std::iter_swap(producer, retirement);
        break;
      }
      case TensorRuntimeMutation::kMissingTransportRetention: {
        const auto transport = std::find_if(
            state->pool_capacity_dependencies.begin(),
            state->pool_capacity_dependencies.end(),
            [&](const auto& dependency) {
              return dependency.pool_id == allocation.pool_id &&
                     dependency.origin == PoolClaimOrigin::kTransportRetention &&
                     dependency.source == PoolCapacityDependencySource::kQueueSizingRecipe;
            });
        const auto provider =
            std::find_if(state->provider_allocation_plans.begin(),
                         state->provider_allocation_plans.end(),
                         [&](const auto& record) { return record.pool_id == allocation.pool_id; });
        if (transport == state->pool_capacity_dependencies.end() ||
            provider == state->provider_allocation_plans.end() || !transport->queue_sizing_recipe ||
            provider->sizing_dependency_count <= 1U) {
          return {};
        }
        const auto recipe = std::ranges::find(state->queue_sizing_recipes,
                                              transport->queue_sizing_recipe,
                                              &QueueSizingRecipePlanRecord::id);
        if (recipe == state->queue_sizing_recipes.end()) {
          return {};
        }
        const auto edge =
            std::ranges::find(state->payload_edges, recipe->edge, &PayloadEdgePlanRecord::edge);
        if (edge == state->payload_edges.end() ||
            edge->queue_depth >
                std::numeric_limits<std::uint32_t>::max() / transport->mapping_multiplicity) {
          return {};
        }
        const std::uint32_t removed_slots = edge->queue_depth * transport->mapping_multiplicity;
        if (removed_slots == 0U || removed_slots >= allocation.slot_count ||
            removed_slots > allocation.control_capacity ||
            removed_slots > allocation.readiness_capacity ||
            removed_slots > allocation.release_capacity ||
            removed_slots > allocation.retirement_capacity ||
            removed_slots > allocation.quarantine_capacity) {
          return {};
        }
        const std::uint32_t removed_dependency_id = transport->id.value;
        const std::uint64_t removed_bytes =
            static_cast<std::uint64_t>(removed_slots) * allocation.bounds.max_byte_span;
        if (removed_bytes > allocation.aggregate_byte_capacity) {
          return {};
        }
        state->pool_capacity_dependencies.erase(transport);
        bool provider_updated{};
        for (auto& record : state->provider_allocation_plans) {
          const std::uint32_t first = record.first_sizing_dependency.value;
          const std::uint32_t last = first + record.sizing_dependency_count;
          if (removed_dependency_id < first) {
            --record.first_sizing_dependency.value;
          } else if (removed_dependency_id < last) {
            --record.sizing_dependency_count;
            provider_updated = provider_updated || record.pool_id == allocation.pool_id;
          }
        }
        if (!provider_updated) {
          return {};
        }
        for (std::size_t index = 0U; index < state->pool_capacity_dependencies.size(); ++index) {
          state->pool_capacity_dependencies[index].id =
              PoolCapacityDependencyId{static_cast<std::uint32_t>(index + 1U)};
        }
        allocation.slot_count -= removed_slots;
        allocation.control_capacity -= removed_slots;
        allocation.readiness_capacity -= removed_slots;
        allocation.release_capacity -= removed_slots;
        allocation.retirement_capacity -= removed_slots;
        allocation.quarantine_capacity -= removed_slots;
        allocation.aggregate_byte_capacity -= removed_bytes;
        for (auto& resource : state->tensor_device_resources) {
          if (resource.process_group != allocation.allocator_process_group ||
              resource.device != allocation.device || resource.pool_count == 0U) {
            continue;
          }
          if (removed_slots > resource.slot_capacity || removed_slots > resource.control_capacity ||
              removed_slots > resource.readiness_capacity ||
              removed_slots > resource.release_capacity ||
              removed_slots > resource.retirement_capacity ||
              removed_slots > resource.quarantine_capacity) {
            return {};
          }
          resource.slot_capacity -= removed_slots;
          resource.control_capacity -= removed_slots;
          resource.readiness_capacity -= removed_slots;
          resource.release_capacity -= removed_slots;
          resource.retirement_capacity -= removed_slots;
          resource.quarantine_capacity -= removed_slots;
        }
        for (auto& capacity : state->profile_manifest.derived_capacities) {
          if (capacity.object_id != allocation.pool_id) {
            continue;
          }
          const bool byte_capacity = capacity.capacity_id == "aggregate_bytes";
          const bool slot_capacity = capacity.capacity_id == "slots" ||
                                     capacity.capacity_id == "control_records" ||
                                     capacity.capacity_id == "readiness_tokens" ||
                                     capacity.capacity_id == "release_records" ||
                                     capacity.capacity_id == "retirement_records" ||
                                     capacity.capacity_id == "quarantine_records";
          if ((byte_capacity && capacity.limit < removed_bytes) ||
              (slot_capacity && capacity.limit < removed_slots)) {
            return {};
          }
          if (byte_capacity) {
            capacity.limit -= removed_bytes;
          } else if (slot_capacity) {
            capacity.limit -= removed_slots;
          }
        }
        break;
      }
      case TensorRuntimeMutation::kAllocationCapacity:
        ++allocation.readiness_capacity;
        break;
      case TensorRuntimeMutation::kAllocationAggregate:
        ++allocation.aggregate_byte_capacity;
        break;
    }
    return ExecutionPlanAccess::make(std::move(state));
  }

  /// @brief Copy one injected plan and corrupt exactly one lifecycle-baseline invariant.
  [[nodiscard]] static ExecutionPlan mutate_lifecycle(const ExecutionPlan& plan,
                                                      LifecycleMutation mutation) {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    auto& transaction_capacity = state->lifecycle_transaction_capacity;
    const OperatorRef first = state->lifecycle_restart_targets.empty()
                                  ? OperatorRef{}
                                  : state->lifecycle_restart_targets.front().target;
    switch (mutation) {
      case LifecycleMutation::kCheckpointMissing:
        state->lifecycle_checkpoints.pop_back();
        break;
      case LifecycleMutation::kCheckpointCallback:
        state->lifecycle_checkpoints.front().callback_present = true;
        break;
      case LifecycleMutation::kCheckpointRequired:
        state->lifecycle_checkpoints.front().required = true;
        break;
      case LifecycleMutation::kCheckpointLevel:
        state->lifecycle_checkpoints.front().stable_topological_level = 1U;
        break;
      case LifecycleMutation::kCheckpointOrder:
        std::swap(state->lifecycle_checkpoints[0], state->lifecycle_checkpoints[1]);
        break;
      case LifecycleMutation::kDependency:
        state->lifecycle_dependencies.push_back({
            .before = first,
            .after = first,
            .stage = LifecycleStage::kStart,
        });
        break;
      case LifecycleMutation::kRestartPermission:
        state->lifecycle_restart_permissions.push_back({.requester = first, .target = first});
        break;
      case LifecycleMutation::kMutableField:
        state->lifecycle_mutable_fields.push_back({.owner = first, .field = 1U});
        break;
      case LifecycleMutation::kTransactionCapacity:
        state->lifecycle_transaction_capacity = LifecycleTransactionCapacityPlanRecord{};
        break;
      case LifecycleMutation::kRestartTargetMissing:
        state->lifecycle_restart_targets.pop_back();
        break;
      case LifecycleMutation::kRestartTargetIdentity:
        ++state->lifecycle_restart_targets.front().target.id;
        break;
      case LifecycleMutation::kRestartTargetAuthority:
        state->lifecycle_restart_targets.front().self_allowed = true;
        state->lifecycle_restart_targets.front().allowed_from_stage_mask = 1U;
        state->lifecycle_restart_targets.front().disposition =
            LifecycleRestartTargetDisposition::kAdmitted;
        break;
      case LifecycleMutation::kRestartClosureMissing:
        state->lifecycle_restart_closures.pop_back();
        break;
      case LifecycleMutation::kRestartStructuralEventCount:
        if (state->lifecycle_restart_closures.empty()) {
          return {};
        }
        --state->lifecycle_restart_closures.front()
              .stage_resolutions.front()
              .required_structural_event_records;
        break;
      case LifecycleMutation::kCapacityDerivationReason:
        if (!transaction_capacity) {
          return {};
        }
        ++transaction_capacity->derivation_reason;
        break;
      case LifecycleMutation::kRequiredCapacityUnderflow:
        if (!transaction_capacity) {
          return {};
        }
        transaction_capacity->configured.max_receipts = 64U;
        transaction_capacity->configured.max_events = 128U;
        transaction_capacity->configured.max_audit_records = 64U;
        transaction_capacity->required_minimum.max_receipts = 64U;
        transaction_capacity->required_minimum.max_events = 128U;
        transaction_capacity->required_minimum.max_audit_records = 64U;
        break;
      case LifecycleMutation::kRequiredCapacityOverflow:
        if (!transaction_capacity) {
          return {};
        }
        transaction_capacity->configured.max_pending_requests =
            std::numeric_limits<std::uint32_t>::max();
        transaction_capacity->configured.max_receipts = std::numeric_limits<std::uint32_t>::max();
        transaction_capacity->configured.max_events = std::numeric_limits<std::uint32_t>::max();
        transaction_capacity->configured.max_audit_records =
            std::numeric_limits<std::uint32_t>::max();
        transaction_capacity->required_minimum.max_receipts =
            std::numeric_limits<std::uint32_t>::max();
        transaction_capacity->required_minimum.max_events =
            std::numeric_limits<std::uint32_t>::max();
        transaction_capacity->required_minimum.max_audit_records =
            std::numeric_limits<std::uint32_t>::max();
        break;
    }
    return ExecutionPlanAccess::make(std::move(state));
  }

  /// @brief Copy one accepted clock-only plan and corrupt inert clock provenance.
  [[nodiscard]] static ExecutionPlan mutate_inert_clock(const ExecutionPlan& plan,
                                                        InertClockMutation mutation) {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source || source->clocks.empty() || source->executable_factories.clocks.empty() ||
        source->clock_conversions.empty()) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    switch (mutation) {
      case InertClockMutation::kPlanClockKey:
        state->clocks.front().key = ClockKey::named("tampered-clock");
        break;
      case InertClockMutation::kMissingFactory:
        state->executable_factories.clocks.pop_back();
        break;
      case InertClockMutation::kFactoryMode:
        state->executable_factories.clocks.front().factory.reset();
        break;
      case InertClockMutation::kRootDefault:
        state->root_default_clock = {};
        break;
      case InertClockMutation::kConversionGeneration:
        state->clock_conversions.front().plan_generation = 0U;
        break;
      case InertClockMutation::kConversionEndpoint:
        state->clock_conversions.front().from = {};
        break;
    }
    return ExecutionPlanAccess::make(std::move(state));
  }

  /// @brief Copy one accepted reusable-Graph plan and corrupt its frozen import chain.
  /// @param plan Accepted plan containing one reusable-Graph clock import and binding.
  /// @param mutation Exact malformed inventory fact to inject.
  /// @return Source-tree-only malformed plan, or an empty facade when unsuitable.
  [[nodiscard]] static ExecutionPlan mutate_clock_import(const ExecutionPlan& plan,
                                                         ClockImportMutation mutation) {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source) {
      return {};
    }
    const auto frozen = GraphCompilerAccess::retain_snapshot(source->graph);
    if (!frozen || frozen->clock_imports.empty() || frozen->clock_bindings.empty()) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    auto snapshot = std::make_shared<GraphSnapshotData>(*frozen);
    switch (mutation) {
      case ClockImportMutation::kMissingImport:
        snapshot->clock_imports.pop_back();
        break;
      case ClockImportMutation::kMissingBinding:
        snapshot->clock_bindings.pop_back();
        break;
      case ClockImportMutation::kMissingImportAndBinding:
        snapshot->clock_imports.pop_back();
        snapshot->clock_bindings.pop_back();
        break;
      case ClockImportMutation::kChangedParentClock: {
        const auto root = std::find_if(snapshot->graphs.begin(),
                                       snapshot->graphs.end(),
                                       [](const auto& graph) { return !graph.parent.has_value(); });
        const auto binding =
            root == snapshot->graphs.end()
                ? snapshot->clock_bindings.end()
                : std::find_if(snapshot->clock_bindings.begin(),
                               snapshot->clock_bindings.end(),
                               [&](const auto& candidate) {
                                 return candidate.parent_graph == root->id &&
                                        candidate.parent_clock == ClockKey::named("runtime_clock");
                               });
        if (binding == snapshot->clock_bindings.end()) {
          return {};
        }
        binding->parent_clock = ClockKey::named("alternate_runtime_clock");
      } break;
      case ClockImportMutation::kNonImmediateParent:
        snapshot->clock_bindings.front().parent_graph =
            snapshot->clock_bindings.front().child_graph;
        break;
      case ClockImportMutation::kFrozenResolvedClock:
        if (state->clock_import_resolutions.empty()) {
          return {};
        }
        state->clock_import_resolutions.front().resolved_clock = {};
        break;
      case ClockImportMutation::kDuplicateImportedDefault:
        if (snapshot->clock_imports.size() < 2U || state->clock_import_resolutions.size() < 2U) {
          return {};
        }
        snapshot->clock_imports[0].is_default = true;
        snapshot->clock_imports[1].is_default = true;
        state->clock_import_resolutions[0].is_default = true;
        state->clock_import_resolutions[1].is_default = true;
        break;
    }
    state->graph = GraphCompilerAccess::canonical_view(std::move(snapshot));
    return ExecutionPlanAccess::make(std::move(state));
  }

  /// @brief Copy one injected plan and replace the first materialization strategy.
  [[nodiscard]] static ExecutionPlan mutate_materialization_strategy(const ExecutionPlan& plan,
                                                                     std::string strategy) {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source || source->profile_manifest.materialization_obligations.empty()) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    state->profile_manifest.materialization_obligations.front().strategy_id = std::move(strategy);
    return ExecutionPlanAccess::make(std::move(state));
  }

  /// @brief Copy one plan and corrupt exactly one package compatibility requirement.
  [[nodiscard]] static ExecutionPlan mutate_profile(const ExecutionPlan& plan,
                                                    ProfileMutation mutation) {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    switch (mutation) {
      case ProfileMutation::kPlanSchema:
        state->profile_manifest.plan_schema_version += ".unsupported";
        break;
      case ProfileMutation::kProfilePolicy:
        ++state->profile_manifest.profile_policy_id.major;
        break;
      case ProfileMutation::kRuntimeAbi:
        state->profile_manifest.runtime_abi_version += ".unsupported";
        break;
      case ProfileMutation::kTarget:
        state->profile_manifest.target_triple += ".unsupported";
        break;
      case ProfileMutation::kCapability:
      case ProfileMutation::kEvidence: {
        const auto& package = runtime_package_capability_manifest();
        if (package.offers.empty()) {
          return {};
        }
        const auto& offer = package.offers.front();
        state->profile_manifest.required_capabilities = {{
            .capability_id = offer.id,
            .version =
                mutation == ProfileMutation::kCapability ? offer.version + 1U : offer.version,
            .evidence_id = mutation == ProfileMutation::kEvidence ? offer.evidence_id + ".wrong"
                                                                  : offer.evidence_id,
        }};
        break;
      }
    }
    return ExecutionPlanAccess::make(std::move(state));
  }
};

}  // namespace holoscan::detail

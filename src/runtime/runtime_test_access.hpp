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
#include <ranges>
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
#include "runtime_phase_counters.hpp"
#include "runtime_support_thread_control.hpp"
#include "runtime_tensor_resources.hpp"

namespace holoscan::detail {

/// Mutate one accepted support-thread affinity after its first successful private observation.
class RuntimeSupportThreadAffinityMutationControl final : public RuntimeSupportThreadControl {
 public:
  RuntimeSupportThreadAffinityMutationControl(std::weak_ptr<ExecutionPlanState> state,
                                              SupportThreadRole role,
                                              std::uint32_t replacement_core)
      : state_(std::move(state)), role_(role), replacement_core_(replacement_core) {}

  [[nodiscard]] expected<void, Error> enter(SupportThreadRole role) override {
    if (role != role_ || entered_.exchange(true, std::memory_order_acq_rel)) {
      return {};
    }
    const auto state = state_.lock();
    if (!state) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    const auto found = std::ranges::find(
        state->support_thread_affinities, role_, &SupportThreadAffinityPlanRecord::role);
    if (found == state->support_thread_affinities.end() || !found->accepted) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    found->requested.affinity = CPUAffinity{replacement_core_};
    found->accepted->affinity = CPUAffinity{replacement_core_};
    return {};
  }

 private:
  std::weak_ptr<ExecutionPlanState> state_;
  SupportThreadRole role_;
  std::uint32_t replacement_core_{};
  std::atomic_bool entered_{};
};

/// @brief Coordinate the physical join owner and peer waiters in source-tree tests.
class RuntimeWaitTestControl final {
 public:
  [[nodiscard]] bool wait_for_join_owner() {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(lock, std::chrono::seconds{5}, [this] { return join_owners_ != 0U; });
  }

  [[nodiscard]] bool wait_for_peer_waiters(std::size_t count) {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(
        lock, std::chrono::seconds{5}, [this, count] { return peer_waiters_ >= count; });
  }

  [[nodiscard]] bool wait_for_bounded_waiters(std::size_t count) {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(
        lock, std::chrono::seconds{5}, [this, count] { return bounded_waiters_ >= count; });
  }

  void release_join_owner() {
    const std::lock_guard lock(mutex_);
    released_ = true;
    cv_.notify_all();
  }

  [[nodiscard]] std::size_t join_owner_count() const {
    const std::lock_guard lock(mutex_);
    return join_owners_;
  }

  void join_owner_selected() {
    std::unique_lock lock(mutex_);
    ++join_owners_;
    cv_.notify_all();
    cv_.wait(lock, [this] { return released_; });
  }

  void peer_waiter_entered() {
    const std::lock_guard lock(mutex_);
    ++peer_waiters_;
    cv_.notify_all();
  }

  void bounded_waiter_entered() {
    const std::lock_guard lock(mutex_);
    ++bounded_waiters_;
    cv_.notify_all();
  }

 private:
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::size_t join_owners_{};
  std::size_t peer_waiters_{};
  std::size_t bounded_waiters_{};
  bool released_{};
};

/// @brief Pause source-tree callbacks immediately before or after their admission claim.
///
/// Production plans leave this control empty. A test can synchronize multiple partition owners
/// at either side of the single session-wide termination linearization point. A bounded initial
/// bypass lets one callback become the requester while peers pause at the selected point.
class RuntimeTerminationAdmissionTestControl final {
 public:
  enum class Point : std::uint8_t {
    kBeforeClaim,
    kAfterClaim,
    kParameterMutexHeld,
  };

  RuntimeTerminationAdmissionTestControl(Point point, std::size_t participant_count,
                                         std::size_t bypass_count = 0U)
      : point_(point), participant_count_(participant_count), bypass_count_(bypass_count) {}

  ~RuntimeTerminationAdmissionTestControl() { release(); }

  void before_claim() { arrive(Point::kBeforeClaim); }
  void after_claim() { arrive(Point::kAfterClaim); }
  void parameter_mutex_held() { arrive(Point::kParameterMutexHeld); }

  /// @brief Make the next source-tree callback fail its parameter-observation boundary.
  void fail_next_parameter_observation() noexcept {
    fail_parameter_observation_.store(true, std::memory_order_release);
  }

  /// @brief Consume one source-tree parameter-observation fault.
  [[nodiscard]] bool consume_parameter_observation_failure() noexcept {
    return fail_parameter_observation_.exchange(false, std::memory_order_acq_rel);
  }

  [[nodiscard]] bool wait_until_arrived() {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(
        lock, std::chrono::seconds{5}, [this] { return arrival_count_ >= participant_count_; });
  }

  void release() {
    const std::lock_guard lock(mutex_);
    released_ = true;
    cv_.notify_all();
  }

  [[nodiscard]] std::size_t arrival_count() const {
    const std::lock_guard lock(mutex_);
    return arrival_count_;
  }

 private:
  void arrive(Point point) {
    std::unique_lock lock(mutex_);
    if (point != point_ || released_) {
      return;
    }
    if (bypassed_count_ < bypass_count_) {
      ++bypassed_count_;
      cv_.notify_all();
      return;
    }
    if (arrival_count_ >= participant_count_) {
      return;
    }
    ++arrival_count_;
    cv_.notify_all();
    cv_.wait(lock, [this] { return released_; });
  }

  mutable std::mutex mutex_;
  std::condition_variable cv_;
  Point point_;
  std::size_t participant_count_{};
  std::size_t bypass_count_{};
  std::size_t bypassed_count_{};
  std::size_t arrival_count_{};
  bool released_{};
  std::atomic_bool fail_parameter_observation_{};
};

/// @brief Hold one frozen source-tree candidate before Dispatcher selection.
///
/// The one-shot barrier makes the lifecycle acceptance boundary deterministic without changing
/// production plan identity. Production plans leave the corresponding inventory pointer empty.
class RuntimePreDispatchTestControl final {
 public:
  explicit RuntimePreDispatchTestControl(OperatorInstanceId blocked_operator)
      : blocked_operator_(blocked_operator) {}

  ~RuntimePreDispatchTestControl() { release(); }

  [[nodiscard]] bool wait_until_paused(
      std::chrono::steady_clock::duration timeout = std::chrono::seconds{5}) {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(lock, timeout, [this] { return paused_; });
  }

  [[nodiscard]] OperatorInstanceId blocked_operator() const noexcept { return blocked_operator_; }

  void candidate_frozen(OperatorInstanceId operator_id) {
    std::unique_lock lock(mutex_);
    if (operator_id != blocked_operator_ || consumed_) {
      return;
    }
    consumed_ = true;
    paused_ = true;
    cv_.notify_all();
    cv_.wait(lock, [this] { return released_; });
  }

  void release() {
    const std::lock_guard lock(mutex_);
    released_ = true;
    cv_.notify_all();
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  OperatorInstanceId blocked_operator_{};
  bool consumed_{};
  bool paused_{};
  bool released_{};
};

/// @brief Pause one source-tree owner immediately before its late readiness fallback drain.
///
/// The one-shot barrier lets a test publish a witness after the ordinary poll and freshness pass
/// but before the fallback binding drain. Production plans leave the inventory pointer empty.
class RuntimeLateReadinessDrainTestControl final {
 public:
  ~RuntimeLateReadinessDrainTestControl() { release(); }

  void arm() {
    const std::lock_guard lock(mutex_);
    armed_ = true;
    cv_.notify_all();
  }

  [[nodiscard]] bool wait_until_paused() {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(lock, std::chrono::seconds{5}, [this] { return paused_; });
  }

  void before_late_drain() {
    std::unique_lock lock(mutex_);
    if (!armed_ || consumed_) {
      return;
    }
    consumed_ = true;
    paused_ = true;
    cv_.notify_all();
    cv_.wait(lock, [this] { return released_; });
  }

  void release() {
    const std::lock_guard lock(mutex_);
    released_ = true;
    cv_.notify_all();
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  bool armed_{};
  bool consumed_{};
  bool paused_{};
  bool released_{};
};

/// @brief Pause a source-tree RunSession after staging direct-continuation evidence.
///
/// The barrier makes the stop/cancellation boundary deterministic without changing production
/// plan identity. Production plans leave the corresponding inventory pointer empty.
class RuntimeDirectContinuationTestControl final {
 public:
  struct CausalFreezeFailureObservation {
    std::size_t pending_before_cancel{};
    std::size_t outstanding_before_cancel{};
    std::size_t pending_after_cancel{};
    std::size_t outstanding_after_cancel{};
    bool receipt_present_after_cancel{};
    bool retired_all{};
    bool recorded{};
  };

  ~RuntimeDirectContinuationTestControl() { release(); }

  [[nodiscard]] bool wait_until_receipt_staged() {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(lock, std::chrono::seconds{5}, [this] { return receipt_staged_; });
  }

  void force_causal_freeze_failure() {
    const std::lock_guard lock(mutex_);
    force_causal_freeze_failure_ = true;
    pause_after_receipt_ = false;
  }

  [[nodiscard]] bool consume_causal_freeze_failure() {
    const std::lock_guard lock(mutex_);
    if (!force_causal_freeze_failure_ || causal_freeze_failure_consumed_) {
      return false;
    }
    causal_freeze_failure_consumed_ = true;
    return true;
  }

  void record_causal_freeze_failure_cleanup(const std::size_t pending_before_cancel,
                                            const std::size_t outstanding_before_cancel,
                                            const std::size_t pending_after_cancel,
                                            const std::size_t outstanding_after_cancel,
                                            const bool receipt_present_after_cancel,
                                            const bool retired_all) {
    const std::lock_guard lock(mutex_);
    causal_freeze_failure_observation_ = CausalFreezeFailureObservation{
        .pending_before_cancel = pending_before_cancel,
        .outstanding_before_cancel = outstanding_before_cancel,
        .pending_after_cancel = pending_after_cancel,
        .outstanding_after_cancel = outstanding_after_cancel,
        .receipt_present_after_cancel = receipt_present_after_cancel,
        .retired_all = retired_all,
        .recorded = true,
    };
    cv_.notify_all();
  }

  [[nodiscard]] CausalFreezeFailureObservation causal_freeze_failure_observation() {
    const std::lock_guard lock(mutex_);
    return causal_freeze_failure_observation_;
  }

  [[nodiscard]] bool wait_until_causal_freeze_failure_cleanup() {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(lock, std::chrono::seconds{5}, [this] {
      return causal_freeze_failure_observation_.recorded;
    });
  }

  void receipt_staged() {
    std::unique_lock lock(mutex_);
    receipt_staged_ = true;
    cv_.notify_all();
    if (pause_after_receipt_) {
      cv_.wait(lock, [this] { return released_; });
    }
  }

  void release() {
    const std::lock_guard lock(mutex_);
    released_ = true;
    cv_.notify_all();
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  bool receipt_staged_{};
  bool released_{};
  bool pause_after_receipt_{true};
  bool force_causal_freeze_failure_{};
  bool causal_freeze_failure_consumed_{};
  CausalFreezeFailureObservation causal_freeze_failure_observation_{};
};

/// @brief Record which partition runner evaluates one source-tree window timeout.
class RuntimeWindowTimeoutTestControl final {
 public:
  struct Observation {
    ExecutionPartitionId partition{};
    VertexId owner{};
  };

  void evaluated(ExecutionPartitionId partition, VertexId owner) {
    const std::lock_guard lock(mutex_);
    observations_.push_back(Observation{.partition = partition, .owner = owner});
  }

  [[nodiscard]] std::vector<Observation> snapshot() const {
    const std::lock_guard lock(mutex_);
    return observations_;
  }

 private:
  mutable std::mutex mutex_;
  std::vector<Observation> observations_;
};

/// @brief Observe and pause one source-tree closed-island transaction.
///
/// Production plans leave this pointer empty. Tests use the control to count allocations on the
/// runner thread, stop precisely between callbacks, or invalidate the generation evidence carried
/// by the source completion report.
class RuntimeClosedIslandTestControl final {
 public:
  using TransactionHook = void (*)(void*) noexcept;
  using CompletionHook = void (*)(void*, const RuntimeDirectCompositeCompletionReport&) noexcept;

  ~RuntimeClosedIslandTestControl() { release(); }

  void set_pause_before_successor(bool pause) noexcept { pause_before_successor_ = pause; }
  void set_pause_at_successor_boundary(bool pause) noexcept {
    pause_at_successor_boundary_ = pause;
  }
  void set_corrupt_generation(bool corrupt) noexcept { corrupt_generation_ = corrupt; }

  void set_transaction_hooks(void* state, TransactionHook begin, TransactionHook end) noexcept {
    transaction_hook_state_ = state;
    begin_transaction_hook_ = begin;
    end_transaction_hook_ = end;
  }

  void set_completion_hook(void* state, CompletionHook completion) noexcept {
    completion_hook_state_ = state;
    completion_hook_ = completion;
  }

  void transaction_started() const noexcept {
    if (begin_transaction_hook_ != nullptr) {
      begin_transaction_hook_(transaction_hook_state_);
    }
  }

  void transaction_finished(const RuntimeDirectCompositeCompletionReport& report) const noexcept {
    if (end_transaction_hook_ != nullptr) {
      end_transaction_hook_(transaction_hook_state_);
    }
    if (completion_hook_ != nullptr) {
      completion_hook_(completion_hook_state_, report);
    }
  }

  [[nodiscard]] bool corrupt_generation() const noexcept { return corrupt_generation_; }

  [[nodiscard]] bool wait_until_before_successor() {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(lock, std::chrono::seconds{5}, [this] { return before_successor_; });
  }

  void before_successor() {
    std::unique_lock lock(mutex_);
    before_successor_ = true;
    cv_.notify_all();
    if (pause_before_successor_) {
      cv_.wait(lock, [this] { return released_; });
    }
  }

  [[nodiscard]] bool wait_until_successor_boundary() {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(lock, std::chrono::seconds{5}, [this] { return successor_boundary_; });
  }

  void successor_boundary() {
    std::unique_lock lock(mutex_);
    successor_boundary_ = true;
    cv_.notify_all();
    if (pause_at_successor_boundary_) {
      cv_.wait(lock, [this] { return released_; });
    }
  }

  void release() {
    const std::lock_guard lock(mutex_);
    released_ = true;
    cv_.notify_all();
  }

 private:
  void* transaction_hook_state_{};
  TransactionHook begin_transaction_hook_{};
  TransactionHook end_transaction_hook_{};
  void* completion_hook_state_{};
  CompletionHook completion_hook_{};
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  bool pause_before_successor_{};
  bool pause_at_successor_boundary_{};
  bool corrupt_generation_{};
  bool before_successor_{};
  bool successor_boundary_{};
  bool released_{};
};

/// @brief Route finite stale source witnesses through the Operator completion final-entry fence.
///
/// Production eagerly cancels a completed receiver. A source-tree test attaches this control to
/// retain already-bounded physical witnesses instead, then counts each callback rejected by the
/// final lane-side completion check. No rejected callback is entered or credited.
class RuntimeOperatorCompletionFinalEntryTestControl final {
 public:
  void checked() noexcept { checked_.fetch_add(1U, std::memory_order_relaxed); }

  void rejected() noexcept { rejected_.fetch_add(1U, std::memory_order_relaxed); }

  void deadline_evaluation_started() noexcept {
    deadline_evaluations_.fetch_add(1U, std::memory_order_relaxed);
  }

  [[nodiscard]] std::size_t rejected_count() const noexcept {
    return rejected_.load(std::memory_order_acquire);
  }

  [[nodiscard]] std::size_t checked_count() const noexcept {
    return checked_.load(std::memory_order_acquire);
  }

  [[nodiscard]] std::size_t deadline_evaluation_count() const noexcept {
    return deadline_evaluations_.load(std::memory_order_acquire);
  }

 private:
  std::atomic<std::size_t> checked_{};
  std::atomic<std::size_t> rejected_{};
  std::atomic<std::size_t> deadline_evaluations_{};
};

/// @brief Carry one source-tree-only initial compute-call progress value across the plan boundary.
class RuntimeOperatorCompletionProgressSeedTestControl final {
 public:
  explicit RuntimeOperatorCompletionProgressSeedTestControl(
      const std::uint64_t successful_compute_calls) noexcept
      : seed_(successful_compute_calls) {}

  [[nodiscard]] bool consume(std::uint64_t& successful_compute_calls) noexcept {
    const std::uint64_t seed = seed_.exchange(kConsumed, std::memory_order_acq_rel);
    if (seed == kConsumed) {
      return false;
    }
    successful_compute_calls = seed;
    return true;
  }

 private:
  static constexpr std::uint64_t kConsumed = std::numeric_limits<std::uint64_t>::max();
  std::atomic<std::uint64_t> seed_;
};

/// @brief Inject one deterministic startup failure after completion progress materializes.
class RuntimeStartupProgressFailureTestControl final {
 public:
  [[nodiscard]] bool consume() noexcept { return fail_.exchange(false, std::memory_order_acq_rel); }

 private:
  std::atomic_bool fail_{true};
};

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
  /// @return Independent plan facade carrying one source-tree-only one-shot retirement fault.
  /// @note The fault is installed before Channel arm and does not enter plan identity or artifacts.
  [[nodiscard]] static ExecutionPlan fail_next_channel_activation_retirement_for_test(
      const ExecutionPlan& plan) noexcept {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    state->executable_factories.test_fail_next_channel_activation_retirement = true;
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
    if (found == state->support_thread_affinities.end() || !found->accepted) {
      return {};
    }
    found->requested.affinity = CPUAffinity{core};
    found->accepted->affinity = CPUAffinity{core};
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

  /// @brief Copy one plan and append a deadline that has no owning temporal trigger.
  [[nodiscard]] static ExecutionPlan with_stray_deadline(const ExecutionPlan& plan) {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    state->deadlines.emplace_back();
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

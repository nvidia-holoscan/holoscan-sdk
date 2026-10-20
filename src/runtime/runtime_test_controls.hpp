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
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_plan_ids.hpp>
#include <holoscan/core/graph_ids.hpp>
#include <holoscan/core/operator_ref.hpp>

#include "../compiler/execution_plan_state.hpp"
#include "port_binding.hpp"
#include "runtime_direct_composite.hpp"
#include "runtime_support_thread_control.hpp"

// Runtime test controls: interleaving, admission, and observation hooks that Runtime consults
// through the executable factory inventory. Production Runtime sources include only this header;
// the source-tree plan builders live in runtime_test_access.hpp.
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
    if (found == state->support_thread_affinities.end()) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    auto& accepted = found->accepted;
    if (!accepted.has_value()) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    found->requested.affinity = CPUAffinity{replacement_core_};
    accepted->affinity = CPUAffinity{replacement_core_};
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
/// but before the fallback binding drain. A partition that may skip that drain still pauses here
/// first. Production plans leave the inventory pointer empty.
class RuntimeLateReadinessDrainTestControl final {
 public:
  ~RuntimeLateReadinessDrainTestControl() { release(); }

  // Pause the first owner to reach its late drain or, given a source, the first owner bound to it.
  void arm(std::optional<SourceId> source = std::nullopt) {
    const std::lock_guard lock(mutex_);
    armed_ = true;
    armed_source_ = source;
    cv_.notify_all();
  }

  [[nodiscard]] bool wait_until_paused() {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(lock, std::chrono::seconds{5}, [this] { return paused_; });
  }

  void before_late_drain(const std::vector<std::unique_ptr<PortBinding>>& bindings) {
    std::unique_lock lock(mutex_);
    if (!armed_ || consumed_) {
      return;
    }
    if (armed_source_ && std::ranges::none_of(bindings, [this](const auto& binding) {
          return binding != nullptr && binding->source() == *armed_source_;
        })) {
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
  std::optional<SourceId> armed_source_;
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

  struct BoundaryFallbackObservation {
    bool group_failed{};
    bool completion_pending{};
    bool input_terminal_pending{};
    bool recorded{};
  };

  void group_failure_published() {
    const std::lock_guard lock{mutex_};
    group_failure_published_ = true;
    cv_.notify_all();
  }

  [[nodiscard]] bool wait_until_group_failure_published() {
    std::unique_lock lock{mutex_};
    return cv_.wait_for(lock, std::chrono::seconds{5}, [this] { return group_failure_published_; });
  }

  void record_boundary_fallback(bool group_failed, bool completion_pending,
                                bool input_terminal_pending) {
    const std::lock_guard lock{mutex_};
    if (!boundary_fallback_observation_.recorded) {
      boundary_fallback_observation_ = BoundaryFallbackObservation{
          group_failed, completion_pending, input_terminal_pending, true};
    }
    cv_.notify_all();
  }

  [[nodiscard]] BoundaryFallbackObservation boundary_fallback_observation() {
    const std::lock_guard lock{mutex_};
    return boundary_fallback_observation_;
  }

  [[nodiscard]] bool wait_until_boundary_fallback() {
    std::unique_lock lock{mutex_};
    return cv_.wait_for(
        lock, std::chrono::seconds{5}, [this] { return boundary_fallback_observation_.recorded; });
  }

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
  bool group_failure_published_{};
  BoundaryFallbackObservation boundary_fallback_observation_;
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

  void max_start_delay_evaluation_started() noexcept {
    max_start_delay_evaluations_.fetch_add(1U, std::memory_order_relaxed);
  }

  [[nodiscard]] std::size_t rejected_count() const noexcept {
    return rejected_.load(std::memory_order_acquire);
  }

  [[nodiscard]] std::size_t checked_count() const noexcept {
    return checked_.load(std::memory_order_acquire);
  }

  [[nodiscard]] std::size_t max_start_delay_evaluation_count() const noexcept {
    return max_start_delay_evaluations_.load(std::memory_order_acquire);
  }

 private:
  std::atomic<std::size_t> checked_{};
  std::atomic<std::size_t> rejected_{};
  std::atomic<std::size_t> max_start_delay_evaluations_{};
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

}  // namespace holoscan::detail

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include <holoscan/core/lifecycle_control.hpp>

#include "../compiler/execution_plan_state.hpp"
#include "../core/lifecycle_transaction_kernel.hpp"
#include "lifecycle_executor.hpp"
#include "parameter_update_coordinator.hpp"
#include "runtime_diagnostic_sink.hpp"

namespace holoscan::detail {

enum class LifecycleRuntimeRequestOrigin : std::uint8_t {
  kParameter,
  kExternal,
  kComputeSelf,
  kComputeCrossTarget,
};

using LifecycleRestartExecutor =
    LifecycleRestartExecutionOutcome (*)(void*, const LifecycleRestartExecutionRequest&) noexcept;

/// Session-local lifecycle ingress, bounded transaction kernel, and tombstone.
class LifecycleControlState final : public std::enable_shared_from_this<LifecycleControlState> {
 public:
  LifecycleControlState(const std::shared_ptr<const ExecutionPlanState>& plan,
                        RunGenerationId run_generation, ProcessGroupId process_group = {});

  LifecycleControlState(const LifecycleControlState&) = delete;
  LifecycleControlState& operator=(const LifecycleControlState&) = delete;
  ~LifecycleControlState() = default;

  [[nodiscard]] expected<LifecycleControlHandle, Error> make_handle(
      const LifecycleAccessPolicy& policy) noexcept;
  [[nodiscard]] expected<LifecycleRestartReceipt, Error> submit(
      LifecycleRuntimeRequestOrigin origin, OperatorRef requester, OperatorRef target,
      LifecycleRestartOptions options, const void* policy) noexcept;

  /// Return the private callback binding used by the run-owned Parameter coordinator.
  [[nodiscard]] ParameterLifecycleSubmitBinding parameter_submit_binding() noexcept;
  [[nodiscard]] expected<LifecycleRestartReceiptView, Error> inspect(
      LifecycleRestartReceipt receipt) const noexcept;
  [[nodiscard]] expected<LifecycleRestartReceiptView, Error> await(
      LifecycleRestartReceipt receipt, std::chrono::nanoseconds timeout) const;

  /// Open external and compute restart submission after successful run startup.
  [[nodiscard]] bool open_ingress() noexcept;

  /// Bind the run-owned wake used to interrupt partition-owner waits after admission.
  void bind_wake(void* state, void (*wake)(void*) noexcept) noexcept;

  /// Return whether accepted lifecycle work requires a control safe point.
  [[nodiscard]] bool has_pending() const noexcept {
    return pending_.load(std::memory_order_acquire);
  }

  /// Reject new submissions without waiting for the worker or acquiring a lock.
  void request_stop_ingress() noexcept;

  /// Execute every currently queued transaction at a Dispatcher control safe point.
  [[nodiscard]] bool process_pending(LifecycleRestartExecutor executor, void* executor_state,
                                     const std::atomic_bool& stop_requested) noexcept;

  /// Complete worker-owned stop cleanup, publish evidence, and retain receipt tombstones.
  void finalize_close() noexcept;

  /// Copy the current bounded receipt-lineage records in acceptance order.
  ///
  /// This private cold-path seam supports runtime handoff and tests without exposing a public
  /// lifecycle cursor. The return value is the number of copied records; callers supply storage.
  [[nodiscard]] std::size_t copy_diagnostic_records(
      std::span<RuntimeLifecycleReceiptDiagnostic> destination) const noexcept;

  [[nodiscard]] bool enabled() const noexcept { return enabled_; }
  [[nodiscard]] RunGenerationId run_generation() const noexcept { return run_generation_; }

 private:
  struct NormalizedPolicy {
    std::vector<OperatorRef> targets;
    std::vector<LifecycleStage> stages;
  };

  struct AcceptedWork {
    LifecycleRestartReceipt receipt;
    LifecycleTransactionId transaction{};
    LifecycleTransactionId related_transaction{};
    LifecycleTransactionGate gate{LifecycleTransactionGate::kControlSafePoint};
    ParamGeneration parameter_generation{};
    bool processed{};
  };

  struct TargetEpoch {
    OperatorInstanceId target{};
    LifecycleEpoch current{1U};
  };

  struct RestartTarget {
    OperatorRef target{};
    std::optional<Profile> effective_profile;
    bool self_allowed{};
    bool external_control_allowed{};
    bool parameter_origin_allowed{};
    std::uint8_t allowed_from_stage_mask{};
    LifecycleRestartTargetDisposition disposition{LifecycleRestartTargetDisposition::kDisabled};
  };

  struct RestartClosure {
    OperatorRef target{};
    std::vector<OperatorInstanceId> closure;
    std::uint8_t valid_requested_stage_mask{};
    std::array<LifecycleRestartStageResolutionPlanRecord, 6> stage_resolutions{};
  };

  struct RestartPermission {
    OperatorInstanceId requester{};
    OperatorInstanceId target{};
  };

  enum class IngressPhase : std::uint8_t {
    kStarting,
    kOpen,
    kStopRequested,
    kClosed,
  };

  friend class LifecycleControlAccess;
  friend class ::holoscan::LifecycleControlHandle;

  [[nodiscard]] expected<std::shared_ptr<const NormalizedPolicy>, Error> normalize_policy(
      const LifecycleAccessPolicy& policy, const NormalizedPolicy* parent = nullptr) const noexcept;
  [[nodiscard]] expected<void, Error> normalize_policy_targets(const LifecycleAccessPolicy& policy,
                                                               const NormalizedPolicy* parent,
                                                               NormalizedPolicy& normalized) const;
  [[nodiscard]] static expected<void, Error> normalize_policy_stages(
      const LifecycleAccessPolicy& policy, const NormalizedPolicy* parent,
      NormalizedPolicy& normalized);
  [[nodiscard]] expected<LifecycleRestartReceipt, Error> submit_locked(
      LifecycleRuntimeRequestOrigin origin, OperatorRef requester, OperatorRef target,
      LifecycleRestartOptions options, const NormalizedPolicy* policy) noexcept;
  [[nodiscard]] expected<ParameterLifecycleSubmitOutcome, Error> submit_parameter(
      const ParameterLifecycleSubmitRequest& request) noexcept;
  [[nodiscard]] expected<ParameterLifecycleSubmitOutcome, Error> submit_parameter_locked(
      const ParameterLifecycleSubmitRequest& request) noexcept;
  static expected<ParameterLifecycleSubmitOutcome, Error> submit_parameter_callback(
      void* state, const ParameterLifecycleSubmitRequest& request) noexcept;
  static void cancel_parameter_callback(void* state,
                                        const ParameterLifecycleSubmitOutcome& outcome) noexcept;
  [[nodiscard]] const RestartTarget* target_record(OperatorInstanceId target) const noexcept;
  [[nodiscard]] const RestartClosure* closure_record(OperatorInstanceId target) const noexcept;
  [[nodiscard]] bool origin_authorized(LifecycleRuntimeRequestOrigin origin, OperatorRef requester,
                                       const RestartTarget& target,
                                       const NormalizedPolicy* policy) const noexcept;
  [[nodiscard]] static Error map_error(const LifecycleTransactionKernelError& error) noexcept;
  [[nodiscard]] expected<LifecycleRestartReceiptView, Error> project_receipt_locked(
      LifecycleRestartReceipt receipt) const noexcept;
  [[nodiscard]] LifecycleRestartReceiptView public_receipt_view(
      const LifecycleTransactionReceiptView& view) const noexcept;
  [[nodiscard]] static RuntimeLifecycleRequestOrigin diagnostic_origin(
      LifecycleRuntimeRequestOrigin origin) noexcept;
  [[nodiscard]] TargetEpoch* target_epoch(OperatorInstanceId target) noexcept;
  [[nodiscard]] const TargetEpoch* target_epoch(OperatorInstanceId target) const noexcept;
  void append_diagnostic_record(LifecycleRuntimeRequestOrigin origin, OperatorRef requester,
                                LifecycleRestartOptions options,
                                const LifecycleTransactionReceiptView& view) noexcept;
  void refresh_transaction_diagnostics(
      LifecycleTransactionId transaction, RuntimeLifecycleRecoveryDisposition recovery,
      const LifecycleRestartExecutionOutcome* execution = nullptr) noexcept;
  void prepare_transaction_diagnostics(LifecycleTransactionId transaction) noexcept;
  void refresh_receipt_diagnostic(const LifecycleTransactionReceiptView& view) noexcept;
  static void publish_diagnostic_records(
      const std::shared_ptr<RuntimeStructuralDiagnosticSink>& sink,
      std::span<const RuntimeLifecycleReceiptDiagnostic> records) noexcept;

  PlanGenerationId plan_generation_{};
  RunGenerationId run_generation_{};
  bool enabled_{};
  std::vector<RestartTarget> restart_targets_;
  std::vector<RestartClosure> restart_closures_;
  std::vector<RestartPermission> restart_permissions_;
  std::shared_ptr<RuntimeStructuralDiagnosticSink> diagnostic_sink_;
  mutable std::mutex mutex_;
  mutable std::condition_variable changed_;
  std::unique_ptr<LifecycleTransactionKernel> kernel_;
  std::vector<AcceptedWork> accepted_;
  std::vector<LifecycleRestartParameterCandidate> transaction_parameter_candidates_;
  std::vector<TargetEpoch> target_epochs_;
  std::vector<RuntimeLifecycleReceiptDiagnostic> diagnostic_records_;
  // Guarded by mutex_. submit() invokes the borrowed callback while holding
  // that lock; finalize_close() acquires it before clearing either pointer.
  void* wake_state_{};
  void (*wake_)(void*) noexcept = nullptr;
  std::atomic_bool pending_{false};
  std::atomic<IngressPhase> ingress_phase_{IngressPhase::kStarting};
  std::atomic_flag finalization_started_ = ATOMIC_FLAG_INIT;
};

/// One process-local, intrusive-reference-counted public-handle representation.
class LifecycleControlHandleState final {
 public:
  LifecycleControlHandleState(const LifecycleControlHandleState&) = delete;
  LifecycleControlHandleState& operator=(const LifecycleControlHandleState&) = delete;

 private:
  LifecycleControlHandleState(std::shared_ptr<LifecycleControlState> control,
                              std::shared_ptr<const void> policy) noexcept
      : control_(std::move(control)), policy_(std::move(policy)) {}
  ~LifecycleControlHandleState() = default;

  void retain() noexcept { references_.fetch_add(1U, std::memory_order_relaxed); }
  void release() noexcept {
    if (references_.fetch_sub(1U, std::memory_order_acq_rel) == 1U) {
      delete this;
    }
  }

  std::atomic<std::size_t> references_{1U};
  std::shared_ptr<LifecycleControlState> control_;
  std::shared_ptr<const void> policy_;

  friend class LifecycleControlAccess;
  friend class ::holoscan::LifecycleControlHandle;
};

/// Private construction bridge used by RunSession without exposing state.
class LifecycleControlAccess {
 public:
  [[nodiscard]] static expected<LifecycleControlHandle, Error> make_handle(
      std::shared_ptr<LifecycleControlState> state, std::shared_ptr<const void> policy) noexcept;
};

}  // namespace holoscan::detail

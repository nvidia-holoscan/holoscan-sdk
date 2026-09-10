/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "lifecycle_control_state.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <utility>
#include <vector>

#include "../core/lifecycle_capacity_derivation.hpp"
#include "runtime_callback_context.hpp"
#include "runtime_worker.hpp"

namespace holoscan {
namespace detail {
namespace {

[[nodiscard]] bool terminal_state(LifecycleTransactionState state) noexcept {
  switch (state) {
    case LifecycleTransactionState::kAcceptedNoChange:
    case LifecycleTransactionState::kPublished:
    case LifecycleTransactionState::kSuperseded:
    case LifecycleTransactionState::kCanceled:
    case LifecycleTransactionState::kRolledBack:
    case LifecycleTransactionState::kFailedStopped:
      return true;
    case LifecycleTransactionState::kRejected:
    case LifecycleTransactionState::kQueued:
    case LifecycleTransactionState::kQuiescing:
    case LifecycleTransactionState::kRestarting:
      return false;
  }
  return false;
}

[[nodiscard]] LifecycleTransactionKernelCapacity kernel_capacity(
    const LifecycleTransactionCapacityValues& values) noexcept {
  return {
      .max_pending_requests = values.max_pending_requests,
      .max_pending_transactions = values.max_pending_transactions,
      .max_receipts = values.max_receipts,
      .max_fenced_operators = values.max_fenced_operators,
      .max_candidate_generations = values.max_candidate_generations,
      .max_events = values.max_events,
      .max_audit_records = values.max_audit_records,
      .max_quiesce_records = values.max_quiesce_records,
      .max_rollback_records = values.max_rollback_records,
  };
}

[[nodiscard]] LifecycleRestartReceiptView base_public_view(
    const LifecycleTransactionReceiptView& view) noexcept {
  return {
      .receipt = view.receipt,
      .state = view.state,
      .transaction_id = view.transaction_id,
      .coalesced_into_transaction = view.coalesced_into_transaction,
      .related_transaction_id = view.related_transaction_id,
      .fence_id = view.fence_id,
      .effective_stage = view.effective_stage,
      .candidate_generation = view.candidate_generation,
  };
}

[[nodiscard]] std::uint8_t stage_bit(LifecycleStage stage) noexcept {
  return static_cast<std::uint8_t>(1U << static_cast<std::uint8_t>(stage));
}

[[nodiscard]] bool valid_transaction_members(std::span<const OperatorInstanceId> members,
                                             OperatorInstanceId target) noexcept {
  if (target == 0U || members.empty()) {
    return false;
  }
  bool contains_target = false;
  OperatorInstanceId previous = 0U;
  for (const OperatorInstanceId member : members) {
    if (member == 0U || member <= previous) {
      return false;
    }
    previous = member;
    contains_target = contains_target || member == target;
  }
  return contains_target;
}

[[nodiscard]] LifecycleTransactionCompatibilityKey shared_compatibility_key(
    std::optional<Profile> profile) noexcept {
  // One LifecycleControlState contains one compiler-admitted authorization,
  // rollback, deadline, and closure-union domain. The effective profile is the
  // only compatibility dimension that can differ within that state.
  return {
      .authorization = 1U,
      .profile = static_cast<std::uint64_t>(profile.value_or(Profile::kStandard)),
      .rollback = 1U,
      .deadline = 0U,
      .closure = 1U,
  };
}

}  // namespace

LifecycleControlState::LifecycleControlState(const std::shared_ptr<const ExecutionPlanState>& plan,
                                             RunGenerationId run_generation,
                                             ProcessGroupId process_group)
    : plan_generation_(plan != nullptr ? plan->plan_generation : PlanGenerationId{}),
      run_generation_(run_generation),
      enabled_(plan != nullptr && plan->lifecycle_transaction_capacity.has_value()),
      diagnostic_sink_(plan != nullptr ? plan->runtime_diagnostic_sink : nullptr) {
  if (plan == nullptr || !plan->lifecycle_transaction_capacity) {
    enabled_ = false;
    return;
  }
  const auto lifecycle_capacity =
      plan->lifecycle_transaction_capacity.value_or(LifecycleTransactionCapacityPlanRecord{});
  const auto& configured = lifecycle_capacity.configured;
  LifecycleTransactionCapacityValues parameter_required{};
  if (const auto& parameter_capacity = plan->parameter_runtime_capacity; parameter_capacity) {
    parameter_required = parameter_capacity->required_lifecycle_capacity;
  }
  std::vector<std::uint8_t> membership_marks(plan->lifecycle_restart_closures.size());
  const auto required = derive_required_lifecycle_capacity(
      configured,
      admitted_lifecycle_restart_capacity(plan->lifecycle_restart_targets,
                                          plan->lifecycle_restart_closures,
                                          plan->plan_generation,
                                          membership_marks),
      parameter_required);
  const bool stage_deltas_valid = std::all_of(
      plan->lifecycle_restart_closures.begin(),
      plan->lifecycle_restart_closures.end(),
      [](const LifecycleRestartClosurePlanRecord& closure) {
        return std::all_of(closure.stage_resolutions.begin(),
                           closure.stage_resolutions.end(),
                           [](const LifecycleRestartStageResolutionPlanRecord& resolution) {
                             return resolution.required_structural_event_records ==
                                    kLifecycleStructuralEventsPerRequest;
                           });
      });
  if (required.arithmetic_overflow || required.invalid_input ||
      !lifecycle_capacity_values_positive(configured) ||
      !lifecycle_capacity_values_equal(lifecycle_capacity.required_minimum, required.values) ||
      !lifecycle_capacity_values_satisfy(configured, required.values) ||
      lifecycle_capacity.derivation_reason != kCanonicalGraphLifecycleCapacityDerivationReason ||
      !stage_deltas_valid) {
    // A malformed injected plan must reach GraphExec's typed cold-start validation without first
    // allocating from wrapped or unbounded capacity evidence.
    enabled_ = false;
    return;
  }
  kernel_ = std::make_unique<LifecycleTransactionKernel>(kernel_capacity(configured));
  accepted_.reserve(configured.max_receipts);
  transaction_parameter_candidates_.reserve(configured.max_candidate_generations);
  diagnostic_records_.reserve(configured.max_receipts);
  restart_targets_.reserve(plan->lifecycle_restart_targets.size());
  target_epochs_.reserve(plan->lifecycle_restart_targets.size());
  for (const auto& target : plan->lifecycle_restart_targets) {
    const auto factory = std::find_if(
        plan->executable_factories.operators.begin(),
        plan->executable_factories.operators.end(),
        [&](const auto& candidate) { return candidate.operator_id == target.target.id; });
    if (factory == plan->executable_factories.operators.end() ||
        (process_group && !worker_owns_vertex(*plan, process_group, factory->vertex))) {
      continue;
    }
    restart_targets_.push_back({
        .target = target.target,
        .effective_profile = target.effective_profile,
        .self_allowed = target.self_allowed,
        .external_control_allowed = target.external_control_allowed,
        .parameter_origin_allowed = target.parameter_origin_allowed,
        .allowed_from_stage_mask = target.allowed_from_stage_mask,
        .disposition = target.disposition,
    });
    if (std::none_of(target_epochs_.begin(), target_epochs_.end(), [&](const TargetEpoch& epoch) {
          return epoch.target == target.target.id;
        })) {
      target_epochs_.push_back({.target = target.target.id, .current = 1U});
    }
  }
  restart_closures_.reserve(plan->lifecycle_restart_closures.size());
  for (const auto& closure : plan->lifecycle_restart_closures) {
    if (std::none_of(restart_targets_.begin(), restart_targets_.end(), [&](const auto& target) {
          return target.target.id == closure.target.id;
        })) {
      continue;
    }
    auto& copied = restart_closures_.emplace_back();
    copied.target = closure.target;
    copied.valid_requested_stage_mask = closure.valid_requested_stage_mask;
    copied.stage_resolutions = closure.stage_resolutions;
    copied.closure.reserve(closure.closure.size());
    for (const OperatorRef ref : closure.closure) {
      copied.closure.push_back(ref.id);
    }
  }
  restart_permissions_.reserve(plan->lifecycle_restart_permissions.size());
  for (const auto& permission : plan->lifecycle_restart_permissions) {
    const bool requester_local =
        std::any_of(restart_targets_.begin(), restart_targets_.end(), [&](const auto& target) {
          return target.target.id == permission.requester.id;
        });
    const bool target_local =
        std::any_of(restart_targets_.begin(), restart_targets_.end(), [&](const auto& target) {
          return target.target.id == permission.target.id;
        });
    if (!requester_local || !target_local) {
      continue;
    }
    restart_permissions_.push_back({
        .requester = permission.requester.id,
        .target = permission.target.id,
    });
  }
}

RuntimeLifecycleRequestOrigin LifecycleControlState::diagnostic_origin(
    LifecycleRuntimeRequestOrigin origin) noexcept {
  switch (origin) {
    case LifecycleRuntimeRequestOrigin::kParameter:
      return RuntimeLifecycleRequestOrigin::kParameter;
    case LifecycleRuntimeRequestOrigin::kExternal:
      return RuntimeLifecycleRequestOrigin::kExternal;
    case LifecycleRuntimeRequestOrigin::kComputeSelf:
      return RuntimeLifecycleRequestOrigin::kComputeSelf;
    case LifecycleRuntimeRequestOrigin::kComputeCrossTarget:
      return RuntimeLifecycleRequestOrigin::kComputeCrossTarget;
  }
  return RuntimeLifecycleRequestOrigin::kExternal;
}

LifecycleControlState::TargetEpoch* LifecycleControlState::target_epoch(
    OperatorInstanceId target) noexcept {
  const auto found = std::find_if(target_epochs_.begin(),
                                  target_epochs_.end(),
                                  [&](const auto& item) { return item.target == target; });
  return found == target_epochs_.end() ? nullptr : &*found;
}

const LifecycleControlState::TargetEpoch* LifecycleControlState::target_epoch(
    OperatorInstanceId target) const noexcept {
  const auto found = std::find_if(target_epochs_.begin(),
                                  target_epochs_.end(),
                                  [&](const auto& item) { return item.target == target; });
  return found == target_epochs_.end() ? nullptr : &*found;
}

void LifecycleControlState::append_diagnostic_record(
    LifecycleRuntimeRequestOrigin origin, OperatorRef requester, LifecycleRestartOptions options,
    const LifecycleTransactionReceiptView& view) noexcept {
  // The kernel and this array share the plan's max_receipts bound. A kernel-accepted receipt must
  // therefore always have a preallocated evidence slot; violating that invariant is an internal
  // contract failure, never a reason to silently drop lineage for an accepted request.
  assert(diagnostic_records_.size() < diagnostic_records_.capacity());
  LifecycleEpoch previous_epoch{};
  LifecycleEpoch candidate_epoch{};
  const auto coalesced =
      std::find_if(diagnostic_records_.begin(), diagnostic_records_.end(), [&](const auto& record) {
        return view.transaction_id != 0U && record.transaction_id == view.transaction_id &&
               record.receipt.target.id == view.receipt.target.id;
      });
  if (coalesced != diagnostic_records_.end()) {
    previous_epoch = coalesced->previous_epoch;
    candidate_epoch = coalesced->candidate_epoch;
  } else if (const TargetEpoch* const epoch = target_epoch(view.receipt.target.id);
             epoch != nullptr) {
    previous_epoch = epoch->current;
    LifecycleEpoch latest_reserved = previous_epoch;
    for (const auto& record : diagnostic_records_) {
      if (record.receipt.target.id == view.receipt.target.id) {
        latest_reserved = std::max(latest_reserved, record.candidate_epoch);
      }
    }
    candidate_epoch =
        latest_reserved == std::numeric_limits<LifecycleEpoch>::max() ? 0U : latest_reserved + 1U;
  }
  const LifecycleRestartReceiptView public_view = public_receipt_view(view);
  diagnostic_records_.push_back({
      .origin = diagnostic_origin(origin),
      .requester = requester,
      .receipt = public_view.receipt,
      .state = public_view.state,
      .transaction_id = public_view.transaction_id,
      .coalesced_into_transaction = public_view.coalesced_into_transaction,
      .related_transaction_id = public_view.related_transaction_id,
      .fence_id = public_view.fence_id,
      .effective_stage = public_view.effective_stage,
      .candidate_generation = public_view.candidate_generation,
      .previous_epoch = previous_epoch,
      .candidate_epoch = candidate_epoch,
      .effective_epoch = previous_epoch,
      .reason = options.reason,
      .recovery = RuntimeLifecycleRecoveryDisposition::kPending,
  });
}

void LifecycleControlState::refresh_receipt_diagnostic(
    const LifecycleTransactionReceiptView& view) noexcept {
  const auto found =
      std::find_if(diagnostic_records_.begin(), diagnostic_records_.end(), [&](const auto& record) {
        return record.receipt.receipt_id == view.receipt.receipt_id;
      });
  if (found == diagnostic_records_.end()) {
    return;
  }
  const LifecycleRestartReceiptView public_view = public_receipt_view(view);
  found->state = public_view.state;
  found->transaction_id = public_view.transaction_id;
  found->coalesced_into_transaction = public_view.coalesced_into_transaction;
  found->related_transaction_id = public_view.related_transaction_id;
  found->fence_id = public_view.fence_id;
  found->effective_stage = public_view.effective_stage;
  found->candidate_generation = public_view.candidate_generation;
}

void LifecycleControlState::prepare_transaction_diagnostics(
    LifecycleTransactionId transaction) noexcept {
  for (auto& record : diagnostic_records_) {
    if (record.transaction_id != transaction) {
      continue;
    }
    const TargetEpoch* const epoch = target_epoch(record.receipt.target.id);
    if (epoch == nullptr) {
      continue;
    }
    record.previous_epoch = epoch->current;
    record.effective_epoch = epoch->current;
    if (record.candidate_epoch <= epoch->current) {
      record.candidate_epoch =
          epoch->current == std::numeric_limits<LifecycleEpoch>::max() ? 0U : epoch->current + 1U;
    }
  }
}

void LifecycleControlState::refresh_transaction_diagnostics(
    LifecycleTransactionId transaction, RuntimeLifecycleRecoveryDisposition recovery,
    const LifecycleRestartExecutionOutcome* execution) noexcept {
  if (recovery == RuntimeLifecycleRecoveryDisposition::kPublished) {
    for (auto& record : diagnostic_records_) {
      if (record.transaction_id != transaction || record.candidate_epoch == 0U) {
        continue;
      }
      TargetEpoch* const epoch = target_epoch(record.receipt.target.id);
      if (epoch != nullptr) {
        epoch->current = record.candidate_epoch;
      }
    }
  }
  for (auto& record : diagnostic_records_) {
    if (record.transaction_id != transaction) {
      continue;
    }
    auto view = kernel_->inspect(record.receipt);
    if (view) {
      refresh_receipt_diagnostic(view.value());
    }
    record.recovery = recovery;
    if (execution != nullptr) {
      record.primary_failure = execution->primary_failure;
      record.rollback_failure = execution->rollback_failure;
      record.restoration_failure = execution->restoration_failure;
      record.restoration_rollback_failure = execution->restoration_rollback_failure;
    }
    switch (recovery) {
      case RuntimeLifecycleRecoveryDisposition::kPending:
        record.effective_epoch = record.previous_epoch;
        break;
      case RuntimeLifecycleRecoveryDisposition::kPublished:
        record.effective_epoch = record.candidate_epoch;
        break;
      case RuntimeLifecycleRecoveryDisposition::kRestoredOldEpoch:
      case RuntimeLifecycleRecoveryDisposition::kCanceled:
        record.effective_epoch = record.previous_epoch;
        break;
      case RuntimeLifecycleRecoveryDisposition::kFailedStopped:
        record.effective_epoch = 0U;
        break;
    }
  }
}

std::size_t LifecycleControlState::copy_diagnostic_records(
    std::span<RuntimeLifecycleReceiptDiagnostic> destination) const noexcept {
  const std::lock_guard lock(mutex_);
  const std::size_t count = std::min(destination.size(), diagnostic_records_.size());
  std::copy_n(diagnostic_records_.begin(), count, destination.begin());
  return count;
}

void LifecycleControlState::publish_diagnostic_records(
    const std::shared_ptr<RuntimeStructuralDiagnosticSink>& sink,
    std::span<const RuntimeLifecycleReceiptDiagnostic> records) noexcept {
  if (sink == nullptr || records.empty()) {
    return;
  }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    sink->publish_lifecycle_receipts(records);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (...) {  // NOLINT(bugprone-empty-catch)
    // Receipt evidence is diagnostic-only. Sink failure cannot change the
    // transaction's terminal receipt, cleanup, or session outcome.
  }
#endif
}

const LifecycleControlState::RestartTarget* LifecycleControlState::target_record(
    OperatorInstanceId target) const noexcept {
  const auto found =
      std::find_if(restart_targets_.begin(), restart_targets_.end(), [target](const auto& record) {
        return record.target.id == target;
      });
  return found == restart_targets_.end() ? nullptr : &*found;
}

const LifecycleControlState::RestartClosure* LifecycleControlState::closure_record(
    OperatorInstanceId target) const noexcept {
  const auto found = std::find_if(
      restart_closures_.begin(), restart_closures_.end(), [target](const auto& record) {
        return record.target.id == target;
      });
  return found == restart_closures_.end() ? nullptr : &*found;
}

expected<std::shared_ptr<const LifecycleControlState::NormalizedPolicy>, Error>
LifecycleControlState::normalize_policy(const LifecycleAccessPolicy& policy,
                                        const NormalizedPolicy* parent) const noexcept {
  constexpr std::size_t kForwardStageCount = 6U;
  const std::size_t target_limit =
      parent == nullptr ? restart_targets_.size() : parent->targets.size();
  const std::size_t stage_limit = parent == nullptr ? kForwardStageCount : parent->stages.size();
  if (policy.allowed_targets.size() > target_limit ||
      policy.allowed_from_stages.size() > stage_limit) {
    return make_unexpected(Error{ErrorCode::kLifecycleRestartUnauthorized});
  }

#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    auto normalized = std::make_shared<NormalizedPolicy>();
    if (auto result = normalize_policy_targets(policy, parent, *normalized); !result) {
      return make_unexpected(std::move(result.error()));
    }
    if (auto result = normalize_policy_stages(policy, parent, *normalized); !result) {
      return make_unexpected(std::move(result.error()));
    }
    return normalized;
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (const std::bad_alloc&) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
#endif
}

expected<void, Error> LifecycleControlState::normalize_policy_targets(
    const LifecycleAccessPolicy& policy, const NormalizedPolicy* parent,
    NormalizedPolicy& normalized) const {
  normalized.targets.reserve(policy.allowed_targets.size());
  for (const OperatorRef target : policy.allowed_targets) {
    if (!target.run_bound() || target.plan_generation != plan_generation_ ||
        target.run_generation != run_generation_) {
      return make_unexpected(Error{ErrorCode::kCrossSessionOperatorRef});
    }
    const auto* record = target_record(target.id);
    if (record == nullptr || record->disposition != LifecycleRestartTargetDisposition::kAdmitted ||
        !record->external_control_allowed) {
      return make_unexpected(Error{ErrorCode::kLifecycleRestartUnauthorized});
    }
    if (parent != nullptr && std::find(parent->targets.begin(), parent->targets.end(), target) ==
                                 parent->targets.end()) {
      return make_unexpected(Error{ErrorCode::kLifecycleRestartUnauthorized});
    }
    if (std::find(normalized.targets.begin(), normalized.targets.end(), target) ==
        normalized.targets.end()) {
      normalized.targets.push_back(target);
    }
  }
  std::sort(normalized.targets.begin(),
            normalized.targets.end(),
            [](OperatorRef lhs, OperatorRef rhs) { return lhs.id < rhs.id; });
  return {};
}

expected<void, Error> LifecycleControlState::normalize_policy_stages(
    const LifecycleAccessPolicy& policy, const NormalizedPolicy* parent,
    NormalizedPolicy& normalized) {
  normalized.stages.reserve(policy.allowed_from_stages.size());
  for (const LifecycleStage stage : policy.allowed_from_stages) {
    const bool valid_stage = is_lifecycle_forward_stage(stage);
    if (!valid_stage ||
        (parent != nullptr &&
         std::find(parent->stages.begin(), parent->stages.end(), stage) == parent->stages.end())) {
      return make_unexpected(Error{valid_stage ? ErrorCode::kLifecycleRestartUnauthorized
                                               : ErrorCode::kLifecycleInvalidRestartStage});
    }
    if (std::find(normalized.stages.begin(), normalized.stages.end(), stage) ==
        normalized.stages.end()) {
      normalized.stages.push_back(stage);
    }
  }
  std::sort(normalized.stages.begin(),
            normalized.stages.end(),
            [](LifecycleStage lhs, LifecycleStage rhs) {
              return static_cast<std::uint8_t>(lhs) < static_cast<std::uint8_t>(rhs);
            });
  return {};
}

expected<LifecycleControlHandle, Error> LifecycleControlState::make_handle(
    const LifecycleAccessPolicy& policy) noexcept {
  const std::lock_guard lock(mutex_);
  if (!enabled_) {
    return make_unexpected(Error{ErrorCode::kLifecycleTransactionsDisabled});
  }
  const IngressPhase phase = ingress_phase_.load(std::memory_order_acquire);
  if (phase == IngressPhase::kStopRequested || phase == IngressPhase::kClosed) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  auto normalized = normalize_policy(policy);
  if (!normalized) {
    return make_unexpected(std::move(normalized.error()));
  }
  return LifecycleControlAccess::make_handle(shared_from_this(), std::move(normalized).value());
}

bool LifecycleControlState::origin_authorized(LifecycleRuntimeRequestOrigin origin,
                                              OperatorRef requester, const RestartTarget& target,
                                              const NormalizedPolicy* policy) const noexcept {
  switch (origin) {
    case LifecycleRuntimeRequestOrigin::kParameter:
      return target.parameter_origin_allowed;
    case LifecycleRuntimeRequestOrigin::kExternal:
      return policy != nullptr && target.external_control_allowed &&
             std::any_of(policy->targets.begin(), policy->targets.end(), [&](OperatorRef allowed) {
               return allowed.id == target.target.id &&
                      allowed.plan_generation == target.target.plan_generation &&
                      allowed.run_generation == run_generation_;
             });
    case LifecycleRuntimeRequestOrigin::kComputeSelf:
      return requester.id == target.target.id && requester.run_generation == run_generation_ &&
             target.self_allowed;
    case LifecycleRuntimeRequestOrigin::kComputeCrossTarget:
      return requester.run_generation == run_generation_ &&
             std::any_of(restart_permissions_.begin(),
                         restart_permissions_.end(),
                         [&](const auto& permission) {
                           return permission.requester == requester.id &&
                                  permission.target == target.target.id;
                         });
  }
  return false;
}

expected<LifecycleRestartReceipt, Error> LifecycleControlState::submit(
    LifecycleRuntimeRequestOrigin origin, OperatorRef requester, OperatorRef target,
    LifecycleRestartOptions options, const void* policy) noexcept {
  expected<LifecycleRestartReceipt, Error> result;
  {
    const std::lock_guard lock(mutex_);
    result = submit_locked(
        origin, requester, target, options, static_cast<const NormalizedPolicy*>(policy));
    if (result) {
      pending_.store(true, std::memory_order_release);
      // The wake state is borrowed from RunSession. Keep its dereference in the same lifetime
      // domain as finalization so a retained public handle cannot race the session's destruction.
      // Invoking after releasing mutex_ would let finalize_close() clear and retire the borrowed
      // state between the copy and call.
      if (wake_ != nullptr) {
        wake_(wake_state_);
      }
    }
  }
  return result;
}

ParameterLifecycleSubmitBinding LifecycleControlState::parameter_submit_binding() noexcept {
  return {
      .state = this,
      .submit = &LifecycleControlState::submit_parameter_callback,
      .cancel_accepted = &LifecycleControlState::cancel_parameter_callback,
  };
}

expected<ParameterLifecycleSubmitOutcome, Error> LifecycleControlState::submit_parameter_callback(
    void* state, const ParameterLifecycleSubmitRequest& request) noexcept {
  if (state == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  return static_cast<LifecycleControlState*>(state)->submit_parameter(request);
}

void LifecycleControlState::cancel_parameter_callback(
    void* state, const ParameterLifecycleSubmitOutcome& outcome) noexcept {
  // A success produced below is assembled from one inspected kernel receipt, so the coordinator's
  // malformed-success unwind is unreachable for this binding. Keep the callback non-null as a
  // defensive contract boundary without canceling unrelated coalesced transaction members.
  static_cast<void>(state);
  static_cast<void>(outcome);
}

expected<ParameterLifecycleSubmitOutcome, Error> LifecycleControlState::submit_parameter(
    const ParameterLifecycleSubmitRequest& request) noexcept {
  expected<ParameterLifecycleSubmitOutcome, Error> result;
  {
    const std::lock_guard lock(mutex_);
    result = submit_parameter_locked(request);
    if (result) {
      pending_.store(true, std::memory_order_release);
      // See submit(): mutex_ also fences the borrowed RunSession wake-state lifetime.
      if (wake_ != nullptr) {
        wake_(wake_state_);
      }
    }
  }
  return result;
}

// Keep Parameter restart admission, predecessor inspection, kernel receipt creation, and
// diagnostic evidence in one mutex-held transaction so no partial outcome becomes observable.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<ParameterLifecycleSubmitOutcome, Error> LifecycleControlState::submit_parameter_locked(
    const ParameterLifecycleSubmitRequest& request) noexcept {
  if (!enabled_ || kernel_ == nullptr) {
    return make_unexpected(Error{ErrorCode::kLifecycleTransactionsDisabled});
  }
  const IngressPhase phase = ingress_phase_.load(std::memory_order_acquire);
  if (phase == IngressPhase::kStarting) {
    return make_unexpected(Error{ErrorCode::kLifecycleRunState});
  }
  if (phase != IngressPhase::kOpen) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  if (request.generation == 0U || !is_lifecycle_forward_stage(request.earliest_stage)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  if (!request.owner.run_bound() || request.owner.plan_generation != plan_generation_) {
    return make_unexpected(Error{ErrorCode::kStaleOperatorRef});
  }
  if (request.owner.run_generation != run_generation_) {
    return make_unexpected(Error{ErrorCode::kCrossSessionOperatorRef});
  }
  const RestartTarget* const target_plan = target_record(request.owner.id);
  const RestartClosure* const closure_plan = closure_record(request.owner.id);
  if (target_plan == nullptr || closure_plan == nullptr || !target_plan->parameter_origin_allowed ||
      target_plan->disposition != LifecycleRestartTargetDisposition::kAdmitted) {
    return make_unexpected(Error{ErrorCode::kLifecycleRestartUnauthorized});
  }
  const std::size_t stage_index = static_cast<std::uint8_t>(request.earliest_stage);
  const std::uint8_t requested_bit = stage_bit(request.earliest_stage);
  if (stage_index >= 6U || (target_plan->allowed_from_stage_mask & requested_bit) == 0U ||
      (closure_plan->valid_requested_stage_mask & requested_bit) == 0U) {
    return make_unexpected(Error{ErrorCode::kLifecycleRestartUnauthorized});
  }
  const auto& resolution = closure_plan->stage_resolutions[stage_index];
  if ((target_plan->allowed_from_stage_mask & stage_bit(resolution.effective_stage)) == 0U) {
    return make_unexpected(Error{ErrorCode::kLifecycleRestartUnauthorized});
  }
  const std::size_t closure_index =
      static_cast<std::size_t>(closure_plan - restart_closures_.data());
  if (closure_index >= restart_closures_.size()) {
    return make_unexpected(Error{ErrorCode::kFailure});
  }

  LifecycleTransactionId prior_parameter_transaction{};
  for (auto candidate = accepted_.rbegin(); candidate != accepted_.rend(); ++candidate) {
    if (candidate->parameter_generation == 0U || candidate->receipt.target.id != request.owner.id ||
        candidate->transaction == 0U) {
      continue;
    }
    auto prior_view = kernel_->inspect(candidate->receipt);
    if (prior_view && !terminal_state(prior_view->state)) {
      prior_parameter_transaction = candidate->transaction;
      break;
    }
  }

  auto accepted = kernel_->submit(LifecycleTransactionRequest{
      .origin = LifecycleTransactionRequestOrigin::kParameter,
      .target = request.owner,
      .requested_stage = request.earliest_stage,
      .effective_stage = resolution.effective_stage,
      .compatibility = shared_compatibility_key(target_plan->effective_profile),
      .closure = restart_closures_[closure_index].closure,
      .candidate_generation = request.generation,
      .parameter_candidate_equivalent = request.equivalent_to_published,
      .target_state = LifecycleTransactionTargetState::kRunning,
      .structural_event_delta = resolution.required_structural_event_records,
  });
  if (!accepted) {
    return make_unexpected(map_error(accepted.error()));
  }
  auto view = kernel_->inspect(*accepted);
  if (!view) {
    return make_unexpected(map_error(view.error()));
  }

  LifecycleTransactionId superseded_transaction{};
  if (prior_parameter_transaction != 0U && terminal_state(view->state)) {
    const auto prior =
        std::find_if(accepted_.rbegin(), accepted_.rend(), [&](const AcceptedWork& candidate) {
          return candidate.parameter_generation != 0U &&
                 candidate.transaction == prior_parameter_transaction &&
                 candidate.receipt.target.id == request.owner.id;
        });
    if (prior != accepted_.rend()) {
      auto prior_view = kernel_->inspect(prior->receipt);
      if (prior_view && prior_view->state == LifecycleTransactionState::kSuperseded) {
        superseded_transaction = prior_parameter_transaction;
      }
    }
  }
  for (auto& candidate : accepted_) {
    if (candidate.processed) {
      continue;
    }
    auto candidate_view = kernel_->inspect(candidate.receipt);
    if (candidate_view && terminal_state(candidate_view->state)) {
      candidate.processed = true;
    }
  }

  LifecycleTransactionId related_transaction{};
  if (view->transaction_id != 0U && view->coalesced_into_transaction == 0U) {
    const auto predecessor =
        std::find_if(accepted_.rbegin(), accepted_.rend(), [&](const AcceptedWork& candidate) {
          return !candidate.processed && candidate.transaction != 0U &&
                 candidate.transaction != view->transaction_id;
        });
    if (predecessor != accepted_.rend()) {
      related_transaction = predecessor->transaction;
      for (auto& candidate : accepted_) {
        if (!candidate.processed && candidate.transaction == related_transaction) {
          candidate.related_transaction = view->transaction_id;
        }
      }
    }
  }
  accepted_.push_back({
      .receipt = *accepted,
      .transaction = view->transaction_id,
      .related_transaction = related_transaction,
      .gate = LifecycleTransactionGate::kActivationRetired,
      .parameter_generation = request.generation,
      .processed = terminal_state(view->state),
  });
  append_diagnostic_record(LifecycleRuntimeRequestOrigin::kParameter,
                           request.owner,
                           LifecycleRestartOptions{.from_stage = request.earliest_stage},
                           *view);
  refresh_transaction_diagnostics(view->transaction_id,
                                  RuntimeLifecycleRecoveryDisposition::kPending);
  if (related_transaction != 0U) {
    refresh_transaction_diagnostics(related_transaction,
                                    RuntimeLifecycleRecoveryDisposition::kPending);
  }
  changed_.notify_all();
  return ParameterLifecycleSubmitOutcome{
      .receipt = *accepted,
      .state = view->state,
      .transaction = view->transaction_id,
      .coalesced_into = view->coalesced_into_transaction,
      .superseded_transaction = superseded_transaction,
  };
}

// Keep authorization, closure resolution, capacity admission, and receipt evidence in one locked
// transaction so no accepted request can escape with only a subset committed.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<LifecycleRestartReceipt, Error> LifecycleControlState::submit_locked(
    LifecycleRuntimeRequestOrigin origin, OperatorRef requester, OperatorRef target,
    LifecycleRestartOptions options, const NormalizedPolicy* policy) noexcept {
  if (!enabled_) {
    return make_unexpected(Error{ErrorCode::kLifecycleTransactionsDisabled});
  }
  const IngressPhase phase = ingress_phase_.load(std::memory_order_acquire);
  if (phase == IngressPhase::kStarting) {
    return make_unexpected(Error{ErrorCode::kLifecycleRunState});
  }
  if (phase != IngressPhase::kOpen) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  if (!is_lifecycle_forward_stage(options.from_stage)) {
    return make_unexpected(Error{ErrorCode::kLifecycleInvalidRestartStage});
  }
  if (!target.run_bound() || target.plan_generation != plan_generation_) {
    return make_unexpected(Error{ErrorCode::kStaleOperatorRef});
  }
  if (origin != LifecycleRuntimeRequestOrigin::kExternal &&
      (!requester.run_bound() || requester.plan_generation != plan_generation_)) {
    return make_unexpected(Error{ErrorCode::kStaleOperatorRef});
  }
  if (target.run_generation != run_generation_ ||
      (requester.valid() && requester.run_generation != run_generation_)) {
    return make_unexpected(Error{ErrorCode::kCrossSessionOperatorRef});
  }
  const auto* target_plan = target_record(target.id);
  const auto* closure_plan = closure_record(target.id);
  if (target_plan == nullptr || closure_plan == nullptr ||
      target_plan->disposition != LifecycleRestartTargetDisposition::kAdmitted ||
      !origin_authorized(origin, requester, *target_plan, policy)) {
    return make_unexpected(Error{ErrorCode::kLifecycleRestartUnauthorized});
  }
  const std::uint8_t bit = stage_bit(options.from_stage);
  const std::size_t stage_index = static_cast<std::uint8_t>(options.from_stage);
  if ((target_plan->allowed_from_stage_mask & bit) == 0U ||
      (closure_plan->valid_requested_stage_mask & bit) == 0U || stage_index >= 6U ||
      (policy != nullptr &&
       std::find(policy->stages.begin(), policy->stages.end(), options.from_stage) ==
           policy->stages.end())) {
    return make_unexpected(Error{ErrorCode::kLifecycleRestartUnauthorized});
  }
  const auto& resolution = closure_plan->stage_resolutions[stage_index];
  if ((target_plan->allowed_from_stage_mask & stage_bit(resolution.effective_stage)) == 0U ||
      (policy != nullptr &&
       std::find(policy->stages.begin(), policy->stages.end(), resolution.effective_stage) ==
           policy->stages.end())) {
    return make_unexpected(Error{ErrorCode::kLifecycleRestartUnauthorized});
  }
  const std::size_t closure_index =
      static_cast<std::size_t>(closure_plan - restart_closures_.data());
  if (closure_index >= restart_closures_.size()) {
    return make_unexpected(Error{ErrorCode::kFailure});
  }

  const auto kernel_origin = origin == LifecycleRuntimeRequestOrigin::kExternal
                                 ? LifecycleTransactionRequestOrigin::kExternalControl
                             : origin == LifecycleRuntimeRequestOrigin::kComputeSelf
                                 ? LifecycleTransactionRequestOrigin::kComputeSelf
                                 : LifecycleTransactionRequestOrigin::kComputeCrossTarget;
  const auto target_state = origin == LifecycleRuntimeRequestOrigin::kExternal
                                ? LifecycleTransactionTargetState::kIdle
                                : LifecycleTransactionTargetState::kRunning;
  const LifecycleTransactionGate gate = origin == LifecycleRuntimeRequestOrigin::kExternal
                                            ? LifecycleTransactionGate::kControlSafePoint
                                            : LifecycleTransactionGate::kActivationRetired;
  auto accepted = kernel_->submit(LifecycleTransactionRequest{
      .origin = kernel_origin,
      .target = target,
      .requested_stage = options.from_stage,
      .effective_stage = resolution.effective_stage,
      .compatibility = shared_compatibility_key(target_plan->effective_profile),
      .closure = restart_closures_[closure_index].closure,
      .candidate_generation = 0U,
      .parameter_candidate_equivalent = false,
      .target_state = target_state,
      .structural_event_delta = resolution.required_structural_event_records,
  });
  if (!accepted) {
    return make_unexpected(map_error(accepted.error()));
  }
  auto view = kernel_->inspect(accepted.value());
  if (!view) {
    return make_unexpected(map_error(view.error()));
  }

  LifecycleTransactionId related_transaction{};
  if (view->transaction_id != 0U && view->coalesced_into_transaction == 0U) {
    const auto predecessor =
        std::find_if(accepted_.rbegin(), accepted_.rend(), [&](const AcceptedWork& candidate) {
          return !candidate.processed && candidate.transaction != 0U &&
                 candidate.transaction != view->transaction_id;
        });
    if (predecessor != accepted_.rend()) {
      related_transaction = predecessor->transaction;
      for (auto& candidate : accepted_) {
        if (!candidate.processed && candidate.transaction == related_transaction) {
          candidate.related_transaction = view->transaction_id;
        }
      }
    }
  }
  accepted_.push_back({
      .receipt = accepted.value(),
      .transaction = view->transaction_id,
      .related_transaction = related_transaction,
      .gate = gate,
      .parameter_generation = 0U,
      .processed = terminal_state(view->state),
  });
  append_diagnostic_record(origin, requester, options, view.value());
  refresh_transaction_diagnostics(view->transaction_id,
                                  RuntimeLifecycleRecoveryDisposition::kPending);
  if (related_transaction != 0U) {
    refresh_transaction_diagnostics(related_transaction,
                                    RuntimeLifecycleRecoveryDisposition::kPending);
  }
  changed_.notify_all();
  return accepted.value();
}

LifecycleRestartReceiptView LifecycleControlState::public_receipt_view(
    const LifecycleTransactionReceiptView& view) const noexcept {
  LifecycleRestartReceiptView result = base_public_view(view);
  if (result.related_transaction_id != 0U) {
    return result;
  }
  const auto accepted =
      std::find_if(accepted_.begin(), accepted_.end(), [&](const AcceptedWork& candidate) {
        return candidate.receipt.receipt_id == view.receipt.receipt_id;
      });
  if (accepted != accepted_.end()) {
    result.related_transaction_id = accepted->related_transaction;
  }
  return result;
}

Error LifecycleControlState::map_error(const LifecycleTransactionKernelError& error) noexcept {
  switch (error.code) {
    case LifecycleTransactionKernelErrorCode::kInvalidRestartStage:
      return Error{ErrorCode::kLifecycleInvalidRestartStage};
    case LifecycleTransactionKernelErrorCode::kCapacityExceeded:
    case LifecycleTransactionKernelErrorCode::kSuccessorUnavailable:
    case LifecycleTransactionKernelErrorCode::kIdentityExhausted:
      return Error{ErrorCode::kLifecycleRestartQueueFull};
    case LifecycleTransactionKernelErrorCode::kIngressClosed:
      return Error{ErrorCode::kClosed};
    case LifecycleTransactionKernelErrorCode::kTargetStateRejected:
      return Error{ErrorCode::kLifecycleRunState};
    case LifecycleTransactionKernelErrorCode::kUnknownReceipt:
      return Error{ErrorCode::kCrossSessionOperatorRef};
    case LifecycleTransactionKernelErrorCode::kGateUnsatisfied:
      return Error{ErrorCode::kLifecycleQuiesceFailed};
    case LifecycleTransactionKernelErrorCode::kInvalidRequest:
    case LifecycleTransactionKernelErrorCode::kUnknownTransaction:
    case LifecycleTransactionKernelErrorCode::kInvalidTransition:
      return Error{ErrorCode::kFailure};
  }
  return Error{ErrorCode::kFailure};
}

expected<LifecycleRestartReceiptView, Error> LifecycleControlState::project_receipt_locked(
    LifecycleRestartReceipt receipt) const noexcept {
  if (!enabled_ || kernel_ == nullptr) {
    return make_unexpected(Error{ErrorCode::kLifecycleTransactionsDisabled});
  }
  if (receipt.target.plan_generation != plan_generation_ ||
      receipt.target.run_generation != run_generation_) {
    return make_unexpected(Error{ErrorCode::kCrossSessionOperatorRef});
  }
  auto view = kernel_->inspect(receipt);
  if (!view) {
    return make_unexpected(map_error(view.error()));
  }
  return public_receipt_view(view.value());
}

expected<LifecycleRestartReceiptView, Error> LifecycleControlState::inspect(
    LifecycleRestartReceipt receipt) const noexcept {
  const std::lock_guard lock(mutex_);
  return project_receipt_locked(receipt);
}

expected<LifecycleRestartReceiptView, Error> LifecycleControlState::await(
    LifecycleRestartReceipt receipt, std::chrono::nanoseconds timeout) const {
  if (timeout.count() < 0) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  std::unique_lock lock(mutex_);
  auto initial = project_receipt_locked(receipt);
  if (!initial || terminal_state(initial->state)) {
    return initial;
  }
  const auto terminal = [&] {
    auto view = kernel_->inspect(receipt);
    return !view || terminal_state(view->state);
  };
  if (!changed_.wait_for(lock, timeout, terminal)) {
    return make_unexpected(Error{ErrorCode::kTimeout});
  }
  return project_receipt_locked(receipt);
}

bool LifecycleControlState::open_ingress() noexcept {
  if (!enabled_) {
    return true;
  }
  IngressPhase expected = IngressPhase::kStarting;
  if (ingress_phase_.compare_exchange_strong(
          expected, IngressPhase::kOpen, std::memory_order_release, std::memory_order_acquire)) {
    changed_.notify_all();
    return true;
  }
  return expected == IngressPhase::kOpen;
}

void LifecycleControlState::bind_wake(void* state, void (*wake)(void*) noexcept) noexcept {
  const std::lock_guard lock(mutex_);
  wake_state_ = state;
  wake_ = wake;
}

void LifecycleControlState::request_stop_ingress() noexcept {
  IngressPhase phase = ingress_phase_.load(std::memory_order_acquire);
  while ((phase == IngressPhase::kStarting || phase == IngressPhase::kOpen) &&
         !ingress_phase_.compare_exchange_weak(phase,
                                               IngressPhase::kStopRequested,
                                               std::memory_order_release,
                                               std::memory_order_acquire)) {
  }
  changed_.notify_all();
}

// The control-safe-point state machine must keep kernel transitions, external execution, and
// terminal receipt publication adjacent to preserve exact rollback ordering.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
bool LifecycleControlState::process_pending(LifecycleRestartExecutor executor, void* executor_state,
                                            const std::atomic_bool& stop_requested) noexcept {
  for (;;) {
    OperatorRef target;
    LifecycleStage effective_stage{LifecycleStage::kConfigure};
    LifecycleTransactionId transaction{};
    std::span<const OperatorInstanceId> transaction_members;
    {
      const std::lock_guard lock(mutex_);
      if (!enabled_ || ingress_phase_.load(std::memory_order_acquire) != IngressPhase::kOpen ||
          stop_requested.load(std::memory_order_acquire)) {
        return true;
      }
      auto work = std::find_if(accepted_.begin(), accepted_.end(), [](const auto& item) {
        return !item.processed && item.transaction != 0U;
      });
      if (work == accepted_.end()) {
        pending_.store(false, std::memory_order_release);
        return true;
      }
      transaction = work->transaction;
      target = work->receipt.target;
      auto effective = kernel_->effective_stage(transaction);
      if (!effective) {
        return false;
      }
      effective_stage = effective.value();
      transaction_parameter_candidates_.clear();
      for (const auto& candidate : accepted_) {
        if (candidate.processed || candidate.transaction != transaction ||
            candidate.parameter_generation == 0U) {
          continue;
        }
        auto candidate_view = kernel_->inspect(candidate.receipt);
        if (!candidate_view || terminal_state(candidate_view->state)) {
          continue;
        }
        const auto duplicate =
            std::find_if(transaction_parameter_candidates_.begin(),
                         transaction_parameter_candidates_.end(),
                         [&](const LifecycleRestartParameterCandidate& retained) {
                           return retained.owner.id == candidate.receipt.target.id;
                         });
        if (duplicate == transaction_parameter_candidates_.end()) {
          assert(transaction_parameter_candidates_.size() <
                 transaction_parameter_candidates_.capacity());
          transaction_parameter_candidates_.push_back({
              .owner = candidate.receipt.target,
              .generation = candidate.parameter_generation,
          });
        } else {
          duplicate->generation = candidate.parameter_generation;
        }
      }
      prepare_transaction_diagnostics(transaction);
      for (auto& candidate : accepted_) {
        if (!candidate.processed && candidate.transaction == transaction) {
          auto acknowledged =
              kernel_->acknowledge_gate(transaction, candidate.receipt.target.id, candidate.gate);
          if (!acknowledged && acknowledged.error().code !=
                                   LifecycleTransactionKernelErrorCode::kInvalidTransition) {
            return false;
          }
        }
      }
      // Inspect the provisional union while holding the control mutex, which
      // serializes every coalescing mutation. Retain the span for execution
      // only after quiescing/restarting closes the coalescing window below.
      auto members = kernel_->transaction_members(transaction);
      if (!members || !valid_transaction_members(*members, target.id)) {
        return false;
      }
      if (!kernel_->begin_quiescing(transaction) || !kernel_->begin_restarting(transaction)) {
        return false;
      }
      transaction_members = *members;
      refresh_transaction_diagnostics(transaction, RuntimeLifecycleRecoveryDisposition::kPending);
      changed_.notify_all();
    }

    const LifecycleRestartExecutionOutcome restarted =
        executor == nullptr
            ? LifecycleRestartExecutionOutcome{}
            : executor(executor_state,
                       LifecycleRestartExecutionRequest{
                           .transaction = transaction,
                           .target = target,
                           .stage = effective_stage,
                           .members = transaction_members,
                           .parameter_candidates = transaction_parameter_candidates_,
                       });

    {
      const std::lock_guard lock(mutex_);
      expected<void, LifecycleTransactionKernelError> completed;
      switch (restarted.disposition) {
        case LifecycleRestartExecutionDisposition::kPublished:
          completed = kernel_->publish(transaction);
          break;
        case LifecycleRestartExecutionDisposition::kRolledBack:
          completed = kernel_->rollback(transaction, true);
          break;
        case LifecycleRestartExecutionDisposition::kFailedStopped:
          completed = kernel_->rollback(transaction, false);
          break;
      }
      if (!completed) {
        return false;
      }
      const RuntimeLifecycleRecoveryDisposition recovery = [&] {
        switch (restarted.disposition) {
          case LifecycleRestartExecutionDisposition::kPublished:
            return RuntimeLifecycleRecoveryDisposition::kPublished;
          case LifecycleRestartExecutionDisposition::kRolledBack:
            return RuntimeLifecycleRecoveryDisposition::kRestoredOldEpoch;
          case LifecycleRestartExecutionDisposition::kFailedStopped:
            return RuntimeLifecycleRecoveryDisposition::kFailedStopped;
        }
        return RuntimeLifecycleRecoveryDisposition::kFailedStopped;
      }();
      refresh_transaction_diagnostics(transaction, recovery, &restarted);
      for (auto& candidate : accepted_) {
        if (candidate.transaction == transaction) {
          candidate.processed = true;
        }
      }
      changed_.notify_all();
      if (restarted.disposition == LifecycleRestartExecutionDisposition::kFailedStopped) {
        return false;
      }
    }
  }
}

void LifecycleControlState::finalize_close() noexcept {
  request_stop_ingress();
  if (finalization_started_.test_and_set(std::memory_order_acq_rel)) {
    return;
  }

  std::shared_ptr<RuntimeStructuralDiagnosticSink> sink;
  {
    const std::lock_guard lock(mutex_);
    // Acquiring the mutex waits for any accepted submitter to finish using the borrowed wake.
    // Clearing it here leaves retained lifecycle handles with a closed, self-contained tombstone.
    wake_ = nullptr;
    wake_state_ = nullptr;
    if (kernel_ != nullptr) {
      kernel_->request_stop();
      for (auto& record : diagnostic_records_) {
        auto view = kernel_->inspect(record.receipt);
        if (view) {
          refresh_receipt_diagnostic(view.value());
          if (view->state == LifecycleTransactionState::kCanceled) {
            record.recovery = RuntimeLifecycleRecoveryDisposition::kCanceled;
            record.effective_epoch = record.previous_epoch;
          }
        }
      }
      static_cast<void>(kernel_->complete_stop_cleanup());
    }
    for (auto& accepted : accepted_) {
      accepted.processed = true;
    }
    changed_.notify_all();
    sink = std::move(diagnostic_sink_);
  }
  publish_diagnostic_records(sink, diagnostic_records_);
  ingress_phase_.store(IngressPhase::kClosed, std::memory_order_release);
  changed_.notify_all();
}

}  // namespace detail

expected<LifecycleControlHandle, Error> detail::LifecycleControlAccess::make_handle(
    std::shared_ptr<detail::LifecycleControlState> state,
    std::shared_ptr<const void> policy) noexcept {
  auto* const handle_state =
      new (std::nothrow) detail::LifecycleControlHandleState{std::move(state), std::move(policy)};
  if (handle_state == nullptr) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
  // Ownership transfers immediately to the intrusive public handle. The analyzer cannot follow
  // the private release path through the move-only expected return.
  LifecycleControlHandle handle{handle_state};  // NOLINT(clang-analyzer-cplusplus.NewDeleteLeaks)
  return expected<LifecycleControlHandle, Error>{std::move(handle)};
}

LifecycleControlHandle::LifecycleControlHandle(detail::LifecycleControlHandleState* state) noexcept
    : state_(state) {}

LifecycleControlHandle::LifecycleControlHandle(const LifecycleControlHandle& other) noexcept
    : state_(other.state_) {
  if (state_ != nullptr) {
    state_->retain();
  }
}

LifecycleControlHandle::LifecycleControlHandle(LifecycleControlHandle&& other) noexcept
    : state_(std::exchange(other.state_, nullptr)) {}

LifecycleControlHandle::~LifecycleControlHandle() noexcept {
  if (state_ != nullptr) {
    state_->release();
  }
}

LifecycleControlHandle& LifecycleControlHandle::operator=(
    const LifecycleControlHandle& other) noexcept {
  if (this == &other) {
    return *this;
  }
  if (other.state_ != nullptr) {
    other.state_->retain();
  }
  if (state_ != nullptr) {
    state_->release();
  }
  state_ = other.state_;
  return *this;
}

LifecycleControlHandle& LifecycleControlHandle::operator=(LifecycleControlHandle&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  if (state_ != nullptr) {
    state_->release();
  }
  state_ = std::exchange(other.state_, nullptr);
  return *this;
}

expected<LifecycleControlHandle, Error> LifecycleControlHandle::restrict(
    const LifecycleAccessPolicy& policy) const noexcept {
  if (state_ == nullptr || state_->control_ == nullptr || state_->policy_ == nullptr) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  const std::lock_guard lock(state_->control_->mutex_);
  const auto phase = state_->control_->ingress_phase_.load(std::memory_order_acquire);
  if (phase == detail::LifecycleControlState::IngressPhase::kStopRequested ||
      phase == detail::LifecycleControlState::IngressPhase::kClosed) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  const auto* parent =
      static_cast<const detail::LifecycleControlState::NormalizedPolicy*>(state_->policy_.get());
  auto normalized = state_->control_->normalize_policy(policy, parent);
  if (!normalized) {
    return make_unexpected(std::move(normalized.error()));
  }
  return detail::LifecycleControlAccess::make_handle(state_->control_,
                                                     std::move(normalized).value());
}

expected<LifecycleRestartReceipt, Error> LifecycleControlHandle::request_restart(
    OperatorRef target, LifecycleRestartOptions options) const noexcept {
  if (detail::runtime_callback_context_restricted()) {
    return make_unexpected(Error{ErrorCode::kLifecycleInvalidCallingContext});
  }
  if (state_ == nullptr || state_->control_ == nullptr || state_->policy_ == nullptr) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  return state_->control_->submit(
      detail::LifecycleRuntimeRequestOrigin::kExternal, {}, target, options, state_->policy_.get());
}

expected<LifecycleRestartReceiptView, Error> LifecycleControlHandle::inspect(
    LifecycleRestartReceipt receipt) const noexcept {
  return state_ != nullptr && state_->control_ != nullptr
             ? state_->control_->inspect(receipt)
             : expected<LifecycleRestartReceiptView, Error>{
                   make_unexpected(Error{ErrorCode::kClosed})};
}

expected<LifecycleRestartReceiptView, Error> LifecycleControlHandle::await(
    LifecycleRestartReceipt receipt, std::chrono::nanoseconds timeout) const {
  if (detail::runtime_callback_context_restricted()) {
    return make_unexpected(Error{ErrorCode::kLifecycleInvalidCallingContext});
  }
  return state_ != nullptr && state_->control_ != nullptr
             ? state_->control_->await(receipt, timeout)
             : expected<LifecycleRestartReceiptView, Error>{
                   make_unexpected(Error{ErrorCode::kClosed})};
}

}  // namespace holoscan

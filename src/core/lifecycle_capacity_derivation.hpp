/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

#include <holoscan/core/execution_plan.hpp>

namespace holoscan::detail {

inline constexpr std::uint32_t kLifecycleStructuralEventsPerRequest = 8U;
inline constexpr ReasonId kCanonicalGraphLifecycleCapacityDerivationReason = 0x4C43415000000001ULL;

struct LifecycleRestartCapacityMaxima {
  std::uint32_t fenced{};
  std::uint32_t quiesce{};
  std::uint32_t rollback{};
  std::uint32_t structural_events_per_request{};
  bool arithmetic_overflow{};
  bool invalid_input{};
};

struct RequiredLifecycleCapacity {
  LifecycleTransactionCapacityValues values;
  bool arithmetic_overflow{};
  bool invalid_input{};
};

[[nodiscard]] constexpr bool lifecycle_capacity_values_positive(
    const LifecycleTransactionCapacityValues& values) noexcept {
  return values.max_pending_requests != 0U && values.max_pending_transactions != 0U &&
         values.max_receipts != 0U && values.max_fenced_operators != 0U &&
         values.max_candidate_generations != 0U && values.max_events != 0U &&
         values.max_audit_records != 0U && values.max_quiesce_records != 0U &&
         values.max_rollback_records != 0U;
}

[[nodiscard]] constexpr bool lifecycle_capacity_values_equal(
    const LifecycleTransactionCapacityValues& left,
    const LifecycleTransactionCapacityValues& right) noexcept {
  return left.max_pending_requests == right.max_pending_requests &&
         left.max_pending_transactions == right.max_pending_transactions &&
         left.max_receipts == right.max_receipts &&
         left.max_fenced_operators == right.max_fenced_operators &&
         left.max_candidate_generations == right.max_candidate_generations &&
         left.max_events == right.max_events && left.max_audit_records == right.max_audit_records &&
         left.max_quiesce_records == right.max_quiesce_records &&
         left.max_rollback_records == right.max_rollback_records;
}

[[nodiscard]] constexpr bool lifecycle_capacity_values_satisfy(
    const LifecycleTransactionCapacityValues& configured,
    const LifecycleTransactionCapacityValues& required) noexcept {
  return configured.max_pending_requests >= required.max_pending_requests &&
         configured.max_pending_transactions >= required.max_pending_transactions &&
         configured.max_receipts >= required.max_receipts &&
         configured.max_fenced_operators >= required.max_fenced_operators &&
         configured.max_candidate_generations >= required.max_candidate_generations &&
         configured.max_events >= required.max_events &&
         configured.max_audit_records >= required.max_audit_records &&
         configured.max_quiesce_records >= required.max_quiesce_records &&
         configured.max_rollback_records >= required.max_rollback_records;
}

[[nodiscard]] constexpr bool checked_lifecycle_multiply(std::uint32_t left, std::uint32_t right,
                                                        std::uint32_t& result) noexcept {
  if (left != 0U && right > std::numeric_limits<std::uint32_t>::max() / left) {
    result = std::numeric_limits<std::uint32_t>::max();
    return false;
  }
  result = left * right;
  return true;
}

// Operator identities are dense in a valid immutable plan. The caller-owned topology-sized mark
// span keeps this pass linear without allocating from untrusted transaction-capacity evidence.
[[nodiscard]] inline LifecycleRestartCapacityMaxima
admitted_lifecycle_restart_capacity(  // NOLINT(readability-function-cognitive-complexity)
    std::span<const LifecycleRestartTargetPlanRecord> targets,
    std::span<const LifecycleRestartClosurePlanRecord> closures, PlanGenerationId plan_generation,
    std::span<std::uint8_t> membership_marks) noexcept {
  LifecycleRestartCapacityMaxima maxima;
  if (plan_generation == 0U || closures.size() > std::numeric_limits<std::uint32_t>::max() ||
      membership_marks.size() < closures.size()) {
    maxima.invalid_input = true;
    return maxima;
  }
  std::fill_n(membership_marks.begin(), closures.size(), 0U);
  if (targets.size() != closures.size()) {
    maxima.invalid_input = true;
  }

  const std::size_t paired_count = std::min(targets.size(), closures.size());
  for (std::size_t target_index = 0U; target_index < paired_count; ++target_index) {
    const auto& target = targets[target_index];
    const auto& closure = closures[target_index];
    const OperatorInstanceId expected_id = static_cast<OperatorInstanceId>(target_index + 1U);
    const bool disposition_valid =
        target.disposition == LifecycleRestartTargetDisposition::kDisabled ||
        target.disposition == LifecycleRestartTargetDisposition::kNotRequested ||
        target.disposition == LifecycleRestartTargetDisposition::kAdmitted ||
        target.disposition == LifecycleRestartTargetDisposition::kRejected;
    if (target.target.id != expected_id || target.target.plan_generation != plan_generation ||
        target.target.run_generation != 0U || closure.target != target.target ||
        !disposition_valid) {
      maxima.invalid_input = true;
      continue;
    }
    OperatorInstanceId previous{};
    bool contains_target = false;
    for (const OperatorRef member : closure.closure) {
      if (member.id == 0U || member.id > closures.size() ||
          member.plan_generation != plan_generation || member.run_generation != 0U ||
          (previous != 0U && member.id <= previous)) {
        maxima.invalid_input = true;
        continue;
      }
      previous = member.id;
      contains_target = contains_target || member.id == target.target.id;
      if (target.disposition == LifecycleRestartTargetDisposition::kAdmitted) {
        auto& mark = membership_marks[member.id - 1U];
        if (mark == 0U) {
          mark = 1U;
          ++maxima.fenced;
        }
      }
    }
    maxima.invalid_input = maxima.invalid_input || !contains_target;
  }

  if (maxima.invalid_input || maxima.fenced == 0U) {
    return maxima;
  }

  // The kernel retains one active transaction plus one post-cutoff successor. Fences are shared
  // by both transactions, while quiesce and rollback journals are charged for both copies of the
  // admitted closure. Each retained request contributes the complete eight-event state history.
  maxima.arithmetic_overflow |= !checked_lifecycle_multiply(maxima.fenced, 2U, maxima.quiesce);
  maxima.arithmetic_overflow |= !checked_lifecycle_multiply(maxima.fenced, 2U, maxima.rollback);
  maxima.structural_events_per_request = kLifecycleStructuralEventsPerRequest;
  return maxima;
}

[[nodiscard]] inline RequiredLifecycleCapacity derive_required_lifecycle_capacity(
    const LifecycleTransactionCapacityValues& configured, LifecycleRestartCapacityMaxima maxima,
    const LifecycleTransactionCapacityValues& parameter_required) noexcept {
  // Every pending slot keeps both its live receipt and a superseded predecessor observable until
  // explicit retirement. Audit records have that same retention lifetime. Active restart work
  // also requires capacity for the current transaction and its post-cutoff successor.
  const std::uint32_t minimum_requests = maxima.structural_events_per_request == 0U ? 1U : 2U;
  const std::uint32_t request_bound = std::max(configured.max_pending_requests, minimum_requests);
  std::uint32_t retained_receipt_bound{};
  bool arithmetic_overflow = maxima.arithmetic_overflow;
  arithmetic_overflow |= !checked_lifecycle_multiply(request_bound, 2U, retained_receipt_bound);
  std::uint32_t required_events{};
  arithmetic_overflow |= !checked_lifecycle_multiply(
      retained_receipt_bound, maxima.structural_events_per_request, required_events);
  const std::uint32_t transaction_bound = std::max(configured.max_pending_transactions, 2U);

  return RequiredLifecycleCapacity{
      .values =
          LifecycleTransactionCapacityValues{
              .max_pending_requests =
                  std::max(minimum_requests, parameter_required.max_pending_requests),
              .max_pending_transactions = std::max(2U, parameter_required.max_pending_transactions),
              .max_receipts = std::max(retained_receipt_bound, parameter_required.max_receipts),
              .max_fenced_operators =
                  std::max(maxima.fenced, parameter_required.max_fenced_operators),
              .max_candidate_generations =
                  std::max(transaction_bound, parameter_required.max_candidate_generations),
              .max_events = std::max(required_events, parameter_required.max_events),
              .max_audit_records =
                  std::max(retained_receipt_bound, parameter_required.max_audit_records),
              .max_quiesce_records =
                  std::max(maxima.quiesce, parameter_required.max_quiesce_records),
              .max_rollback_records =
                  std::max(maxima.rollback, parameter_required.max_rollback_records),
          },
      .arithmetic_overflow = arithmetic_overflow,
      .invalid_input = maxima.invalid_input,
  };
}

}  // namespace holoscan::detail

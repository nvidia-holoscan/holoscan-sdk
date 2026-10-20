/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

#include <holoscan/core/execution_plan.hpp>

#include "../core/diagnostic_registry.hpp"
#include "../threading/thread_placement_contracts.hpp"

namespace holoscan::detail {

[[nodiscard]] inline std::string_view support_thread_role_name(SupportThreadRole role) noexcept {
  switch (role) {
    case SupportThreadRole::kController:
      return "controller";
    case SupportThreadRole::kChannelPump:
      return "channel_pump";
  }
  return "unknown";
}

/// Format a bounded CPU-set summary for cold diagnostics.
[[nodiscard]] inline std::string support_thread_affinity_summary(const CPUAffinity& affinity) {
  constexpr std::size_t kMaximumRenderedCores = 16U;
  std::string summary{"["};
  const std::size_t rendered = std::min(affinity.cores.size(), kMaximumRenderedCores);
  for (std::size_t index = 0U; index < rendered; ++index) {
    if (index != 0U) {
      summary += ',';
    }
    summary += std::to_string(affinity.cores[index]);
  }
  if (rendered < affinity.cores.size()) {
    if (rendered != 0U) {
      summary += ',';
    }
    summary += "...+" + std::to_string(affinity.cores.size() - rendered);
  }
  summary += ']';
  return summary;
}

[[nodiscard]] inline bool valid_support_thread_role(SupportThreadRole role) noexcept {
  switch (role) {
    case SupportThreadRole::kController:
    case SupportThreadRole::kChannelPump:
      return true;
  }
  return false;
}

[[nodiscard]] inline bool affinity_only_support_placement(
    const ThreadPlacement& placement) noexcept {
  return placement.affinity.has_value() &&
         placement.scheduling.policy == SchedulingPolicy::kOther &&
         !placement.scheduling.priority && !placement.scheduling.deadline &&
         placement.wake_mode == WakeMode::kSleepingWaitset &&
         placement.irq_isolation == IRQIsolation::kNotRequested &&
         placement.probe_failure_policy == ProbeFailurePolicy::kFailAdmission;
}

[[nodiscard]] inline bool support_thread_affinity_record_less(
    const SupportThreadAffinityPlanRecord& lhs, const SupportThreadAffinityPlanRecord& rhs) {
  if (lhs.role != rhs.role) {
    return static_cast<std::uint8_t>(lhs.role) < static_cast<std::uint8_t>(rhs.role);
  }
  return thread_placement_less(lhs.requested, rhs.requested);
}

[[nodiscard]] inline bool support_thread_affinity_normalized(
    const ThreadPlacement& requested) noexcept {
  return !requested.affinity ||
         (std::is_sorted(requested.affinity->cores.begin(), requested.affinity->cores.end()) &&
          std::adjacent_find(requested.affinity->cores.begin(), requested.affinity->cores.end()) ==
              requested.affinity->cores.end());
}

/// Validate canonical support-thread records for artifact finalization or admitted Runtime use.
[[nodiscard]] inline bool support_thread_affinity_records_valid(
    std::span<const SupportThreadAffinityPlanRecord> records, bool require_accepted) {
  constexpr auto kUnknown = "SUPPORT_THREAD_PLACEMENT_UNKNOWN_OR_DUPLICATE";
  constexpr auto kRejected = "THREAD_PLACEMENT_REJECTED";
  for (std::size_t index = 0U; index < records.size(); ++index) {
    const auto& record = records[index];
    if (index != 0U && support_thread_affinity_record_less(record, records[index - 1U])) {
      return false;
    }
    const bool role_valid = valid_support_thread_role(record.role);
    const bool request_valid = affinity_only_support_placement(record.requested) &&
                               static_cast<bool>(record.requested.validate());
    const bool normalized = support_thread_affinity_normalized(record.requested);
    const bool duplicate_role =
        (index != 0U && records[index - 1U].role == record.role) ||
        (index + 1U < records.size() && records[index + 1U].role == record.role);
    if (!normalized) {
      return false;
    }
    if (record.disposition == PlanDisposition::kAccepted) {
      if (!role_valid || !request_valid || duplicate_role || record.reason || !record.accepted ||
          !thread_placement_equal(record.requested, *record.accepted)) {
        return false;
      }
      continue;
    }
    if (require_accepted || record.disposition != PlanDisposition::kRejected || record.accepted) {
      return false;
    }
    const auto expected_reason =
        !role_valid || duplicate_role ? diagnostic_id(kUnknown) : diagnostic_id(kRejected);
    if (record.reason != expected_reason) {
      return false;
    }
  }
  return true;
}

}  // namespace holoscan::detail

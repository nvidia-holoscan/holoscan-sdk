/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>

#include <holoscan/threading/thread_placement.hpp>

namespace holoscan::detail {

/// Compare every semantic SCHED_DEADLINE field without depending on object representation.
[[nodiscard]] inline bool deadline_scheduling_equal(
    const std::optional<DeadlineScheduling>& lhs,
    const std::optional<DeadlineScheduling>& rhs) noexcept {
  if (lhs.has_value() != rhs.has_value()) {
    return false;
  }
  return !lhs || (lhs->runtime_ns == rhs->runtime_ns && lhs->deadline_ns == rhs->deadline_ns &&
                  lhs->period_ns == rhs->period_ns);
}

/// Compare every semantic field in one complete thread-placement value.
[[nodiscard]] inline bool thread_placement_equal(const ThreadPlacement& lhs,
                                                 const ThreadPlacement& rhs) noexcept {
  const bool affinity_equal = lhs.affinity.has_value() == rhs.affinity.has_value() &&
                              (!lhs.affinity || lhs.affinity->cores == rhs.affinity->cores);
  return affinity_equal && lhs.scheduling.policy == rhs.scheduling.policy &&
         lhs.scheduling.priority == rhs.scheduling.priority &&
         deadline_scheduling_equal(lhs.scheduling.deadline, rhs.scheduling.deadline) &&
         lhs.wake_mode == rhs.wake_mode && lhs.irq_isolation == rhs.irq_isolation &&
         lhs.probe_failure_policy == rhs.probe_failure_policy;
}

}  // namespace holoscan::detail

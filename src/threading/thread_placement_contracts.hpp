/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

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

/// Compare an optional accepted affinity with one observed thread snapshot.
[[nodiscard]] inline bool thread_affinity_readback_matches(
    const std::optional<CPUAffinity>& accepted, const CPUAffinity& observed) noexcept {
  return !accepted || accepted->cores == observed.cores;
}

/// Order every semantic field in one complete thread-placement value.
[[nodiscard]] inline bool thread_placement_less(const ThreadPlacement& lhs,
                                                const ThreadPlacement& rhs) {
  const std::vector<std::uint32_t> empty;
  const auto& lhs_cores = lhs.affinity ? lhs.affinity->cores : empty;
  const auto& rhs_cores = rhs.affinity ? rhs.affinity->cores : empty;
  return std::tuple{lhs.affinity.has_value(),
                    lhs_cores,
                    lhs.scheduling.policy,
                    lhs.scheduling.priority,
                    lhs.scheduling.deadline.has_value(),
                    lhs.scheduling.deadline ? lhs.scheduling.deadline->runtime_ns : 0U,
                    lhs.scheduling.deadline ? lhs.scheduling.deadline->deadline_ns : 0U,
                    lhs.scheduling.deadline ? lhs.scheduling.deadline->period_ns : 0U,
                    lhs.wake_mode,
                    lhs.irq_isolation,
                    lhs.probe_failure_policy} <
         std::tuple{rhs.affinity.has_value(),
                    rhs_cores,
                    rhs.scheduling.policy,
                    rhs.scheduling.priority,
                    rhs.scheduling.deadline.has_value(),
                    rhs.scheduling.deadline ? rhs.scheduling.deadline->runtime_ns : 0U,
                    rhs.scheduling.deadline ? rhs.scheduling.deadline->deadline_ns : 0U,
                    rhs.scheduling.deadline ? rhs.scheduling.deadline->period_ns : 0U,
                    rhs.wake_mode,
                    rhs.irq_isolation,
                    rhs.probe_failure_policy};
}

}  // namespace holoscan::detail

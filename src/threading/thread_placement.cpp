/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifdef __linux__
#include <sched.h>
#endif

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <string>
#include <utility>

#include <holoscan/threading/thread_placement.hpp>

namespace holoscan {
namespace {

constexpr uint64_t kMaxAffinityRangeExpansion = 1'048'576;

unexpected<Error> invalid_argument(std::string message) {
  return make_unexpected(Error{ErrorCode::kInvalidArgument, std::move(message)});
}

[[nodiscard]] expected<void, Error> append_range(std::vector<uint32_t>* cores, CPUCoreRange range) {
  if (range.last < range.first) {
    return invalid_argument("CPU_AFFINITY_RANGE_INVALID reason=end_before_start");
  }

  const uint64_t expanded_size = static_cast<uint64_t>(range.last) - range.first + 1;
  if (expanded_size > kMaxAffinityRangeExpansion ||
      cores->size() > kMaxAffinityRangeExpansion - expanded_size) {
    return invalid_argument("CPU_AFFINITY_RANGE_INVALID reason=expansion_exceeds_limit");
  }

  for (uint32_t core = range.first;; ++core) {
    cores->push_back(core);
    if (core == range.last) {
      break;
    }
  }
  return {};
}

[[nodiscard]] expected<void, Error> validate_other_policy(const ThreadScheduling& scheduling) {
  if (scheduling.priority.has_value()) {
    return invalid_argument("SCHED_OTHER must not set a real-time priority");
  }
  if (scheduling.deadline.has_value()) {
    return invalid_argument("SCHED_OTHER must not set deadline parameters");
  }
  return {};
}

[[nodiscard]] expected<void, Error> validate_priority_policy(const ThreadScheduling& scheduling,
                                                             const char* policy_name) {
  if (!scheduling.priority.has_value()) {
    return invalid_argument(std::string(policy_name) + " requires a priority");
  }
  if (scheduling.deadline.has_value()) {
    return invalid_argument(std::string(policy_name) + " must not set deadline parameters");
  }

#ifdef __linux__
  const int native_policy = (scheduling.policy == SchedulingPolicy::kFIFO) ? SCHED_FIFO : SCHED_RR;
  const int min_priority = sched_get_priority_min(native_policy);
  const int max_priority = sched_get_priority_max(native_policy);
  if (min_priority >= 0 && max_priority >= 0 &&
      (*scheduling.priority < min_priority || *scheduling.priority > max_priority)) {
    return invalid_argument(std::string(policy_name) + " priority is outside the Linux range [" +
                            std::to_string(min_priority) + ", " + std::to_string(max_priority) +
                            "]");
  }
#else
  return invalid_argument("real-time scheduling policies require Linux support");
#endif

  return {};
}

[[nodiscard]] expected<void, Error> validate_deadline_policy(const ThreadScheduling& scheduling) {
  if (scheduling.priority.has_value()) {
    return invalid_argument("SCHED_DEADLINE must not set a FIFO/RR priority");
  }
  if (!scheduling.deadline.has_value()) {
    return invalid_argument("SCHED_DEADLINE requires deadline parameters");
  }
  auto deadline_result = scheduling.deadline->validate();
  if (!deadline_result) {
    return forward_error(deadline_result);
  }
#ifndef __linux__
  return invalid_argument("SCHED_DEADLINE requires Linux support");
#endif
  return {};
}

}  // namespace

CPUAffinity::CPUAffinity(std::vector<uint32_t> core_ids) : cores(std::move(core_ids)) {
  normalize();
}

CPUAffinity::CPUAffinity(std::initializer_list<uint32_t> core_ids) : cores(core_ids) {
  normalize();
}

expected<CPUAffinity, Error> CPUAffinity::from_ranges(std::span<const CPUCoreRange> ranges) {
  CPUAffinity affinity;
  for (const CPUCoreRange range : ranges) {
    auto append_result = append_range(&affinity.cores, range);
    if (!append_result) {
      return forward_error(append_result);
    }
  }
  affinity.normalize();
  return affinity;
}

expected<CPUAffinity, Error> CPUAffinity::from_ranges(std::initializer_list<CPUCoreRange> ranges) {
  return from_ranges(std::span<const CPUCoreRange>(ranges.begin(), ranges.size()));
}

expected<CPUAffinity, Error> CPUAffinity::from_cores_and_ranges(
    std::span<const uint32_t> core_ids, std::span<const CPUCoreRange> ranges) {
  CPUAffinity affinity;
  affinity.cores.assign(core_ids.begin(), core_ids.end());

  for (const CPUCoreRange range : ranges) {
    auto append_result = append_range(&affinity.cores, range);
    if (!append_result) {
      return forward_error(append_result);
    }
  }

  affinity.normalize();
  return affinity;
}

bool CPUAffinity::empty() const noexcept {
  return cores.empty();
}

void CPUAffinity::normalize() {
  std::sort(cores.begin(), cores.end());
  cores.erase(std::unique(cores.begin(), cores.end()), cores.end());
}

expected<void, Error> CPUAffinity::validate() const {
  if (cores.empty()) {
    return invalid_argument("CPU affinity must contain at least one core");
  }
  return {};
}

expected<void, Error> DeadlineScheduling::validate() const {
  if (runtime_ns == 0 || deadline_ns == 0 || period_ns == 0) {
    return invalid_argument("deadline runtime, deadline, and period must be non-zero");
  }
  if (runtime_ns > deadline_ns) {
    return invalid_argument("deadline runtime must be <= deadline");
  }
  if (deadline_ns > period_ns) {
    return invalid_argument("deadline must be <= period");
  }
  return {};
}

expected<void, Error> ThreadScheduling::validate() const {
  switch (policy) {
    case SchedulingPolicy::kOther:
      return validate_other_policy(*this);
    case SchedulingPolicy::kFIFO:
      return validate_priority_policy(*this, "SCHED_FIFO");
    case SchedulingPolicy::kRoundRobin:
      return validate_priority_policy(*this, "SCHED_RR");
    case SchedulingPolicy::kDeadline:
      return validate_deadline_policy(*this);
  }
  return invalid_argument("unknown scheduling policy");
}

expected<void, Error> ThreadPlacement::validate() const {
  if (affinity.has_value()) {
    auto affinity_result = affinity->validate();
    if (!affinity_result) {
      return forward_error(affinity_result);
    }
  }
  auto scheduling_result = scheduling.validate();
  if (!scheduling_result) {
    return forward_error(scheduling_result);
  }
  switch (wake_mode) {
    case WakeMode::kSleepingWaitset:
    case WakeMode::kBusyPoll:
      break;
    default:
      return invalid_argument("unknown thread wake mode");
  }
  switch (irq_isolation) {
    case IRQIsolation::kNotRequested:
    case IRQIsolation::kPreferIsolated:
    case IRQIsolation::kRequireIsolated:
      break;
    default:
      return invalid_argument("unknown IRQ-isolation request");
  }
  switch (probe_failure_policy) {
    case ProbeFailurePolicy::kWarnAndRecordFallback:
    case ProbeFailurePolicy::kFailAdmission:
      break;
    default:
      return invalid_argument("unknown placement probe failure policy");
  }
  return {};
}

}  // namespace holoscan

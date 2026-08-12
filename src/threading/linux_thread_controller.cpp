/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "linux_thread_controller.hpp"

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#ifdef __linux__
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <fstream>
#include <sstream>
#endif

namespace holoscan::detail {
namespace {

#ifdef __linux__

#ifndef SCHED_DEADLINE
constexpr int kSchedDeadline = 6;
#else
constexpr int kSchedDeadline = SCHED_DEADLINE;
#endif

#ifndef SCHED_FLAG_RESET_ON_FORK
constexpr uint64_t kSchedFlagResetOnFork = 0x01;
#else
constexpr uint64_t kSchedFlagResetOnFork = SCHED_FLAG_RESET_ON_FORK;
#endif

constexpr int kCapSysNice = 23;

struct sched_attr {
  uint32_t size;
  uint32_t sched_policy;
  uint64_t sched_flags;
  int32_t sched_nice;
  uint32_t sched_priority;
  uint64_t sched_runtime;
  uint64_t sched_deadline;
  uint64_t sched_period;
};

unexpected<Error> unexpected_errno(int errno_value, const std::string& operation) {
  return make_unexpected(linux_thread_error_for_errno(errno_value, operation));
}

unexpected<Error> invalid_argument(std::string message) {
  return make_unexpected(Error{ErrorCode::kInvalidArgument, std::move(message)});
}

constexpr uint64_t rlim_to_uint64(rlim_t value) noexcept {
  if (value == RLIM_INFINITY) {
    return std::numeric_limits<uint64_t>::max();
  }
  return static_cast<uint64_t>(value);
}

constexpr int native_priority_policy(SchedulingPolicy policy) noexcept {
  return (policy == SchedulingPolicy::kFIFO) ? SCHED_FIFO : SCHED_RR;
}

expected<void, Error> for_each_requested_core(const CPUAffinity& affinity, cpu_set_t* cpuset,
                                              std::size_t cpuset_bytes) {
  for (const uint32_t core : affinity.cores) {
    if (core >= CPU_SETSIZE) {
      return invalid_argument("CPU affinity core exceeds CPU_SETSIZE");
    }
    CPU_SET_S(core, cpuset_bytes, cpuset);
  }
  return {};
}

expected<CPUAffinity, Error> read_current_affinity() {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  if (sched_getaffinity(0, sizeof(cpuset), &cpuset) != 0) {
    return unexpected_errno(errno, "sched_getaffinity");
  }

  CPUAffinity affinity;
  for (uint32_t core = 0; core < CPU_SETSIZE; ++core) {
    if (CPU_ISSET(core, &cpuset)) {
      affinity.cores.push_back(core);
    }
  }
  return affinity;
}

bool effective_capability_bit(int capability) {
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    if (line.rfind("CapEff:", 0) != 0) {
      continue;
    }
    std::istringstream stream(line.substr(std::strlen("CapEff:")));
    std::string value;
    stream >> value;
    if (value.empty()) {
      return false;
    }
    uint64_t mask = 0;
    const auto* first = value.data();
    const auto* last = value.data() + value.size();
    const auto result = std::from_chars(first, last, mask, 16);
    if (result.ec != std::errc() || result.ptr != last) {
      return false;
    }
    return (mask & (uint64_t{1} << capability)) != 0;
  }
  return false;
}

constexpr bool policy_requests_realtime(const ThreadScheduling& scheduling) noexcept {
  return scheduling.policy == SchedulingPolicy::kFIFO ||
         scheduling.policy == SchedulingPolicy::kRoundRobin ||
         scheduling.policy == SchedulingPolicy::kDeadline;
}

bool can_apply_realtime(const ThreadPlacement& placement, const LinuxThreadProbe& probe) noexcept {
  if (!policy_requests_realtime(placement.scheduling)) {
    return true;
  }
  if (probe.has_cap_sys_nice) {
    return true;
  }
  if (placement.scheduling.policy == SchedulingPolicy::kDeadline) {
    return false;
  }
  return placement.scheduling.priority.has_value() &&
         static_cast<uint64_t>(*placement.scheduling.priority) <= probe.rlimit_rtprio_soft;
}

expected<void, Error> apply_affinity(const CPUAffinity& affinity) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  auto fill_result = for_each_requested_core(affinity, &cpuset, sizeof(cpuset));
  if (!fill_result) {
    return forward_error(fill_result);
  }

  const int status = pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
  if (status != 0) {
    return unexpected_errno(status, "pthread_setaffinity_np");
  }
  return {};
}

expected<void, Error> apply_priority_scheduling(const ThreadScheduling& scheduling) {
  if (!scheduling.priority.has_value()) {
    return invalid_argument("priority scheduling requires a priority value");
  }

  // ThreadScheduling::validate() enforces priority for FIFO/RR; keep this assertion near the
  // dereference for direct internal calls.
  assert(scheduling.priority.has_value());
  sched_param param{};
  param.sched_priority = scheduling.priority.value();
  const int policy = native_priority_policy(scheduling.policy) | SCHED_RESET_ON_FORK;
  if (sched_setscheduler(0, policy, &param) != 0) {
    return unexpected_errno(errno, "sched_setscheduler");
  }
  return {};
}

expected<void, Error> apply_deadline_scheduling(const DeadlineScheduling& deadline) {
  sched_attr attr{};
  attr.size = sizeof(attr);
  attr.sched_policy = kSchedDeadline;
  attr.sched_flags = kSchedFlagResetOnFork;
  attr.sched_runtime = deadline.runtime_ns;
  attr.sched_deadline = deadline.deadline_ns;
  attr.sched_period = deadline.period_ns;

  if (syscall(SYS_sched_setattr, 0, &attr, 0) != 0) {
    return unexpected_errno(errno, "sched_setattr");
  }
  return {};
}

std::optional<DeadlineScheduling> read_deadline_scheduling() {
  sched_attr attr{};
  attr.size = sizeof(attr);
  if (syscall(SYS_sched_getattr, 0, &attr, sizeof(attr), 0) != 0 ||
      attr.sched_policy != static_cast<uint32_t>(kSchedDeadline)) {
    return std::nullopt;
  }
  return DeadlineScheduling{
      .runtime_ns = attr.sched_runtime,
      .deadline_ns = attr.sched_deadline,
      .period_ns = attr.sched_period,
  };
}

constexpr SchedulingPolicy scheduling_policy_from_native(int policy) noexcept {
  const int base_policy = policy & ~SCHED_RESET_ON_FORK;
  if (base_policy == SCHED_FIFO) {
    return SchedulingPolicy::kFIFO;
  }
  if (base_policy == SCHED_RR) {
    return SchedulingPolicy::kRoundRobin;
  }
  if (base_policy == kSchedDeadline) {
    return SchedulingPolicy::kDeadline;
  }
  return SchedulingPolicy::kOther;
}

#endif  // __linux__

}  // namespace

Error linux_thread_error_for_errno(int errno_value, const std::string& operation) {
  ErrorCode code = ErrorCode::kFailure;
  std::string errno_name;
  switch (errno_value) {
    case EPERM:
      errno_name = "EPERM";
      break;
    case EINVAL:
      code = ErrorCode::kInvalidArgument;
      errno_name = "EINVAL";
      break;
    case EBUSY:
      code = ErrorCode::kNotReady;
      errno_name = "EBUSY";
      break;
    default:
      errno_name = "errno " + std::to_string(errno_value);
      break;
  }

  return Error{code, operation + " failed with " + errno_name + ": " + std::strerror(errno_value)};
}

expected<LinuxThreadProbe, Error> LinuxThreadController::probe(const ThreadPlacement& placement) {
#ifndef __linux__
  (void)placement;
  return make_unexpected(
      Error{ErrorCode::kNotImplemented, "LinuxThreadController is only supported on Linux"});
#else
  auto placement_result = placement.validate();
  if (!placement_result) {
    return forward_error(placement_result);
  }

  LinuxThreadProbe probe;
  probe.supported = true;
  probe.fifo_priority_min = sched_get_priority_min(SCHED_FIFO);
  probe.fifo_priority_max = sched_get_priority_max(SCHED_FIFO);
  probe.round_robin_priority_min = sched_get_priority_min(SCHED_RR);
  probe.round_robin_priority_max = sched_get_priority_max(SCHED_RR);
  probe.has_cap_sys_nice = effective_capability_bit(kCapSysNice);

  rlimit rtprio{};
  if (getrlimit(RLIMIT_RTPRIO, &rtprio) != 0) {
    return unexpected_errno(errno, "getrlimit(RLIMIT_RTPRIO)");
  }
  probe.rlimit_rtprio_soft = rlim_to_uint64(rtprio.rlim_cur);
  probe.rlimit_rtprio_hard = rlim_to_uint64(rtprio.rlim_max);

  auto effective_affinity = read_current_affinity();
  if (!effective_affinity) {
    return forward_error(effective_affinity);
  }
  probe.effective_affinity = std::move(*effective_affinity);

  if (placement.affinity.has_value()) {
    for (const uint32_t requested_core : placement.affinity->cores) {
      if (!std::binary_search(probe.effective_affinity.cores.begin(),
                              probe.effective_affinity.cores.end(),
                              requested_core)) {
        probe.unavailable_cores.push_back(requested_core);
      }
    }
  }

  probe.can_apply_realtime = can_apply_realtime(placement, probe);
  return probe;
#endif
}

expected<void, Error> LinuxThreadController::validate_only(const ThreadPlacement& placement) {
#ifndef __linux__
  (void)placement;
  return make_unexpected(
      Error{ErrorCode::kNotImplemented, "LinuxThreadController is only supported on Linux"});
#else
  auto probe_result = probe(placement);
  if (!probe_result) {
    return forward_error(probe_result);
  }

  if (!probe_result->unavailable_cores.empty()) {
    return invalid_argument("requested CPU affinity includes cores outside the effective cpuset");
  }
  if (!probe_result->can_apply_realtime) {
    return make_unexpected(
        Error{ErrorCode::kNotReady, "real-time scheduling requires CAP_SYS_NICE or RLIMIT_RTPRIO"});
  }
  return {};
#endif
}

expected<void, Error> LinuxThreadController::apply(const ThreadPlacement& placement) {
#ifndef __linux__
  (void)placement;
  return make_unexpected(
      Error{ErrorCode::kNotImplemented, "LinuxThreadController is only supported on Linux"});
#else
  auto validation_result = placement.validate();
  if (!validation_result) {
    return forward_error(validation_result);
  }

  if (placement.affinity.has_value()) {
    auto affinity_result = apply_affinity(*placement.affinity);
    if (!affinity_result) {
      return forward_error(affinity_result);
    }
  }

  switch (placement.scheduling.policy) {
    case SchedulingPolicy::kOther:
      // kOther is a placement no-op for freshly-created SCHED_OTHER threads; it does not demote an
      // already real-time thread.
      return {};
    case SchedulingPolicy::kFIFO:
    case SchedulingPolicy::kRoundRobin:
      return apply_priority_scheduling(placement.scheduling);
    case SchedulingPolicy::kDeadline:
      if (!placement.scheduling.deadline.has_value()) {
        return invalid_argument("deadline scheduling requires deadline parameters");
      }
      return apply_deadline_scheduling(placement.scheduling.deadline.value());
  }

  return invalid_argument("unknown scheduling policy");
#endif
}

expected<LinuxThreadSnapshot, Error> LinuxThreadController::snapshot() {
#ifndef __linux__
  return make_unexpected(
      Error{ErrorCode::kNotImplemented, "LinuxThreadController is only supported on Linux"});
#else
  LinuxThreadSnapshot snapshot;

  auto affinity = read_current_affinity();
  if (!affinity) {
    return forward_error(affinity);
  }
  snapshot.affinity = std::move(*affinity);

  const int scheduler_policy = sched_getscheduler(0);
  if (scheduler_policy < 0) {
    return unexpected_errno(errno, "sched_getscheduler");
  }
  snapshot.reset_on_fork = (scheduler_policy & SCHED_RESET_ON_FORK) != 0;
  snapshot.policy = scheduling_policy_from_native(scheduler_policy);

  sched_param param{};
  if (sched_getparam(0, &param) != 0) {
    return unexpected_errno(errno, "sched_getparam");
  }
  if (snapshot.policy == SchedulingPolicy::kFIFO ||
      snapshot.policy == SchedulingPolicy::kRoundRobin) {
    snapshot.priority = param.sched_priority;
  }
  if (snapshot.policy == SchedulingPolicy::kDeadline) {
    snapshot.deadline = read_deadline_scheduling();
  }

  return snapshot;
#endif
}

expected<void, Error> LinuxThreadController::restore(const LinuxThreadSnapshot& snapshot) {
#ifndef __linux__
  (void)snapshot;
  return make_unexpected(
      Error{ErrorCode::kNotImplemented, "LinuxThreadController is only supported on Linux"});
#else
  sched_param parameter{};
  int native_policy = SCHED_OTHER;
  switch (snapshot.policy) {
    case SchedulingPolicy::kOther:
      if (snapshot.reset_on_fork) {
        native_policy |= SCHED_RESET_ON_FORK;
      }
      break;
    case SchedulingPolicy::kFIFO:
    case SchedulingPolicy::kRoundRobin:
      if (!snapshot.priority) {
        return invalid_argument("restoring priority scheduling requires a priority value");
      }
      parameter.sched_priority = *snapshot.priority;
      native_policy = native_priority_policy(snapshot.policy);
      if (snapshot.reset_on_fork) {
        native_policy |= SCHED_RESET_ON_FORK;
      }
      break;
    case SchedulingPolicy::kDeadline: {
      if (!snapshot.deadline) {
        return invalid_argument("restoring deadline scheduling requires deadline parameters");
      }
      sched_attr attribute{};
      attribute.size = sizeof(attribute);
      attribute.sched_policy = kSchedDeadline;
      attribute.sched_flags = snapshot.reset_on_fork ? kSchedFlagResetOnFork : 0U;
      attribute.sched_runtime = snapshot.deadline->runtime_ns;
      attribute.sched_deadline = snapshot.deadline->deadline_ns;
      attribute.sched_period = snapshot.deadline->period_ns;
      if (syscall(SYS_sched_setattr, 0, &attribute, 0) != 0) {
        return unexpected_errno(errno, "sched_setattr");
      }
      auto affinity_result = apply_affinity(snapshot.affinity);
      return affinity_result ? expected<void, Error>{} : forward_error(affinity_result);
    }
  }

  if (sched_setscheduler(0, native_policy, &parameter) != 0) {
    return unexpected_errno(errno, "sched_setscheduler");
  }
  auto affinity_result = apply_affinity(snapshot.affinity);
  return affinity_result ? expected<void, Error>{} : forward_error(affinity_result);
#endif
}

}  // namespace holoscan::detail

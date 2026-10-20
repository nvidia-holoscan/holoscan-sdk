/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/threading/thread_placement.hpp>

namespace holoscan::detail {

struct LinuxThreadProbe {
  bool supported = false;
  CPUAffinity effective_affinity;
  bool has_cap_sys_nice = false;
  uint64_t rlimit_rtprio_soft = 0;
  uint64_t rlimit_rtprio_hard = 0;
  int fifo_priority_min = 0;
  int fifo_priority_max = 0;
  int round_robin_priority_min = 0;
  int round_robin_priority_max = 0;
  std::vector<uint32_t> unavailable_cores;
  bool can_apply_realtime = false;
};

struct LinuxThreadSnapshot {
  CPUAffinity affinity;
  SchedulingPolicy policy = SchedulingPolicy::kOther;
  std::optional<int> priority;
  std::optional<DeadlineScheduling> deadline;
  bool reset_on_fork = false;
};

class LinuxThreadController {
 public:
  [[nodiscard]] static expected<LinuxThreadProbe, Error> probe(const ThreadPlacement& placement);
  [[nodiscard]] static expected<void, Error> validate_only(const ThreadPlacement& placement);
  [[nodiscard]] static expected<void, Error> apply(const ThreadPlacement& placement);
  [[nodiscard]] static expected<LinuxThreadSnapshot, Error> snapshot();
  [[nodiscard]] static expected<void, Error> restore(const LinuxThreadSnapshot& snapshot);
};

[[nodiscard]] Error linux_thread_error_for_errno(int errno_value, const std::string& operation);

}  // namespace holoscan::detail

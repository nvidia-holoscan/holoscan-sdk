/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>

namespace holoscan::benchmarking {

struct PlanningAllocationStats {
  std::size_t allocation_count{};
  std::size_t cumulative_requested_bytes{};
  std::size_t peak_live_requested_bytes{};
  std::size_t live_requested_bytes{};
  bool cross_thread_activity{};
  bool cross_generation_activity{};
};

class PlanningAllocationProbe {
 public:
  /// @warning At most one probe may be active process-wide. The benchmark runs probes serially;
  /// overlapping probes intentionally have no supported ownership semantics.
  PlanningAllocationProbe() noexcept;
  ~PlanningAllocationProbe() noexcept;

  PlanningAllocationProbe(const PlanningAllocationProbe&) = delete;
  PlanningAllocationProbe& operator=(const PlanningAllocationProbe&) = delete;

  [[nodiscard]] PlanningAllocationStats snapshot() const noexcept;
  [[nodiscard]] PlanningAllocationStats finish() noexcept;

 private:
  bool finished_{};
};

}  // namespace holoscan::benchmarking

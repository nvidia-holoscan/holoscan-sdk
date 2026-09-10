/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Smoke: holoscan::run() executes a compiled plan end-to-end using a direct
// (in-process fake) backend; verifies operator callback ordering, successful
// clock reads, and that the activation clock ID matches the registered clock.

#include <atomic>
#include <cstdint>
#include <vector>

#include <holoscan/core/run.hpp>

#include "runtime_test_graph.hpp"
#include "support/direct_port_binding.hpp"

// REQ: RUNTIME-001
// REQ: RUNTIME-002
// REQ: RUNTIME-004
int main() {
  holoscan::test::RuntimeProbeState::reset();

  // The private, bounded direct witness is the baseline fake-input backend. The
  // plan still reconstructs its own SyntheticClock-like test clock and runs
  // through the same free run/session path as ordinary accepted plans.
  const holoscan::ExecutionPlan plan =
      holoscan::test::enable_direct_runtime(holoscan::test::make_runtime_test_plan());
  if (!plan.ok() || holoscan::run(plan) != 0) {
    return 1;  // exit 1: plan compilation failed or holoscan::run() returned a non-zero status
  }

  const std::vector<std::int32_t> expected_events{11, 12, 13};
  if (holoscan::test::RuntimeProbeState::snapshot() != expected_events ||
      !holoscan::test::RuntimeProbeState::observed_clock_read_succeeded.load() ||
      holoscan::test::RuntimeProbeState::observed_clock_id.load() == 0U ||
      holoscan::test::RuntimeProbeState::observed_activation_clock_id.load() !=
          holoscan::test::RuntimeProbeState::observed_clock_id.load()) {
    return 2;  // exit 2: callback sequence wrong, clock read failed, or clock ID mismatch
  }
  return 0;
}

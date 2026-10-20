// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Shared helpers for the graph-level operator tests: run a session to completion (or a
// deadline), assert a clean stop, and the CTest skip protocol for a fully skipped binary.

#pragma once

#include <chrono>
#include <thread>

#include <gtest/gtest.h>

#include <holoscan/core/run.hpp>

namespace holoscan::visualization::test_util {

constexpr int kSkipExitCode = 77;

/// Run the session until it terminates on its own or \p timeout elapses, then stop it and wait.
inline void run_until_done(holoscan::RunSession& session, std::chrono::seconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (session.termination() == holoscan::RunTermination::kRunning &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds{20});
  }
  session.request_stop();
  session.wait();
}

/// Fail on any diagnostic and require a cooperative kStopped termination.
inline void expect_clean_stop(const holoscan::RunSession& session) {
  for (const auto& diagnostic : session.diagnostics()) {
    ADD_FAILURE() << "diagnostic: " << diagnostic.token
                  << " vertex=" << diagnostic.source.vertex.value;
  }
  EXPECT_EQ(session.termination(), holoscan::RunTermination::kStopped);
}

/// Map a gtest result onto the CTest skip protocol (SKIP_RETURN_CODE 77): every selected test
/// skipped => Skipped; otherwise the gtest result.
inline int skip_aware_exit_code(int gtest_result) {
  const auto* unit = ::testing::UnitTest::GetInstance();
  if (gtest_result == 0 && unit->test_to_run_count() > 0 &&
      unit->skipped_test_count() == unit->test_to_run_count()) {
    return kSkipExitCode;
  }
  return gtest_result;
}

}  // namespace holoscan::visualization::test_util

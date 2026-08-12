/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/run.hpp>

#include "runtime_test_graph.hpp"
#include "support/direct_port_binding.hpp"

namespace holoscan {
namespace {

// Builds a single-operator plan with RuntimeProbeOperator under the given Profile.
// RuntimeProbeClock is used so execution completes deterministically without real-time waits.
// Used by ExecutesEveryAdmittedRootProfileWithoutRuntimeReselection.
[[nodiscard]] ExecutionPlan make_runtime_profile_test_plan(Profile profile) {
  Graph graph{"runtime-profile-test"};
  const auto probe = graph.op<test::RuntimeProbeOperator>("operator", 1);
  graph.partition("runtime-spin-worker").add(probe);
  const ClockKey default_clock = graph.add_clock<test::RuntimeProbeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  CompileOptions options;
  options.profile = profile;
  options.deployment.bind_thread_placement("runtime-spin-worker",
                                           ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  return compile(graph, std::move(options));
}

// REQ: RUNTIME-010
// GoogleTest fatal assertions terminate this test before optional access, but clang-tidy does not
// model the macro's control flow.
// NOLINTBEGIN(bugprone-unchecked-optional-access)
TEST(RuntimeProfileTest, ExecutesEveryAdmittedRootProfileWithoutRuntimeReselection) {
  constexpr std::array admitted_profiles{
      Profile::kStandard,
      Profile::kRobotics,
      Profile::kResearch,
  };

  // Key setup: the three admitted root profiles are each compiled and run in isolation.
  for (const Profile profile : admitted_profiles) {
    SCOPED_TRACE(static_cast<std::uint32_t>(profile));
    test::RuntimeProbeState::reset();
    const ExecutionPlan compiled = make_runtime_profile_test_plan(profile);
    ASSERT_TRUE(compiled.ok()) << compiled.json();
    ASSERT_TRUE(compiled.profile_manifest().requested_profile.has_value());
    ASSERT_TRUE(compiled.profile_manifest().effective_graph_profile.has_value());
    EXPECT_EQ(*compiled.profile_manifest().requested_profile, profile);
    EXPECT_EQ(*compiled.profile_manifest().effective_graph_profile, profile);

    const ExecutionPlan plan = test::enable_direct_runtime(compiled);
    ASSERT_TRUE(plan.ok()) << plan.json();
    test::RuntimeProbeState::reset();

    RunSession session = run_async(plan);
    session.wait();

    EXPECT_EQ(session.termination(), RunTermination::kCompleted);
    EXPECT_TRUE(session.diagnostics().empty());
    // Key assertion: effective_graph_profile matches the requested profile; probe sequence
    //               {11,12,13} confirms the operator ran without runtime profile reselection.
    EXPECT_EQ(test::RuntimeProbeState::snapshot(), (std::vector<std::int32_t>{11, 12, 13}));
    EXPECT_EQ(plan.profile_manifest().effective_graph_profile, profile);
  }
}
// NOLINTEND(bugprone-unchecked-optional-access)

}  // namespace
}  // namespace holoscan

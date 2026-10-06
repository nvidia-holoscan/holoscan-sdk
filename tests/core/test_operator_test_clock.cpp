/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <chrono>
#include <memory>

#include <gtest/gtest.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/run.hpp>
#include "testing/controlled_clock_run.hpp"

// REQ: OPERATOR-058
TEST(OperatorTestingClock, TerminalObservationStillRequiresEveryCollectorRetirement) {
  using namespace holoscan;
  using namespace holoscan::detail;
  using namespace std::chrono_literals;
  // A terminal session makes the observer's final-iteration boundary deterministic. The records
  // model the immutable Runtime snapshot independently of scheduling or a thread timing guess.
  for (const bool collected : {false, true}) {
    Graph graph{"terminal-observer"};
    auto session = run_async(compile(graph));
    session.wait();
    ASSERT_FALSE(session.is_running());
    auto observation = std::make_shared<RuntimeFixtureObservation>(4U, 1U, 1U);
    const std::array publication{FixtureObservedPublication{
        .output = PortId{1U}, .sequence = 1U, .sequence_consumed = true}};
    FixtureObservationRecord tick{.kind = FixtureObservationKind::kCandidateFinished,
                                  .owner = VertexId{1U},
                                  .outcome = FixtureCandidateOutcome::kRetired,
                                  .admitted = true};
    tick.witness.kind = RuntimeEventKind::kClockTick;
    tick.witness.reference_time = TimePoint{.timestamp_ns = 1000000000, .clock_id = ClockId{1U}};
    ASSERT_TRUE(observation->append(tick, {}, publication));
    if (collected) {
      FixtureObservedInput input{.input = PortId{2U},
                                 .selection = {InputState::kSelected, SelectionReason::kNone},
                                 .has_sample = true};
      input.sample.source_sequence = 1U;
      const std::array inputs{input};
      ASSERT_TRUE(observation->append(
          FixtureObservationRecord{.kind = FixtureObservationKind::kCandidateFinished,
                                   .owner = VertexId{2U},
                                   .outcome = FixtureCandidateOutcome::kRetired,
                                   .admitted = true},
          inputs,
          {}));
    }
    holoscan::testing::detail::ControlledClockAttempt attempt;
    attempt.observation = observation;
    attempt.tested = VertexId{1U};
    attempt.clock = ClockId{1U};
    attempt.outputs.push_back({.name = "out",
                               .input = PortId{2U},
                               .output = PortId{1U},
                               .source = VertexId{1U},
                               .target = VertexId{2U}});
    SyntheticClock clock{1000000000};
    attempt.drive(session, clock, holoscan::testing::ControlledClockRun{.tick = 1000ms});
    EXPECT_EQ(attempt.boundary_reached, collected);
    EXPECT_EQ(attempt.outcome(),
              collected ? holoscan::testing::ClockTestOutcome::kRetired
                        : holoscan::testing::ClockTestOutcome::kFailed);
    EXPECT_EQ(attempt.failure.empty(), collected);
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

#include <holoscan/sensor_io/sequence_tracker.hpp>

namespace holoscan::sensor_io {
namespace {

// REQ: SENSOR-IO-003
TEST(SensorIoSequenceTracker, FirstObservationReportsNoGapWhateverTheCounterStartsAt) {
  SequenceTracker tracker;

  const SequenceObservation observation = tracker.observe(9'001U);

  EXPECT_TRUE(observation.first);
  EXPECT_EQ(observation.gap, 0U);
  EXPECT_FALSE(observation.regressed);
  EXPECT_EQ(observation.sequence, 9'001U);
  EXPECT_EQ(tracker.total_gap(), 0U);
}

// REQ: SENSOR-IO-003
TEST(SensorIoSequenceTracker, ConsecutiveSamplesReportNoGap) {
  SequenceTracker tracker;

  (void)tracker.observe(4U);
  const SequenceObservation observation = tracker.observe(5U);

  EXPECT_FALSE(observation.first);
  EXPECT_EQ(observation.gap, 0U);
  EXPECT_EQ(tracker.total_gap(), 0U);
}

// The sample after a device overrun is indistinguishable from an on-time sample except for this
// counter, which is the reason the counter is tracked at all.
// REQ: SENSOR-IO-003
TEST(SensorIoSequenceTracker, JumpInCounterReportsTheMissingSampleCount) {
  SequenceTracker tracker;

  (void)tracker.observe(10U);
  const SequenceObservation observation = tracker.observe(14U);

  EXPECT_EQ(observation.gap, 3U);
  EXPECT_FALSE(observation.regressed);
  EXPECT_EQ(tracker.total_gap(), 3U);
}

// REQ: SENSOR-IO-003
TEST(SensorIoSequenceTracker, GapsAccumulateAcrossObservations) {
  SequenceTracker tracker;

  (void)tracker.observe(1U);
  (void)tracker.observe(3U);  // one missed
  (void)tracker.observe(7U);  // three missed
  (void)tracker.observe(8U);  // none missed

  EXPECT_EQ(tracker.total_gap(), 4U);
  EXPECT_EQ(tracker.previous(), 8U);
}

// A counter that repeats or moves backwards is a restarted device or a narrower counter than
// assumed, which is a different problem from a dropped sample and is reported separately.
// REQ: SENSOR-IO-003
TEST(SensorIoSequenceTracker, RepeatedCounterIsReportedAsRegressionNotGap) {
  SequenceTracker tracker;

  (void)tracker.observe(6U);
  const SequenceObservation observation = tracker.observe(6U);

  EXPECT_TRUE(observation.regressed);
  EXPECT_EQ(observation.gap, 0U);
  EXPECT_EQ(tracker.total_gap(), 0U);
}

// REQ: SENSOR-IO-003
TEST(SensorIoSequenceTracker, BackwardsCounterIsReportedAsRegression) {
  SequenceTracker tracker;

  (void)tracker.observe(100U);
  const SequenceObservation observation = tracker.observe(2U);

  EXPECT_TRUE(observation.regressed);
  EXPECT_EQ(observation.gap, 0U);
  EXPECT_EQ(tracker.previous(), 2U);
}

// REQ: SENSOR-IO-003
TEST(SensorIoSequenceTracker, ResetStartsANewEpochSoARestartIsNotAnEnormousGap) {
  SequenceTracker tracker;

  (void)tracker.observe(std::numeric_limits<std::uint64_t>::max() - 1U);
  tracker.reset();
  EXPECT_FALSE(tracker.seen());

  const SequenceObservation observation = tracker.observe(0U);

  EXPECT_TRUE(observation.first);
  EXPECT_EQ(observation.gap, 0U);
  EXPECT_FALSE(observation.regressed);
  EXPECT_EQ(tracker.total_gap(), 0U);
}

}  // namespace
}  // namespace holoscan::sensor_io

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include <holoscan/core/lifecycle.hpp>
#include <holoscan/sensor_io/clock_discipline.hpp>
#include <holoscan/time/time_point.hpp>

namespace holoscan::sensor_io {
namespace {

constexpr ClockId kReferenceDomain{7};

// The documented default admits one millisecond, which is where a measured offset stops beating the
// arrival-time stamping it exists to replace.
static_assert(ClockDiscipline::kDefaultMaxUncertaintyNs == 1'000'000);

/// Replays a scripted sequence of clock reads so a test can control the bracket width of each
/// measurement round exactly.
class ScriptedClock {
 public:
  explicit ScriptedClock(std::vector<std::optional<std::int64_t>> readings)
      : readings_(std::move(readings)) {}

  [[nodiscard]] auto reader() {
    return [this]() noexcept -> std::optional<std::int64_t> {
      if (index_ >= readings_.size()) {
        return std::nullopt;
      }
      return readings_[index_++];
    };
  }

 private:
  std::vector<std::optional<std::int64_t>> readings_;
  std::size_t index_{};
};

[[nodiscard]] auto constant_reader(std::int64_t value) {
  return [value]() noexcept { return std::optional<std::int64_t>{value}; };
}

[[nodiscard]] auto unavailable_reader() {
  return []() noexcept { return std::optional<std::int64_t>{}; };
}

// REQ: SENSOR-IO-002
TEST(SensorIoClockDiscipline, UnboundDisciplineProjectsNothing) {
  const ClockDiscipline discipline;

  EXPECT_FALSE(discipline.settled());
  EXPECT_EQ(discipline.domain(), ClockId{});
  EXPECT_FALSE(discipline.project(1'000).has_value());
}

// Binding names the domain but does not establish the offset. Reporting absence here is what stops
// an unmeasured device timestamp from being published as if it were a reference-domain coordinate.
// REQ: SENSOR-IO-002
TEST(SensorIoClockDiscipline, BoundButUnprimedDisciplineProjectsNothing) {
  ClockDiscipline discipline;
  discipline.bind(kReferenceDomain);

  EXPECT_FALSE(discipline.settled());
  EXPECT_EQ(discipline.domain(), kReferenceDomain);
  EXPECT_FALSE(discipline.project(1'000).has_value());
}

// REQ: SENSOR-IO-002
TEST(SensorIoClockDiscipline, IdentityBindingProjectsWithoutMeasurement) {
  ClockDiscipline discipline;
  discipline.bind_identity(kReferenceDomain);

  ASSERT_TRUE(discipline.settled());
  const std::optional<TimePoint> projected = discipline.project(12'345);

  ASSERT_TRUE(projected.has_value());
  EXPECT_EQ(projected->timestamp_ns, 12'345);
  EXPECT_EQ(projected->clock_id, kReferenceDomain);
  EXPECT_EQ(discipline.estimate().offset_ns, 0);
}

// An identity binding asserts that the device stamps in a named domain, so it asserts nothing when
// no domain is named. Reporting settled for an unset ClockId is the damaging half: operators guard
// priming with settled() precisely because the Core clock resolver gap forces the call into the
// first compute, so a discipline that claimed to be settled would never measure and would then
// project nothing, for the rest of the epoch, without failing anything.
// REQ: SENSOR-IO-002
TEST(SensorIoClockDiscipline, IdentityBindingAnUnsetDomainLeavesTheDisciplineUnsettled) {
  ClockDiscipline discipline;
  discipline.bind_identity(ClockId{});

  EXPECT_FALSE(discipline.settled());
  EXPECT_FALSE(discipline.project(12'345).has_value());
  // Still primeable, which is the point of not claiming to be settled: the operator that binds a
  // domain it can resolve gets a measurement rather than a discipline stuck asserting nothing.
  discipline.bind(kReferenceDomain);
  ScriptedClock reference{{1'000, 1'010}};
  EXPECT_EQ(discipline.prime(constant_reader(100), reference.reader(), 1U), LifecycleStatus::kOk);
}

// REQ: SENSOR-IO-002
TEST(SensorIoClockDiscipline, PrimingMeasuresTheOffsetAndProjectsThroughIt) {
  ClockDiscipline discipline;
  discipline.bind(kReferenceDomain);

  // One round: reference reads 1000 and 1010 around a device read of 100. The bracket midpoint is
  // 1005, so the device trails the reference by 905ns.
  ScriptedClock reference{{1'000, 1'010}};
  ASSERT_EQ(discipline.prime(constant_reader(100), reference.reader(), 1U), LifecycleStatus::kOk);

  EXPECT_TRUE(discipline.settled());
  EXPECT_EQ(discipline.estimate().offset_ns, 905);
  EXPECT_EQ(discipline.estimate().uncertainty_ns, 5);
  EXPECT_EQ(discipline.estimate().samples, 1U);

  const std::optional<TimePoint> projected = discipline.project(200);
  ASSERT_TRUE(projected.has_value());
  EXPECT_EQ(projected->timestamp_ns, 1'105);
  EXPECT_EQ(projected->clock_id, kReferenceDomain);
}

// The tightest bracket is the round least contaminated by scheduling delay, so it must win even
// when a wider round is measured first.
// REQ: SENSOR-IO-002
TEST(SensorIoClockDiscipline, TightestBracketWinsAcrossRounds) {
  ClockDiscipline discipline;
  discipline.bind(kReferenceDomain);

  // Round one brackets 1000..2000 (wide, midpoint 1500). Round two brackets 3000..3004 (tight,
  // midpoint 3002). The device reads 100 then 2100, so only the tight round's offset of 902
  // should survive.
  ScriptedClock reference{{1'000, 2'000, 3'000, 3'004}};
  ScriptedClock device{{100, 2'100}};

  ASSERT_EQ(discipline.prime(device.reader(), reference.reader(), 2U), LifecycleStatus::kOk);

  EXPECT_EQ(discipline.estimate().samples, 2U);
  EXPECT_EQ(discipline.estimate().uncertainty_ns, 2);
  EXPECT_EQ(discipline.estimate().offset_ns, 902);
}

// An odd bracket cannot be halved exactly, and the honest bound is the ceiling. The midpoint
// truncates toward the first reference read, so the second one sits two nanoseconds away from a
// three nanosecond bracket; reporting one would let a consumer order two capture times that this
// measurement cannot actually order.
// REQ: SENSOR-IO-002
TEST(SensorIoClockDiscipline, OddBracketReportsTheCeilingOfItsHalfWidth) {
  ClockDiscipline discipline;
  discipline.bind(kReferenceDomain);

  ScriptedClock reference{{1'000, 1'003}};
  ASSERT_EQ(discipline.prime(constant_reader(100), reference.reader(), 1U), LifecycleStatus::kOk);

  EXPECT_EQ(discipline.estimate().uncertainty_ns, 2);
  EXPECT_EQ(discipline.estimate().offset_ns, 901);
}

// REQ: SENSOR-IO-002
TEST(SensorIoClockDiscipline, ZeroWidthBracketReportsZeroUncertainty) {
  ClockDiscipline discipline;
  discipline.bind(kReferenceDomain);

  ScriptedClock reference{{4'000, 4'000}};
  ASSERT_EQ(discipline.prime(constant_reader(1'000), reference.reader(), 1U), LifecycleStatus::kOk);

  EXPECT_EQ(discipline.estimate().uncertainty_ns, 0);
  EXPECT_EQ(discipline.estimate().offset_ns, 3'000);
}

// The ceiling has to be computed without overflowing, which rules out the `(bracket + 1) / 2` form.
// REQ: SENSOR-IO-002
TEST(SensorIoClockDiscipline, WidestPossibleBracketComputesItsCeilingWithoutOverflow) {
  constexpr std::int64_t kMax = std::numeric_limits<std::int64_t>::max();
  ClockDiscipline discipline;
  discipline.bind(kReferenceDomain);

  ScriptedClock reference{{0, kMax}};
  ASSERT_EQ(
      discipline.prime(
          constant_reader(0), reference.reader(), 1U, ClockDiscipline::kUnboundedUncertaintyNs),
      LifecycleStatus::kOk);

  EXPECT_EQ(discipline.estimate().uncertainty_ns, 4'611'686'018'427'387'904);
  EXPECT_GT(discipline.estimate().uncertainty_ns, kMax / 2);
}

// A bracket the caller has said it cannot justify must not settle the discipline, and must not
// project either. The clocks are not at fault, so the failure is classified retryable.
// REQ: SENSOR-IO-002
TEST(SensorIoClockDiscipline, PrimingIsRetryableWhenEveryBracketExceedsTheAdmittedMaximum) {
  ClockDiscipline discipline;
  discipline.bind(kReferenceDomain);

  ScriptedClock reference{{1'000, 1'100}};
  EXPECT_EQ(
      discipline.prime(constant_reader(100), reference.reader(), 1U, /*max_uncertainty_ns=*/10),
      LifecycleStatus::kRetryableFailure);

  EXPECT_FALSE(discipline.settled());
  EXPECT_FALSE(discipline.project(200).has_value());
  EXPECT_EQ(discipline.estimate().offset_ns, 0);
  EXPECT_EQ(discipline.estimate().uncertainty_ns, 0);
}

// REQ: SENSOR-IO-002
TEST(SensorIoClockDiscipline, PrimingAcceptsABracketExactlyAtTheAdmittedMaximum) {
  ClockDiscipline discipline;
  discipline.bind(kReferenceDomain);

  ScriptedClock reference{{1'000, 1'004}};
  ASSERT_EQ(
      discipline.prime(constant_reader(100), reference.reader(), 1U, /*max_uncertainty_ns=*/2),
      LifecycleStatus::kOk);

  EXPECT_TRUE(discipline.settled());
  EXPECT_EQ(discipline.estimate().uncertainty_ns, 2);
}

// The bound is a rejection, not a poisoning: the binding survives so a later cold call can measure
// again, which is the only retry Core performs today.
// REQ: SENSOR-IO-002
TEST(SensorIoClockDiscipline, ARejectedPrimeLeavesTheBindingOpenToALaterAttempt) {
  ClockDiscipline discipline;
  discipline.bind(kReferenceDomain);

  ScriptedClock wide{{1'000, 1'100}};
  ASSERT_EQ(discipline.prime(constant_reader(100), wide.reader(), 1U, /*max_uncertainty_ns=*/10),
            LifecycleStatus::kRetryableFailure);
  ASSERT_EQ(discipline.domain(), kReferenceDomain);

  ScriptedClock tight{{2'000, 2'004}};
  ASSERT_EQ(discipline.prime(constant_reader(100), tight.reader(), 1U, /*max_uncertainty_ns=*/10),
            LifecycleStatus::kOk);

  EXPECT_TRUE(discipline.settled());
  EXPECT_EQ(discipline.estimate().offset_ns, 1'902);
}

// The default bound exists so an operator that never inspects the estimate still refuses an offset
// no better than the arrival-time stamping this class replaces. The two-argument call also pins the
// defaulted signature that existing consumers rely on.
// REQ: SENSOR-IO-002
TEST(SensorIoClockDiscipline, DefaultAdmittedMaximumRejectsAFourMillisecondBracket) {
  ClockDiscipline discipline;
  discipline.bind(kReferenceDomain);

  // A 4ms bracket reports 2ms of uncertainty, twice the admitted default.
  ScriptedClock reference{{0, 4'000'000}};
  EXPECT_EQ(discipline.prime(constant_reader(0), reference.reader()),
            LifecycleStatus::kRetryableFailure);
  EXPECT_FALSE(discipline.settled());
}

// REQ: SENSOR-IO-002
TEST(SensorIoClockDiscipline, DefaultAdmittedMaximumAcceptsAMicrosecondWideBracket) {
  ClockDiscipline discipline;
  discipline.bind(kReferenceDomain);

  ScriptedClock reference{{0, 1'000}};
  EXPECT_EQ(discipline.prime(constant_reader(0), reference.reader()), LifecycleStatus::kOk);
  EXPECT_EQ(discipline.estimate().uncertainty_ns, 500);
}

// The rounds budget is also the retry budget for scheduling noise: one wide round only has to be
// beaten by a tighter one for the admitted bound to be met.
// REQ: SENSOR-IO-002
TEST(SensorIoClockDiscipline, ATighterLaterRoundSatisfiesABoundTheFirstRoundMissed) {
  ClockDiscipline discipline;
  discipline.bind(kReferenceDomain);

  ScriptedClock reference{{1'000, 1'100, 3'000, 3'004}};
  ScriptedClock device{{100, 2'100}};

  ASSERT_EQ(discipline.prime(device.reader(), reference.reader(), 2U, /*max_uncertainty_ns=*/10),
            LifecycleStatus::kOk);

  EXPECT_EQ(discipline.estimate().uncertainty_ns, 2);
  EXPECT_EQ(discipline.estimate().offset_ns, 902);
}

// REQ: SENSOR-IO-002
TEST(SensorIoClockDiscipline, PrimingFailsWhenNoRoundProducesAUsablePair) {
  ClockDiscipline discipline;
  discipline.bind(kReferenceDomain);

  EXPECT_EQ(discipline.prime(unavailable_reader(), constant_reader(1'000), 4U),
            LifecycleStatus::kFatalFailure);
  EXPECT_FALSE(discipline.settled());
  EXPECT_FALSE(discipline.project(10).has_value());
}

// REQ: SENSOR-IO-002
TEST(SensorIoClockDiscipline, PrimingFailsWhenNoDomainIsBound) {
  ClockDiscipline discipline;

  EXPECT_EQ(discipline.prime(constant_reader(100), constant_reader(1'000), 4U),
            LifecycleStatus::kFatalFailure);
}

// REQ: SENSOR-IO-002
TEST(SensorIoClockDiscipline, PrimingAnAlreadySettledDisciplineIsSkippedNotRepeated) {
  ClockDiscipline discipline;
  discipline.bind_identity(kReferenceDomain);

  EXPECT_EQ(discipline.prime(constant_reader(100), constant_reader(1'000), 4U),
            LifecycleStatus::kSkipped);
  EXPECT_EQ(discipline.estimate().offset_ns, 0);
}

// REQ: SENSOR-IO-002
TEST(SensorIoClockDiscipline, RoundsWithUnusableReadsAreDiscardedWithoutFailingTheRest) {
  ClockDiscipline discipline;
  discipline.bind(kReferenceDomain);

  // The first round's reference reads go backwards, which cannot be a valid bracket. The second
  // round is well-formed and must be the one that settles the discipline.
  ScriptedClock reference{{2'000, 1'000, 5'000, 5'002}};
  ScriptedClock device{{100, 4'000}};

  ASSERT_EQ(discipline.prime(device.reader(), reference.reader(), 2U), LifecycleStatus::kOk);
  EXPECT_EQ(discipline.estimate().samples, 1U);
  EXPECT_EQ(discipline.estimate().offset_ns, 1'001);
}

// REQ: SENSOR-IO-002
TEST(SensorIoClockDiscipline, OverflowingProjectionReportsAbsenceRatherThanWrapping) {
  ClockDiscipline discipline;
  discipline.bind(kReferenceDomain);

  ScriptedClock reference{{1'000, 1'000}};
  ASSERT_EQ(discipline.prime(constant_reader(0), reference.reader(), 1U), LifecycleStatus::kOk);
  ASSERT_EQ(discipline.estimate().offset_ns, 1'000);

  EXPECT_FALSE(discipline.project(std::numeric_limits<std::int64_t>::max()).has_value());
}

// REQ: SENSOR-IO-002
TEST(SensorIoClockDiscipline, UnbindDiscardsTheOffsetSoALaterEpochRemeasures) {
  ClockDiscipline discipline;
  discipline.bind_identity(kReferenceDomain);
  ASSERT_TRUE(discipline.project(5).has_value());

  discipline.unbind();

  EXPECT_FALSE(discipline.settled());
  EXPECT_EQ(discipline.domain(), ClockId{});
  EXPECT_FALSE(discipline.project(5).has_value());
}

// REQ: SENSOR-IO-002
TEST(SensorIoClockDiscipline, RebindingClearsAPreviouslyMeasuredOffset) {
  ClockDiscipline discipline;
  discipline.bind_identity(kReferenceDomain);
  ASSERT_TRUE(discipline.settled());

  discipline.bind(ClockId{9});

  EXPECT_FALSE(discipline.settled());
  EXPECT_EQ(discipline.domain(), ClockId{9});
}

}  // namespace
}  // namespace holoscan::sensor_io

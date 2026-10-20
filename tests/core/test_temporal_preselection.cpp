/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

#include "../common/enum_test_utils.hpp"
#include "runtime/temporal_preselection.hpp"

namespace holoscan::test {
namespace {

using namespace std::chrono_literals;

// REQ: CONTRACT-016
TEST(TemporalPreselection, ReadableIsLevelSensitiveAndRearmsOnlyAfterRetirement) {
  auto state = detail::ReadinessPreselectionState::create(ReadinessSourceKind::kReadable, 1U);
  ASSERT_TRUE(state);

  auto not_ready = state->observe_readable(false);
  ASSERT_TRUE(not_ready);
  EXPECT_EQ(not_ready->disposition, detail::ReadinessWitnessDisposition::kNoWitness);

  auto first = state->observe_readable(true);
  ASSERT_TRUE(first);
  EXPECT_EQ(first->disposition, detail::ReadinessWitnessDisposition::kAccepted);
  EXPECT_EQ(first->source_sequence, 1U);
  ASSERT_TRUE(state->observe_readable(true));
  EXPECT_EQ(state->outstanding(), 1U);

  ASSERT_TRUE(state->retire_one());
  auto rearmed = state->observe_readable(true);
  ASSERT_TRUE(rearmed);
  EXPECT_EQ(rearmed->source_sequence, 2U);
}

// REQ: CONTRACT-016
TEST(TemporalPreselection, NotificationCountsEveryAcceptedPostAndBackpressuresAtCapacity) {
  auto state = detail::ReadinessPreselectionState::create(ReadinessSourceKind::kNotification, 2U);
  ASSERT_TRUE(state);

  auto first = state->post_notification();
  auto second = state->post_notification();
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  EXPECT_EQ(first->source_sequence, 1U);
  EXPECT_EQ(second->source_sequence, 2U);

  auto full = state->post_notification();
  ASSERT_FALSE(full);
  EXPECT_EQ(full.error().code, ErrorCode::kBackpressured);
  EXPECT_EQ(state->last_source_sequence(), 2U);

  ASSERT_TRUE(state->retire_one());
  auto third = state->post_notification();
  ASSERT_TRUE(third);
  EXPECT_EQ(third->source_sequence, 3U);
}

// REQ: CONTRACT-016
TEST(TemporalPreselection, DoorbellCoalescesEqualRejectsLowerAndPreservesRetryOnBackpressure) {
  // Key setup: kDoorbell state with capacity 1; tokens must advance monotonically (no rollback).
  // Key assertion: equal token -> kCoalesced (no new slot consumed); strictly lower ->
  //   kProtocolError; at capacity -> kBackpressured (token unchanged); retire_one() re-arms.
  auto state = detail::ReadinessPreselectionState::create(ReadinessSourceKind::kDoorbell, 1U);
  ASSERT_TRUE(state);

  auto first = state->observe_doorbell(4U);
  ASSERT_TRUE(first);
  EXPECT_EQ(first->source_sequence, 1U);
  EXPECT_EQ(first->skipped_doorbell_tokens, 3U);
  EXPECT_EQ(state->accepted_doorbell_token(), 4U);

  auto equal = state->observe_doorbell(4U);
  ASSERT_TRUE(equal);
  EXPECT_EQ(equal->disposition, detail::ReadinessWitnessDisposition::kCoalesced);

  auto lower = state->observe_doorbell(3U);
  ASSERT_FALSE(lower);
  EXPECT_EQ(lower.error().code, ErrorCode::kProtocolError);

  auto full = state->observe_doorbell(7U);
  ASSERT_FALSE(full);
  EXPECT_EQ(full.error().code, ErrorCode::kBackpressured);
  EXPECT_EQ(state->accepted_doorbell_token(), 4U);

  ASSERT_TRUE(state->retire_one());
  auto retry = state->observe_doorbell(7U);
  ASSERT_TRUE(retry);
  EXPECT_EQ(retry->skipped_doorbell_tokens, 2U);
  EXPECT_EQ(state->accepted_doorbell_token(), 7U);
}

// REQ: CONTRACT-016
TEST(TemporalPreselection, SpinAcceptsOnlyFalseToTrueEdges) {
  auto state = detail::ReadinessPreselectionState::create(ReadinessSourceKind::kSpin, 1U);
  ASSERT_TRUE(state);

  auto first = state->observe_spin(true);
  ASSERT_TRUE(first);
  EXPECT_EQ(first->source_sequence, 1U);

  auto repeated = state->observe_spin(true);
  ASSERT_TRUE(repeated);
  EXPECT_EQ(repeated->disposition, detail::ReadinessWitnessDisposition::kNoWitness);

  ASSERT_TRUE(state->observe_spin(false));
  auto full_edge = state->observe_spin(true);
  ASSERT_FALSE(full_edge);
  EXPECT_EQ(full_edge.error().code, ErrorCode::kBackpressured);

  ASSERT_TRUE(state->retire_one());
  auto still_true = state->observe_spin(true);
  ASSERT_TRUE(still_true);
  EXPECT_EQ(still_true->disposition, detail::ReadinessWitnessDisposition::kNoWitness);
  ASSERT_TRUE(state->observe_spin(false));
  auto next_edge = state->observe_spin(true);
  ASSERT_TRUE(next_edge);
  EXPECT_EQ(next_edge->source_sequence, 2U);
}

// REQ: CONTRACT-016
TEST(TemporalPreselection, ValidatesCapacityKindAndRetirementWithoutMutation) {
  EXPECT_FALSE(detail::ReadinessPreselectionState::create(ReadinessSourceKind::kReadable, 2U));
  EXPECT_FALSE(detail::ReadinessPreselectionState::create(ReadinessSourceKind::kSpin, 0U));
  EXPECT_FALSE(detail::ReadinessPreselectionState::create(
      enum_from_underlying<ReadinessSourceKind>(std::numeric_limits<std::uint8_t>::max()), 1U));

  auto state = detail::ReadinessPreselectionState::create(ReadinessSourceKind::kNotification, 1U);
  ASSERT_TRUE(state);
  auto wrong_kind = state->observe_readable(true);
  ASSERT_FALSE(wrong_kind);
  EXPECT_EQ(wrong_kind.error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(state->outstanding(), 0U);
  EXPECT_FALSE(state->retire_one());
}

// REQ: CONTRACT-004
TEST(TemporalPreselection, ClockConversionUsesOnlyIdenticalOrExplicitDirectDirection) {
  // Key setup: source->target affine edge with scale=3/2 and offset=+10 ns.
  // Key assertion: identical-clock pass-through (ts=7) bypasses the edge; direct conversion of
  //   ts=3 yields 14 ns (4.5 rounds-to-even -> 4, then +10); reverse direction and an undeclared
  //   third clock both return kNotFound.
  constexpr ClockId source{1U};
  constexpr ClockId target{2U};
  constexpr detail::DirectAffineClockEdge direct{
      .source = source,
      .target = target,
      .conversion = {.scale = {.numerator = 3, .denominator = 2}, .offset_ns = 10},
  };

  const auto identical = detail::convert_identical_or_direct(
      TimePoint{.timestamp_ns = 7, .clock_id = source}, source, nullptr);
  ASSERT_TRUE(identical);
  EXPECT_EQ(identical->timestamp_ns, 7);

  auto converted = detail::convert_identical_or_direct(
      TimePoint{.timestamp_ns = 3, .clock_id = source}, target, &direct);
  ASSERT_TRUE(converted);
  EXPECT_EQ(converted->timestamp_ns, 14);  // 4.5 rounds to even, then adds ten.
  EXPECT_EQ(converted->clock_id, target);

  auto reverse = detail::convert_identical_or_direct(
      TimePoint{.timestamp_ns = 14, .clock_id = target}, source, &direct);
  ASSERT_FALSE(reverse);
  EXPECT_EQ(reverse.error().code, ErrorCode::kNotFound);

  auto missing = detail::convert_identical_or_direct(
      TimePoint{.timestamp_ns = 3, .clock_id = source}, ClockId{3U}, &direct);
  ASSERT_FALSE(missing);
  EXPECT_EQ(missing.error().code, ErrorCode::kNotFound);
}

// REQ: CONTRACT-004
TEST(TemporalPreselection, ClockConversionRoundsTiesToEvenAndChecksOverflow) {
  // Key setup: scale=1/2 for rounding cases; scale=2/1 for overflow; and 2/1 + INT64_MIN offset
  //   for the cancellation case. Signed inputs exercise both positive and negative rounding.
  // Key assertion: ties-to-even for positive and negative halves; offset folded into the dividend
  //   before rounding; 2*INT64_MAX overflows; 2*INT64_MAX + INT64_MIN does not (cancels cleanly).
  constexpr ClockId source{1U};
  constexpr ClockId target{2U};
  constexpr detail::DirectAffineClockEdge half{
      .source = source,
      .target = target,
      .conversion = {.scale = {.numerator = 1, .denominator = 2}},
  };
  EXPECT_EQ(detail::convert_identical_or_direct({1, source}, target, &half)->timestamp_ns, 0);
  EXPECT_EQ(detail::convert_identical_or_direct({3, source}, target, &half)->timestamp_ns, 2);
  EXPECT_EQ(detail::convert_identical_or_direct({-1, source}, target, &half)->timestamp_ns, 0);
  EXPECT_EQ(detail::convert_identical_or_direct({-3, source}, target, &half)->timestamp_ns, -2);

  constexpr detail::DirectAffineClockEdge half_plus_one{
      .source = source,
      .target = target,
      .conversion = {.scale = {.numerator = 1, .denominator = 2}, .offset_ns = 1},
  };
  EXPECT_EQ(detail::convert_identical_or_direct({1, source}, target, &half_plus_one)->timestamp_ns,
            2);  // Fold the integer offset before ties-to-even rounding.

  constexpr detail::DirectAffineClockEdge twice{
      .source = source,
      .target = target,
      .conversion = {.scale = {.numerator = 2, .denominator = 1}},
  };
  auto overflow = detail::convert_identical_or_direct(
      {std::numeric_limits<std::int64_t>::max(), source}, target, &twice);
  ASSERT_FALSE(overflow);
  EXPECT_EQ(overflow.error().code, ErrorCode::kInvalidArgument);

  constexpr AffineClockConversion cancellation_conversion{
      .scale = {.numerator = 2, .denominator = 1},
      .offset_ns = std::numeric_limits<std::int64_t>::min(),
  };
  constexpr detail::DirectAffineClockEdge cancellation{
      .source = source,
      .target = target,
      .conversion = cancellation_conversion,
  };
  auto canceled = detail::convert_identical_or_direct(
      {std::numeric_limits<std::int64_t>::max(), source}, target, &cancellation);
  ASSERT_TRUE(canceled);
  EXPECT_EQ(canceled->timestamp_ns, std::numeric_limits<std::int64_t>::max() - 1);
}

// REQ: CONTRACT-016
TEST(TemporalPreselection, ExactAffineComparisonPreservesSubNanosecondOrderAndEquality) {
  // Key setup: three affine edges (1/3, 1/2, 2/4) applied to timestamp=1; results are stored as
  //   exact rational coordinates rather than rounded nanoseconds.
  // Key assertion: 1/3 < 1/2 (kLess); 1/2 == 2/4 (kEqual, reduced fractions match); negative
  //   timestamps (-3*1/2 = -3/2, -4*1/3 = -4/3) preserve sign-aware order (-3/2 < -4/3).
  constexpr ClockId source{1U};
  constexpr ClockId target{2U};
  constexpr detail::DirectAffineClockEdge third{
      .source = source,
      .target = target,
      .conversion = {.scale = {.numerator = 1, .denominator = 3}},
  };
  constexpr detail::DirectAffineClockEdge half{
      .source = source,
      .target = target,
      .conversion = {.scale = {.numerator = 1, .denominator = 2}},
  };
  constexpr detail::DirectAffineClockEdge two_quarters{
      .source = source,
      .target = target,
      .conversion = {.scale = {.numerator = 2, .denominator = 4}},
  };

  auto one_third = detail::convert_identical_or_direct_exact({1, source}, target, &third);
  auto one_half = detail::convert_identical_or_direct_exact({1, source}, target, &half);
  auto equal_half = detail::convert_identical_or_direct_exact({1, source}, target, &two_quarters);
  ASSERT_TRUE(one_third);
  ASSERT_TRUE(one_half);
  ASSERT_TRUE(equal_half);
  EXPECT_EQ(detail::materialize_exact_coordinate(*one_third)->timestamp_ns, 0);
  EXPECT_EQ(detail::materialize_exact_coordinate(*one_half)->timestamp_ns, 0);
  EXPECT_EQ(*detail::compare_exact_coordinates(*one_third, *one_half),
            detail::ExactCoordinateOrder::kLess);
  EXPECT_EQ(*detail::compare_exact_coordinates(*one_half, *equal_half),
            detail::ExactCoordinateOrder::kEqual);

  auto negative_half = detail::convert_identical_or_direct_exact({-3, source}, target, &half);
  auto negative_third = detail::convert_identical_or_direct_exact({-4, source}, target, &third);
  ASSERT_TRUE(negative_half);
  ASSERT_TRUE(negative_third);
  EXPECT_EQ(*detail::compare_exact_coordinates(*negative_half, *negative_third),
            detail::ExactCoordinateOrder::kLess);
}

// REQ: CONTRACT-016
TEST(TemporalPreselection, ExactAffineWindowChecksBoundaryBeforeMaterialization) {
  // Key setup: low = 1*(1/3) ns, exact_boundary = 4*(1/3) ns (= low+1 ns in exact arithmetic),
  //   epsilon_over = 3*(1/2) ns (strictly above the 1 ns boundary).
  // Key assertion: [low, exact_boundary] fits the inclusive 1 ns window; [low, epsilon_over] does
  //   not; a coordinate with a mismatched target clock fails compare_exact_coordinates.
  constexpr ClockId source{1U};
  constexpr ClockId target{2U};
  constexpr detail::DirectAffineClockEdge third{
      .source = source,
      .target = target,
      .conversion = {.scale = {.numerator = 1, .denominator = 3}},
  };
  constexpr detail::DirectAffineClockEdge half{
      .source = source,
      .target = target,
      .conversion = {.scale = {.numerator = 1, .denominator = 2}},
  };

  auto low = detail::convert_identical_or_direct_exact({1, source}, target, &third);
  auto exact_boundary = detail::convert_identical_or_direct_exact({4, source}, target, &third);
  auto epsilon_over = detail::convert_identical_or_direct_exact({3, source}, target, &half);
  ASSERT_TRUE(low);
  ASSERT_TRUE(exact_boundary);
  ASSERT_TRUE(epsilon_over);
  EXPECT_EQ(*detail::exact_interval_within_inclusive_window(*low, *exact_boundary, 1ns), true);
  EXPECT_EQ(*detail::exact_interval_within_inclusive_window(*low, *epsilon_over, 1ns), false);

  auto wrong_domain = *epsilon_over;
  wrong_domain.target_clock = ClockId{3U};
  EXPECT_FALSE(detail::compare_exact_coordinates(*low, wrong_domain));
}

// REQ: CONTRACT-016
TEST(TemporalPreselection, ExactAffineCoordinateSupportsExtremaWithoutInventingPaths) {
  // Key setup: scale=2/1 applied to INT64_MAX yields an exact coordinate whose materialized
  //   result overflows int64_t.
  // Key assertion: materialize_exact_coordinate() fails for the overflowed value; reverse and
  //   undeclared-path conversions both return kNotFound without inventing intermediate nodes.
  constexpr ClockId source{1U};
  constexpr ClockId target{2U};
  constexpr detail::DirectAffineClockEdge twice{
      .source = source,
      .target = target,
      .conversion = {.scale = {.numerator = 2, .denominator = 1}},
  };

  auto exact = detail::convert_identical_or_direct_exact(
      {std::numeric_limits<std::int64_t>::max(), source}, target, &twice);
  ASSERT_TRUE(exact);
  EXPECT_FALSE(detail::materialize_exact_coordinate(*exact));

  auto reverse = detail::convert_identical_or_direct_exact({1, target}, source, &twice);
  ASSERT_FALSE(reverse);
  EXPECT_EQ(reverse.error().code, ErrorCode::kNotFound);
  auto unrecorded_path =
      detail::convert_identical_or_direct_exact({1, source}, ClockId{3U}, &twice);
  ASSERT_FALSE(unrecorded_path);
  EXPECT_EQ(unrecorded_path.error().code, ErrorCode::kNotFound);
}

// REQ: CONTRACT-016
TEST(TemporalPreselection, HealthPolicyCannotCreateEligibilityOrMutateAContract) {
  // Key setup: each EndpointHealthAction tested against is_eligible=false (loop), then each
  //   action paired with is_eligible=true and a matching health state.
  // Key assertion: non-eligible always yields kNotEligible; eligible maps kAdmit->kAdmitted,
  //   kDegrade->kDegraded, kReject->kRejected; invalid action returns kInvalidArgument.
  using detail::EndpointHealthAction;
  using detail::EndpointHealthDisposition;
  using detail::EndpointHealthState;
  using detail::HealthPreselectionResult;

  for (const auto action : {EndpointHealthAction::kAdmit,
                            EndpointHealthAction::kDegrade,
                            EndpointHealthAction::kReject}) {
    auto not_eligible = detail::apply_endpoint_health(
        false, EndpointHealthDisposition{EndpointHealthState::kFreshUnhealthy, action});
    ASSERT_TRUE(not_eligible);
    EXPECT_EQ(*not_eligible, HealthPreselectionResult::kNotEligible);
  }

  EXPECT_EQ(*detail::apply_endpoint_health(
                true, {EndpointHealthState::kFreshHealthy, EndpointHealthAction::kAdmit}),
            HealthPreselectionResult::kAdmitted);
  EXPECT_EQ(*detail::apply_endpoint_health(
                true, {EndpointHealthState::kStale, EndpointHealthAction::kDegrade}),
            HealthPreselectionResult::kDegraded);
  EXPECT_EQ(*detail::apply_endpoint_health(
                true, {EndpointHealthState::kFreshUnhealthy, EndpointHealthAction::kReject}),
            HealthPreselectionResult::kRejected);

  auto invalid = detail::apply_endpoint_health(
      false,
      {EndpointHealthState::kFreshHealthy,
       enum_from_underlying<EndpointHealthAction>(std::numeric_limits<std::uint8_t>::max())});
  ASSERT_FALSE(invalid);
  EXPECT_EQ(invalid.error().code, ErrorCode::kInvalidArgument);
}

}  // namespace
}  // namespace holoscan::test

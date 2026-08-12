/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <holoscan/core/expected.hpp>

#include "../common/allocation_counter.hpp"
#include "../common/enum_test_utils.hpp"
#include "runtime/temporal_input_cache.hpp"

namespace holoscan::test {
namespace {

using namespace std::chrono_literals;
constexpr ClockId kClock{7};

// Convenience factory for TemporalSampleRecord using integer nanosecond timestamps and
// the shared kClock.  Tests that need a different clock domain must set
// capture_time.clock_id on the returned value before use.
[[nodiscard]] constexpr detail::TemporalSampleRecord sample(std::int64_t timestamp,
                                                            std::uint64_t source_sequence,
                                                            std::uint64_t arrival,
                                                            std::uint64_t epoch = 1) noexcept {
  return {.capture_time = TimePoint{timestamp, kClock},
          .source_sequence = source_sequence,
          .producer_epoch = epoch,
          .arrival_sequence = arrival};
}

// REQ: CONTRACT-004
TEST(TemporalInputCache, LatestWithinIsCausalOrderIndependentAndInclusive) {
  detail::TemporalInputCache cache{8};
  // Key setup: samples pushed in arrival order {90,110,80,90}; two share t=90 with
  // different source_sequence values to exercise tie-breaking by highest sequence.
  ASSERT_TRUE(cache.push(sample(90, 1, 1)));
  ASSERT_TRUE(cache.push(sample(110, 2, 2)));
  ASSERT_TRUE(cache.push(sample(80, 3, 3)));
  ASSERT_TRUE(cache.push(sample(90, 4, 4)));

  {
    auto selected = cache.select_latest_within_transaction(TimePoint{100, kClock}, 10ns);
    ASSERT_TRUE(selected);
    ASSERT_TRUE(selected->result().sample);
    const auto selected_sample = selected->result().sample.value_or(detail::TemporalSampleRecord{});
    EXPECT_EQ(selected_sample.capture_time.timestamp_ns, 90);
    EXPECT_EQ(selected_sample.source_sequence, 4U);
  }

  // Key assertion: window narrowed to 9ns excludes both t=90 samples (age=10 > 9).
  auto boundary = cache.select_latest_within_transaction(TimePoint{100, kClock}, 9ns);
  ASSERT_TRUE(boundary);
  EXPECT_EQ(boundary->result().selection.reason, SelectionReason::kTooStale);
  EXPECT_FALSE(boundary->result().sample);
}

// REQ: CONTRACT-004
TEST(TemporalInputCache, OptionalCaptureTimeNeverBecomesNumericZero) {
  detail::TemporalInputCache cache{3};
  ASSERT_TRUE(cache.push(detail::TemporalSampleRecord{
      .source_sequence = 1, .producer_epoch = 1, .arrival_sequence = 1}));
  auto selected = cache.select_exact_transaction(TimePoint{0, kClock});
  ASSERT_TRUE(selected);
  EXPECT_EQ(selected->result().selection.reason, SelectionReason::kCaptureTimeAbsent);
  EXPECT_FALSE(selected->result().sample);
}

// REQ: CONTRACT-005
TEST(TemporalInputCache, DrainAllCommitsOnceOnlyAfterWholeActivationAcceptance) {
  detail::TemporalInputCache cache{4};
  // Key setup: snapshot_drain_all() captures a watermarked prefix. A cancelled snapshot
  // (scope exit without commit) leaves the cache unchanged; only a committed snapshot
  // drains. A second commit() on the same snapshot must fail (one-shot semantics).
  ASSERT_TRUE(cache.push(sample(1, 1, 1)));
  ASSERT_TRUE(cache.push(sample(2, 2, 2)));

  {
    auto provisional = cache.snapshot_drain_all(2, 4);
    ASSERT_TRUE(provisional);
    EXPECT_EQ(provisional->samples().size(), 2U);
  }
  EXPECT_EQ(cache.size(), 2U);

  auto accepted = cache.snapshot_drain_all(1, 4);
  ASSERT_TRUE(accepted);
  ASSERT_TRUE(cache.push(sample(3, 3, 3)));
  ASSERT_TRUE(accepted->commit());
  ASSERT_EQ(cache.size(), 2U);
  EXPECT_EQ(cache.records()[0].source_sequence, 2U);
  EXPECT_EQ(cache.records()[1].source_sequence, 3U);
  // Key assertion: commit() is one-shot — a second call on the same snapshot returns false.
  auto second_commit = accepted->commit();
  EXPECT_FALSE(second_commit);
}

// REQ: CONTRACT-005
TEST(TemporalInputCache, DrainAllCommitRetiresFrozenPrefixAndPreservesLaterArrivals) {
  detail::TemporalInputCache cache{8};
  for (std::uint64_t sequence = 1; sequence <= 6; ++sequence) {
    ASSERT_TRUE(cache.push(sample(static_cast<std::int64_t>(sequence), sequence, sequence)));
  }

  // Key setup: record 7 is pushed after the snapshot is taken but before commit().
  // commit() retires only the frozen prefix {1..4}; arrivals {5,6,7} survive.
  auto accepted = cache.snapshot_drain_all(4, 8);
  ASSERT_TRUE(accepted);
  ASSERT_EQ(accepted->samples().size(), 4U);
  ASSERT_TRUE(cache.push(sample(7, 7, 7)));

  ASSERT_TRUE(accepted->commit());
  ASSERT_EQ(cache.size(), 3U);
  EXPECT_EQ(cache.records()[0].arrival_sequence, 5U);
  EXPECT_EQ(cache.records()[1].arrival_sequence, 6U);
  EXPECT_EQ(cache.records()[2].arrival_sequence, 7U);
}

// REQ: CONTRACT-004
TEST(TemporalInputCache, ScalarSelectionConsumesOnlyAfterAcceptedCommit) {
  detail::TemporalInputCache cache{4};
  ASSERT_TRUE(cache.push(sample(10, 1, 1)));
  // Key setup: a cancelled selection (scope exit without commit) leaves the cache intact;
  // the subsequently committed selection actually removes the sample.
  {
    auto cancelled = cache.select_exact_transaction(TimePoint{10, kClock});
    ASSERT_TRUE(cancelled);
    ASSERT_TRUE(cancelled->result().sample);
  }
  EXPECT_EQ(cache.size(), 1U);

  auto accepted = cache.select_exact_transaction(TimePoint{10, kClock});
  ASSERT_TRUE(accepted);
  ASSERT_TRUE(accepted->commit());
  EXPECT_EQ(cache.size(), 0U);
}

// REQ: CONTRACT-004
// REQ: CONTRACT-005
TEST(TemporalInputCache, SeededModelPreservesOrderAcrossTransactionsAndOverflow) {
  // Key setup: boundary timestamps (INT64_MIN×2, -1, 0×2, 1, INT64_MAX-1, INT64_MAX)
  // stress overflow and equal-time tie-breaking before the 512-step random walk begins.
  // A parallel model vector mirrors the expected cache state for per-step verification.
  constexpr std::array<std::int64_t, 8> kTimestamps{
      std::numeric_limits<std::int64_t>::min(),
      std::numeric_limits<std::int64_t>::min(),
      -1,
      0,
      0,
      1,
      std::numeric_limits<std::int64_t>::max() - 1,
      std::numeric_limits<std::int64_t>::max(),
  };
  detail::TemporalInputCache cache{8};
  std::vector<detail::TemporalSampleRecord> model;
  std::uint64_t next_identity = 1;
  for (const auto timestamp : kTimestamps) {
    const auto record = sample(timestamp, next_identity, next_identity);
    ASSERT_TRUE(cache.push(record));
    model.push_back(record);
    ++next_identity;
  }

  auto overflow = cache.push(sample(0, next_identity, next_identity));
  ASSERT_FALSE(overflow);
  EXPECT_EQ(overflow.error().code, ErrorCode::kResourceExhausted);
  ++next_identity;

  {
    auto equal_time_tie =
        cache.select_exact_transaction(TimePoint{std::numeric_limits<std::int64_t>::min(), kClock});
    ASSERT_TRUE(equal_time_tie);
    ASSERT_TRUE(equal_time_tie->result().sample);
    EXPECT_EQ(equal_time_tie->result().sample->source_sequence, 2U);
  }
  {
    detail::TemporalInputCache extrema{1};
    ASSERT_TRUE(extrema.push(sample(std::numeric_limits<std::int64_t>::min(), 1, 1)));
    auto selected = extrema.select_latest_within_transaction(
        TimePoint{std::numeric_limits<std::int64_t>::max(), kClock},
        std::chrono::nanoseconds::max());
    ASSERT_TRUE(selected);
    EXPECT_EQ(selected->result().selection.reason, SelectionReason::kTooStale);
    EXPECT_FALSE(selected->result().sample);
  }

  std::uint64_t random_state = 0x7a5b'3c19'd204'8ef1ULL;
  const auto next_random = [&random_state]() noexcept {
    random_state = random_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return random_state >> 32U;
  };
  const auto expect_cache_matches_model = [&]() {
    const auto records = cache.records();
    ASSERT_EQ(records.size(), model.size());
    EXPECT_TRUE(std::equal(records.begin(), records.end(), model.begin(), model.end()));
  };
  const auto erase_selected = [&](const detail::TemporalSampleRecord& selected) {
    const auto found = std::find_if(model.begin(), model.end(), [&](const auto& candidate) {
      return candidate.arrival_sequence == selected.arrival_sequence;
    });
    ASSERT_NE(found, model.end());
    model.erase(found);
  };

  std::size_t scalar_commits = 0;
  std::size_t scalar_cancels = 0;
  std::size_t drain_commits = 0;
  std::size_t drain_cancels = 0;
  std::size_t overflows = 1;
  for (std::size_t step = 0; step < 512; ++step) {
    const auto operation = next_random() % 4U;
    if (operation == 0U) {
      const auto timestamp = kTimestamps[next_random() % kTimestamps.size()];
      const auto record = sample(timestamp, next_identity, next_identity);
      auto pushed = cache.push(record);
      if (model.size() == cache.capacity()) {
        ASSERT_FALSE(pushed);
        EXPECT_EQ(pushed.error().code, ErrorCode::kResourceExhausted);
        ++overflows;
      } else {
        ASSERT_TRUE(pushed);
        model.push_back(record);
      }
      ++next_identity;
    } else if (operation == 1U) {
      const auto reference = kTimestamps[next_random() % kTimestamps.size()];
      std::optional<detail::TemporalSampleRecord> expected;
      for (const auto& candidate : model) {
        if (candidate.capture_time.timestamp_ns == reference &&
            (!expected || candidate.source_sequence > expected->source_sequence)) {
          expected = candidate;
        }
      }
      const bool commit = (next_random() & 1U) != 0U;
      auto selected = cache.select_exact_transaction(TimePoint{reference, kClock});
      ASSERT_TRUE(selected);
      EXPECT_EQ(selected->result().sample, expected);
      EXPECT_EQ(selected->result().selection.state,
                expected ? InputState::kSelected : InputState::kAbsent);
      if (commit) {
        ASSERT_TRUE(selected->commit());
        if (expected) {
          erase_selected(*expected);
        }
        ++scalar_commits;
      } else {
        ++scalar_cancels;
      }
    } else if (operation == 2U) {
      const auto reference = kTimestamps[next_random() % kTimestamps.size()];
      const auto max_age =
          std::chrono::nanoseconds{1 + static_cast<std::int64_t>(next_random() % 64)};
      std::optional<detail::TemporalSampleRecord> expected;
      for (const auto& candidate : model) {
        const auto timestamp = candidate.capture_time.timestamp_ns;
        if (timestamp <= reference &&
            (!expected || timestamp > expected->capture_time.timestamp_ns ||
             (timestamp == expected->capture_time.timestamp_ns &&
              candidate.source_sequence > expected->source_sequence))) {
          expected = candidate;
        }
      }
      SelectionReason reason = SelectionReason::kNeverSeen;
      if (expected) {
        const auto age = static_cast<std::uint64_t>(reference) -
                         static_cast<std::uint64_t>(expected->capture_time.timestamp_ns);
        if (age > static_cast<std::uint64_t>(max_age.count())) {
          expected.reset();
          reason = SelectionReason::kTooStale;
        } else {
          reason = SelectionReason::kNone;
        }
      }
      const bool commit = (next_random() & 1U) != 0U;
      auto selected = cache.select_latest_within_transaction(TimePoint{reference, kClock}, max_age);
      ASSERT_TRUE(selected);
      EXPECT_EQ(selected->result().sample, expected);
      EXPECT_EQ(selected->result().selection.reason, reason);
      if (commit) {
        ASSERT_TRUE(selected->commit());
        if (expected) {
          erase_selected(*expected);
        }
        ++scalar_commits;
      } else {
        ++scalar_cancels;
      }
    } else {
      const auto watermark = next_random() % (next_identity + 1U);
      std::size_t frozen_count = 0;
      while (frozen_count < model.size() && model[frozen_count].arrival_sequence <= watermark) {
        ++frozen_count;
      }
      const bool commit = (next_random() & 1U) != 0U;
      auto drained = cache.snapshot_drain_all(watermark, cache.capacity());
      ASSERT_TRUE(drained);
      ASSERT_EQ(drained->samples().size(), frozen_count);
      EXPECT_TRUE(std::equal(drained->samples().begin(),
                             drained->samples().end(),
                             model.begin(),
                             model.begin() + frozen_count));
      if (commit) {
        ASSERT_TRUE(drained->commit());
        model.erase(model.begin(), model.begin() + static_cast<std::ptrdiff_t>(frozen_count));
        ++drain_commits;
      } else {
        ++drain_cancels;
      }
    }
    expect_cache_matches_model();
  }

  // Key assertion: all four operation types must appear in both commit and cancel variants,
  // confirming the LCG seed provides adequate coverage of the operation space.
  EXPECT_GT(scalar_commits, 0U);
  EXPECT_GT(scalar_cancels, 0U);
  EXPECT_GT(drain_commits, 0U);
  EXPECT_GT(drain_cancels, 0U);
  EXPECT_GT(overflows, 1U);
}

// REQ: CONTRACT-003
TEST(TemporalInputCache, SafeDefaultDominatesAbsentAfterCompleteEvaluation) {
  const std::array selections{
      InputSelection{InputState::kAbsent, SelectionReason::kNeverSeen},
      InputSelection{InputState::kSelected, SelectionReason::kNone},
      InputSelection{InputState::kSafeDefault, SelectionReason::kTooStale},
  };
  EXPECT_EQ(detail::aggregate_missing_inputs(selections),
            detail::MissingAggregateDisposition::kDegraded);
}

// REQ: CONTRACT-003
TEST(TemporalInputCache, ScalarReuseRequiresCurrentProducerEpochAndInclusiveAge) {
  const detail::ScalarSelectionResult missing{
      .selection = {InputState::kAbsent, SelectionReason::kNeverSeen}};
  const auto retained = sample(10, 1, 1, 2);

  auto boundary = detail::apply_scalar_missing_policy(
      missing, ReuseLast{.max_age = 5ns}, retained, 2, TimePoint{15, kClock});
  ASSERT_TRUE(boundary.sample);
  EXPECT_EQ(boundary.selection.state, InputState::kReused);
  EXPECT_EQ(boundary.sample ? boundary.sample->producer_epoch : 0U, 2U);

  auto expired = detail::apply_scalar_missing_policy(
      missing, ReuseLast{.max_age = 5ns}, retained, 2, TimePoint{16, kClock});
  EXPECT_EQ(expired.selection.state, InputState::kAbsent);
  EXPECT_FALSE(expired.sample);

  auto old_epoch = detail::apply_scalar_missing_policy(
      missing, ReuseLast{.max_age = 5ns}, retained, 3, TimePoint{15, kClock});
  EXPECT_EQ(old_epoch.selection.state, InputState::kAbsent);
  EXPECT_FALSE(old_epoch.sample);

  auto wrong_clock = detail::apply_scalar_missing_policy(
      missing, ReuseLast{.max_age = 5ns}, retained, 2, TimePoint{15, ClockId{8}});
  EXPECT_EQ(wrong_clock.selection.state, InputState::kAbsent);
  EXPECT_FALSE(wrong_clock.sample);
}

// REQ: CONTRACT-006
TEST(TemporalInputCache, AlignedJoinEnforcesAuthoredCapacityIndependentOfVectorGrowth) {
  detail::AlignedJoinEngine engine{1, 1};
  ASSERT_TRUE(engine.push(0, sample(1, 1, 1)));

  auto full = engine.push(0, sample(2, 2, 2));
  ASSERT_FALSE(full);
  EXPECT_EQ(full.error().code, ErrorCode::kResourceExhausted);
  EXPECT_EQ(engine.queue_size(0), 1U);
}

// REQ: CONTRACT-006
// Queue state before first select:
//   input[0]: t=0(seq=1) --- t=10(seq=2)
//   input[1]:      t=5(seq=1) --------- t=11(seq=2)
// select(2ns): head of input[0] at t=0 misaligned vs input[1]'s t=5 (spread 5 > 2).
// select(5ns): after retiring t=0, heads are t=10 and t=5; spread 5 ≤ 5 → kNominal.
TEST(TemporalInputCache, AlignedJoinRetiresAllMinimumHeadsBeforeExactMatch) {
  detail::AlignedJoinEngine engine{2, 4};
  ASSERT_TRUE(engine.push(0, sample(0, 1, 1)));
  ASSERT_TRUE(engine.push(0, sample(10, 2, 2)));
  ASSERT_TRUE(engine.push(1, sample(5, 1, 3)));
  ASSERT_TRUE(engine.push(1, sample(11, 2, 4)));
  const std::array policies{detail::AlignedMissingPolicy{}, detail::AlignedMissingPolicy{}};
  const std::array closure{detail::AlignmentClosureProof::kOpen,
                           detail::AlignmentClosureProof::kOpen};

  {
    auto provisional = engine.select(2ns, policies, closure);
    ASSERT_TRUE(provisional);
    EXPECT_EQ(provisional->disposition(), detail::AlignedCandidateDisposition::kRetireMisaligned);
    EXPECT_EQ(engine.queue_size(0), 2U);
  }
  auto retired = engine.select(2ns, policies, closure);
  ASSERT_TRUE(retired);
  ASSERT_TRUE(retired->commit());
  EXPECT_EQ(engine.queue_size(0), 1U);

  auto match = engine.select(5ns, policies, closure);
  ASSERT_TRUE(match);
  EXPECT_EQ(match->disposition(), detail::AlignedCandidateDisposition::kNominal);
  EXPECT_EQ(match->reference_time(), (TimePoint{10, kClock}));
  ASSERT_EQ(match->members().size(), 2U);
  ASSERT_TRUE(match->commit());
  EXPECT_EQ(engine.queue_size(0), 0U);
  EXPECT_EQ(engine.queue_size(1), 1U);
}

// REQ: CONTRACT-006
// Inputs at INT64_MIN and 0 produce a timestamp spread that overflows uint64.
// The engine must return kInvalidArgument without mutating either queue.
TEST(TemporalInputCache, AlignedRejectsUnrepresentableTimestampSpreadWithoutRetirement) {
  const std::array policies{detail::AlignedMissingPolicy{}, detail::AlignedMissingPolicy{}};
  const std::array open{detail::AlignmentClosureProof::kOpen, detail::AlignmentClosureProof::kOpen};

  detail::AlignedJoinEngine complete{2, 1};
  ASSERT_TRUE(complete.push(0, sample(std::numeric_limits<std::int64_t>::min(), 1, 1)));
  ASSERT_TRUE(complete.push(1, sample(0, 1, 1)));
  auto strict = complete.select(std::chrono::nanoseconds::max(), policies, open);
  ASSERT_FALSE(strict);
  EXPECT_EQ(strict.error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(complete.queue_size(0), 1U);
  EXPECT_EQ(complete.queue_size(1), 1U);

  detail::AlignedJoinEngine partial{2, 1};
  ASSERT_TRUE(partial.push(0, sample(std::numeric_limits<std::int64_t>::min(), 1, 1)));
  const std::array partial_policies{
      detail::AlignedMissingPolicy{},
      detail::AlignedMissingPolicy{.kind = detail::AlignedMissingPolicyKind::kSafeDefault},
  };
  const std::array closed{detail::AlignmentClosureProof::kOpen,
                          detail::AlignmentClosureProof::kEndOfStream};
  auto degraded = partial.select(
      std::chrono::nanoseconds::max(), partial_policies, closed, TimePoint{0, kClock});
  ASSERT_FALSE(degraded);
  EXPECT_EQ(degraded.error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(partial.queue_size(0), 1U);
}

// REQ: CONTRACT-006
TEST(TemporalInputCache, AlignedPartialRequiresProofAndNoFireConsumesNothing) {
  detail::AlignedJoinEngine engine{2, 2};
  ASSERT_TRUE(engine.push(0, sample(10, 1, 1)));
  const std::array policies{
      detail::AlignedMissingPolicy{},
      detail::AlignedMissingPolicy{.kind = detail::AlignedMissingPolicyKind::kSkip},
  };
  const std::array open{detail::AlignmentClosureProof::kOpen, detail::AlignmentClosureProof::kOpen};
  auto pending = engine.select(1ns, policies, open);
  ASSERT_TRUE(pending);
  EXPECT_EQ(pending->disposition(), detail::AlignedCandidateDisposition::kPending);

  const std::array closed{detail::AlignmentClosureProof::kOpen,
                          detail::AlignmentClosureProof::kEndOfStream};
  auto no_fire = engine.select(1ns, policies, closed, TimePoint{12, kClock});
  ASSERT_TRUE(no_fire);
  EXPECT_EQ(no_fire->disposition(), detail::AlignedCandidateDisposition::kNoFire);
  ASSERT_EQ(no_fire->closure_evidence().size(), 2U);
  EXPECT_EQ(no_fire->closure_evidence()[1], detail::AlignmentClosureProof::kEndOfStream);
  EXPECT_EQ(no_fire->selections()[1].reason, SelectionReason::kEndOfStream);
  ASSERT_TRUE(no_fire->commit());
  EXPECT_EQ(engine.queue_size(0), 1U);
}

// REQ: CONTRACT-006
TEST(TemporalInputCache, AlignedProofClosedUnspecifiedMemberBecomesNoFire) {
  detail::AlignedJoinEngine engine{2, 2};
  ASSERT_TRUE(engine.push(0, sample(10, 1, 1)));
  const std::array policies{detail::AlignedMissingPolicy{}, detail::AlignedMissingPolicy{}};
  const std::array closed{detail::AlignmentClosureProof::kOpen,
                          detail::AlignmentClosureProof::kEndOfStream};

  auto no_fire = engine.select(1ns, policies, closed, TimePoint{12, kClock});
  ASSERT_TRUE(no_fire);
  EXPECT_EQ(no_fire->disposition(), detail::AlignedCandidateDisposition::kNoFire);
  ASSERT_EQ(no_fire->closure_evidence().size(), 2U);
  EXPECT_EQ(no_fire->closure_evidence()[1], detail::AlignmentClosureProof::kEndOfStream);
  ASSERT_EQ(no_fire->selections().size(), 2U);
  EXPECT_EQ(no_fire->selections()[1].state, InputState::kAbsent);
  EXPECT_EQ(no_fire->selections()[1].reason, SelectionReason::kEndOfStream);
  EXPECT_TRUE(no_fire->members().empty());
  ASSERT_TRUE(no_fire->commit());
  EXPECT_EQ(engine.queue_size(0), 1U);
}

// REQ: CONTRACT-006
TEST(TemporalInputCache, AlignedPartialRejectsAvailableHeadOutsideClosureWindow) {
  detail::AlignedJoinEngine engine{2, 2};
  ASSERT_TRUE(engine.push(0, sample(10, 1, 1)));
  const std::array policies{
      detail::AlignedMissingPolicy{},
      detail::AlignedMissingPolicy{.kind = detail::AlignedMissingPolicyKind::kSafeDefault},
  };
  const std::array closed{detail::AlignmentClosureProof::kOpen,
                          detail::AlignmentClosureProof::kEndOfStream};
  auto misaligned = engine.select(1ns, policies, closed, TimePoint{20, kClock});
  ASSERT_TRUE(misaligned);
  EXPECT_EQ(misaligned->disposition(), detail::AlignedCandidateDisposition::kRetireMisaligned);
  ASSERT_TRUE(misaligned->commit());
  EXPECT_EQ(engine.queue_size(0), 0U);
}

// REQ: CONTRACT-006
// After a committed activation the engine remembers the last delivered (timestamp,
// source_sequence) per member. Pushes that go backwards in time or reuse a consumed
// source_sequence are rejected until transition_member_epoch() advances the epoch.
TEST(TemporalInputCache, AlignedOrderingWitnessSurvivesConsumptionUntilExplicitEpochTransition) {
  detail::AlignedJoinEngine engine{2, 2};
  ASSERT_TRUE(engine.push(0, sample(10, 1, 1)));
  ASSERT_TRUE(engine.push(1, sample(10, 1, 1)));
  const std::array policies{detail::AlignedMissingPolicy{}, detail::AlignedMissingPolicy{}};
  const std::array open{detail::AlignmentClosureProof::kOpen, detail::AlignmentClosureProof::kOpen};

  auto activation = engine.select(0ns, policies, open);
  ASSERT_TRUE(activation);
  ASSERT_TRUE(activation->commit());
  EXPECT_EQ(engine.queue_size(0), 0U);

  auto regression = engine.push(0, sample(9, 2, 2));
  ASSERT_FALSE(regression);
  EXPECT_EQ(regression.error().code, ErrorCode::kInvalidArgument);

  auto source_regression = engine.push(0, sample(11, 1, 2));
  ASSERT_FALSE(source_regression);
  EXPECT_EQ(source_regression.error().code, ErrorCode::kInvalidArgument);

  auto spontaneous_epoch = engine.push(0, sample(20, 1, 2, 2));
  ASSERT_FALSE(spontaneous_epoch);
  EXPECT_EQ(spontaneous_epoch.error().code, ErrorCode::kProtocolError);

  ASSERT_TRUE(engine.transition_member_epoch(
      0, 2, detail::AlignedMemberEpochTransitionReason::kProducerEpoch));
  ASSERT_TRUE(engine.push(0, sample(20, 1, 2, 2)));
}

// REQ: CONTRACT-006
// transition_member_epoch() is blocked while a select() transaction is outstanding
// (kBackpressured). After commit, the epoch advances and the reuse-last fallback
// buffer is cleared so the pre-transition sample cannot be delivered as kReused.
TEST(TemporalInputCache, AlignedEpochTransitionRejectsActiveTransactionAndClearsFallback) {
  detail::AlignedJoinEngine engine{2, 2};
  ASSERT_TRUE(engine.push(0, sample(10, 1, 1)));
  ASSERT_TRUE(engine.push(1, sample(10, 1, 1)));
  const std::array policies{detail::AlignedMissingPolicy{}, detail::AlignedMissingPolicy{}};
  const std::array open{detail::AlignmentClosureProof::kOpen, detail::AlignmentClosureProof::kOpen};

  auto active = engine.select(0ns, policies, open);
  ASSERT_TRUE(active);
  auto while_active = engine.transition_member_epoch(
      1, 2, detail::AlignedMemberEpochTransitionReason::kProducerEpoch);
  ASSERT_FALSE(while_active);
  EXPECT_EQ(while_active.error().code, ErrorCode::kBackpressured);
  ASSERT_TRUE(active->commit());

  ASSERT_TRUE(engine.transition_member_epoch(
      1, 2, detail::AlignedMemberEpochTransitionReason::kProducerEpoch));
  ASSERT_TRUE(engine.push(0, sample(20, 2, 2)));
  const std::array reuse{
      detail::AlignedMissingPolicy{},
      detail::AlignedMissingPolicy{.kind = detail::AlignedMissingPolicyKind::kReuseLast,
                                   .reuse_max_age = 100ns},
  };
  const std::array closed{detail::AlignmentClosureProof::kOpen,
                          detail::AlignmentClosureProof::kProducerEpochClosed};
  auto no_old_epoch = engine.select(100ns, reuse, closed, TimePoint{20, kClock});
  ASSERT_TRUE(no_old_epoch);
  EXPECT_EQ(no_old_epoch->disposition(), detail::AlignedCandidateDisposition::kNoFire);
  EXPECT_EQ(no_old_epoch->selections()[1].state, InputState::kAbsent);
  EXPECT_TRUE(no_old_epoch->members().empty());
}

// REQ: CONTRACT-006
TEST(TemporalInputCache, AlignedRejectsUnknownClosedValuesAndInvalidReuseShapeWithoutMutation) {
  detail::AlignedJoinEngine engine{2, 2};
  ASSERT_TRUE(engine.push(0, sample(10, 1, 1)));
  const std::array valid_policies{
      detail::AlignedMissingPolicy{},
      detail::AlignedMissingPolicy{.kind = detail::AlignedMissingPolicyKind::kSkip},
  };
  const std::array valid_closure{detail::AlignmentClosureProof::kOpen,
                                 detail::AlignmentClosureProof::kEndOfStream};

  auto unknown_closure = valid_closure;
  unknown_closure[1] =
      enum_from_underlying<detail::AlignmentClosureProof>(std::numeric_limits<std::uint8_t>::max());
  EXPECT_FALSE(engine.select(0ns, valid_policies, unknown_closure, TimePoint{10, kClock}));
  EXPECT_EQ(engine.queue_size(0), 1U);

  auto unknown_policy = valid_policies;
  unknown_policy[1].kind = enum_from_underlying<detail::AlignedMissingPolicyKind>(
      std::numeric_limits<std::uint8_t>::max());
  EXPECT_FALSE(engine.select(0ns, unknown_policy, valid_closure, TimePoint{10, kClock}));
  EXPECT_EQ(engine.queue_size(0), 1U);

  auto invalid_skip_age = valid_policies;
  invalid_skip_age[1].reuse_max_age = 1ns;
  EXPECT_FALSE(engine.select(0ns, invalid_skip_age, valid_closure, TimePoint{10, kClock}));

  auto invalid_reuse_age = valid_policies;
  invalid_reuse_age[1].kind = detail::AlignedMissingPolicyKind::kReuseLast;
  EXPECT_FALSE(engine.select(0ns, invalid_reuse_age, valid_closure, TimePoint{10, kClock}));

  auto valid = engine.select(0ns, valid_policies, valid_closure, TimePoint{10, kClock});
  ASSERT_TRUE(valid);
  EXPECT_EQ(valid->disposition(), detail::AlignedCandidateDisposition::kNoFire);
  EXPECT_EQ(engine.queue_size(0), 1U);
}

// REQ: CONTRACT-007
// Anchor and sample timeline (lookbehind=1, lookahead=1 anchor-interval each side):
//   anchors: t=10(primed) t=20(pending) t=30(closes window@10)
//   samples:  t=5  t=10  t=15  t=20  t=25  then t=35 after pop
// Window@10 closes with all five samples; window@20 reuses t=15 onward (overlap).
TEST(TemporalInputCache, WindowPrimesClosesAndRetainsOverlappingSamples) {
  detail::BracketedWindowEngine engine{
      BracketedWindow{.lookbehind = 1, .lookahead = 1, .max_samples = 8}, 16, 2};
  ASSERT_TRUE(engine.push_sample(sample(5, 1, 1)));
  ASSERT_EQ(engine.push_anchor(sample(10, 1, 1))->disposition,
            detail::WindowAnchorDisposition::kPrimed);
  ASSERT_TRUE(engine.push_sample(sample(10, 2, 2)));
  ASSERT_TRUE(engine.push_sample(sample(15, 3, 3)));
  ASSERT_EQ(engine.push_anchor(sample(20, 2, 2))->disposition,
            detail::WindowAnchorDisposition::kPending);
  ASSERT_TRUE(engine.push_sample(sample(20, 4, 4)));
  ASSERT_TRUE(engine.push_sample(sample(25, 5, 5)));
  const auto* closed = engine.front_closed();
  ASSERT_NE(closed, nullptr);
  ASSERT_EQ(closed->samples.size(), 5U);
  EXPECT_EQ(closed->samples.front().capture_time.timestamp_ns, 5);
  EXPECT_EQ(closed->samples.back().capture_time.timestamp_ns, 25);
  engine.pop_closed();

  ASSERT_EQ(engine.push_anchor(sample(30, 3, 3))->disposition,
            detail::WindowAnchorDisposition::kPending);
  ASSERT_TRUE(engine.push_sample(sample(35, 6, 6)));
  const auto* overlap = engine.front_closed();
  ASSERT_NE(overlap, nullptr);
  EXPECT_EQ(overlap->samples.front().capture_time.timestamp_ns, 15);
  engine.pop_closed();
}

// REQ: CONTRACT-007
// A window with max_samples=3 that accumulates 4 samples fires with an empty result
// (kWindowSampleLimit). A subsequent anchor at an earlier timestamp triggers
// kResetAndReprimed so the engine can restart with a new window.
TEST(TemporalInputCache, WindowLimitIsAllOrNothingAndRegressionReprimes) {
  detail::BracketedWindowEngine engine{BracketedWindow{.max_samples = 3}, 8, 1};
  ASSERT_TRUE(engine.push_sample(sample(0, 1, 1)));
  ASSERT_TRUE(engine.push_anchor(sample(1, 1, 1)));
  ASSERT_TRUE(engine.push_sample(sample(1, 2, 2)));
  ASSERT_TRUE(engine.push_sample(sample(2, 3, 3)));
  ASSERT_TRUE(engine.push_anchor(sample(3, 2, 2)));
  ASSERT_TRUE(engine.push_sample(sample(4, 4, 4)));
  const auto* limited = engine.front_closed();
  ASSERT_NE(limited, nullptr);
  EXPECT_TRUE(limited->samples.empty());
  EXPECT_EQ(limited->selection.reason, SelectionReason::kWindowSampleLimit);
  engine.pop_closed();

  auto reset = engine.push_anchor(sample(2, 3, 3));
  ASSERT_TRUE(reset);
  EXPECT_EQ(reset->disposition, detail::WindowAnchorDisposition::kResetAndReprimed);
  EXPECT_EQ(reset->reset_reason, detail::WindowResetReason::kTimestampRegression);
}

// REQ: CONTRACT-007
TEST(TemporalInputCache, WindowPendingLimitClosesTheRejectedAttemptWithTypedReason) {
  detail::BracketedWindowEngine engine{BracketedWindow{.max_samples = 8}, 8, 1};
  ASSERT_TRUE(engine.push_sample(sample(5, 1, 1)));
  ASSERT_TRUE(engine.push_anchor(sample(10, 1, 1)));
  ASSERT_EQ(engine.push_anchor(sample(20, 2, 2))->disposition,
            detail::WindowAnchorDisposition::kPending);

  auto limited = engine.push_anchor(sample(30, 3, 3));
  ASSERT_TRUE(limited);
  EXPECT_EQ(limited->disposition, detail::WindowAnchorDisposition::kClosed);
  const auto* terminal = engine.front_closed();
  ASSERT_NE(terminal, nullptr);
  EXPECT_EQ(terminal->selection.state, InputState::kAbsent);
  EXPECT_EQ(terminal->selection.reason, SelectionReason::kWindowPendingLimit);
  EXPECT_EQ(terminal->reference_time, (TimePoint{30, kClock}));
}

// REQ: CONTRACT-007
TEST(TemporalInputCache, WindowPendingLimitAdvancesTheConsecutiveAnchorBoundary) {
  detail::BracketedWindowEngine engine{BracketedWindow{.max_samples = 8}, 8, 1};
  ASSERT_TRUE(engine.push_anchor(sample(10, 1, 1)));
  ASSERT_TRUE(engine.push_anchor(sample(20, 2, 2)));
  ASSERT_TRUE(engine.push_anchor(sample(30, 3, 3)));

  auto regression = engine.push_anchor(sample(25, 4, 4));
  ASSERT_TRUE(regression);
  EXPECT_EQ(regression->disposition, detail::WindowAnchorDisposition::kResetAndReprimed);
  EXPECT_EQ(regression->reset_reason, detail::WindowResetReason::kTimestampRegression);
}

// REQ: CONTRACT-007
TEST(TemporalInputCache, WindowFullResultRingPreservesFailedAnchorForExactRetry) {
  detail::BracketedWindowEngine engine{BracketedWindow{.max_samples = 8}, 8, 1};
  ASSERT_TRUE(engine.push_anchor(sample(10, 1, 1)));
  ASSERT_TRUE(engine.push_anchor(sample(20, 2, 2)));
  ASSERT_TRUE(engine.push_anchor(sample(30, 3, 3)));
  ASSERT_TRUE(engine.push_anchor(sample(40, 4, 4)));

  auto full = engine.push_anchor(sample(50, 5, 5));
  ASSERT_FALSE(full);
  EXPECT_EQ(full.error().code, ErrorCode::kResourceExhausted);

  engine.pop_closed();
  auto retry = engine.push_anchor(sample(50, 5, 5));
  ASSERT_TRUE(retry);
  EXPECT_EQ(retry->disposition, detail::WindowAnchorDisposition::kClosed);
}

// REQ: CONTRACT-007
TEST(TemporalInputCache, WindowOlderPendingClosesAfterNewAttemptInObservedClosureOrder) {
  detail::BracketedWindowEngine engine{
      BracketedWindow{.lookbehind = 1, .lookahead = 1, .max_samples = 8}, 8, 1};
  ASSERT_TRUE(engine.push_sample(sample(5, 1, 1)));
  ASSERT_TRUE(engine.push_anchor(sample(10, 1, 1)));
  ASSERT_TRUE(engine.push_sample(sample(10, 2, 2)));
  ASSERT_TRUE(engine.push_anchor(sample(20, 2, 2)));
  ASSERT_TRUE(engine.push_sample(sample(20, 3, 3)));

  ASSERT_TRUE(engine.push_anchor(sample(30, 3, 3)));
  const auto* capacity_close = engine.front_closed();
  ASSERT_NE(capacity_close, nullptr);
  EXPECT_EQ(capacity_close->reference_time, (TimePoint{30, kClock}));
  EXPECT_EQ(capacity_close->selection.reason, SelectionReason::kWindowPendingLimit);

  ASSERT_TRUE(engine.push_sample(sample(25, 4, 4)));
  engine.pop_closed();
  const auto* older_close = engine.front_closed();
  ASSERT_NE(older_close, nullptr);
  EXPECT_EQ(older_close->reference_time, (TimePoint{20, kClock}));
  EXPECT_EQ(older_close->selection.state, InputState::kSelected);
}

// REQ: CONTRACT-007
TEST(TemporalInputCache, WindowSuccessorProofClosesMissingPredecessorTerminally) {
  detail::BracketedWindowEngine engine{BracketedWindow{.max_samples = 8}, 8, 1};
  ASSERT_TRUE(engine.push_anchor(sample(10, 1, 1)));
  ASSERT_TRUE(engine.push_sample(sample(10, 1, 1)));
  ASSERT_TRUE(engine.push_anchor(sample(20, 2, 2)));
  ASSERT_TRUE(engine.push_sample(sample(20, 2, 2)));
  ASSERT_TRUE(engine.push_sample(sample(21, 3, 3)));

  const auto* terminal = engine.front_closed();
  ASSERT_NE(terminal, nullptr);
  EXPECT_TRUE(terminal->samples.empty());
  EXPECT_EQ(terminal->selection.state, InputState::kAbsent);
  EXPECT_EQ(terminal->selection.reason, SelectionReason::kInsufficientHistory);
}

// REQ: CONTRACT-007
TEST(TemporalInputCache, WindowReportsSampleEpochClockAndTimestampResets) {
  detail::BracketedWindowEngine engine{BracketedWindow{.max_samples = 8}, 8, 2};
  ASSERT_TRUE(engine.push_sample(sample(10, 4, 1)));
  ASSERT_TRUE(engine.push_anchor(sample(10, 4, 1)));

  auto epoch = engine.push_sample(sample(10, 1, 1, 2));
  ASSERT_TRUE(epoch);
  EXPECT_EQ(epoch->disposition, detail::WindowInputDisposition::kResetAndAccepted);
  EXPECT_EQ(epoch->reset_reason, detail::WindowResetReason::kProducerEpoch);

  auto regression = engine.push_sample(sample(9, 2, 2, 2));
  ASSERT_TRUE(regression);
  EXPECT_EQ(regression->reset_reason, detail::WindowResetReason::kTimestampRegression);

  auto other_clock = sample(11, 3, 3, 2);
  other_clock.capture_time.clock_id = ClockId{8};
  auto clock = engine.push_sample(other_clock);
  ASSERT_TRUE(clock);
  EXPECT_EQ(clock->reset_reason, detail::WindowResetReason::kClockDomain);
}

// REQ: CONTRACT-007
// A producer epoch reset while a closed window is in the output ring must not
// overwrite that closed result; the caller can still pop and process it.
TEST(TemporalInputCache, WindowDiscontinuityPreservesAlreadyClosedSnapshot) {
  detail::BracketedWindowEngine engine{BracketedWindow{.max_samples = 8}, 8, 2};
  ASSERT_TRUE(engine.push_sample(sample(5, 1, 1)));
  ASSERT_TRUE(engine.push_anchor(sample(10, 1, 1)));
  ASSERT_TRUE(engine.push_sample(sample(10, 2, 2)));
  ASSERT_TRUE(engine.push_anchor(sample(20, 2, 2)));
  ASSERT_TRUE(engine.push_sample(sample(20, 3, 3)));
  ASSERT_TRUE(engine.push_sample(sample(25, 4, 4)));
  // Key setup: a closed window is already in the ring before the epoch reset sample.
  const auto* before_reset = engine.front_closed();
  ASSERT_NE(before_reset, nullptr);
  ASSERT_FALSE(before_reset->samples.empty());
  const auto closed_reference = before_reset->reference_time;
  const auto closed_first = before_reset->samples.front();

  auto reset = engine.push_sample(sample(1, 1, 1, 2));
  ASSERT_TRUE(reset);
  EXPECT_EQ(reset->reset_reason, detail::WindowResetReason::kProducerEpoch);
  const auto* after_reset = engine.front_closed();
  ASSERT_NE(after_reset, nullptr);
  EXPECT_EQ(after_reset->reference_time, closed_reference);
  ASSERT_FALSE(after_reset->samples.empty());
  EXPECT_EQ(after_reset->samples.front(), closed_first);

  engine.pop_closed();
  EXPECT_EQ(engine.front_closed(), nullptr);
}

// REQ: CONTRACT-007
TEST(TemporalInputCache, WindowRejectsNonmonotonicSourceAndArrivalWithinEpoch) {
  detail::BracketedWindowEngine engine{BracketedWindow{.max_samples = 8}, 8, 2};
  ASSERT_TRUE(engine.push_sample(sample(10, 2, 2)));
  EXPECT_FALSE(engine.push_sample(sample(11, 2, 3)));
  EXPECT_FALSE(engine.push_sample(sample(11, 3, 2)));
  ASSERT_TRUE(engine.push_anchor(sample(10, 2, 2)));
  EXPECT_FALSE(engine.push_anchor(sample(11, 2, 3)));
  EXPECT_FALSE(engine.push_anchor(sample(11, 3, 2)));
}

// REQ: CONTRACT-007
TEST(TemporalInputCache, WindowLatchesComparisonClockRegardlessOfArrivalOrder) {
  detail::BracketedWindowEngine engine{BracketedWindow{.max_samples = 8}, 8, 2};
  ASSERT_TRUE(engine.push_sample(sample(5, 1, 1)));
  auto other_clock_anchor = sample(10, 1, 1);
  other_clock_anchor.capture_time.clock_id = ClockId{8};
  auto reset = engine.push_anchor(other_clock_anchor);
  ASSERT_TRUE(reset);
  EXPECT_EQ(reset->disposition, detail::WindowAnchorDisposition::kResetAndReprimed);
  EXPECT_EQ(reset->reset_reason, detail::WindowResetReason::kClockDomain);
}

// REQ: CONTRACT-007
TEST(TemporalInputCache, WindowTimeoutAndEndOfStreamClosePendingInOrder) {
  detail::BracketedWindowEngine engine{BracketedWindow{.max_samples = 8}, 8, 2};
  ASSERT_TRUE(engine.push_anchor(sample(10, 1, 1)));
  ASSERT_TRUE(engine.push_anchor(sample(20, 2, 2)));
  ASSERT_TRUE(engine.push_anchor(sample(30, 3, 3)));
  ASSERT_EQ(engine.pending_count(), 2U);

  auto invalid = engine.close_oldest(SelectionReason::kNeverSeen);
  ASSERT_FALSE(invalid);
  EXPECT_EQ(invalid.error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(engine.pending_count(), 2U);

  ASSERT_TRUE(engine.close_oldest(SelectionReason::kWindowTimeout));
  const auto* timeout = engine.front_closed();
  ASSERT_NE(timeout, nullptr);
  EXPECT_EQ(timeout->selection.state, InputState::kAbsent);
  EXPECT_EQ(timeout->selection.reason, SelectionReason::kWindowTimeout);
  EXPECT_EQ(timeout->reference_time, (TimePoint{20, kClock}));
  EXPECT_TRUE(timeout->samples.empty());
  EXPECT_EQ(engine.pending_count(), 1U);
  engine.pop_closed();

  ASSERT_TRUE(engine.close_oldest(SelectionReason::kEndOfStream));
  const auto* end_of_stream = engine.front_closed();
  ASSERT_NE(end_of_stream, nullptr);
  EXPECT_EQ(end_of_stream->selection.state, InputState::kAbsent);
  EXPECT_EQ(end_of_stream->selection.reason, SelectionReason::kEndOfStream);
  EXPECT_EQ(end_of_stream->reference_time, (TimePoint{30, kClock}));
  EXPECT_TRUE(end_of_stream->samples.empty());
  EXPECT_EQ(engine.pending_count(), 0U);
  engine.pop_closed();

  auto empty = engine.close_oldest(SelectionReason::kEndOfStream);
  ASSERT_FALSE(empty);
  EXPECT_EQ(empty.error().code, ErrorCode::kInvalidArgument);
}

// REQ: CONTRACT-007
TEST(TemporalInputCache, WindowEqualTimeSamplesRetainSourceSequenceOrder) {
  detail::BracketedWindowEngine engine{
      BracketedWindow{.lookbehind = 1, .lookahead = 1, .max_samples = 8}, 8, 2};
  ASSERT_TRUE(engine.push_sample(sample(5, 1, 1)));
  ASSERT_TRUE(engine.push_anchor(sample(10, 1, 1)));
  ASSERT_TRUE(engine.push_sample(sample(10, 2, 2)));
  ASSERT_TRUE(engine.push_sample(sample(10, 3, 3)));
  ASSERT_TRUE(engine.push_anchor(sample(10, 2, 2)));
  ASSERT_TRUE(engine.push_sample(sample(11, 4, 4)));

  const auto* closed = engine.front_closed();
  ASSERT_NE(closed, nullptr);
  ASSERT_EQ(closed->samples.size(), 4U);
  EXPECT_EQ(closed->reference_time, (TimePoint{10, kClock}));
  EXPECT_EQ(closed->samples[0], sample(5, 1, 1));
  EXPECT_EQ(closed->samples[1], sample(10, 2, 2));
  EXPECT_EQ(closed->samples[2], sample(10, 3, 3));
  EXPECT_EQ(closed->samples[3], sample(11, 4, 4));
}

// REQ: CONTRACT-008
TEST(TemporalInputCache, ClockScheduleCoalescesInConstantStateAndAcceptsExactNow) {
  // Key setup: schedule starting at t=10 with period 5ns; query at t=27 lands between
  // tick 3 (t=25) and tick 4 (t=30), so index=3 with missed_count=2 (ticks 1 and 2).
  detail::CoalescingClockSchedule schedule{TimePoint{10, kClock}, 5ns};
  auto first = schedule.due(TimePoint{10, kClock});
  ASSERT_TRUE(first);
  ASSERT_TRUE(first->has_value());
  const auto first_tick = first->value_or(detail::ClockTickCandidate{});
  EXPECT_EQ(first_tick.index, 0U);

  auto late = schedule.due(TimePoint{27, kClock});
  ASSERT_TRUE(late);
  ASSERT_TRUE(late->has_value());
  const auto late_tick = late->value_or(detail::ClockTickCandidate{});
  EXPECT_EQ(late_tick.index, 3U);
  EXPECT_EQ(late_tick.missed_first, 1U);
  EXPECT_EQ(late_tick.missed_last, 2U);
  EXPECT_EQ(late_tick.missed_count, 2U);
  EXPECT_EQ(late_tick.scheduled_time, (TimePoint{25, kClock}));

  auto regression = schedule.due(TimePoint{20, kClock});
  ASSERT_FALSE(regression);
  EXPECT_EQ(regression.error().code, ErrorCode::kProtocolError);
}

// REQ: CONTRACT-008
TEST(TemporalInputCache, ClockScheduleUsesFullUint64IndexAcrossNegativePhase) {
  detail::CoalescingClockSchedule schedule{
      TimePoint{std::numeric_limits<std::int64_t>::min(), kClock}, 1ns};
  auto terminal = schedule.due(TimePoint{std::numeric_limits<std::int64_t>::max(), kClock});
  ASSERT_TRUE(terminal);
  ASSERT_TRUE(terminal->has_value());
  const auto terminal_tick = terminal->value_or(detail::ClockTickCandidate{});
  EXPECT_EQ(terminal_tick.index, std::numeric_limits<std::uint64_t>::max());
  EXPECT_EQ(terminal_tick.scheduled_time.timestamp_ns, std::numeric_limits<std::int64_t>::max());
  EXPECT_EQ(terminal_tick.missed_first, 0U);
  EXPECT_EQ(terminal_tick.missed_last, std::numeric_limits<std::uint64_t>::max() - 1U);
  EXPECT_EQ(terminal_tick.missed_count, std::numeric_limits<std::uint64_t>::max());

  auto repeated = schedule.due(TimePoint{std::numeric_limits<std::int64_t>::max(), kClock});
  ASSERT_TRUE(repeated);
  EXPECT_FALSE(*repeated);
}

// REQ: CONTRACT-012
TEST(TemporalInputCache, DeadlineGateUsesInclusiveBoundaryAndClosedViolationActions) {
  // Key setup: reference t=10 with slack 5ns; arrival at t=15 is exactly on the boundary
  // (inclusive pass), arrival at t=16 violates it. Each OnViolation policy maps to a
  // distinct DeadlineGateDisposition.
  auto boundary = detail::evaluate_deadline_gate(
      TimePoint{10, kClock}, 5ns, TimePoint{15, kClock}, OnViolation::kFailRun);
  ASSERT_TRUE(boundary);
  EXPECT_EQ(boundary->disposition, detail::DeadlineGateDisposition::kInvokeNominal);
  EXPECT_EQ(boundary->status.reason, ActivationReason::kNominal);

  auto witness = detail::evaluate_deadline_gate(
      TimePoint{10, kClock}, 5ns, TimePoint{16, kClock}, OnViolation::kEmitWitness);
  ASSERT_TRUE(witness);
  EXPECT_EQ(witness->disposition, detail::DeadlineGateDisposition::kInvokeViolation);
  EXPECT_EQ(witness->status.reason, ActivationReason::kDeadlineExceeded);
  EXPECT_EQ(witness->status.policy, OnViolation::kEmitWitness);

  auto safe = detail::evaluate_deadline_gate(
      TimePoint{10, kClock}, 5ns, TimePoint{16, kClock}, OnViolation::kSafeDefault);
  ASSERT_TRUE(safe);
  EXPECT_EQ(safe->disposition, detail::DeadlineGateDisposition::kInvokeViolation);

  auto fail = detail::evaluate_deadline_gate(
      TimePoint{10, kClock}, 5ns, TimePoint{16, kClock}, OnViolation::kFailRun);
  ASSERT_TRUE(fail);
  EXPECT_EQ(fail->disposition, detail::DeadlineGateDisposition::kFailRun);
  EXPECT_FALSE(detail::evaluate_deadline_gate(
      TimePoint{10, kClock}, 5ns, TimePoint{16, ClockId{8}}, OnViolation::kFailRun));
}

// REQ: CONTRACT-014
TEST(TemporalInputCache, WarmedSelectionHotPathsDoNotAllocate) {
  // Key setup: each engine type performs one warm-up operation outside the monitored
  // scope so that lazy internal allocations do not inflate the allocation count.
  detail::TemporalInputCache scalar{4};
  ASSERT_TRUE(scalar.push(sample(10, 1, 1)));
  {
    auto warm = scalar.select_exact_transaction(TimePoint{10, kClock});
    ASSERT_TRUE(warm);
  }

  detail::TemporalInputCache drain{4};
  ASSERT_TRUE(drain.push(sample(10, 1, 1)));
  {
    auto warm = drain.snapshot_drain_all(1, 4);
    ASSERT_TRUE(warm);
  }

  detail::AlignedJoinEngine aligned{2, 2};
  ASSERT_TRUE(aligned.push(0, sample(10, 1, 1)));
  ASSERT_TRUE(aligned.push(1, sample(10, 1, 1)));
  const std::array policies{detail::AlignedMissingPolicy{}, detail::AlignedMissingPolicy{}};
  const std::array closure{detail::AlignmentClosureProof::kOpen,
                           detail::AlignmentClosureProof::kOpen};
  {
    auto warm = aligned.select(0ns, policies, closure);
    ASSERT_TRUE(warm);
  }

  detail::BracketedWindowEngine window{BracketedWindow{.max_samples = 8}, 8, 2};
  ASSERT_TRUE(window.push_sample(sample(5, 1, 1)));
  ASSERT_TRUE(window.push_anchor(sample(10, 1, 1)));

  detail::CoalescingClockSchedule clock{TimePoint{0, kClock}, 1ns};
  ASSERT_TRUE(clock.due(TimePoint{0, kClock}));

  std::size_t allocations = 0;
  bool operations_succeeded = false;
  {
    const ScopedAllocationCounter counter;
    auto scalar_selection = scalar.select_exact_transaction(TimePoint{10, kClock});
    auto drain_selection = drain.snapshot_drain_all(1, 4);
    auto aligned_selection = aligned.select(0ns, policies, closure);
    auto window_sample = window.push_sample(sample(10, 2, 2));
    auto window_anchor = window.push_anchor(sample(20, 2, 2));
    auto window_successor = window.push_sample(sample(25, 3, 3));
    auto tick = clock.due(TimePoint{1, kClock});
    const bool scalar_committed = scalar_selection && scalar_selection->commit();
    const bool drain_committed = drain_selection && drain_selection->commit();
    const bool aligned_committed = aligned_selection && aligned_selection->commit();
    const bool window_closed = window.front_closed() != nullptr;
    window.pop_closed();
    operations_succeeded = scalar_committed && drain_committed && aligned_committed &&
                           window_sample && window_anchor && window_successor && window_closed &&
                           tick;
    allocations = counter.count();
  }
  // Key assertion: zero heap allocations across all engine types on the steady-state path.
  EXPECT_TRUE(operations_succeeded);
  EXPECT_EQ(allocations, 0U);
}

}  // namespace
}  // namespace holoscan::test

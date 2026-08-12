/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <thread>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include <holoscan/core/source_id.hpp>
#include <holoscan/core/tag.hpp>

#include <core/source_id_internal.hpp>
#include <core/tag_internal.hpp>

namespace holoscan {
namespace {

template <typename T>
concept LessComparable = requires(const T lhs, const T rhs) {
  { lhs < rhs } -> std::convertible_to<bool>;
};

template <typename T>
concept HasTag = requires(const T value) { value.tag; };

template <typename T>
concept HasTagTime = requires(const T value) { value.tag_time; };

// REQ: CLOCK-016
static_assert(sizeof(Tag) == 16);
static_assert(sizeof(TagTime) == 16);
static_assert(offsetof(Tag, sequence) == 0);
static_assert(offsetof(Tag, generation) == 8);
static_assert(offsetof(Tag, reserved) == 12);
static_assert(offsetof(TagTime, logical_ns) == 0);
static_assert(offsetof(TagTime, microstep) == 8);
static_assert(offsetof(TagTime, reserved) == 12);
static_assert(std::is_standard_layout_v<Tag>);
static_assert(std::is_trivially_copyable_v<Tag>);
static_assert(std::is_standard_layout_v<TagTime>);
static_assert(std::is_trivially_copyable_v<TagTime>);
static_assert(!LessComparable<Tag>);
// REQ: CLOCK-017
static_assert(LessComparable<TagTime>);
// REQ: CLOCK-019
static_assert(sizeof(SourceId) == 4);
static_assert(std::is_standard_layout_v<SourceId>);
static_assert(std::is_trivially_copyable_v<SourceId>);
static_assert(!LessComparable<SourceId>);
// REQ: CLOCK-020
static_assert(!HasTag<detail::SourceAcceptanceStamp>);
static_assert(!HasTagTime<detail::SourceAcceptanceStamp>);
static_assert(!std::is_copy_constructible_v<detail::TagAllocator>);
static_assert(!std::is_move_constructible_v<detail::TagAllocator>);
static_assert(!std::is_copy_constructible_v<detail::SourceAcceptanceAllocator>);
static_assert(!std::is_move_constructible_v<detail::SourceAcceptanceAllocator>);

// REQ: CLOCK-016
TEST(Tag, ValidityHasOneCanonicalInvalidValueAndRejectsMalformedFields) {
  constexpr Tag invalid{};
  constexpr Tag valid{.sequence = 1, .generation = 1, .reserved = 0};
  constexpr Tag zero_sequence{.sequence = 0, .generation = 1, .reserved = 0};
  constexpr Tag zero_generation{.sequence = 1, .generation = 0, .reserved = 0};
  constexpr Tag unsupported_version{.sequence = 1, .generation = 1, .reserved = 1};

  static_assert(!invalid.valid());
  static_assert(valid.valid());
  static_assert(!zero_sequence.valid());
  static_assert(!zero_generation.valid());
  static_assert(!unsupported_version.valid());
}

// REQ: CLOCK-016
TEST(Tag, EqualityAndHashDoNotImplyTemporalOrder) {
  const Tag first{.sequence = 1, .generation = 7, .reserved = 0};
  const Tag same{.sequence = 1, .generation = 7, .reserved = 0};
  const Tag later_sequence{.sequence = 2, .generation = 7, .reserved = 0};
  const Tag other_generation{.sequence = 1, .generation = 8, .reserved = 0};

  EXPECT_EQ(first, same);
  EXPECT_NE(first, later_sequence);
  EXPECT_NE(first, other_generation);

  std::unordered_set<Tag> tags;
  tags.insert(first);
  tags.insert(same);
  tags.insert(later_sequence);
  tags.insert(other_generation);
  EXPECT_EQ(tags.size(), 3);
}

// REQ: CLOCK-017
TEST(TagTime, ZeroIsValidAndOrderingIsLexicographicWithoutReservedBits) {
  // Key setup: TagTime values including negative logical_ns, microstep tie-breaking, and
  //   instances with reserved != 0 (unsupported versions).
  // Key assertion: equality checks all fields including reserved; ordering uses only (logical_ns,
  //   microstep) so an unsupported-version value is incomparable with its same-coordinate twin
  //   yet still less than a value with a larger microstep; hash keys on (logical_ns, microstep).
  constexpr TagTime zero{};
  constexpr TagTime earlier{.logical_ns = -1, .microstep = 99, .reserved = 0};
  constexpr TagTime first{.logical_ns = 5, .microstep = 0, .reserved = 0};
  constexpr TagTime second{.logical_ns = 5, .microstep = 1, .reserved = 0};
  constexpr TagTime later{.logical_ns = 6, .microstep = 0, .reserved = 0};
  constexpr TagTime unsupported_version{.logical_ns = 5, .microstep = 0, .reserved = 1};

  static_assert(zero.valid());
  static_assert(earlier < first);
  static_assert(first < second);
  static_assert(second < later);
  static_assert(!unsupported_version.valid());
  static_assert(first == TagTime{.logical_ns = 5, .microstep = 0, .reserved = 0});
  static_assert(first != unsupported_version);
  static_assert(!(first < unsupported_version));
  static_assert(!(unsupported_version < first));
  static_assert(unsupported_version < second);

  constexpr TagTime unsupported_earlier{.logical_ns = 4, .microstep = 99, .reserved = 7};
  static_assert(!unsupported_earlier.valid());
  static_assert(unsupported_earlier < first);
  static_assert(!(first < unsupported_earlier));

  std::unordered_set<TagTime> times;
  times.insert(first);
  times.insert(TagTime{.logical_ns = 5, .microstep = 0, .reserved = 0});
  times.insert(second);
  EXPECT_EQ(times.size(), 2);
}

// REQ: CLOCK-018
TEST(TagAllocator, ReservesEveryFrozenCandidateIncludingNoComputeOutcomes) {
  // Key setup: 'no_compute' models a candidate whose activation skips compute() (e.g., predicate
  //   false or deadline miss); it permanently occupies a sequence slot and cannot be recycled.
  // Key assertion: sequences advance monotonically; the third candidate follows the frozen slot
  //   rather than reusing it.
  detail::TagAllocator allocator;
  EXPECT_EQ(allocator.generation(), 1);

  auto fired = allocator.reserve_candidate();
  ASSERT_TRUE(fired);
  EXPECT_EQ(*fired, (Tag{.sequence = 1, .generation = 1, .reserved = 0}));

  // This candidate models a predicate false/missing/deadline disposition. Not invoking compute
  // cannot return its identity to the allocator.
  auto no_compute = allocator.reserve_candidate();
  ASSERT_TRUE(no_compute);
  EXPECT_EQ(*no_compute, (Tag{.sequence = 2, .generation = 1, .reserved = 0}));

  auto next = allocator.reserve_candidate();
  ASSERT_TRUE(next);
  EXPECT_EQ(*next, (Tag{.sequence = 3, .generation = 1, .reserved = 0}));
}

// REQ: CLOCK-018
TEST(TagAllocator, PublishesRestartGenerationOnlyAtSuccessfulStart) {
  detail::TagAllocator allocator;
  ASSERT_TRUE(allocator.reserve_candidate());

  // A failed restart/rollback does not call publish_restart(), so the published epoch and sequence
  // continue without consuming a generation.
  auto after_failed_restart = allocator.reserve_candidate();
  ASSERT_TRUE(after_failed_restart);
  EXPECT_EQ(*after_failed_restart, (Tag{.sequence = 2, .generation = 1, .reserved = 0}));

  ASSERT_TRUE(allocator.publish_restart());
  EXPECT_EQ(allocator.generation(), 2);
  auto first_after_restart = allocator.reserve_candidate();
  ASSERT_TRUE(first_after_restart);
  EXPECT_EQ(*first_after_restart, (Tag{.sequence = 1, .generation = 2, .reserved = 0}));
}

// REQ: CLOCK-018
TEST(TagAllocator, SequenceExhaustionReturnsTerminalFailureBeforeWrapOrReuse) {
  detail::TagAllocator allocator(/*generation=*/7,
                                 /*next_sequence=*/std::numeric_limits<std::uint64_t>::max());

  auto last = allocator.reserve_candidate();
  ASSERT_TRUE(last);
  EXPECT_EQ(last->sequence, std::numeric_limits<std::uint64_t>::max());
  EXPECT_EQ(last->generation, 7);

  auto exhausted = allocator.reserve_candidate();
  ASSERT_FALSE(exhausted);
  EXPECT_EQ(exhausted.error().code, ErrorCode::kResourceExhausted);
  EXPECT_FALSE(allocator.reserve_candidate());
}

// REQ: CLOCK-018
TEST(TagAllocator, GenerationExhaustionPreservesThePublishedEpochAndSequence) {
  detail::TagAllocator allocator(/*generation=*/std::numeric_limits<std::uint32_t>::max(),
                                 /*next_sequence=*/9);

  auto exhausted = allocator.publish_restart();
  ASSERT_FALSE(exhausted);
  EXPECT_EQ(exhausted.error().code, ErrorCode::kResourceExhausted);
  EXPECT_EQ(allocator.generation(), std::numeric_limits<std::uint32_t>::max());

  auto still_published = allocator.reserve_candidate();
  ASSERT_TRUE(still_published);
  EXPECT_EQ(still_published->generation, std::numeric_limits<std::uint32_t>::max());
  EXPECT_EQ(still_published->sequence, 9);
}

// REQ: CLOCK-019
TEST(SourceId, IsStrongNonzeroEqualityHashOnlyPlanLocalIdentity) {
  const SourceId invalid;
  const SourceId source = detail::make_source_id(42);
  const SourceId same = detail::make_source_id(42);
  const SourceId different = detail::make_source_id(43);

  EXPECT_FALSE(static_cast<bool>(invalid));
  EXPECT_TRUE(static_cast<bool>(source));
  EXPECT_EQ(source, same);
  EXPECT_NE(source, different);

  std::unordered_set<SourceId> sources;
  sources.insert(source);
  sources.insert(same);
  sources.insert(different);
  EXPECT_EQ(sources.size(), 2);
}

// REQ: CLOCK-020
TEST(SourceAcceptanceAllocator, StartsAtOneAndReservesInGenerationLocalOrder) {
  detail::SourceAcceptanceAllocator allocator;
  EXPECT_EQ(allocator.generation(), 1);

  auto first = allocator.reserve_acceptance();
  auto second = allocator.reserve_acceptance();
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  EXPECT_EQ(first->generation, 1);
  EXPECT_EQ(first->sequence, 1);
  EXPECT_EQ(second->generation, 1);
  EXPECT_EQ(second->sequence, 2);
}

// REQ: CLOCK-020
TEST(SourceAcceptanceAllocator, ConcurrentProducersReserveUniqueAcceptedWitnesses) {
  // Key setup: 8 threads each call reserve_acceptance() 256 times on the same allocator; stamps
  //   are stored in a per-thread slice of a shared vector.
  // Key assertion: all 2048 stamps succeed; after sorting by sequence they form [1..2048] with
  //   generation=1, confirming the atomic counter has no loss or duplication across threads.
  constexpr std::size_t kProducerCount = 8;
  constexpr std::size_t kReservationsPerProducer = 256;
  constexpr std::size_t kReservationCount = kProducerCount * kReservationsPerProducer;
  detail::SourceAcceptanceAllocator allocator;
  std::vector<detail::SourceAcceptanceStamp> stamps(kReservationCount);
  std::vector<std::thread> producers;
  std::atomic<bool> succeeded{true};
  producers.reserve(kProducerCount);

  for (std::size_t producer = 0; producer < kProducerCount; ++producer) {
    producers.emplace_back([&, producer] {
      const std::size_t first = producer * kReservationsPerProducer;
      for (std::size_t offset = 0; offset < kReservationsPerProducer; ++offset) {
        auto stamp = allocator.reserve_acceptance();
        if (!stamp) {
          succeeded.store(false, std::memory_order_relaxed);
          return;
        }
        stamps[first + offset] = *stamp;
      }
    });
  }
  for (auto& producer : producers) {
    producer.join();
  }
  ASSERT_TRUE(succeeded.load(std::memory_order_relaxed));

  std::sort(stamps.begin(), stamps.end(), [](const auto lhs, const auto rhs) {
    return lhs.sequence < rhs.sequence;
  });
  for (std::size_t index = 0; index < stamps.size(); ++index) {
    EXPECT_EQ(stamps[index].generation, 1);
    EXPECT_EQ(stamps[index].sequence, index + 1U);
    EXPECT_EQ(stamps[index].reserved, 0);
  }
}

// REQ: CLOCK-020
TEST(SourceAcceptanceAllocator, RestartPublicationResetsSequenceOnlyOnSuccess) {
  detail::SourceAcceptanceAllocator allocator;
  ASSERT_TRUE(allocator.reserve_acceptance());

  // Failed lifecycle publication leaves the old allocator state published, so the caller does not
  // invoke publish_restart() and acceptance continues in the old generation.
  auto after_failed_publication = allocator.reserve_acceptance();
  ASSERT_TRUE(after_failed_publication);
  EXPECT_EQ(after_failed_publication->generation, 1);
  EXPECT_EQ(after_failed_publication->sequence, 2);

  ASSERT_TRUE(allocator.publish_restart());
  auto first_after_restart = allocator.reserve_acceptance();
  ASSERT_TRUE(first_after_restart);
  EXPECT_EQ(first_after_restart->generation, 2);
  EXPECT_EQ(first_after_restart->sequence, 1);
}

// REQ: CLOCK-020
TEST(SourceAcceptanceAllocator, SequenceExhaustionUsesMaximumOnceThenFailsBeforeWrap) {
  detail::SourceAcceptanceAllocator allocator(
      /*generation=*/11, /*next_sequence=*/std::numeric_limits<std::uint64_t>::max());

  auto last = allocator.reserve_acceptance();
  ASSERT_TRUE(last);
  EXPECT_EQ(last->generation, 11);
  EXPECT_EQ(last->sequence, std::numeric_limits<std::uint64_t>::max());

  auto exhausted = allocator.reserve_acceptance();
  ASSERT_FALSE(exhausted);
  EXPECT_EQ(exhausted.error().code, ErrorCode::kResourceExhausted);
  EXPECT_FALSE(allocator.reserve_acceptance());
}

// REQ: CLOCK-020
TEST(SourceAcceptanceAllocator, GenerationExhaustionDoesNotAlterPublishedState) {
  detail::SourceAcceptanceAllocator allocator(
      /*generation=*/std::numeric_limits<std::uint32_t>::max(), /*next_sequence=*/5);

  auto exhausted = allocator.publish_restart();
  ASSERT_FALSE(exhausted);
  EXPECT_EQ(exhausted.error().code, ErrorCode::kResourceExhausted);
  EXPECT_EQ(allocator.generation(), std::numeric_limits<std::uint32_t>::max());

  auto still_published = allocator.reserve_acceptance();
  ASSERT_TRUE(still_published);
  EXPECT_EQ(still_published->generation, std::numeric_limits<std::uint32_t>::max());
  EXPECT_EQ(still_published->sequence, 5);
}

}  // namespace
}  // namespace holoscan

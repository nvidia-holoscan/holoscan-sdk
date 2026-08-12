/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/payload_kind.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/memory/memory_buffer.hpp>

namespace holoscan {

// Minimal generated-root types used to verify PublishedPayload admission rules:
// CopyableGeneratedRoot satisfies the concept; MoveOnlyGeneratedRoot does not because
// PublishedPayload requires CopyConstructible (fanout duplicates the value).
struct CopyableGeneratedRoot {};

struct MoveOnlyGeneratedRoot {
  MoveOnlyGeneratedRoot() = default;
  MoveOnlyGeneratedRoot(const MoveOnlyGeneratedRoot&) = delete;
  MoveOnlyGeneratedRoot& operator=(const MoveOnlyGeneratedRoot&) = delete;
  MoveOnlyGeneratedRoot(MoveOnlyGeneratedRoot&&) = default;
  MoveOnlyGeneratedRoot& operator=(MoveOnlyGeneratedRoot&&) = default;
};

template <>
consteval SchemaIdentity schema_identity<CopyableGeneratedRoot>() {
  return SchemaIdentity{.canonical_name = "nvidia.holoscan.test.CopyableGeneratedRoot",
                        .version = 1};
}

template <>
consteval SchemaIdentity schema_identity<MoveOnlyGeneratedRoot>() {
  return SchemaIdentity{.canonical_name = "nvidia.holoscan.test.MoveOnlyGeneratedRoot",
                        .version = 1};
}

namespace {

// ArbitraryCopyableClass is copyable but has no schema_identity specialisation, so it
// must NOT satisfy PublishedPayload — copyability alone is not sufficient.
struct ArbitraryCopyableClass {
  std::uint64_t value{};
};

// The implementation's positive POD admission machinery admits SDK scalar
// values. Copyability alone must not admit ArbitraryCopyableClass.
template <typename T>
concept TestPublishedPayload = PublishedPayload<T>;

static_assert(std::is_same_v<std::underlying_type_t<PayloadKind>, std::uint8_t>);
static_assert(static_cast<std::uint8_t>(PayloadKind::kInlineTyped) == 0);
static_assert(static_cast<std::uint8_t>(PayloadKind::kSerializedBytes) == 1);
static_assert(static_cast<std::uint8_t>(PayloadKind::kDescriptorBacked) == 2);
static_assert(static_cast<std::uint8_t>(PayloadKind::kCompatibilityPayload) == 3);
static_assert(static_cast<std::uint8_t>(PayloadKind::kControlEmpty) == 4);

static_assert(TestPublishedPayload<std::uint32_t>);
static_assert(TestPublishedPayload<float>);
static_assert(TestPublishedPayload<Tensor>);
static_assert(TestPublishedPayload<CopyableGeneratedRoot>);
static_assert(!TestPublishedPayload<MoveOnlyGeneratedRoot>);
static_assert(!TestPublishedPayload<const std::uint32_t>);
static_assert(!TestPublishedPayload<std::uint32_t&>);
static_assert(!TestPublishedPayload<std::uint32_t*>);
static_assert(!TestPublishedPayload<std::unique_ptr<std::uint32_t>>);
static_assert(!TestPublishedPayload<std::reference_wrapper<std::uint32_t>>);
static_assert(!TestPublishedPayload<std::span<std::uint32_t>>);
static_assert(!TestPublishedPayload<std::string_view>);
static_assert(!TestPublishedPayload<std::string>);
static_assert(!TestPublishedPayload<MemoryBuffer>);
static_assert(!TestPublishedPayload<ArbitraryCopyableClass>);

static_assert(std::is_same_v<decltype(EmitOptions::capture_time), std::optional<TimePoint>>);
static_assert(std::is_same_v<decltype(EmitOptions::frame_id), std::optional<std::uint64_t>>);
static_assert(std::is_same_v<decltype(EmitOptions::trace), std::optional<TraceContext>>);
static_assert(std::is_same_v<EmitFlags, SampleFlags>);
static_assert(((SampleFlags::kInvalid | SampleFlags::kDegraded) & SampleFlags::kInvalid) ==
              SampleFlags::kInvalid);

// REQ: OPERATOR-001
// A default SampleMetadata is version 1.0, valid (not invalid, not degraded), inline-typed,
// and has no capture time (timestamp=0, clock_id=default).
TEST(SampleMetadata, DefaultsAreVersionedValidAndCaptureTimeIsCanonicallyAbsent) {
  const SampleMetadata metadata;

  EXPECT_EQ(metadata.version.major, 1);
  EXPECT_EQ(metadata.version.minor, 0);
  EXPECT_TRUE(metadata.valid());
  EXPECT_FALSE(metadata.degraded());
  EXPECT_EQ(metadata.payload_kind, PayloadKind::kInlineTyped);

  const TimePoint capture_time = metadata.capture_time();
  EXPECT_FALSE(capture_time.valid());
  EXPECT_EQ(capture_time.timestamp_ns, 0);
  EXPECT_EQ(capture_time.clock_id, ClockId{});
}

// REQ: OPERATOR-001
// Validity (kInvalid/kDegraded flags) and capture-time presence are orthogonal:
// a degraded sample may have no capture time; an invalid sample may still have one.
TEST(SampleMetadata, PayloadValidityAndCaptureTimePresenceAreIndependent) {
  SampleMetadata without_capture_time;
  without_capture_time.flags = SampleFlags::kDegraded;
  EXPECT_TRUE(without_capture_time.valid());
  EXPECT_TRUE(without_capture_time.degraded());
  EXPECT_FALSE(without_capture_time.capture_time().valid());

  SampleMetadata invalid_with_capture_time;
  invalid_with_capture_time.flags = SampleFlags::kInvalid;
  invalid_with_capture_time.capture_timestamp_ns = -7;
  invalid_with_capture_time.source_clock_id = ClockId{.value = 11};
  EXPECT_FALSE(invalid_with_capture_time.valid());
  EXPECT_TRUE(invalid_with_capture_time.capture_time().valid());
  EXPECT_EQ(invalid_with_capture_time.capture_time().timestamp_ns, -7);
  EXPECT_EQ(invalid_with_capture_time.capture_time().clock_id.value, 11);
}

// REQ: OPERATOR-015
// SampleFlags supports bitwise OR and AND so callers can combine and test individual
// flag bits without accidentally clearing unrelated flags.
TEST(SampleMetadata, FlagOperatorsComposeIndependentValidityEvidence) {
  SampleFlags flags = SampleFlags::kNone;
  flags |= SampleFlags::kDegraded;
  flags |= SampleFlags::kInvalid;

  EXPECT_EQ(flags & SampleFlags::kDegraded, SampleFlags::kDegraded);
  EXPECT_EQ(flags & SampleFlags::kInvalid, SampleFlags::kInvalid);
  EXPECT_EQ(flags & SampleFlags::kNone, SampleFlags::kNone);
}

// REQ: OPERATOR-001
// Sample<T> is a value type: it stores both the payload and its SampleMetadata inline,
// so no heap allocation occurs for scalar types. Copy and move construction are available.
TEST(Sample, OwnsPayloadAndTransportNeutralMetadataByValue) {
  // Key setup: explicitly populated metadata with every transport-neutral field set
  // so that all accessor round-trips can be checked in one pass.
  const SampleMetadata metadata{
      .sequence = 9,
      .capture_timestamp_ns = 123,
      .source_clock_id = ClockId{.value = 4},
      .frame_id = 17,
      .byte_size = sizeof(std::uint32_t),
      .payload_kind = PayloadKind::kInlineTyped,
  };
  const Sample<std::uint32_t> sample{42, metadata};

  EXPECT_EQ(sample.data, 42);
  EXPECT_EQ(sample.metadata.sequence, 9);
  EXPECT_EQ(sample.metadata.frame_id, 17);
  EXPECT_EQ(sample.metadata.capture_time(),
            (TimePoint{.timestamp_ns = 123, .clock_id = ClockId{.value = 4}}));

  static_assert(std::is_copy_constructible_v<Sample<std::uint32_t>>);
  static_assert(std::is_move_constructible_v<Sample<std::uint32_t>>);
  static_assert(std::is_nothrow_destructible_v<Sample<std::uint32_t>>);
}

}  // namespace
}  // namespace holoscan

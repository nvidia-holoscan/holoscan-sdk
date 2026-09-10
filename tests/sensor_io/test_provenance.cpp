/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <cstdint>

#include <holoscan/core/sample.hpp>
#include <holoscan/sensor_io/provenance.hpp>
#include <holoscan/time/time_point.hpp>

namespace holoscan::sensor_io {
namespace {

constexpr ClockId kCaptureDomain{3};
constexpr ClockId kOtherDomain{4};

[[nodiscard]] SampleMetadata stamped(std::int64_t capture_ns, std::uint64_t frame_id = 0U,
                                     ClockId domain = kCaptureDomain) {
  SampleMetadata metadata{};
  metadata.capture_timestamp_ns = capture_ns;
  metadata.source_clock_id = domain;
  metadata.frame_id = frame_id;
  return metadata;
}

// REQ: SENSOR-IO-004
TEST(SensorIoProvenance, ForwardsCaptureTimeFrameAndTrace) {
  SampleMetadata source = stamped(1'700, 42U);
  source.trace_id = 99U;

  const EmitOptions options = derive_from(source);

  ASSERT_TRUE(options.capture_time.has_value());
  EXPECT_EQ(options.capture_time->timestamp_ns, 1'700);
  EXPECT_EQ(options.capture_time->clock_id, kCaptureDomain);
  ASSERT_TRUE(options.frame_id.has_value());
  EXPECT_EQ(*options.frame_id, 42U);
  ASSERT_TRUE(options.trace.has_value());
  EXPECT_EQ(options.trace->trace_id, 99U);
}

// A zero coordinate in a valid clock domain is a real instant, so an absent capture time must stay
// absent rather than becoming a plausible-looking zero.
// REQ: SENSOR-IO-004
TEST(SensorIoProvenance, AbsentCaptureTimeStaysAbsent) {
  SampleMetadata source{};
  source.capture_timestamp_ns = 5'000;  // stale coordinate with no valid domain

  const EmitOptions options = derive_from(source);

  EXPECT_FALSE(options.capture_time.has_value());
  EXPECT_FALSE(options.frame_id.has_value());
  EXPECT_FALSE(options.trace.has_value());
}

// REQ: SENSOR-IO-004
TEST(SensorIoProvenance, ZeroCoordinateInAValidDomainIsForwardedAsReal) {
  const EmitOptions options = derive_from(stamped(0));

  ASSERT_TRUE(options.capture_time.has_value());
  EXPECT_EQ(options.capture_time->timestamp_ns, 0);
  EXPECT_EQ(options.capture_time->clock_id, kCaptureDomain);
}

// REQ: SENSOR-IO-004
TEST(SensorIoProvenance, DegradedEvidenceIsCarriedForward) {
  SampleMetadata source = stamped(10);
  source.flags = SampleFlags::kDegraded;

  const EmitOptions options = derive_from(source);

  EXPECT_TRUE((options.flags & SampleFlags::kDegraded) == SampleFlags::kDegraded);
}

// REQ: SENSOR-IO-004
TEST(SensorIoProvenance, DerivesFromAnOwningSample) {
  const Sample<std::int32_t> source{7, stamped(2'500, 11U)};

  const EmitOptions options = derive_from(source);

  ASSERT_TRUE(options.capture_time.has_value());
  EXPECT_EQ(options.capture_time->timestamp_ns, 2'500);
  ASSERT_TRUE(options.frame_id.has_value());
  EXPECT_EQ(*options.frame_id, 11U);
}

// REQ: SENSOR-IO-004
TEST(SensorIoProvenanceAccumulator, EmptyAccumulatorHasNoDefensibleAnswer) {
  const ProvenanceAccumulator accumulator;

  const auto options = accumulator.emit_options(CaptureTimePolicy::kEarliest);

  ASSERT_FALSE(options.has_value());
  EXPECT_EQ(options.error().code, ErrorCode::kInvalidArgument);
}

// REQ: SENSOR-IO-004
TEST(SensorIoProvenanceAccumulator, EarliestPolicyInheritsTheStalestInput) {
  ProvenanceAccumulator accumulator;
  accumulator.observe(stamped(300));
  accumulator.observe(stamped(100));
  accumulator.observe(stamped(200));

  const auto options = accumulator.emit_options(CaptureTimePolicy::kEarliest);

  ASSERT_TRUE(options.has_value());
  ASSERT_TRUE(options->capture_time.has_value());
  EXPECT_EQ(options->capture_time->timestamp_ns, 100);
  EXPECT_EQ(options->capture_time->clock_id, kCaptureDomain);
  EXPECT_EQ(accumulator.observed(), 3U);
}

// REQ: SENSOR-IO-004
TEST(SensorIoProvenanceAccumulator, LatestPolicyInheritsTheFreshestInput) {
  ProvenanceAccumulator accumulator;
  accumulator.observe(stamped(300));
  accumulator.observe(stamped(100));

  const auto options = accumulator.emit_options(CaptureTimePolicy::kLatest);

  ASSERT_TRUE(options.has_value());
  ASSERT_TRUE(options->capture_time.has_value());
  EXPECT_EQ(options->capture_time->timestamp_ns, 300);
}

// Coordinates from different domains have no ordering without an admitted conversion, so there is
// no honest "earliest" to pick.
// REQ: SENSOR-IO-004
TEST(SensorIoProvenanceAccumulator, MixedClockDomainsAreRejectedRatherThanSilentlyOrdered) {
  ProvenanceAccumulator accumulator;
  accumulator.observe(stamped(100, 0U, kCaptureDomain));
  accumulator.observe(stamped(200, 0U, kOtherDomain));

  EXPECT_TRUE(accumulator.mixed_domains());
  const auto options = accumulator.emit_options(CaptureTimePolicy::kEarliest);

  ASSERT_FALSE(options.has_value());
  EXPECT_EQ(options.error().code, ErrorCode::kInvalidArgument);
}

// The extremum of a set with unknown members is unknown, and the result should say so rather than
// quietly reporting the extremum of the members it happened to see.
// REQ: SENSOR-IO-004
TEST(SensorIoProvenanceAccumulator, AnUnstampedContributorDemotesTheResult) {
  ProvenanceAccumulator accumulator;
  accumulator.observe(stamped(100));
  accumulator.observe(SampleMetadata{});

  EXPECT_TRUE(accumulator.has_unstamped_contributor());
  const auto options = accumulator.emit_options(CaptureTimePolicy::kEarliest);

  ASSERT_TRUE(options.has_value());
  EXPECT_FALSE(options->capture_time.has_value());
  EXPECT_TRUE((options->flags & SampleFlags::kDegraded) == SampleFlags::kDegraded);
}

// When nothing was stamped there is no provenance to lose, so the result is unstamped but not
// degraded.
// REQ: SENSOR-IO-004
TEST(SensorIoProvenanceAccumulator, AllUnstampedContributorsProduceNoDegradation) {
  ProvenanceAccumulator accumulator;
  accumulator.observe(SampleMetadata{});
  accumulator.observe(SampleMetadata{});

  const auto options = accumulator.emit_options(CaptureTimePolicy::kEarliest);

  ASSERT_TRUE(options.has_value());
  EXPECT_FALSE(options->capture_time.has_value());
  EXPECT_EQ(options->flags, SampleFlags::kNone);
}

// REQ: SENSOR-IO-004
TEST(SensorIoProvenanceAccumulator, OneInvalidContributorInvalidatesTheResult) {
  SampleMetadata invalid = stamped(50);
  invalid.flags = SampleFlags::kInvalid;

  ProvenanceAccumulator accumulator;
  accumulator.observe(stamped(10));
  accumulator.observe(invalid);

  const auto options = accumulator.emit_options(CaptureTimePolicy::kLatest);

  ASSERT_TRUE(options.has_value());
  EXPECT_TRUE((options->flags & SampleFlags::kInvalid) == SampleFlags::kInvalid);
}

// REQ: SENSOR-IO-004
TEST(SensorIoProvenanceAccumulator, UnanimousFrameIdentitySurvives) {
  ProvenanceAccumulator accumulator;
  accumulator.observe(stamped(10, 77U));
  accumulator.observe(stamped(20, 77U));

  const auto options = accumulator.emit_options(CaptureTimePolicy::kEarliest);

  ASSERT_TRUE(options.has_value());
  ASSERT_TRUE(options->frame_id.has_value());
  EXPECT_EQ(*options->frame_id, 77U);
}

// REQ: SENSOR-IO-004
TEST(SensorIoProvenanceAccumulator, DisagreeingFrameIdentityIsDropped) {
  ProvenanceAccumulator accumulator;
  accumulator.observe(stamped(10, 77U));
  accumulator.observe(stamped(20, 78U));

  const auto options = accumulator.emit_options(CaptureTimePolicy::kEarliest);

  ASSERT_TRUE(options.has_value());
  EXPECT_FALSE(options->frame_id.has_value());
}

// REQ: SENSOR-IO-004
TEST(SensorIoProvenanceAccumulator, FrameIdentityIsDroppedWhenAContributorHasNone) {
  ProvenanceAccumulator accumulator;
  accumulator.observe(stamped(10, 77U));
  accumulator.observe(stamped(20, 0U));

  const auto options = accumulator.emit_options(CaptureTimePolicy::kEarliest);

  ASSERT_TRUE(options.has_value());
  EXPECT_FALSE(options->frame_id.has_value());
}

// REQ: SENSOR-IO-004
TEST(SensorIoProvenanceAccumulator, FrameIdentityStaysRetiredAfterAConflict) {
  ProvenanceAccumulator accumulator;
  accumulator.observe(stamped(10, 77U));
  accumulator.observe(stamped(20, 78U));
  accumulator.observe(stamped(30, 77U));

  const auto options = accumulator.emit_options(CaptureTimePolicy::kEarliest);

  ASSERT_TRUE(options.has_value());
  EXPECT_FALSE(options->frame_id.has_value());
}

// REQ: SENSOR-IO-004
TEST(SensorIoProvenanceAccumulator, AgreedTraceIdentityCorrelatesTheResult) {
  SampleMetadata traced = stamped(10);
  traced.trace_id = 500U;
  SampleMetadata same_root = stamped(20);
  same_root.trace_id = 500U;

  ProvenanceAccumulator accumulator;
  accumulator.observe(traced);
  accumulator.observe(same_root);

  const auto options = accumulator.emit_options(CaptureTimePolicy::kEarliest);

  ASSERT_TRUE(options.has_value());
  ASSERT_TRUE(options->trace.has_value());
  EXPECT_EQ(options->trace->trace_id, 500U);
}

// REQ: SENSOR-IO-004
TEST(SensorIoProvenanceAccumulator, UntracedContributorDoesNotRetireTheTrace) {
  SampleMetadata traced = stamped(10);
  traced.trace_id = 500U;

  // Tracing is sampled, so most contributors carry no root. Treating that as a disagreement would
  // retire the trace on nearly every activation and make the feature useless under fusion.
  ProvenanceAccumulator accumulator;
  accumulator.observe(traced);
  accumulator.observe(stamped(20));

  const auto options = accumulator.emit_options(CaptureTimePolicy::kEarliest);

  ASSERT_TRUE(options.has_value());
  ASSERT_TRUE(options->trace.has_value());
  EXPECT_EQ(options->trace->trace_id, 500U);
}

// REQ: SENSOR-IO-004
TEST(SensorIoProvenanceAccumulator, ConflictingTraceRootsLeaveTheResultUncorrelated) {
  SampleMetadata traced = stamped(10);
  traced.trace_id = 500U;
  SampleMetadata other_root = stamped(20);
  other_root.trace_id = 600U;

  ProvenanceAccumulator accumulator;
  accumulator.observe(traced);
  accumulator.observe(other_root);

  const auto options = accumulator.emit_options(CaptureTimePolicy::kEarliest);

  ASSERT_TRUE(options.has_value());
  EXPECT_FALSE(options->trace.has_value());
}

// REQ: SENSOR-IO-004
TEST(SensorIoProvenanceAccumulator, RetiredTraceIsNotReadoptedByALaterContributor) {
  SampleMetadata traced = stamped(10);
  traced.trace_id = 500U;
  SampleMetadata other_root = stamped(20);
  other_root.trace_id = 600U;
  SampleMetadata revisits_first_root = stamped(30);
  revisits_first_root.trace_id = 500U;

  ProvenanceAccumulator accumulator;
  accumulator.observe(traced);
  accumulator.observe(other_root);
  accumulator.observe(revisits_first_root);

  const auto options = accumulator.emit_options(CaptureTimePolicy::kEarliest);

  ASSERT_TRUE(options.has_value());
  EXPECT_FALSE(options->trace.has_value());
}

// REQ: SENSOR-IO-004
TEST(SensorIoProvenanceAccumulator, ObservesOwningSamples) {
  ProvenanceAccumulator accumulator;
  accumulator.observe(Sample<std::int32_t>{1, stamped(400)});
  accumulator.observe(Sample<std::int32_t>{2, stamped(200)});

  const auto options = accumulator.emit_options(CaptureTimePolicy::kEarliest);

  ASSERT_TRUE(options.has_value());
  ASSERT_TRUE(options->capture_time.has_value());
  EXPECT_EQ(options->capture_time->timestamp_ns, 200);
}

// REQ: SENSOR-IO-004
TEST(SensorIoProvenanceAccumulator, ResetClearsProvenanceBetweenActivations) {
  ProvenanceAccumulator accumulator;
  accumulator.observe(stamped(100, 0U, kCaptureDomain));
  accumulator.observe(stamped(200, 0U, kOtherDomain));
  ASSERT_TRUE(accumulator.mixed_domains());

  accumulator.reset();
  EXPECT_EQ(accumulator.observed(), 0U);
  EXPECT_FALSE(accumulator.mixed_domains());

  accumulator.observe(stamped(900));
  const auto options = accumulator.emit_options(CaptureTimePolicy::kEarliest);

  ASSERT_TRUE(options.has_value());
  ASSERT_TRUE(options->capture_time.has_value());
  EXPECT_EQ(options->capture_time->timestamp_ns, 900);
}

}  // namespace
}  // namespace holoscan::sensor_io

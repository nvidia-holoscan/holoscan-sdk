/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <semaphore>
#include <span>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/time/manual_clock.hpp>
#include "compiler/execution_plan_state.hpp"
#include "core/operator_test_binding.hpp"
#include "core/source_id_internal.hpp"
#include "runtime/dispatcher.hpp"
#include "runtime/runtime_channel_binding.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "time/clock_registry_internal.hpp"

#include "../support/optional_assertions.hpp"
#include "runtime_channel_binding_test_support.hpp"

namespace holoscan::detail {
namespace {

using namespace std::chrono_literals;

using namespace runtime_channel_binding_test;

[[nodiscard]] expected<FrozenRuntimeCandidate, Error> wait_for_frozen_channel_candidate(
    RuntimeChannelSession& session, PortBinding& binding, RuntimeCandidateCoordinator& coordinator,
    std::int64_t& observed_ns, const ClockId clock = ClockId{1U}) {
  const auto deadline = std::chrono::steady_clock::now() + 2s;
  while (std::chrono::steady_clock::now() < deadline) {
    auto polled = binding.poll(TimePoint{observed_ns++, clock});
    if (!polled) {
      return make_unexpected(std::move(polled).error());
    }
    auto ingested = coordinator.ingest(binding);
    if (!ingested) {
      return make_unexpected(std::move(ingested).error());
    }
    if (*ingested != 0U) {
      if (*ingested != 1U) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      return coordinator.freeze_next();
    }
    if (auto status = session.status(); !status) {
      return make_unexpected(std::move(status).error());
    }
    std::this_thread::sleep_for(1ms);
  }
  return make_unexpected(Error{ErrorCode::kNotReady});
}

// REQ: RUNTIME-017
// REQ: CHAN-131
// Freshness recovery is useful only when drop-oldest has bounded staging behind a pinned typed
// slot. A blocking LatestWithin input has the temporal contract but no lossy handoff to reconcile,
// so registering it would add status queries and retained-claim scans to an unrelated path.
TEST(RuntimeChannelBindingTemporal, FreshnessSupportRequiresDropOldestStaging) {
  for (const bool drop_oldest : {false, true}) {
    SCOPED_TRACE(drop_oldest ? "drop-oldest" : "blocking");
    SessionFixture fixture;
    fixture.enable_latest_within();
    if (drop_oldest) {
      fixture.use_one_slot_drop_oldest_queue();
    } else {
      fixture.use_one_slot_blocking_transport();
    }
    auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
    ASSERT_TRUE(session) << runtime_error_message(session.error());

    auto supported = (*session)->supports_freshness_refill(PortId{2U});
    ASSERT_TRUE(supported) << runtime_error_message(supported.error());
    EXPECT_EQ(*supported, drop_oldest);
  }
}

TEST(RuntimeChannelBindingTemporal, UpcomingClockFreshnessRequiresDropOldestStaging) {
  for (const bool drop_oldest : {false, true}) {
    SCOPED_TRACE(drop_oldest ? "drop-oldest" : "blocking");
    SessionFixture fixture;
    fixture.enable_latest_within();
    fixture.plan.input_selections.front().missing = TemporalRecordMissingKind::kReuseLast;
    fixture.plan.input_selections.front().reuse_max_age_ns = 100;
    if (drop_oldest) {
      fixture.use_one_slot_drop_oldest_queue();
    } else {
      fixture.use_one_slot_blocking_transport();
    }
    auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
    ASSERT_TRUE(session) << runtime_error_message(session.error());

    auto supported = (*session)->supports_drop_oldest_freshness_refill(PortId{2U});
    ASSERT_TRUE(supported) << runtime_error_message(supported.error());
    EXPECT_EQ(*supported, drop_oldest);
  }
}

// A test reference clock for capture-less drain successors: returns `now` and counts readings.
struct TestReferenceClock {
  TimePoint now;
  int readings{};

  [[nodiscard]] RuntimeChannelClockRead reader() noexcept {
    return RuntimeChannelClockRead{
        .state = this,
        .read = [](void* state) noexcept -> expected<TimePoint, Error> {
          auto& clock = *static_cast<TestReferenceClock*>(state);
          ++clock.readings;
          return clock.now;
        },
    };
  }
};

// One-slot drop-oldest DrainAll input with direct delivery, armed, holding retained sample 11.
struct CaptureAbsentDrainFixture {
  SessionFixture fixture;
  std::unique_ptr<RuntimeChannelSession> session;
  std::unique_ptr<PortBinding> binding;
  TemporalSampleRecord retained;

  void start() {
    fixture.enable_drain_all(1U);
    fixture.use_one_slot_drop_oldest_queue();
    fixture.use_direct_retained_delivery();
    auto created = RuntimeChannelSession::create(fixture.plan, fixture.replay);
    ASSERT_TRUE(created) << runtime_error_message(created.error());
    session = std::move(*created);
    auto created_binding =
        session->create_readiness_binding(VertexId{2U},
                                          RuntimePortBindingRequest{
                                              .plan_generation = fixture.plan.plan_generation,
                                              .run_generation = 1U,
                                              .kind = RuntimeEventKind::kInputArrival,
                                              .source = make_source_id(1U),
                                              .source_clock = ClockId{1U},
                                              .reference_clock = ClockId{1U},
                                              .capacity = 8U,
                                          });
    ASSERT_TRUE(created_binding) << runtime_error_message(created_binding.error());
    binding = std::move(*created_binding);
    ASSERT_TRUE(session->refresh_connections());
    ASSERT_TRUE(session->arm());
    ASSERT_TRUE(session->prepare_activation(VertexId{1U}, 1U, {}));

    ASSERT_TRUE(fixture.source.output.emit(std::int32_t{11}));
    auto retained_event = wait_for_event(*session, *binding);
    ASSERT_TRUE(retained_event) << runtime_error_message(retained_event.error());
    auto record = session->temporal_sample(retained_event->port, retained_event->source_sequence);
    ASSERT_TRUE(record) << runtime_error_message(record.error());
    retained = *record;
    ASSERT_TRUE(binding->retire(retained_event->source_sequence));
  }

  void stop() {
    ASSERT_TRUE(session->retire_activation(VertexId{1U}, 1U, true));
    session->cancel();
    ASSERT_TRUE(session->join());
  }
};

// REQ: CHAN-131
// A capture-less successor has no coordinate of its own. The reservation reads the reference
// clock for it, and qualifies it against the floor like a capture time. The slot keeps that
// coordinate: a later reservation of the same record reuses it rather than a newer reading, so the
// witness that eventually publishes the record carries the coordinate the trade was qualified
// with. Other modes decline a capture-less successor.
TEST(RuntimeChannelBindingTemporal, ReserveCaptureAbsentDrainSuccessorKeepsItsFirstReading) {
  constexpr std::int64_t kReadingNs = 20;
  struct Vector {
    std::int64_t floor_ns{};
    bool expected_reserved{};
  };
  constexpr std::array vectors{
      Vector{kReadingNs, true},
      Vector{kReadingNs - 1, false},
  };

  for (const Vector& vector : vectors) {
    SCOPED_TRACE(vector.floor_ns);
    CaptureAbsentDrainFixture drain;
    ASSERT_NO_FATAL_FAILURE(drain.start());
    // The sole typed slot is claimed, so the capture-less successor can only stage.
    ASSERT_TRUE(drain.fixture.source.output.emit(std::int32_t{22}));

    const TimePoint floor{vector.floor_ns, ClockId{1U}};
    // Only a drain gives a capture-less successor a coordinate; another mode declines it.
    auto declined = drain.session->reserve_freshness_refill(PortId{2U}, drain.retained, floor);
    ASSERT_TRUE(declined) << runtime_error_message(declined.error());
    EXPECT_FALSE(declined->has_value());

    TestReferenceClock clock{.now = TimePoint{kReadingNs, ClockId{1U}}};
    auto reserved =
        drain.session->reserve_freshness_refill(PortId{2U},
                                                drain.retained,
                                                floor,
                                                RuntimeChannelFreshnessSelectionMode::kDrainCut,
                                                clock.reader());
    ASSERT_TRUE(reserved) << runtime_error_message(reserved.error());
    const auto& optional_reserved = *reserved;
    ASSERT_EQ(optional_reserved.has_value(), vector.expected_reserved);
    EXPECT_EQ(clock.readings, 1);
    if (optional_reserved.has_value()) {
      EXPECT_EQ(optional_reserved->successor_reference, (TimePoint{kReadingNs, ClockId{1U}}));
      drain.session->release_freshness_reservation(*optional_reserved);
      // Key assertion: a later, newer reading does not replace the coordinate already fixed.
      clock.now = TimePoint{kReadingNs + 50, ClockId{1U}};
      auto again =
          drain.session->reserve_freshness_refill(PortId{2U},
                                                  drain.retained,
                                                  TimePoint{kReadingNs + 100, ClockId{1U}},
                                                  RuntimeChannelFreshnessSelectionMode::kDrainCut,
                                                  clock.reader());
      ASSERT_TRUE(again) << runtime_error_message(again.error());
      const auto& optional_again = *again;
      HOLOSCAN_ASSERT_HAS_VALUE(optional_again) << "Expected freshness reservation";
      EXPECT_EQ(optional_again->successor_reference, (TimePoint{kReadingNs, ClockId{1U}}));
      EXPECT_EQ(clock.readings, 1);
      drain.session->release_freshness_reservation(*optional_again);
    }
    drain.stop();
  }
}

// REQ: CHAN-131
// A drain cut can hold a capture-less record ahead of a captured target whose capture time is older
// than that record. The capture-less member takes the clock reading made after it was stored,
// never the target's coordinate. It keeps that reading when its reservation is released and taken
// again at a later reading, as a deferred handoff is, and its witness publishes it. The target
// keeps its own capture coordinate through the next handoff of the committed cut.
TEST(RuntimeChannelBindingTemporal, CaptureAbsentDrainCutMemberKeepsItsOwnReading) {
  constexpr std::int64_t kReadingNs = 100;
  constexpr std::int64_t kTargetCaptureNs = 40;
  SessionFixture fixture;
  fixture.enable_drain_all(2U);
  fixture.use_two_slot_drop_oldest_queue();
  fixture.use_direct_retained_delivery();
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto binding =
      (*session)->create_readiness_binding(VertexId{2U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.plan.plan_generation,
                                               .run_generation = 1U,
                                               .kind = RuntimeEventKind::kInputArrival,
                                               .source = make_source_id(1U),
                                               .source_clock = ClockId{1U},
                                               .reference_clock = ClockId{1U},
                                               .capacity = 8U,
                                           });
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));

  // Two retained records fill both typed slots, so the next arrivals stage.
  std::vector<TemporalSampleRecord> retained;
  for (const std::int32_t value : {11, 12}) {
    ASSERT_TRUE(fixture.source.output.emit(value));
    auto event = wait_for_event(**session, **binding);
    ASSERT_TRUE(event) << runtime_error_message(event.error());
    auto record = (*session)->temporal_sample(event->port, event->source_sequence);
    ASSERT_TRUE(record) << runtime_error_message(record.error());
    retained.push_back(*record);
    ASSERT_TRUE((*binding)->retire(event->source_sequence));
  }
  // Producer order C then D: C has no capture time, and D was captured before C's reading.
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{21}));
  ASSERT_TRUE(fixture.source.output.emit(Sample<std::int32_t>{
      22,
      SampleMetadata{.capture_timestamp_ns = kTargetCaptureNs, .source_clock_id = ClockId{1U}}}));

  const TimePoint floor{200, ClockId{1U}};
  const TimePoint reading{kReadingNs, ClockId{1U}};
  const TimePoint target_capture{kTargetCaptureNs, ClockId{1U}};
  TestReferenceClock clock{.now = reading};
  const auto reserve = [&](const TemporalSampleRecord& record) {
    return (*session)->reserve_freshness_refill(
        PortId{2U}, record, floor, RuntimeChannelFreshnessSelectionMode::kDrainCut, clock.reader());
  };
  // Poll after the reading, as the dispatcher does at the current time. A poll that predates a
  // fixed coordinate publishes the poll time instead.
  const auto poll_event = [&]() -> expected<RuntimeEvent, Error> {
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    while (std::chrono::steady_clock::now() < deadline) {
      auto polled = (*binding)->poll(TimePoint{kReadingNs + 60, ClockId{1U}});
      if (!polled) {
        return make_unexpected(std::move(polled).error());
      }
      RuntimeEvent event;
      auto drained = (*binding)->drain(std::span<RuntimeEvent>{&event, 1U});
      if (!drained) {
        return make_unexpected(std::move(drained).error());
      }
      if (*drained == 1U) {
        return event;
      }
      std::this_thread::sleep_for(1ms);
    }
    return make_unexpected(Error{ErrorCode::kNotReady});
  };
  auto first = reserve(retained[0]);
  ASSERT_TRUE(first) << runtime_error_message(first.error());
  const auto& optional_first = *first;
  HOLOSCAN_ASSERT_HAS_VALUE(optional_first) << "Expected freshness reservation";
  EXPECT_EQ(optional_first->successor.source_sequence, retained[0].source_sequence + 2U);
  EXPECT_EQ(optional_first->target.source_sequence, retained[0].source_sequence + 3U);
  // Key assertion: C takes the reading made after it was stored, not D's older capture time.
  EXPECT_EQ(optional_first->successor_reference, reading);
  EXPECT_EQ(optional_first->target_reference, target_capture);
  EXPECT_EQ(clock.readings, 1);

  (*session)->release_freshness_reservation(*optional_first);
  clock.now = TimePoint{kReadingNs + 50, ClockId{1U}};
  auto deferred = reserve(retained[0]);
  ASSERT_TRUE(deferred) << runtime_error_message(deferred.error());
  const auto& optional_deferred = *deferred;
  HOLOSCAN_ASSERT_HAS_VALUE(optional_deferred) << "Expected freshness reservation";
  EXPECT_EQ(optional_deferred->successor_reference, reading);
  EXPECT_EQ(clock.readings, 1);

  std::uint32_t promoted = 0U;
  ASSERT_TRUE((*session)->retire_samples(PortId{2U},
                                         std::span<const TemporalSampleRecord>{retained.data(), 1U},
                                         &promoted,
                                         &*optional_deferred));
  ASSERT_EQ(promoted, 1U);
  auto member_event = poll_event();
  ASSERT_TRUE(member_event) << runtime_error_message(member_event.error());
  EXPECT_EQ(member_event->reference_time, reading);
  ASSERT_TRUE((*binding)->retire(member_event->source_sequence));

  // The committed cut hands out its target next, with the target's own coordinate.
  auto last = reserve(retained[1]);
  ASSERT_TRUE(last) << runtime_error_message(last.error());
  const auto& optional_last = *last;
  HOLOSCAN_ASSERT_HAS_VALUE(optional_last) << "Expected freshness reservation";
  EXPECT_EQ(optional_last->successor.source_sequence, retained[0].source_sequence + 3U);
  EXPECT_EQ(optional_last->successor_reference, target_capture);
  ASSERT_TRUE((*session)->retire_samples(PortId{2U},
                                         std::span<const TemporalSampleRecord>{&retained[1], 1U},
                                         &promoted,
                                         &*optional_last));
  ASSERT_EQ(promoted, 1U);
  auto target_event = poll_event();
  ASSERT_TRUE(target_event) << runtime_error_message(target_event.error());
  EXPECT_EQ(target_event->reference_time, target_capture);
  ASSERT_TRUE((*binding)->retire(target_event->source_sequence));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// REQ: CHAN-131
// A drain cut is a producer-order prefix. When the older of two staged records was captured after
// the trigger and the newer one before it, the cut ends before the older record instead of
// admitting the newer one ahead of it. Once both qualify, the cut starts at the older record.
TEST(RuntimeChannelBindingTemporal, DrainCutEndsBeforeTheOldestRecordAboveTheFloor) {
  SessionFixture fixture;
  fixture.enable_drain_all(2U);
  fixture.use_two_slot_drop_oldest_queue();
  fixture.use_direct_retained_delivery();
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto binding =
      (*session)->create_readiness_binding(VertexId{2U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.plan.plan_generation,
                                               .run_generation = 1U,
                                               .kind = RuntimeEventKind::kInputArrival,
                                               .source = make_source_id(1U),
                                               .source_clock = ClockId{1U},
                                               .reference_clock = ClockId{1U},
                                               .capacity = 8U,
                                           });
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));

  // Two retained records fill both typed slots, so the next arrivals stage.
  TemporalSampleRecord retained;
  for (const std::int32_t value : {11, 12}) {
    ASSERT_TRUE(fixture.source.output.emit(value));
    auto event = wait_for_event(**session, **binding);
    ASSERT_TRUE(event) << runtime_error_message(event.error());
    auto record = (*session)->temporal_sample(event->port, event->source_sequence);
    ASSERT_TRUE(record) << runtime_error_message(record.error());
    if (value == 11) {
      retained = *record;
    }
    ASSERT_TRUE((*binding)->retire(event->source_sequence));
  }
  // Producer order B then C, captured out of order: B after the trigger at 20, C before it.
  ASSERT_TRUE(fixture.source.output.emit(Sample<std::int32_t>{
      21, SampleMetadata{.capture_timestamp_ns = 30, .source_clock_id = ClockId{1U}}}));
  ASSERT_TRUE(fixture.source.output.emit(Sample<std::int32_t>{
      22, SampleMetadata{.capture_timestamp_ns = 10, .source_clock_id = ClockId{1U}}}));

  auto declined =
      (*session)->reserve_freshness_refill(PortId{2U},
                                           retained,
                                           TimePoint{20, ClockId{1U}},
                                           RuntimeChannelFreshnessSelectionMode::kDrainCut);
  ASSERT_TRUE(declined) << runtime_error_message(declined.error());
  // Key assertion: C does not enter ahead of B, the older record that cannot qualify yet.
  EXPECT_FALSE(declined->has_value());

  auto reserved =
      (*session)->reserve_freshness_refill(PortId{2U},
                                           retained,
                                           TimePoint{40, ClockId{1U}},
                                           RuntimeChannelFreshnessSelectionMode::kDrainCut);
  ASSERT_TRUE(reserved) << runtime_error_message(reserved.error());
  const auto& optional_reserved = *reserved;
  HOLOSCAN_ASSERT_HAS_VALUE(optional_reserved) << "Expected freshness reservation";
  EXPECT_EQ(optional_reserved->successor.source_sequence, retained.source_sequence + 2U);
  EXPECT_EQ(optional_reserved->target.source_sequence, retained.source_sequence + 3U);
  (*session)->release_freshness_reservation(*optional_reserved);
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// REQ: CHAN-131
// The dispatcher decides to trade before it reserves, and a fast producer can replace the staged
// successor in between. The replacement did not exist when the dispatcher decided, so it must not
// qualify for a trigger that precedes it. The reservation reads the clock itself, under the mutex
// that orders stores, so the replacement is qualified by a reading taken after it was stored.
TEST(RuntimeChannelBindingTemporal, CaptureAbsentDrainReplacementIsQualifiedAfterItWasStored) {
  constexpr std::int64_t kTriggerNs = 200;
  CaptureAbsentDrainFixture drain;
  ASSERT_NO_FATAL_FAILURE(drain.start());
  TestReferenceClock clock{.now = TimePoint{150, ClockId{1U}}};
  ASSERT_TRUE(drain.fixture.source.output.emit(std::int32_t{22}));  // staged before the trigger
  // The dispatcher would qualify sample 22 for the trigger at 200 now. Before it reserves, the
  // producer replaces 22 with 33 after the trigger time.
  clock.now = TimePoint{250, ClockId{1U}};
  ASSERT_TRUE(drain.fixture.source.output.emit(std::int32_t{33}));

  const TimePoint floor{kTriggerNs, ClockId{1U}};
  auto declined =
      drain.session->reserve_freshness_refill(PortId{2U},
                                              drain.retained,
                                              floor,
                                              RuntimeChannelFreshnessSelectionMode::kDrainCut,
                                              clock.reader());
  ASSERT_TRUE(declined) << runtime_error_message(declined.error());
  // Key assertion: the replacement does not qualify for the earlier trigger, so the retained
  // sample is not traded for it.
  EXPECT_FALSE(declined->has_value());
  EXPECT_EQ(clock.readings, 1);

  // A later trigger admits the replacement with the coordinate read after it was stored.
  auto reserved =
      drain.session->reserve_freshness_refill(PortId{2U},
                                              drain.retained,
                                              TimePoint{300, ClockId{1U}},
                                              RuntimeChannelFreshnessSelectionMode::kDrainCut,
                                              clock.reader());
  ASSERT_TRUE(reserved) << runtime_error_message(reserved.error());
  const auto& optional_reserved = *reserved;
  HOLOSCAN_ASSERT_HAS_VALUE(optional_reserved) << "Expected freshness reservation";
  EXPECT_EQ(optional_reserved->successor_reference, (TimePoint{250, ClockId{1U}}));
  EXPECT_GT(optional_reserved->successor.source_sequence, drain.retained.source_sequence + 1U);
  drain.session->release_freshness_reservation(*optional_reserved);
  drain.stop();
}

// REQ: RUNTIME-017
// REQ: CHAN-131
// The causality qualification, stated as the reconcile depends on it: a staged successor captured
// after the coordinate of a witness this round can freeze must not be admitted in place of a
// record that witness could already have selected. Evicting for it would strand that witness --
// the successor sorts behind it in freeze_next's reference-time ordering, so the activation reads
// a cache the eviction emptied and selects nothing.
//
// This is the deterministic form of the end-to-end HighRateSource case. That one can only observe
// the outcome at a producer fast enough to lose the race; this one fixes both coordinates.
TEST(RuntimeChannelBindingTemporal, ReserveDeclinesASuccessorCapturedAfterTheWitness) {
  constexpr std::int64_t kRetainedCaptureNs = 10;
  constexpr std::int64_t kSuccessorCaptureNs = 20;

  // The witness coordinate is the only variable: at the successor's own capture the successor is
  // causal to it, and one nanosecond earlier it is not.
  struct Vector {
    std::int64_t witness_reference_ns{};
    bool expected_reserved{};
  };
  constexpr std::array vectors{
      Vector{kSuccessorCaptureNs, true},
      Vector{kSuccessorCaptureNs - 1, false},
  };

  for (const Vector& vector : vectors) {
    SCOPED_TRACE(vector.witness_reference_ns);
    SessionFixture fixture;
    fixture.enable_scalar(InputSelectorKind::kLatestWithin, TemporalRecordMissingKind::kSkip);
    fixture.use_one_slot_drop_oldest_queue();
    fixture.use_direct_retained_delivery();

    auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
    ASSERT_TRUE(session) << runtime_error_message(session.error());
    auto binding =
        (*session)->create_readiness_binding(VertexId{2U},
                                             RuntimePortBindingRequest{
                                                 .plan_generation = fixture.plan.plan_generation,
                                                 .run_generation = 1U,
                                                 .kind = RuntimeEventKind::kInputArrival,
                                                 .source = make_source_id(1U),
                                                 .source_clock = ClockId{1U},
                                                 .reference_clock = ClockId{1U},
                                                 .capacity = 8U,
                                             });
    ASSERT_TRUE(binding) << runtime_error_message(binding.error());
    ASSERT_TRUE((*session)->refresh_connections());
    ASSERT_TRUE((*session)->arm());
    ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));

    ASSERT_TRUE(fixture.source.output.emit(
        Sample<std::int32_t>{11,
                             SampleMetadata{
                                 .capture_timestamp_ns = kRetainedCaptureNs,
                                 .source_clock_id = ClockId{1U},
                             }}));
    auto retained_event = wait_for_event(**session, **binding);
    ASSERT_TRUE(retained_event) << runtime_error_message(retained_event.error());
    auto retained =
        (*session)->temporal_sample(retained_event->port, retained_event->source_sequence);
    ASSERT_TRUE(retained) << runtime_error_message(retained.error());

    // Match the dispatcher state this unit-level reservation normally observes. The selector
    // still owns the payload claim, but its readiness witness has already been ingested and
    // retired, so releasing the temporal claim can make the typed slot reclaimable.
    ASSERT_TRUE((*binding)->retire(retained_event->source_sequence));

    // The sole typed slot is now claimed, so the successor can only stage.
    ASSERT_TRUE(fixture.source.output.emit(
        Sample<std::int32_t>{22,
                             SampleMetadata{
                                 .capture_timestamp_ns = kSuccessorCaptureNs,
                                 .source_clock_id = ClockId{1U},
                             }}));

    auto reserved = (*session)->reserve_freshness_refill(
        PortId{2U}, *retained, TimePoint{vector.witness_reference_ns, ClockId{1U}});
    ASSERT_TRUE(reserved) << runtime_error_message(reserved.error());
    const auto& optional_reserved = *reserved;
    EXPECT_EQ(optional_reserved.has_value(), vector.expected_reserved);

    if (optional_reserved.has_value()) {
      (*session)->release_freshness_reservation(*optional_reserved);
    }
    ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
    (*session)->cancel();
    ASSERT_TRUE((*session)->join());
  }
}

// REQ: CHAN-136
// REQ: RUNTIME-050
TEST(RuntimeChannelBindingTemporal, DirectAffineReuseLastReservationUsesReferenceCoordinates) {
  SessionFixture fixture;
  fixture.require_direct_capture_conversion();
  fixture.plan.input_selections.front().missing = TemporalRecordMissingKind::kReuseLast;
  fixture.plan.input_selections.front().reuse_max_age_ns = 1000;
  fixture.use_one_slot_drop_oldest_queue();
  fixture.use_direct_retained_delivery();
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto binding = (*session)->create_readiness_binding(
      VertexId{2U},
      RuntimePortBindingRequest{.plan_generation = fixture.plan.plan_generation,
                                .run_generation = 1U,
                                .kind = RuntimeEventKind::kInputArrival,
                                .source = make_source_id(1U),
                                .source_clock = ClockId{1U},
                                .reference_clock = ClockId{2U},
                                .source_to_reference_conversion_id = 1U,
                                .capacity = 8U});
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(Sample<std::int32_t>{
      11, SampleMetadata{.capture_timestamp_ns = 10, .source_clock_id = ClockId{1U}}}));
  auto event = wait_for_event(**session, **binding, ClockId{2U});
  ASSERT_TRUE(event) << runtime_error_message(event.error());
  auto retained = (*session)->temporal_sample(event->port, event->source_sequence);
  ASSERT_TRUE(retained) << runtime_error_message(retained.error());
  ASSERT_TRUE((*binding)->retire(event->source_sequence));
  ASSERT_TRUE(fixture.source.output.emit(Sample<std::int32_t>{
      22, SampleMetadata{.capture_timestamp_ns = 20, .source_clock_id = ClockId{1U}}}));

  auto too_early =
      (*session)->reserve_freshness_refill(PortId{2U}, *retained, TimePoint{42, ClockId{2U}});
  ASSERT_TRUE(too_early) << runtime_error_message(too_early.error());
  EXPECT_FALSE(too_early->has_value());
  auto reserved =
      (*session)->reserve_freshness_refill(PortId{2U}, *retained, TimePoint{43, ClockId{2U}});
  ASSERT_TRUE(reserved) << runtime_error_message(reserved.error());
  const auto& optional_reserved = *reserved;
  HOLOSCAN_ASSERT_HAS_VALUE(optional_reserved) << "Expected freshness reservation";
  EXPECT_EQ(optional_reserved->successor.capture_time, (TimePoint{20, ClockId{1U}}));
  EXPECT_EQ(optional_reserved->successor_reference, (TimePoint{43, ClockId{2U}}));
  auto selector = RuntimeInputSelectionState::create(fixture.plan.input_selections.front());
  ASSERT_TRUE(selector) << runtime_error_message(selector.error());
  ASSERT_TRUE(selector->push(*retained));
  EXPECT_FALSE(selector->staged_reuse_eligible(optional_reserved->successor.capture_time,
                                               TimePoint{43, ClockId{2U}}));
  EXPECT_TRUE(selector->staged_reuse_eligible(optional_reserved->successor_reference,
                                              TimePoint{43, ClockId{2U}}));
  ASSERT_TRUE(selector->evict_retained(retained->arrival_sequence));
  std::uint32_t promoted = 0U;
  ASSERT_TRUE((*session)->retire_samples(PortId{2U},
                                         std::span<const TemporalSampleRecord>{&*retained, 1U},
                                         &promoted,
                                         &*optional_reserved));
  ASSERT_EQ(promoted, 1U);
  auto promoted_event = wait_for_event(**session, **binding, ClockId{2U});
  ASSERT_TRUE(promoted_event) << runtime_error_message(promoted_event.error());
  EXPECT_EQ(promoted_event->reference_time, optional_reserved->successor_reference);
  auto promoted_sample =
      (*session)->temporal_sample(promoted_event->port, promoted_event->source_sequence);
  ASSERT_TRUE(promoted_sample) << runtime_error_message(promoted_sample.error());
  TemporalSampleRecord normalized_successor = optional_reserved->successor;
  normalized_successor.capture_time = optional_reserved->successor_reference;
  EXPECT_EQ(*promoted_sample, normalized_successor);
  ASSERT_TRUE(selector->push(*promoted_sample));
  ASSERT_TRUE((*binding)->retire(promoted_event->source_sequence));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// REQ: RUNTIME-017
// REQ: CHAN-131
// The invariant the reconcile owes its caller: after a pass in which it evicts, the selector cache
// is non-empty whenever it was non-empty before. Releasing a retained record is only ever a trade,
// and a trade that hands back nothing converts a bounded staleness into a lost activation, which
// is strictly worse and is the regression this branch had to clear.
//
// Both halves are asserted, because the invariant holds two ways. When a successor qualifies, the
// same pass admits it. When none qualifies, the reservation refuses and the caller never evicts,
// so there is nothing to hand back.
TEST(RuntimeChannelBindingTemporal, EvictionRefillsTheCacheItEmptiedWithinTheSamePass) {
  constexpr std::int64_t kRetainedCaptureNs = 10;
  constexpr std::int64_t kSuccessorCaptureNs = 20;
  constexpr std::int64_t kLaterSuccessorCaptureNs = 30;

  for (const bool successor_qualifies : {true, false}) {
    SCOPED_TRACE(successor_qualifies ? "successor qualifies" : "no successor qualifies");
    SessionFixture fixture;
    fixture.enable_scalar(InputSelectorKind::kLatestWithin, TemporalRecordMissingKind::kSkip);
    fixture.use_one_slot_drop_oldest_queue();
    fixture.use_direct_retained_delivery();

    auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
    ASSERT_TRUE(session) << runtime_error_message(session.error());
    auto binding =
        (*session)->create_readiness_binding(VertexId{2U},
                                             RuntimePortBindingRequest{
                                                 .plan_generation = fixture.plan.plan_generation,
                                                 .run_generation = 1U,
                                                 .kind = RuntimeEventKind::kInputArrival,
                                                 .source = make_source_id(1U),
                                                 .source_clock = ClockId{1U},
                                                 .reference_clock = ClockId{1U},
                                                 .capacity = 8U,
                                             });
    ASSERT_TRUE(binding) << runtime_error_message(binding.error());
    ASSERT_TRUE((*session)->refresh_connections());
    ASSERT_TRUE((*session)->arm());
    ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));

    ASSERT_TRUE(fixture.source.output.emit(
        Sample<std::int32_t>{11,
                             SampleMetadata{
                                 .capture_timestamp_ns = kRetainedCaptureNs,
                                 .source_clock_id = ClockId{1U},
                             }}));
    auto retained_event = wait_for_event(**session, **binding);
    ASSERT_TRUE(retained_event) << runtime_error_message(retained_event.error());
    auto retained =
        (*session)->temporal_sample(retained_event->port, retained_event->source_sequence);
    ASSERT_TRUE(retained) << runtime_error_message(retained.error());

    // Retire the readiness witness the way the dispatcher does once it has ingested the candidate.
    // The slot stays occupied and protected regardless, because the selector still holds its
    // payload claim -- and that claim is precisely what the reconcile trades away below.
    ASSERT_TRUE((*binding)->retire(retained_event->source_sequence));

    ASSERT_TRUE(fixture.source.output.emit(
        Sample<std::int32_t>{22,
                             SampleMetadata{
                                 .capture_timestamp_ns = kSuccessorCaptureNs,
                                 .source_clock_id = ClockId{1U},
                             }}));

    // A floor at the successor's capture qualifies it; one below it disqualifies every staged
    // record, which is the "nothing to trade for" half of the invariant.
    const TimePoint floor{successor_qualifies ? kSuccessorCaptureNs : kSuccessorCaptureNs - 1,
                          ClockId{1U}};
    auto reserved = (*session)->reserve_freshness_refill(
        PortId{2U},
        *retained,
        floor,
        RuntimeChannelFreshnessSelectionMode::kNewestWithArrivalPrefix);
    ASSERT_TRUE(reserved) << runtime_error_message(reserved.error());
    ASSERT_EQ(reserved->has_value(), successor_qualifies);

    if (!successor_qualifies) {
      // No eviction happens, so the record the witness could already have selected is untouched.
      auto still_held =
          (*session)->temporal_sample(retained_event->port, retained_event->source_sequence);
      ASSERT_TRUE(still_held) << runtime_error_message(still_held.error());
      EXPECT_EQ(still_held->source_sequence, retained->source_sequence);
      ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
      (*session)->cancel();
      ASSERT_TRUE((*session)->join());
      continue;
    }

    // Stage a second, later successor. A reserved record does not count against the authored
    // staging depth, so both are now staged. Re-reserving must keep the oldest successor: bypassing
    // it would publish readiness out of arrival order and violate the temporal cache watermark.
    ASSERT_TRUE(fixture.source.output.emit(
        Sample<std::int32_t>{33,
                             SampleMetadata{
                                 .capture_timestamp_ns = kLaterSuccessorCaptureNs,
                                 .source_clock_id = ClockId{1U},
                             }}));
    auto requalified = (*session)->reserve_freshness_refill(
        PortId{2U},
        *retained,
        TimePoint{kLaterSuccessorCaptureNs, ClockId{1U}},
        RuntimeChannelFreshnessSelectionMode::kNewestWithArrivalPrefix);
    ASSERT_TRUE(requalified) << runtime_error_message(requalified.error());
    const auto& optional_requalified = *requalified;
    HOLOSCAN_ASSERT_HAS_VALUE(optional_requalified) << "Expected freshness reservation";

    // Key assertion: the release that empties the cache admits the reserved record in the same
    // call, rather than leaving the refill to whatever the pipeline does afterwards.
    std::uint32_t promoted = 0U;
    ASSERT_TRUE((*session)->retire_samples(PortId{2U},
                                           std::span<const TemporalSampleRecord>{&*retained, 1U},
                                           &promoted,
                                           &*optional_requalified));
    EXPECT_EQ(promoted, 1U);

    // And the record admitted is the oldest qualified successor, not the later staged arrival.
    auto refilled_event = wait_for_event(**session, **binding);
    ASSERT_TRUE(refilled_event) << runtime_error_message(refilled_event.error());
    auto refilled =
        (*session)->temporal_sample(refilled_event->port, refilled_event->source_sequence);
    ASSERT_TRUE(refilled) << runtime_error_message(refilled.error());
    EXPECT_EQ(refilled->capture_time.timestamp_ns, kSuccessorCaptureNs);
    EXPECT_GT(refilled->source_sequence, retained->source_sequence);

    ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
    (*session)->cancel();
    ASSERT_TRUE((*session)->join());
  }
}

// Arrange the state the freshness reconcile operates on, which every case below needs and none of
// them is about: a depth-one drop-oldest direct-retained input holding one published record whose
// payload claim has transferred to the selector and whose readiness witness the dispatcher has
// already retired. The sole typed slot is then claimed, so anything published afterwards can only
// stage, which is the condition the reconcile inspects.
struct RetainedRefillArrangement {
  SessionFixture fixture;
  std::unique_ptr<RuntimeChannelSession> session;
  std::unique_ptr<PortBinding> binding;
  TemporalSampleRecord retained{};

  [[nodiscard]] bool emit(std::int32_t value, std::int64_t capture_ns) {
    return fixture.source.output
        .emit(Sample<std::int32_t>{value,
                                   SampleMetadata{
                                       .capture_timestamp_ns = capture_ns,
                                       .source_clock_id = ClockId{1U},
                                   }})
        .has_value();
  }
};

struct FreshnessTypedIngressBarrier {
  static void observe(void* const opaque) noexcept {
    auto& state = *static_cast<FreshnessTypedIngressBarrier*>(opaque);
    state.calls.fetch_add(1U, std::memory_order_relaxed);
    if (!state.block.load(std::memory_order_acquire)) {
      return;
    }
    state.entered.store(true, std::memory_order_release);
    while (!state.release.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
  }

  std::atomic<std::uint32_t> calls{};
  std::atomic<bool> block{};
  std::atomic<bool> entered{};
  std::atomic<bool> release{};
};

void arrange_retained_refill(RetainedRefillArrangement& arrangement, std::int64_t retained_capture,
                             FreshnessTypedIngressBarrier* const ingress_barrier = nullptr,
                             const bool transported = false) {
  arrangement.fixture.enable_scalar(InputSelectorKind::kLatestWithin,
                                    TemporalRecordMissingKind::kSkip);
  arrangement.fixture.use_one_slot_drop_oldest_queue();
  if (!transported) {
    arrangement.fixture.use_direct_retained_delivery();
  }

  auto session =
      RuntimeChannelSession::create(arrangement.fixture.plan, arrangement.fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  arrangement.session = std::move(*session);
  if (ingress_barrier != nullptr) {
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_typed_ingress_observer(
        *arrangement.session, PortId{2U}, ingress_barrier, &FreshnessTypedIngressBarrier::observe));
  }
  auto binding = arrangement.session->create_readiness_binding(
      VertexId{2U},
      RuntimePortBindingRequest{
          .plan_generation = arrangement.fixture.plan.plan_generation,
          .run_generation = 1U,
          .kind = RuntimeEventKind::kInputArrival,
          .source = make_source_id(1U),
          .source_clock = ClockId{1U},
          .reference_clock = ClockId{1U},
          .capacity = 8U,
      });
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());
  arrangement.binding = std::move(*binding);
  ASSERT_TRUE(arrangement.session->refresh_connections());
  ASSERT_TRUE(arrangement.session->arm());
  ASSERT_TRUE(arrangement.session->prepare_activation(VertexId{1U}, 1U, {}));

  ASSERT_TRUE(arrangement.emit(11, retained_capture));
  auto event = wait_for_event(*arrangement.session, *arrangement.binding);
  ASSERT_TRUE(event) << runtime_error_message(event.error());
  auto record = arrangement.session->temporal_sample(event->port, event->source_sequence);
  ASSERT_TRUE(record) << runtime_error_message(record.error());
  arrangement.retained = *record;
  // The dispatcher retires the readiness witness once it has ingested the candidate. The slot
  // stays occupied because the selector still holds the payload claim.
  ASSERT_TRUE(arrangement.binding->retire(event->source_sequence));
}

// REQ: RUNTIME-017
// REQ: CHAN-131
// A transported receive can reserve a staging slot after a freshness successor is reserved but
// before the selector releases its retained typed claim. That in-flight staging reservation must
// not hide the typed capacity the exact retained slot releases for the reserved successor.
TEST(RuntimeChannelBindingTemporal,
     ReservedRefillUsesReleasedTypedCapacityWhileNextReceiveIsStaging) {
  constexpr std::int64_t kRetainedCaptureNs = 10;
  constexpr std::int64_t kReservedCaptureNs = 20;
  constexpr std::int64_t kFollowingCaptureNs = 30;

  FreshnessTypedIngressBarrier ingress_barrier;
  RetainedRefillArrangement arrangement;
  ASSERT_NO_FATAL_FAILURE(
      arrange_retained_refill(arrangement, kRetainedCaptureNs, &ingress_barrier, true));
  ASSERT_TRUE(arrangement.emit(22, kReservedCaptureNs));
  bool successor_staged = false;
  const auto successor_deadline = std::chrono::steady_clock::now() + 2s;
  while (!successor_staged && std::chrono::steady_clock::now() < successor_deadline) {
    auto available =
        arrangement.session->has_freshness_refill_candidate(PortId{2U}, arrangement.retained);
    ASSERT_TRUE(available) << runtime_error_message(available.error());
    successor_staged = *available;
    if (!successor_staged) {
      std::this_thread::yield();
    }
  }
  ASSERT_TRUE(successor_staged);
  auto reserved = arrangement.session->reserve_freshness_refill(
      PortId{2U}, arrangement.retained, TimePoint{kReservedCaptureNs, ClockId{1U}});
  ASSERT_TRUE(reserved) << runtime_error_message(reserved.error());
  const auto& optional_reserved = *reserved;
  HOLOSCAN_ASSERT_HAS_VALUE(optional_reserved) << "Expected freshness reservation";

  ingress_barrier.block.store(true, std::memory_order_release);
  bool following_emitted = false;
  std::thread producer{[&] { following_emitted = arrangement.emit(33, kFollowingCaptureNs); }};
  auto producer_cleanup = ScopeExit{[&]() noexcept {
    ingress_barrier.release.store(true, std::memory_order_release);
    if (producer.joinable()) {
      producer.join();
    }
  }};
  const auto deadline = std::chrono::steady_clock::now() + 2s;
  while (!ingress_barrier.entered.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  ASSERT_TRUE(ingress_barrier.entered.load(std::memory_order_acquire));
  const auto in_flight =
      RuntimeChannelSessionTestAccess::typed_ingress_reservations(*arrangement.session, PortId{2U});
  ASSERT_TRUE(in_flight) << runtime_error_message(in_flight.error());
  ASSERT_EQ(*in_flight, 1U);

  std::uint32_t promoted = 0U;
  ASSERT_TRUE(arrangement.session->retire_samples(
      PortId{2U},
      std::span<const TemporalSampleRecord>{&arrangement.retained, 1U},
      &promoted,
      &*optional_reserved));
  ASSERT_EQ(promoted, 1U);

  ingress_barrier.release.store(true, std::memory_order_release);
  producer.join();
  ASSERT_TRUE(following_emitted);
  auto refilled_event = wait_for_event(*arrangement.session, *arrangement.binding);
  ASSERT_TRUE(refilled_event) << runtime_error_message(refilled_event.error());
  auto refilled =
      arrangement.session->temporal_sample(refilled_event->port, refilled_event->source_sequence);
  ASSERT_TRUE(refilled) << runtime_error_message(refilled.error());
  EXPECT_EQ(refilled->capture_time.timestamp_ns, kReservedCaptureNs);

  ASSERT_TRUE(arrangement.binding->retire(refilled_event->source_sequence));
  ASSERT_TRUE(arrangement.session->retire_activation(VertexId{1U}, 1U, true));
  arrangement.session->cancel();
  ASSERT_TRUE(arrangement.session->join());
}

// REQ: RUNTIME-017
// A reservation is useful only if the controller can make the exact retained slot writable.
// An active selection remains observable and must decline. The exact ReuseLast slot is instead the
// supported two-owner handoff: reserving it is reversible and does not mutate either claim until
// the selector and Channel commit together.
TEST(RuntimeChannelBindingTemporal, FreshnessReservationPreflightsEverySurvivingTypedClaim) {
  struct Vector {
    bool active_selected{};
    bool reuse_retained{};
    bool reservation_expected{};
  };
  constexpr std::array vectors{
      Vector{.active_selected = true},
      Vector{.reuse_retained = true, .reservation_expected = true},
  };

  for (const Vector& vector : vectors) {
    SCOPED_TRACE(vector.active_selected ? "active selection" : "finite reuse");
    RetainedRefillArrangement arrangement;
    ASSERT_NO_FATAL_FAILURE(arrange_retained_refill(arrangement, 10));
    ASSERT_TRUE(arrangement.emit(22, 20));
    ASSERT_TRUE(
        RuntimeChannelSessionTestAccess::set_single_occupied_claim_state(*arrangement.session,
                                                                         PortId{2U},
                                                                         vector.active_selected,
                                                                         vector.reuse_retained,
                                                                         true,
                                                                         false));

    auto available =
        arrangement.session->has_freshness_refill_candidate(PortId{2U}, arrangement.retained);
    ASSERT_TRUE(available) << runtime_error_message(available.error());
    EXPECT_EQ(*available, vector.reservation_expected);
    auto reservation = arrangement.session->reserve_freshness_refill(
        PortId{2U}, arrangement.retained, TimePoint{20, ClockId{1U}});
    ASSERT_TRUE(reservation) << runtime_error_message(reservation.error());
    const auto& optional_reservation = *reservation;
    EXPECT_EQ(optional_reservation.has_value(), vector.reservation_expected);
    if (optional_reservation.has_value()) {
      arrangement.session->release_freshness_reservation(*optional_reservation);
      auto available_after_release =
          arrangement.session->has_freshness_refill_candidate(PortId{2U}, arrangement.retained);
      ASSERT_TRUE(available_after_release)
          << runtime_error_message(available_after_release.error());
      EXPECT_TRUE(*available_after_release);
    }
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_single_occupied_claim_state(
        *arrangement.session, PortId{2U}, false, false, true, false));
    ASSERT_TRUE(arrangement.session->retire_activation(VertexId{1U}, 1U, true));
    arrangement.session->cancel();
    ASSERT_TRUE(arrangement.session->join());
  }
}

// REQ: RUNTIME-017
// Authored Graphs currently reject input fan-in, so a source-tree alias injects the
// multi-subscriber topology at this private binding boundary. Subscriber-local arrival identities
// can coincide; when both match, recovery cannot prove which queue owns the successor and must
// safely decline.
TEST(RuntimeChannelBindingTemporal, AmbiguousSubscriberIdentityDeclinesFreshnessRecovery) {
  SessionFixture fixture;
  fixture.enable_scalar(InputSelectorKind::kLatestWithin, TemporalRecordMissingKind::kSkip);
  fixture.use_one_slot_drop_oldest_fanout();

  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto first_binding =
      (*session)->create_readiness_binding(VertexId{2U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.plan.plan_generation,
                                               .run_generation = 1U,
                                               .kind = RuntimeEventKind::kInputArrival,
                                               .source = make_source_id(1U),
                                               .source_clock = ClockId{1U},
                                               .reference_clock = ClockId{1U},
                                               .capacity = 8U,
                                           });
  ASSERT_TRUE(first_binding) << runtime_error_message(first_binding.error());
  auto second_binding =
      (*session)->create_readiness_binding(VertexId{3U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.plan.plan_generation,
                                               .run_generation = 1U,
                                               .kind = RuntimeEventKind::kInputArrival,
                                               .source = make_source_id(2U),
                                               .source_clock = ClockId{1U},
                                               .reference_clock = ClockId{1U},
                                               .capacity = 8U,
                                           });
  ASSERT_TRUE(second_binding) << runtime_error_message(second_binding.error());
  const SampleMetadata retained_metadata{
      .sequence = 1U,
      .capture_timestamp_ns = 10,
      .source_clock_id = ClockId{1U},
      .byte_size = sizeof(std::int32_t),
      .payload_kind = PayloadKind::kInlineTyped,
  };
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::inject_received_metadata(
      **session, PortId{2U}, retained_metadata));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::inject_received_metadata(
      **session, PortId{3U}, retained_metadata));
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  std::array<TemporalSampleRecord, 2U> retained;
  const std::array bindings{first_binding->get(), second_binding->get()};
  for (std::size_t index = 0U; index < retained.size(); ++index) {
    auto event = wait_for_event(**session, *bindings[index]);
    ASSERT_TRUE(event) << runtime_error_message(event.error());
    auto sample = (*session)->temporal_sample(event->port, event->source_sequence);
    ASSERT_TRUE(sample) << runtime_error_message(sample.error());
    retained[index] = *sample;
    ASSERT_TRUE(bindings[index]->retire(event->source_sequence));
  }
  ASSERT_EQ(retained[0], retained[1]);

  auto live_alias = RuntimeChannelSessionTestAccess::set_freshness_subscriber_alias(
      **session, PortId{2U}, PortId{3U}, true);
  ASSERT_FALSE(live_alias);
  EXPECT_EQ(live_alias.error().code, ErrorCode::kInvalidArgument);
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());

  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_freshness_subscriber_alias(
      **session, PortId{2U}, PortId{3U}, true));
  auto available = (*session)->has_freshness_refill_candidate(PortId{2U}, retained[0]);
  EXPECT_TRUE(available) << runtime_error_message(available.error());
  if (available) {
    EXPECT_FALSE(*available);
  }
  auto reservation =
      (*session)->reserve_freshness_refill(PortId{2U}, retained[0], TimePoint{20, ClockId{1U}});
  EXPECT_TRUE(reservation) << runtime_error_message(reservation.error());
  if (reservation) {
    EXPECT_FALSE(reservation->has_value());
  }
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_freshness_subscriber_alias(
      **session, PortId{2U}, PortId{3U}, false));
}

// REQ: RUNTIME-017
// Reservation generations make the handoff exclusive: a token superseded by a later probe cannot
// release the retained claim or consume whatever staging happens to occupy the slot now.
TEST(RuntimeChannelBindingTemporal, SupersededReservationCannotCommitALaterHandoff) {
  RetainedRefillArrangement arrangement;
  ASSERT_NO_FATAL_FAILURE(arrange_retained_refill(arrangement, 10));
  ASSERT_TRUE(arrangement.emit(22, 20));

  auto first = arrangement.session->reserve_freshness_refill(
      PortId{2U}, arrangement.retained, TimePoint{20, ClockId{1U}});
  ASSERT_TRUE(first) << runtime_error_message(first.error());
  const auto& optional_first = *first;
  HOLOSCAN_ASSERT_HAS_VALUE(optional_first) << "Expected freshness reservation";
  auto second = arrangement.session->reserve_freshness_refill(
      PortId{2U}, arrangement.retained, TimePoint{20, ClockId{1U}});
  ASSERT_TRUE(second) << runtime_error_message(second.error());
  const auto& optional_second = *second;
  HOLOSCAN_ASSERT_HAS_VALUE(optional_second) << "Expected freshness reservation";
  ASSERT_NE(optional_first->generation, optional_second->generation);

  std::uint32_t promoted = 0U;
  auto stale = arrangement.session->retire_samples(
      PortId{2U},
      std::span<const TemporalSampleRecord>{&arrangement.retained, 1U},
      &promoted,
      &*optional_first);
  ASSERT_FALSE(stale);
  EXPECT_EQ(stale.error().code, ErrorCode::kStaleEndpoint);
  EXPECT_EQ(promoted, 0U);
  auto retained = arrangement.session->temporal_sample(PortId{2U}, 1U);
  ASSERT_TRUE(retained) << runtime_error_message(retained.error());

  ASSERT_TRUE(arrangement.session->retire_samples(
      PortId{2U},
      std::span<const TemporalSampleRecord>{&arrangement.retained, 1U},
      &promoted,
      &*optional_second));
  EXPECT_EQ(promoted, 1U);
  ASSERT_TRUE(arrangement.session->retire_activation(VertexId{1U}, 1U, true));
  arrangement.session->cancel();
  ASSERT_TRUE(arrangement.session->join());
}

// REQ: RUNTIME-017
// REQ: RUNTIME-043
// REQ: CHAN-128
// A freshness refill must promote staged records in arrival order. Publishing a newer record
// before an older staged predecessor would make a real temporal cache reject the predecessor at
// its monotonic arrival watermark.
TEST(RuntimeChannelBindingTemporal, FreshnessRefillPromotesStagedArrivalsInOrder) {
  SessionFixture fixture;
  fixture.enable_scalar(InputSelectorKind::kLatestWithin, TemporalRecordMissingKind::kSkip);
  fixture.use_two_slot_drop_oldest_queue();
  fixture.use_direct_retained_delivery();
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto binding =
      (*session)->create_readiness_binding(VertexId{2U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.plan.plan_generation,
                                               .run_generation = 1U,
                                               .kind = RuntimeEventKind::kInputArrival,
                                               .source = make_source_id(1U),
                                               .source_clock = ClockId{1U},
                                               .reference_clock = ClockId{1U},
                                               .capacity = 8U,
                                           });
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());
  const RuntimeSourceBindingRecord source{
      .candidate_kind = RuntimeEventKind::kInputArrival,
      .source = make_source_id(1U),
      .port = PortId{2U},
      .source_clock = ClockId{1U},
      .reference_clock = ClockId{1U},
      .source_to_reference_conversion_id = 0U,
      .source_order = 0U,
      .capacity = 8U,
      .source_generation = 1U,
      .tag_generation = 1U,
      .receiver_index = 0U,
      .partition = ExecutionPartitionId{1U},
      .execution_lane = 0U,
      .vertex = VertexId{2U},
      .plan_vertex_order = 0U,
      .stable_candidate_tie = 1U,
  };
  auto coordinator = RuntimeCandidateCoordinator::create({&source, 1U}, 1U, 8U);
  ASSERT_TRUE(coordinator) << runtime_error_message(coordinator.error());
  std::int64_t observed_ns = 1;
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  auto selector = RuntimeInputSelectionState::create(fixture.plan.input_selections.front());
  ASSERT_TRUE(selector) << runtime_error_message(selector.error());

  std::array<TemporalSampleRecord, 2U> retained;
  for (std::size_t index = 0U; index < retained.size(); ++index) {
    ASSERT_TRUE(fixture.source.output.emit(
        Sample<std::int32_t>{static_cast<std::int32_t>(11 + index),
                             SampleMetadata{
                                 .capture_timestamp_ns = static_cast<std::int64_t>(10 + index * 10),
                                 .source_clock_id = ClockId{1U},
                             }}));
    auto candidate =
        wait_for_frozen_channel_candidate(**session, **binding, *coordinator, observed_ns);
    ASSERT_TRUE(candidate) << runtime_error_message(candidate.error());
    auto record =
        (*session)->temporal_sample(candidate->witness.port, candidate->witness.source_sequence);
    ASSERT_TRUE(record) << runtime_error_message(record.error());
    retained[index] = *record;
    auto retained_pushed = selector->push(*record);
    ASSERT_TRUE(retained_pushed) << runtime_error_message(retained_pushed.error());
    ASSERT_TRUE(coordinator->retire(*candidate));
    ASSERT_TRUE((*binding)->retire(candidate->witness.source_sequence));
  }

  ASSERT_TRUE(fixture.source.output.emit(Sample<std::int32_t>{
      33, SampleMetadata{.capture_timestamp_ns = 30, .source_clock_id = ClockId{1U}}}));
  ASSERT_TRUE(fixture.source.output.emit(Sample<std::int32_t>{
      44, SampleMetadata{.capture_timestamp_ns = 40, .source_clock_id = ClockId{1U}}}));
  auto reservation = (*session)->reserve_freshness_refill(
      PortId{2U},
      retained[0],
      TimePoint{40, ClockId{1U}},
      RuntimeChannelFreshnessSelectionMode::kNewestWithArrivalPrefix);
  ASSERT_TRUE(reservation) << runtime_error_message(reservation.error());
  const auto& optional_reservation = *reservation;
  HOLOSCAN_ASSERT_HAS_VALUE(optional_reservation) << "Expected freshness reservation";
  EXPECT_EQ(optional_reservation->successor.capture_time.timestamp_ns, 30);
  EXPECT_EQ(optional_reservation->target.capture_time.timestamp_ns, 40);
  ASSERT_TRUE(selector->evict_retained(retained[0].arrival_sequence));
  std::uint32_t promoted = 0U;
  ASSERT_TRUE((*session)->retire_samples(PortId{2U},
                                         std::span<const TemporalSampleRecord>{&retained[0], 1U},
                                         &promoted,
                                         &*optional_reservation));
  ASSERT_EQ(promoted, 1U);
  auto older_candidate =
      wait_for_frozen_channel_candidate(**session, **binding, *coordinator, observed_ns);
  ASSERT_TRUE(older_candidate) << runtime_error_message(older_candidate.error());
  auto older = (*session)->temporal_sample(older_candidate->witness.port,
                                           older_candidate->witness.source_sequence);
  ASSERT_TRUE(older) << runtime_error_message(older.error());
  auto older_pushed = selector->push(*older);
  ASSERT_TRUE(older_pushed) << runtime_error_message(older_pushed.error());
  ASSERT_TRUE(coordinator->retire(*older_candidate));
  ASSERT_TRUE((*binding)->retire(older_candidate->witness.source_sequence));

  auto final_reservation = (*session)->reserve_freshness_refill(
      PortId{2U},
      retained[1],
      TimePoint{40, ClockId{1U}},
      RuntimeChannelFreshnessSelectionMode::kNewestWithArrivalPrefix);
  ASSERT_TRUE(final_reservation) << runtime_error_message(final_reservation.error());
  const auto& optional_final_reservation = *final_reservation;
  HOLOSCAN_ASSERT_HAS_VALUE(optional_final_reservation) << "Expected freshness reservation";
  EXPECT_EQ(optional_final_reservation->successor.capture_time.timestamp_ns, 40);
  EXPECT_EQ(optional_final_reservation->target.capture_time.timestamp_ns, 40);
  ASSERT_TRUE(selector->evict_retained(retained[1].arrival_sequence));
  promoted = 0U;
  ASSERT_TRUE((*session)->retire_samples(PortId{2U},
                                         std::span<const TemporalSampleRecord>{&retained[1], 1U},
                                         &promoted,
                                         &*optional_final_reservation));
  ASSERT_EQ(promoted, 1U);
  auto newer_candidate =
      wait_for_frozen_channel_candidate(**session, **binding, *coordinator, observed_ns);
  ASSERT_TRUE(newer_candidate) << runtime_error_message(newer_candidate.error());
  auto newer = (*session)->temporal_sample(newer_candidate->witness.port,
                                           newer_candidate->witness.source_sequence);
  ASSERT_TRUE(newer) << runtime_error_message(newer.error());
  auto newer_pushed = selector->push(*newer);
  ASSERT_TRUE(newer_pushed) << runtime_error_message(newer_pushed.error());
  EXPECT_EQ(older->capture_time.timestamp_ns, 30);
  EXPECT_EQ(newer->capture_time.timestamp_ns, 40);
  EXPECT_LT(older->arrival_sequence, newer->arrival_sequence);

  auto newest_selection =
      selector->begin(TimePoint{40, ClockId{1U}}, newer->arrival_sequence, newer->producer_epoch);
  ASSERT_TRUE(newest_selection) << runtime_error_message(newest_selection.error());
  ASSERT_EQ(newest_selection->selected().size(), 1U);
  EXPECT_EQ(newest_selection->selected().front(), *newer);
  ASSERT_TRUE(newest_selection->finish(false));

  ASSERT_TRUE(coordinator->retire(*newer_candidate));
  ASSERT_TRUE((*binding)->retire(newer_candidate->witness.source_sequence));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// REQ: RUNTIME-017
// REQ: CHAN-128
// Rotate the retained typed record away from slots.begin(), then reserve its successor in the now
// first staging slot. Every ordinary promotion candidate is ineligible, so min_element returns the
// reserved sentinel at begin(); that result must remain untouched for reservation recovery.
TEST(RuntimeChannelBindingTemporal, AllIneligibleOrdinaryPromotionPreservesReservedRefill) {
  constexpr std::int64_t kFirstCaptureNs = 10;
  constexpr std::int64_t kRetainedCaptureNs = 20;
  constexpr std::int64_t kReservedCaptureNs = 30;

  RetainedRefillArrangement arrangement;
  ASSERT_NO_FATAL_FAILURE(arrange_retained_refill(arrangement, kFirstCaptureNs));

  ASSERT_TRUE(arrangement.emit(22, kRetainedCaptureNs));
  std::uint32_t promoted = 0U;
  ASSERT_TRUE(arrangement.session->retire_samples(
      PortId{2U}, std::span<const TemporalSampleRecord>{&arrangement.retained, 1U}, &promoted));
  ASSERT_EQ(promoted, 1U);
  auto retained_event = wait_for_event(*arrangement.session, *arrangement.binding);
  ASSERT_TRUE(retained_event) << runtime_error_message(retained_event.error());
  auto retained =
      arrangement.session->temporal_sample(retained_event->port, retained_event->source_sequence);
  ASSERT_TRUE(retained) << runtime_error_message(retained.error());
  ASSERT_EQ(retained->capture_time.timestamp_ns, kRetainedCaptureNs);
  ASSERT_TRUE(arrangement.binding->retire(retained_event->source_sequence));

  ASSERT_TRUE(arrangement.emit(33, kReservedCaptureNs));
  auto reservation = arrangement.session->reserve_freshness_refill(
      PortId{2U}, *retained, TimePoint{kReservedCaptureNs, ClockId{1U}});
  ASSERT_TRUE(reservation) << runtime_error_message(reservation.error());
  ASSERT_TRUE(reservation->has_value());
  const auto before =
      RuntimeChannelSessionTestAccess::input_reset_state(*arrangement.session, PortId{2U});
  ASSERT_TRUE(before) << runtime_error_message(before.error());
  ASSERT_EQ(before->occupied_samples, 1U);
  ASSERT_EQ(before->pending_direct_ingress_samples, 1U);

  promoted = 1U;
  ASSERT_TRUE(arrangement.session->retire_samples(
      PortId{2U}, std::span<const TemporalSampleRecord>{&*retained, 1U}, &promoted));
  EXPECT_EQ(promoted, 0U);
  const auto preserved =
      RuntimeChannelSessionTestAccess::input_reset_state(*arrangement.session, PortId{2U});
  ASSERT_TRUE(preserved) << runtime_error_message(preserved.error());
  EXPECT_EQ(preserved->occupied_samples, 0U);
  EXPECT_EQ(preserved->pending_direct_ingress_samples, 1U);
  EXPECT_EQ(preserved->last_producer_sequence, before->last_producer_sequence);
  EXPECT_EQ(preserved->next_global_arrival_order, before->next_global_arrival_order);

  auto recovered = arrangement.session->recover_abandoned_freshness_reservation(PortId{2U});
  ASSERT_TRUE(recovered) << runtime_error_message(recovered.error());
  ASSERT_TRUE(*recovered);
  auto recovered_event = wait_for_event(*arrangement.session, *arrangement.binding);
  ASSERT_TRUE(recovered_event) << runtime_error_message(recovered_event.error());
  auto recovered_record =
      arrangement.session->temporal_sample(recovered_event->port, recovered_event->source_sequence);
  ASSERT_TRUE(recovered_record) << runtime_error_message(recovered_record.error());
  EXPECT_EQ(recovered_record->capture_time.timestamp_ns, kReservedCaptureNs);

  const std::array selected{*recovered_record};
  const std::array selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = selected,
      .destructive = true,
  }};
  ASSERT_TRUE(arrangement.session->prepare_activation(VertexId{2U}, 1U, selection));
  auto value = arrangement.fixture.sink.input.receive_data();
  ASSERT_TRUE(value) << runtime_error_message(value.error());
  EXPECT_EQ(*value, 33);

  ASSERT_TRUE(arrangement.binding->retire(recovered_event->source_sequence));
  ASSERT_TRUE(arrangement.session->retire_activation(VertexId{2U}, 1U, true));
  ASSERT_TRUE(arrangement.session->retire_activation(VertexId{1U}, 1U, true));
  arrangement.session->cancel();
  ASSERT_TRUE(arrangement.session->join());
}

// Re-reserving while a retained claim is still live preserves the prior uncommitted target as a
// recovery marker. Even when another typed slot has become writable, ordinary promotion must not
// consume that marker before the new reservation publishes.
// REQ: RUNTIME-017
// REQ: CHAN-128
TEST(RuntimeChannelBindingTemporal, PreservedUncommittedRefillTargetIsNotOrdinarilyPromoted) {
  SessionFixture fixture;
  fixture.enable_scalar(InputSelectorKind::kLatestWithin, TemporalRecordMissingKind::kSkip);
  fixture.use_two_slot_drop_oldest_queue();
  fixture.use_direct_retained_delivery();
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto binding =
      (*session)->create_readiness_binding(VertexId{2U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.plan.plan_generation,
                                               .run_generation = 1U,
                                               .kind = RuntimeEventKind::kInputArrival,
                                               .source = make_source_id(1U),
                                               .source_clock = ClockId{1U},
                                               .reference_clock = ClockId{1U},
                                               .capacity = 8U,
                                           });
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  const auto emit = [&](const std::int32_t value, const std::int64_t capture_ns) {
    return fixture.source.output.emit(Sample<std::int32_t>{
        value, SampleMetadata{.capture_timestamp_ns = capture_ns, .source_clock_id = ClockId{1U}}});
  };

  std::array<TemporalSampleRecord, 2U> retained;
  for (std::size_t index = 0U; index < retained.size(); ++index) {
    ASSERT_TRUE(
        emit(static_cast<std::int32_t>(11 + index), static_cast<std::int64_t>(10 + index * 10)));
    auto event = wait_for_event(**session, **binding);
    ASSERT_TRUE(event) << runtime_error_message(event.error());
    auto record = (*session)->temporal_sample(event->port, event->source_sequence);
    ASSERT_TRUE(record) << runtime_error_message(record.error());
    retained[index] = *record;
    ASSERT_TRUE((*binding)->retire(event->source_sequence));
  }

  ASSERT_TRUE(emit(33, 30));
  std::uint32_t promoted = 0U;
  ASSERT_TRUE((*session)->retire_samples(
      PortId{2U}, std::span<const TemporalSampleRecord>{&retained[0], 1U}, &promoted));
  ASSERT_EQ(promoted, 1U);
  auto third_event = wait_for_event(**session, **binding);
  ASSERT_TRUE(third_event) << runtime_error_message(third_event.error());
  auto third = (*session)->temporal_sample(third_event->port, third_event->source_sequence);
  ASSERT_TRUE(third) << runtime_error_message(third.error());
  ASSERT_EQ(third->capture_time.timestamp_ns, 30);
  ASSERT_TRUE((*binding)->retire(third_event->source_sequence));

  ASSERT_TRUE(emit(44, 40));
  auto first_reservation =
      (*session)->reserve_freshness_refill(PortId{2U}, retained[1], TimePoint{40, ClockId{1U}});
  ASSERT_TRUE(first_reservation) << runtime_error_message(first_reservation.error());
  const auto& optional_first_reservation = *first_reservation;
  HOLOSCAN_ASSERT_HAS_VALUE(optional_first_reservation) << "Expected freshness reservation";
  EXPECT_EQ(optional_first_reservation->target.capture_time.timestamp_ns, 40);

  promoted = 1U;
  ASSERT_TRUE((*session)->retire_samples(
      PortId{2U}, std::span<const TemporalSampleRecord>{&*third, 1U}, &promoted));
  ASSERT_EQ(promoted, 0U);
  auto replacement_reservation =
      (*session)->reserve_freshness_refill(PortId{2U}, retained[1], TimePoint{40, ClockId{1U}});
  ASSERT_TRUE(replacement_reservation) << runtime_error_message(replacement_reservation.error());
  const auto& optional_replacement_reservation = *replacement_reservation;
  HOLOSCAN_ASSERT_HAS_VALUE(optional_replacement_reservation) << "Expected freshness reservation";
  EXPECT_EQ(optional_replacement_reservation->successor, optional_first_reservation->successor);
  EXPECT_EQ(optional_replacement_reservation->target, optional_first_reservation->target);

  (*session)->release_freshness_reservation(*optional_replacement_reservation);
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// REQ: RUNTIME-017
// A lifecycle restart must clear reservation-only state even when clear_samples() makes its
// occupied staged slot unoccupied directly. Otherwise the first post-restart sample inherits
// refill_reserved and is invisible to ordinary promotion.
TEST(RuntimeChannelBindingTemporal, LifecycleRestartClearsAnAbandonedFreshnessReservation) {
  RetainedRefillArrangement arrangement;
  ASSERT_NO_FATAL_FAILURE(arrange_retained_refill(arrangement, 10));
  ASSERT_TRUE(arrangement.emit(22, 20));
  auto reservation = arrangement.session->reserve_freshness_refill(
      PortId{2U}, arrangement.retained, TimePoint{20, ClockId{1U}});
  ASSERT_TRUE(reservation) << runtime_error_message(reservation.error());
  ASSERT_TRUE(reservation->has_value());

  // Ordinary retirement deliberately skips the reserved staged slot. This leaves a claim-free,
  // occupied reservation for reset() to discard through clear_samples().
  std::uint32_t promoted = 1U;
  ASSERT_TRUE(arrangement.session->retire_samples(
      PortId{2U}, std::span<const TemporalSampleRecord>{&arrangement.retained, 1U}, &promoted));
  ASSERT_EQ(promoted, 0U);
  ASSERT_TRUE(arrangement.session->retire_activation(VertexId{1U}, 1U, true));
  arrangement.session->cancel();
  ASSERT_TRUE(arrangement.session->join());
  const std::array affected{VertexId{1U}, VertexId{2U}};
  ASSERT_TRUE(arrangement.session->commit_restart(VertexId{1U}, affected));

  ASSERT_TRUE(arrangement.session->refresh_connections());
  ASSERT_TRUE(arrangement.session->arm());
  ASSERT_TRUE(arrangement.session->prepare_activation(VertexId{1U}, 2U, {}));
  ASSERT_TRUE(arrangement.emit(33, 30));
  auto event = wait_for_event(*arrangement.session, *arrangement.binding);
  ASSERT_TRUE(event) << runtime_error_message(event.error());
  auto record = arrangement.session->temporal_sample(event->port, event->source_sequence);
  ASSERT_TRUE(record) << runtime_error_message(record.error());
  EXPECT_EQ(record->capture_time.timestamp_ns, 30);

  ASSERT_TRUE(arrangement.binding->retire(event->source_sequence));
  ASSERT_TRUE(arrangement.session->retire_activation(VertexId{1U}, 2U, true));
  arrangement.session->cancel();
  ASSERT_TRUE(arrangement.session->join());
}

// REQ: RUNTIME-017
// A reserved record does not consume the authored staging depth, so a producer that keeps
// publishing neither displaces it nor is itself blocked. Both halves matter: protecting the record
// without the headroom closes the producer's only staging destination, which measured as a 12x
// throughput loss, while bypassing the reserved record would publish readiness out of arrival
// order.
TEST(RuntimeChannelBindingTemporal, AReservedRefillSurvivesAProducerThatKeepsStaging) {
  constexpr std::int64_t kRetainedCaptureNs = 10;
  constexpr std::int64_t kReservedCaptureNs = 20;
  constexpr std::int64_t kLaterCaptureNs = 30;
  constexpr std::int64_t kNewestCaptureNs = 40;

  RetainedRefillArrangement arrangement;
  ASSERT_NO_FATAL_FAILURE(arrange_retained_refill(arrangement, kRetainedCaptureNs));

  ASSERT_TRUE(arrangement.emit(22, kReservedCaptureNs));
  auto reserved = arrangement.session->reserve_freshness_refill(
      PortId{2U}, arrangement.retained, TimePoint{kReservedCaptureNs, ClockId{1U}});
  ASSERT_TRUE(reserved) << runtime_error_message(reserved.error());
  ASSERT_TRUE(reserved->has_value());

  // The producer keeps going. Staging has room for this because the reserved record is spoken for
  // and does not count against the authored depth.
  ASSERT_TRUE(arrangement.emit(33, kLaterCaptureNs));
  // And once more, which leaves no free slot at all and so drives the replacement branch rather
  // than the free-slot one. Drop-oldest must take the unreserved record here even though the
  // reserved one is older; without that this case never exercises the exclusion it claims to.
  ASSERT_TRUE(arrangement.emit(44, kNewestCaptureNs));

  // Key assertion: the reserved record is still staged. Requalifying at its own coordinate finds
  // it, which it could not if the later arrival had displaced it.
  auto still_staged = arrangement.session->reserve_freshness_refill(
      PortId{2U}, arrangement.retained, TimePoint{kReservedCaptureNs, ClockId{1U}});
  ASSERT_TRUE(still_staged) << runtime_error_message(still_staged.error());
  EXPECT_TRUE(still_staged->has_value());

  // And the other half: the later arrival staged rather than being turned away, which it could not
  // have done if the reserved record had consumed the authored depth. Even at the later floor, the
  // refill must admit the reserved predecessor before the newer staged record.
  auto both_staged = arrangement.session->reserve_freshness_refill(
      PortId{2U},
      arrangement.retained,
      TimePoint{kNewestCaptureNs, ClockId{1U}},
      RuntimeChannelFreshnessSelectionMode::kNewestWithArrivalPrefix);
  ASSERT_TRUE(both_staged) << runtime_error_message(both_staged.error());
  const auto& optional_both_staged = *both_staged;
  HOLOSCAN_ASSERT_HAS_VALUE(optional_both_staged) << "Expected freshness reservation";
  std::uint32_t promoted = 0U;
  ASSERT_TRUE(arrangement.session->retire_samples(
      PortId{2U},
      std::span<const TemporalSampleRecord>{&arrangement.retained, 1U},
      &promoted,
      &*optional_both_staged));
  ASSERT_EQ(promoted, 1U);
  auto admitted_event = wait_for_event(*arrangement.session, *arrangement.binding);
  ASSERT_TRUE(admitted_event) << runtime_error_message(admitted_event.error());
  auto admitted =
      arrangement.session->temporal_sample(admitted_event->port, admitted_event->source_sequence);
  ASSERT_TRUE(admitted) << runtime_error_message(admitted.error());
  EXPECT_EQ(admitted->capture_time.timestamp_ns, kReservedCaptureNs);

  ASSERT_TRUE(arrangement.binding->retire(admitted_event->source_sequence));
  auto target_reservation = arrangement.session->reserve_freshness_refill(
      PortId{2U},
      *admitted,
      TimePoint{kNewestCaptureNs, ClockId{1U}},
      RuntimeChannelFreshnessSelectionMode::kNewestWithArrivalPrefix);
  ASSERT_TRUE(target_reservation) << runtime_error_message(target_reservation.error());
  const auto& optional_target_reservation = *target_reservation;
  HOLOSCAN_ASSERT_HAS_VALUE(optional_target_reservation) << "Expected freshness reservation";
  EXPECT_EQ(optional_target_reservation->successor.capture_time.timestamp_ns, kNewestCaptureNs);
  EXPECT_EQ(optional_target_reservation->target.capture_time.timestamp_ns, kNewestCaptureNs);
  promoted = 0U;
  ASSERT_TRUE(
      arrangement.session->retire_samples(PortId{2U},
                                          std::span<const TemporalSampleRecord>{&*admitted, 1U},
                                          &promoted,
                                          &*optional_target_reservation));
  ASSERT_EQ(promoted, 1U);
  auto newest_event = wait_for_event(*arrangement.session, *arrangement.binding);
  ASSERT_TRUE(newest_event) << runtime_error_message(newest_event.error());
  auto newest =
      arrangement.session->temporal_sample(newest_event->port, newest_event->source_sequence);
  ASSERT_TRUE(newest) << runtime_error_message(newest.error());
  EXPECT_EQ(newest->capture_time.timestamp_ns, kNewestCaptureNs);
  EXPECT_LT(admitted->arrival_sequence, newest->arrival_sequence);
  ASSERT_TRUE(arrangement.binding->retire(newest_event->source_sequence));
  ASSERT_TRUE(arrangement.session->retire_activation(VertexId{1U}, 1U, true));
  arrangement.session->cancel();
  ASSERT_TRUE(arrangement.session->join());
}

// REQ: RUNTIME-017
// REQ: RUNTIME-043
// A committed prefix target remains the next ordering fence even if ordinary activation
// retirement frees typed capacity before the controller opens the target's final reservation.
// Newer producer traffic must remain staged until that target publishes.
TEST(RuntimeChannelBindingTemporal, OrdinaryRetirementCannotOvertakeACommittedRefillTarget) {
  RetainedRefillArrangement arrangement;
  ASSERT_NO_FATAL_FAILURE(arrange_retained_refill(arrangement, 10));

  ASSERT_TRUE(arrangement.emit(22, 20));
  auto initial = arrangement.session->reserve_freshness_refill(
      PortId{2U}, arrangement.retained, TimePoint{20, ClockId{1U}});
  ASSERT_TRUE(initial) << runtime_error_message(initial.error());
  ASSERT_TRUE(initial->has_value());
  ASSERT_TRUE(arrangement.emit(33, 30));

  auto prefix = arrangement.session->reserve_freshness_refill(
      PortId{2U},
      arrangement.retained,
      TimePoint{30, ClockId{1U}},
      RuntimeChannelFreshnessSelectionMode::kNewestWithArrivalPrefix);
  ASSERT_TRUE(prefix) << runtime_error_message(prefix.error());
  const auto& optional_prefix = *prefix;
  HOLOSCAN_ASSERT_HAS_VALUE(optional_prefix) << "Expected freshness reservation";
  ASSERT_EQ(optional_prefix->successor.capture_time.timestamp_ns, 20);
  ASSERT_EQ(optional_prefix->target.capture_time.timestamp_ns, 30);
  std::uint32_t promoted = 0U;
  ASSERT_TRUE(arrangement.session->retire_samples(
      PortId{2U},
      std::span<const TemporalSampleRecord>{&arrangement.retained, 1U},
      &promoted,
      &*optional_prefix));
  ASSERT_EQ(promoted, 1U);
  auto first_event = wait_for_event(*arrangement.session, *arrangement.binding);
  ASSERT_TRUE(first_event) << runtime_error_message(first_event.error());
  auto first =
      arrangement.session->temporal_sample(first_event->port, first_event->source_sequence);
  ASSERT_TRUE(first) << runtime_error_message(first.error());
  ASSERT_EQ(first->capture_time.timestamp_ns, 20);
  ASSERT_TRUE(arrangement.binding->retire(first_event->source_sequence));

  auto changed_mode =
      arrangement.session->reserve_freshness_refill(PortId{2U},
                                                    *first,
                                                    TimePoint{40, ClockId{1U}},
                                                    RuntimeChannelFreshnessSelectionMode::kNewest);
  ASSERT_FALSE(changed_mode);
  EXPECT_EQ(changed_mode.error().code, ErrorCode::kProtocolError);

  ASSERT_TRUE(arrangement.emit(44, 40));
  promoted = 0U;
  ASSERT_TRUE(arrangement.session->retire_samples(
      PortId{2U}, std::span<const TemporalSampleRecord>{&*first, 1U}, &promoted));
  ASSERT_EQ(promoted, 1U);
  auto target_event = wait_for_event(*arrangement.session, *arrangement.binding);
  ASSERT_TRUE(target_event) << runtime_error_message(target_event.error());
  auto target =
      arrangement.session->temporal_sample(target_event->port, target_event->source_sequence);
  ASSERT_TRUE(target) << runtime_error_message(target.error());
  EXPECT_EQ(target->capture_time.timestamp_ns, 30);
  EXPECT_LT(first->arrival_sequence, target->arrival_sequence);
  ASSERT_TRUE(arrangement.binding->retire(target_event->source_sequence));

  promoted = 0U;
  ASSERT_TRUE(arrangement.session->retire_samples(
      PortId{2U}, std::span<const TemporalSampleRecord>{&*target, 1U}, &promoted));
  ASSERT_EQ(promoted, 1U);
  auto newest_event = wait_for_event(*arrangement.session, *arrangement.binding);
  ASSERT_TRUE(newest_event) << runtime_error_message(newest_event.error());
  auto newest =
      arrangement.session->temporal_sample(newest_event->port, newest_event->source_sequence);
  ASSERT_TRUE(newest) << runtime_error_message(newest.error());
  EXPECT_EQ(newest->capture_time.timestamp_ns, 40);
  EXPECT_LT(target->arrival_sequence, newest->arrival_sequence);
  ASSERT_TRUE(arrangement.binding->retire(newest_event->source_sequence));

  ASSERT_TRUE(arrangement.session->retire_activation(VertexId{1U}, 1U, true));
  arrangement.session->cancel();
  ASSERT_TRUE(arrangement.session->join());
}

// REQ: RUNTIME-043
// A refill target is the newest causal sample under LatestWithin's capture-time ordering, not the
// greatest producer sequence. Admission still follows producer order through that target, but a
// later arrival with an older capture coordinate must not make the actually fresh sample
// ineligible.
TEST(RuntimeChannelBindingTemporal, RefillTargetUsesLatestWithinCaptureOrdering) {
  RetainedRefillArrangement arrangement;
  ASSERT_NO_FATAL_FAILURE(arrange_retained_refill(arrangement, 10));

  ASSERT_TRUE(arrangement.emit(22, 99));
  auto protected_fresh = arrangement.session->reserve_freshness_refill(
      PortId{2U}, arrangement.retained, TimePoint{100, ClockId{1U}});
  ASSERT_TRUE(protected_fresh) << runtime_error_message(protected_fresh.error());
  const auto& optional_protected_fresh = *protected_fresh;
  HOLOSCAN_ASSERT_HAS_VALUE(optional_protected_fresh) << "Expected freshness reservation";
  ASSERT_EQ(optional_protected_fresh->target.capture_time.timestamp_ns, 99);

  // The protected record occupies refill headroom, so this later producer sequence survives too.
  // Its capture coordinate is older even though its arrival identity is newer.
  ASSERT_TRUE(arrangement.emit(33, 20));
  auto newest_by_producer =
      arrangement.session->reserve_freshness_refill(PortId{2U},
                                                    arrangement.retained,
                                                    TimePoint{100, ClockId{1U}},
                                                    RuntimeChannelFreshnessSelectionMode::kNewest);
  ASSERT_TRUE(newest_by_producer) << runtime_error_message(newest_by_producer.error());
  const auto& optional_newest_by_producer = *newest_by_producer;
  HOLOSCAN_ASSERT_HAS_VALUE(optional_newest_by_producer) << "Expected freshness reservation";
  EXPECT_EQ(optional_newest_by_producer->successor.capture_time.timestamp_ns, 20);

  auto reservation = arrangement.session->reserve_freshness_refill(
      PortId{2U},
      arrangement.retained,
      TimePoint{100, ClockId{1U}},
      RuntimeChannelFreshnessSelectionMode::kNewestWithArrivalPrefix);
  ASSERT_TRUE(reservation) << runtime_error_message(reservation.error());
  const auto& optional_reservation = *reservation;
  HOLOSCAN_ASSERT_HAS_VALUE(optional_reservation) << "Expected freshness reservation";
  EXPECT_EQ(optional_reservation->successor.capture_time.timestamp_ns, 99);
  EXPECT_EQ(optional_reservation->target.capture_time.timestamp_ns, 99);

  arrangement.session->release_freshness_reservation(*optional_reservation);
  ASSERT_TRUE(arrangement.session->retire_activation(VertexId{1U}, 1U, true));
  arrangement.session->cancel();
  ASSERT_TRUE(arrangement.session->join());
}

// REQ: RUNTIME-017
// Releasing a reservation hands the record back to ordinary drop-oldest. The reconcile abandons a
// reservation whose eviction did not proceed, restores authored capacity after refill headroom is
// used, and leaves the newest replacement reservable.
TEST(RuntimeChannelBindingTemporal, ReleasingAReservationReturnsTheRecordToDropOldest) {
  constexpr std::int64_t kRetainedCaptureNs = 10;
  constexpr std::int64_t kReservedCaptureNs = 20;
  constexpr std::int64_t kHeadroomCaptureNs = 30;
  constexpr std::int64_t kNewestCaptureNs = 40;

  RetainedRefillArrangement arrangement;
  ASSERT_NO_FATAL_FAILURE(arrange_retained_refill(arrangement, kRetainedCaptureNs));

  ASSERT_TRUE(arrangement.emit(22, kReservedCaptureNs));
  auto reserved = arrangement.session->reserve_freshness_refill(
      PortId{2U}, arrangement.retained, TimePoint{kReservedCaptureNs, ClockId{1U}});
  ASSERT_TRUE(reserved) << runtime_error_message(reserved.error());
  const auto& optional_reserved = *reserved;
  HOLOSCAN_ASSERT_HAS_VALUE(optional_reserved) << "Expected freshness reservation";

  // Exercise the one extra slot while the reservation protects the authored-depth record.
  ASSERT_TRUE(arrangement.emit(33, kHeadroomCaptureNs));
  arrangement.session->release_freshness_reservation(*optional_reserved);

  // With the protection gone the authored depth applies again, so the newest arrival can replace
  // the remaining staged record.
  ASSERT_TRUE(arrangement.emit(44, kNewestCaptureNs));

  // Key assertion: nothing qualifies at the released record's coordinate any more, because that
  // record is gone. A release that did nothing would leave it staged and findable here.
  auto requalified = arrangement.session->reserve_freshness_refill(
      PortId{2U}, arrangement.retained, TimePoint{kReservedCaptureNs, ClockId{1U}});
  ASSERT_TRUE(requalified) << runtime_error_message(requalified.error());
  EXPECT_FALSE(requalified->has_value());

  auto newest = arrangement.session->reserve_freshness_refill(
      PortId{2U}, arrangement.retained, TimePoint{kNewestCaptureNs, ClockId{1U}});
  ASSERT_TRUE(newest) << runtime_error_message(newest.error());
  const auto& optional_newest = *newest;
  HOLOSCAN_ASSERT_HAS_VALUE(optional_newest) << "Expected freshness reservation";
  EXPECT_EQ(optional_newest->successor.capture_time.timestamp_ns, kNewestCaptureNs);
  arrangement.session->release_freshness_reservation(*optional_newest);

  ASSERT_TRUE(arrangement.session->retire_activation(VertexId{1U}, 1U, true));
  arrangement.session->cancel();
  ASSERT_TRUE(arrangement.session->join());
}

// REQ: RUNTIME-017
// A refused selector eviction releases its reservation from the dispatcher thread. Contention
// with a producer must defer that cleanup instead of parking the dispatcher behind the cache lock.
TEST(RuntimeChannelBindingTemporal, ReleasingAReservationDoesNotWaitForTheProducer) {
  RetainedRefillArrangement arrangement;
  ASSERT_NO_FATAL_FAILURE(arrange_retained_refill(arrangement, 10));
  ASSERT_TRUE(arrangement.emit(22, 20));
  auto reserved = arrangement.session->reserve_freshness_refill(
      PortId{2U}, arrangement.retained, TimePoint{20, ClockId{1U}});
  ASSERT_TRUE(reserved) << runtime_error_message(reserved.error());
  const auto& optional_reserved = *reserved;
  HOLOSCAN_ASSERT_HAS_VALUE(optional_reserved) << "Expected freshness reservation";

  struct LockScript {
    std::binary_semaphore held{0};
    std::binary_semaphore release{0};
    std::binary_semaphore caller_started{0};
    std::binary_semaphore caller_returned{0};
  } script;
  std::thread holder{[&] {
    static_cast<void>(RuntimeChannelSessionTestAccess::with_input_lock(
        *arrangement.session, PortId{2U}, &script, [](void* state) noexcept {
          auto& owned = *static_cast<LockScript*>(state);
          owned.held.release();
          owned.release.acquire();
        }));
  }};
  script.held.acquire();
  std::thread caller{[&] {
    script.caller_started.release();
    arrangement.session->release_freshness_reservation(*optional_reserved);
    script.caller_returned.release();
  }};
  script.caller_started.acquire();
  const bool returned_while_contended = script.caller_returned.try_acquire_for(50ms);
  script.release.release();
  if (!returned_while_contended) {
    script.caller_returned.acquire();
  }
  caller.join();
  holder.join();

  EXPECT_TRUE(returned_while_contended);
  // A deferred release deliberately leaves the generation-fenced reservation for the next pass.
  arrangement.session->release_freshness_reservation(*optional_reserved);
  ASSERT_TRUE(arrangement.session->retire_activation(VertexId{1U}, 1U, true));
  arrangement.session->cancel();
  ASSERT_TRUE(arrangement.session->join());
}

// REQ: RUNTIME-017
// A deferred reservation release must not leave an uncommitted staged successor fenced after its
// reservation identity is cleared. Recovery restores ordinary promotion even while the retained
// claim is still live, so retiring that claim needs no later producer arrival or reservation probe.
TEST(RuntimeChannelBindingTemporal, DeferredReservationSelfHealPromotesAnIdleSuccessor) {
  constexpr std::int64_t kSuccessorCaptureNs = 20;

  RetainedRefillArrangement arrangement;
  const auto counters = std::make_shared<RuntimePhaseCounters>();
  arrangement.fixture.plan.executable_factories.phase_counters = counters;
  ASSERT_NO_FATAL_FAILURE(arrange_retained_refill(arrangement, 10));
  ASSERT_TRUE(arrangement.emit(22, kSuccessorCaptureNs));
  auto reserved = arrangement.session->reserve_freshness_refill(
      PortId{2U}, arrangement.retained, TimePoint{kSuccessorCaptureNs, ClockId{1U}});
  ASSERT_TRUE(reserved) << runtime_error_message(reserved.error());
  const auto& optional_reserved = *reserved;
  HOLOSCAN_ASSERT_HAS_VALUE(optional_reserved) << "Expected freshness reservation";

  struct LockScript {
    std::binary_semaphore held{0};
    std::binary_semaphore release{0};
    std::binary_semaphore caller_returned{0};
  } script;
  std::thread holder{[&] {
    static_cast<void>(RuntimeChannelSessionTestAccess::with_input_lock(
        *arrangement.session, PortId{2U}, &script, [](void* state) noexcept {
          auto& owned = *static_cast<LockScript*>(state);
          owned.held.release();
          owned.release.acquire();
        }));
  }};
  script.held.acquire();
  std::thread caller{[&] {
    arrangement.session->release_freshness_reservation(*optional_reserved);
    script.caller_returned.release();
  }};
  const bool returned_while_contended = script.caller_returned.try_acquire_for(50ms);
  script.release.release();
  if (!returned_while_contended) {
    script.caller_returned.acquire();
  }
  caller.join();
  holder.join();
  ASSERT_TRUE(returned_while_contended);
  const auto deferred = counters->snapshot();
  EXPECT_EQ(deferred[RuntimePhaseCounters::index(
                RuntimePhaseCounter::kChannelFreshnessReservationReleaseDeferred)],
            1U);

  auto recovered = arrangement.session->recover_abandoned_freshness_reservation(PortId{2U});
  ASSERT_TRUE(recovered) << runtime_error_message(recovered.error());
  ASSERT_TRUE(*recovered);

  // The retained claim was live during recovery. Its later retirement must nevertheless promote
  // the successor through the ordinary path; no second reserve call is available to self-heal it.
  std::uint32_t promoted = 0U;
  ASSERT_TRUE(arrangement.session->retire_samples(
      PortId{2U}, std::span<const TemporalSampleRecord>{&arrangement.retained, 1U}, &promoted));
  ASSERT_EQ(promoted, 1U);
  const auto recovered_state =
      RuntimeChannelSessionTestAccess::input_reset_state(*arrangement.session, PortId{2U});
  ASSERT_TRUE(recovered_state) << runtime_error_message(recovered_state.error());
  EXPECT_EQ(recovered_state->pending_direct_ingress_samples, 0U);
  EXPECT_EQ(recovered_state->occupied_samples, 1U);

  auto successor = wait_for_event(*arrangement.session, *arrangement.binding);
  EXPECT_TRUE(successor) << runtime_error_message(successor.error());
  if (successor) {
    auto record = arrangement.session->temporal_sample(successor->port, successor->source_sequence);
    ASSERT_TRUE(record) << runtime_error_message(record.error());
    EXPECT_EQ(record->capture_time.timestamp_ns, kSuccessorCaptureNs);
    ASSERT_TRUE(arrangement.binding->retire(successor->source_sequence));
  }

  ASSERT_TRUE(arrangement.session->retire_activation(VertexId{1U}, 1U, true));
  arrangement.session->cancel();
  ASSERT_TRUE(arrangement.session->join());
}

// REQ: RUNTIME-017
// The next reconcile self-heals an abandoned reservation. That retry is still speculative and
// must decline immediately when the producer owns the cache lock rather than waiting it out.
TEST(RuntimeChannelBindingTemporal, ReservationSelfHealDoesNotWaitForTheProducer) {
  RetainedRefillArrangement arrangement;
  ASSERT_NO_FATAL_FAILURE(arrange_retained_refill(arrangement, 10));
  ASSERT_TRUE(arrangement.emit(22, 20));
  auto reserved = arrangement.session->reserve_freshness_refill(
      PortId{2U}, arrangement.retained, TimePoint{20, ClockId{1U}});
  ASSERT_TRUE(reserved) << runtime_error_message(reserved.error());
  const auto& optional_reserved = *reserved;
  HOLOSCAN_ASSERT_HAS_VALUE(optional_reserved) << "Expected freshness reservation";

  struct LockScript {
    std::binary_semaphore held{0};
    std::binary_semaphore release{0};
    std::binary_semaphore caller_started{0};
    std::binary_semaphore caller_returned{0};
  } script;
  std::thread holder{[&] {
    static_cast<void>(RuntimeChannelSessionTestAccess::with_input_lock(
        *arrangement.session, PortId{2U}, &script, [](void* state) noexcept {
          auto& owned = *static_cast<LockScript*>(state);
          owned.held.release();
          owned.release.acquire();
        }));
  }};
  script.held.acquire();
  bool probe_succeeded = false;
  std::optional<RuntimeChannelFreshnessReservation> replacement;
  std::thread caller{[&] {
    script.caller_started.release();
    auto probed = arrangement.session->reserve_freshness_refill(
        PortId{2U}, arrangement.retained, TimePoint{20, ClockId{1U}});
    probe_succeeded = probed.has_value();
    if (probed) {
      replacement = *probed;
    }
    script.caller_returned.release();
  }};
  script.caller_started.acquire();
  const bool returned_while_contended = script.caller_returned.try_acquire_for(50ms);
  script.release.release();
  if (!returned_while_contended) {
    script.caller_returned.acquire();
  }
  caller.join();
  holder.join();

  EXPECT_TRUE(returned_while_contended);
  EXPECT_TRUE(probe_succeeded);
  EXPECT_FALSE(replacement.has_value());
  if (replacement) {
    arrangement.session->release_freshness_reservation(*replacement);
  } else {
    arrangement.session->release_freshness_reservation(*optional_reserved);
  }
  ASSERT_TRUE(arrangement.session->retire_activation(VertexId{1U}, 1U, true));
  arrangement.session->cancel();
  ASSERT_TRUE(arrangement.session->join());
}

// REQ: RUNTIME-017
// REQ: CHAN-131
// A promoted refill carries no claim until its readiness publishes, and pre-witness replacement
// would otherwise admit exactly that shape. So a fast producer could take back the very record the
// eviction gave up a retained one to obtain, and because the replacement is captured after the
// witness the eviction was justified against, it sorts behind that witness and the activation
// selects nothing. Measured at 89.8% of lost selections against 6.1% of all refills.
TEST(RuntimeChannelBindingTemporal, APromotedRefillIsNotReplacedBeforeItsReadinessPublishes) {
  constexpr std::int64_t kRetainedCaptureNs = 10;
  constexpr std::int64_t kRefillCaptureNs = 20;
  constexpr std::int64_t kIntruderCaptureNs = 30;

  RetainedRefillArrangement arrangement;
  ASSERT_NO_FATAL_FAILURE(arrange_retained_refill(arrangement, kRetainedCaptureNs));

  ASSERT_TRUE(arrangement.emit(22, kRefillCaptureNs));
  auto reserved = arrangement.session->reserve_freshness_refill(
      PortId{2U}, arrangement.retained, TimePoint{kRefillCaptureNs, ClockId{1U}});
  ASSERT_TRUE(reserved) << runtime_error_message(reserved.error());
  const auto& optional_reserved = *reserved;
  HOLOSCAN_ASSERT_HAS_VALUE(optional_reserved) << "Expected freshness reservation";

  std::uint32_t promoted = 0U;
  ASSERT_TRUE(arrangement.session->retire_samples(
      PortId{2U},
      std::span<const TemporalSampleRecord>{&arrangement.retained, 1U},
      &promoted,
      &*optional_reserved));
  ASSERT_EQ(promoted, 1U);
  auto protected_outcome = arrangement.session->last_refill_outcome(PortId{2U});
  ASSERT_TRUE(protected_outcome) << runtime_error_message(protected_outcome.error());
  ASSERT_EQ(*protected_outcome, 0U)
      << "post-handoff capacity probes must not report a replacement attempt";

  // The refill now occupies the typed slot and has not yet published. This is the window the
  // producer used to win. Unlike the post-handoff capacity query above, this publish reaches the
  // staged-promotion path and genuinely attempts to replace the protected refill.
  ASSERT_TRUE(arrangement.emit(33, kIntruderCaptureNs));
  protected_outcome = arrangement.session->last_refill_outcome(PortId{2U});
  ASSERT_TRUE(protected_outcome) << runtime_error_message(protected_outcome.error());
  EXPECT_EQ(*protected_outcome, 2U)
      << "the rejected replacement must be observable as refill protection";

  // Key assertion: the readiness that surfaces is the refill the eviction was traded for, not the
  // arrival that came after it.
  auto refilled_event = wait_for_event(*arrangement.session, *arrangement.binding);
  ASSERT_TRUE(refilled_event) << runtime_error_message(refilled_event.error());
  auto refilled =
      arrangement.session->temporal_sample(refilled_event->port, refilled_event->source_sequence);
  ASSERT_TRUE(refilled) << runtime_error_message(refilled.error());
  EXPECT_EQ(refilled->capture_time.timestamp_ns, kRefillCaptureNs);

  ASSERT_TRUE(arrangement.binding->retire(refilled_event->source_sequence));
  ASSERT_TRUE(arrangement.session->retire_activation(VertexId{1U}, 1U, true));
  arrangement.session->cancel();
  ASSERT_TRUE(arrangement.session->join());
}

struct ReuseReplacementRollbackState {
  RuntimeChannelSession* session{};
  bool retire_newer{};
  bool observed{};
  expected<std::size_t, Error> reservations = make_unexpected(Error{ErrorCode::kNotReady});
  expected<RuntimeChannelInputResetState, Error> reserved_state =
      make_unexpected(Error{ErrorCode::kNotReady});
  expected<void, Error> retirement;

  static void observe(void* const opaque) noexcept {
    auto& state = *static_cast<ReuseReplacementRollbackState*>(opaque);
    state.observed = true;
    state.reservations =
        RuntimeChannelSessionTestAccess::typed_ingress_reservations(*state.session, PortId{2U});
    state.reserved_state =
        RuntimeChannelSessionTestAccess::input_reset_state(*state.session, PortId{2U});
    if (state.retire_newer) {
      state.retirement = state.session->retire_activation(VertexId{2U}, 2U, true);
    }
  }
};

void run_reuse_replacement_rollback_case(const bool retire_newer) {
  SessionFixture fixture;
  fixture.use_two_slot_blocking_queue();
  fixture.enable_reuse_last();
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto binding = (*session)->create_readiness_binding(
      VertexId{2U},
      RuntimePortBindingRequest{.plan_generation = fixture.plan.plan_generation,
                                .run_generation = 1U,
                                .kind = RuntimeEventKind::kInputArrival,
                                .source = make_source_id(1U),
                                .source_clock = ClockId{1U},
                                .reference_clock = ClockId{1U},
                                .capacity = 2U});
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());
  ASSERT_TRUE((*session)->refresh_connections());

  // Keep the session disarmed so the cold seam schedules the ownership transition exactly.
  // Both payloads still enter, become selected, and retire through the production Channel paths.
  constexpr std::array<std::int32_t, 2U> values{11, 22};
  std::array<RuntimeEvent, 2U> events{};
  std::array<TemporalSampleRecord, 2U> records{};
  for (std::size_t index = 0U; index < records.size(); ++index) {
    const auto generation = static_cast<std::uint64_t>(index + 1U);
    ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, generation, {}));
    ASSERT_TRUE(fixture.source.output.emit(values[index]));
    ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, generation, true));
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U}));
    auto event = wait_for_event(**session, **binding);
    ASSERT_TRUE(event) << runtime_error_message(event.error());
    events[index] = *event;
    auto record = (*session)->temporal_sample(event->port, event->source_sequence);
    ASSERT_TRUE(record) << runtime_error_message(record.error());
    records[index] = *record;
    const std::array selected{RuntimeChannelActivationSelection{
        .port = PortId{2U},
        .selection = {InputState::kSelected, SelectionReason::kNone},
        .samples = std::span<const TemporalSampleRecord>{&records[index], 1U},
        .destructive = true}};
    ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, generation, selected));
    {
      auto received = fixture.sink.input.receive();
      ASSERT_TRUE(received) << runtime_error_message(received.error());
      EXPECT_EQ(received->data, values[index]);
    }
    ASSERT_TRUE((*binding)->retire(event->source_sequence));
    if (index == 0U) {
      ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, generation, true));
      // Model released selector history while the independently owned ReuseLast fallback lives.
      ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_single_occupied_claim_state(
          **session, PortId{2U}, false, true, false, false));
    }
  }

  ReuseReplacementRollbackState observation{.session = session->get(),
                                            .retire_newer = retire_newer};
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::with_reuse_replacement_reservation(
      **session, PortId{2U}, &observation, &ReuseReplacementRollbackState::observe));
  ASSERT_TRUE(observation.observed);
  ASSERT_TRUE(observation.reservations) << runtime_error_message(observation.reservations.error());
  EXPECT_EQ(*observation.reservations, 1U);
  ASSERT_TRUE(observation.reserved_state)
      << runtime_error_message(observation.reserved_state.error());
  EXPECT_EQ(observation.reserved_state->occupied_samples, 2U);
  EXPECT_FALSE(observation.reserved_state->reuse_retained);
  EXPECT_FALSE(observation.reserved_state->reuse_slot_matches_single_occupied);
  ASSERT_TRUE(observation.retirement) << runtime_error_message(observation.retirement.error());

  const auto reservations =
      RuntimeChannelSessionTestAccess::typed_ingress_reservations(**session, PortId{2U});
  ASSERT_TRUE(reservations) << runtime_error_message(reservations.error());
  EXPECT_EQ(*reservations, 0U);
  if (!retire_newer) {
    ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 2U, false));
    ASSERT_TRUE((*session)->retire_samples(
        PortId{2U}, std::span<const TemporalSampleRecord>{&records[1U], 1U}));
  }
  const auto retained = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
  ASSERT_TRUE(retained) << runtime_error_message(retained.error());
  EXPECT_EQ(retained->occupied_samples, 1U);
  EXPECT_TRUE(retained->reuse_retained);
  EXPECT_TRUE(retained->reuse_slot_matches_single_occupied);

  const std::size_t retained_index = retire_newer ? 1U : 0U;
  const auto discarded =
      (*session)->temporal_sample(PortId{2U}, events[1U - retained_index].source_sequence);
  ASSERT_FALSE(discarded);
  EXPECT_EQ(discarded.error().code, ErrorCode::kNotFound);
  const auto survivor =
      (*session)->temporal_sample(PortId{2U}, events[retained_index].source_sequence);
  ASSERT_TRUE(survivor) << runtime_error_message(survivor.error());
  EXPECT_EQ(*survivor, records[retained_index]);
  const std::array reused{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = {InputState::kReused, SelectionReason::kTooStale},
      .samples = std::span<const TemporalSampleRecord>{&records[retained_index], 1U},
      .destructive = true}};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 3U, reused));
  {
    auto received = fixture.sink.input.receive();
    ASSERT_TRUE(received) << runtime_error_message(received.error());
    EXPECT_EQ(received->data, values[retained_index]);
    EXPECT_EQ(received->metadata.sequence, retained_index + 1U);
  }
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 3U, true));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// REQ: CHAN-146
// REQ: RUNTIME-017
// A failed receive restores the prior fallback when no activation has installed a newer one.
TEST(RuntimeChannelBindingTemporal, ReuseReplacementRollbackRestoresPriorFallback) {
  ASSERT_NO_FATAL_FAILURE(run_reuse_replacement_rollback_case(false));
}

// REQ: CHAN-146
// REQ: RUNTIME-017
// A rollback after activation retirement preserves the newly published fallback and drops the
// obsolete reservation; the newer payload remains available to the next activation.
TEST(RuntimeChannelBindingTemporal, ReuseReplacementRollbackPreservesNewerRetiredFallback) {
  ASSERT_NO_FATAL_FAILURE(run_reuse_replacement_rollback_case(true));
}

// REQ: CHAN-136
// REQ: RUNTIME-050
// A direct freshness handoff replaces the Channel payload before the optional process-local retry.
// If that retry fails, selector rollback would resurrect metadata for the destroyed payload. Model
// the controller's two-party ordering and prove the retry observes two already-committed ledgers.
TEST(RuntimeChannelBindingTemporal, PostHandoffRetryFailureCannotRestoreTheReplacedReuseIdentity) {
  constexpr std::int64_t kRetainedCaptureNs = 10;
  constexpr std::int64_t kSuccessorCaptureNs = 20;
  RetainedRefillArrangement arrangement;
  ASSERT_NO_FATAL_FAILURE(arrange_retained_refill(arrangement, kRetainedCaptureNs));

  auto selection = arrangement.fixture.plan.input_selections.front();
  selection.missing = TemporalRecordMissingKind::kReuseLast;
  selection.reuse_max_age_ns = 1000;
  auto selector = RuntimeInputSelectionState::create(selection);
  ASSERT_TRUE(selector) << runtime_error_message(selector.error());
  ASSERT_TRUE(selector->push(arrangement.retained));
  auto selected = selector->begin(TimePoint{kRetainedCaptureNs, ClockId{1U}}, 1U, 1U);
  ASSERT_TRUE(selected) << runtime_error_message(selected.error());
  ASSERT_TRUE(selected->finish(true));
  ASSERT_EQ(selector->reuse_retained_record(), arrangement.retained);

  ASSERT_TRUE(arrangement.emit(22, kSuccessorCaptureNs));
  auto channel_reservation = arrangement.session->reserve_freshness_refill(
      PortId{2U}, arrangement.retained, TimePoint{kSuccessorCaptureNs, ClockId{1U}});
  ASSERT_TRUE(channel_reservation) << runtime_error_message(channel_reservation.error());
  const auto& optional_channel_reservation = *channel_reservation;
  HOLOSCAN_ASSERT_HAS_VALUE(optional_channel_reservation) << "Expected freshness reservation";
  auto selector_reservation = selector->reserve_reuse_replacement(arrangement.retained);
  ASSERT_TRUE(selector_reservation) << runtime_error_message(selector_reservation.error());
  ASSERT_TRUE(
      RuntimeChannelSessionTestAccess::fail_next_freshness_handoff_retry(*arrangement.session));

  std::uint32_t promoted = 0U;
  ASSERT_TRUE(arrangement.session->retire_samples(
      PortId{2U},
      std::span<const TemporalSampleRecord>{&arrangement.retained, 1U},
      &promoted,
      &*optional_channel_reservation));
  ASSERT_EQ(promoted, 1U);
  selector_reservation->commit();
  auto retried = arrangement.session->retry_after_freshness_handoff();
  ASSERT_FALSE(retried);
  EXPECT_EQ(retried.error().code, ErrorCode::kProtocolError);

  std::array<TemporalSampleRecord, 1U> selector_claims{};
  auto selector_claim_count = selector->copy_payload_claims(selector_claims);
  ASSERT_TRUE(selector_claim_count) << runtime_error_message(selector_claim_count.error());
  EXPECT_EQ(*selector_claim_count, 0U);
  auto channel_state =
      RuntimeChannelSessionTestAccess::input_reset_state(*arrangement.session, PortId{2U});
  ASSERT_TRUE(channel_state) << runtime_error_message(channel_state.error());
  EXPECT_FALSE(channel_state->reuse_retained);
  EXPECT_FALSE(channel_state->reuse_slot_matches_single_occupied);
  EXPECT_EQ(channel_state->occupied_samples, 1U);

  // Failed-run cleanup may discard the committed successor, but it must not reconstruct N.
  arrangement.session->cancel();
  ASSERT_TRUE(arrangement.session->join());
}

// REQ: RUNTIME-017
// poll() abandons its pass on a contended cache lock, reporting no error and creating no event.
// That is right for the ordinary pass, which must never wait on a producer and can retry for free.
// It is wrong directly after a freshness eviction, where the retained record is already released
// and the refill is held unpublished until a poll publishes it: there the abandoned pass costs an
// activation rather than deferring one. poll_blocking() is the same pass that waits instead.
TEST(RuntimeChannelBindingTemporal, PollBlockingCompletesAPassAContendedPollAbandons) {
  constexpr std::int64_t kRetainedCaptureNs = 10;
  constexpr std::int64_t kRefillCaptureNs = 20;

  RetainedRefillArrangement arrangement;
  ASSERT_NO_FATAL_FAILURE(arrange_retained_refill(arrangement, kRetainedCaptureNs));

  ASSERT_TRUE(arrangement.emit(22, kRefillCaptureNs));
  auto reserved = arrangement.session->reserve_freshness_refill(
      PortId{2U}, arrangement.retained, TimePoint{kRefillCaptureNs, ClockId{1U}});
  ASSERT_TRUE(reserved) << runtime_error_message(reserved.error());
  const auto& optional_reserved = *reserved;
  HOLOSCAN_ASSERT_HAS_VALUE(optional_reserved) << "Expected freshness reservation";
  std::uint32_t promoted = 0U;
  ASSERT_TRUE(arrangement.session->retire_samples(
      PortId{2U},
      std::span<const TemporalSampleRecord>{&arrangement.retained, 1U},
      &promoted,
      &*optional_reserved));
  ASSERT_EQ(promoted, 1U);

  struct LockScript {
    std::binary_semaphore held{0};
    std::binary_semaphore release{0};
    std::binary_semaphore waiter_started{0};
  } script;

  // Hold the cache lock the poll must acquire, from another thread, so the contention is arranged
  // rather than raced for.
  std::thread holder{[&] {
    static_cast<void>(RuntimeChannelSessionTestAccess::with_input_lock(
        *arrangement.session, PortId{2U}, &script, [](void* state) noexcept {
          auto& owned = *static_cast<LockScript*>(state);
          owned.held.release();
          owned.release.acquire();
        }));
  }};
  script.held.acquire();

  // The contended pass reports success and does nothing, which is the outcome that cannot be told
  // from "nothing was ready" without asking.
  const TimePoint observed{kRefillCaptureNs + 1, ClockId{1U}};
  ASSERT_TRUE(arrangement.binding->poll(observed));
  EXPECT_TRUE(arrangement.binding->last_poll_incomplete());

  // The blocking pass waits out the holder and completes. It is issued from another thread because
  // this one has to release the lock for it to finish.
  std::atomic<bool> blocking_poll_succeeded{false};
  std::thread waiter{[&] {
    script.waiter_started.release();
    blocking_poll_succeeded.store(arrangement.binding->poll_blocking(observed).has_value(),
                                  std::memory_order_relaxed);
  }};
  script.waiter_started.acquire();
  // Bound how long the wrong behaviour has to reveal itself. A pass that does not wait returns
  // within microseconds and reports itself incomplete; one that waits is still parked on the
  // holder, so this delay only ever delays the correct case. Without it the waiter can start after
  // the lock is already free, and a non-blocking pass would pass this test.
  std::this_thread::sleep_for(std::chrono::milliseconds{50});
  script.release.release();
  waiter.join();
  holder.join();

  EXPECT_TRUE(blocking_poll_succeeded.load(std::memory_order_relaxed));
  EXPECT_FALSE(arrangement.binding->last_poll_incomplete());
  auto event = wait_for_event(*arrangement.session, *arrangement.binding);
  ASSERT_TRUE(event) << runtime_error_message(event.error());
  auto record = arrangement.session->temporal_sample(event->port, event->source_sequence);
  ASSERT_TRUE(record) << runtime_error_message(record.error());
  EXPECT_EQ(record->capture_time.timestamp_ns, kRefillCaptureNs);

  ASSERT_TRUE(arrangement.binding->retire(event->source_sequence));
  ASSERT_TRUE(arrangement.session->retire_activation(VertexId{1U}, 1U, true));
  arrangement.session->cancel();
  ASSERT_TRUE(arrangement.session->join());
}

// REQ: CHAN-131
// A LatestWithin prefix whose oldest record carries a capture time can still hold a capture-less
// record before its target. Every prefix record is checked before the first prefix step commits,
// so the reservation declines instead of committing a prefix whose next step has no coordinate.
// Once newer captured arrivals replace the staged records, the trade proceeds through the prefix.
//
//   typed: 1 2 3    staged: 4 (capture 40), 5 (no capture), 6 (capture 60)
TEST(RuntimeChannelBindingTemporal, MixedCapturePrefixDeclinesUntilItsCapturelessRecordIsReplaced) {
  SessionFixture fixture;
  fixture.enable_scalar(InputSelectorKind::kLatestWithin, TemporalRecordMissingKind::kSkip);
  fixture.use_drop_oldest_queue_policy();
  fixture.set_queue_depth(0U, 3U);
  fixture.use_direct_retained_delivery();
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto binding =
      (*session)->create_readiness_binding(VertexId{2U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.plan.plan_generation,
                                               .run_generation = 1U,
                                               .kind = RuntimeEventKind::kInputArrival,
                                               .source = make_source_id(1U),
                                               .source_clock = ClockId{1U},
                                               .reference_clock = ClockId{1U},
                                               .capacity = 8U,
                                           });
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  const auto emit = [&](const std::int32_t value, const std::int64_t capture_ns) {
    return fixture.source.output.emit(Sample<std::int32_t>{
        value,
        SampleMetadata{.capture_timestamp_ns = capture_ns, .source_clock_id = ClockId{1U}},
    });
  };
  const auto reserve = [&](const TemporalSampleRecord& retained, const std::int64_t floor_ns) {
    return (*session)->reserve_freshness_refill(
        PortId{2U},
        retained,
        TimePoint{floor_ns, ClockId{1U}},
        RuntimeChannelFreshnessSelectionMode::kNewestWithArrivalPrefix);
  };
  std::array<TemporalSampleRecord, 3U> retained;
  for (std::size_t index = 0U; index < retained.size(); ++index) {
    const auto value = static_cast<std::int32_t>(index + 1U);
    ASSERT_TRUE(emit(value, value * 10));
    auto event = wait_for_event(**session, **binding);
    ASSERT_TRUE(event) << runtime_error_message(event.error());
    auto record = (*session)->temporal_sample(event->port, event->source_sequence);
    ASSERT_TRUE(record) << runtime_error_message(record.error());
    retained[index] = *record;
    ASSERT_TRUE((*binding)->retire(event->source_sequence));
  }
  ASSERT_TRUE(emit(4, 40));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{5}));
  ASSERT_TRUE(emit(6, 60));

  // Key assertion: the capture-less record behind a captured oldest one declines the trade.
  auto declined = reserve(retained[2], 60);
  ASSERT_TRUE(declined) << runtime_error_message(declined.error());
  EXPECT_FALSE(declined->has_value());

  // Newer captured arrivals, from the producer's next activation, replace every staged record,
  // and the trade then walks the prefix.
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 2U, {}));
  for (const std::int32_t value : {7, 8, 9}) {
    auto emitted = emit(value, value * 10);
    ASSERT_TRUE(emitted) << runtime_error_message(emitted.error());
  }
  auto prefix = reserve(retained[2], 90);
  ASSERT_TRUE(prefix) << runtime_error_message(prefix.error());
  const auto& optional_prefix = *prefix;
  HOLOSCAN_ASSERT_HAS_VALUE(optional_prefix) << "Expected freshness reservation";
  EXPECT_EQ(optional_prefix->successor.capture_time.timestamp_ns, 70);
  EXPECT_EQ(optional_prefix->target.capture_time.timestamp_ns, 90);
  std::uint32_t promoted = 0U;
  ASSERT_TRUE((*session)->retire_samples(PortId{2U},
                                         std::span<const TemporalSampleRecord>{&retained[2], 1U},
                                         &promoted,
                                         &*optional_prefix));
  ASSERT_EQ(promoted, 1U);
  auto event = wait_for_event(**session, **binding);
  ASSERT_TRUE(event) << runtime_error_message(event.error());
  auto first = (*session)->temporal_sample(event->port, event->source_sequence);
  ASSERT_TRUE(first) << runtime_error_message(first.error());
  ASSERT_TRUE((*binding)->retire(event->source_sequence));
  auto next = reserve(*first, 90);
  static_cast<void>((*session)->retire_activation(VertexId{1U}, 2U, true));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
  ASSERT_TRUE(next) << runtime_error_message(next.error());
  const auto& optional_next = *next;
  HOLOSCAN_ASSERT_HAS_VALUE(optional_next) << "Expected freshness reservation";
  EXPECT_EQ(optional_next->successor.capture_time.timestamp_ns, 80);
  EXPECT_EQ(optional_next->target.capture_time.timestamp_ns, 90);
}

// REQ: CHAN-131
// The freshness candidate probe counts a capture-less staged successor only for a drain, whose
// reservation gives it a coordinate. Every other selection mode declines that successor, so the
// probe declines it too; otherwise a capture-less stream would attempt a reservation every round.
TEST(RuntimeChannelBindingTemporal, FreshnessProbeCountsCapturelessSuccessorsOnlyForADrain) {
  RetainedRefillArrangement arrangement;
  ASSERT_NO_FATAL_FAILURE(arrange_retained_refill(arrangement, 10));
  ASSERT_TRUE(arrangement.fixture.source.output.emit(std::int32_t{22}));

  auto drain =
      arrangement.session->has_freshness_refill_candidate(PortId{2U}, arrangement.retained, true);
  ASSERT_TRUE(drain) << runtime_error_message(drain.error());
  EXPECT_TRUE(*drain);
  // Key assertion: outside a drain the probe agrees with the reservation, which declines.
  auto scalar =
      arrangement.session->has_freshness_refill_candidate(PortId{2U}, arrangement.retained, false);
  ASSERT_TRUE(scalar) << runtime_error_message(scalar.error());
  EXPECT_FALSE(*scalar);
  auto reservation =
      arrangement.session->reserve_freshness_refill(PortId{2U},
                                                    arrangement.retained,
                                                    TimePoint{20, ClockId{1U}},
                                                    RuntimeChannelFreshnessSelectionMode::kNewest);
  ASSERT_TRUE(reservation) << runtime_error_message(reservation.error());
  EXPECT_FALSE(reservation->has_value());

  ASSERT_TRUE(arrangement.session->retire_activation(VertexId{1U}, 1U, true));
  arrangement.session->cancel();
  ASSERT_TRUE(arrangement.session->join());
}

// A tick-reconciled drop-oldest DrainAll input stages one survivor beyond its authored depth. A
// publication that races the tick then displaces an older staged survivor rather than the newest
// one at or before the tick, so the tick's drain cut admits it; at the authored depth the trade
// declines and the tick drains the stale retained sample. Once both staged survivors are eligible,
// the cut targets the newer one and admits its producer-order prefix from the older one.
// REQ: RUNTIME-088
// REQ: RUNTIME-089
// REQ: CHAN-131
TEST(RuntimeChannelBindingTemporal, TickReconciledDrainCutAdmitsTheNewestSurvivorAtTheTick) {
  constexpr std::int64_t kTickNs = 200;
  for (const bool extra_survivor : {false, true}) {
    SCOPED_TRACE(extra_survivor ? "extra staging survivor" : "authored depth");
    SessionFixture fixture;
    fixture.enable_drain_all(1U);
    fixture.use_one_slot_drop_oldest_queue();
    fixture.use_direct_retained_delivery();
    const auto counters = std::make_shared<RuntimePhaseCounters>();
    fixture.plan.executable_factories.phase_counters = counters;
    auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
    ASSERT_TRUE(session) << runtime_error_message(session.error());
    if (extra_survivor) {
      auto enabled = (*session)->enable_tick_reconciled_staging(PortId{2U});
      ASSERT_TRUE(enabled) << runtime_error_message(enabled.error());
      ASSERT_TRUE(*enabled);
    }
    auto binding =
        (*session)->create_readiness_binding(VertexId{2U},
                                             RuntimePortBindingRequest{
                                                 .plan_generation = fixture.plan.plan_generation,
                                                 .run_generation = 1U,
                                                 .kind = RuntimeEventKind::kInputArrival,
                                                 .source = make_source_id(1U),
                                                 .source_clock = ClockId{1U},
                                                 .reference_clock = ClockId{1U},
                                                 .capacity = 8U,
                                             });
    ASSERT_TRUE(binding) << runtime_error_message(binding.error());
    ASSERT_TRUE((*session)->refresh_connections());
    ASSERT_TRUE((*session)->arm());
    ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
    const auto emit_captured = [&](const std::int32_t value, const std::int64_t capture_ns) {
      return fixture.source.output.emit(Sample<std::int32_t>{
          value,
          SampleMetadata{.capture_timestamp_ns = capture_ns, .source_clock_id = ClockId{1U}}});
    };
    ASSERT_TRUE(emit_captured(11, 10));
    auto retained_event = wait_for_event(**session, **binding);
    ASSERT_TRUE(retained_event) << runtime_error_message(retained_event.error());
    auto retained =
        (*session)->temporal_sample(retained_event->port, retained_event->source_sequence);
    ASSERT_TRUE(retained) << runtime_error_message(retained.error());
    ASSERT_TRUE((*binding)->retire(retained_event->source_sequence));

    // Key setup: the sole typed slot holds the retained sample, so later samples stage. The last
    // one is captured after the tick, as a publication racing the tick scan would be.
    ASSERT_TRUE(emit_captured(22, 100));
    ASSERT_TRUE(emit_captured(33, 150));
    ASSERT_TRUE(emit_captured(44, 250));
    auto raced =
        (*session)->reserve_freshness_refill(PortId{2U},
                                             *retained,
                                             TimePoint{kTickNs, ClockId{1U}},
                                             RuntimeChannelFreshnessSelectionMode::kDrainCut);
    ASSERT_TRUE(raced) << runtime_error_message(raced.error());

    // Key assertion: only the extra survivor keeps 33, the newest sample at or before the tick.
    const auto& optional_raced = *raced;
    ASSERT_EQ(optional_raced.has_value(), extra_survivor);
    const std::uint64_t declined = counters->snapshot()[RuntimePhaseCounters::index(
        RuntimePhaseCounter::kChannelFreshnessDeclinedStagedAboveFloor)];
    if (!extra_survivor) {
      EXPECT_EQ(declined, 1U);
    } else {
      HOLOSCAN_ASSERT_HAS_VALUE(optional_raced) << "Expected refill with the extra survivor";
      EXPECT_EQ(declined, 0U);
      EXPECT_EQ(optional_raced->successor.capture_time.timestamp_ns, 150);
      EXPECT_EQ(optional_raced->target.capture_time.timestamp_ns, 150);
      (*session)->release_freshness_reservation(*optional_raced);

      // Key assertion: with both staged survivors eligible, the cut targets the newer one and
      // admits the older one first, so the batch ends at the newest survivor in producer order.
      auto later =
          (*session)->reserve_freshness_refill(PortId{2U},
                                               *retained,
                                               TimePoint{300, ClockId{1U}},
                                               RuntimeChannelFreshnessSelectionMode::kDrainCut);
      ASSERT_TRUE(later) << runtime_error_message(later.error());
      const auto& optional_later = *later;
      HOLOSCAN_ASSERT_HAS_VALUE(optional_later) << "Expected freshness reservation";
      EXPECT_EQ(optional_later->successor.capture_time.timestamp_ns, 150);
      EXPECT_EQ(optional_later->target.capture_time.timestamp_ns, 250);
      (*session)->release_freshness_reservation(*optional_later);
    }
    ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
    (*session)->cancel();
    ASSERT_TRUE((*session)->join());
  }
}

// A tick scan that runs late can find several publications past the tick. Staging keeps one
// survivor beyond the authored depth, and with the owner's next tick published it keeps the newest
// survivor at or before that tick while later publications displace each other, so the late
// scan's drain cut still admits it. Without the fence staging keeps only the newest survivors, all
// past the tick, and the trade declines.
// REQ: RUNTIME-089
TEST(RuntimeChannelBindingTemporal, TickReconciledStagingKeepsTheNewestSurvivorAtOrBeforeTheTick) {
  constexpr std::int64_t kTickNs = 200;
  for (const bool fenced : {false, true}) {
    SCOPED_TRACE(fenced ? "fenced" : "unfenced");
    SessionFixture fixture;
    fixture.enable_drain_all(1U);
    fixture.use_one_slot_drop_oldest_queue();
    fixture.use_direct_retained_delivery();
    const auto counters = std::make_shared<RuntimePhaseCounters>();
    fixture.plan.executable_factories.phase_counters = counters;
    auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
    ASSERT_TRUE(session) << runtime_error_message(session.error());
    auto enabled = (*session)->enable_tick_reconciled_staging(PortId{2U});
    ASSERT_TRUE(enabled) << runtime_error_message(enabled.error());
    ASSERT_TRUE(*enabled);
    auto binding =
        (*session)->create_readiness_binding(VertexId{2U},
                                             RuntimePortBindingRequest{
                                                 .plan_generation = fixture.plan.plan_generation,
                                                 .run_generation = 1U,
                                                 .kind = RuntimeEventKind::kInputArrival,
                                                 .source = make_source_id(1U),
                                                 .source_clock = ClockId{1U},
                                                 .reference_clock = ClockId{1U},
                                                 .capacity = 8U,
                                             });
    ASSERT_TRUE(binding) << runtime_error_message(binding.error());
    ASSERT_TRUE((*session)->refresh_connections());
    ASSERT_TRUE((*session)->arm());
    ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
    const auto emit_captured = [&](const std::int32_t value, const std::int64_t capture_ns) {
      return fixture.source.output.emit(Sample<std::int32_t>{
          value,
          SampleMetadata{.capture_timestamp_ns = capture_ns, .source_clock_id = ClockId{1U}}});
    };
    ASSERT_TRUE(emit_captured(11, 10));
    auto retained_event = wait_for_event(**session, **binding);
    ASSERT_TRUE(retained_event) << runtime_error_message(retained_event.error());
    auto retained =
        (*session)->temporal_sample(retained_event->port, retained_event->source_sequence);
    ASSERT_TRUE(retained) << runtime_error_message(retained.error());
    ASSERT_TRUE((*binding)->retire(retained_event->source_sequence));
    if (fenced) {
      (*session)->set_tick_fence(PortId{2U}, TimePoint{kTickNs, ClockId{1U}});
    }

    // Key setup: the sole typed slot holds the retained sample, so later samples stage. One is
    // captured before the tick and three after it, as publications during a delayed scan would be.
    ASSERT_TRUE(emit_captured(22, 150));
    ASSERT_TRUE(emit_captured(33, 250));
    ASSERT_TRUE(emit_captured(44, 260));
    ASSERT_TRUE(emit_captured(55, 270));
    auto late =
        (*session)->reserve_freshness_refill(PortId{2U},
                                             *retained,
                                             TimePoint{kTickNs, ClockId{1U}},
                                             RuntimeChannelFreshnessSelectionMode::kDrainCut);
    ASSERT_TRUE(late) << runtime_error_message(late.error());

    // Key assertion: only the fenced input still holds 22, the newest sample at or before the tick.
    const auto& optional_late = *late;
    ASSERT_EQ(optional_late.has_value(), fenced);
    const std::uint64_t declined = counters->snapshot()[RuntimePhaseCounters::index(
        RuntimePhaseCounter::kChannelFreshnessDeclinedStagedAboveFloor)];
    if (!fenced) {
      EXPECT_EQ(declined, 1U);
    } else {
      HOLOSCAN_ASSERT_HAS_VALUE(optional_late) << "Expected refill with the tick survivor";
      EXPECT_EQ(declined, 0U);
      EXPECT_EQ(optional_late->successor.capture_time.timestamp_ns, 150);
      EXPECT_EQ(optional_late->target.capture_time.timestamp_ns, 150);
      (*session)->release_freshness_reservation(*optional_late);
    }
    ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
    (*session)->cancel();
    ASSERT_TRUE((*session)->join());
  }
}

// A long-parked owner witnesses its tick-reconciled input only at the tick scan, so between ticks
// promotion replaces the unwitnessed typed record with each newer one. A publication racing the
// tick would replace the newest record at or before the tick and leave the tick nothing eligible.
// The owner's published next tick fences that promotion: the record published after it stays
// staged, and the tick's poll witnesses the record published before it. Without the fence the
// typed record is the later one.
// REQ: RUNTIME-088
// REQ: RUNTIME-089
TEST(RuntimeChannelBindingTemporal, TickFencedPromotionKeepsTheNewestRecordAtOrBeforeTheTick) {
  constexpr std::int64_t kTickNs = 200;
  for (const bool fenced : {false, true}) {
    SCOPED_TRACE(fenced ? "fenced" : "unfenced");
    ClockRegistryBuilder registry_builder;
    const ClockKey publish_key = ClockKey::named("publish-domain");
    const auto publish_source = std::make_shared<ManualClock>(150);
    ASSERT_TRUE(registry_builder.add_clock(
        publish_key, publish_source, ClockCapabilityOptions{.can_read_now = true, .steady = true}));
    ASSERT_TRUE(registry_builder.set_default(publish_key));
    auto registry = registry_builder.build();
    ASSERT_TRUE(registry) << runtime_error_message(registry.error());
    auto publish_clock = registry->resolve(publish_key);
    ASSERT_TRUE(publish_clock) << runtime_error_message(publish_clock.error());
    const ClockId bound = publish_clock->id();

    SessionFixture fixture;
    fixture.enable_drain_all(1U);
    fixture.use_one_slot_drop_oldest_queue();
    fixture.use_direct_retained_delivery();
    fixture.plan.input_selections.front().reference_clock = bound;
    const auto counters = std::make_shared<RuntimePhaseCounters>();
    fixture.plan.executable_factories.phase_counters = counters;
    auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
    ASSERT_TRUE(session) << runtime_error_message(session.error());
    ASSERT_TRUE((*session)->bind_publish_clock(*publish_clock));
    auto ordered = (*session)->order_capture_absent_arrivals_by_publication(PortId{2U});
    ASSERT_TRUE(ordered) << runtime_error_message(ordered.error());
    ASSERT_TRUE(*ordered);
    auto enabled = (*session)->enable_tick_reconciled_staging(PortId{2U});
    ASSERT_TRUE(enabled) << runtime_error_message(enabled.error());
    ASSERT_TRUE(*enabled);
    if (fenced) {
      (*session)->set_tick_fence(PortId{2U}, TimePoint{kTickNs, bound});
    }
    auto binding =
        (*session)->create_readiness_binding(VertexId{2U},
                                             RuntimePortBindingRequest{
                                                 .plan_generation = fixture.plan.plan_generation,
                                                 .run_generation = 1U,
                                                 .kind = RuntimeEventKind::kInputArrival,
                                                 .source = make_source_id(1U),
                                                 .source_clock = bound,
                                                 .reference_clock = bound,
                                                 .capacity = 2U,
                                             });
    ASSERT_TRUE(binding) << runtime_error_message(binding.error());
    ASSERT_TRUE((*session)->refresh_connections());
    ASSERT_TRUE((*session)->arm());
    ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));

    // Key setup: 11 is published before the tick and fills the unwitnessed typed slot; 22 is
    // published after it, as a publication racing the tick scan would be.
    ASSERT_TRUE(fixture.source.output.emit(std::int32_t{11}));
    publish_source->advance_to(250);
    ASSERT_TRUE(fixture.source.output.emit(std::int32_t{22}));
    ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));

    ASSERT_TRUE((*binding)->poll(TimePoint{.timestamp_ns = 260, .clock_id = bound}));
    RuntimeEvent event;
    auto drained = (*binding)->drain(std::span<RuntimeEvent>{&event, 1U});
    ASSERT_TRUE(drained) << runtime_error_message(drained.error());
    ASSERT_EQ(*drained, 1U);

    // Key assertion: the fence keeps 11, published at 150, for the tick at 200.
    EXPECT_EQ(event.reference_time.timestamp_ns, fenced ? 150 : 250);
    // Each promotion pass that meets the fence defers again, so count presence, not passes.
    const std::uint64_t deferred = counters->snapshot()[RuntimePhaseCounters::index(
        RuntimePhaseCounter::kChannelTickFenceDeferredPromotions)];
    EXPECT_EQ(deferred != 0U, fenced);
    ASSERT_TRUE((*binding)->retire(event.source_sequence));
    (*session)->cancel();
    ASSERT_TRUE((*session)->join());
  }
}

}  // namespace
}  // namespace holoscan::detail

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <semaphore>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_state.hpp"
#include "core/operator_test_binding.hpp"
#include "core/source_id_internal.hpp"
#include "runtime/dispatcher.hpp"
#include "runtime/runtime_channel_binding.hpp"
#include "runtime/runtime_phase_counters.hpp"

#include "runtime_channel_binding_test_support.hpp"

namespace holoscan::detail {
namespace {

using namespace std::chrono_literals;
using namespace runtime_channel_binding_test;

struct HeldInputLock {
  std::atomic<bool> acquired{};
  std::atomic<bool> release{};
};

void hold_input_lock(void* opaque) noexcept {
  auto& state = *static_cast<HeldInputLock*>(opaque);
  state.acquired.store(true, std::memory_order_release);
  while (!state.release.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
}

struct BoundedHeldInputLock {
  std::atomic<bool> acquired{};
  std::atomic<bool> release{};
  std::atomic<bool> gave_up{};
};

// Hold the lock until released, but never past a deadline, so an unbounded waiter under test fails
// its assertions instead of hanging the suite.
void hold_input_lock_bounded(void* opaque) noexcept {
  auto& state = *static_cast<BoundedHeldInputLock*>(opaque);
  const auto deadline = std::chrono::steady_clock::now() + 2s;
  state.acquired.store(true, std::memory_order_release);
  while (!state.release.load(std::memory_order_acquire)) {
    if (std::chrono::steady_clock::now() >= deadline) {
      state.gave_up.store(true, std::memory_order_release);
      return;
    }
    std::this_thread::yield();
  }
}

// Release a held input lock as soon as a contention retry reports a failed attempt from inside
// its loop, and count those attempts.
struct ReleaseOnRetryAttempt {
  BoundedHeldInputLock* held{};
  std::atomic<std::uint32_t> attempts{};
};

void release_holder_on_retry_attempt(void* opaque, VertexId, PortId, SourceId) noexcept {
  auto& state = *static_cast<ReleaseOnRetryAttempt*>(opaque);
  state.attempts.fetch_add(1U, std::memory_order_acq_rel);
  state.held->release.store(true, std::memory_order_release);
}

struct LatestReplacementBarrier {
  std::atomic<bool> selection_saved{};
  std::atomic<bool> replacement_complete{};
  std::atomic<bool> observer_timed_out{};
};

void wait_for_latest_replacement(void* opaque) noexcept {
  auto& state = *static_cast<LatestReplacementBarrier*>(opaque);
  state.selection_saved.store(true, std::memory_order_release);
  const auto deadline = std::chrono::steady_clock::now() + 2s;
  while (!state.replacement_complete.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  state.observer_timed_out.store(!state.replacement_complete.load(std::memory_order_acquire),
                                 std::memory_order_release);
}

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

// The pump resolves ready handles and its periodic drain set through cold prebound indexes rather
// than rescanning the complete subscriber inventory on every wake.
TEST(RuntimeChannelBinding, PrebindsExactWaitAndDrainSubscriberIndexes) {
  SessionFixture fixture;
  fixture.add_blocking_fanout();
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());

  const auto indexes = RuntimeChannelSessionTestAccess::pump_index_state(**session);
  ASSERT_TRUE(indexes) << runtime_error_message(indexes.error());
  EXPECT_TRUE(indexes->exact);
  EXPECT_EQ(indexes->wait_handle_count, 2U);
  EXPECT_EQ(indexes->drain_subscriber_count, 2U);
}

// Only kDropOldest needs an auxiliary protected terminal lane. Keep the lossless kBlocking fanout
// on one wait handle per logical recipient while indexing both handles for the lossy recipient.
// REQ: CHAN-110
TEST(RuntimeChannelBinding, PrebindsProtectedTerminalHandleForDropOldestSubscribers) {
  SessionFixture fixture;
  fixture.add_mixed_policy_fanout();
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());

  const auto indexes = RuntimeChannelSessionTestAccess::pump_index_state(**session);
  ASSERT_TRUE(indexes) << runtime_error_message(indexes.error());
  EXPECT_TRUE(indexes->exact);
  EXPECT_EQ(indexes->wait_handle_count, 3U);
  EXPECT_EQ(indexes->drain_subscriber_count, 2U);
}

// A foreign transport thread publishes typed input readiness while the serial controller may be
// parked. The pump must signal only after the complete typed mutation is visible.
// REQ: RUNTIME-030
TEST(RuntimeChannelBinding, PumpSignalsControllerAfterProductiveTypedIngress) {
  SessionFixture fixture;
  // Sized with headroom rather than exactly: wait_kernel polls readiness once before it
  // blocks, so a productive scan now records one more transport-wait point than the eight
  // slots this originally held and the trace silently dropped the tail. The order assertions
  // below are the contract; the capacity is only what lets them be observed.
  std::array<RuntimePhaseTraceRecord, 16U> trace_storage;
  const auto counters = std::make_shared<RuntimePhaseCounters>(std::span{trace_storage});
  fixture.plan.executable_factories.phase_counters = counters;
  std::binary_semaphore wake{0};
  auto session = RuntimeChannelSession::create(
      fixture.plan,
      fixture.replay,
      {},
      RuntimeChannelReadinessWake{
          .state = &wake,
          .signal =
              [](void* state) noexcept { static_cast<std::binary_semaphore*>(state)->release(); },
      });
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
                                               .capacity = 2U,
                                           });
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());
  ASSERT_TRUE((*session)->refresh_connections());

  // Key setup: queue one sample before arming so only the joined pump can publish typed readiness.
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{42}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  EXPECT_FALSE(wake.try_acquire());
  ASSERT_TRUE((*session)->arm());
  ASSERT_TRUE(wake.try_acquire_for(1s));

  auto event = wait_for_event(**session, **binding);
  ASSERT_TRUE(event) << runtime_error_message(event.error());
  // Key assertion: typed readiness is observable when the pump wake arrives, and empty scans add
  // no further signal.
  EXPECT_EQ(event->kind, RuntimeEventKind::kInputArrival);
  EXPECT_FALSE(wake.try_acquire_for(20ms));

  (*session)->cancel();
  ASSERT_TRUE((*session)->join());

  std::optional<std::int64_t> transport_ready;
  std::optional<std::int64_t> ingress_retained;
  std::optional<std::int64_t> wake_signaled;
  // First occurrence of each point, not the last. The contract is the ordering within the one
  // productive scan, but the pump keeps scanning: cancel() below wakes it again and records a
  // second transport-wait point after the ingress. Taking the last occurrence compared the
  // shutdown wake against the productive retention, which only ever held because the trace buffer
  // was too small to keep that record.
  for (const RuntimePhaseTraceRecord& record : counters->trace_records()) {
    switch (record.point) {
      case RuntimePhaseTracePoint::kChannelTransportWaitReturned:
        if (!transport_ready.has_value()) {
          transport_ready = record.timestamp_ns;
        }
        break;
      case RuntimePhaseTracePoint::kChannelIngressRetained:
        if (!ingress_retained.has_value()) {
          ingress_retained = record.timestamp_ns;
        }
        EXPECT_EQ(record.source_sequence, 1U);
        break;
      case RuntimePhaseTracePoint::kChannelReadinessWakeSignaled:
        if (!wake_signaled.has_value()) {
          wake_signaled = record.timestamp_ns;
        }
        break;
      default:
        break;
    }
  }
  // Key assertion: the opt-in trace preserves the exact transport-to-retention-to-wake order.
  ASSERT_TRUE(transport_ready.has_value());
  ASSERT_TRUE(ingress_retained.has_value());
  ASSERT_TRUE(wake_signaled.has_value());
  // The fatal assertions above prove all three optionals are engaged.
  // NOLINTBEGIN(bugprone-unchecked-optional-access)
  EXPECT_LE(*transport_ready, *ingress_retained);
  EXPECT_LE(*ingress_retained, *wake_signaled);
  // NOLINTEND(bugprone-unchecked-optional-access)
  EXPECT_EQ(counters->trace_dropped(), 0U);
}

// The cold metadata-contract seam must preserve the same admission invariant as production
// create(): capture-time selectors require an exact catalog/plan clock pair. DrainAll remains
// valid without capture coordinates because it consumes arrival evidence only.
// REQ: RUNTIME-034
TEST(RuntimeChannelBinding, ExternalMetadataContractRejectsMissingClocksForCaptureSelector) {
  SessionFixture capture_fixture;
  capture_fixture.enable_scalar(InputSelectorKind::kLatestWithin,
                                TemporalRecordMissingKind::kSafeDefault);
  auto capture_session =
      RuntimeChannelSession::create(capture_fixture.plan, capture_fixture.replay);
  ASSERT_TRUE(capture_session) << runtime_error_message(capture_session.error());
  auto rejected = RuntimeChannelSessionTestAccess::set_external_metadata_contract(
      **capture_session, PortId{2U}, ClockId{}, ClockId{});
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, ErrorCode::kInvalidArgument);

  SessionFixture drain_all_fixture;
  drain_all_fixture.enable_drain_all(2U);
  auto drain_all_session =
      RuntimeChannelSession::create(drain_all_fixture.plan, drain_all_fixture.replay);
  ASSERT_TRUE(drain_all_session) << runtime_error_message(drain_all_session.error());
  EXPECT_TRUE(RuntimeChannelSessionTestAccess::set_external_metadata_contract(
      **drain_all_session, PortId{2U}, ClockId{}, ClockId{}));
}

// REQ: RUNTIME-018
// REQ: RUNTIME-034
// REQ: RUNTIME-019
TEST(ChannelTemporalDispatchTest, CommitsAndRollsBackTypedSelectionsTransactionally) {
  SessionFixture fixture;
  fixture.enable_drain_all(3U);
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

  constexpr std::array values{10, 20, 30, 40, 50, 60, 70};
  constexpr std::array<std::int64_t, values.size()> capture_times{10, 20, 30, 40, 50, 60, 70};
  auto script = publish_scalar_script(**session, **binding, fixture.source, values, capture_times);
  ASSERT_TRUE(script) << runtime_error_message(script.error());

  auto direct = RuntimeInputSelectionState::create(fixture.plan.input_selections.front());
  ASSERT_TRUE(direct);
  for (std::size_t index = 0U; index < 3U; ++index) {
    ASSERT_TRUE(direct->push(script->records[index]));
  }

  auto provisional = direct->begin(TimePoint{30, ClockId{1U}},
                                   script->records[2].arrival_sequence,
                                   script->records[2].producer_epoch);
  ASSERT_TRUE(provisional);
  ASSERT_EQ(provisional->selected().size(), 3U);
  for (std::size_t index = 0U; index < provisional->selected().size(); ++index) {
    EXPECT_EQ(provisional->selected()[index], script->records[index]);
  }
  const std::array provisional_selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = provisional->selection(),
      .samples = provisional->selected(),
      .batch = true,
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, provisional_selection));
  auto first_view = fixture.sink.input.selected_batch();
  auto repeated_view = fixture.sink.input.selected_batch();
  ASSERT_TRUE(first_view);
  ASSERT_TRUE(repeated_view);
  auto scalar_mismatch = fixture.sink.input.receive();
  ASSERT_FALSE(scalar_mismatch);
  EXPECT_EQ(scalar_mismatch.error().code, ErrorCode::kInputSelectionCardinalityMismatch);
  EXPECT_EQ(batch_values(*first_view), (std::vector<std::int32_t>{10, 20, 30}));
  EXPECT_EQ(batch_values(*repeated_view), (std::vector<std::int32_t>{10, 20, 30}));
  ASSERT_EQ(first_view->size(), repeated_view->size());
  for (std::size_t index = 0U; index < first_view->size(); ++index) {
    EXPECT_EQ(&(*first_view)[index], &(*repeated_view)[index]);
    EXPECT_EQ((*first_view)[index].metadata.sequence,
              provisional->selected()[index].source_sequence);
  }
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, false));
  ASSERT_TRUE(provisional->finish(false));
  EXPECT_EQ(direct->retained_count(), 3U);

  auto accepted = direct->begin(TimePoint{30, ClockId{1U}},
                                script->records[2].arrival_sequence,
                                script->records[2].producer_epoch);
  ASSERT_TRUE(accepted);
  const std::array accepted_selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = accepted->selection(),
      .samples = accepted->selected(),
      .batch = true,
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 2U, accepted_selection));
  auto accepted_view = fixture.sink.input.selected_batch();
  ASSERT_TRUE(accepted_view);
  EXPECT_EQ(batch_values(*accepted_view), (std::vector<std::int32_t>{10, 20, 30}));
  for (std::size_t index = 0U; index < 3U; ++index) {
    ASSERT_TRUE((*binding)->retire(script->events[index].source_sequence));
  }
  ASSERT_TRUE(accepted->finish(true));
  EXPECT_EQ(direct->retained_count(), 0U);
  // Match the production retirement order: selector commit first proves the drained prefix is no
  // longer observable, Channel releases that temporal claim while the active batch still pins the
  // typed payloads, then activation retirement performs the final bounded reclamation.
  ASSERT_TRUE((*session)->retire_samples(
      PortId{2U}, std::span<const TemporalSampleRecord>{script->records}.first(3U)));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 2U, true));
  for (std::size_t index = 0U; index < 3U; ++index) {
    auto retired = (*session)->temporal_sample(PortId{2U}, script->events[index].source_sequence);
    ASSERT_FALSE(retired);
    EXPECT_EQ(retired.error().code, ErrorCode::kNotFound);
  }

  auto empty = direct->begin(TimePoint{31, ClockId{1U}},
                             script->records[2].arrival_sequence,
                             script->records[2].producer_epoch);
  ASSERT_TRUE(empty);
  EXPECT_TRUE(empty->selected().empty());
  const std::array empty_selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = empty->selection(),
      .samples = empty->selected(),
      .batch = true,
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 3U, empty_selection));
  auto empty_view = fixture.sink.input.selected_batch();
  ASSERT_TRUE(empty_view);
  EXPECT_TRUE(empty_view->empty());
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 3U, true));
  ASSERT_TRUE(empty->finish(true));

  for (std::size_t index = 3U; index < script->records.size(); ++index) {
    ASSERT_TRUE(direct->push(script->records[index]));
  }
  auto expanded = direct->begin(TimePoint{70, ClockId{1U}},
                                script->records.back().arrival_sequence,
                                script->records.back().producer_epoch);
  ASSERT_TRUE(expanded);
  ASSERT_EQ(expanded->selected().size(), 4U);
  const std::array expanded_selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = expanded->selection(),
      .samples = expanded->selected(),
      .batch = true,
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 4U, expanded_selection));
  auto expanded_view = fixture.sink.input.selected_batch();
  ASSERT_TRUE(expanded_view);
  EXPECT_EQ(batch_values(*expanded_view), (std::vector<std::int32_t>{40, 50, 60, 70}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 4U, false));
  ASSERT_TRUE(expanded->finish(false));
  EXPECT_EQ(direct->retained_count(), 4U);
  auto rolled_back_view = fixture.sink.input.selected_batch();
  ASSERT_FALSE(rolled_back_view);
  EXPECT_EQ(rolled_back_view.error().code, ErrorCode::kStaleEndpoint);

  for (std::size_t index = 3U; index < script->events.size(); ++index) {
    ASSERT_TRUE((*binding)->retire(script->events[index].source_sequence));
  }
  ASSERT_TRUE((*session)->retire_samples(
      PortId{2U}, std::span<const TemporalSampleRecord>{script->records}.subspan(3U)));
  (*session)->cancel();
  EXPECT_TRUE((*session)->join());
}

// REQ: RUNTIME-019
TEST(ChannelBindingParityTest, LatestWithinUsesDirectSelectorIdentityInChannelScalarView) {
  SessionFixture fixture;
  fixture.enable_latest_within();
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
  ASSERT_TRUE(binding);
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());

  constexpr std::array values{11, 22, 33};
  constexpr std::array<std::int64_t, values.size()> capture_times{10, 20, 30};
  auto script = publish_scalar_script(**session, **binding, fixture.source, values, capture_times);
  ASSERT_TRUE(script) << runtime_error_message(script.error());

  auto direct = RuntimeInputSelectionState::create(fixture.plan.input_selections.front());
  ASSERT_TRUE(direct);
  for (const auto& record : script->records) {
    ASSERT_TRUE(direct->push(record));
  }
  auto selected = direct->begin(TimePoint{25, ClockId{1U}},
                                script->records.back().arrival_sequence,
                                script->records.back().producer_epoch);
  ASSERT_TRUE(selected);
  ASSERT_EQ(selected->selected().size(), 1U);
  EXPECT_EQ(selected->selected().front(), script->records[1]);
  const std::array channel_selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = selected->selection(),
      .samples = selected->selected(),
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, channel_selection));
  auto first = fixture.sink.input.receive();
  auto repeated = fixture.sink.input.receive_data();
  ASSERT_TRUE(first);
  ASSERT_TRUE(repeated);
  EXPECT_EQ(first->data, 22);
  EXPECT_EQ(*repeated, 22);
  EXPECT_EQ(first->metadata.sequence, selected->selected().front().source_sequence);
  ASSERT_TRUE((*binding)->retire(script->events[1].source_sequence));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  ASSERT_TRUE(selected->finish(true));

  ASSERT_TRUE((*binding)->retire(script->events[0].source_sequence));
  ASSERT_TRUE((*binding)->retire(script->events[2].source_sequence));
  const std::array unselected_records{script->records[0], script->records[2]};
  ASSERT_TRUE((*session)->retire_samples(PortId{2U}, unselected_records));
  (*session)->cancel();
  EXPECT_TRUE((*session)->join());
}

// A local producer may omit capture time even when a side-input selector needs event time. The
// arrival remains valid readiness evidence, while the selector observes the canonical absence and
// applies its authored missing policy without inventing a timestamp.
// REQ: RUNTIME-017
// REQ: RUNTIME-034
// REQ: RUNTIME-043
TEST(ChannelBindingParityTest, MissingLocalCaptureTimeUsesAuthoredSafeDefault) {
  SessionFixture fixture;
  fixture.enable_scalar(InputSelectorKind::kLatestWithin, TemporalRecordMissingKind::kSafeDefault);
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
  ASSERT_TRUE(fixture.source.output.emit(Sample<std::int32_t>{37, {}}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));

  auto event = wait_for_event(**session, **binding);
  ASSERT_TRUE(event) << runtime_error_message(event.error());
  auto record = (*session)->temporal_sample(event->port, event->source_sequence);
  ASSERT_TRUE(record) << runtime_error_message(record.error());
  EXPECT_FALSE(record->capture_time.valid());

  auto selector = RuntimeInputSelectionState::create(fixture.plan.input_selections.front());
  ASSERT_TRUE(selector) << runtime_error_message(selector.error());
  ASSERT_TRUE(selector->push(*record));
  auto selected =
      selector->begin(event->reference_time, record->arrival_sequence, record->producer_epoch);
  ASSERT_TRUE(selected) << runtime_error_message(selected.error());
  EXPECT_EQ(selected->selection().state, InputState::kSafeDefault);
  EXPECT_EQ(selected->selection().reason, SelectionReason::kCaptureTimeAbsent);
  EXPECT_TRUE(selected->selected().empty());

  ASSERT_TRUE((*binding)->retire(event->source_sequence));
  ASSERT_TRUE(selected->finish(true));
  ASSERT_TRUE(selector->reset());
  const std::array records{*record};
  ASSERT_TRUE((*session)->retire_samples(PortId{2U}, records));
  (*session)->cancel();
  EXPECT_TRUE((*session)->join());
}

// An externally driven import admitted for capture-time selection may not silently downgrade a
// missing capture coordinate to arrival time. The readiness boundary must preserve the registered
// structural diagnostic identity so RunSession can attribute the protocol failure to its owner.
// REQ: RUNTIME-034
TEST(ChannelBindingParityTest, MissingExternalCaptureTimeFailsWithRegisteredProtocolIdentity) {
  SessionFixture fixture;
  fixture.enable_scalar(InputSelectorKind::kLatestWithin, TemporalRecordMissingKind::kSafeDefault);
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_external_metadata_contract(
      **session, PortId{2U}, ClockId{1U}, ClockId{1U}));
  const auto before = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
  ASSERT_TRUE(before) << runtime_error_message(before.error());
  std::optional<StructuralDiagnosticId> terminal_diagnostic;
  auto rejected = RuntimeChannelSessionTestAccess::inject_received_metadata(
      **session,
      PortId{2U},
      SampleMetadata{
          .sequence = 1U,
          .byte_size = sizeof(std::int32_t),
          .payload_kind = PayloadKind::kInlineTyped,
      },
      &terminal_diagnostic);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, ErrorCode::kProtocolError);
  EXPECT_TRUE(rejected.error().message.empty());
  EXPECT_EQ(terminal_diagnostic, StructuralDiagnosticId::kTemporalCaptureTimeRequired);
  const auto after = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
  ASSERT_TRUE(after) << runtime_error_message(after.error());
  EXPECT_EQ(after->next_arrival_sequence, before->next_arrival_sequence);
  EXPECT_EQ(after->next_global_arrival_order, before->next_global_arrival_order);
  EXPECT_EQ(after->occupied_samples, before->occupied_samples);
  EXPECT_EQ(after->active_generation, before->active_generation);
  EXPECT_EQ(after->view_generation, before->view_generation);
  EXPECT_EQ(after->producer_epoch, before->producer_epoch);
}

// REQ: RUNTIME-018
// REQ: RUNTIME-055
TEST(ChannelBindingParityTest, ScalarSelectorsAndMissingPoliciesUseOneDirectExpectationTable) {
  struct Vector {
    InputSelectorKind selector;
    TemporalRecordMissingKind missing;
    std::int64_t reference_ns;
    InputSelection expected_selection;
    std::optional<std::int32_t> expected_value;
  };
  constexpr std::array vectors{
      Vector{InputSelectorKind::kExact,
             TemporalRecordMissingKind::kSkip,
             20,
             {InputState::kSelected, SelectionReason::kNone},
             33},
      Vector{InputSelectorKind::kLatestWithin,
             TemporalRecordMissingKind::kSkip,
             25,
             {InputState::kSelected, SelectionReason::kNone},
             33},
      Vector{InputSelectorKind::kExact,
             TemporalRecordMissingKind::kSkip,
             25,
             {InputState::kAbsent, SelectionReason::kNeverSeen},
             std::nullopt},
      Vector{InputSelectorKind::kExact,
             TemporalRecordMissingKind::kSafeDefault,
             25,
             {InputState::kSafeDefault, SelectionReason::kNeverSeen},
             std::nullopt},
  };

  for (const Vector& vector : vectors) {
    SCOPED_TRACE(static_cast<unsigned>(vector.selector));
    SCOPED_TRACE(static_cast<unsigned>(vector.missing));
    SCOPED_TRACE(vector.reference_ns);
    SessionFixture fixture;
    fixture.enable_scalar(vector.selector, vector.missing);
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

    // Equal-time candidates deliberately make the stable source-sequence tie
    // part of the shared direct/Channel expectation table.
    constexpr std::array values{11, 22, 33};
    constexpr std::array<std::int64_t, values.size()> capture_times{10, 20, 20};
    auto script =
        publish_scalar_script(**session, **binding, fixture.source, values, capture_times);
    ASSERT_TRUE(script) << runtime_error_message(script.error());

    auto direct = RuntimeInputSelectionState::create(fixture.plan.input_selections.front());
    ASSERT_TRUE(direct);
    for (const TemporalSampleRecord& record : script->records) {
      ASSERT_TRUE(direct->push(record));
    }
    auto decision = direct->begin(TimePoint{vector.reference_ns, ClockId{1U}},
                                  script->records.back().arrival_sequence,
                                  script->records.back().producer_epoch);
    ASSERT_TRUE(decision);
    EXPECT_EQ(decision->selection().state, vector.expected_selection.state);
    EXPECT_EQ(decision->selection().reason, vector.expected_selection.reason);
    EXPECT_EQ(decision->selected().size(), vector.expected_value.has_value() ? 1U : 0U);

    const std::array selected{RuntimeChannelActivationSelection{
        .port = PortId{2U},
        .selection = decision->selection(),
        .samples = decision->selected(),
        .destructive = true,
    }};
    ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, selected));
    EXPECT_EQ(fixture.sink.input.selection().state, vector.expected_selection.state);
    EXPECT_EQ(fixture.sink.input.selection().reason, vector.expected_selection.reason);

    auto batch_mismatch = fixture.sink.input.selected_batch();
    ASSERT_FALSE(batch_mismatch);
    EXPECT_EQ(batch_mismatch.error().code, ErrorCode::kInputSelectionCardinalityMismatch);
    auto scalar = fixture.sink.input.receive();
    if (vector.expected_value.has_value()) {
      ASSERT_TRUE(scalar);
      EXPECT_EQ(scalar->data, *vector.expected_value);
      EXPECT_EQ(scalar->metadata.sequence, decision->selected().front().source_sequence);
    } else {
      ASSERT_FALSE(scalar);
      EXPECT_EQ(scalar.error().code, ErrorCode::kInputEmpty);
    }

    for (const RuntimeEvent& event : script->events) {
      ASSERT_TRUE((*binding)->retire(event.source_sequence));
    }
    ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, vector.expected_value.has_value()));
    ASSERT_TRUE(decision->finish(vector.expected_value.has_value()));
    (*session)->cancel();
    EXPECT_TRUE((*session)->join());
  }
}

// REQ: OPERATOR-012
// REQ: OPERATOR-020
// REQ: RUNTIME-019
TEST(ChannelBindingParityTest, InputAccessorsUseTheSyntheticGuardPrecedenceTable) {
  constexpr auto state = [](SyntheticSelectorShape shape,
                            InputSelection selection,
                            bool stale = false,
                            bool closed = false,
                            ErrorCode backend = ErrorCode::kSuccess) {
    return SyntheticInputState{
        .selector_shape = shape,
        .selection = selection,
        .stale_generation = stale,
        .closed = closed,
        .advertises_value =
            selection.state == InputState::kSelected || selection.state == InputState::kReused,
        .backend_error = backend,
    };
  };
  constexpr InputSelection not_selected{InputState::kNotSelected,
                                        SelectionReason::kNotSelectedForActivation};
  constexpr InputSelection absent{InputState::kAbsent, SelectionReason::kNeverSeen};
  constexpr InputSelection selected{InputState::kSelected, SelectionReason::kNone};

  // Production Channel accessors classify only activation-local materialized state. A transport
  // close destroys/unbinds that state (the public result is kInputUndeclared), transport receive
  // failures terminate the session before a selection is exposed, and Channel backpressure is
  // reported by publication/retirement control (covered by the existing publication-phase and
  // premature-retirement tests). Those three synthetic backend-injection rows remain in this one
  // oracle so their precedence is not lost, but are deliberately marked non-applicable to a
  // production Input accessor instead of inventing a second Channel-side error contract.
  constexpr std::array expectations{
      InputGuardExpectation{"unbound sample",
                            InputAccessorKind::kSample,
                            {},
                            ErrorCode::kInputUndeclared,
                            ChannelGuardApplicability::kAccessor,
                            true},
      InputGuardExpectation{"unbound data",
                            InputAccessorKind::kData,
                            {},
                            ErrorCode::kInputUndeclared,
                            ChannelGuardApplicability::kAccessor,
                            true},
      InputGuardExpectation{"unbound batch",
                            InputAccessorKind::kBatch,
                            {},
                            ErrorCode::kInputUndeclared,
                            ChannelGuardApplicability::kAccessor,
                            true},
      InputGuardExpectation{"stale before closed/cardinality sample",
                            InputAccessorKind::kSample,
                            state(SyntheticSelectorShape::kBatch, not_selected, true, true),
                            ErrorCode::kStaleEndpoint,
                            ChannelGuardApplicability::kAccessor},
      InputGuardExpectation{"stale before closed/cardinality data",
                            InputAccessorKind::kData,
                            state(SyntheticSelectorShape::kBatch, not_selected, true, true),
                            ErrorCode::kStaleEndpoint,
                            ChannelGuardApplicability::kAccessor},
      InputGuardExpectation{"stale before closed/cardinality batch",
                            InputAccessorKind::kBatch,
                            state(SyntheticSelectorShape::kScalar, not_selected, true, true),
                            ErrorCode::kStaleEndpoint,
                            ChannelGuardApplicability::kAccessor},
      InputGuardExpectation{"closed before cardinality sample",
                            InputAccessorKind::kSample,
                            state(SyntheticSelectorShape::kBatch, not_selected, false, true),
                            ErrorCode::kClosed,
                            ChannelGuardApplicability::kSyntheticClosedOnly},
      InputGuardExpectation{"closed before cardinality data",
                            InputAccessorKind::kData,
                            state(SyntheticSelectorShape::kBatch, not_selected, false, true),
                            ErrorCode::kClosed,
                            ChannelGuardApplicability::kSyntheticClosedOnly},
      InputGuardExpectation{"closed before cardinality batch",
                            InputAccessorKind::kBatch,
                            state(SyntheticSelectorShape::kScalar, not_selected, false, true),
                            ErrorCode::kClosed,
                            ChannelGuardApplicability::kSyntheticClosedOnly},
      InputGuardExpectation{"not selected before cardinality sample",
                            InputAccessorKind::kSample,
                            state(SyntheticSelectorShape::kBatch, not_selected),
                            ErrorCode::kInputNotSelected,
                            ChannelGuardApplicability::kAccessor},
      InputGuardExpectation{"not selected before cardinality data",
                            InputAccessorKind::kData,
                            state(SyntheticSelectorShape::kBatch, not_selected),
                            ErrorCode::kInputNotSelected,
                            ChannelGuardApplicability::kAccessor},
      InputGuardExpectation{"not selected before cardinality batch",
                            InputAccessorKind::kBatch,
                            state(SyntheticSelectorShape::kScalar, not_selected),
                            ErrorCode::kInputNotSelected,
                            ChannelGuardApplicability::kAccessor},
      InputGuardExpectation{"scalar as batch",
                            InputAccessorKind::kBatch,
                            state(SyntheticSelectorShape::kScalar, selected),
                            ErrorCode::kInputSelectionCardinalityMismatch,
                            ChannelGuardApplicability::kAccessor},
      InputGuardExpectation{"batch as sample",
                            InputAccessorKind::kSample,
                            state(SyntheticSelectorShape::kBatch, selected),
                            ErrorCode::kInputSelectionCardinalityMismatch,
                            ChannelGuardApplicability::kAccessor},
      InputGuardExpectation{"batch as data",
                            InputAccessorKind::kData,
                            state(SyntheticSelectorShape::kBatch, selected),
                            ErrorCode::kInputSelectionCardinalityMismatch,
                            ChannelGuardApplicability::kAccessor},
      InputGuardExpectation{"empty scalar sample",
                            InputAccessorKind::kSample,
                            state(SyntheticSelectorShape::kScalar, absent),
                            ErrorCode::kInputEmpty,
                            ChannelGuardApplicability::kAccessor},
      InputGuardExpectation{"empty scalar data",
                            InputAccessorKind::kData,
                            state(SyntheticSelectorShape::kScalar, absent),
                            ErrorCode::kInputEmpty,
                            ChannelGuardApplicability::kAccessor},
      InputGuardExpectation{"empty batch",
                            InputAccessorKind::kBatch,
                            state(SyntheticSelectorShape::kBatch, absent),
                            ErrorCode::kInputEmpty,
                            ChannelGuardApplicability::kAccessor},
      InputGuardExpectation{
          "backend sample error",
          InputAccessorKind::kSample,
          state(SyntheticSelectorShape::kScalar, selected, false, false, ErrorCode::kReceiveError),
          ErrorCode::kReceiveError,
          ChannelGuardApplicability::kSyntheticBackendOnly},
      InputGuardExpectation{
          "backend data error",
          InputAccessorKind::kData,
          state(SyntheticSelectorShape::kScalar, selected, false, false, ErrorCode::kReceiveError),
          ErrorCode::kReceiveError,
          ChannelGuardApplicability::kSyntheticBackendOnly},
      InputGuardExpectation{
          "backend batch error",
          InputAccessorKind::kBatch,
          state(SyntheticSelectorShape::kBatch, selected, false, false, ErrorCode::kReceiveError),
          ErrorCode::kReceiveError,
          ChannelGuardApplicability::kSyntheticBackendOnly},
      InputGuardExpectation{
          "backpressure remains a session-control result",
          InputAccessorKind::kSample,
          state(SyntheticSelectorShape::kScalar, selected, false, false, ErrorCode::kBackpressured),
          ErrorCode::kBackpressured,
          ChannelGuardApplicability::kSessionBackpressureOnly},
      InputGuardExpectation{"finite scalar read claims",
                            InputAccessorKind::kSample,
                            state(SyntheticSelectorShape::kScalar, selected),
                            ErrorCode::kResourceExhausted,
                            ChannelGuardApplicability::kAccessor,
                            false,
                            true},
  };

  for (const InputGuardExpectation& expectation : expectations) {
    SCOPED_TRACE(expectation.name);
    SinkOperator direct_sink;
    SyntheticOperatorBinding direct;
    if (!expectation.unbound) {
      ASSERT_TRUE(direct.declare_ports(direct_sink));
      expected<void, Error> bound;
      if (expectation.state.selector_shape == SyntheticSelectorShape::kScalar &&
          expectation.state.advertises_value) {
        bound = direct.bind_scalar_state(direct_sink.input,
                                         Sample<std::int32_t>{71, SampleMetadata{.sequence = 1U}},
                                         expectation.state);
      } else {
        bound = direct.bind_state(direct_sink.input, expectation.state);
      }
      ASSERT_TRUE(bound) << runtime_error_message(bound.error());
    }
    const ErrorCode direct_code =
        expectation.exhaust_claims
            ? observe_exhausted_scalar_claims(direct_sink.input)
            : observe_input_accessor(direct_sink.input, expectation.accessor);
    EXPECT_EQ(direct_code, expectation.expected);

    if (expectation.channel == ChannelGuardApplicability::kAccessor) {
      auto channel_code = observe_channel_guard(expectation);
      ASSERT_TRUE(channel_code) << runtime_error_message(channel_code.error());
      EXPECT_EQ(*channel_code, expectation.expected);
      EXPECT_EQ(*channel_code, direct_code);
    }
  }
}

// REQ: RUNTIME-019
// REQ: CHAN-115
TEST(ChannelBindingParityTest, QueuePolicyDoesNotRedefineExactOrLatestWithinSelection) {
  struct Vector {
    QueuePolicy queue_policy;
    InputSelectorKind selector;
    std::size_t survivor_count;
    std::int64_t reference_ns;
    std::int32_t expected_value;
    std::int64_t expected_capture_ns;
  };
  constexpr std::array vectors{
      Vector{QueuePolicy::kBlocking, InputSelectorKind::kExact, 3U, 20, 22, 20},
      Vector{QueuePolicy::kBlocking, InputSelectorKind::kLatestWithin, 3U, 25, 22, 20},
      Vector{QueuePolicy::kDropOldest, InputSelectorKind::kExact, 2U, 30, 33, 30},
      Vector{QueuePolicy::kDropOldest, InputSelectorKind::kLatestWithin, 2U, 35, 33, 30},
  };
  constexpr std::array values{11, 22, 33};
  constexpr std::array<std::int64_t, values.size()> capture_times{10, 20, 30};

  for (const Vector& vector : vectors) {
    SCOPED_TRACE(static_cast<unsigned>(vector.queue_policy));
    SCOPED_TRACE(static_cast<unsigned>(vector.selector));
    SessionFixture fixture;
    fixture.enable_scalar(vector.selector, TemporalRecordMissingKind::kSkip);
    if (vector.queue_policy == QueuePolicy::kDropOldest) {
      fixture.use_two_slot_drop_oldest_queue();
    }

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
    // Publish before arming the pump so the storage-policy survivor set is deterministic:
    // blocking retains all candidates, while drop-oldest retains only the two newest candidates.
    ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
    for (std::size_t index = 0U; index < values.size(); ++index) {
      ASSERT_TRUE(fixture.source.output.emit(
          Sample<std::int32_t>{values[index],
                               SampleMetadata{
                                   .capture_timestamp_ns = capture_times[index],
                                   .source_clock_id = ClockId{1U},
                               }}));
    }
    ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
    ASSERT_TRUE((*session)->arm());
    PublishedScalarScript script;
    for (std::size_t index = 0U; index < vector.survivor_count; ++index) {
      auto event = wait_for_event(**session, **binding);
      ASSERT_TRUE(event) << runtime_error_message(event.error());
      auto record = (*session)->temporal_sample(event->port, event->source_sequence);
      ASSERT_TRUE(record) << runtime_error_message(record.error());
      script.events.push_back(*event);
      script.records.push_back(*record);
    }

    auto selector = RuntimeInputSelectionState::create(fixture.plan.input_selections.front());
    ASSERT_TRUE(selector);
    for (const TemporalSampleRecord& record : script.records) {
      ASSERT_TRUE(selector->push(record));
    }
    auto decision = selector->begin(TimePoint{vector.reference_ns, ClockId{1U}},
                                    script.records.back().arrival_sequence,
                                    script.records.back().producer_epoch);
    ASSERT_TRUE(decision);
    ASSERT_EQ(decision->selected().size(), 1U);
    EXPECT_EQ(decision->selected().front().capture_time.timestamp_ns, vector.expected_capture_ns);

    SinkOperator direct_sink;
    SyntheticOperatorBinding direct;
    ASSERT_TRUE(direct.declare_ports(direct_sink));
    ASSERT_TRUE(direct.bind_scalar(
        direct_sink.input,
        Sample<std::int32_t>{vector.expected_value,
                             SampleMetadata{
                                 .sequence = decision->selected().front().source_sequence,
                                 .capture_timestamp_ns = vector.expected_capture_ns,
                                 .source_clock_id = ClockId{1U},
                             }},
        decision->selection()));
    auto direct_sample = direct_sink.input.receive();
    ASSERT_TRUE(direct_sample);

    const std::array selected_input{RuntimeChannelActivationSelection{
        .port = PortId{2U},
        .selection = decision->selection(),
        .samples = decision->selected(),
        .destructive = true,
    }};
    ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, selected_input));
    auto channel_sample = fixture.sink.input.receive();
    ASSERT_TRUE(channel_sample);
    EXPECT_EQ(channel_sample->data, vector.expected_value);
    EXPECT_EQ(channel_sample->data, direct_sample->data);
    EXPECT_EQ(channel_sample->metadata.sequence, direct_sample->metadata.sequence);
    EXPECT_EQ(channel_sample->metadata.capture_timestamp_ns,
              direct_sample->metadata.capture_timestamp_ns);
    EXPECT_EQ(channel_sample->metadata.source_clock_id, direct_sample->metadata.source_clock_id);

    ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, false));
    ASSERT_TRUE(decision->finish(false));
    for (const RuntimeEvent& event : script.events) {
      ASSERT_TRUE((*binding)->retire(event.source_sequence));
    }
    ASSERT_TRUE((*session)->retire_samples(PortId{2U}, script.records));
    (*session)->cancel();
    EXPECT_TRUE((*session)->join());
  }
}

// REQ: RUNTIME-018
// REQ: RUNTIME-019
TEST(ChannelTemporalDispatchTest, PreservesBracketedOverlapAndRejectsPartialOverflowView) {
  SessionFixture fixture;
  fixture.enable_bracketed_window(5U);
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
  ASSERT_TRUE(binding);
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());

  constexpr std::array values{5, 10, 15, 20, 25, 30, 35};
  constexpr std::array<std::int64_t, values.size()> capture_times{5, 10, 15, 20, 25, 30, 35};
  auto script = publish_scalar_script(**session, **binding, fixture.source, values, capture_times);
  ASSERT_TRUE(script) << runtime_error_message(script.error());

  auto direct = RuntimeInputSelectionState::create(
      fixture.plan.input_selections.front(), &fixture.plan.bracketed_windows.front(), 2U);
  ASSERT_TRUE(direct);
  ASSERT_TRUE(direct->push(script->records[0]));
  ASSERT_TRUE(direct->push_anchor(script->records[1]));
  ASSERT_TRUE(direct->push(script->records[1]));
  ASSERT_TRUE(direct->push(script->records[2]));
  ASSERT_TRUE(direct->push_anchor(script->records[3]));
  ASSERT_TRUE(direct->push(script->records[3]));
  ASSERT_TRUE(direct->push(script->records[4]));

  auto first_window = direct->begin(TimePoint{20, ClockId{1U}},
                                    script->records[4].arrival_sequence,
                                    script->records[4].producer_epoch);
  ASSERT_TRUE(first_window);
  EXPECT_EQ(first_window->semantic_reference_time(), (TimePoint{20, ClockId{1U}}));
  ASSERT_EQ(first_window->selected().size(), 5U);
  const std::array first_selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = first_window->selection(),
      .samples = first_window->selected(),
      .batch = true,
      .destructive = false,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, first_selection));
  auto first_view = fixture.sink.input.selected_batch();
  auto first_repeat = fixture.sink.input.selected_batch();
  ASSERT_TRUE(first_view);
  ASSERT_TRUE(first_repeat);
  EXPECT_EQ(batch_values(*first_view), (std::vector<std::int32_t>{5, 10, 15, 20, 25}));
  EXPECT_EQ(batch_values(*first_repeat), (std::vector<std::int32_t>{5, 10, 15, 20, 25}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  ASSERT_TRUE(first_window->finish(true));

  ASSERT_TRUE(direct->push_anchor(script->records[5]));
  ASSERT_TRUE(direct->push(script->records[5]));
  ASSERT_TRUE(direct->push(script->records[6]));
  auto overlap = direct->begin(TimePoint{30, ClockId{1U}},
                               script->records[6].arrival_sequence,
                               script->records[6].producer_epoch);
  ASSERT_TRUE(overlap);
  ASSERT_EQ(overlap->selected().size(), 5U);
  EXPECT_EQ(overlap->selected()[0], script->records[2]);
  EXPECT_EQ(overlap->selected()[1], script->records[3]);
  EXPECT_EQ(overlap->selected()[2], script->records[4]);
  const std::array overlap_selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = overlap->selection(),
      .samples = overlap->selected(),
      .batch = true,
      .destructive = false,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 2U, overlap_selection));
  auto overlap_view = fixture.sink.input.selected_batch();
  ASSERT_TRUE(overlap_view);
  EXPECT_EQ(batch_values(*overlap_view), (std::vector<std::int32_t>{15, 20, 25, 30, 35}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 2U, true));
  ASSERT_TRUE(overlap->finish(true));

  auto overflow_plan = fixture.plan.input_selections.front();
  overflow_plan.max_selected_samples = 3U;
  auto overflow_window = fixture.plan.bracketed_windows.front();
  overflow_window.max_samples = 3U;
  overflow_window.retained_overlap_capacity = 1U;
  auto overflow_state = RuntimeInputSelectionState::create(overflow_plan, &overflow_window, 1U);
  ASSERT_TRUE(overflow_state);
  ASSERT_TRUE(overflow_state->push(script->records[0]));
  ASSERT_TRUE(overflow_state->push_anchor(script->records[1]));
  ASSERT_TRUE(overflow_state->push(script->records[1]));
  ASSERT_TRUE(overflow_state->push_anchor(script->records[2]));
  ASSERT_TRUE(overflow_state->push(script->records[2]));
  ASSERT_TRUE(overflow_state->push(script->records[3]));
  auto overflow = overflow_state->begin(TimePoint{15, ClockId{1U}},
                                        script->records[3].arrival_sequence,
                                        script->records[3].producer_epoch);
  ASSERT_TRUE(overflow);
  EXPECT_EQ(overflow->selection().state, InputState::kAbsent);
  EXPECT_EQ(overflow->selection().reason, SelectionReason::kWindowSampleLimit);
  EXPECT_TRUE(overflow->selected().empty());
  const std::array overflow_selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = overflow->selection(),
      .samples = overflow->selected(),
      .batch = true,
      .destructive = false,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 3U, overflow_selection));
  EXPECT_EQ(fixture.sink.input.selection().reason, SelectionReason::kWindowSampleLimit);
  auto no_partial = fixture.sink.input.selected_batch();
  ASSERT_FALSE(no_partial);
  EXPECT_EQ(no_partial.error().code, ErrorCode::kInputEmpty);
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 3U, false));
  ASSERT_TRUE(overflow->finish(false));

  for (const auto& event : script->events) {
    ASSERT_TRUE((*binding)->retire(event.source_sequence));
  }
  ASSERT_TRUE((*session)->retire_samples(PortId{2U}, script->records));
  (*session)->cancel();
  EXPECT_TRUE((*session)->join());
}

// REQ: RUNTIME-018
TEST(ChannelBindingParityTest, BracketedEqualTimeOrderAndResetFactsMatchTheDirectEngine) {
  SessionFixture fixture;
  fixture.enable_bracketed_window(5U);
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
  ASSERT_TRUE(binding);
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());

  constexpr std::array values{5, 10, 11, 12, 9, 13};
  constexpr std::array<std::int64_t, values.size()> capture_times{5, 10, 10, 11, 9, 10};
  auto script = publish_scalar_script(**session, **binding, fixture.source, values, capture_times);
  ASSERT_TRUE(script) << runtime_error_message(script.error());

  auto direct = RuntimeInputSelectionState::create(
      fixture.plan.input_selections.front(), &fixture.plan.bracketed_windows.front(), 2U);
  ASSERT_TRUE(direct);
  ASSERT_TRUE(direct->push(script->records[0]));
  ASSERT_TRUE(direct->push_anchor(script->records[1]));
  ASSERT_TRUE(direct->push(script->records[1]));
  ASSERT_TRUE(direct->push(script->records[2]));
  ASSERT_TRUE(direct->push_anchor(script->records[2]));
  ASSERT_TRUE(direct->push(script->records[3]));

  auto equal_time = direct->begin(TimePoint{11, ClockId{1U}},
                                  script->records[3].arrival_sequence,
                                  script->records[3].producer_epoch);
  ASSERT_TRUE(equal_time);
  ASSERT_EQ(equal_time->selected().size(), 4U);
  EXPECT_EQ(equal_time->selected()[0], script->records[0]);
  EXPECT_EQ(equal_time->selected()[1], script->records[1]);
  EXPECT_EQ(equal_time->selected()[2], script->records[2]);
  EXPECT_EQ(equal_time->selected()[3], script->records[3]);
  const std::array selected{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = equal_time->selection(),
      .samples = equal_time->selected(),
      .batch = true,
      .destructive = false,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, selected));
  auto channel_view = fixture.sink.input.selected_batch();
  ASSERT_TRUE(channel_view);
  EXPECT_EQ(batch_values(*channel_view), (std::vector<std::int32_t>{5, 10, 11, 12}));
  for (std::size_t index = 0U; index < 4U; ++index) {
    EXPECT_EQ((*channel_view)[index].metadata.sequence,
              equal_time->selected()[index].source_sequence);
  }
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  ASSERT_TRUE(equal_time->finish(true));

  auto regressed = direct->push_anchor(script->records[4]);
  ASSERT_TRUE(regressed);
  EXPECT_EQ(regressed->disposition, WindowAnchorDisposition::kResetAndReprimed);
  EXPECT_EQ(regressed->reset_reason, WindowResetReason::kTimestampRegression);

  TemporalSampleRecord restarted = script->records[5];
  restarted.source_sequence = 1U;
  restarted.arrival_sequence = 1U;
  restarted.producer_epoch = 2U;
  auto new_epoch = direct->push(restarted);
  ASSERT_TRUE(new_epoch);
  EXPECT_EQ(direct->pending_window_count(), 0U);
  EXPECT_EQ(direct->retained_count(), 0U);

  for (const RuntimeEvent& event : script->events) {
    ASSERT_TRUE((*binding)->retire(event.source_sequence));
  }
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
  const std::array affected{VertexId{1U}, VertexId{2U}};
  ASSERT_TRUE((*session)->commit_restart(VertexId{1U}, affected));
  auto old_epoch = (*session)->prepare_activation(VertexId{2U}, 2U, selected);
  ASSERT_FALSE(old_epoch);
  EXPECT_EQ(old_epoch.error().code, ErrorCode::kStaleEndpoint);
}

// REQ: RUNTIME-016
// REQ: RUNTIME-019
// REQ: RUNTIME-019
// REQ: RUNTIME-019
TEST(RuntimeChannelBinding, EnqueuesBoundedWindowTimeoutWitnessWithoutPayloadLookup) {
  SessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto binding =
      (*session)->create_readiness_binding(VertexId{2U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.plan.plan_generation,
                                               .run_generation = 1U,
                                               .kind = RuntimeEventKind::kInputArrival,
                                               .source = make_source_id(7U),
                                               .source_clock = ClockId{1U},
                                               .reference_clock = ClockId{1U},
                                               .capacity = 2U,
                                               .source_generation = 3U,
                                               .initial_source_sequence = 10U,
                                           });
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());

  const TimePoint first_terminal{.timestamp_ns = 200, .clock_id = ClockId{1U}};
  const TimePoint second_terminal{.timestamp_ns = 201, .clock_id = ClockId{1U}};
  ASSERT_TRUE((*binding)->enqueue_window_timeout(PortId{2U}, first_terminal));
  ASSERT_TRUE((*binding)->enqueue_window_timeout(PortId{2U}, second_terminal));

  auto full = (*binding)->enqueue_window_timeout(PortId{2U}, TimePoint{202, ClockId{1U}});
  ASSERT_FALSE(full);
  EXPECT_EQ(full.error().code, ErrorCode::kBackpressured);

  std::array<RuntimeEvent, 2U> events;
  auto drained = (*binding)->drain(events);
  ASSERT_TRUE(drained);
  ASSERT_EQ(*drained, events.size());
  EXPECT_EQ(events[0].kind, RuntimeEventKind::kWindowTimeout);
  EXPECT_EQ(events[0].source, make_source_id(7U));
  EXPECT_EQ(events[0].port, PortId{2U});
  EXPECT_EQ(events[0].source_decision_time, first_terminal);
  EXPECT_EQ(events[0].reference_time, first_terminal);
  EXPECT_EQ(events[0].source_generation, 3U);
  EXPECT_EQ(events[0].source_sequence, 11U);
  EXPECT_EQ(events[1].kind, RuntimeEventKind::kWindowTimeout);
  EXPECT_EQ(events[1].source, make_source_id(7U));
  EXPECT_EQ(events[1].port, PortId{2U});
  EXPECT_EQ(events[1].source_decision_time, second_terminal);
  EXPECT_EQ(events[1].reference_time, second_terminal);
  EXPECT_EQ(events[1].source_generation, 3U);
  EXPECT_EQ(events[1].source_sequence, 12U);
  ASSERT_TRUE((*binding)->retire(events[0].source_sequence));
  ASSERT_TRUE((*binding)->retire(events[1].source_sequence));
  auto advanced = (*binding)->advance_source_generation();
  ASSERT_TRUE(advanced);
  EXPECT_EQ(*advanced, 4U);

  auto wrong_clock = (*binding)->enqueue_window_timeout(PortId{2U}, TimePoint{203, ClockId{2U}});
  ASSERT_FALSE(wrong_clock);
  EXPECT_EQ(wrong_clock.error().code, ErrorCode::kInvalidArgument);
  (*binding)->cancel();
  auto closed = (*binding)->enqueue_window_timeout(PortId{2U}, TimePoint{204, ClockId{1U}});
  ASSERT_FALSE(closed);
  EXPECT_EQ(closed.error().code, ErrorCode::kClosed);
}

TEST(RuntimeChannelBinding, RejectsWindowTimeoutOnUnsupportedGenericBinding) {
  BoundedEventPortBinding binding{1U};
  auto unsupported = binding.enqueue_window_timeout(PortId{2U}, TimePoint{200, ClockId{1U}});
  ASSERT_FALSE(unsupported);
  EXPECT_EQ(unsupported.error().code, ErrorCode::kNotSupported);
}

// REQ: RUNTIME-019
TEST(RuntimeChannelBinding, RejectsReadinessRequestWithoutSourceClock) {
  SessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());

  auto binding =
      (*session)->create_readiness_binding(VertexId{2U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.plan.plan_generation,
                                               .run_generation = 1U,
                                               .kind = RuntimeEventKind::kInputArrival,
                                               .source = make_source_id(1U),
                                               .reference_clock = ClockId{1U},
                                               .capacity = 8U,
                                           });
  ASSERT_FALSE(binding);
  EXPECT_EQ(binding.error().code, ErrorCode::kInvalidArgument);
}

// REQ: RUNTIME-018
TEST(RuntimeChannelBinding, RollsBackPartiallyResolvedActivationSelection) {
  SessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session);
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
  ASSERT_TRUE(binding);
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{31}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  auto event = wait_for_event(**session, **binding);
  ASSERT_TRUE(event);
  auto retained = (*session)->temporal_sample(event->port, event->source_sequence);
  ASSERT_TRUE(retained);
  TemporalSampleRecord missing = *retained;
  missing.source_sequence += 100U;
  const std::array invalid_samples{*retained, missing};
  const std::array invalid{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = invalid_samples,
      .batch = true,
  }};
  auto rejected = (*session)->prepare_activation(VertexId{2U}, 1U, invalid);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, ErrorCode::kStaleEndpoint);

  const std::array valid_samples{*retained};
  const std::array valid{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = valid_samples,
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, valid));
  auto predicate_view = (*session)->predicate_view(PortId{2U}, 1U);
  ASSERT_TRUE(predicate_view);
  auto value = fixture.sink.input.receive_data();
  ASSERT_TRUE(value);
  EXPECT_EQ(*value, 31);
  ASSERT_TRUE((*binding)->retire(event->source_sequence));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  (*session)->cancel();
  EXPECT_TRUE((*session)->join());
}

// REQ: RUNTIME-018
TEST(RuntimeChannelBinding, IncrementallyBindsDisjointPredicateAndComputeInputs) {
  SessionFixture fixture;
  fixture.add_compute_only_input_for_sink();
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
                                               .capacity = 16U,
                                           });
  ASSERT_TRUE(binding);
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{81}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));

  std::array<TemporalSampleRecord, 2U> temporal;
  std::array<RuntimeEvent, 2U> events;
  const PortId first_input{2U};
  for (std::size_t index = 0U; index < events.size(); ++index) {
    auto event = wait_for_event(**session, **binding);
    ASSERT_TRUE(event) << runtime_error_message(event.error());
    events[index] = *event;
    auto record = (*session)->temporal_sample(event->port, event->source_sequence);
    ASSERT_TRUE(record);
    const std::size_t temporal_index = event->port == first_input ? 0U : 1U;
    temporal[temporal_index] = *record;
  }

  const std::array predicate_phase{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = std::span<const TemporalSampleRecord>{temporal}.first<1U>(),
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, predicate_phase));
  auto first_view = (*session)->predicate_view(PortId{2U}, 1U);
  ASSERT_TRUE(first_view);
  auto omitted = fixture.second_sink.input.receive_data();
  ASSERT_FALSE(omitted);
  EXPECT_EQ(omitted.error().code, ErrorCode::kInputNotSelected);

  const std::array compute_phase{RuntimeChannelActivationSelection{
      .port = PortId{3U},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = std::span<const TemporalSampleRecord>{temporal}.subspan<1U, 1U>(),
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, compute_phase));
  EXPECT_EQ(*(*session)->predicate_view(PortId{2U}, 1U), first_view.value());
  ASSERT_TRUE((*session)->predicate_view(PortId{3U}, 1U));
  auto first_value = fixture.sink.input.receive_data();
  auto second_value = fixture.second_sink.input.receive_data();
  ASSERT_TRUE(first_value);
  ASSERT_TRUE(second_value);
  EXPECT_EQ(*first_value, 81);
  EXPECT_EQ(*second_value, 81);

  auto duplicate = (*session)->prepare_activation(VertexId{2U}, 1U, predicate_phase);
  ASSERT_FALSE(duplicate);
  EXPECT_EQ(duplicate.error().code, ErrorCode::kInvalidArgument);
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, false));
  for (const RuntimeEvent& event : events) {
    ASSERT_TRUE((*binding)->retire(event.source_sequence));
  }
  EXPECT_TRUE((*session)->temporal_sample(PortId{2U}, events[0].source_sequence) ||
              (*session)->temporal_sample(PortId{2U}, events[1].source_sequence));
  (*session)->cancel();
  EXPECT_TRUE((*session)->join());
}

// REQ: RUNTIME-016
// REQ: RUNTIME-017
TEST(RuntimeChannelBinding, ProtectsWitnessAndTemporalClaimFromLossyCacheReplacement) {
  SessionFixture fixture;
  fixture.use_one_slot_drop_oldest_queue();
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session);
  auto binding =
      (*session)->create_readiness_binding(VertexId{2U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.plan.plan_generation,
                                               .run_generation = 1U,
                                               .kind = RuntimeEventKind::kInputArrival,
                                               .source = make_source_id(1U),
                                               .source_clock = ClockId{1U},
                                               .reference_clock = ClockId{1U},
                                               .capacity = 2U,
                                           });
  ASSERT_TRUE(binding);
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{1}));
  auto first_event = wait_for_event(**session, **binding);
  ASSERT_TRUE(first_event);
  auto first_temporal =
      (*session)->temporal_sample(first_event->port, first_event->source_sequence);
  ASSERT_TRUE(first_temporal);
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{2}));
  std::this_thread::sleep_for(20ms);
  EXPECT_TRUE((*session)->temporal_sample(first_event->port, first_event->source_sequence));

  ASSERT_TRUE((*binding)->retire(first_event->source_sequence));
  std::this_thread::sleep_for(20ms);
  EXPECT_TRUE((*session)->temporal_sample(first_event->port, first_event->source_sequence));

  // Witness retirement releases transport attribution only. The lossy slot stays immutable until
  // the selector proves that no cache, pending/closed window, reuse state, or activation can still
  // observe the payload.
  const std::array first_claim{*first_temporal};
  ASSERT_TRUE((*session)->retire_samples(first_event->port, first_claim));
  auto second_event = wait_for_event(**session, **binding);
  ASSERT_TRUE(second_event) << runtime_error_message(second_event.error());
  EXPECT_GT(second_event->source_sequence, first_event->source_sequence);
  auto replaced = (*session)->temporal_sample(first_event->port, first_event->source_sequence);
  ASSERT_FALSE(replaced);
  EXPECT_EQ(replaced.error().code, ErrorCode::kNotFound);
  auto second_temporal =
      (*session)->temporal_sample(second_event->port, second_event->source_sequence);
  ASSERT_TRUE(second_temporal);
  ASSERT_TRUE((*binding)->retire(second_event->source_sequence));
  const std::array second_claim{*second_temporal};
  ASSERT_TRUE((*session)->retire_samples(second_event->port, second_claim));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  (*session)->cancel();
  EXPECT_TRUE((*session)->join());
}

// A transported drop-oldest receive reserves the typed destination it will overwrite. Count that
// actual claim as a pre-witness replacement even though the receive commits through its reserved
// destination and therefore does not query writable_slot() a second time.
// REQ: RUNTIME-017
// REQ: CHAN-131
TEST(RuntimeChannelBinding, TransportedReservationCountsItsPreWitnessReplacement) {
  SessionFixture fixture;
  fixture.use_one_slot_drop_oldest_queue();
  const auto counters = std::make_shared<RuntimePhaseCounters>();
  fixture.plan.executable_factories.phase_counters = counters;
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
                                               .capacity = 2U,
                                           });
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));

  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{11}));
  ASSERT_TRUE(wait_for_occupied_sample(**session, PortId{2U}));
  const auto before = counters->snapshot();
  EXPECT_EQ(
      before[RuntimePhaseCounters::index(RuntimePhaseCounter::kChannelPreWitnessReplacements)], 0U);

  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{22}));
  const auto received_deadline = std::chrono::steady_clock::now() + 2s;
  RuntimePhaseCounters::Snapshot received;
  do {
    received = counters->snapshot();
    if (received[RuntimePhaseCounters::index(RuntimePhaseCounter::kChannelSamplesReceived)] >= 2U) {
      break;
    }
    std::this_thread::yield();
  } while (std::chrono::steady_clock::now() < received_deadline);
  ASSERT_GE(received[RuntimePhaseCounters::index(RuntimePhaseCounter::kChannelSamplesReceived)],
            2U);
  const auto replacement_state =
      RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
  ASSERT_TRUE(replacement_state) << runtime_error_message(replacement_state.error());
  EXPECT_EQ(replacement_state->occupied_samples, 1U);
  EXPECT_EQ(replacement_state->pending_direct_ingress_samples, 0U);
  auto replacement = wait_for_event(**session, **binding);
  ASSERT_TRUE(replacement) << runtime_error_message(replacement.error());
  auto replacement_record =
      (*session)->temporal_sample(replacement->port, replacement->source_sequence);
  ASSERT_TRUE(replacement_record) << runtime_error_message(replacement_record.error());
  EXPECT_EQ(replacement_record->source_sequence, 2U);
  const auto after = counters->snapshot();
  EXPECT_EQ(after[RuntimePhaseCounters::index(RuntimePhaseCounter::kChannelPreWitnessReplacements)],
            1U);

  ASSERT_TRUE((*binding)->retire(replacement->source_sequence));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// Selector commit can release temporal history before compute returns. The activation and witness
// claims independently keep the typed payload stable until both later retirement boundaries pass.
// REQ: RUNTIME-017
// REQ: RUNTIME-018
// REQ: RUNTIME-019
TEST(RuntimeChannelBinding, TemporalRetirementDefersReclamationAcrossActiveSelection) {
  SessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session);
  auto binding =
      (*session)->create_readiness_binding(VertexId{2U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.plan.plan_generation,
                                               .run_generation = 1U,
                                               .kind = RuntimeEventKind::kInputArrival,
                                               .source = make_source_id(1U),
                                               .source_clock = ClockId{1U},
                                               .reference_clock = ClockId{1U},
                                               .capacity = 2U,
                                           });
  ASSERT_TRUE(binding);
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{9}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));

  auto event = wait_for_event(**session, **binding);
  ASSERT_TRUE(event);
  auto temporal = (*session)->temporal_sample(event->port, event->source_sequence);
  ASSERT_TRUE(temporal);
  const std::array records{*temporal};
  const std::array selected{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = records,
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, selected));
  ASSERT_TRUE((*session)->retire_samples(PortId{2U}, records));
  EXPECT_TRUE((*session)->temporal_sample(event->port, event->source_sequence));
  auto value = fixture.sink.input.receive_data();
  ASSERT_TRUE(value);
  EXPECT_EQ(*value, 9);

  ASSERT_TRUE((*binding)->retire(event->source_sequence));
  EXPECT_TRUE((*session)->temporal_sample(event->port, event->source_sequence));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  auto reclaimed = (*session)->temporal_sample(event->port, event->source_sequence);
  ASSERT_FALSE(reclaimed);
  EXPECT_EQ(reclaimed.error().code, ErrorCode::kNotFound);
  (*session)->cancel();
  EXPECT_TRUE((*session)->join());
}

// A selector synchronizes every removed temporal claim for one input as one Channel retirement
// batch. Validation of a later record must therefore fail before an earlier valid record loses its
// claim; the unchanged batch remains retryable after the caller corrects the bad record.
// REQ: RUNTIME-018
// REQ: RUNTIME-018
TEST(RuntimeChannelBinding, TemporalRetirementBatchFailureDoesNotReleaseEarlierClaims) {
  SessionFixture fixture;
  fixture.use_two_slot_blocking_queue();
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
                                               .capacity = 2U,
                                           });
  ASSERT_TRUE(binding);
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());

  constexpr std::array values{31, 32};
  constexpr std::array<std::int64_t, values.size()> capture_times{10, 20};
  auto script = publish_scalar_script(**session, **binding, fixture.source, values, capture_times);
  ASSERT_TRUE(script) << runtime_error_message(script.error());
  ASSERT_EQ(script->records.size(), 2U);
  for (const RuntimeEvent& event : script->events) {
    ASSERT_TRUE((*binding)->retire(event.source_sequence));
  }

  TemporalSampleRecord malformed = script->records[1];
  ASSERT_NE(malformed.source_sequence, std::numeric_limits<std::uint64_t>::max());
  ++malformed.source_sequence;
  const std::array failed_batch{script->records[0], malformed};
  auto failed = (*session)->retire_samples(PortId{2U}, failed_batch);
  ASSERT_FALSE(failed);
  EXPECT_EQ(failed.error().code, ErrorCode::kStaleEndpoint);

  // The valid first record was inspected before the malformed second record, but the failed batch
  // must leave both temporal claims intact.
  EXPECT_TRUE((*session)->temporal_sample(PortId{2U}, script->records[0].source_sequence));
  EXPECT_TRUE((*session)->temporal_sample(PortId{2U}, script->records[1].source_sequence));
  ASSERT_TRUE((*session)->retire_samples(PortId{2U}, script->records));
  for (const TemporalSampleRecord& record : script->records) {
    auto retired = (*session)->temporal_sample(PortId{2U}, record.source_sequence);
    ASSERT_FALSE(retired);
    EXPECT_EQ(retired.error().code, ErrorCode::kNotFound);
  }

  (*session)->cancel();
  EXPECT_TRUE((*session)->join());
}

// REQ: RUNTIME-017
TEST(RuntimeChannelBinding, FreezesCaptureAndConvertedReferenceCoordinates) {
  SessionFixture fixture;
  fixture.require_direct_capture_conversion();
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
                                               .reference_clock = ClockId{2U},
                                               .source_to_reference_conversion_id = 1U,
                                               .capacity = 8U,
                                           });
  ASSERT_TRUE(binding);
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  const SampleMetadata metadata{.capture_timestamp_ns = 10, .source_clock_id = ClockId{1U}};
  ASSERT_TRUE(fixture.source.output.emit(Sample<std::int32_t>{9, metadata}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  auto event = wait_for_event(**session, **binding, ClockId{2U});
  ASSERT_TRUE(event) << runtime_error_message(event.error());
  EXPECT_EQ(event->source_decision_time, (TimePoint{.timestamp_ns = 10, .clock_id = ClockId{1U}}));
  EXPECT_EQ(event->reference_time, (TimePoint{.timestamp_ns = 23, .clock_id = ClockId{2U}}));
  (*session)->cancel();
  EXPECT_TRUE((*session)->join());
}

// Arrival identities are bounded runtime protocol counters. Neither the subscriber-local sequence
// nor the session-global order may wrap and make two retained samples observationally identical.
// Exhaustion is terminal and must occur before typed retention or either counter is advanced.
// REQ: RUNTIME-018
// REQ: RUNTIME-020
// REQ: RUNTIME-024
TEST(RuntimeChannelBinding, ArrivalCounterExhaustionFailsBeforeRetainingTheSample) {
  constexpr std::uint64_t exhausted = std::numeric_limits<std::uint64_t>::max();
  for (const bool exhaust_global_order : {false, true}) {
    SCOPED_TRACE(exhaust_global_order ? "global arrival order" : "subscriber arrival sequence");
    SessionFixture fixture;
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
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_arrival_counters(
        **session,
        PortId{2U},
        exhaust_global_order ? 1U : exhausted,
        exhaust_global_order ? exhausted : 1U));
    auto before = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
    ASSERT_TRUE(before) << runtime_error_message(before.error());

    ASSERT_TRUE((*session)->refresh_connections());
    ASSERT_TRUE((*session)->arm());
    ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
    ASSERT_TRUE(fixture.source.output.emit(std::int32_t{97}));
    ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
    auto event = wait_for_event(**session, **binding);
    ASSERT_FALSE(event);
    EXPECT_EQ(event.error().code, ErrorCode::kResourceExhausted);
    (*session)->cancel();
    auto joined = (*session)->join();
    ASSERT_FALSE(joined);
    EXPECT_EQ(joined.error().code, ErrorCode::kResourceExhausted);

    auto after = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
    ASSERT_TRUE(after) << runtime_error_message(after.error());
    EXPECT_EQ(after->next_arrival_sequence, before->next_arrival_sequence);
    EXPECT_EQ(after->next_global_arrival_order, before->next_global_arrival_order);
    EXPECT_EQ(after->occupied_samples, 0U);
  }
}

// Direct lossy ingress performs the same arrival-identity preflight before accepting a sample.
// Exhaustion must not leave an unpublished mailbox survivor that a retry could later duplicate.
// REQ: CHAN-116
// REQ: RUNTIME-018
// REQ: RUNTIME-020
// REQ: RUNTIME-024
TEST(RuntimeChannelBinding, DirectLossyArrivalCounterExhaustionLeavesIngressEmpty) {
  constexpr std::uint64_t exhausted = std::numeric_limits<std::uint64_t>::max();
  for (const bool exhaust_global_order : {false, true}) {
    SCOPED_TRACE(exhaust_global_order ? "global arrival order" : "subscriber arrival sequence");
    SessionFixture fixture;
    fixture.use_one_slot_drop_oldest_queue();
    fixture.use_direct_retained_delivery();
    auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
    ASSERT_TRUE(session) << runtime_error_message(session.error());
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_arrival_counters(
        **session,
        PortId{2U},
        exhaust_global_order ? 1U : exhausted,
        exhaust_global_order ? exhausted : 1U));
    auto before = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
    ASSERT_TRUE(before) << runtime_error_message(before.error());

    ASSERT_TRUE((*session)->refresh_connections());
    ASSERT_TRUE((*session)->arm());
    ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
    auto emitted = fixture.source.output.emit(std::int32_t{97});
    ASSERT_FALSE(emitted);
    EXPECT_EQ(emitted.error().code, ErrorCode::kResourceExhausted);

    auto after = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
    ASSERT_TRUE(after) << runtime_error_message(after.error());
    EXPECT_EQ(after->next_arrival_sequence, before->next_arrival_sequence);
    EXPECT_EQ(after->next_global_arrival_order, before->next_global_arrival_order);
    EXPECT_EQ(after->occupied_samples, 0U);
    EXPECT_EQ(after->pending_direct_ingress_samples, 0U);

    ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
    (*session)->cancel();
    EXPECT_TRUE((*session)->join());
  }
}

// A contended earlier subscriber hides a potentially smaller arrival identity. Poll therefore
// makes no progress instead of selecting an observable later arrival from another input.
// REQ: RUNTIME-016
// REQ: RUNTIME-018
// REQ: RUNTIME-026
TEST(RuntimeChannelBinding, ContendedInputCannotBeOvertakenByLaterArrival) {
  SessionFixture fixture;
  fixture.add_compute_only_input_for_sink();
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
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{31}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  ASSERT_TRUE(wait_for_occupied_sample(**session, PortId{2U}));
  ASSERT_TRUE(wait_for_occupied_sample(**session, PortId{3U}));

  // Key setup: hold the input lock on PortId{2U} to block the pump from publishing that event.
  HeldInputLock held;
  bool lock_operation_succeeded = false;
  std::thread lock_owner([&] {
    lock_operation_succeeded = static_cast<bool>(RuntimeChannelSessionTestAccess::with_input_lock(
        **session, PortId{2U}, &held, &hold_input_lock));
  });
  const auto lock_deadline = std::chrono::steady_clock::now() + 2s;
  while (!held.acquired.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < lock_deadline) {
    std::this_thread::yield();
  }
  std::array<RuntimeEvent, 2U> events{};
  expected<void, Error> blocked_poll = make_unexpected(Error{ErrorCode::kNotReady});
  expected<std::size_t, Error> blocked_drain = make_unexpected(Error{ErrorCode::kNotReady});
  if (held.acquired.load(std::memory_order_acquire)) {
    EXPECT_TRUE((*binding)->poll_target().has_value());
    blocked_poll = (*binding)->poll(TimePoint{1, ClockId{1U}});
    blocked_drain = (*binding)->drain(events);
  }

  held.release.store(true, std::memory_order_release);
  lock_owner.join();
  ASSERT_TRUE(held.acquired.load(std::memory_order_acquire));
  ASSERT_TRUE(lock_operation_succeeded);
  ASSERT_TRUE(blocked_poll) << runtime_error_message(blocked_poll.error());
  ASSERT_TRUE(blocked_drain) << runtime_error_message(blocked_drain.error());
  // Key assertion: drain returns 0 while the lock is held — later arrival cannot overtake.
  EXPECT_EQ(*blocked_drain, 0U);
  EXPECT_TRUE((*binding)->poll_target().has_value());
  ASSERT_TRUE((*binding)->poll(TimePoint{2, ClockId{1U}}));
  EXPECT_FALSE((*binding)->poll_target().has_value());
  auto ordered_drain = (*binding)->drain(events);
  ASSERT_TRUE(ordered_drain);
  ASSERT_EQ(*ordered_drain, 2U);
  EXPECT_EQ(events[0].port, PortId{2U});
  EXPECT_EQ(events[1].port, PortId{3U});
  ASSERT_TRUE((*binding)->retire(events[0].source_sequence));
  ASSERT_TRUE((*binding)->retire(events[1].source_sequence));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// The contention retry is a bounded spin against the real input cache mutex. Against a holder that
// outlives the budget it must run for at least that budget and then report the same incomplete
// decline as the lost try_lock it follows, keeping the dirty revision for the next round; the next
// ordinary poll after the release publishes the retained sample. A missing retry loop or a zero
// budget returns immediately and fails the lower bound; an unbounded loop outlives the holder,
// converts the revision, and fails the incomplete-decline assertions.
// REQ: RUNTIME-025
// REQ: RUNTIME-026
TEST(RuntimeChannelBinding, ContentionRetryExpiresOnHeldInputLockAndRetainsDirtyRevision) {
  SessionFixture fixture;
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
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{31}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  ASSERT_TRUE(wait_for_occupied_sample(**session, PortId{2U}));

  // Key setup: hold the real input cache lock on PortId{2U} across the ordinary poll and the whole
  // contention retry. The holder gives up after two seconds so an unbounded retry cannot hang.
  BoundedHeldInputLock held;
  bool lock_operation_succeeded = false;
  std::thread lock_owner([&] {
    lock_operation_succeeded = static_cast<bool>(RuntimeChannelSessionTestAccess::with_input_lock(
        **session, PortId{2U}, &held, &hold_input_lock_bounded));
  });
  const auto lock_deadline = std::chrono::steady_clock::now() + 2s;
  while (!held.acquired.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < lock_deadline) {
    std::this_thread::yield();
  }
  expected<void, Error> contended_poll = make_unexpected(Error{ErrorCode::kNotReady});
  expected<void, Error> retried_poll = make_unexpected(Error{ErrorCode::kNotReady});
  expected<std::size_t, Error> blocked_drain = make_unexpected(Error{ErrorCode::kNotReady});
  bool contended_poll_incomplete = false;
  bool contended_poll_lock_contended = false;
  bool retried_poll_incomplete = false;
  bool retried_poll_lock_contended = false;
  bool retried_poll_target_retained = false;
  std::chrono::steady_clock::duration retry_elapsed{};
  std::array<RuntimeEvent, 2U> events{};
  if (held.acquired.load(std::memory_order_acquire)) {
    contended_poll = (*binding)->poll(TimePoint{1, ClockId{1U}});
    contended_poll_incomplete = (*binding)->last_poll_incomplete();
    contended_poll_lock_contended = (*binding)->last_poll_lock_contended();
    const auto retry_started = std::chrono::steady_clock::now();
    retried_poll = (*binding)->poll_after_contention(TimePoint{2, ClockId{1U}});
    retry_elapsed = std::chrono::steady_clock::now() - retry_started;
    retried_poll_incomplete = (*binding)->last_poll_incomplete();
    retried_poll_lock_contended = (*binding)->last_poll_lock_contended();
    retried_poll_target_retained = (*binding)->poll_target().has_value();
    blocked_drain = (*binding)->drain(events);
  }
  held.release.store(true, std::memory_order_release);
  lock_owner.join();
  ASSERT_TRUE(held.acquired.load(std::memory_order_acquire));
  ASSERT_TRUE(lock_operation_succeeded);
  ASSERT_FALSE(held.gave_up.load(std::memory_order_acquire))
      << "the holder outlived the retry: the contention retry is not bounded";
  ASSERT_TRUE(contended_poll) << runtime_error_message(contended_poll.error());
  EXPECT_TRUE(contended_poll_incomplete);
  EXPECT_TRUE(contended_poll_lock_contended);
  // Key assertion: the retry waited out at least its budget against the held lock, then declined
  // exactly like the ordinary poll and kept the dirty revision: the poll target is still
  // advertised and nothing drains.
  ASSERT_TRUE(retried_poll) << runtime_error_message(retried_poll.error());
  EXPECT_GE(retry_elapsed, kContentionRetryBudget)
      << "retry returned after "
      << std::chrono::duration_cast<std::chrono::nanoseconds>(retry_elapsed).count() << " ns";
  EXPECT_TRUE(retried_poll_incomplete);
  EXPECT_TRUE(retried_poll_lock_contended);
  EXPECT_TRUE(retried_poll_target_retained);
  ASSERT_TRUE(blocked_drain) << runtime_error_message(blocked_drain.error());
  EXPECT_EQ(*blocked_drain, 0U);

  // Key assertion: once the holder is gone, the next ordinary poll publishes the retained sample.
  ASSERT_TRUE((*binding)->poll(TimePoint{3, ClockId{1U}}));
  EXPECT_FALSE((*binding)->last_poll_incomplete());
  EXPECT_FALSE((*binding)->poll_target().has_value());
  auto drained = (*binding)->drain(events);
  ASSERT_TRUE(drained) << runtime_error_message(drained.error());
  ASSERT_EQ(*drained, 1U);
  EXPECT_EQ(events[0].port, PortId{2U});
  ASSERT_TRUE((*binding)->retire(events[0].source_sequence));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// The contention retry exists for one production case: the holder releases the input cache mutex
// inside the budget and the same poll_after_contention() call acquires it and converts the
// revision. Force that ordering exactly: the real holder releases only when the retry reports a
// failed try_lock from inside its loop, and the budget is widened so a scheduler pause cannot
// expire the retry under test. A loop that reports attempts but never re-tries the lock releases
// the holder yet still returns unowned at its deadline, so the retry is incomplete, the poll target
// stays advertised and nothing drains; a loop that never reports leaves the holder in place and
// expires the same way; a loop that acquires but spins on to its deadline fails the elapsed bound.
// REQ: RUNTIME-025
// REQ: RUNTIME-026
TEST(RuntimeChannelBinding, ContentionRetryAcquiresLockReleasedDuringItsBudgetAndPublishes) {
  SessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  BoundedHeldInputLock held;
  ReleaseOnRetryAttempt release{.held = &held};
  // Wide enough that a scheduler pause of the holder cannot expire the retry under test, and
  // strictly shorter than the holder's two-second give-up so an expired retry is still observed
  // against a held lock.
  constexpr std::chrono::nanoseconds kWidenedRetryBudget = 1s;
  ASSERT_TRUE(
      RuntimeChannelSessionTestAccess::set_contention_retry_budget(**session, kWidenedRetryBudget));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_input_contention_retry_attempt_observer(
      **session, PortId{2U}, &release, &release_holder_on_retry_attempt));
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
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{31}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  ASSERT_TRUE(wait_for_occupied_sample(**session, PortId{2U}));

  // Key setup: hold the real input cache lock on PortId{2U}. The holder releases only when the
  // retry loop has reported a failed attempt, never on a timer; it gives up after two seconds so
  // a retry that never reports fails its assertions instead of hanging.
  bool lock_operation_succeeded = false;
  std::thread lock_owner([&] {
    lock_operation_succeeded = static_cast<bool>(RuntimeChannelSessionTestAccess::with_input_lock(
        **session, PortId{2U}, &held, &hold_input_lock_bounded));
  });
  const auto lock_deadline = std::chrono::steady_clock::now() + 2s;
  while (!held.acquired.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < lock_deadline) {
    std::this_thread::yield();
  }
  expected<void, Error> contended_poll = make_unexpected(Error{ErrorCode::kNotReady});
  expected<void, Error> retried_poll = make_unexpected(Error{ErrorCode::kNotReady});
  bool contended_poll_incomplete = false;
  bool contended_poll_lock_contended = false;
  bool retried_poll_incomplete = true;
  bool retried_poll_lock_contended = true;
  bool retried_poll_target_retained = true;
  std::chrono::steady_clock::duration retry_elapsed = kWidenedRetryBudget;
  if (held.acquired.load(std::memory_order_acquire)) {
    contended_poll = (*binding)->poll(TimePoint{1, ClockId{1U}});
    contended_poll_incomplete = (*binding)->last_poll_incomplete();
    contended_poll_lock_contended = (*binding)->last_poll_lock_contended();
    const auto retry_started = std::chrono::steady_clock::now();
    retried_poll = (*binding)->poll_after_contention(TimePoint{2, ClockId{1U}});
    retry_elapsed = std::chrono::steady_clock::now() - retry_started;
    retried_poll_incomplete = (*binding)->last_poll_incomplete();
    retried_poll_lock_contended = (*binding)->last_poll_lock_contended();
    retried_poll_target_retained = (*binding)->poll_target().has_value();
  }
  held.release.store(true, std::memory_order_release);
  lock_owner.join();
  ASSERT_TRUE(held.acquired.load(std::memory_order_acquire));
  ASSERT_TRUE(lock_operation_succeeded);
  ASSERT_FALSE(held.gave_up.load(std::memory_order_acquire))
      << "the holder gave up before the retry reported a failed attempt";
  ASSERT_TRUE(contended_poll) << runtime_error_message(contended_poll.error());
  EXPECT_TRUE(contended_poll_incomplete);
  EXPECT_TRUE(contended_poll_lock_contended);
  // Key assertion: the retry re-armed against the held lock at least once, the holder released in
  // response, and the same retry call acquired the lock and converted the revision before its
  // budget expired: complete, no contention reported, poll target cleared, and the retained sample
  // drains exactly once.
  ASSERT_TRUE(retried_poll) << runtime_error_message(retried_poll.error());
  EXPECT_GE(release.attempts.load(std::memory_order_acquire), 1U);
  EXPECT_LT(retry_elapsed, kWidenedRetryBudget)
      << "retry returned after "
      << std::chrono::duration_cast<std::chrono::nanoseconds>(retry_elapsed).count() << " ns";
  EXPECT_FALSE(retried_poll_incomplete);
  EXPECT_FALSE(retried_poll_lock_contended);
  EXPECT_FALSE(retried_poll_target_retained);
  std::array<RuntimeEvent, 2U> events{};
  auto drained = (*binding)->drain(events);
  ASSERT_TRUE(drained) << runtime_error_message(drained.error());
  ASSERT_EQ(*drained, 1U);
  EXPECT_EQ(events[0].port, PortId{2U});
  ASSERT_TRUE((*binding)->retire(events[0].source_sequence));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// On the HoloMQ path, a kDropOldest replacement can reuse the selected slot after the poll's
// owner-local snapshot releases its mutex. The poll must validate the saved arrival identity after
// reacquisition and reheap the replacement rather than publish it ahead of an older candidate in
// another slot.
// REQ: RUNTIME-018
// REQ: RUNTIME-043
TEST(RuntimeChannelBinding, DropOldestReplacementCannotOvertakeOlderSavedArrival) {
  SessionFixture fixture;
  fixture.use_two_slot_drop_oldest_queue();
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
                                               .capacity = 4U,
                                           });
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());

  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{11}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));

  // Let the pump remove the first kDropOldest transport sample before publishing the second. This
  // deterministically fills both Runtime cache slots without depending on transport scheduling.
  ASSERT_TRUE(wait_for_occupied_sample(**session, PortId{2U}));
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 2U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{22}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 2U, true));

  const auto occupied_deadline = std::chrono::steady_clock::now() + 2s;
  expected<RuntimeChannelInputResetState, Error> occupied =
      make_unexpected(Error{ErrorCode::kNotReady});
  do {
    occupied = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
    ASSERT_TRUE(occupied) << runtime_error_message(occupied.error());
    if (occupied->occupied_samples == 2U) {
      break;
    }
    std::this_thread::yield();
  } while (std::chrono::steady_clock::now() < occupied_deadline);
  ASSERT_EQ(occupied->occupied_samples, 2U);

  LatestReplacementBarrier barrier;
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_poll_selection_observer(
      **session, &barrier, &wait_for_latest_replacement));
  std::atomic<ErrorCode> replacement_result{ErrorCode::kFailure};
  std::thread replacement([&] {
    const auto selection_deadline = std::chrono::steady_clock::now() + 2s;
    while (!barrier.selection_saved.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < selection_deadline) {
      std::this_thread::yield();
    }
    if (!barrier.selection_saved.load(std::memory_order_acquire)) {
      replacement_result.store(ErrorCode::kNotReady, std::memory_order_release);
      barrier.replacement_complete.store(true, std::memory_order_release);
      return;
    }
    auto prepared = (*session)->prepare_activation(VertexId{1U}, 3U, {});
    if (!prepared) {
      replacement_result.store(prepared.error().code, std::memory_order_release);
    } else {
      auto emitted = fixture.source.output.emit(std::int32_t{33});
      if (!emitted) {
        replacement_result.store(emitted.error().code, std::memory_order_release);
      } else {
        auto retired = (*session)->retire_activation(VertexId{1U}, 3U, true);
        if (!retired) {
          replacement_result.store(retired.error().code, std::memory_order_release);
        } else {
          auto flushed = (*session)->flush_process_local_activation_cut();
          replacement_result.store(flushed ? ErrorCode::kSuccess : flushed.error().code,
                                   std::memory_order_release);
        }
      }
    }
    barrier.replacement_complete.store(true, std::memory_order_release);
  });

  auto polled = (*binding)->poll(TimePoint{1, ClockId{1U}});
  replacement.join();
  ASSERT_TRUE(polled) << runtime_error_message(polled.error());
  ASSERT_FALSE(barrier.observer_timed_out.load(std::memory_order_acquire));
  ASSERT_EQ(replacement_result.load(std::memory_order_acquire), ErrorCode::kSuccess);

  std::array<RuntimeEvent, 2U> events{};
  auto drained = (*binding)->drain(events);
  ASSERT_TRUE(drained) << runtime_error_message(drained.error());
  ASSERT_EQ(*drained, 2U);
  auto first = (*session)->temporal_sample(events[0].port, events[0].source_sequence);
  auto second = (*session)->temporal_sample(events[1].port, events[1].source_sequence);
  ASSERT_TRUE(first) << runtime_error_message(first.error());
  ASSERT_TRUE(second) << runtime_error_message(second.error());
  EXPECT_EQ(first->source_sequence, 2U);
  EXPECT_EQ(second->source_sequence, 3U);
  ASSERT_TRUE((*binding)->retire(events[0].source_sequence));
  ASSERT_TRUE((*binding)->retire(events[1].source_sequence));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// The owner-local candidate snapshot is bounded by typed storage rather than the event ring. A
// smaller ring may publish an ordered prefix and backpressure, but its retained suffix must remain
// available for the next poll without rescanning already emitted witnesses.
// REQ: RUNTIME-016
// REQ: RUNTIME-018
TEST(RuntimeChannelBinding, ReadinessRingBackpressureRetainsOrderedCandidateSuffix) {
  SessionFixture fixture;
  fixture.use_two_slot_blocking_queue();
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
                                               .capacity = 1U,
                                           });
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  for (std::uint64_t generation = 1U; generation <= 2U; ++generation) {
    ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, generation, {}));
    ASSERT_TRUE(fixture.source.output.emit(static_cast<std::int32_t>(generation * 11U)));
    ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, generation, true));
  }

  const auto occupied_deadline = std::chrono::steady_clock::now() + 2s;
  expected<RuntimeChannelInputResetState, Error> occupied =
      make_unexpected(Error{ErrorCode::kNotReady});
  do {
    occupied = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
    ASSERT_TRUE(occupied) << runtime_error_message(occupied.error());
    if (occupied->occupied_samples == 2U) {
      break;
    }
    std::this_thread::yield();
  } while (std::chrono::steady_clock::now() < occupied_deadline);
  ASSERT_EQ(occupied->occupied_samples, 2U);

  auto first_poll = (*binding)->poll(TimePoint{1, ClockId{1U}});
  ASSERT_FALSE(first_poll);
  EXPECT_EQ(first_poll.error().code, ErrorCode::kBackpressured);
  RuntimeEvent first_event;
  auto first_count = (*binding)->drain(std::span<RuntimeEvent>{&first_event, 1U});
  ASSERT_TRUE(first_count);
  ASSERT_EQ(*first_count, 1U);
  auto first = (*session)->temporal_sample(first_event.port, first_event.source_sequence);
  ASSERT_TRUE(first);
  EXPECT_EQ(first->source_sequence, 1U);
  ASSERT_TRUE((*binding)->retire(first_event.source_sequence));

  ASSERT_TRUE((*binding)->poll(TimePoint{2, ClockId{1U}}));
  RuntimeEvent second_event;
  auto second_count = (*binding)->drain(std::span<RuntimeEvent>{&second_event, 1U});
  ASSERT_TRUE(second_count);
  ASSERT_EQ(*second_count, 1U);
  auto second = (*session)->temporal_sample(second_event.port, second_event.source_sequence);
  ASSERT_TRUE(second);
  EXPECT_EQ(second->source_sequence, 2U);
  ASSERT_TRUE((*binding)->retire(second_event.source_sequence));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// Readiness-ring pressure is retryable, while exhausting its monotonic source identity is not.
// Sequence exhaustion leaves the retained sample unemitted so a non-exhausted binding can still
// publish its one witness exactly once.
// REQ: RUNTIME-016
// REQ: RUNTIME-018
TEST(RuntimeChannelBinding, ReadinessSequenceExhaustionIsResourceExhaustedBeforeWitnessEmission) {
  SessionFixture fixture;
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
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{37}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  ASSERT_TRUE(wait_for_occupied_sample(**session, PortId{2U}));

  constexpr std::uint64_t exhausted = std::numeric_limits<std::uint64_t>::max();
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_readiness_sequence(**binding, exhausted));
  auto exhausted_poll = (*binding)->poll(TimePoint{1, ClockId{1U}});
  ASSERT_FALSE(exhausted_poll);
  EXPECT_EQ(exhausted_poll.error().code, ErrorCode::kResourceExhausted);
  RuntimeEvent event;
  auto empty = (*binding)->drain(std::span<RuntimeEvent>{&event, 1U});
  ASSERT_TRUE(empty);
  EXPECT_EQ(*empty, 0U);

  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_readiness_sequence(**binding, exhausted - 1U));
  ASSERT_TRUE((*binding)->poll(TimePoint{2, ClockId{1U}}));
  auto drained = (*binding)->drain(std::span<RuntimeEvent>{&event, 1U});
  ASSERT_TRUE(drained);
  ASSERT_EQ(*drained, 1U);
  EXPECT_EQ(event.source_sequence, exhausted);
  ASSERT_TRUE((*binding)->retire(event.source_sequence));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// Zero is the unassigned witness sentinel, and a nonzero sequence in an occupied cache slot does
// not become retireable until its readiness event has actually entered the bounded ring.
// REQ: RUNTIME-016
TEST(RuntimeChannelBinding, RejectsZeroAndOccupiedButUnemittedWitnessRetirement) {
  SessionFixture fixture;
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
  auto zero = (*binding)->retire(0U);
  ASSERT_FALSE(zero);
  EXPECT_EQ(zero.error().code, ErrorCode::kInvalidArgument);

  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{43}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  ASSERT_TRUE(wait_for_occupied_sample(**session, PortId{2U}));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_single_occupied_witness_state(
      **session, PortId{2U}, 17U, false, false));
  auto unemitted = (*binding)->retire(17U);
  ASSERT_FALSE(unemitted);
  EXPECT_EQ(unemitted.error().code, ErrorCode::kStaleEndpoint);

  auto event = wait_for_event(**session, **binding);
  ASSERT_TRUE(event) << runtime_error_message(event.error());
  ASSERT_TRUE((*binding)->retire(event->source_sequence));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
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
    EXPECT_EQ(reserved->has_value(), vector.expected_reserved);

    if (reserved->has_value()) {
      (*session)->release_freshness_reservation(**reserved);
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
  ASSERT_TRUE(reserved->has_value());
  EXPECT_EQ((**reserved).successor.capture_time, (TimePoint{20, ClockId{1U}}));
  EXPECT_EQ((**reserved).successor_reference, (TimePoint{43, ClockId{2U}}));
  auto selector = RuntimeInputSelectionState::create(fixture.plan.input_selections.front());
  ASSERT_TRUE(selector) << runtime_error_message(selector.error());
  ASSERT_TRUE(selector->push(*retained));
  EXPECT_FALSE(selector->staged_reuse_eligible((**reserved).successor.capture_time,
                                               TimePoint{43, ClockId{2U}}));
  EXPECT_TRUE(selector->staged_reuse_eligible((**reserved).successor_reference,
                                              TimePoint{43, ClockId{2U}}));
  ASSERT_TRUE(selector->evict_retained(retained->arrival_sequence));
  std::uint32_t promoted = 0U;
  ASSERT_TRUE((*session)->retire_samples(
      PortId{2U}, std::span<const TemporalSampleRecord>{&*retained, 1U}, &promoted, &**reserved));
  ASSERT_EQ(promoted, 1U);
  auto promoted_event = wait_for_event(**session, **binding, ClockId{2U});
  ASSERT_TRUE(promoted_event) << runtime_error_message(promoted_event.error());
  EXPECT_EQ(promoted_event->reference_time, (**reserved).successor_reference);
  auto promoted_sample =
      (*session)->temporal_sample(promoted_event->port, promoted_event->source_sequence);
  ASSERT_TRUE(promoted_sample) << runtime_error_message(promoted_sample.error());
  TemporalSampleRecord normalized_successor = (**reserved).successor;
  normalized_successor.capture_time = (**reserved).successor_reference;
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
    ASSERT_TRUE(requalified->has_value());

    // Key assertion: the release that empties the cache admits the reserved record in the same
    // call, rather than leaving the refill to whatever the pipeline does afterwards.
    std::uint32_t promoted = 0U;
    ASSERT_TRUE((*session)->retire_samples(PortId{2U},
                                           std::span<const TemporalSampleRecord>{&*retained, 1U},
                                           &promoted,
                                           &**requalified));
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
  ASSERT_TRUE(reserved->has_value());

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
      &**reserved));
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
    EXPECT_EQ(reservation->has_value(), vector.reservation_expected);
    if (reservation->has_value()) {
      arrangement.session->release_freshness_reservation(**reservation);
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
  ASSERT_TRUE(first->has_value());
  auto second = arrangement.session->reserve_freshness_refill(
      PortId{2U}, arrangement.retained, TimePoint{20, ClockId{1U}});
  ASSERT_TRUE(second) << runtime_error_message(second.error());
  ASSERT_TRUE(second->has_value());
  ASSERT_NE((*first)->generation, (*second)->generation);

  std::uint32_t promoted = 0U;
  auto stale = arrangement.session->retire_samples(
      PortId{2U},
      std::span<const TemporalSampleRecord>{&arrangement.retained, 1U},
      &promoted,
      &**first);
  ASSERT_FALSE(stale);
  EXPECT_EQ(stale.error().code, ErrorCode::kStaleEndpoint);
  EXPECT_EQ(promoted, 0U);
  auto retained = arrangement.session->temporal_sample(PortId{2U}, 1U);
  ASSERT_TRUE(retained) << runtime_error_message(retained.error());

  ASSERT_TRUE(arrangement.session->retire_samples(
      PortId{2U},
      std::span<const TemporalSampleRecord>{&arrangement.retained, 1U},
      &promoted,
      &**second));
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
  ASSERT_TRUE(reservation->has_value());
  EXPECT_EQ((**reservation).successor.capture_time.timestamp_ns, 30);
  EXPECT_EQ((**reservation).target.capture_time.timestamp_ns, 40);
  ASSERT_TRUE(selector->evict_retained(retained[0].arrival_sequence));
  std::uint32_t promoted = 0U;
  ASSERT_TRUE((*session)->retire_samples(PortId{2U},
                                         std::span<const TemporalSampleRecord>{&retained[0], 1U},
                                         &promoted,
                                         &**reservation));
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
  ASSERT_TRUE(final_reservation->has_value());
  EXPECT_EQ((**final_reservation).successor.capture_time.timestamp_ns, 40);
  EXPECT_EQ((**final_reservation).target.capture_time.timestamp_ns, 40);
  ASSERT_TRUE(selector->evict_retained(retained[1].arrival_sequence));
  promoted = 0U;
  ASSERT_TRUE((*session)->retire_samples(PortId{2U},
                                         std::span<const TemporalSampleRecord>{&retained[1], 1U},
                                         &promoted,
                                         &**final_reservation));
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
  ASSERT_TRUE(first_reservation->has_value());
  EXPECT_EQ((**first_reservation).target.capture_time.timestamp_ns, 40);

  promoted = 1U;
  ASSERT_TRUE((*session)->retire_samples(
      PortId{2U}, std::span<const TemporalSampleRecord>{&*third, 1U}, &promoted));
  ASSERT_EQ(promoted, 0U);
  auto replacement_reservation =
      (*session)->reserve_freshness_refill(PortId{2U}, retained[1], TimePoint{40, ClockId{1U}});
  ASSERT_TRUE(replacement_reservation) << runtime_error_message(replacement_reservation.error());
  ASSERT_TRUE(replacement_reservation->has_value());
  EXPECT_EQ((**replacement_reservation).successor, (**first_reservation).successor);
  EXPECT_EQ((**replacement_reservation).target, (**first_reservation).target);

  (*session)->release_freshness_reservation(**replacement_reservation);
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
  ASSERT_TRUE(both_staged->has_value());
  std::uint32_t promoted = 0U;
  ASSERT_TRUE(arrangement.session->retire_samples(
      PortId{2U},
      std::span<const TemporalSampleRecord>{&arrangement.retained, 1U},
      &promoted,
      &**both_staged));
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
  ASSERT_TRUE(target_reservation->has_value());
  EXPECT_EQ((**target_reservation).successor.capture_time.timestamp_ns, kNewestCaptureNs);
  EXPECT_EQ((**target_reservation).target.capture_time.timestamp_ns, kNewestCaptureNs);
  promoted = 0U;
  ASSERT_TRUE(
      arrangement.session->retire_samples(PortId{2U},
                                          std::span<const TemporalSampleRecord>{&*admitted, 1U},
                                          &promoted,
                                          &**target_reservation));
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
  ASSERT_TRUE(prefix->has_value());
  ASSERT_EQ((**prefix).successor.capture_time.timestamp_ns, 20);
  ASSERT_EQ((**prefix).target.capture_time.timestamp_ns, 30);
  std::uint32_t promoted = 0U;
  ASSERT_TRUE(arrangement.session->retire_samples(
      PortId{2U},
      std::span<const TemporalSampleRecord>{&arrangement.retained, 1U},
      &promoted,
      &**prefix));
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
  ASSERT_TRUE(protected_fresh->has_value());
  ASSERT_EQ((**protected_fresh).target.capture_time.timestamp_ns, 99);

  // The protected record occupies refill headroom, so this later producer sequence survives too.
  // Its capture coordinate is older even though its arrival identity is newer.
  ASSERT_TRUE(arrangement.emit(33, 20));
  auto newest_by_producer =
      arrangement.session->reserve_freshness_refill(PortId{2U},
                                                    arrangement.retained,
                                                    TimePoint{100, ClockId{1U}},
                                                    RuntimeChannelFreshnessSelectionMode::kNewest);
  ASSERT_TRUE(newest_by_producer) << runtime_error_message(newest_by_producer.error());
  ASSERT_TRUE(newest_by_producer->has_value());
  EXPECT_EQ((**newest_by_producer).successor.capture_time.timestamp_ns, 20);

  auto reservation = arrangement.session->reserve_freshness_refill(
      PortId{2U},
      arrangement.retained,
      TimePoint{100, ClockId{1U}},
      RuntimeChannelFreshnessSelectionMode::kNewestWithArrivalPrefix);
  ASSERT_TRUE(reservation) << runtime_error_message(reservation.error());
  ASSERT_TRUE(reservation->has_value());
  EXPECT_EQ((**reservation).successor.capture_time.timestamp_ns, 99);
  EXPECT_EQ((**reservation).target.capture_time.timestamp_ns, 99);

  arrangement.session->release_freshness_reservation(**reservation);
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
  ASSERT_TRUE(reserved->has_value());

  // Exercise the one extra slot while the reservation protects the authored-depth record.
  ASSERT_TRUE(arrangement.emit(33, kHeadroomCaptureNs));
  arrangement.session->release_freshness_reservation(**reserved);

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
  ASSERT_TRUE(newest->has_value());
  EXPECT_EQ((**newest).successor.capture_time.timestamp_ns, kNewestCaptureNs);
  arrangement.session->release_freshness_reservation(**newest);

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
  ASSERT_TRUE(reserved->has_value());

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
    arrangement.session->release_freshness_reservation(**reserved);
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
  arrangement.session->release_freshness_reservation(**reserved);
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
  ASSERT_TRUE(reserved->has_value());

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
    arrangement.session->release_freshness_reservation(**reserved);
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
  ASSERT_TRUE(reserved->has_value());

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
    arrangement.session->release_freshness_reservation(**reserved);
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
  ASSERT_TRUE(reserved->has_value());

  std::uint32_t promoted = 0U;
  ASSERT_TRUE(arrangement.session->retire_samples(
      PortId{2U},
      std::span<const TemporalSampleRecord>{&arrangement.retained, 1U},
      &promoted,
      &**reserved));
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
  ASSERT_TRUE(channel_reservation->has_value());
  auto selector_reservation = selector->reserve_reuse_replacement(arrangement.retained);
  ASSERT_TRUE(selector_reservation) << runtime_error_message(selector_reservation.error());
  ASSERT_TRUE(
      RuntimeChannelSessionTestAccess::fail_next_freshness_handoff_retry(*arrangement.session));

  std::uint32_t promoted = 0U;
  ASSERT_TRUE(arrangement.session->retire_samples(
      PortId{2U},
      std::span<const TemporalSampleRecord>{&arrangement.retained, 1U},
      &promoted,
      &**channel_reservation));
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
  ASSERT_TRUE(reserved->has_value());
  std::uint32_t promoted = 0U;
  ASSERT_TRUE(arrangement.session->retire_samples(
      PortId{2U},
      std::span<const TemporalSampleRecord>{&arrangement.retained, 1U},
      &promoted,
      &**reserved));
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

}  // namespace
}  // namespace holoscan::detail

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <fmt/format.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <holomq/errors.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/time/manual_clock.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "channel/channel_common.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_state.hpp"
#include "compiler/process_placement_internal.hpp"
#include "core/operator_spec_compiler_access.hpp"
#include "core/source_id_internal.hpp"
#include "runtime/endpoint_clock_health_runtime.hpp"
#include "runtime/graph_exec.hpp"
#include "runtime/runtime_channel_binding.hpp"
#include "runtime/runtime_channel_storage.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "runtime/runtime_rate_state.hpp"
#include "runtime/runtime_worker.hpp"
#include "support/host_unique_test_name.hpp"
#include "time/clock_registry_internal.hpp"

#include "runtime_channel_binding_test_support.hpp"

namespace holoscan::detail {
namespace {

using namespace std::chrono_literals;

using namespace runtime_channel_binding_test;

// One shared HoloMQ publication accounts for every accepted recipient before the serial
// controller establishes its post-compute activation cut.
// REQ: CHAN-101
// REQ: RUNTIME-019
TEST(RuntimeChannelBinding, EventlessScalarFanoutMaterializesEveryRecipientAtActivationCut) {
  SessionFixture fixture;
  fixture.use_one_slot_blocking_fanout();
  fixture.use_same_partition_eventless_transport();
  fixture.use_single_partition_runner();
  const auto counters = std::make_shared<RuntimePhaseCounters>();
  fixture.plan.executable_factories.phase_counters = counters;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto left_binding =
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
  auto right_binding =
      (*session)->create_readiness_binding(VertexId{3U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.plan.plan_generation,
                                               .run_generation = 1U,
                                               .kind = RuntimeEventKind::kInputArrival,
                                               .source = make_source_id(2U),
                                               .source_clock = ClockId{1U},
                                               .reference_clock = ClockId{1U},
                                               .capacity = 1U,
                                           });
  ASSERT_TRUE(left_binding) << runtime_error_message(left_binding.error());
  ASSERT_TRUE(right_binding) << runtime_error_message(right_binding.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());

  // Key setup: publish once to two eventless subscribers in the same execution partition, then
  // establish the ordinary activation cut before observing either readiness stream.
  auto indexes = RuntimeChannelSessionTestAccess::pump_index_state(**session);
  ASSERT_TRUE(indexes) << runtime_error_message(indexes.error());
  EXPECT_TRUE(indexes->exact);
  EXPECT_EQ(indexes->wait_handle_count, 0U);
  EXPECT_EQ(indexes->drain_subscriber_count, 2U);
  EXPECT_FALSE(indexes->background_thread_required);
  EXPECT_FALSE(indexes->background_thread_running);
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{42}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  ASSERT_TRUE((*session)->flush_process_local_activation_cut());
  auto left_event = wait_for_event(**session, **left_binding);
  auto right_event = wait_for_event(**session, **right_binding);
  ASSERT_TRUE(left_event) << runtime_error_message(left_event.error());
  ASSERT_TRUE(right_event) << runtime_error_message(right_event.error());

  // Key assertion: the single publication becomes two typed samples/readiness witnesses while
  // using no WaitSet attachment and exactly one bounded-or-skipped activation cut.
  const auto snapshot = counters->snapshot();
  const auto phase = [&](RuntimePhaseCounter counter) {
    return snapshot[RuntimePhaseCounters::index(counter)];
  };
  EXPECT_EQ(left_event->port, PortId{2U});
  EXPECT_EQ(right_event->port, PortId{3U});
  EXPECT_EQ(phase(RuntimePhaseCounter::kChannelActivationCuts), 1U);
  EXPECT_EQ(phase(RuntimePhaseCounter::kChannelActivationCutDrains) +
                phase(RuntimePhaseCounter::kChannelActivationCutSkips),
            1U);
  EXPECT_EQ(phase(RuntimePhaseCounter::kChannelSamplesReceived), 2U);
  EXPECT_EQ(phase(RuntimePhaseCounter::kChannelReadinessPublished), 2U);
  EXPECT_EQ(phase(RuntimePhaseCounter::kChannelActivationCutInventoryScans), 0U);
  EXPECT_EQ(phase(RuntimePhaseCounter::kChannelActivationCutSubscriberProbes), 2U);
  // The accepted publication supplies the exact recipients, so the cut performs no redundant
  // backend readiness query before receiving the two immediately visible HoloMQ heads.
  EXPECT_EQ(phase(RuntimePhaseCounter::kChannelActivationCutTransportProbes), 0U);
  EXPECT_EQ(phase(RuntimePhaseCounter::kChannelActivationCutMaterializations), 2U);

  auto left_sample = (*session)->temporal_sample(left_event->port, left_event->source_sequence);
  auto right_sample = (*session)->temporal_sample(right_event->port, right_event->source_sequence);
  ASSERT_TRUE(left_sample) << runtime_error_message(left_sample.error());
  ASSERT_TRUE(right_sample) << runtime_error_message(right_sample.error());
  EXPECT_EQ(left_sample->producer_epoch, right_sample->producer_epoch);
  EXPECT_EQ(left_sample->source_sequence, right_sample->source_sequence);
  ASSERT_TRUE((*left_binding)->retire(left_event->source_sequence));
  ASSERT_TRUE((*right_binding)->retire(right_event->source_sequence));
  ASSERT_TRUE((*session)->retire_samples(left_event->port,
                                         std::span<const TemporalSampleRecord>{&*left_sample, 1U}));
  ASSERT_TRUE((*session)->retire_samples(
      right_event->port, std::span<const TemporalSampleRecord>{&*right_sample, 1U}));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// A depth-one lossy service can replace both subscriber heads during one producer callback. The
// activation cut must materialize the common newest sequence without relying on event handles.
// REQ: CHAN-101
// REQ: CHAN-115
// REQ: RUNTIME-019
TEST(RuntimeChannelBinding, EventlessDropOldestFanoutMaterializesNewestSequenceForEveryRecipient) {
  SessionFixture fixture;
  fixture.use_one_slot_drop_oldest_fanout();
  fixture.use_same_partition_eventless_transport();
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto left_binding =
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
  auto right_binding =
      (*session)->create_readiness_binding(VertexId{3U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.plan.plan_generation,
                                               .run_generation = 1U,
                                               .kind = RuntimeEventKind::kInputArrival,
                                               .source = make_source_id(2U),
                                               .source_clock = ClockId{1U},
                                               .reference_clock = ClockId{1U},
                                               .capacity = 1U,
                                           });
  ASSERT_TRUE(left_binding) << runtime_error_message(left_binding.error());
  ASSERT_TRUE(right_binding) << runtime_error_message(right_binding.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());

  // Key setup: publish two values in one activation to replace both depth-one transport heads
  // before the eventless service reaches its post-compute activation cut.
  auto indexes = RuntimeChannelSessionTestAccess::pump_index_state(**session);
  ASSERT_TRUE(indexes) << runtime_error_message(indexes.error());
  EXPECT_EQ(indexes->wait_handle_count, 0U);
  EXPECT_EQ(indexes->drain_subscriber_count, 2U);
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{41}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{42}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  ASSERT_TRUE((*session)->flush_process_local_activation_cut());
  auto left_event = wait_for_event(**session, **left_binding);
  auto right_event = wait_for_event(**session, **right_binding);
  ASSERT_TRUE(left_event) << runtime_error_message(left_event.error());
  ASSERT_TRUE(right_event) << runtime_error_message(right_event.error());
  auto left_sample = (*session)->temporal_sample(left_event->port, left_event->source_sequence);
  auto right_sample = (*session)->temporal_sample(right_event->port, right_event->source_sequence);
  ASSERT_TRUE(left_sample) << runtime_error_message(left_sample.error());
  ASSERT_TRUE(right_sample) << runtime_error_message(right_sample.error());

  // Key assertion: both legs expose only sequence two, and each typed read observes the common
  // newest payload exactly once despite the absent WaitSet attachments.
  EXPECT_EQ(left_sample->source_sequence, 2U);
  EXPECT_EQ(right_sample->source_sequence, 2U);
  const std::array left_selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = std::span<const TemporalSampleRecord>{&*left_sample, 1U},
      .destructive = true,
  }};
  const std::array right_selection{RuntimeChannelActivationSelection{
      .port = PortId{3U},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = std::span<const TemporalSampleRecord>{&*right_sample, 1U},
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, left_selection));
  auto left_value = fixture.sink.input.receive();
  ASSERT_TRUE(left_value) << runtime_error_message(left_value.error());
  EXPECT_EQ(left_value->data, 42);
  ASSERT_TRUE((*left_binding)->retire(left_event->source_sequence));
  ASSERT_TRUE((*session)->retire_samples(left_event->port, left_selection.front().samples));
  auto left_retired = (*session)->retire_activation(VertexId{2U}, 1U, true);
  ASSERT_TRUE(left_retired) << runtime_error_message(left_retired.error());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{3U}, 1U, right_selection));
  auto right_value = fixture.second_sink.input.receive();
  ASSERT_TRUE(right_value) << runtime_error_message(right_value.error());
  EXPECT_EQ(right_value->data, 42);
  ASSERT_TRUE((*right_binding)->retire(right_event->source_sequence));
  ASSERT_TRUE((*session)->retire_samples(right_event->port, right_selection.front().samples));
  auto right_retired = (*session)->retire_activation(VertexId{3U}, 1U, true);
  ASSERT_TRUE(right_retired) << runtime_error_message(right_retired.error());
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

struct DropOldestFanoutObservation {
  std::uint64_t source_sequence{};
  std::uint64_t producer_epoch{};
  std::uint64_t metadata_sequence{};
  std::int32_t value{};

  friend bool operator==(const DropOldestFanoutObservation&,
                         const DropOldestFanoutObservation&) = default;
};

// Publish 41 then 42 in one source activation into a depth-one drop-oldest fanout, then read what
// each recipient selects, in recipient order.
void observe_drop_oldest_fanout(const bool direct,
                                std::vector<DropOldestFanoutObservation>& observed) {
  SessionFixture fixture;
  if (direct) {
    fixture.use_direct_drop_oldest_fanout();
  } else {
    fixture.use_one_slot_drop_oldest_fanout();
    fixture.use_same_partition_eventless_transport();
  }
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  const auto bind = [&](const std::uint32_t sink) {
    return (*session)->create_readiness_binding(VertexId{sink},
                                                RuntimePortBindingRequest{
                                                    .plan_generation = fixture.plan.plan_generation,
                                                    .run_generation = 1U,
                                                    .kind = RuntimeEventKind::kInputArrival,
                                                    .source = make_source_id(sink - 1U),
                                                    .source_clock = ClockId{1U},
                                                    .reference_clock = ClockId{1U},
                                                    .capacity = 1U,
                                                });
  };
  auto first_binding = bind(2U);
  auto second_binding = bind(3U);
  ASSERT_TRUE(first_binding) << runtime_error_message(first_binding.error());
  ASSERT_TRUE(second_binding) << runtime_error_message(second_binding.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  auto indexes = RuntimeChannelSessionTestAccess::pump_index_state(**session);
  ASSERT_TRUE(indexes) << runtime_error_message(indexes.error());
  EXPECT_EQ(indexes->wait_handle_count, 0U);
  EXPECT_EQ(indexes->drain_subscriber_count, direct ? 0U : 2U);

  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{41}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{42}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  ASSERT_TRUE((*session)->flush_process_local_activation_cut());

  const auto consume = [&](PortBinding& binding, const std::uint32_t sink, SinkOperator& reader) {
    auto event = wait_for_event(**session, binding);
    ASSERT_TRUE(event) << runtime_error_message(event.error());
    auto record = (*session)->temporal_sample(event->port, event->source_sequence);
    ASSERT_TRUE(record) << runtime_error_message(record.error());
    const std::array selection{RuntimeChannelActivationSelection{
        .port = event->port,
        .selection = {InputState::kSelected, SelectionReason::kNone},
        .samples = std::span<const TemporalSampleRecord>{&*record, 1U},
        .destructive = true,
    }};
    ASSERT_TRUE((*session)->prepare_activation(VertexId{sink}, 1U, selection));
    auto value = reader.input.receive();
    ASSERT_TRUE(value) << runtime_error_message(value.error());
    observed.push_back(DropOldestFanoutObservation{
        .source_sequence = record->source_sequence,
        .producer_epoch = record->producer_epoch,
        .metadata_sequence = value->metadata.sequence,
        .value = value->data,
    });
    ASSERT_TRUE(binding.retire(event->source_sequence));
    ASSERT_TRUE((*session)->retire_samples(event->port, selection.front().samples));
    auto retired = (*session)->retire_activation(VertexId{sink}, 1U, true);
    ASSERT_TRUE(retired) << runtime_error_message(retired.error());
  };
  ASSERT_NO_FATAL_FAILURE(consume(**first_binding, 2U, fixture.sink));
  ASSERT_NO_FATAL_FAILURE(consume(**second_binding, 3U, fixture.second_sink));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// A depth-one drop-oldest scalar fanout family stores each publication into every recipient by
// direct retention. Each recipient keeps the survivor, sequence, and producer epoch that the
// eventless HoloMQ family delivers, with no HoloMQ subscriber to drain.
// REQ: CHAN-160
TEST(RuntimeChannelBinding, DirectDropOldestFanoutKeepsTheHoloMQSurvivorForEveryRecipient) {
  std::vector<DropOldestFanoutObservation> holomq;
  std::vector<DropOldestFanoutObservation> direct;
  ASSERT_NO_FATAL_FAILURE(observe_drop_oldest_fanout(false, holomq));
  ASSERT_NO_FATAL_FAILURE(observe_drop_oldest_fanout(true, direct));

  // Key assertion: both recipients select only sequence two, carrying 42, under either delivery.
  ASSERT_EQ(direct.size(), 2U);
  for (const auto& recipient : direct) {
    EXPECT_EQ(recipient.source_sequence, 2U);
    EXPECT_EQ(recipient.metadata_sequence, 2U);
    EXPECT_EQ(recipient.value, 42);
    EXPECT_EQ(recipient.producer_epoch, direct.front().producer_epoch);
  }
  EXPECT_EQ(direct, holomq);
}

// Terminal close stores one end of stream into every recipient at the sequence after the last
// publication, and a repeated close does not advance it.
// REQ: CHAN-160
// REQ: CHAN-136
TEST(RuntimeChannelBinding, DirectDropOldestFanoutClosesEveryRecipientAtOneSequence) {
  SessionFixture fixture;
  fixture.use_direct_drop_oldest_fanout();
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{7}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));

  ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
  ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());

  // Key assertion: both recipients hold the data sample and end of stream at sequence two, and the
  // output consumed exactly one sequence for the terminal.
  for (const PortId input : {PortId{2U}, PortId{3U}}) {
    SCOPED_TRACE(::testing::Message() << "input " << input.value());
    const auto state = RuntimeChannelSessionTestAccess::input_reset_state(**session, input);
    ASSERT_TRUE(state) << runtime_error_message(state.error());
    EXPECT_EQ(state->occupied_samples, 1U);
    EXPECT_TRUE(state->end_of_stream_received);
    EXPECT_EQ(state->end_of_stream_transport_sequence, 2U);
  }
  const auto next = RuntimeChannelSessionTestAccess::output_next_sequence(**session, PortId{1U});
  ASSERT_TRUE(next) << runtime_error_message(next.error());
  EXPECT_EQ(*next, 3U);
}

// A store failure at the first recipient leaves the publication unsubmitted, as at fanout one. At
// a later recipient the earlier recipients already hold the sequence, so emit() consumes it,
// reports a partial publication, and fails the session.
// REQ: CHAN-160
TEST(RuntimeChannelBinding, DirectDropOldestFanoutContainsALaterRecipientFailure) {
  for (const PortId failing : {PortId{2U}, PortId{3U}}) {
    SCOPED_TRACE(::testing::Message() << "failing input " << failing.value());
    SessionFixture fixture;
    fixture.use_direct_drop_oldest_fanout();
    auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
    ASSERT_TRUE(session) << runtime_error_message(session.error());
    // Key setup: retain a later sequence in one recipient, so the producer's first sequence is
    // non-monotonic there and only there.
    const std::int32_t injected = 5;
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::inject_direct_inline_payload(
        **session,
        failing,
        std::as_bytes(std::span{&injected, 1U}),
        SampleMetadata{
            .sequence = 5U,
            .byte_size = sizeof(injected),
            .payload_kind = PayloadKind::kInlineTyped,
        }));
    ASSERT_TRUE((*session)->refresh_connections());
    ASSERT_TRUE((*session)->arm());
    ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));

    auto published = fixture.source.output.emit(std::int32_t{7});
    const bool session_healthy = (*session)->status().has_value();
    static_cast<void>((*session)->retire_activation(VertexId{1U}, 1U, false));
    (*session)->cancel();
    static_cast<void>((*session)->join());

    // Key assertion: recipients are stored in plan order. A first-recipient failure returns the
    // store error and keeps the sequence; a second-recipient failure consumes it and fails the
    // session with a partial publication.
    ASSERT_FALSE(published);
    const auto next = RuntimeChannelSessionTestAccess::output_next_sequence(**session, PortId{1U});
    ASSERT_TRUE(next) << runtime_error_message(next.error());
    if (failing == PortId{2U}) {
      EXPECT_EQ(published.error().code, ErrorCode::kProtocolError);
      EXPECT_EQ(*next, 1U);
      EXPECT_TRUE(session_healthy);
    } else {
      EXPECT_EQ(published.error().code, ErrorCode::kPublicationPartial);
      EXPECT_EQ(*next, 2U);
      EXPECT_FALSE(session_healthy);
    }
  }
}

// A blocking direct fanout family checks every recipient's typed capacity before storing into
// any. While one recipient is full, a publication that the other recipient has room for is still
// rejected before submission: neither recipient receives it, and the sequence is kept.
// REQ: CHAN-167
TEST(RuntimeChannelBinding, DirectBlockingFanoutRejectsBeforeAnyStoreWhileARecipientIsFull) {
  SessionFixture fixture;
  fixture.use_direct_blocking_fanout();
  // Key setup: the first recipient holds one sample, the second two.
  fixture.set_queue_depth(1U, 2U);
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{7}));
  auto rejected = fixture.source.output.emit(std::int32_t{8});
  static_cast<void>((*session)->retire_activation(VertexId{1U}, 1U, true));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());

  // Key assertion: the second publication is pre-submit backpressure, the output keeps its
  // sequence, and the recipient with room did not receive it.
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, ErrorCode::kBackpressured);
  const auto next = RuntimeChannelSessionTestAccess::output_next_sequence(**session, PortId{1U});
  ASSERT_TRUE(next) << runtime_error_message(next.error());
  EXPECT_EQ(*next, 2U);
  for (const PortId input : {PortId{2U}, PortId{3U}}) {
    SCOPED_TRACE(::testing::Message() << "input " << input.value());
    const auto state = RuntimeChannelSessionTestAccess::input_reset_state(**session, input);
    ASSERT_TRUE(state) << runtime_error_message(state.error());
    EXPECT_EQ(state->occupied_samples, 1U);
    EXPECT_EQ(state->last_producer_sequence, 1U);
  }
}

// Install one scripted transported leg, such as a cross-process or exported member, beside a direct
// drop-oldest fanout family.
[[nodiscard]] ScriptedInlinePublisher* add_scripted_transported_leg(
    RuntimeChannelSession& session, std::vector<channel_internal::InlinePublicationResult> results,
    std::vector<ErrorCode> close_results = {}) {
  auto publisher = std::make_unique<ScriptedInlinePublisher>(
      std::move(results), std::vector<ErrorCode>{}, std::move(close_results));
  auto* const observed = publisher.get();
  std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
  publishers.push_back(std::move(publisher));
  // A refreshed endpoint is not process-local transport, as a cross-process or export leg is not.
  const auto replaced = RuntimeChannelSessionTestAccess::replace_output_publishers(
      session, PortId{1U}, std::move(publishers), {}, true);
  EXPECT_TRUE(replaced) << (replaced ? std::string{} : runtime_error_message(replaced.error()));
  return replaced ? observed : nullptr;
}

// A mixed-locality family stores each publication into its direct recipients and then publishes it
// once to its transported leg at the same sequence. A leg that then rejects before submission
// leaves the publication indeterminate: the direct recipients already hold it, so the sequence is
// consumed, and the session stays healthy. A later close still ends the direct recipients at the
// next sequence. The leg's receiver infers its end of stream from the last publication it
// received, so after such a rejection it would infer an earlier sequence.
// REQ: CHAN-165
TEST(RuntimeChannelBinding, MixedLocalityFanoutStoresDirectRecipientsBeforeItsTransportedLeg) {
  SessionFixture fixture;
  fixture.use_direct_drop_oldest_fanout();
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto* const leg = add_scripted_transported_leg(
      **session,
      {
          channel_internal::InlinePublicationResult{
              .phase = channel_internal::InlinePublicationPhase::kAccepted,
              .recipient_count = 1U,
              .error = Error{ErrorCode::kSuccess},
          },
          channel_internal::InlinePublicationResult{
              .phase = channel_internal::InlinePublicationPhase::kPreSubmitRejected,
              .error = Error{ErrorCode::kBackpressured},
          },
      });
  ASSERT_NE(leg, nullptr);
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());

  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{7}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 2U, {}));
  auto rejected = fixture.source.output.emit(std::int32_t{8});
  const bool session_healthy = (*session)->status().has_value();
  static_cast<void>((*session)->retire_activation(VertexId{1U}, 2U, false));

  // Key assertion: the leg saw both sequences after the direct recipients held them, and its
  // pre-submit rejection of the second consumed that sequence without failing the session.
  EXPECT_EQ(leg->sequences, (std::vector<std::uint64_t>{1U, 2U}));
  EXPECT_EQ(leg->values, (std::vector<std::int32_t>{7, 8}));
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, ErrorCode::kPublicationIndeterminate);
  EXPECT_TRUE(session_healthy);
  for (const PortId input : {PortId{2U}, PortId{3U}}) {
    SCOPED_TRACE(::testing::Message() << "input " << input.value());
    const auto state = RuntimeChannelSessionTestAccess::input_reset_state(**session, input);
    ASSERT_TRUE(state) << runtime_error_message(state.error());
    EXPECT_EQ(state->occupied_samples, 1U);
    EXPECT_EQ(state->last_producer_sequence, 2U);
  }

  // The caller handled the error, so the producer still closes its output.
  ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
  EXPECT_EQ(leg->close_count, 1U);
  for (const PortId input : {PortId{2U}, PortId{3U}}) {
    SCOPED_TRACE(::testing::Message() << "input " << input.value());
    const auto state = RuntimeChannelSessionTestAccess::input_reset_state(**session, input);
    ASSERT_TRUE(state) << runtime_error_message(state.error());
    EXPECT_TRUE(state->end_of_stream_received);
    EXPECT_EQ(state->end_of_stream_transport_sequence, 3U);
  }
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
  const auto next = RuntimeChannelSessionTestAccess::output_next_sequence(**session, PortId{1U});
  ASSERT_TRUE(next) << runtime_error_message(next.error());
  EXPECT_EQ(*next, 4U);
}

// Terminal close stores end of stream into the direct recipients at the next sequence and then
// closes the transported leg, whose receiver infers the same sequence. A leg that backpressures
// is retried: the direct stores repeat harmlessly, and the sequence advances once.
// REQ: CHAN-165
// REQ: CHAN-136
TEST(RuntimeChannelBinding, MixedLocalityFanoutClosesEveryArmAtOneSequence) {
  SessionFixture fixture;
  fixture.use_direct_drop_oldest_fanout();
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto* const leg = add_scripted_transported_leg(**session, {}, {ErrorCode::kBackpressured});
  ASSERT_NE(leg, nullptr);
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{7}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));

  auto backpressured = (*session)->close_owner_outputs(VertexId{1U});
  ASSERT_FALSE(backpressured);
  EXPECT_EQ(backpressured.error().code, ErrorCode::kBackpressured);
  ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
  ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());

  // Key assertion: each direct recipient holds the data and end of stream at sequence two, the
  // leg closed once after its retry, and the output consumed exactly one terminal sequence.
  EXPECT_EQ(leg->close_count, 2U);
  for (const PortId input : {PortId{2U}, PortId{3U}}) {
    SCOPED_TRACE(::testing::Message() << "input " << input.value());
    const auto state = RuntimeChannelSessionTestAccess::input_reset_state(**session, input);
    ASSERT_TRUE(state) << runtime_error_message(state.error());
    EXPECT_EQ(state->occupied_samples, 1U);
    EXPECT_TRUE(state->end_of_stream_received);
    EXPECT_EQ(state->end_of_stream_transport_sequence, 2U);
  }
  const auto next = RuntimeChannelSessionTestAccess::output_next_sequence(**session, PortId{1U});
  ASSERT_TRUE(next) << runtime_error_message(next.error());
  EXPECT_EQ(*next, 3U);
}

// A drop-oldest family with two members in this process group and one in another stores each
// publication into the local recipients and publishes it once to the remote one, at one sequence.
// Terminal close ends every recipient at the next sequence: the local ones by a direct store, and
// the remote one by the transported close, whose receiver infers the same sequence. Each process
// group gets its own Channel session, as its worker would, both in this test process.
// REQ: CHAN-165
TEST(RuntimeChannelBinding, MixedLocalityFanoutDeliversOneSequenceAcrossProcessGroups) {
  Graph graph{test::make_host_unique_test_name("mixed-locality-fanout")};
  graph.set_default_clock(graph.bind_shared_clock("clock", std::make_shared<ManualClock>(0)));
  const auto source = graph.op<SourceOperator>("source");
  const auto first = graph.op<SinkOperator>("first");
  const auto second = graph.op<SinkOperator>("second");
  const auto remote = graph.op<SinkOperator>("remote");
  const ConnectionOptions drop_oldest{.queue_policy = QueuePolicy::kDropOldest, .queue_depth = 1U};
  graph.add_flow(source->output, first->input, drop_oldest);
  graph.add_flow(source->output, second->input, drop_oldest);
  graph.add_flow(source->output, remote->input, drop_oldest);
  constexpr std::array local_members{GraphObjectSelector{.graph_path = "source"},
                                     GraphObjectSelector{.graph_path = "first"},
                                     GraphObjectSelector{.graph_path = "second"}};
  constexpr std::array remote_members{GraphObjectSelector{.graph_path = "remote"}};
  ProcessPlacementBinding placement;
  placement
      .bind_process_role(
          "local", ProcessPlacement{.members = local_members, .launch_group = "local-process"})
      .bind_process_role(
          "remote", ProcessPlacement{.members = remote_members, .launch_group = "remote-process"});
  const ExecutionPlan plan = compile_with_process_placement(graph, CompileOptions{}, placement);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->process_local_channel_services.size(), 1U);
  EXPECT_EQ(state->process_local_channel_services.front().delivery_kind,
            ProcessLocalChannelDeliveryKind::kDirectRetained);
  EXPECT_EQ(state->process_local_channel_services.front().max_subscribers, 2U);
  ASSERT_EQ(state->cross_process_channel_services.size(), 1U);
  const ProcessGroupId local_group =
      state->cross_process_channel_services.front().source_process_group;
  const ProcessGroupId remote_group =
      state->cross_process_channel_services.front().target_process_group;
  ASSERT_NE(local_group, remote_group);

  // Each process group's session binds only the declarations of its own Operators.
  SourceOperator producer;
  std::array<SinkOperator, 3U> sinks;
  std::vector<RuntimeChannelPortReplayRecord> local_replay;
  std::vector<RuntimeChannelPortReplayRecord> remote_replay;
  std::vector<PortId> local_inputs;
  PortId remote_input{};
  PortId output{};
  VertexId source_vertex{};
  std::size_t next_sink = 0U;
  for (const auto& planned : state->ports) {
    if (planned.owner_kind != PortOwnerKind::kOperator) {
      continue;
    }
    const bool local = worker_owns_vertex(*state, local_group, planned.owner);
    void* declaration = nullptr;
    if (planned.direction == PortDirection::kOutput) {
      declaration = &producer.output;
      output = planned.id;
      source_vertex = planned.owner;
    } else {
      ASSERT_LT(next_sink, sinks.size());
      declaration = &sinks[next_sink++].input;
      if (local) {
        local_inputs.push_back(planned.id);
      } else {
        remote_input = planned.id;
      }
    }
    (local ? local_replay : remote_replay)
        .push_back(RuntimeChannelPortReplayRecord{
            .port = planned.id,
            .owner = planned.owner,
            .direction = planned.direction,
            .declaration = declaration,
            .type_ops = runtime_port_type_ops<std::int32_t>(),
        });
  }
  ASSERT_EQ(local_inputs.size(), 2U);
  ASSERT_TRUE(remote_input);
  // The subscriber exists before the publisher, as when the remote worker starts first.
  auto remote_session = RuntimeChannelSession::create(*state, remote_replay, remote_group);
  ASSERT_TRUE(remote_session) << runtime_error_message(remote_session.error());
  auto local_session = RuntimeChannelSession::create(*state, local_replay, local_group);
  ASSERT_TRUE(local_session) << runtime_error_message(local_session.error());
  ASSERT_TRUE((*remote_session)->refresh_connections());
  ASSERT_TRUE((*local_session)->refresh_connections());
  const auto shutdown = [&] {
    for (auto* const session : {local_session->get(), remote_session->get()}) {
      session->cancel();
      EXPECT_TRUE(session->join());
    }
  };
  // Receive into the remote input until `done` holds for its state.
  const auto pump_remote_until =
      [&](const auto done) -> expected<RuntimeChannelInputResetState, Error> {
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    for (;;) {
      auto observed =
          RuntimeChannelSessionTestAccess::input_reset_state(**remote_session, remote_input);
      if (!observed || done(*observed)) {
        return observed;
      }
      if (std::chrono::steady_clock::now() >= deadline) {
        return make_unexpected(Error{ErrorCode::kTimeout});
      }
      static_cast<void>(
          RuntimeChannelSessionTestAccess::receive_one(**remote_session, remote_input));
      std::this_thread::sleep_for(1ms);
    }
  };

  for (const std::int32_t value : {41, 42}) {
    const auto activation = static_cast<std::uint64_t>(value - 40);
    ASSERT_TRUE((*local_session)->prepare_activation(source_vertex, activation, {}));
    ASSERT_TRUE(producer.output.emit(value));
    ASSERT_TRUE((*local_session)->retire_activation(source_vertex, activation, true));
  }
  auto remote_data = pump_remote_until([](const RuntimeChannelInputResetState& observed) {
    return observed.last_producer_sequence == 2U;
  });
  std::vector<RuntimeChannelInputResetState> local_data;
  for (const PortId input : local_inputs) {
    auto observed = RuntimeChannelSessionTestAccess::input_reset_state(**local_session, input);
    ASSERT_TRUE(observed) << runtime_error_message(observed.error());
    local_data.push_back(*observed);
  }
  ASSERT_TRUE((*local_session)->close_owner_outputs(source_vertex));
  auto remote_terminal = pump_remote_until([](const RuntimeChannelInputResetState& observed) {
    return observed.end_of_stream_received;
  });

  // Key assertion: every recipient holds sequence two, the survivor of the depth-one family, and
  // reads end of stream at sequence three; the output consumed one terminal sequence.
  ASSERT_TRUE(remote_data) << runtime_error_message(remote_data.error());
  EXPECT_EQ(remote_data->occupied_samples, 1U);
  ASSERT_TRUE(remote_terminal) << runtime_error_message(remote_terminal.error());
  EXPECT_EQ(remote_terminal->end_of_stream_transport_sequence, 3U);
  for (std::size_t index = 0U; index < local_inputs.size(); ++index) {
    SCOPED_TRACE(::testing::Message() << "local input " << local_inputs[index].value());
    EXPECT_EQ(local_data[index].occupied_samples, 1U);
    EXPECT_EQ(local_data[index].last_producer_sequence, 2U);
    const auto observed =
        RuntimeChannelSessionTestAccess::input_reset_state(**local_session, local_inputs[index]);
    ASSERT_TRUE(observed) << runtime_error_message(observed.error());
    EXPECT_TRUE(observed->end_of_stream_received);
    EXPECT_EQ(observed->end_of_stream_transport_sequence, 3U);
  }
  shutdown();
  const auto next = RuntimeChannelSessionTestAccess::output_next_sequence(**local_session, output);
  ASSERT_TRUE(next) << runtime_error_message(next.error());
  EXPECT_EQ(*next, 4U);
}

// REQ: RUNTIME-019
TEST(RuntimeChannelBinding, PublishesOneAuthorityAcrossMixedPolicyFanout) {
  SessionFixture fixture;
  fixture.add_mixed_policy_fanout();
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
  ASSERT_TRUE(first_binding);
  ASSERT_TRUE(second_binding);
  EXPECT_EQ(*(*session)->readiness_capacity(VertexId{2U}), 8U);
  EXPECT_EQ(*(*session)->readiness_capacity(VertexId{3U}), 8U);
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{17}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));

  auto first_event = wait_for_event(**session, **first_binding);
  auto second_event = wait_for_event(**session, **second_binding);
  ASSERT_TRUE(first_event) << runtime_error_message(first_event.error());
  ASSERT_TRUE(second_event) << runtime_error_message(second_event.error());
  EXPECT_EQ(first_event->port, PortId{2U});
  EXPECT_EQ(second_event->port, PortId{3U});
  (*session)->cancel();
  EXPECT_TRUE((*session)->join());
}

}  // namespace
}  // namespace holoscan::detail

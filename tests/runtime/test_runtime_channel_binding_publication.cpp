/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <holomq/errors.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/profiler/nvtx3.hpp>
#include <holoscan/time/manual_clock.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "channel/channel_common.hpp"
#include "compiler/execution_plan_state.hpp"
#include "core/operator_spec_compiler_access.hpp"
#include "core/source_id_internal.hpp"
#include "runtime/endpoint_clock_health_runtime.hpp"
#include "runtime/graph_exec.hpp"
#include "runtime/runtime_channel_binding.hpp"
#include "runtime/runtime_channel_storage.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "runtime/runtime_rate_state.hpp"

#include "runtime_channel_binding_test_support.hpp"

namespace holoscan::test::nvtx_emit_marker_capture {

bool reset(std::string_view label);
std::size_t emitted() noexcept;

}  // namespace holoscan::test::nvtx_emit_marker_capture

namespace holoscan::detail {
namespace {

using namespace std::chrono_literals;

using namespace runtime_channel_binding_test;

// A transport close follows the last producer sequence without displacing its retained data.
// In particular, once the sole typed slot holds data, the terminal head must still be classified
// and recorded; neither borrowed nor hook-capable copy fallback may turn EOS into a data-capacity
// retry. Legacy copy-only endpoints conservatively backpressure until they implement the bounded
// operation-classification hook.
// REQ: CHAN-136
TEST(RuntimeChannelBinding, InlineTransportCloseRetainsDataThenPublishesOneEndOfStream) {
  constexpr std::int32_t kValue = 37;
  constexpr std::uint64_t kDataSequence = 41U;
  for (const InlineReceivePath path :
       {InlineReceivePath::kBorrowed, InlineReceivePath::kCopyFallback}) {
    SCOPED_TRACE(path == InlineReceivePath::kBorrowed ? "borrowed" : "copy fallback");
    SessionFixture fixture;
    fixture.use_one_slot_blocking_transport();
    auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
    ASSERT_TRUE(session) << runtime_error_message(session.error());
    auto replacement = std::make_unique<ScriptedInlineSubscriber>(kValue, kDataSequence, path);
    auto* const observed = replacement.get();
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_input_subscriber(
        **session, PortId{2U}, std::move(replacement)));
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

    // The first receive occupies the only typed destination. The next receive observes the
    // queued terminal without needing a data reservation, and the third proves idempotence.
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U}));
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U}));
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U}));
    const auto retained = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
    ASSERT_TRUE(retained) << runtime_error_message(retained.error());
    EXPECT_EQ(retained->occupied_samples, 1U);
    EXPECT_EQ(retained->next_arrival_sequence, 2U);
    EXPECT_TRUE(retained->end_of_stream_received);

    // The first kClosed stored exactly predecessor+1; repeating kClosed did not advance it.
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::store_end_of_stream(
        **session, PortId{2U}, kDataSequence + 1U));
    auto conflicting = RuntimeChannelSessionTestAccess::store_end_of_stream(
        **session, PortId{2U}, kDataSequence + 2U);
    ASSERT_FALSE(conflicting);
    EXPECT_EQ(conflicting.error().code, ErrorCode::kProtocolError);

    EXPECT_EQ(observed->borrowed_receive_count, 1U);
    EXPECT_EQ(observed->copy_receive_count, path == InlineReceivePath::kCopyFallback ? 1U : 0U);
    EXPECT_EQ(observed->loan_release_count, path == InlineReceivePath::kBorrowed ? 1U : 0U);
    EXPECT_EQ(observed->staged_data_query_count, 0U);

    ASSERT_TRUE((*binding)->poll(TimePoint{1, ClockId{1U}}));
    std::array<RuntimeEvent, 3U> events{};
    auto drained = (*binding)->drain(events);
    ASSERT_TRUE(drained) << runtime_error_message(drained.error());
    ASSERT_EQ(*drained, 2U);
    EXPECT_EQ(events[0].kind, RuntimeEventKind::kInputArrival);
    EXPECT_EQ(events[1].kind, RuntimeEventKind::kInputEndOfStream);
    EXPECT_EQ(events[0].source_sequence, 1U);
    EXPECT_EQ(events[1].source_sequence, 2U);

    auto record = (*session)->temporal_sample(events[0].port, events[0].source_sequence);
    ASSERT_TRUE(record) << runtime_error_message(record.error());
    EXPECT_EQ(record->source_sequence, kDataSequence);
    const std::array selection{RuntimeChannelActivationSelection{
        .port = PortId{2U},
        .selection = {InputState::kSelected, SelectionReason::kNone},
        .samples = std::span<const TemporalSampleRecord>{&*record, 1U},
        .destructive = true,
    }};
    ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, selection));
    auto received = fixture.sink.input.receive();
    ASSERT_TRUE(received) << runtime_error_message(received.error());
    EXPECT_EQ(received->data, kValue);
    EXPECT_EQ(received->metadata.sequence, kDataSequence);
    ASSERT_TRUE((*binding)->retire(events[0].source_sequence));
    ASSERT_TRUE((*binding)->retire(events[1].source_sequence));
    ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
    ASSERT_TRUE((*session)->retire_samples(PortId{2U},
                                           std::span<const TemporalSampleRecord>{&*record, 1U}));

    EXPECT_FALSE((*binding)->poll(TimePoint{2, ClockId{1U}}));
    auto repeated = (*binding)->drain(events);
    ASSERT_TRUE(repeated) << runtime_error_message(repeated.error());
    EXPECT_EQ(*repeated, 0U);
  }
}

// REQ: RUNTIME-019
TEST(RuntimeChannelBinding, PublishesAConstructedGenericLoan) {
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
  ASSERT_TRUE(binding);
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  std::this_thread::sleep_for(20ms);
  auto loan = fixture.source.output.allocate();
  ASSERT_TRUE(loan) << runtime_error_message(loan.error());
  loan->value() = 73;
  ASSERT_TRUE(fixture.source.output.emit(std::move(*loan)));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));

  auto event = wait_for_event(**session, **binding);
  ASSERT_TRUE(event) << runtime_error_message(event.error());
  auto temporal = (*session)->temporal_sample(event->port, event->source_sequence);
  ASSERT_TRUE(temporal);
  const std::array selected{*temporal};
  const std::array selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = selected,
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, selection));
  auto value = fixture.sink.input.receive_data();
  ASSERT_TRUE(value);
  EXPECT_EQ(*value, 73);
  ASSERT_TRUE((*binding)->retire(event->source_sequence));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  (*session)->cancel();
  EXPECT_TRUE((*session)->join());
}

// A proven pre-submit rejection leaves the exact caller-owned loan armed. The Runtime does not
// allocate a replacement or retry it; the caller may correct the condition and explicitly submit
// the same authority again during the same activation.
// REQ: RUNTIME-019
TEST(RuntimeChannelBinding, PreservesGenericLoanAcrossPreSubmitBackpressure) {
  Graph graph;
  SessionFixture fixture;
  (void)graph.add_operator<SourceOperator>("source");
  (void)graph.add_operator<SinkOperator>("sink");
  fixture.plan.graph = graph.view();
  fixture.plan.generic_output_loans.front().max_live_loans = 1U;
  (void)profiler::operator_profile_domain();
  ASSERT_TRUE(test::nvtx_emit_marker_capture::reset("source.output:emit"));
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto publisher = std::make_unique<ScriptedInlinePublisher>(
      std::vector<channel_internal::InlinePublicationResult>{
          channel_internal::InlinePublicationResult{
              .phase = channel_internal::InlinePublicationPhase::kPreSubmitRejected,
              .recipient_count = 0U,
              .error = Error{ErrorCode::kBackpressured},
          },
          channel_internal::InlinePublicationResult{
              .phase = channel_internal::InlinePublicationPhase::kAccepted,
              .recipient_count = 1U,
              .error = Error{ErrorCode::kSuccess},
          },
      });
  auto* observed = publisher.get();
  std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
  publishers.push_back(std::move(publisher));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_output_publishers(
      **session, PortId{1U}, std::move(publishers)));
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));

  auto loan = fixture.source.output.allocate();
  ASSERT_TRUE(loan) << runtime_error_message(loan.error());
  loan->value() = 22;
  auto rejected = fixture.source.output.emit(std::move(*loan));
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, ErrorCode::kBackpressured);
  EXPECT_EQ(test::nvtx_emit_marker_capture::emitted(), 0U);
  auto still_held = fixture.source.output.allocate();
  ASSERT_FALSE(still_held);
  EXPECT_EQ(still_held.error().code, ErrorCode::kResourceExhausted);

  ASSERT_TRUE(fixture.source.output.emit(std::move(*loan)));
  EXPECT_EQ(test::nvtx_emit_marker_capture::emitted(), 1U);
  auto replacement = fixture.source.output.allocate();
  ASSERT_TRUE(replacement) << runtime_error_message(replacement.error());
  replacement->cancel();
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  EXPECT_EQ(observed->sequences, (std::vector<std::uint64_t>{1U, 1U}));
  EXPECT_EQ(observed->values, (std::vector<std::int32_t>{22, 22}));
}

// Nonconsuming and consuming outcomes retain their existing accounting across a legal cutover.
// REQ: RUNTIME-052
TEST(RuntimeChannelBinding, PublicationOutcomeAccountingSurvivesRestart) {
  // Key setup: source.output -> scripted publisher; reject, consume uncertain authority, then
  // accept, with a committed restart between each publication attempt.
  // Key assertion: observed sequences are 1, 1, 2 and next is 3; only consuming outcomes advance.
  SessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session);
  auto publisher = std::make_unique<ScriptedInlinePublisher>(
      std::vector<channel_internal::InlinePublicationResult>{
          {.phase = channel_internal::InlinePublicationPhase::kPreSubmitRejected,
           .recipient_count = 0U,
           .error = Error{ErrorCode::kBackpressured}},
          {.phase = channel_internal::InlinePublicationPhase::kPostSubmitUncertain,
           .recipient_count = 0U,
           .error = Error{ErrorCode::kPublicationIndeterminate}},
          {.phase = channel_internal::InlinePublicationPhase::kAccepted,
           .recipient_count = 1U,
           .error = Error{ErrorCode::kSuccess}},
      });
  auto* const observed = publisher.get();
  std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
  publishers.push_back(std::move(publisher));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_output_publishers(
      **session, PortId{1U}, std::move(publishers)));
  for (const std::uint64_t generation : {1U, 2U, 3U}) {
    ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, generation, {}));
    auto emitted = fixture.source.output.emit(std::int32_t{31});
    if (generation < 3U) {
      ASSERT_FALSE(emitted);
      EXPECT_EQ(
          emitted.error().code,
          generation == 1U ? ErrorCode::kBackpressured : ErrorCode::kPublicationIndeterminate);
    } else {
      ASSERT_TRUE(emitted);
    }
    ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, generation, true));
    const std::array affected{VertexId{1U}, VertexId{2U}};
    ASSERT_TRUE((*session)->commit_restart(VertexId{1U}, affected));
  }
  EXPECT_EQ(observed->sequences, (std::vector<std::uint64_t>{1U, 1U, 2U}));
  auto counter = RuntimeChannelSessionTestAccess::output_next_sequence(**session, PortId{1U});
  ASSERT_TRUE(counter);
  EXPECT_EQ(*counter, 3U);
}

// A pre-submit refresh failure owns no close authority; retry publishes one terminal operation.
// REQ: CHAN-136
TEST(RuntimeChannelBinding, CloseOwnerOutputsRefreshesBeforePublishingEndOfStream) {
  SessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto publisher = std::make_unique<ScriptedInlinePublisher>(
      std::vector<channel_internal::InlinePublicationResult>{},
      std::vector{ErrorCode::kBackpressured, ErrorCode::kSuccess},
      std::vector{ErrorCode::kSuccess});
  auto* const observed = publisher.get();
  std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
  publishers.push_back(std::move(publisher));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_output_publishers(
      **session, PortId{1U}, std::move(publishers)));

  auto pressured = (*session)->close_owner_outputs(VertexId{1U});
  ASSERT_FALSE(pressured);
  EXPECT_EQ(pressured.error().code, ErrorCode::kBackpressured);
  EXPECT_EQ(observed->close_count, 0U);

  ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
  EXPECT_EQ(observed->close_count, 1U);
  ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
  EXPECT_EQ(observed->close_count, 1U);
}

// Bounded catalog-publisher loan exhaustion happens before terminal submission. Transport
// locality does not change that authority fact: retry must reuse the same unresolved close.
// REQ: CHAN-136
TEST(RuntimeChannelBinding, NonLocalPreSubmitResourceExhaustionRemainsRetryable) {
  SessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto publisher = std::make_unique<ScriptedInlinePublisher>(
      std::vector<channel_internal::InlinePublicationResult>{},
      std::vector<ErrorCode>{},
      std::vector{ErrorCode::kResourceExhausted, ErrorCode::kSuccess});
  auto* const observed = publisher.get();
  std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
  publishers.push_back(std::move(publisher));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_output_publishers(
      **session, PortId{1U}, std::move(publishers), {}, true));

  auto pressured = (*session)->close_owner_outputs(VertexId{1U});
  ASSERT_FALSE(pressured);
  EXPECT_EQ(pressured.error().code, ErrorCode::kBackpressured);
  EXPECT_EQ(observed->close_count, 1U);

  ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
  EXPECT_EQ(observed->close_count, 2U);
  ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
  EXPECT_EQ(observed->close_count, 2U);
}

// A retry preserves every accepted fanout arm and submits only the still-unresolved suffix.
// REQ: CHAN-136
// REQ: RUNTIME-052
TEST(RuntimeChannelBinding, CloseOwnerOutputsCompletesAfterPartialFanoutRetry) {
  // Key setup: source.output --+-> accepting arm
  //                          +-> backpressured-then-accepting arm
  // Start at one and at UINT64_MAX - 1; retry the partially completed fanout.
  // Key assertion: the resolved arm is never refreshed or closed again, and final completion
  // charges one coordinate while subsequent completed closes remain idempotent at exhaustion.
  for (const std::uint64_t sequence :
       {std::uint64_t{1U}, std::numeric_limits<std::uint64_t>::max() - 1U}) {
    SCOPED_TRACE(sequence);
    SessionFixture fixture;
    auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
    ASSERT_TRUE(session) << runtime_error_message(session.error());
    ASSERT_TRUE(
        RuntimeChannelSessionTestAccess::set_output_next_sequence(**session, PortId{1U}, sequence));
    auto first = std::make_unique<ScriptedInlinePublisher>(
        std::vector<channel_internal::InlinePublicationResult>{},
        std::vector{ErrorCode::kSuccess, ErrorCode::kProtocolError},
        std::vector{ErrorCode::kSuccess, ErrorCode::kClosed});
    auto second = std::make_unique<ScriptedInlinePublisher>(
        std::vector<channel_internal::InlinePublicationResult>{},
        std::vector{ErrorCode::kSuccess, ErrorCode::kSuccess},
        std::vector{ErrorCode::kBackpressured, ErrorCode::kSuccess});
    auto* const first_observed = first.get();
    auto* const second_observed = second.get();
    std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
    publishers.push_back(std::move(first));
    publishers.push_back(std::move(second));
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_output_publishers(
        **session, PortId{1U}, std::move(publishers), {}, true));

    auto pressured = (*session)->close_owner_outputs(VertexId{1U});
    ASSERT_FALSE(pressured);
    EXPECT_EQ(pressured.error().code, ErrorCode::kBackpressured);
    EXPECT_EQ(first_observed->close_count, 1U);
    EXPECT_EQ(second_observed->close_count, 1U);

    ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
    EXPECT_EQ(first_observed->close_count, 1U);
    EXPECT_EQ(second_observed->close_count, 2U);

    ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
    EXPECT_EQ(first_observed->close_count, 1U);
    EXPECT_EQ(second_observed->close_count, 2U);
    auto counter = RuntimeChannelSessionTestAccess::output_next_sequence(**session, PortId{1U});
    ASSERT_TRUE(counter);
    EXPECT_EQ(*counter, sequence + 1U);
  }
}

// Observe cold terminal accounting without conflating arm resolution with published EOS.
// Representative caller: ExhaustedOptionalNoPeerCloseResolvesLocally.
void expect_output_close_state(const RuntimeChannelSession& session,
                               const std::uint64_t next_sequence, const std::size_t resolved_arms) {
  auto sequence = RuntimeChannelSessionTestAccess::output_next_sequence(session, PortId{1U});
  ASSERT_TRUE(sequence) << runtime_error_message(sequence.error());
  EXPECT_EQ(*sequence, next_sequence);
  auto resolved =
      RuntimeChannelSessionTestAccess::resolved_output_terminal_arms(session, PortId{1U});
  ASSERT_TRUE(resolved) << runtime_error_message(resolved.error());
  EXPECT_EQ(*resolved, resolved_arms);
}

// REQ: RUNTIME-052
// REQ: CHAN-136
TEST(RuntimeChannelBinding, ExhaustedOptionalNoPeerCloseResolvesLocally) {
  // Key setup: an optional exported arm has neither a peer nor accepted recipient data.
  // Fixture topology: source.output -> scripted optional export (no receiver).
  for (const std::uint64_t sequence :
       {std::uint64_t{1U}, std::numeric_limits<std::uint64_t>::max()}) {
    SCOPED_TRACE(sequence);
    SessionFixture fixture;
    auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
    ASSERT_TRUE(session);
    auto publisher = std::make_unique<ScriptedInlinePublisher>(
        std::vector<channel_internal::InlinePublicationResult>{});
    auto* const observed = publisher.get();
    observed->subscribers_present = false;
    std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
    publishers.push_back(std::move(publisher));
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_output_publishers(
        **session, PortId{1U}, std::move(publishers), {}, true, false));
    ASSERT_TRUE(
        RuntimeChannelSessionTestAccess::set_output_next_sequence(**session, PortId{1U}, sequence));

    // Key assertion: local completion resolves the arm without EOS or sequence consumption,
    // even when the next counter is exhausted; a repeated completed close performs no work.
    ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
    EXPECT_EQ(observed->refresh_count, 1U);
    EXPECT_EQ(observed->close_count, 0U);
    expect_output_close_state(**session, sequence, 1U);
    ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
    EXPECT_EQ(observed->refresh_count, 1U);
    EXPECT_EQ(observed->close_count, 0U);
    expect_output_close_state(**session, sequence, 1U);
    auto reset = (*session)->reset();
    ASSERT_FALSE(reset);
    EXPECT_EQ(reset.error().code, ErrorCode::kClosed);
  }
}

// REQ: RUNTIME-052
// REQ: CHAN-136
// REQ: CHAN-144
TEST(RuntimeChannelBinding, ExhaustedRequiredExportAbandonmentReturnsClosed) {
  // Key setup: source.output -> required export -> recipient; accept data before the
  // recipient disappears at terminal refresh, with the Runtime next counter exhausted.
  SessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session);
  auto publisher = std::make_unique<ScriptedInlinePublisher>(
      std::vector<channel_internal::InlinePublicationResult>{});
  auto* const observed = publisher.get();
  std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
  publishers.push_back(std::move(publisher));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_output_publishers(
      **session, PortId{1U}, std::move(publishers), {}, true));
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{41}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  ASSERT_EQ(observed->sequences, (std::vector<std::uint64_t>{1U}));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_output_next_sequence(
      **session, PortId{1U}, std::numeric_limits<std::uint64_t>::max()));
  const auto refreshes = observed->refresh_count;
  observed->remove_subscribers_on_refresh = true;

  // Key assertion: abandonment resolves the arm with kClosed, not exhaustion, and never
  // submits/counts EOS. Neither reset nor a close retry can reopen the abandoned arm.
  auto closed = (*session)->close_owner_outputs(VertexId{1U});
  ASSERT_FALSE(closed);
  EXPECT_EQ(closed.error().code, ErrorCode::kClosed);
  EXPECT_EQ(observed->refresh_count, refreshes + 1U);
  EXPECT_EQ(observed->close_count, 0U);
  expect_output_close_state(**session, std::numeric_limits<std::uint64_t>::max(), 1U);
  auto reset = (*session)->reset();
  ASSERT_FALSE(reset);
  EXPECT_EQ(reset.error().code, ErrorCode::kClosed);
  ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
  EXPECT_EQ(observed->refresh_count, refreshes + 1U);
  EXPECT_EQ(observed->close_count, 0U);
  expect_output_close_state(**session, std::numeric_limits<std::uint64_t>::max(), 1U);
}

// REQ: RUNTIME-052
// REQ: CHAN-136
TEST(RuntimeChannelBinding, ExhaustedOptionalAcceptedRecipientRemainsObligated) {
  // Key setup: source.output -> optional export -> recipient; accept data before the
  // recipient disappears at terminal refresh, with the Runtime next counter exhausted.
  SessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session);
  auto publisher = std::make_unique<ScriptedInlinePublisher>(
      std::vector<channel_internal::InlinePublicationResult>{});
  auto* const observed = publisher.get();
  std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
  publishers.push_back(std::move(publisher));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_output_publishers(
      **session, PortId{1U}, std::move(publishers), {}, true, false));
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{41}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  ASSERT_EQ(observed->sequences, (std::vector<std::uint64_t>{1U}));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_output_next_sequence(
      **session, PortId{1U}, std::numeric_limits<std::uint64_t>::max()));
  const auto refreshes = observed->refresh_count;
  observed->remove_subscribers_on_refresh = true;

  // Key assertion: accepted recipient data preserves the optional arm's terminal obligation;
  // losing that peer cannot turn exhausted EOS into successful local resolution.
  auto closed = (*session)->close_owner_outputs(VertexId{1U});
  ASSERT_FALSE(closed);
  EXPECT_EQ(closed.error().code, ErrorCode::kResourceExhausted);
  EXPECT_EQ(observed->refresh_count, refreshes + 1U);
  EXPECT_EQ(observed->close_count, 0U);
  expect_output_close_state(**session, std::numeric_limits<std::uint64_t>::max(), 0U);
}

// REQ: RUNTIME-052
// REQ: CHAN-136
TEST(RuntimeChannelBinding, ExhaustedClosePreservesRefreshFailure) {
  // Key setup: source.output -> scripted connected export; terminal refresh fails before
  // the exhausted output can classify its recipient.
  SessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session);
  auto publisher = std::make_unique<ScriptedInlinePublisher>(
      std::vector<channel_internal::InlinePublicationResult>{},
      std::vector{ErrorCode::kBackpressured});
  auto* const observed = publisher.get();
  std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
  publishers.push_back(std::move(publisher));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_output_publishers(
      **session, PortId{1U}, std::move(publishers), {}, true));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_output_next_sequence(
      **session, PortId{1U}, std::numeric_limits<std::uint64_t>::max()));
  // Key assertion: the existing refresh error survives without EOS or terminal resolution.
  auto closed = (*session)->close_owner_outputs(VertexId{1U});
  ASSERT_FALSE(closed);
  EXPECT_EQ(closed.error().code, ErrorCode::kBackpressured);
  EXPECT_EQ(observed->refresh_count, 1U);
  EXPECT_EQ(observed->close_count, 0U);
  expect_output_close_state(**session, std::numeric_limits<std::uint64_t>::max(), 0U);
}

// REQ: RUNTIME-052
// REQ: CHAN-136
TEST(RuntimeChannelBinding, ExhaustedClosePreservesEarlierLocalResolution) {
  // Key setup: source.output -> optional absent arm
  //                         -> optional connected arm requiring EOS.
  SessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session);
  auto first = std::make_unique<ScriptedInlinePublisher>(
      std::vector<channel_internal::InlinePublicationResult>{});
  auto second = std::make_unique<ScriptedInlinePublisher>(
      std::vector<channel_internal::InlinePublicationResult>{});
  auto* const local = first.get();
  auto* const connected = second.get();
  local->subscribers_present = false;
  std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
  publishers.push_back(std::move(first));
  publishers.push_back(std::move(second));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_output_publishers(
      **session, PortId{1U}, std::move(publishers), {}, true, false));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_output_next_sequence(
      **session, PortId{1U}, std::numeric_limits<std::uint64_t>::max()));
  // Key assertion: exhaustion on the connected arm preserves the earlier local resolution;
  // retries skip that resolved arm and neither arm submits EOS.
  for (std::size_t attempt = 0U; attempt < 2U; ++attempt) {
    auto closed = (*session)->close_owner_outputs(VertexId{1U});
    ASSERT_FALSE(closed);
    EXPECT_EQ(closed.error().code, ErrorCode::kResourceExhausted);
    EXPECT_EQ(local->refresh_count, 1U);
    EXPECT_EQ(local->close_count, 0U);
    EXPECT_EQ(connected->close_count, 0U);
    expect_output_close_state(**session, std::numeric_limits<std::uint64_t>::max(), 1U);
  }
}

// REQ: RUNTIME-052
// REQ: CHAN-136
TEST(RuntimeChannelBinding, TerminalCompletionRetryRemembersConsumedAuthority) {
  // Key setup: source.output --+-> optional accepting arm (second scenario only)
  //                          +-> authority-consuming uncertain arm
  // Exercise uncertainty alone and after an accepted arm so either can establish accounting.
  for (const bool accepted_prefix : {false, true}) {
    SCOPED_TRACE(accepted_prefix);
    SessionFixture fixture;
    auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
    ASSERT_TRUE(session);
    auto publisher = std::make_unique<ScriptedInlinePublisher>(
        std::vector<channel_internal::InlinePublicationResult>{},
        std::vector<ErrorCode>{},
        std::vector{ErrorCode::kPublicationIndeterminate});
    auto* const observed = publisher.get();
    std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
    auto accepted = std::make_unique<ScriptedInlinePublisher>(
        std::vector<channel_internal::InlinePublicationResult>{});
    auto* const first = accepted.get();
    if (accepted_prefix) {
      publishers.push_back(std::move(accepted));
    }
    publishers.push_back(std::move(publisher));
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_output_publishers(
        **session, PortId{1U}, std::move(publishers), {}, true));
    const auto last_sequence = std::numeric_limits<std::uint64_t>::max() - 1U;
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_output_next_sequence(
        **session, PortId{1U}, last_sequence));
    auto uncertain = (*session)->close_owner_outputs(VertexId{1U});
    ASSERT_FALSE(uncertain);
    EXPECT_EQ(uncertain.error().code, ErrorCode::kPublicationIndeterminate);
    expect_output_close_state(**session, last_sequence, accepted_prefix ? 2U : 1U);
    // Key assertion: a retry which submits no EOS still charges the earlier consumed operation
    // exactly once; a completed retry at exhaustion remains idempotent.
    ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
    ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
    if (accepted_prefix) {
      EXPECT_EQ(first->close_count, 1U);
    }
    EXPECT_EQ(observed->close_count, 1U);
    expect_output_close_state(
        **session, std::numeric_limits<std::uint64_t>::max(), accepted_prefix ? 2U : 1U);
  }
}

// REQ: RUNTIME-052
// REQ: CHAN-136
TEST(RuntimeChannelBinding, ExhaustedDirectCloseDoesNotStoreEndOfStream) {
  // Key setup: source.output -> direct-retained sink.input, with exhausted output capacity.
  // Key assertion: no terminal storage, delivery count, or sequence changes on repeated close.
  SessionFixture fixture;
  fixture.use_direct_retained_delivery();
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session);
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_output_next_sequence(
      **session, PortId{1U}, std::numeric_limits<std::uint64_t>::max()));
  for (std::size_t attempt = 0U; attempt < 2U; ++attempt) {
    auto closed = (*session)->close_owner_outputs(VertexId{1U});
    ASSERT_FALSE(closed);
    EXPECT_EQ(closed.error().code, ErrorCode::kResourceExhausted);
    expect_output_close_state(**session, std::numeric_limits<std::uint64_t>::max(), 0U);
    auto input = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
    ASSERT_TRUE(input);
    EXPECT_FALSE(input->end_of_stream_received);
    auto progress = RuntimeChannelSessionTestAccess::process_local_progress_state(**session);
    ASSERT_TRUE(progress);
    EXPECT_EQ(progress->published_deliveries, 0U);
    EXPECT_EQ(progress->publications_in_flight, 0U);
  }
}

// A newly exhausted close may refresh its recipient but must not submit EOS.
// REQ: RUNTIME-052
TEST(RuntimeChannelBinding, ExhaustedCloseDoesNotSubmitEndOfStream) {
  // Key setup: source.output -> scripted connected export; refresh succeeds at exhaustion.
  SessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session);
  auto publisher = std::make_unique<ScriptedInlinePublisher>(
      std::vector<channel_internal::InlinePublicationResult>{}, std::vector{ErrorCode::kSuccess});
  auto* const observed = publisher.get();
  std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
  publishers.push_back(std::move(publisher));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_output_publishers(
      **session, PortId{1U}, std::move(publishers), {}, true));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_output_next_sequence(
      **session, PortId{1U}, std::numeric_limits<std::uint64_t>::max()));
  auto closed = (*session)->close_owner_outputs(VertexId{1U});
  // Key assertion: classification refresh runs, but exhausted EOS consumes no authority.
  ASSERT_FALSE(closed);
  EXPECT_EQ(closed.error().code, ErrorCode::kResourceExhausted);
  EXPECT_EQ(observed->close_count, 0U);
  EXPECT_EQ(observed->refresh_count, 1U);
  EXPECT_TRUE(observed->sequences.empty());
  expect_output_close_state(**session, std::numeric_limits<std::uint64_t>::max(), 0U);
}

// Terminal authority is monotonic within one session; an administrative reset cannot reopen an
// endpoint after any arm has committed EOS.
// REQ: CHAN-136
TEST(RuntimeChannelBinding, FullResetCannotReopenClosedOwnerOutputs) {
  SessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto publisher = std::make_unique<ScriptedInlinePublisher>(
      std::vector<channel_internal::InlinePublicationResult>{});
  auto* const observed = publisher.get();
  std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
  publishers.push_back(std::move(publisher));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_output_publishers(
      **session, PortId{1U}, std::move(publishers)));

  ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
  auto reset = (*session)->reset();
  ASSERT_FALSE(reset);
  EXPECT_EQ(reset.error().code, ErrorCode::kClosed);
  ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));

  EXPECT_EQ(observed->close_count, 1U);
}

// An authority-consuming close result resolves only that fanout arm. Lifecycle control cannot
// reopen the consumed authority, while an owner-close retry skips it and closes the unresolved
// suffix exactly once.
// REQ: CHAN-136
TEST(RuntimeChannelBinding, AuthorityConsumingCloseFailureResolvesOnlyThatArmAcrossRetry) {
  SessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto first = std::make_unique<ScriptedInlinePublisher>(
      std::vector<channel_internal::InlinePublicationResult>{},
      std::vector<ErrorCode>{},
      std::vector{ErrorCode::kPublicationIndeterminate});
  auto second = std::make_unique<ScriptedInlinePublisher>(
      std::vector<channel_internal::InlinePublicationResult>{},
      std::vector<ErrorCode>{},
      std::vector{ErrorCode::kSuccess});
  auto* const first_observed = first.get();
  auto* const second_observed = second.get();
  std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
  publishers.push_back(std::move(first));
  publishers.push_back(std::move(second));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_output_publishers(
      **session, PortId{1U}, std::move(publishers), {}, true));

  auto closed = (*session)->close_owner_outputs(VertexId{1U});
  ASSERT_FALSE(closed);
  EXPECT_EQ(closed.error().code, ErrorCode::kPublicationIndeterminate);
  EXPECT_EQ(first_observed->close_count, 1U);
  EXPECT_EQ(second_observed->close_count, 0U);

  const std::array affected{VertexId{1U}, VertexId{2U}};
  auto full_reset = (*session)->reset();
  ASSERT_FALSE(full_reset);
  EXPECT_EQ(full_reset.error().code, ErrorCode::kClosed);
  auto scoped_reset = (*session)->reset(VertexId{1U});
  ASSERT_FALSE(scoped_reset);
  EXPECT_EQ(scoped_reset.error().code, ErrorCode::kClosed);
  auto restart = (*session)->commit_restart(VertexId{1U}, affected);
  ASSERT_FALSE(restart);
  EXPECT_EQ(restart.error().code, ErrorCode::kClosed);

  ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
  EXPECT_EQ(first_observed->close_count, 1U);
  EXPECT_EQ(second_observed->close_count, 1U);
  ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
  EXPECT_EQ(first_observed->close_count, 1U);
  EXPECT_EQ(second_observed->close_count, 1U);
}

// A throwable backend boundary is classified as indeterminate instead of escaping this noexcept
// owner-close transaction. The affected arm and process-local accounting are consumed exactly
// once, while a retry may close only the unresolved suffix.
// REQ: CHAN-136
TEST(RuntimeChannelBinding, ThrowingCloseResolvesOnlyThatArmAndInvalidatesLocalAccounting) {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  SessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto first = std::make_unique<ScriptedInlinePublisher>(
      std::vector<channel_internal::InlinePublicationResult>{},
      std::vector<ErrorCode>{},
      std::vector<ErrorCode>{},
      true);
  auto second = std::make_unique<ScriptedInlinePublisher>(
      std::vector<channel_internal::InlinePublicationResult>{},
      std::vector<ErrorCode>{},
      std::vector{ErrorCode::kSuccess});
  auto* const first_observed = first.get();
  auto* const second_observed = second.get();
  std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
  publishers.push_back(std::move(first));
  publishers.push_back(std::move(second));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_output_publishers(
      **session, PortId{1U}, std::move(publishers)));

  auto closed = (*session)->close_owner_outputs(VertexId{1U});
  ASSERT_FALSE(closed);
  EXPECT_EQ(closed.error().code, ErrorCode::kPublicationIndeterminate);
  EXPECT_EQ(first_observed->close_count, 1U);
  EXPECT_EQ(second_observed->close_count, 0U);
  const auto progress_state =
      RuntimeChannelSessionTestAccess::process_local_progress_state(**session);
  ASSERT_TRUE(progress_state) << runtime_error_message(progress_state.error());
  EXPECT_FALSE(progress_state->tracking_valid);

  ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
  EXPECT_EQ(first_observed->close_count, 1U);
  EXPECT_EQ(second_observed->close_count, 1U);
  ASSERT_TRUE((*session)->close_owner_outputs(VertexId{1U}));
  EXPECT_EQ(first_observed->close_count, 1U);
  EXPECT_EQ(second_observed->close_count, 1U);
#else
  GTEST_SKIP() << "exception containment requires language exception support";
#endif
}

// Catalog and inline publishers share one refresh-failure classifier. A failure before any arm
// submits preserves the current sequence, while a failure after an earlier arm submits consumes
// that authority and reports an indeterminate fanout.
// REQ: SYSTEM-003
TEST(RuntimeChannelBinding, RefreshFailureClassifiesPublicationAuthority) {
  const auto accepted = [] {
    return channel_internal::InlinePublicationResult{
        .phase = channel_internal::InlinePublicationPhase::kAccepted,
        .recipient_count = 1U,
        .error = Error{ErrorCode::kSuccess},
    };
  };

  {
    SessionFixture fixture;
    auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
    ASSERT_TRUE(session) << runtime_error_message(session.error());
    auto publisher = std::make_unique<ScriptedInlinePublisher>(
        std::vector{accepted()}, std::vector{ErrorCode::kBackpressured, ErrorCode::kSuccess});
    auto* const observed = publisher.get();
    std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
    publishers.push_back(std::move(publisher));
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_output_publishers(
        **session, PortId{1U}, std::move(publishers), {}, true));
    ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));

    auto rejected = fixture.source.output.emit(std::int32_t{50});
    ASSERT_FALSE(rejected);
    EXPECT_EQ(rejected.error().code, ErrorCode::kBackpressured);
    ASSERT_TRUE(fixture.source.output.emit(std::int32_t{51}));
    ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
    EXPECT_EQ(observed->sequences, (std::vector<std::uint64_t>{1U}));
    EXPECT_EQ(observed->values, (std::vector<std::int32_t>{51}));
  }

  {
    SessionFixture fixture;
    auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
    ASSERT_TRUE(session) << runtime_error_message(session.error());
    auto first = std::make_unique<ScriptedInlinePublisher>(std::vector{accepted(), accepted()});
    auto second = std::make_unique<ScriptedInlinePublisher>(
        std::vector{accepted()}, std::vector{ErrorCode::kBackpressured, ErrorCode::kSuccess});
    auto* const first_observed = first.get();
    auto* const second_observed = second.get();
    std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
    publishers.push_back(std::move(first));
    publishers.push_back(std::move(second));
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_output_publishers(
        **session, PortId{1U}, std::move(publishers), {}, true));
    ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));

    auto uncertain = fixture.source.output.emit(std::int32_t{60});
    ASSERT_FALSE(uncertain);
    EXPECT_EQ(uncertain.error().code, ErrorCode::kPublicationIndeterminate);
    ASSERT_TRUE(fixture.source.output.emit(std::int32_t{61}));
    ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
    EXPECT_EQ(first_observed->sequences, (std::vector<std::uint64_t>{1U, 2U}));
    EXPECT_EQ(second_observed->sequences, (std::vector<std::uint64_t>{2U}));
  }
}

// A malformed backend phase after another fanout arm submitted is indeterminate, but the
// accepted publication still consumes the sample sequence just like every other post-submit path.
// REQ: RUNTIME-019
TEST(RuntimeChannelBinding, MalformedPostSubmitPhaseConsumesPublicationSequence) {
  SessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  const auto accepted = channel_internal::InlinePublicationResult{
      .phase = channel_internal::InlinePublicationPhase::kAccepted,
      .recipient_count = 1U,
      .error = Error{ErrorCode::kSuccess},
  };
  auto first = std::make_unique<ScriptedInlinePublisher>(std::vector{accepted, accepted});
  auto second = std::make_unique<ScriptedInlinePublisher>(
      std::vector{channel_internal::InlinePublicationResult{
                      // The out-of-range phase exercises malformed backend handling.
                      // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
                      .phase = static_cast<channel_internal::InlinePublicationPhase>(0xffU),
                      .recipient_count = 0U,
                      .error = Error{ErrorCode::kSuccess},
                  },
                  accepted});
  auto* const first_observed = first.get();
  auto* const second_observed = second.get();
  std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
  publishers.push_back(std::move(first));
  publishers.push_back(std::move(second));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_output_publishers(
      **session, PortId{1U}, std::move(publishers)));
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));

  // Key setup: the first arm submits before the second arm returns an invalid private phase.
  auto uncertain = fixture.source.output.emit(std::int32_t{70});
  ASSERT_FALSE(uncertain);
  EXPECT_EQ(uncertain.error().code, ErrorCode::kPublicationIndeterminate);
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{71}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));

  // Key assertion: the retry uses sequence two on every fanout arm.
  EXPECT_EQ(first_observed->sequences, (std::vector<std::uint64_t>{1U, 2U}));
  EXPECT_EQ(second_observed->sequences, (std::vector<std::uint64_t>{1U, 2U}));
}

// An accepted backend phase is still post-submit indeterminate unless its exact delivered count
// matches the compiler-frozen fanout. Both zero delivery and partial delivery consume the output
// sequence because Runtime cannot prove that no recipient observed the attempted publication.
// REQ: CHAN-067
// REQ: CHAN-108
// REQ: CHAN-076
// REQ: RUNTIME-019
// REQ: RUNTIME-037
TEST(RuntimeChannelBinding, RejectsAcceptedPublicationWithIncompleteRecipientCount) {
  for (const std::uint32_t delivered : {0U, 1U}) {
    SCOPED_TRACE(delivered);
    SessionFixture fixture;
    auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
    ASSERT_TRUE(session) << runtime_error_message(session.error());
    auto publisher = std::make_unique<ScriptedInlinePublisher>(
        std::vector<channel_internal::InlinePublicationResult>{
            channel_internal::InlinePublicationResult{
                .phase = channel_internal::InlinePublicationPhase::kAccepted,
                .recipient_count = delivered,
                .error = Error{ErrorCode::kSuccess},
            },
            channel_internal::InlinePublicationResult{
                .phase = channel_internal::InlinePublicationPhase::kAccepted,
                .recipient_count = 2U,
                .error = Error{ErrorCode::kSuccess},
            },
        });
    auto* const observed = publisher.get();
    std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
    publishers.push_back(std::move(publisher));
    const std::array<std::uint32_t, 1U> expected_recipients{2U};
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_output_publishers(
        **session, PortId{1U}, std::move(publishers), expected_recipients));
    ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));

    auto incomplete = fixture.source.output.emit(std::int32_t{40});
    ASSERT_FALSE(incomplete);
    // Key assertion: the accepted strict subset is quantified and still consumes sequence
    // authority, so the next call advances rather than retrying sequence one.
    EXPECT_EQ(incomplete.error().code, ErrorCode::kPublicationPartial);
    EXPECT_EQ(incomplete.error().message,
              fmt::format("PUBLICATION_PARTIAL delivered={} expected=2", delivered));
    ASSERT_TRUE(fixture.source.output.emit(std::int32_t{41}));
    ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));

    EXPECT_EQ(observed->sequences, (std::vector<std::uint64_t>{1U, 2U}));
    EXPECT_EQ(observed->values, (std::vector<std::int32_t>{40, 41}));
  }
}

// Once one fanout endpoint accepts a publication, a later endpoint's rejection is indeterminate
// to the caller. Runtime consumes sequence and loan authority exactly once and never retries.
// REQ: CHAN-076
// REQ: CHAN-101
// REQ: RUNTIME-019
TEST(RuntimeChannelBinding, ConsumesGenericLoanAfterIndeterminateFanoutPublication) {
  SessionFixture fixture;
  fixture.plan.generic_output_loans.front().max_live_loans = 1U;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto first_publisher = std::make_unique<ScriptedInlinePublisher>(
      std::vector<channel_internal::InlinePublicationResult>{
          channel_internal::InlinePublicationResult{
              .phase = channel_internal::InlinePublicationPhase::kAccepted,
              .recipient_count = 1U,
              .error = Error{ErrorCode::kSuccess},
          },
          channel_internal::InlinePublicationResult{
              .phase = channel_internal::InlinePublicationPhase::kAccepted,
              .recipient_count = 1U,
              .error = Error{ErrorCode::kSuccess},
          },
      });
  auto second_publisher = std::make_unique<ScriptedInlinePublisher>(
      std::vector<channel_internal::InlinePublicationResult>{
          channel_internal::InlinePublicationResult{
              .phase = channel_internal::InlinePublicationPhase::kPreSubmitRejected,
              .recipient_count = 0U,
              .error = Error{ErrorCode::kBackpressured},
          },
          channel_internal::InlinePublicationResult{
              .phase = channel_internal::InlinePublicationPhase::kAccepted,
              .recipient_count = 1U,
              .error = Error{ErrorCode::kSuccess},
          },
      });
  auto* first_observed = first_publisher.get();
  auto* second_observed = second_publisher.get();
  std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
  publishers.push_back(std::move(first_publisher));
  publishers.push_back(std::move(second_publisher));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_output_publishers(
      **session, PortId{1U}, std::move(publishers)));
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));

  auto loan = fixture.source.output.allocate();
  ASSERT_TRUE(loan) << runtime_error_message(loan.error());
  loan->value() = 32;
  auto uncertain = fixture.source.output.emit(std::move(*loan));
  ASSERT_FALSE(uncertain);
  EXPECT_EQ(uncertain.error().code, ErrorCode::kPublicationIndeterminate);
  auto replacement = fixture.source.output.allocate();
  ASSERT_TRUE(replacement) << runtime_error_message(replacement.error());
  replacement->cancel();
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{33}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  EXPECT_EQ(first_observed->sequences, (std::vector<std::uint64_t>{1U, 2U}));
  EXPECT_EQ(first_observed->values, (std::vector<std::int32_t>{32, 33}));
  EXPECT_EQ(second_observed->sequences, (std::vector<std::uint64_t>{1U, 2U}));
  EXPECT_EQ(second_observed->values, (std::vector<std::int32_t>{32, 33}));
}

// REQ: RUNTIME-019
TEST(RuntimeChannelBinding, EnforcesUnpublishedLoanCreditIndependentlyOfSlotCount) {
  SessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));

  auto first = fixture.source.output.allocate();
  auto second = fixture.source.output.allocate();
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  auto exhausted = fixture.source.output.allocate();
  ASSERT_FALSE(exhausted);
  EXPECT_EQ(exhausted.error().code, ErrorCode::kResourceExhausted);

  first->cancel();
  auto replacement = fixture.source.output.allocate();
  ASSERT_TRUE(replacement);
  replacement->cancel();
  second->cancel();
  EXPECT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
}

// Reports scripted HoloMQ send results through the Channel layer's publication mapping, as the
// inline publisher does, so a test can script cancelled, expired, or faulted sends. The last
// result repeats.
class HoloMQSendResultPublisher final : public channel_internal::InlinePayloadPublisher {
 public:
  explicit HoloMQSendResultPublisher(std::vector<holomq::ErrorCode> codes)
      : codes_(std::move(codes)) {}

  [[nodiscard]] channel_internal::InlinePublicationResult emit_phase(
      std::span<const std::byte>, const SampleMetadata&) override {
    const holomq::ErrorCode code = codes_[std::min(next_, codes_.size() - 1U)];
    ++next_;
    auto published = channel_internal::from_holomq_publication_result(
        holomq::make_unexpected(holomq::Error{code}), 1U, true);
    if (published) {
      return {.phase = channel_internal::InlinePublicationPhase::kAccepted,
              .recipient_count = *published};
    }
    const bool pre_submit = published.error().code == ErrorCode::kBackpressured;
    return {.phase = pre_submit ? channel_internal::InlinePublicationPhase::kPreSubmitRejected
                                : channel_internal::InlinePublicationPhase::kPostSubmitUncertain,
            .error = std::move(published).error()};
  }
  [[nodiscard]] expected<void, Error> update_connections() noexcept override { return {}; }
  [[nodiscard]] bool has_subscribers() const noexcept override { return true; }

 private:
  std::vector<holomq::ErrorCode> codes_;
  std::size_t next_{};
};

// Only the stop's cancellation of a publication's own wait attributes its failure to the stop.
// With the stop accepted, a send that HoloMQ reports cancelled is attributed, alone or after
// another fanout leg accepted. A send whose budget expired, so it failed before the stop
// cancelled anything, and a fanout leg that failed for a reason other than capacity are not,
// although the stop is accepted when each failure is classified. The arms run in that order on
// one thread, so a cancelled arm's evidence cannot leak into a later arm.
// REQ: RUNTIME-076
TEST(RuntimeChannelBinding, StopAttributesOnlyPublicationsItCancelled) {
  struct Arm {
    std::string_view name;
    bool fanout;
    holomq::ErrorCode send;
    ErrorCode returned;
    bool attributed;
  };
  constexpr std::array arms{
      Arm{"cancelled", false, holomq::ErrorCode::kInterrupted, ErrorCode::kBackpressured, true},
      Arm{"cancelled again",
          false,
          holomq::ErrorCode::kInterrupted,
          ErrorCode::kBackpressured,
          true},
      Arm{"expired", false, holomq::ErrorCode::kBackpressured, ErrorCode::kBackpressured, false},
      Arm{"fanout cancelled",
          true,
          holomq::ErrorCode::kInterrupted,
          ErrorCode::kPublicationIndeterminate,
          true},
      Arm{"fanout faulted",
          true,
          holomq::ErrorCode::kStaleEndpoint,
          ErrorCode::kPublicationIndeterminate,
          false},
  };
  std::optional<Error> prior_cancelled;
  for (const Arm& arm : arms) {
    SCOPED_TRACE(arm.name);
    SessionFixture fixture;
    const std::atomic<bool> stop{true};
    auto session = RuntimeChannelSession::create(
        fixture.plan, fixture.replay, {}, RuntimeChannelReadinessWake{.publication_stop = &stop});
    ASSERT_TRUE(session) << runtime_error_message(session.error());
    std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
    std::vector<std::uint32_t> recipients{1U};
    if (arm.fanout) {
      publishers.push_back(std::make_unique<ScriptedInlinePublisher>(
          std::vector<channel_internal::InlinePublicationResult>{
              channel_internal::InlinePublicationResult{
                  .phase = channel_internal::InlinePublicationPhase::kAccepted,
                  .recipient_count = 1U,
                  .error = Error{ErrorCode::kSuccess},
              },
          }));
      recipients.push_back(1U);
    }
    publishers.push_back(
        std::make_unique<HoloMQSendResultPublisher>(std::vector<holomq::ErrorCode>{arm.send}));
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_output_publishers(
        **session, PortId{1U}, std::move(publishers), recipients));
    ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));

    auto emitted = fixture.source.output.emit(std::int32_t{7});
    ASSERT_FALSE(emitted);
    EXPECT_EQ(emitted.error().code, arm.returned);
    // Key assertion: the stop owns the failure exactly when it cancelled the failing send.
    EXPECT_EQ((*session)->publication_cancelled_by_stop(VertexId{1U}, 1U, emitted.error()),
              arm.attributed);
    EXPECT_EQ(emitted.error().message.starts_with("RUNTIME_PUBLICATION_STOPPED"), arm.attributed);
    if (arm.attributed) {
      if (prior_cancelled) {
        EXPECT_FALSE((*session)->publication_cancelled_by_stop(VertexId{1U}, 1U, *prior_cancelled));
      }
      prior_cancelled = emitted.error();
    }
    static_cast<void>((*session)->retire_activation(VertexId{1U}, 1U, false));
    (*session)->cancel();
    static_cast<void>((*session)->join());
  }
}

// A stop-cancelled publication's error identifies it: only that error, returned unchanged, is the
// stop's. In one activation, a send that the stop cancelled is followed on the same output by a
// send that fails for another reason: an expired budget after a pre-submit cancellation, which
// leaves retry authority, or a fault after another fanout leg accepted. The later failure, a
// synthetic pressure error, and the cancelled error with another code or message are not
// attributed, while the cancelled error stays attributed whatever followed it.
// REQ: RUNTIME-076
TEST(RuntimeChannelBinding, StopAttributesOnlyTheCancelledPublicationsOwnError) {
  struct Arm {
    std::string_view name;
    bool fanout;
    holomq::ErrorCode later_send;
    ErrorCode returned;
  };
  constexpr std::array arms{
      Arm{"expired after cancelled",
          false,
          holomq::ErrorCode::kBackpressured,
          ErrorCode::kBackpressured},
      Arm{"fanout faulted after cancelled",
          true,
          holomq::ErrorCode::kStaleEndpoint,
          ErrorCode::kPublicationIndeterminate},
  };
  for (const Arm& arm : arms) {
    SCOPED_TRACE(arm.name);
    SessionFixture fixture;
    const std::atomic<bool> stop{true};
    auto session = RuntimeChannelSession::create(
        fixture.plan, fixture.replay, {}, RuntimeChannelReadinessWake{.publication_stop = &stop});
    ASSERT_TRUE(session) << runtime_error_message(session.error());
    std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
    std::vector<std::uint32_t> recipients{1U};
    if (arm.fanout) {
      const channel_internal::InlinePublicationResult accepted{
          .phase = channel_internal::InlinePublicationPhase::kAccepted,
          .recipient_count = 1U,
          .error = Error{ErrorCode::kSuccess},
      };
      publishers.push_back(std::make_unique<ScriptedInlinePublisher>(
          std::vector<channel_internal::InlinePublicationResult>{accepted, accepted}));
      recipients.push_back(1U);
    }
    publishers.push_back(std::make_unique<HoloMQSendResultPublisher>(
        std::vector<holomq::ErrorCode>{holomq::ErrorCode::kInterrupted, arm.later_send}));
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_output_publishers(
        **session, PortId{1U}, std::move(publishers), recipients));
    ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));

    auto cancelled = fixture.source.output.emit(std::int32_t{7});
    ASSERT_FALSE(cancelled);
    EXPECT_EQ(cancelled.error().code, arm.returned);
    EXPECT_TRUE(cancelled.error().message.starts_with("RUNTIME_PUBLICATION_STOPPED"));
    auto later = fixture.source.output.emit(std::int32_t{8});
    ASSERT_FALSE(later);
    EXPECT_EQ(later.error().code, arm.returned);
    EXPECT_FALSE(later.error().message.starts_with("RUNTIME_PUBLICATION_STOPPED"));

    const auto attributed = [&](const Error& error) {
      return (*session)->publication_cancelled_by_stop(VertexId{1U}, 1U, error);
    };
    // Key assertion: a later failure in the same activation is not attributed to the stop.
    EXPECT_FALSE(attributed(later.error()));
    EXPECT_TRUE(attributed(cancelled.error()));
    const Error copied = cancelled.error();
    EXPECT_TRUE(attributed(copied));
    EXPECT_FALSE(attributed(Error{cancelled.error().code, cancelled.error().message}));
    EXPECT_FALSE(attributed(Error{arm.returned}));
    EXPECT_FALSE(attributed(Error{arm.returned, cancelled.error().message + " (handled)"}));
    EXPECT_FALSE(attributed(Error{ErrorCode::kPublicationPartial, cancelled.error().message}));
    EXPECT_FALSE((*session)->publication_cancelled_by_stop(VertexId{2U}, 1U, cancelled.error()));
    EXPECT_FALSE((*session)->publication_cancelled_by_stop(VertexId{1U}, 2U, cancelled.error()));
    static_cast<void>((*session)->retire_activation(VertexId{1U}, 1U, false));
    (*session)->cancel();
    static_cast<void>((*session)->join());
  }
}

}  // namespace
}  // namespace holoscan::detail

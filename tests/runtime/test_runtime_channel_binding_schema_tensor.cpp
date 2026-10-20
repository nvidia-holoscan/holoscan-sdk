/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <limits>
#include <memory>
#include <optional>
#include <semaphore>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/schema/sdk_schemas.hpp>
#include <holoscan/time/manual_clock.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "../../src/channel/channel_common.hpp"
#include "../../src/channel/channel_payload_frame.hpp"
#include "../../src/channel/channel_tensor_test_access.hpp"
#include "../../src/channel/endpoint_catalog_state.hpp"
#include "../common/cuda_test_utils.hpp"
#include "compiler/execution_plan_state.hpp"
#include "core/graph_compiler_access.hpp"
#include "core/source_id_internal.hpp"
#include "runtime/channel_runtime_binding_access.hpp"
#include "runtime/dispatcher.hpp"
#include "runtime/graph_exec.hpp"
#include "runtime/runtime_channel_binding.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "runtime/runtime_test_access.hpp"

#include "runtime_channel_binding_schema_tensor_test_support.hpp"
#include "runtime_channel_binding_test_support.hpp"

namespace holoscan::detail {
namespace {

using namespace std::chrono_literals;

using namespace runtime_channel_binding_test;

using namespace runtime_channel_binding_schema_tensor_test;

// Direct-retained descriptor Tensor delivery can arm causal continuation while preserving the
// ordinary readiness record, schema metadata, and retained Tensor lifetime authority.
// REQ: CHAN-099
// REQ: CHAN-128
// REQ: RUNTIME-019
// REQ: RUNTIME-037
TEST(ChannelBindingParityTest,
     DirectDescriptorTensorArmsContinuationAndPreservesOrdinaryReadiness) {
  // Key setup: create an otherwise eligible exact shared-lane direct descriptor Tensor edge.
  ImageSessionFixture fixture;
  fixture.use_direct_retained_delivery();
  auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto binding =
      (*session)->create_readiness_binding(VertexId{2U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.base.plan.plan_generation,
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
  EXPECT_TRUE((*session)->arm_direct_continuation(VertexId{1U}, 0U));

  auto published = publish_image_sample(**session, **binding, fixture.source, 91U);
  ASSERT_TRUE(published) << runtime_error_message(published.error());
  const std::array selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = InputSelection{InputState::kSelected, SelectionReason::kNone},
      .samples = std::span<const TemporalSampleRecord>{&published->record, 1U},
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, selection));
  ASSERT_TRUE((*session)->prepare_tensor_access(VertexId{2U}, 1U, nullptr));

  // Key assertion: continuation eligibility does not consume ordinary readiness; schema metadata
  // and embedded Tensor backing remain valid through typed receive and retirement.
  {
    auto received = fixture.sink.input.receive();
    ASSERT_TRUE(received) << runtime_error_message(received.error());
    ASSERT_NE(received->data.data, nullptr);
    ASSERT_NE(received->data.data->data_as<std::uint64_t>(), nullptr);
    EXPECT_EQ(*received->data.data->data_as<std::uint64_t>(), 91U);
    ASSERT_NE(received->data.header, nullptr);
    EXPECT_EQ(received->data.header->frame_id, "runtime_image");
    EXPECT_EQ(published->releases->load(std::memory_order_relaxed), 0U);
  }
  ASSERT_TRUE((*binding)->retire(published->event.source_sequence));
  ASSERT_TRUE((*session)->retire_samples(PortId{2U}, selection.front().samples));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  EXPECT_EQ(published->releases->load(std::memory_order_relaxed), 1U);
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// Raw Tensor publication uses the same graph-owned finite-pressure policy as scalar payloads, but
// additionally transfers one backing claim into the transport release callback. A rejected
// publication must retire only its own claim and must not consume a logical sequence.
// REQ: CHAN-083
// REQ: CHAN-101
// REQ: CHAN-102
// REQ: RUNTIME-019
// REQ: TENSOR-STORAGE-010
TEST(ChannelBindingParityTest, TensorBlockingPressureRetiresRejectedBackingAndPreservesRetry) {
  TensorSessionFixture fixture;
  fixture.use_one_slot_blocking_transport();
  auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto binding =
      (*session)->create_readiness_binding(VertexId{2U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.base.plan.plan_generation,
                                               .run_generation = 1U,
                                               .kind = RuntimeEventKind::kInputArrival,
                                               .source = make_source_id(1U),
                                               .source_clock = ClockId{1U},
                                               .reference_clock = ClockId{1U},
                                               .capacity = 2U,
                                           });
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));

  auto first = make_owned_tensor_sample(11U);
  auto rejected = make_owned_tensor_sample(22U);
  ASSERT_TRUE(first) << runtime_error_message(first.error());
  ASSERT_TRUE(rejected) << runtime_error_message(rejected.error());
  auto first_releases = first->releases;
  auto rejected_releases = rejected->releases;
  ASSERT_TRUE(fixture.source.output.emit(std::move(first->tensor)));

  std::promise<void> entered_emit;
  auto entered = entered_emit.get_future();
  auto pressure = std::async(
      std::launch::async,
      [&fixture, tensor = std::move(rejected->tensor), signal = std::move(entered_emit)]() mutable {
        signal.set_value();
        return fixture.source.output.emit(std::move(tensor));
      });
  auto pressure_cleanup = ScopeExit{[&]() noexcept {
    if (!pressure.valid()) {
      return;
    }
    if (pressure.wait_for(0s) != std::future_status::ready) {
      static_cast<void>((*session)->arm());
    }
    (*session)->cancel();
    static_cast<void>((*session)->join());
    pressure.wait();
  }};
  ASSERT_EQ(entered.wait_for(2s), std::future_status::ready);
  if (pressure.wait_for(1s) != std::future_status::ready) {
    ASSERT_TRUE((*session)->arm());
    ASSERT_EQ(pressure.wait_for(2s), std::future_status::ready);
    static_cast<void>(pressure.get());
    (*session)->cancel();
    ASSERT_TRUE((*session)->join());
    FAIL() << "raw Tensor publication waited for full recipient capacity";
  }
  const auto pressure_result = pressure.get();
  pressure_cleanup.release();
  ASSERT_FALSE(pressure_result);
  EXPECT_EQ(pressure_result.error().code, ErrorCode::kBackpressured);
  EXPECT_EQ(first_releases->load(std::memory_order_relaxed), 0U);
  EXPECT_EQ(rejected_releases->load(std::memory_order_relaxed), 1U);

  ASSERT_TRUE((*session)->arm());
  auto first_event = wait_for_event(**session, **binding);
  ASSERT_TRUE(first_event) << runtime_error_message(first_event.error());
  auto first_record = (*session)->temporal_sample(first_event->port, first_event->source_sequence);
  ASSERT_TRUE(first_record) << runtime_error_message(first_record.error());
  const std::array first_selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = std::span<const TemporalSampleRecord>{&*first_record, 1U},
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, first_selection));
  ASSERT_TRUE((*session)->prepare_tensor_access(VertexId{2U}, 1U, nullptr));
  {
    auto sample = fixture.sink.input.receive();
    ASSERT_TRUE(sample) << runtime_error_message(sample.error());
    ASSERT_NE(sample->data.data_as<std::uint64_t>(), nullptr);
    EXPECT_EQ(*sample->data.data_as<std::uint64_t>(), 11U);
    EXPECT_EQ(sample->metadata.sequence, 1U);
  }
  ASSERT_TRUE((*binding)->retire(first_event->source_sequence));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  ASSERT_TRUE((*session)->retire_samples(
      PortId{2U}, std::span<const TemporalSampleRecord>{&*first_record, 1U}));
  EXPECT_EQ(first_releases->load(std::memory_order_relaxed), 1U);

  auto retry = make_owned_tensor_sample(33U);
  ASSERT_TRUE(retry) << runtime_error_message(retry.error());
  auto retry_releases = retry->releases;
  ASSERT_TRUE(fixture.source.output.emit(std::move(retry->tensor)));
  auto retry_event = wait_for_event(**session, **binding);
  ASSERT_TRUE(retry_event) << runtime_error_message(retry_event.error());
  auto retry_record = (*session)->temporal_sample(retry_event->port, retry_event->source_sequence);
  ASSERT_TRUE(retry_record) << runtime_error_message(retry_record.error());
  EXPECT_EQ(retry_record->source_sequence, 2U);
  const std::array retry_selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = std::span<const TemporalSampleRecord>{&*retry_record, 1U},
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 2U, retry_selection));
  ASSERT_TRUE((*session)->prepare_tensor_access(VertexId{2U}, 2U, nullptr));
  {
    auto sample = fixture.sink.input.receive();
    ASSERT_TRUE(sample) << runtime_error_message(sample.error());
    ASSERT_NE(sample->data.data_as<std::uint64_t>(), nullptr);
    EXPECT_EQ(*sample->data.data_as<std::uint64_t>(), 33U);
    EXPECT_EQ(sample->metadata.sequence, 2U);
  }
  ASSERT_TRUE((*binding)->retire(retry_event->source_sequence));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 2U, true));
  ASSERT_TRUE((*session)->retire_samples(
      PortId{2U}, std::span<const TemporalSampleRecord>{&*retry_record, 1U}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
  EXPECT_EQ(retry_releases->load(std::memory_order_relaxed), 1U);
}

// Real HoloMQ fanout can accept one recipient before discovering that another recipient remains
// full. Runtime must classify that result as indeterminate, consume the logical sequence, and
// retain the shared Tensor backing until the accepting recipient retires its exact claim.
// REQ: CHAN-076
// REQ: CHAN-083
// REQ: CHAN-101
// REQ: RUNTIME-019
// REQ: TENSOR-STORAGE-010
TEST(ChannelBindingParityTest, TensorPartialFanoutPressureRetainsAcceptedRecipientBacking) {
  TensorSessionFixture fixture;
  fixture.use_one_slot_blocking_fanout();
  // HoloMQ's own blocking pressure is under test, so turn off the credit check that prevents it.
  fixture.base.disable_publication_credits();
  auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  const auto bind = [&](const VertexId owner, const SourceId source) {
    return (*session)->create_readiness_binding(
        owner,
        RuntimePortBindingRequest{
            .plan_generation = fixture.base.plan.plan_generation,
            .run_generation = 1U,
            .kind = RuntimeEventKind::kInputArrival,
            .source = source,
            .source_clock = ClockId{1U},
            .reference_clock = ClockId{1U},
            .capacity = 1U,
        });
  };
  auto first_binding = bind(VertexId{2U}, make_source_id(1U));
  auto second_binding = bind(VertexId{3U}, make_source_id(2U));
  ASSERT_TRUE(first_binding) << runtime_error_message(first_binding.error());
  ASSERT_TRUE(second_binding) << runtime_error_message(second_binding.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));

  const auto receive_first = [&](const RuntimeEvent& event,
                                 const std::uint64_t activation_generation,
                                 const std::uint64_t expected_value,
                                 const std::uint64_t expected_sequence) -> expected<void, Error> {
    auto record = (*session)->temporal_sample(event.port, event.source_sequence);
    if (!record) {
      return make_unexpected(std::move(record).error());
    }
    const std::array selection{RuntimeChannelActivationSelection{
        .port = PortId{2U},
        .selection = {InputState::kSelected, SelectionReason::kNone},
        .samples = std::span<const TemporalSampleRecord>{&*record, 1U},
        .destructive = true,
    }};
    if (auto prepared =
            (*session)->prepare_activation(VertexId{2U}, activation_generation, selection);
        !prepared) {
      return make_unexpected(std::move(prepared).error());
    }
    if (auto access =
            (*session)->prepare_tensor_access(VertexId{2U}, activation_generation, nullptr);
        !access) {
      return make_unexpected(std::move(access).error());
    }
    auto sample = fixture.sink.input.receive();
    if (!sample || sample->data.data_as<std::uint64_t>() == nullptr ||
        *sample->data.data_as<std::uint64_t>() != expected_value ||
        sample->metadata.sequence != expected_sequence) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    if (auto retired = (*first_binding)->retire(event.source_sequence); !retired) {
      return make_unexpected(std::move(retired).error());
    }
    if (auto retired = (*session)->retire_activation(VertexId{2U}, activation_generation, true);
        !retired) {
      return make_unexpected(std::move(retired).error());
    }
    return (*session)->retire_samples(PortId{2U},
                                      std::span<const TemporalSampleRecord>{&*record, 1U});
  };

  auto first = make_owned_tensor_sample(11U);
  ASSERT_TRUE(first) << runtime_error_message(first.error());
  auto first_releases = first->releases;
  ASSERT_TRUE(fixture.source.output.emit(std::move(first->tensor)));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U}));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::receive_one(**session, PortId{3U}));
  auto first_event = wait_for_event(**session, **first_binding);
  auto held_second_event = wait_for_event(**session, **second_binding);
  ASSERT_TRUE(first_event) << runtime_error_message(first_event.error());
  ASSERT_TRUE(held_second_event) << runtime_error_message(held_second_event.error());
  auto received_first = receive_first(*first_event, 1U, 11U, 1U);
  ASSERT_TRUE(received_first) << runtime_error_message(received_first.error());
  // The second recipient still owns the first publication in its typed retention slot.
  EXPECT_EQ(first_releases->load(std::memory_order_relaxed), 0U);

  auto queued_second = make_owned_tensor_sample(22U);
  ASSERT_TRUE(queued_second) << runtime_error_message(queued_second.error());
  auto second_releases = queued_second->releases;
  ASSERT_TRUE(fixture.source.output.emit(std::move(queued_second->tensor)));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U}));
  auto second_recipient_staged =
      RuntimeChannelSessionTestAccess::receive_one(**session, PortId{3U});
  ASSERT_FALSE(second_recipient_staged);
  EXPECT_EQ(second_recipient_staged.error().code, ErrorCode::kBackpressured);
  auto second_event = wait_for_event(**session, **first_binding);
  ASSERT_TRUE(second_event) << runtime_error_message(second_event.error());

  // A full typed slot no longer implies a full transport queue: the bounded endpoint transaction
  // above stages sequence two for the second recipient so terminal control remains observable. Fill
  // that recipient's transport queue with sequence three, then stage the same sequence for the
  // first recipient. Sequence four now crosses a deterministic partial-fanout visibility boundary:
  // first accepts it into an empty queue and second rejects it behind typed + staged + queued data.
  auto staged = make_owned_tensor_sample(33U);
  ASSERT_TRUE(staged) << runtime_error_message(staged.error());
  auto staged_releases = staged->releases;
  ASSERT_TRUE(fixture.source.output.emit(std::move(staged->tensor)));
  auto first_recipient_staged = RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U});
  ASSERT_FALSE(first_recipient_staged);
  EXPECT_EQ(first_recipient_staged.error().code, ErrorCode::kBackpressured);

  auto partial = make_owned_tensor_sample(44U);
  ASSERT_TRUE(partial) << runtime_error_message(partial.error());
  auto partial_releases = partial->releases;
  auto partially_published = fixture.source.output.emit(std::move(partial->tensor));
  ASSERT_FALSE(partially_published);
  EXPECT_EQ(partially_published.error().code, ErrorCode::kPublicationIndeterminate);
  EXPECT_EQ(partial_releases->load(std::memory_order_relaxed), 0U);

  auto received_second = receive_first(*second_event, 2U, 22U, 2U);
  ASSERT_TRUE(received_second) << runtime_error_message(received_second.error());
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U}));
  auto staged_event = wait_for_event(**session, **first_binding);
  ASSERT_TRUE(staged_event) << runtime_error_message(staged_event.error());
  auto received_staged = receive_first(*staged_event, 3U, 33U, 3U);
  ASSERT_TRUE(received_staged) << runtime_error_message(received_staged.error());
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::receive_one(**session, PortId{2U}));
  auto partial_event = wait_for_event(**session, **first_binding);
  ASSERT_TRUE(partial_event) << runtime_error_message(partial_event.error());
  auto received_partial = receive_first(*partial_event, 4U, 44U, 4U);
  ASSERT_TRUE(received_partial) << runtime_error_message(received_partial.error());
  // The accepting recipient has retired its exact claim, but the indeterminate publication may
  // conservatively retain unresolved peer authority until teardown. It must never release sooner.
  EXPECT_EQ(partial_releases->load(std::memory_order_relaxed), 0U);

  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
  first_binding->reset();
  second_binding->reset();
  session->reset();
  EXPECT_EQ(first_releases->load(std::memory_order_relaxed), 1U);
  EXPECT_EQ(second_releases->load(std::memory_order_relaxed), 1U);
  EXPECT_EQ(staged_releases->load(std::memory_order_relaxed), 1U);
  EXPECT_EQ(partial_releases->load(std::memory_order_relaxed), 1U);
}

// Generated values with an embedded Tensor use the same finite graph-owned pressure policy as raw
// Tensor payloads. A rejected schema envelope must retire only its own embedded backing, while a
// later retry remains the second logical publication and reconstructs the owning value normally.
// REQ: CHAN-084
// REQ: CHAN-119
// REQ: CHAN-101
// REQ: CHAN-102
// REQ: RUNTIME-019
// REQ: TENSOR-SCHEMA-005
// REQ: TENSOR-STORAGE-010
TEST(ChannelBindingParityTest,
     DescriptorTensorImageBlockingPressureRetiresRejectedBackingAndPreservesRetry) {
  ImageSessionFixture fixture;
  fixture.use_one_slot_blocking_transport();
  auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto binding =
      (*session)->create_readiness_binding(VertexId{2U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.base.plan.plan_generation,
                                               .run_generation = 1U,
                                               .kind = RuntimeEventKind::kInputArrival,
                                               .source = make_source_id(1U),
                                               .source_clock = ClockId{1U},
                                               .reference_clock = ClockId{1U},
                                               .capacity = 2U,
                                           });
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));

  const auto make_image = [](std::uint64_t value, std::string frame_id)
      -> expected<std::pair<schema::ImageT, std::shared_ptr<std::atomic<std::uint32_t>>>, Error> {
    auto owned = make_owned_tensor_sample(value);
    if (!owned) {
      return make_unexpected(std::move(owned).error());
    }
    schema::ImageT image{};
    image.height = 1;
    image.width = 1;
    image.encoding = schema::ImageEncoding_CUSTOM;
    image.header = std::make_shared<schema::HeaderT>();
    image.header->frame_id = std::move(frame_id);
    image.data = std::make_shared<Tensor>(std::move(owned->tensor));
    return std::pair{std::move(image), std::move(owned->releases)};
  };
  const auto emit_image = [&fixture](schema::ImageT image, std::int64_t capture_timestamp_ns) {
    return fixture.source.output.emit(Sample<schema::ImageT>{
        std::move(image),
        SampleMetadata{.capture_timestamp_ns = capture_timestamp_ns,
                       .source_clock_id = ClockId{1U}},
    });
  };

  auto first = make_image(11U, "first_image");
  auto rejected = make_image(22U, "rejected_image");
  ASSERT_TRUE(first) << runtime_error_message(first.error());
  ASSERT_TRUE(rejected) << runtime_error_message(rejected.error());
  auto first_releases = first->second;
  auto rejected_releases = rejected->second;
  ASSERT_TRUE(emit_image(std::move(first->first), 10));

  std::promise<void> entered_emit;
  auto entered = entered_emit.get_future();
  auto pressure = std::async(std::launch::async,
                             [image = std::move(rejected->first),
                              &emit_image,
                              signal = std::move(entered_emit)]() mutable {
                               signal.set_value();
                               return emit_image(std::move(image), 20);
                             });
  auto pressure_cleanup = ScopeExit{[&]() noexcept {
    if (!pressure.valid()) {
      return;
    }
    if (pressure.wait_for(0s) != std::future_status::ready) {
      static_cast<void>((*session)->arm());
    }
    (*session)->cancel();
    static_cast<void>((*session)->join());
    pressure.wait();
  }};
  ASSERT_EQ(entered.wait_for(2s), std::future_status::ready);
  if (pressure.wait_for(1s) != std::future_status::ready) {
    ASSERT_TRUE((*session)->arm());
    ASSERT_EQ(pressure.wait_for(2s), std::future_status::ready);
    static_cast<void>(pressure.get());
    (*session)->cancel();
    ASSERT_TRUE((*session)->join());
    FAIL() << "descriptor-Tensor publication waited for full recipient capacity";
  }
  const auto pressure_result = pressure.get();
  pressure_cleanup.release();
  ASSERT_FALSE(pressure_result);
  EXPECT_EQ(pressure_result.error().code, ErrorCode::kBackpressured);
  EXPECT_EQ(first_releases->load(std::memory_order_relaxed), 0U);
  EXPECT_EQ(rejected_releases->load(std::memory_order_relaxed), 1U);

  ASSERT_TRUE((*session)->arm());
  const auto receive_image = [&](std::uint64_t activation_generation,
                                 std::uint64_t expected_value,
                                 std::string_view expected_frame_id,
                                 std::uint64_t expected_sequence) {
    auto event = wait_for_event(**session, **binding);
    ASSERT_TRUE(event) << runtime_error_message(event.error());
    auto record = (*session)->temporal_sample(event->port, event->source_sequence);
    ASSERT_TRUE(record) << runtime_error_message(record.error());
    EXPECT_EQ(record->source_sequence, expected_sequence);
    const std::array selection{RuntimeChannelActivationSelection{
        .port = PortId{2U},
        .selection = {InputState::kSelected, SelectionReason::kNone},
        .samples = std::span<const TemporalSampleRecord>{&*record, 1U},
        .destructive = true,
    }};
    ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, activation_generation, selection));
    ASSERT_TRUE((*session)->prepare_tensor_access(VertexId{2U}, activation_generation, nullptr));
    {
      auto sample = fixture.sink.input.receive();
      ASSERT_TRUE(sample) << runtime_error_message(sample.error());
      ASSERT_NE(sample->data.data, nullptr);
      ASSERT_NE(sample->data.data->data_as<std::uint64_t>(), nullptr);
      EXPECT_EQ(*sample->data.data->data_as<std::uint64_t>(), expected_value);
      ASSERT_NE(sample->data.header, nullptr);
      EXPECT_EQ(sample->data.header->frame_id, expected_frame_id);
      EXPECT_EQ(sample->metadata.sequence, expected_sequence);
    }
    ASSERT_TRUE((*binding)->retire(event->source_sequence));
    ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, activation_generation, true));
    ASSERT_TRUE((*session)->retire_samples(PortId{2U},
                                           std::span<const TemporalSampleRecord>{&*record, 1U}));
  };
  receive_image(1U, 11U, "first_image", 1U);
  EXPECT_EQ(first_releases->load(std::memory_order_relaxed), 1U);

  auto retry = make_image(33U, "retry_image");
  ASSERT_TRUE(retry) << runtime_error_message(retry.error());
  auto retry_releases = retry->second;
  ASSERT_TRUE(emit_image(std::move(retry->first), 30));
  receive_image(2U, 33U, "retry_image", 2U);

  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
  EXPECT_EQ(retry_releases->load(std::memory_order_relaxed), 1U);
}

// A controller activation cut may drain a raw-Tensor descriptor on a thread other than the
// background pump. Cold initialization must consume the exact frozen ordinal, restore the
// controller's ambient device, cache a same-thread repeat, and reject an unavailable ordinal.
// The exec-child matrix separately proves that the initialized import and callback expose the
// correct bytes on device zero and on a nonzero device when the host provides one.
// REQ: CHAN-103
TEST(ChannelBindingParityTest,
     ControllerDescriptorImportInitializesExactCudaDeviceAndRestoresCaller) {
  test::skip_without_cuda_device();

  int device_count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
  ASSERT_GT(device_count, 0);
  int ambient_device = -1;
  ASSERT_EQ(cudaGetDevice(&ambient_device), cudaSuccess);
  const int frozen_device = device_count > 1 ? (ambient_device == 0 ? 1 : 0) : ambient_device;

  TensorSessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::initialize_controller_descriptor_import(
      **session, PortId{2U}, DeviceId{frozen_device}));
  auto initialization_count =
      RuntimeChannelSessionTestAccess::controller_descriptor_import_initialization_count(**session);
  ASSERT_TRUE(initialization_count);
  EXPECT_EQ(*initialization_count, 1U);
  int restored_device = -1;
  ASSERT_EQ(cudaGetDevice(&restored_device), cudaSuccess);
  EXPECT_EQ(restored_device, ambient_device);

  // A repeated cut on the same sequential controller thread uses the cold-bound result and must
  // leave the caller unchanged.
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::initialize_controller_descriptor_import(
      **session, PortId{2U}, DeviceId{frozen_device}));
  initialization_count =
      RuntimeChannelSessionTestAccess::controller_descriptor_import_initialization_count(**session);
  ASSERT_TRUE(initialization_count);
  EXPECT_EQ(*initialization_count, 1U);
  ASSERT_EQ(cudaGetDevice(&restored_device), cudaSuccess);
  EXPECT_EQ(restored_device, ambient_device);

  if (device_count > 1) {
    const int replacement_device = frozen_device == 0 ? 1 : 0;
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::initialize_controller_descriptor_import(
        **session, PortId{2U}, DeviceId{replacement_device}));
    initialization_count =
        RuntimeChannelSessionTestAccess::controller_descriptor_import_initialization_count(
            **session);
    ASSERT_TRUE(initialization_count);
    EXPECT_EQ(*initialization_count, 2U);
    ASSERT_EQ(cudaGetDevice(&restored_device), cudaSuccess);
    EXPECT_EQ(restored_device, ambient_device);
  }

  TensorSessionFixture invalid_fixture;
  auto invalid_session =
      RuntimeChannelSession::create(invalid_fixture.base.plan, invalid_fixture.replay);
  ASSERT_TRUE(invalid_session) << runtime_error_message(invalid_session.error());
  auto unavailable = RuntimeChannelSessionTestAccess::initialize_controller_descriptor_import(
      **invalid_session, PortId{2U}, DeviceId{device_count});
  ASSERT_FALSE(unavailable);
  EXPECT_EQ(unavailable.error().code, ErrorCode::kNotSupported);
  ASSERT_EQ(cudaGetDevice(&restored_device), cudaSuccess);
  EXPECT_EQ(restored_device, ambient_device);
}

// Activation cuts and owner polls on different lanes may import descriptors at the same time.
// Each thread must initialize a frozen import inventory exactly once: alternating threads must not
// repeat the initialization, and concurrent callers must not share mutable controller state.
// REQ: CHAN-103
TEST(ChannelBindingParityTest, ControllerDescriptorImportInitializesOncePerThread) {
  test::skip_without_cuda_device();

  int ambient_device = -1;
  ASSERT_EQ(cudaGetDevice(&ambient_device), cudaSuccess);
  TensorSessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::configure_descriptor_import_device(
      **session, PortId{2U}, DeviceId{ambient_device}));
  const auto initialize = [&]() {
    return RuntimeChannelSessionTestAccess::initialize_descriptor_import_on_this_thread(**session);
  };
  const auto count = [&]() {
    auto initialized =
        RuntimeChannelSessionTestAccess::controller_descriptor_import_initialization_count(
            **session);
    return initialized ? *initialized : std::numeric_limits<std::uint64_t>::max();
  };
  const auto initialize_on_new_thread = [&]() {
    expected<void, Error> result = make_unexpected(Error{ErrorCode::kNotReady});
    std::thread worker([&] { result = initialize(); });
    worker.join();
    return result;
  };

  // Key setup: this thread and a second thread each initialize, then this thread repeats.
  ASSERT_TRUE(initialize());
  EXPECT_EQ(count(), 1U);
  std::thread second([&] {
    EXPECT_TRUE(initialize());
    EXPECT_TRUE(initialize());
  });
  second.join();
  EXPECT_EQ(count(), 2U);
  // Key assertion: alternating back to an initialized thread does not initialize it again.
  ASSERT_TRUE(initialize());
  EXPECT_EQ(count(), 2U);

  // Key assertion: concurrent first use on two new threads initializes each thread once.
  constexpr int kRepeats = 1'000;
  std::atomic<bool> start{false};
  std::atomic<int> failures{0};
  const auto hammer = [&] {
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    for (int repeat = 0; repeat < kRepeats; ++repeat) {
      if (!initialize()) {
        failures.fetch_add(1, std::memory_order_relaxed);
      }
    }
  };
  std::thread third(hammer);
  std::thread fourth(hammer);
  start.store(true, std::memory_order_release);
  third.join();
  fourth.join();
  EXPECT_EQ(failures.load(std::memory_order_relaxed), 0);
  EXPECT_EQ(count(), 4U);
  ASSERT_TRUE(initialize_on_new_thread());
  EXPECT_EQ(count(), 5U);
  int restored_device = -1;
  ASSERT_EQ(cudaGetDevice(&restored_device), cudaSuccess);
  EXPECT_EQ(restored_device, ambient_device);
}

// A raw-Tensor scan can move from a device-pinned subscriber to a host/unspecified subscriber.
// The latter import belongs to the controller's ambient device, not the peer selected for the
// preceding endpoint, and the scan must leave that ambient selection intact for its caller.
// REQ: CHAN-103
TEST(ChannelBindingParityTest, HostDescriptorImportRestoresAmbientDeviceAfterPeerSelection) {
  test::skip_without_cuda_device();

  int device_count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
  if (device_count < 2) {
    GTEST_SKIP() << "peer-to-host descriptor selection requires two CUDA devices";
  }
  int ambient_device = -1;
  ASSERT_EQ(cudaGetDevice(&ambient_device), cudaSuccess);
  const int peer_device = ambient_device == 0 ? 1 : 0;

  auto observed = RuntimeChannelSessionTestAccess::cuda_device_after_import_selection(
      DeviceId{peer_device}, DeviceId{});
  ASSERT_TRUE(observed) << runtime_error_message(observed.error());
  EXPECT_EQ(*observed, DeviceId{ambient_device});

  int restored_device = -1;
  ASSERT_EQ(cudaGetDevice(&restored_device), cudaSuccess);
  EXPECT_EQ(restored_device, ambient_device);
}

// Event-wake-enabled raw Tensor and descriptor-Tensor services use typed Channel endpoints and
// join the same wait set as inline payloads instead of relying on the pump timeout.
// REQ: RUNTIME-032
TEST(RuntimeChannelBinding, PrebindsTypedTensorSubscribersIntoTheSharedWaitSet) {
  {
    TensorSessionFixture fixture;
    auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
    ASSERT_TRUE(session) << runtime_error_message(session.error());

    // Key assertion: one raw-Tensor endpoint contributes one wait handle and one drain entry.
    const auto indexes = RuntimeChannelSessionTestAccess::pump_index_state(**session);
    ASSERT_TRUE(indexes) << runtime_error_message(indexes.error());
    EXPECT_TRUE(indexes->exact);
    EXPECT_EQ(indexes->wait_handle_count, 1U);
    EXPECT_EQ(indexes->drain_subscriber_count, 1U);
  }
  {
    ImageSessionFixture fixture;
    auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
    ASSERT_TRUE(session) << runtime_error_message(session.error());

    // Key assertion: the descriptor-Tensor endpoint participates in the identical bounded index.
    const auto indexes = RuntimeChannelSessionTestAccess::pump_index_state(**session);
    ASSERT_TRUE(indexes) << runtime_error_message(indexes.error());
    EXPECT_TRUE(indexes->exact);
    EXPECT_EQ(indexes->wait_handle_count, 1U);
    EXPECT_EQ(indexes->drain_subscriber_count, 1U);
  }
}

// An exact same-partition Tensor edge needs no transport wake when the controller's serialized
// post-compute activation cut drains the typed endpoint before the next readiness freeze.
// REQ: CHAN-098
// REQ: CHAN-099
// REQ: RUNTIME-019
// REQ: RUNTIME-032
TEST(RuntimeChannelBinding, TensorActivationCutDrainsExactEventlessServices) {
  {
    // Key setup: publish a raw Tensor through an exact 1:1 HoloMQ service with no wait handle.
    TensorSessionFixture fixture;
    fixture.use_one_slot_blocking_transport();
    fixture.base.use_same_partition_eventless_transport();
    fixture.base.use_single_partition_runner();
    const auto counters = std::make_shared<RuntimePhaseCounters>();
    fixture.base.plan.executable_factories.phase_counters = counters;
    auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
    ASSERT_TRUE(session) << runtime_error_message(session.error());
    auto binding = (*session)->create_readiness_binding(
        VertexId{2U},
        RuntimePortBindingRequest{
            .plan_generation = fixture.base.plan.plan_generation,
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
    auto indexes = RuntimeChannelSessionTestAccess::pump_index_state(**session);
    ASSERT_TRUE(indexes) << runtime_error_message(indexes.error());
    EXPECT_TRUE(indexes->exact);
    EXPECT_EQ(indexes->wait_handle_count, 0U);
    EXPECT_EQ(indexes->drain_subscriber_count, 1U);
    EXPECT_FALSE(indexes->background_thread_required);
    EXPECT_FALSE(indexes->background_thread_running);
    const auto controller_initializations =
        RuntimeChannelSessionTestAccess::controller_descriptor_import_initialization_count(
            **session);
    ASSERT_TRUE(controller_initializations)
        << runtime_error_message(controller_initializations.error());
    EXPECT_EQ(*controller_initializations, 1U);

    ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
    auto owned = make_owned_tensor_sample(0xA11CEU);
    ASSERT_TRUE(owned) << runtime_error_message(owned.error());
    const void* const expected_data = owned->data;
    auto releases = owned->releases;
    ASSERT_TRUE(fixture.source.output.emit(std::move(owned->tensor)));
    ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
    ASSERT_TRUE((*session)->flush_process_local_activation_cut());

    // Key assertion: the ordinary cut publishes one typed Tensor readiness record and preserves
    // the backing identity and lifetime without a transport event attachment.
    auto event = wait_for_event(**session, **binding);
    ASSERT_TRUE(event) << runtime_error_message(event.error());
    auto record = (*session)->temporal_sample(event->port, event->source_sequence);
    ASSERT_TRUE(record) << runtime_error_message(record.error());
    const std::array selection{RuntimeChannelActivationSelection{
        .port = PortId{2U},
        .selection = {InputState::kSelected, SelectionReason::kNone},
        .samples = std::span<const TemporalSampleRecord>{&*record, 1U},
        .destructive = true,
    }};
    ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, selection));
    ASSERT_TRUE((*session)->prepare_tensor_access(VertexId{2U}, 1U, nullptr));
    {
      auto received = fixture.sink.input.receive();
      ASSERT_TRUE(received) << runtime_error_message(received.error());
      EXPECT_EQ(received->data.data(), expected_data);
      ASSERT_NE(received->data.data_as<std::uint64_t>(), nullptr);
      EXPECT_EQ(*received->data.data_as<std::uint64_t>(), 0xA11CEU);
      EXPECT_EQ(releases->load(std::memory_order_relaxed), 0U);
    }
    ASSERT_TRUE((*binding)->retire(event->source_sequence));
    ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
    ASSERT_TRUE((*session)->retire_samples(PortId{2U}, selection.front().samples));
    EXPECT_EQ(releases->load(std::memory_order_relaxed), 1U);
    const auto snapshot = counters->snapshot();
    EXPECT_EQ(snapshot[RuntimePhaseCounters::index(RuntimePhaseCounter::kChannelActivationCuts)],
              1U);
    EXPECT_EQ(snapshot[RuntimePhaseCounters::index(RuntimePhaseCounter::kChannelSamplesReceived)],
              1U);
    EXPECT_EQ(
        snapshot[RuntimePhaseCounters::index(RuntimePhaseCounter::kChannelReadinessPublished)], 1U);
    EXPECT_EQ(snapshot[RuntimePhaseCounters::index(
                  RuntimePhaseCounter::kChannelActivationCutSubscriberProbes)],
              1U);
    EXPECT_EQ(snapshot[RuntimePhaseCounters::index(
                  RuntimePhaseCounter::kChannelActivationCutTransportProbes)],
              0U);
    EXPECT_EQ(snapshot[RuntimePhaseCounters::index(
                  RuntimePhaseCounter::kChannelActivationCutMaterializations)],
              1U);
    (*session)->cancel();
    ASSERT_TRUE((*session)->join());
  }

  {
    // Key setup: publish Image metadata plus its Tensor sidecar through the same eventless shape.
    ImageSessionFixture fixture;
    fixture.use_one_slot_blocking_transport();
    fixture.base.use_same_partition_eventless_transport();
    fixture.base.use_single_partition_runner();
    const auto counters = std::make_shared<RuntimePhaseCounters>();
    fixture.base.plan.executable_factories.phase_counters = counters;
    auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
    ASSERT_TRUE(session) << runtime_error_message(session.error());
    auto binding = (*session)->create_readiness_binding(
        VertexId{2U},
        RuntimePortBindingRequest{
            .plan_generation = fixture.base.plan.plan_generation,
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
    auto indexes = RuntimeChannelSessionTestAccess::pump_index_state(**session);
    ASSERT_TRUE(indexes) << runtime_error_message(indexes.error());
    EXPECT_TRUE(indexes->exact);
    EXPECT_EQ(indexes->wait_handle_count, 0U);
    EXPECT_EQ(indexes->drain_subscriber_count, 1U);
    EXPECT_FALSE(indexes->background_thread_required);
    EXPECT_FALSE(indexes->background_thread_running);

    ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
    auto owned = make_owned_tensor_sample(0x1A6EU);
    ASSERT_TRUE(owned) << runtime_error_message(owned.error());
    auto releases = owned->releases;
    schema::ImageT image{};
    image.height = 1;
    image.width = 1;
    image.encoding = schema::ImageEncoding_CUSTOM;
    image.header = std::make_shared<schema::HeaderT>();
    image.header->frame_id = "eventless-image";
    image.data = std::make_shared<Tensor>(std::move(owned->tensor));
    ASSERT_TRUE(fixture.source.output.emit(Sample<schema::ImageT>{
        std::move(image),
        SampleMetadata{.capture_timestamp_ns = 10, .source_clock_id = ClockId{1U}},
    }));
    ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
    ASSERT_TRUE((*session)->flush_process_local_activation_cut());

    // Key assertion: the activation cut exposes one verified Image and keeps its Tensor sidecar
    // alive through typed access and sample retirement.
    auto event = wait_for_event(**session, **binding);
    ASSERT_TRUE(event) << runtime_error_message(event.error());
    auto record = (*session)->temporal_sample(event->port, event->source_sequence);
    ASSERT_TRUE(record) << runtime_error_message(record.error());
    const std::array selection{RuntimeChannelActivationSelection{
        .port = PortId{2U},
        .selection = {InputState::kSelected, SelectionReason::kNone},
        .samples = std::span<const TemporalSampleRecord>{&*record, 1U},
        .destructive = true,
    }};
    ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, selection));
    ASSERT_TRUE((*session)->prepare_tensor_access(VertexId{2U}, 1U, nullptr));
    {
      auto received = fixture.sink.input.receive();
      ASSERT_TRUE(received) << runtime_error_message(received.error());
      ASSERT_NE(received->data.header, nullptr);
      EXPECT_EQ(received->data.header->frame_id, "eventless-image");
      ASSERT_NE(received->data.data, nullptr);
      ASSERT_NE(received->data.data->data_as<std::uint64_t>(), nullptr);
      EXPECT_EQ(*received->data.data->data_as<std::uint64_t>(), 0x1A6EU);
      EXPECT_EQ(releases->load(std::memory_order_relaxed), 0U);
    }
    ASSERT_TRUE((*binding)->retire(event->source_sequence));
    ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
    ASSERT_TRUE((*session)->retire_samples(PortId{2U}, selection.front().samples));
    EXPECT_EQ(releases->load(std::memory_order_relaxed), 1U);
    const auto snapshot = counters->snapshot();
    EXPECT_EQ(snapshot[RuntimePhaseCounters::index(
                  RuntimePhaseCounter::kChannelActivationCutSubscriberProbes)],
              1U);
    EXPECT_EQ(snapshot[RuntimePhaseCounters::index(
                  RuntimePhaseCounter::kChannelActivationCutTransportProbes)],
              0U);
    EXPECT_EQ(snapshot[RuntimePhaseCounters::index(
                  RuntimePhaseCounter::kChannelActivationCutMaterializations)],
              1U);
    (*session)->cancel();
    ASSERT_TRUE((*session)->join());
  }
}

// A queued raw Tensor must wake the event-driven pump. The transport-wait trace is emitted only
// after a real wait-set readiness result, so it distinguishes this path from the timed fallback.
// REQ: RUNTIME-032
// REQ: RUNTIME-030
TEST(RuntimeChannelBinding, TensorPumpUsesEventWakeBeforeTheTimedFallback) {
  TensorSessionFixture fixture;
  std::array<RuntimePhaseTraceRecord, 16U> trace_storage;
  const auto counters = std::make_shared<RuntimePhaseCounters>(std::span{trace_storage});
  fixture.base.plan.executable_factories.phase_counters = counters;
  std::binary_semaphore wake{0};
  auto session = RuntimeChannelSession::create(
      fixture.base.plan,
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
                                               .plan_generation = fixture.base.plan.plan_generation,
                                               .run_generation = 1U,
                                               .kind = RuntimeEventKind::kInputArrival,
                                               .source = make_source_id(1U),
                                               .source_clock = ClockId{1U},
                                               .reference_clock = ClockId{1U},
                                               .capacity = 2U,
                                           });
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());
  ASSERT_TRUE((*session)->refresh_connections());

  // Key setup: publish before arm so the retained backend event is the pump's first observation.
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  auto owned = make_owned_tensor_sample(0xCAFE'BEEFU);
  ASSERT_TRUE(owned) << runtime_error_message(owned.error());
  ASSERT_TRUE(fixture.source.output.emit(std::move(owned->tensor)));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  EXPECT_FALSE(wake.try_acquire());
  ASSERT_TRUE((*session)->arm());
  ASSERT_TRUE(wake.try_acquire_for(1s));

  auto event = wait_for_event(**session, **binding);
  ASSERT_TRUE(event) << runtime_error_message(event.error());
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());

  // Key assertion: typed ingress followed an event-wait return, not the no-attachment sleep path.
  EXPECT_TRUE(std::any_of(counters->trace_records().begin(),
                          counters->trace_records().end(),
                          [](const RuntimePhaseTraceRecord& record) {
                            return record.point ==
                                   RuntimePhaseTracePoint::kChannelTransportWaitReturned;
                          }));
  constexpr std::array kTensorEmitPoints{
      RuntimePhaseTracePoint::kTensorEmitStarted,
      RuntimePhaseTracePoint::kTensorEmitValidated,
      RuntimePhaseTracePoint::kTensorTransportPublicationStarted,
      RuntimePhaseTracePoint::kTensorTransportPublicationReturned,
      RuntimePhaseTracePoint::kTensorEmitCompleted,
  };
  std::vector<RuntimePhaseTraceRecord> tensor_emit_trace;
  std::copy_if(counters->trace_records().begin(),
               counters->trace_records().end(),
               std::back_inserter(tensor_emit_trace),
               [&](const RuntimePhaseTraceRecord& record) {
                 return std::find(kTensorEmitPoints.begin(),
                                  kTensorEmitPoints.end(),
                                  record.point) != kTensorEmitPoints.end();
               });
  // Key assertion: the transport-backed Tensor emit records every ordered publication phase with
  // one stable output-port and sample identity for post-run attribution.
  ASSERT_EQ(tensor_emit_trace.size(), kTensorEmitPoints.size());
  for (std::size_t index = 0U; index < kTensorEmitPoints.size(); ++index) {
    EXPECT_EQ(tensor_emit_trace[index].point, kTensorEmitPoints[index]);
    EXPECT_EQ(tensor_emit_trace[index].output_port_id, 1U);
    EXPECT_EQ(tensor_emit_trace[index].source_sequence, 1U);
  }
  EXPECT_EQ(counters->trace_dropped(), 0U);
}

// The accepted cross-process Tensor plan constructs the same typed runtime binding but selects the
// BufferHandle transfer arm. A pass-through source owns no fabricated allocation pool.
// REQ: CHAN-085
// REQ: CHAN-088
TEST(ChannelBindingParityTest, CrossProcessTensorPassThroughConstructsBufferHandleEndpoints) {
  TensorSessionFixture fixture;
  fixture.use_cross_process_edge();

  auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto transfer =
      RuntimeChannelSessionTestAccess::tensor_transfer_state(**session, PortId{1U}, PortId{2U});
  ASSERT_TRUE(transfer) << runtime_error_message(transfer.error());
  EXPECT_EQ(transfer->buffer_handle_publishers, 1U);
  EXPECT_EQ(transfer->inline_publishers, 0U);
  EXPECT_TRUE(transfer->buffer_handle_subscriber);
  EXPECT_FALSE(transfer->inline_subscriber);
  (*session)->cancel();
  EXPECT_TRUE((*session)->join());
}

// An unpooled source cannot prove its concrete Tensor layout at compile time. The target's frozen
// expectation therefore remains a receive-time contract, checked before assigning arrival identity
// or retaining the sample. Rejected and matching representations use fresh fail-closed sessions.
// REQ: OPERATOR-028
// REQ: OPERATOR-041
// REQ: OPERATOR-046
TEST(ChannelBindingParityTest, PassThroughTensorEnforcesTargetLayoutBeforeCacheMutation) {
  const auto exercise = [](Tensor tensor,
                           const bool should_match,
                           const bool target_declares_layout,
                           const std::optional<TensorContractRecord>& target_contract =
                               std::nullopt) {
    TensorSessionFixture fixture;
    if (target_contract) {
      fixture.base.plan.ports[1].tensor_layout = target_contract->representation;
      fixture.base.plan.ports[1].tensor_contract = target_contract;
    } else if (target_declares_layout) {
      fixture.base.plan.ports[1].tensor_layout = TensorRepresentation{
          .memory_kind = MemoryKind::kHost,
          .dtype = DLDataType{kDLUInt, 64U, 1U},
          .rank = 1U,
      };
    }
    auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
    ASSERT_TRUE(session) << runtime_error_message(session.error());
    auto binding = (*session)->create_readiness_binding(
        VertexId{2U},
        RuntimePortBindingRequest{
            .plan_generation = fixture.base.plan.plan_generation,
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

    // Capture the mutation baseline before publication. A matching sample may be drained into the
    // cache as soon as emit returns, independently of when this test begins polling readiness.
    const auto before = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
    ASSERT_TRUE(before) << runtime_error_message(before.error());
    ASSERT_EQ(before->occupied_samples, 0U);

    ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
    ASSERT_TRUE(fixture.source.output.emit(Sample<Tensor>{std::move(tensor), {}}));
    ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));

    auto event = wait_for_event(**session, **binding);
    if (should_match) {
      ASSERT_TRUE(event) << runtime_error_message(event.error());
      auto record = (*session)->temporal_sample(event->port, event->source_sequence);
      ASSERT_TRUE(record) << runtime_error_message(record.error());
      EXPECT_EQ(record->source_sequence, 1U);
      EXPECT_EQ(record->arrival_sequence, 1U);
      ASSERT_TRUE((*binding)->retire(event->source_sequence));
      ASSERT_TRUE((*session)->retire_samples(PortId{2U},
                                             std::span<const TemporalSampleRecord>{&*record, 1U}));
    } else {
      ASSERT_FALSE(event);
      EXPECT_EQ(event.error().code, ErrorCode::kProtocolError);
      const auto after = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
      ASSERT_TRUE(after) << runtime_error_message(after.error());
      EXPECT_EQ(after->occupied_samples, before->occupied_samples);
      EXPECT_EQ(after->next_arrival_sequence, before->next_arrival_sequence);
      EXPECT_EQ(after->next_global_arrival_order, before->next_global_arrival_order);
    }
    (*session)->cancel();
    EXPECT_EQ(static_cast<bool>((*session)->join()), should_match);
  };

  const std::array<std::int64_t, 1U> rank_one{1};
  const std::array<std::int64_t, 2U> rank_two{1, 1};
  auto wrong_memory = make_stream_tracking_tensor();
  auto wrong_dtype = make_contract_test_tensor(rank_one, DLDataType{kDLUInt, 32U, 1U});
  auto wrong_rank = make_contract_test_tensor(rank_two, DLDataType{kDLUInt, 64U, 1U});
  auto unconstrained = make_contract_test_tensor(rank_two, DLDataType{kDLUInt, 32U, 1U});
  auto matching = make_owned_tensor_sample(42U);
  ASSERT_TRUE(wrong_dtype) << runtime_error_message(wrong_dtype.error());
  ASSERT_TRUE(wrong_rank) << runtime_error_message(wrong_rank.error());
  ASSERT_TRUE(unconstrained) << runtime_error_message(unconstrained.error());
  ASSERT_TRUE(matching) << runtime_error_message(matching.error());

  {
    SCOPED_TRACE("wrong memory kind");
    exercise(std::move(wrong_memory.tensor), false, true);
  }
  {
    SCOPED_TRACE("wrong dtype");
    exercise(std::move(*wrong_dtype), false, true);
  }
  {
    SCOPED_TRACE("wrong rank");
    exercise(std::move(*wrong_rank), false, true);
  }
  {
    SCOPED_TRACE("matching layout");
    exercise(std::move(matching->tensor), true, true);
  }
  {
    SCOPED_TRACE("undeclared target layout");
    exercise(std::move(*unconstrained), true, false);
  }

  const auto normalized_input = [](const TensorInputSpec& contract) {
    auto normalized = normalize_tensor_input_spec(contract);
    EXPECT_TRUE(normalized) << (normalized ? "" : normalized.error().message);
    return normalized ? std::optional<TensorContractRecord>{std::move(*normalized)} : std::nullopt;
  };
  {
    SCOPED_TRACE("maximum byte span");
    constexpr std::array<std::int64_t, 1U> shape{3};
    auto tensor = make_contract_test_tensor(shape, DLDataType{kDLUInt, 64U, 1U});
    ASSERT_TRUE(tensor) << runtime_error_message(tensor.error());
    const auto contract = normalized_input(TensorInputSpec{
        .representation = {.memory_kind = MemoryKind::kHost,
                           .dtype = DLDataType{kDLUInt, 64U, 1U},
                           .rank = 1U},
        .bounds = tensor_bounds(16U),
    });
    ASSERT_TRUE(contract);
    exercise(std::move(*tensor), false, true, contract);
  }
  {
    SCOPED_TRACE("maximum axis extent with in-range span");
    constexpr std::array<std::int64_t, 2U> maximum_shape{2, 3};
    constexpr std::array<std::int64_t, 2U> delivered_shape{3, 2};
    auto tensor = make_contract_test_tensor(delivered_shape, DLDataType{kDLUInt, 8U, 1U});
    ASSERT_TRUE(tensor) << runtime_error_message(tensor.error());
    const auto contract = normalized_input(TensorInputSpec{
        .representation = {.memory_kind = MemoryKind::kHost,
                           .dtype = DLDataType{kDLUInt, 8U, 1U},
                           .rank = 2U},
        .bounds = tensor_bounds(maximum_shape),
    });
    ASSERT_TRUE(contract);
    exercise(std::move(*tensor), false, true, contract);
  }
  {
    SCOPED_TRACE("contiguous stride family");
    constexpr std::array<std::int64_t, 2U> shape{2, 2};
    constexpr std::array<std::int64_t, 2U> padded_byte_strides{3, 1};
    auto tensor =
        make_contract_test_tensor(shape, DLDataType{kDLUInt, 8U, 1U}, padded_byte_strides);
    ASSERT_TRUE(tensor) << runtime_error_message(tensor.error());
    const auto contract = normalized_input(TensorInputSpec{
        .representation = {.memory_kind = MemoryKind::kHost,
                           .dtype = DLDataType{kDLUInt, 8U, 1U},
                           .rank = 2U},
        .bounds = tensor_bounds(std::array<std::int64_t, 2U>{2, 3}),
    });
    ASSERT_TRUE(contract);
    exercise(std::move(*tensor), false, true, contract);
  }
  {
    SCOPED_TRACE("maximum per-axis stride");
    constexpr std::array<std::int64_t, 2U> shape{2, 2};
    constexpr std::array<std::int64_t, 2U> padded_byte_strides{3, 1};
    constexpr std::array axes{
        TensorAxisBounds{.max_extent = 2U, .max_abs_stride_bytes = 2U},
        TensorAxisBounds{.max_extent = 2U, .max_abs_stride_bytes = 1U},
    };
    auto tensor =
        make_contract_test_tensor(shape, DLDataType{kDLUInt, 8U, 1U}, padded_byte_strides);
    ASSERT_TRUE(tensor) << runtime_error_message(tensor.error());
    const auto contract = normalized_input(TensorInputSpec{
        .representation = {.memory_kind = MemoryKind::kHost,
                           .dtype = DLDataType{kDLUInt, 8U, 1U},
                           .rank = 2U,
                           .strides = TensorStridePolicy::kBoundedStrided},
        .bounds = TensorBounds{.max_rank = 2U, .max_byte_span = 5U, .axes = axes},
    });
    ASSERT_TRUE(contract);
    exercise(std::move(*tensor), false, true, contract);
  }
}

// A pass-through producer contract remains authoritative even though its storage is supplied by
// the application. Runtime validates the complete contract before assigning a publication
// sequence or entering fanout, so a rejected axis bound cannot consume emit authority.
// REQ: OPERATOR-042
TEST(ChannelBindingParityTest, PassThroughTensorEnforcesProducerAxisBoundsBeforeFanout) {
  TensorSessionFixture fixture;
  constexpr std::array maximum_shape{std::int64_t{1}};
  auto contract = normalize_tensor_output_spec(TensorOutputSpec{
      .representation = {.memory_kind = MemoryKind::kHost,
                         .dtype = DLDataType{kDLUInt, 64U, 1U},
                         .rank = 1U},
      .bounds = tensor_bounds(maximum_shape),
  });
  ASSERT_TRUE(contract) << (contract ? "" : contract.error().message);
  fixture.base.plan.ports[0].tensor_layout = contract->representation;
  fixture.base.plan.ports[0].tensor_contract = *contract;

  auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));

  // Key setup: the first Tensor fits aggregate service storage but exceeds the producer's exact
  // per-axis bound; the second Tensor satisfies the same frozen contract.
  constexpr std::array oversized_shape{std::int64_t{2}};
  auto oversized = make_contract_test_tensor(oversized_shape, DLDataType{kDLUInt, 64U, 1U});
  auto matching = make_contract_test_tensor(maximum_shape, DLDataType{kDLUInt, 64U, 1U});
  ASSERT_TRUE(oversized) << runtime_error_message(oversized.error());
  ASSERT_TRUE(matching) << runtime_error_message(matching.error());

  const auto rejected = fixture.source.output.emit(std::move(*oversized));
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, ErrorCode::kInvalidArgument);

  // Key assertion: rejecting the producer-contract violation preserves publication authority, so
  // the next conforming Tensor is accepted as the first sample.
  ASSERT_TRUE(fixture.source.output.emit(std::move(*matching)));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// A catalog Tensor import retains the complete normalized logical record, not only its physical
// byte-span and memory-kind subset. Every mismatch is rejected before reserving arrival identity
// or constructing a typed cache slot, even when the backing allocation itself fits the pool.
// REQ: SYSTEM-001
TEST(ChannelBindingParityTest, ExternalTensorImportRejectsLogicalContractDriftBeforeCacheMutation) {
  TensorSessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  const PortId input{2U};
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_external_tensor_contract(
      **session,
      input,
      TensorBoundaryContractRecord{
          .memory_kind = MemoryKind::kHost,
          .dtype = DLDataType{kDLUInt, 64U, 1U},
          .rank = 2U,
          .max_rank = 2U,
          .max_byte_span = 64U,
          .axes = {{.max_extent = 2U, .max_abs_stride_bytes = 16U},
                   {.max_extent = 2U, .max_abs_stride_bytes = 8U}},
          .strides = TensorStridePolicy::kContiguousOnly,
      }));
  const auto before = RuntimeChannelSessionTestAccess::input_reset_state(**session, input);
  ASSERT_TRUE(before) << runtime_error_message(before.error());

  const auto expect_rejected_without_mutation = [&](Tensor tensor, const std::uint64_t byte_size) {
    auto rejected = RuntimeChannelSessionTestAccess::inject_received_tensor(
        **session,
        input,
        Sample<Tensor>{std::move(tensor),
                       SampleMetadata{
                           .sequence = 1U,
                           .byte_size = byte_size,
                           .payload_kind = PayloadKind::kDescriptorBacked,
                       }});
    ASSERT_FALSE(rejected);
    EXPECT_EQ(rejected.error().code, ErrorCode::kProtocolError);
    const auto after = RuntimeChannelSessionTestAccess::input_reset_state(**session, input);
    ASSERT_TRUE(after) << runtime_error_message(after.error());
    EXPECT_EQ(after->next_arrival_sequence, before->next_arrival_sequence);
    EXPECT_EQ(after->next_global_arrival_order, before->next_global_arrival_order);
    EXPECT_EQ(after->occupied_samples, before->occupied_samples);
  };

  constexpr std::array<std::int64_t, 2U> shape_2x2{2, 2};
  auto wrong_dtype = make_contract_test_tensor(shape_2x2, DLDataType{kDLUInt, 32U, 1U});
  ASSERT_TRUE(wrong_dtype) << runtime_error_message(wrong_dtype.error());
  expect_rejected_without_mutation(std::move(*wrong_dtype), 16U);

  constexpr std::array<std::int64_t, 1U> shape_rank_one{4};
  auto wrong_rank = make_contract_test_tensor(shape_rank_one, DLDataType{kDLUInt, 64U, 1U});
  ASSERT_TRUE(wrong_rank) << runtime_error_message(wrong_rank.error());
  expect_rejected_without_mutation(std::move(*wrong_rank), 32U);

  constexpr std::array<std::int64_t, 2U> shape_3x2{3, 2};
  auto oversized_axis = make_contract_test_tensor(shape_3x2, DLDataType{kDLUInt, 64U, 1U});
  ASSERT_TRUE(oversized_axis) << runtime_error_message(oversized_axis.error());
  expect_rejected_without_mutation(std::move(*oversized_axis), 48U);

  constexpr std::array<std::int64_t, 2U> noncontiguous_byte_strides{24, 8};
  auto noncontiguous = make_contract_test_tensor(
      shape_2x2, DLDataType{kDLUInt, 64U, 1U}, noncontiguous_byte_strides);
  ASSERT_TRUE(noncontiguous) << runtime_error_message(noncontiguous.error());
  expect_rejected_without_mutation(std::move(*noncontiguous), 32U);

  {
    SCOPED_TRACE("max_rank");
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_external_tensor_contract(
        **session,
        input,
        TensorBoundaryContractRecord{
            .memory_kind = MemoryKind::kHost,
            .dtype = DLDataType{kDLUInt, 64U, 1U},
            .max_rank = 1U,
            .max_byte_span = 64U,
            .strides = TensorStridePolicy::kContiguousOnly,
        }));
    auto excessive_rank = make_contract_test_tensor(shape_2x2, DLDataType{kDLUInt, 64U, 1U});
    ASSERT_TRUE(excessive_rank) << runtime_error_message(excessive_rank.error());
    expect_rejected_without_mutation(std::move(*excessive_rank), 32U);
  }

  {
    SCOPED_TRACE("max_byte_span");
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_external_tensor_contract(
        **session,
        input,
        TensorBoundaryContractRecord{
            .memory_kind = MemoryKind::kHost,
            .dtype = DLDataType{kDLUInt, 64U, 1U},
            .rank = 2U,
            .max_rank = 2U,
            .max_byte_span = 16U,
            .axes = {{.max_extent = 2U, .max_abs_stride_bytes = 16U},
                     {.max_extent = 2U, .max_abs_stride_bytes = 8U}},
            .strides = TensorStridePolicy::kContiguousOnly,
        }));
    auto excessive_span = make_contract_test_tensor(shape_2x2, DLDataType{kDLUInt, 64U, 1U});
    ASSERT_TRUE(excessive_span) << runtime_error_message(excessive_span.error());
    expect_rejected_without_mutation(std::move(*excessive_span), 32U);
  }

  {
    SCOPED_TRACE("memory_kind");
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_external_tensor_contract(
        **session,
        input,
        TensorBoundaryContractRecord{
            .memory_kind = MemoryKind::kHost,
            .dtype = DLDataType{kDLUInt, 64U, 1U},
            .rank = 2U,
            .max_rank = 2U,
            .max_byte_span = 64U,
            .axes = {{.max_extent = 2U, .max_abs_stride_bytes = 16U},
                     {.max_extent = 2U, .max_abs_stride_bytes = 8U}},
            .strides = TensorStridePolicy::kContiguousOnly,
        }));
    auto wrong_memory = make_contract_test_tensor(shape_2x2,
                                                  DLDataType{kDLUInt, 64U, 1U},
                                                  std::span<const std::int64_t>{},
                                                  MemoryStorageType::kHost);
    ASSERT_TRUE(wrong_memory) << runtime_error_message(wrong_memory.error());
    expect_rejected_without_mutation(std::move(*wrong_memory), 32U);
  }

  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_external_tensor_contract(
      **session,
      input,
      TensorBoundaryContractRecord{
          .memory_kind = MemoryKind::kHost,
          .dtype = DLDataType{kDLUInt, 64U, 1U},
          .rank = 2U,
          .max_rank = 2U,
          .max_byte_span = 64U,
          .axes = {{.max_extent = 2U, .max_abs_stride_bytes = 8U},
                   {.max_extent = 2U, .max_abs_stride_bytes = 8U}},
          .strides = TensorStridePolicy::kContiguousOnly,
      }));
  auto oversized_derived_stride =
      make_contract_test_tensor(shape_2x2, DLDataType{kDLUInt, 64U, 1U});
  ASSERT_TRUE(oversized_derived_stride) << runtime_error_message(oversized_derived_stride.error());
  expect_rejected_without_mutation(std::move(*oversized_derived_stride), 32U);

  (*session)->cancel();
  EXPECT_TRUE((*session)->join());
}

// Tensor-only test seams are deliberately narrower than the generic input state helpers. Reject a
// scalar subscriber before taking its lock or interpreting its typed slot storage as
// Sample<Tensor>. REQ: SYSTEM-001
TEST(ChannelBindingParityTest, TensorTestSeamsRejectScalarInputWithoutMutation) {
  SessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  const PortId input{2U};
  const auto before = RuntimeChannelSessionTestAccess::input_reset_state(**session, input);
  ASSERT_TRUE(before) << runtime_error_message(before.error());

  auto tensor =
      make_contract_test_tensor(std::array<std::int64_t, 1U>{1U}, DLDataType{kDLUInt, 8U, 1U});
  ASSERT_TRUE(tensor) << runtime_error_message(tensor.error());
  auto injected = RuntimeChannelSessionTestAccess::inject_received_tensor(
      **session,
      input,
      Sample<Tensor>{
          std::move(*tensor),
          SampleMetadata{.byte_size = 1U, .payload_kind = PayloadKind::kDescriptorBacked}});
  ASSERT_FALSE(injected);
  EXPECT_EQ(injected.error().code, ErrorCode::kInvalidArgument);

  auto configured = RuntimeChannelSessionTestAccess::set_external_tensor_contract(
      **session,
      input,
      TensorBoundaryContractRecord{
          .memory_kind = MemoryKind::kHost,
          .max_byte_span = 1U,
          .strides = TensorStridePolicy::kContiguousOnly,
      });
  ASSERT_FALSE(configured);
  EXPECT_EQ(configured.error().code, ErrorCode::kInvalidArgument);

  const auto after = RuntimeChannelSessionTestAccess::input_reset_state(**session, input);
  ASSERT_TRUE(after) << runtime_error_message(after.error());
  EXPECT_EQ(after->active_generation, before->active_generation);
  EXPECT_EQ(after->view_generation, before->view_generation);
  EXPECT_EQ(after->producer_epoch, before->producer_epoch);
  EXPECT_EQ(after->next_arrival_sequence, before->next_arrival_sequence);
  EXPECT_EQ(after->next_global_arrival_order, before->next_global_arrival_order);
  EXPECT_EQ(after->occupied_samples, before->occupied_samples);
}

// Every lossy raw-Tensor transport reserves the exact occupied replacement destination before
// consuming its staged head. Readiness and activation cannot claim that incumbent while first-use
// Tensor materialization is complete but typed commit is deliberately paused.
// REQ: RUNTIME-026
// REQ: RUNTIME-042
// REQ: RUNTIME-044
TEST(ChannelBindingParityTest, RawTensorIngressRetainsExactLossyDestinationAcrossMaterialization) {
  // Key setup: pause raw-Tensor ingress after materialization while a pre-witness incumbent
  // occupies the only typed slot.
  for (const QueuePolicy policy : {QueuePolicy::kDropOldest}) {
    SCOPED_TRACE(static_cast<std::uint32_t>(policy));
    const std::string endpoint_name = unique_service();
    Graph graph{"raw-tensor-ingress-transaction"};
    const ClockKey clock =
        graph.bind_shared_clock("runtime-clock", std::make_shared<ManualClock>(0));
    graph.set_default_clock(clock);
    constexpr std::array axes{TensorAxisBounds{.max_extent = 8U, .max_abs_stride_bytes = 1U}};
    TopicOptions options;
    options.endpoint_name = endpoint_name;
    options.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = 2U};
    options.tensor = TensorBoundaryContract{
        .representation = {.memory_kind = MemoryKind::kHost,
                           .dtype = DLDataType{kDLUInt, 8U, 1U},
                           .rank = 1U,
                           .strides = TensorStridePolicy::kContiguousOnly},
        .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 8U, .axes = axes},
        .transfer =
            TensorTransferPolicy{.foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded},
    };
    const auto imported =
        graph.import_topic<Tensor>("holomq://" + endpoint_name, std::move(options));
    const auto sink = graph.op<TensorSinkOperator>("sink");
    auto& sink_input = const_cast<Input<Tensor>&>(sink->input);
    graph.add_flow(imported.output(),
                   sink->input,
                   ConnectionOptions{.queue_policy = policy, .queue_depth = 1U});

    const ExecutionPlan plan = compile(graph);
    ASSERT_TRUE(plan.ok()) << plan.json();
    const auto state = ExecutionPlanAccess::retain(plan);
    ASSERT_NE(state, nullptr);
    const auto input =
        std::find_if(state->ports.begin(), state->ports.end(), [](const PortPlanRecord& candidate) {
          return candidate.owner_kind == PortOwnerKind::kOperator &&
                 candidate.direction == PortDirection::kInput;
        });
    ASSERT_NE(input, state->ports.end());
    const std::array replay_ports{RuntimeChannelPortReplayRecord{
        .port = input->id,
        .owner = input->owner,
        .direction = PortDirection::kInput,
        .declaration = &sink_input,
        .type_ops = runtime_port_type_ops<Tensor>(),
    }};
    BlockingTypedIngressObserver observer;
    auto session = RuntimeChannelSession::create(*state, replay_ports);
    ASSERT_TRUE(session) << runtime_error_message(session.error());

    auto client_runtime = ChannelRuntime::create();
    ASSERT_TRUE(client_runtime) << runtime_error_message(client_runtime.error());
    auto publisher =
        client_runtime->create_publisher<Tensor>(plan.endpoint_catalog(), endpoint_name);
    ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());
    ASSERT_TRUE(wait_for_catalog_subscriber(*publisher));
    const auto publish_tensor = [&](const std::byte value) -> expected<void, Error> {
      const auto deadline = std::chrono::steady_clock::now() + 2s;
      while (std::chrono::steady_clock::now() < deadline) {
        auto published = publish_catalog_host_tensor(*publisher, value);
        if (published || published.error().code != ErrorCode::kNotReady) {
          return published;
        }
        auto refreshed = publisher->update_connections();
        if (!refreshed) {
          return make_unexpected(std::move(refreshed).error());
        }
        std::this_thread::sleep_for(1ms);
      }
      return make_unexpected(Error{ErrorCode::kTimeout});
    };
    ASSERT_TRUE(publish_tensor(std::byte{0x11}));
    ASSERT_TRUE(wait_for_catalog_input(**session, input->id));
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_pre_witness_replacement_allowed(
        **session, input->id, true));

    auto binding =
        (*session)->create_readiness_binding(input->owner,
                                             RuntimePortBindingRequest{
                                                 .plan_generation = state->plan_generation,
                                                 .run_generation = 1U,
                                                 .kind = RuntimeEventKind::kInputArrival,
                                                 .source = make_source_id(1U),
                                                 .source_clock = ClockId{1U},
                                                 .reference_clock = ClockId{1U},
                                                 .capacity = 2U,
                                             });
    ASSERT_TRUE(binding) << runtime_error_message(binding.error());
    observer.block.store(true, std::memory_order_release);
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_typed_ingress_observer(
        **session, input->id, &observer, &BlockingTypedIngressObserver::observe));
    ASSERT_TRUE(publish_tensor(std::byte{0x22}));
    expected<void, Error> receive_result = make_unexpected(Error{ErrorCode::kNotReady});
    std::thread receiver([&] { receive_result = wait_for_catalog_input(**session, input->id); });
    const auto observer_deadline = std::chrono::steady_clock::now() + 2s;
    while (!observer.entered.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < observer_deadline) {
      std::this_thread::yield();
    }
    std::promise<void> terminal_probe_started_promise;
    auto terminal_probe_started = terminal_probe_started_promise.get_future();
    auto terminal_probe = std::async(std::launch::async, [&] {
      terminal_probe_started_promise.set_value();
      return (*session)->input_terminal_observed(input->id);
    });

    // Key assertion: the exact occupied destination remains reserved and hidden from readiness
    // until the newer Tensor backing commits once. The owner-side terminal probe must remain
    // pending while that first-use/import seam owns the transport mutex; it may inspect the exact
    // staged state only after the receive transaction releases the mutex.
    const auto reservations =
        RuntimeChannelSessionTestAccess::typed_ingress_reservations(**session, input->id);
    const auto retained = RuntimeChannelSessionTestAccess::input_reset_state(**session, input->id);
    const auto blocked_poll = (*binding)->poll(TimePoint{1, ClockId{1U}});
    RuntimeEvent blocked_event;
    const auto blocked_drain = (*binding)->drain(std::span<RuntimeEvent>{&blocked_event, 1U});
    const bool retry_remains_visible = (*binding)->poll_target().has_value();
    const auto terminal_probe_start_status = terminal_probe_started.wait_for(2s);
    const auto terminal_probe_status_before_release =
        terminal_probe_start_status == std::future_status::ready ? terminal_probe.wait_for(10ms)
                                                                 : std::future_status::deferred;
    observer.release.store(true, std::memory_order_release);
    receiver.join();
    const auto terminal_probe_status_after_release = terminal_probe.wait_for(2s);
    ASSERT_TRUE(observer.entered.load(std::memory_order_acquire))
        << (receive_result ? std::string{} : runtime_error_message(receive_result.error()));
    EXPECT_EQ(observer.calls.load(std::memory_order_relaxed), 1U);
    EXPECT_EQ(terminal_probe_start_status, std::future_status::ready);
    EXPECT_EQ(terminal_probe_status_before_release, std::future_status::timeout);
    ASSERT_EQ(terminal_probe_status_after_release, std::future_status::ready);
    auto terminal_observed = terminal_probe.get();
    ASSERT_TRUE(terminal_observed) << runtime_error_message(terminal_observed.error());
    EXPECT_FALSE(*terminal_observed);
    ASSERT_TRUE(reservations) << runtime_error_message(reservations.error());
    EXPECT_EQ(*reservations, 1U);
    ASSERT_TRUE(retained) << runtime_error_message(retained.error());
    EXPECT_EQ(retained->occupied_samples, 1U);
    EXPECT_EQ(retained->next_arrival_sequence, 2U);
    ASSERT_TRUE(blocked_poll) << runtime_error_message(blocked_poll.error());
    ASSERT_TRUE(blocked_drain) << runtime_error_message(blocked_drain.error());
    EXPECT_EQ(*blocked_drain, 0U);
    EXPECT_TRUE(retry_remains_visible);
    ASSERT_TRUE(receive_result) << runtime_error_message(receive_result.error());
    const auto released =
        RuntimeChannelSessionTestAccess::typed_ingress_reservations(**session, input->id);
    ASSERT_TRUE(released) << runtime_error_message(released.error());
    EXPECT_EQ(*released, 0U);

    ASSERT_TRUE((*binding)->poll(TimePoint{1, ClockId{1U}}));
    RuntimeEvent event;
    auto drained = (*binding)->drain(std::span<RuntimeEvent>{&event, 1U});
    ASSERT_TRUE(drained) << runtime_error_message(drained.error());
    ASSERT_EQ(*drained, 1U);
    auto record = (*session)->temporal_sample(event.port, event.source_sequence);
    ASSERT_TRUE(record) << runtime_error_message(record.error());
    EXPECT_EQ(record->source_sequence, 2U);
    EXPECT_EQ(record->arrival_sequence, 2U);
    const std::array selection{RuntimeChannelActivationSelection{
        .port = input->id,
        .selection = {InputState::kSelected, SelectionReason::kNone},
        .samples = std::span<const TemporalSampleRecord>{&*record, 1U},
        .destructive = true,
    }};
    ASSERT_TRUE((*session)->prepare_activation(input->owner, 1U, selection));
    ASSERT_TRUE((*session)->prepare_tensor_access(input->owner, 1U, nullptr));
    auto sample = sink_input.receive();
    ASSERT_TRUE(sample) << runtime_error_message(sample.error());
    ASSERT_NE(sample->data.data(), nullptr);
    EXPECT_EQ(*static_cast<const std::byte*>(sample->data.data()), std::byte{0x22});
    EXPECT_EQ(sample->metadata.sequence, 2U);
    ASSERT_TRUE((*binding)->retire(event.source_sequence));
    ASSERT_TRUE((*session)->retire_activation(input->owner, 1U, true));
    ASSERT_TRUE(
        (*session)->retire_samples(input->id, std::span<const TemporalSampleRecord>{&*record, 1U}));
    (*session)->cancel();
    EXPECT_TRUE((*session)->join());
  }
}

// A host-only raw-Tensor endpoint has no compiler-frozen CUDA device and therefore must not enter
// the device-affinity inventory or initialize a controller cursor during destructive restart.
// HoloMQ's pageable-host transport still imports a CUDA-VMM split-generation page, but no payload
// device affinity exists for the controller to initialize before a destructive restart.
// REQ: CHAN-104
// REQ: RUNTIME-020
// REQ: RUNTIME-024
TEST(ChannelBindingParityTest, HostTensorRestartSkipsCudaDescriptorImportInitialization) {
  TensorSessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto initialization_count =
      RuntimeChannelSessionTestAccess::controller_descriptor_import_initialization_count(**session);
  ASSERT_TRUE(initialization_count) << runtime_error_message(initialization_count.error());
  EXPECT_EQ(*initialization_count, 0U);

  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  auto owned = make_owned_tensor_sample(71U);
  ASSERT_TRUE(owned) << runtime_error_message(owned.error());
  auto releases = owned->releases;
  ASSERT_TRUE(fixture.source.output.emit(std::move(owned->tensor)));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));

  const std::array affected{VertexId{1U}, VertexId{2U}};
  ASSERT_TRUE((*session)->commit_restart(VertexId{1U}, affected));
  EXPECT_EQ(releases->load(std::memory_order_relaxed), 1U);
  initialization_count =
      RuntimeChannelSessionTestAccess::controller_descriptor_import_initialization_count(**session);
  ASSERT_TRUE(initialization_count) << runtime_error_message(initialization_count.error());
  EXPECT_EQ(*initialization_count, 0U);
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// Lifecycle restart drains device-affine raw-Tensor descriptors on the controller thread. That
// thread must initialize every frozen import device before crossing the destructive discard
// boundary, select the subscriber's device, and restore the caller afterward.
// REQ: CHAN-103
// REQ: RUNTIME-020
// REQ: RUNTIME-024
TEST(ChannelBindingParityTest, RestartInitializesRawTensorDiscardDeviceAndRestoresCaller) {
  test::skip_without_cuda_device();

  int device_count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
  ASSERT_GT(device_count, 0);
  int ambient_device = -1;
  ASSERT_EQ(cudaGetDevice(&ambient_device), cudaSuccess);
  const int import_device = device_count > 1 ? (ambient_device == 0 ? 1 : 0) : ambient_device;

  TensorSessionFixture fixture;
  auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::configure_descriptor_import_device(
      **session, PortId{2U}, DeviceId{import_device}));
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  auto owned = make_owned_tensor_sample(73U);
  ASSERT_TRUE(owned) << runtime_error_message(owned.error());
  auto releases = owned->releases;
  ASSERT_TRUE(fixture.source.output.emit(std::move(owned->tensor)));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));

  const std::array affected{VertexId{1U}, VertexId{2U}};
  ASSERT_TRUE((*session)->commit_restart(VertexId{1U}, affected));
  EXPECT_EQ(releases->load(std::memory_order_relaxed), 1U);
  auto initialization_count =
      RuntimeChannelSessionTestAccess::controller_descriptor_import_initialization_count(**session);
  ASSERT_TRUE(initialization_count) << runtime_error_message(initialization_count.error());
  EXPECT_EQ(*initialization_count, 1U);

  int restored_device = -1;
  ASSERT_EQ(cudaGetDevice(&restored_device), cudaSuccess);
  EXPECT_EQ(restored_device, ambient_device);
}

}  // namespace
}  // namespace holoscan::detail

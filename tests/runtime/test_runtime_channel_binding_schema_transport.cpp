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
#include <cstring>
#include <future>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/core/graph.hpp>
#include <holoscan/memory/block_memory_pool.hpp>
#include <holoscan/memory/tensor_storage.hpp>

#include "../common/cuda_test_utils.hpp"
#include "channel/channel_catalog_binding.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_state.hpp"
#include "core/source_id_internal.hpp"
#include "runtime/runtime_channel_binding.hpp"
#include "runtime/runtime_phase_counters.hpp"

#include "runtime_channel_binding_schema_tensor_test_support.hpp"
#include "runtime_channel_binding_test_support.hpp"

namespace holoscan::detail {
namespace {

using namespace std::chrono_literals;
using namespace runtime_channel_binding_test;
using namespace runtime_channel_binding_schema_tensor_test;

// Generated schemas must expose bounded encoding and transport attribution without changing the
// production publication path or allocating trace state during execution.
TEST(RuntimeChannelBinding, GeneratedSchemaTransportEmitRecordsOrderedPhases) {
  GeneratedSchemaSessionFixture fixture;
  std::array<RuntimePhaseTraceRecord, 32U> trace_storage;
  const auto counters = std::make_shared<RuntimePhaseCounters>(std::span{trace_storage});
  fixture.base.plan.executable_factories.phase_counters = counters;
  auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());

  // Key setup: publish generated owning values through both generic-loan and direct-value APIs.
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  auto loan = fixture.source.output.allocate();
  ASSERT_TRUE(loan) << runtime_error_message(loan.error());
  loan->value().data = "trace generated schema publication";
  ASSERT_TRUE(fixture.source.output.emit(std::move(*loan)));
  schema::StringT direct_value;
  direct_value.data = "trace direct schema publication";
  ASSERT_TRUE(fixture.source.output.emit(direct_value));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());

  constexpr std::array kExpectedPoints{
      RuntimePhaseTracePoint::kInlinePayloadEmitStarted,
      RuntimePhaseTracePoint::kInlinePayloadEncoded,
      RuntimePhaseTracePoint::kInlinePayloadTransportPublicationStarted,
      RuntimePhaseTracePoint::kInlinePayloadTransportPublicationReturned,
      RuntimePhaseTracePoint::kInlinePayloadEmitCompleted,
  };
  std::vector<RuntimePhaseTraceRecord> emit_trace;
  std::copy_if(counters->trace_records().begin(),
               counters->trace_records().end(),
               std::back_inserter(emit_trace),
               [&](const RuntimePhaseTraceRecord& record) {
                 return std::find(kExpectedPoints.begin(), kExpectedPoints.end(), record.point) !=
                        kExpectedPoints.end();
               });

  // Key assertion: every ordered phase retains the exact output port and source sequence.
  ASSERT_EQ(emit_trace.size(), kExpectedPoints.size() * 2U);
  for (std::size_t sequence_index = 0U; sequence_index < 2U; ++sequence_index) {
    for (std::size_t point_index = 0U; point_index < kExpectedPoints.size(); ++point_index) {
      const RuntimePhaseTraceRecord& record =
          emit_trace[sequence_index * kExpectedPoints.size() + point_index];
      EXPECT_EQ(record.point, kExpectedPoints[point_index]);
      EXPECT_EQ(record.output_port_id, 1U);
      EXPECT_EQ(record.source_sequence, sequence_index + 1U);
    }
  }
  EXPECT_EQ(counters->trace_dropped(), 0U);
}

// Eventless HoloMQ generated schemas use the ordinary activation cut while preserving the schema
// envelope verifier and owning-value materialization path.
// REQ: CHAN-117
// REQ: RUNTIME-019
TEST(RuntimeChannelBinding, HoloMQGeneratedSchemaUsesOrdinaryActivationCut) {
  // Key setup: publish one generated schema on an exact 1:1 eventless HoloMQ edge.
  GeneratedSchemaSessionFixture fixture;
  fixture.base.use_one_slot_blocking_transport();
  fixture.base.use_same_partition_eventless_transport();
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
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  EXPECT_FALSE((*session)->arm_direct_continuation(VertexId{1U}, 0U));

  constexpr std::string_view kExpected{"HoloMQ generated-schema activation cut"};
  schema::StringT value;
  value.data = kExpected;
  ASSERT_TRUE(fixture.source.output.emit(
      std::move(value), EmitOptions{.frame_id = 313U, .trace = TraceContext::root(0xA11CEU)}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  ASSERT_TRUE((*session)->flush_process_local_activation_cut());
  const auto post_cut_indexes = RuntimeChannelSessionTestAccess::pump_index_state(**session);
  ASSERT_TRUE(post_cut_indexes) << runtime_error_message(post_cut_indexes.error());
  EXPECT_FALSE(post_cut_indexes->background_thread_required);
  EXPECT_FALSE(post_cut_indexes->background_thread_running);

  // Key assertion: ordinary cut delivery exposes one verified owning value with intact metadata.
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
  auto received = fixture.sink.input.receive();
  ASSERT_TRUE(received) << runtime_error_message(received.error());
  EXPECT_EQ(received->data.data, kExpected);
  EXPECT_EQ(received->metadata.sequence, 1U);
  EXPECT_EQ(received->metadata.frame_id, 313U);
  EXPECT_EQ(received->metadata.trace_id, 0xA11CEU);
  ASSERT_TRUE((*binding)->retire(event->source_sequence));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  ASSERT_TRUE((*session)->retire_samples(PortId{2U}, selection.front().samples));

  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// One bounded generated-schema publication is independently verified and materialized for every
// recipient before the serial controller completes its post-compute activation cut.
// REQ: CHAN-101
// REQ: CHAN-117
// REQ: RUNTIME-019
TEST(RuntimeChannelBinding, HoloMQGeneratedSchemaFanoutUsesOrdinaryActivationCut) {
  GeneratedSchemaSessionFixture fixture;
  fixture.add_blocking_fanout();
  fixture.base.use_same_partition_eventless_transport();
  fixture.base.use_single_partition_runner();
  const auto counters = std::make_shared<RuntimePhaseCounters>();
  fixture.base.plan.executable_factories.phase_counters = counters;
  auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto left_binding =
      (*session)->create_readiness_binding(VertexId{2U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.base.plan.plan_generation,
                                               .run_generation = 1U,
                                               .kind = RuntimeEventKind::kInputArrival,
                                               .source = make_source_id(1U),
                                               .source_clock = ClockId{1U},
                                               .reference_clock = ClockId{1U},
                                               .capacity = 8U,
                                           });
  auto right_binding =
      (*session)->create_readiness_binding(VertexId{3U},
                                           RuntimePortBindingRequest{
                                               .plan_generation = fixture.base.plan.plan_generation,
                                               .run_generation = 1U,
                                               .kind = RuntimeEventKind::kInputArrival,
                                               .source = make_source_id(2U),
                                               .source_clock = ClockId{1U},
                                               .reference_clock = ClockId{1U},
                                               .capacity = 8U,
                                           });
  ASSERT_TRUE(left_binding) << runtime_error_message(left_binding.error());
  ASSERT_TRUE(right_binding) << runtime_error_message(right_binding.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());

  // Key setup: publish one owning generated schema to two eventless subscribers and establish
  // the ordinary activation cut before observing either readiness stream.
  auto indexes = RuntimeChannelSessionTestAccess::pump_index_state(**session);
  ASSERT_TRUE(indexes) << runtime_error_message(indexes.error());
  EXPECT_TRUE(indexes->exact);
  EXPECT_EQ(indexes->wait_handle_count, 0U);
  EXPECT_EQ(indexes->drain_subscriber_count, 2U);
  EXPECT_FALSE(indexes->background_thread_required);
  EXPECT_FALSE(indexes->background_thread_running);
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  constexpr std::string_view kExpected{"HoloMQ generated-schema fanout activation cut"};
  schema::StringT value;
  value.data = kExpected;
  ASSERT_TRUE(fixture.source.output.emit(
      std::move(value), EmitOptions{.frame_id = 317U, .trace = TraceContext::root(0xFAD0U)}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  ASSERT_TRUE((*session)->flush_process_local_activation_cut());
  const auto post_cut_indexes = RuntimeChannelSessionTestAccess::pump_index_state(**session);
  ASSERT_TRUE(post_cut_indexes) << runtime_error_message(post_cut_indexes.error());
  EXPECT_FALSE(post_cut_indexes->background_thread_required);
  EXPECT_FALSE(post_cut_indexes->background_thread_running);
  const auto snapshot = counters->snapshot();
  EXPECT_EQ(snapshot[RuntimePhaseCounters::index(
                RuntimePhaseCounter::kChannelActivationCutSubscriberProbes)],
            2U);
  EXPECT_EQ(snapshot[RuntimePhaseCounters::index(
                RuntimePhaseCounter::kChannelActivationCutTransportProbes)],
            0U);
  EXPECT_EQ(snapshot[RuntimePhaseCounters::index(
                RuntimePhaseCounter::kChannelActivationCutMaterializations)],
            2U);
  auto left_event = wait_for_event(**session, **left_binding);
  auto right_event = wait_for_event(**session, **right_binding);
  ASSERT_TRUE(left_event) << runtime_error_message(left_event.error());
  ASSERT_TRUE(right_event) << runtime_error_message(right_event.error());
  auto left_sample = (*session)->temporal_sample(left_event->port, left_event->source_sequence);
  auto right_sample = (*session)->temporal_sample(right_event->port, right_event->source_sequence);
  ASSERT_TRUE(left_sample) << runtime_error_message(left_sample.error());
  ASSERT_TRUE(right_sample) << runtime_error_message(right_sample.error());
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

  // Key assertion: both subscribers independently expose the same verified owning value and
  // metadata without any WaitSet attachment.
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, left_selection));
  auto left_value = fixture.sink.input.receive();
  ASSERT_TRUE(left_value) << runtime_error_message(left_value.error());
  EXPECT_EQ(left_value->data.data, kExpected);
  EXPECT_EQ(left_value->metadata.sequence, 1U);
  EXPECT_EQ(left_value->metadata.frame_id, 317U);
  EXPECT_EQ(left_value->metadata.trace_id, 0xFAD0U);
  ASSERT_TRUE((*left_binding)->retire(left_event->source_sequence));
  ASSERT_TRUE((*session)->retire_samples(left_event->port, left_selection.front().samples));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  ASSERT_TRUE((*session)->prepare_activation(VertexId{3U}, 1U, right_selection));
  auto right_value = fixture.second_sink.input.receive();
  ASSERT_TRUE(right_value) << runtime_error_message(right_value.error());
  EXPECT_EQ(right_value->data.data, kExpected);
  EXPECT_EQ(right_value->metadata.sequence, 1U);
  EXPECT_EQ(right_value->metadata.frame_id, 317U);
  EXPECT_EQ(right_value->metadata.trace_id, 0xFAD0U);
  ASSERT_TRUE((*right_binding)->retire(right_event->source_sequence));
  ASSERT_TRUE((*session)->retire_samples(right_event->port, right_selection.front().samples));
  ASSERT_TRUE((*session)->retire_activation(VertexId{3U}, 1U, true));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// A generated owning value uses the same public loan verbs as a scalar, while Runtime must route
// the nontrivial value through its bounded generated-schema encoder and verifier/materializer
// instead of copying the C++ object representation into the transport sample.
// REQ: CHAN-075
// REQ: CHAN-117
// REQ: CHAN-120
// REQ: RUNTIME-019
TEST(RuntimeChannelBinding, PublishesConstructedGeneratedSchemaLoanThroughCodec) {
  GeneratedSchemaSessionFixture fixture;
  fixture.base.use_drop_oldest_queue_policy();
  fixture.base.use_direct_retained_delivery();
  ASSERT_NE(fixture.replay[0].type_ops, nullptr);
  ASSERT_EQ(fixture.replay[0].type_ops, fixture.replay[1].type_ops);
  EXPECT_EQ(fixture.replay[0].type_ops->codec_family, RuntimePayloadCodecFamily::kGeneratedSchema);
  EXPECT_EQ(fixture.replay[0].type_ops->wire_capacity,
            generated_schema_max_serialized_size_v<schema::StringT>);

  auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  const auto indexes = RuntimeChannelSessionTestAccess::pump_index_state(**session);
  ASSERT_TRUE(indexes) << runtime_error_message(indexes.error());
  EXPECT_EQ(indexes->wait_handle_count, 0U);
  EXPECT_EQ(indexes->drain_subscriber_count, 0U);
  auto binding =
      (*session)->create_readiness_binding(VertexId{2U},
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
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  std::this_thread::sleep_for(20ms);

  constexpr std::string_view kExpected{"generated schema from an output loan"};
  auto loan = fixture.source.output.allocate();
  ASSERT_TRUE(loan) << runtime_error_message(loan.error());
  loan->value().data = kExpected;
  ASSERT_TRUE(fixture.source.output.emit(
      std::move(*loan), EmitOptions{.frame_id = 313U, .trace = TraceContext::root(0xA11CEU)}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));

  auto event = wait_for_event(**session, **binding);
  ASSERT_TRUE(event) << runtime_error_message(event.error());
  auto temporal = (*session)->temporal_sample(event->port, event->source_sequence);
  ASSERT_TRUE(temporal) << runtime_error_message(temporal.error());
  const std::array selected{*temporal};
  const std::array selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = selected,
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, selection));
  auto received = fixture.sink.input.receive();
  ASSERT_TRUE(received) << runtime_error_message(received.error());
  EXPECT_EQ(received->data.data, kExpected);
  EXPECT_EQ(received->metadata.sequence, temporal->source_sequence);
  EXPECT_EQ(received->metadata.frame_id, 313U);
  EXPECT_EQ(received->metadata.trace_id, 0xA11CEU);
  EXPECT_EQ(received->metadata.payload_kind, PayloadKind::kSerializedBytes);
  EXPECT_GT(received->metadata.byte_size, kExpected.size());
  EXPECT_LE(received->metadata.byte_size, fixture.replay[0].type_ops->wire_capacity);
  EXPECT_FALSE(received->metadata.source_clock_id.valid());
  EXPECT_EQ(received->metadata.capture_timestamp_ns, 0);

  ASSERT_TRUE((*binding)->retire(event->source_sequence));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  (*session)->cancel();
  EXPECT_TRUE((*session)->join());
}

// Cross-process generated schemas carry canonical generated bytes through HoloMQ. They do not use
// the additional private envelope that protects direct/process-local retained storage.
// REQ: CHAN-075
// REQ: CHAN-120
// REQ: CHAN-085
TEST(RuntimeChannelBinding, CrossProcessGeneratedSchemaMaterializesCanonicalWirePayload) {
  GeneratedSchemaSessionFixture fixture;
  fixture.use_cross_process_edge();

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
                                               .capacity = 8U,
                                           });
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  std::this_thread::sleep_for(20ms);

  constexpr std::string_view kExpected{"cross-process generated schema"};
  auto loan = fixture.source.output.allocate();
  ASSERT_TRUE(loan) << runtime_error_message(loan.error());
  loan->value().data = kExpected;
  ASSERT_TRUE(fixture.source.output.emit(std::move(*loan), EmitOptions{.frame_id = 417U}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));

  auto event = wait_for_event(**session, **binding);
  ASSERT_TRUE(event) << runtime_error_message(event.error());
  auto temporal = (*session)->temporal_sample(event->port, event->source_sequence);
  ASSERT_TRUE(temporal) << runtime_error_message(temporal.error());
  const std::array selected{*temporal};
  const std::array selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = selected,
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, selection));
  auto received = fixture.sink.input.receive();
  ASSERT_TRUE(received) << runtime_error_message(received.error());
  EXPECT_EQ(received->data.data, kExpected);
  EXPECT_EQ(received->metadata.frame_id, 417U);
  EXPECT_EQ(received->metadata.payload_kind, PayloadKind::kSerializedBytes);
  alignas(std::max_align_t)
      std::array<std::byte, generated_schema_max_serialized_size_v<schema::StringT>>
          canonical{};
  schema::StringT expected_value;
  expected_value.data = kExpected;
  auto canonical_size = schema_package<schema::StringT>().encode(expected_value, canonical);
  ASSERT_TRUE(canonical_size) << runtime_error_message(canonical_size.error());
  EXPECT_EQ(received->metadata.byte_size, *canonical_size);

  ASSERT_TRUE((*binding)->retire(event->source_sequence));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  (*session)->cancel();
  EXPECT_TRUE((*session)->join());
}

// A transport-boundary observer captures the exact bytes Runtime hands to HoloMQ. This
// distinguishes the canonical cross-process FlatBuffer from the larger private process-local
// schema envelope without inferring framing from public SampleMetadata.
// REQ: CHAN-085
TEST(RuntimeChannelBinding, CrossProcessGeneratedSchemaEmitsExactCanonicalWireFrame) {
  GeneratedSchemaSessionFixture fixture;
  fixture.use_cross_process_edge();
  auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());

  auto publisher = std::make_unique<ScriptedInlinePublisher>(
      std::vector<channel_internal::InlinePublicationResult>{
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

  constexpr std::string_view kExpected{"canonical bytes observed by raw peer"};
  schema::StringT expected_value;
  expected_value.data = kExpected;
  alignas(std::max_align_t)
      std::array<std::byte, generated_schema_max_serialized_size_v<schema::StringT>>
          canonical{};
  auto canonical_size = schema_package<schema::StringT>().encode(expected_value, canonical);
  ASSERT_TRUE(canonical_size) << runtime_error_message(canonical_size.error());

  auto loan = fixture.source.output.allocate();
  ASSERT_TRUE(loan) << runtime_error_message(loan.error());
  loan->value() = expected_value;
  ASSERT_TRUE(fixture.source.output.emit(std::move(*loan), EmitOptions{.frame_id = 419U}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));

  ASSERT_EQ(observed->payloads.size(), 1U);
  ASSERT_EQ(observed->metadata_records.size(), 1U);
  ASSERT_EQ(observed->payloads.front().size(), *canonical_size);
  ASSERT_EQ(observed->metadata_records.front().byte_size, *canonical_size);
  EXPECT_TRUE(std::equal(canonical.begin(),
                         canonical.begin() + static_cast<std::ptrdiff_t>(*canonical_size),
                         observed->payloads.front().begin()));

  (*session)->cancel();
  EXPECT_TRUE((*session)->join());
}

// Cross-process Image publication stages an ordinary host Tensor into one compiler-bounded HoloMQ
// pool, then reconstructs the generated value around a read-only imported buffer.
// REQ: CHAN-085
// REQ: CHAN-119
// REQ: CHAN-105
// REQ: TENSOR-SCHEMA-005
TEST(RuntimeChannelBinding, CrossProcessImageStagesAndImportsOrdinaryHostTensor) {
  test::skip_without_cuda_device();
  ImageSessionFixture fixture;
  fixture.use_cross_process_staging(sizeof(std::uint64_t));

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
                                               .capacity = 8U,
                                           });
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());

  // Key setup: the producer emits a normal heap-backed host Tensor, not an exportable HoloMQ
  // buffer. Runtime must copy it into the fixed staging pool before publishing the descriptor.
  constexpr std::uint64_t kExpectedValue = 0xCAFE'BEEFU;
  auto published = publish_image_sample(**session, **binding, fixture.source, kExpectedValue);
  ASSERT_TRUE(published) << runtime_error_message(published.error());
  EXPECT_EQ(published->releases->load(std::memory_order_relaxed), 1U);

  const std::array selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = std::span<const TemporalSampleRecord>{&published->record, 1U},
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, selection));
  ASSERT_TRUE((*session)->prepare_tensor_access(VertexId{2U}, 1U, nullptr));
  {
    auto received = fixture.sink.input.receive();
    ASSERT_TRUE(received) << runtime_error_message(received.error());
    ASSERT_NE(received->data.data, nullptr);
    ASSERT_NE(received->data.data->data_as<std::uint64_t>(), nullptr);
    // Key assertion: the imported Tensor has distinct exportable backing, preserves the generated
    // Image metadata/value, and is immutable to the receiving application.
    EXPECT_NE(received->data.data->data(), published->data);
    EXPECT_EQ(*received->data.data->data_as<std::uint64_t>(), kExpectedValue);
    EXPECT_EQ(received->data.header->frame_id, "runtime_image");
    EXPECT_EQ(received->metadata.trace_id, 0xABCDU);
    auto mutable_data = received->data.data->mutable_data();
    ASSERT_FALSE(mutable_data);
    EXPECT_EQ(mutable_data.error().code, ErrorCode::kPermissionDenied);
  }

  ASSERT_TRUE((*binding)->retire(published->event.source_sequence));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  ASSERT_TRUE((*session)->retire_samples(
      PortId{2U}, std::span<const TemporalSampleRecord>{&published->record, 1U}));
  (*session)->cancel();
  EXPECT_TRUE((*session)->join());
}

// The second descriptor schema must use TensorSchemaStagingPool exactly as Image does; its
// application-defined scalar and string metadata round-trip beside the imported Tensor.
// REQ: CHAN-075
// REQ: CHAN-120
// REQ: OPERATOR-035
TEST(RuntimeChannelBinding, CrossProcessTensorPacketUsesGenericDescriptorStagingPool) {
  test::skip_without_cuda_device();
  TensorPacketSessionFixture fixture;
  fixture.use_cross_process_staging(sizeof(std::uint64_t));

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
                                               .capacity = 8U,
                                           });
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());

  constexpr std::uint64_t kExpectedValue = 0x1020'3040U;
  auto owned = make_owned_tensor_sample(kExpectedValue);
  ASSERT_TRUE(owned) << runtime_error_message(owned.error());
  const void* const source_data = owned->data;
  auto published =
      publish_tensor_packet_tensor(**session, **binding, fixture.source, std::move(owned->tensor));
  ASSERT_TRUE(published) << runtime_error_message(published.error());
  EXPECT_EQ(owned->releases->load(std::memory_order_relaxed), 1U);

  const std::array selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = std::span<const TemporalSampleRecord>{&published->record, 1U},
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, selection));
  ASSERT_TRUE((*session)->prepare_tensor_access(VertexId{2U}, 1U, nullptr));
  {
    auto received = fixture.sink.input.receive();
    ASSERT_TRUE(received) << runtime_error_message(received.error());
    ASSERT_NE(received->data.data, nullptr);
    ASSERT_NE(received->data.data->data_as<std::uint64_t>(), nullptr);
    EXPECT_NE(received->data.data->data(), source_data);
    EXPECT_EQ(*received->data.data->data_as<std::uint64_t>(), kExpectedValue);
    EXPECT_TRUE(received->data.data->is_read_only());
    EXPECT_EQ(received->data.sequence, 99U);
    EXPECT_EQ(received->data.source, "external-sensor");
    EXPECT_EQ(received->data.scale, 1.25);
    EXPECT_EQ(received->metadata.trace_id, 0xFACEU);
  }

  ASSERT_TRUE((*binding)->retire(published->event.source_sequence));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  ASSERT_TRUE((*session)->retire_samples(
      PortId{2U}, std::span<const TemporalSampleRecord>{&published->record, 1U}));
  (*session)->cancel();
  EXPECT_TRUE((*session)->join());
}

// Cross-process Image staging preserves CUDA-pinned host memory instead of routing the embedded
// Tensor through the pageable-host provider.
// REQ: CHAN-106
// REQ: CHAN-135
// REQ: TENSOR-SCHEMA-005
TEST(RuntimeChannelBinding, CrossProcessImagePreservesPinnedHostStaging) {
  test::skip_without_cuda_device();
  int device_ordinal = -1;
  ASSERT_EQ(cudaGetDevice(&device_ordinal), cudaSuccess);

  constexpr std::uint64_t kExpectedValue = 0xA1B2'C3D4U;
  auto source_pool = std::make_shared<BlockMemoryPool>(
      MemoryStorageType::kHost, sizeof(kExpectedValue), 1U, device_ordinal);
  ASSERT_TRUE(source_pool->initialize_pool());
  constexpr std::array<std::int64_t, 1U> kShape{1};
  auto source_tensor = make_tensor(
      kShape, DLDataType{kDLUInt, 64U, 1U}, std::nullopt, MemoryStorageType::kHost, source_pool);
  ASSERT_TRUE(source_tensor) << runtime_error_message(source_tensor.error());
  auto source_data = source_tensor->mutable_data();
  ASSERT_TRUE(source_data) << runtime_error_message(source_data.error());
  std::memcpy(*source_data, &kExpectedValue, sizeof(kExpectedValue));

  ImageSessionFixture fixture;
  fixture.use_cross_process_staging(
      sizeof(kExpectedValue), MemoryKind::kPinnedHost, DeviceId{device_ordinal});
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
                                               .capacity = 8U,
                                           });
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());

  // Key setup: publish from a fresh thread without inheriting the pool-creation CUDA context.
  // The source is pinned but not HoloMQ-backed, so staging must use context-independent pinned
  // payload SHM rather than a devicemap lease or pageable fallback.
  auto published_future = std::async(std::launch::async, [&] {
    return publish_image_tensor(**session, **binding, fixture.source, std::move(*source_tensor));
  });
  auto published = published_future.get();
  ASSERT_TRUE(published) << runtime_error_message(published.error());
  const std::array selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = std::span<const TemporalSampleRecord>{&published->record, 1U},
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, selection));
  // Key setup: a pinned-host-only descriptor consumer has no planned CUDA compute stream.
  ASSERT_TRUE((*session)->prepare_tensor_access(VertexId{2U}, 1U, nullptr));
  {
    auto received = fixture.sink.input.receive();
    ASSERT_TRUE(received) << runtime_error_message(received.error());
    ASSERT_NE(received->data.data, nullptr);
    // Key assertion: the null-stream activation waits for BufferHandle readiness before CPU
    // access, while the import remains CUDA-pinned and read-only instead of becoming pageable.
    EXPECT_EQ(received->data.data->device().device_type, kDLCUDAHost);
    EXPECT_EQ(received->data.data->device().device_id, 0);
    EXPECT_TRUE(received->data.data->is_read_only());
    EXPECT_EQ(*received->data.data->data_as<std::uint64_t>(), kExpectedValue);
  }

  ASSERT_TRUE((*binding)->retire(published->event.source_sequence));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  ASSERT_TRUE((*session)->retire_samples(
      PortId{2U}, std::span<const TemporalSampleRecord>{&published->record, 1U}));
  (*session)->cancel();
  EXPECT_TRUE((*session)->join());
}

// Cross-process Image staging performs a producer-stream-ordered D2D copy and imports the same
// device memory kind on the receiving activation.
// REQ: CHAN-106
// REQ: TENSOR-SCHEMA-005
TEST(RuntimeChannelBinding, CrossProcessImagePreservesCUDADeviceStaging) {
  test::skip_without_cuda_device();
  int device_ordinal = -1;
  ASSERT_EQ(cudaGetDevice(&device_ordinal), cudaSuccess);
  test::UniqueCudaStream producer_stream;
  test::UniqueCudaStream consumer_stream;
  ASSERT_EQ(test::create_non_blocking_cuda_stream(producer_stream), cudaSuccess);
  ASSERT_EQ(test::create_non_blocking_cuda_stream(consumer_stream), cudaSuccess);

  // Key setup: hold S0 before the source write, staging D2D, and exported generation record. A
  // consumer that does not honor the BufferHandle generation can incorrectly complete on S1.
  test::BlockingHostState producer_state;
  ASSERT_EQ(
      cudaLaunchHostFunc(
          producer_stream.get(), test::blocking_host_callback, static_cast<void*>(&producer_state)),
      cudaSuccess);
  test::BlockingHostReleaseGuard producer_release_guard(producer_state);
  ASSERT_TRUE(test::wait_until_started(producer_state));

  constexpr std::uint8_t kExpectedByte = 0x4DU;
  constexpr std::uint64_t kExpectedValue = 0x4D4D'4D4D'4D4D'4D4DU;
  auto source_pool = std::make_shared<BlockMemoryPool>(
      MemoryStorageType::kDevice, sizeof(kExpectedValue), 1U, device_ordinal);
  ASSERT_TRUE(source_pool->initialize_pool());
  constexpr std::array<std::int64_t, 1U> kShape{1};
  auto source_tensor = make_tensor(kShape,
                                   DLDataType{kDLUInt, 64U, 1U},
                                   std::nullopt,
                                   MemoryStorageType::kDevice,
                                   source_pool,
                                   ::cuda::stream_ref{producer_stream.get()});
  ASSERT_TRUE(source_tensor) << runtime_error_message(source_tensor.error());
  auto source_data = source_tensor->mutable_data();
  ASSERT_TRUE(source_data) << runtime_error_message(source_data.error());
  ASSERT_EQ(
      cudaMemsetAsync(*source_data, kExpectedByte, sizeof(kExpectedValue), producer_stream.get()),
      cudaSuccess);
  ASSERT_TRUE(source_tensor->set_readiness_stream(producer_stream.get()));

  ImageSessionFixture fixture;
  fixture.use_cross_process_staging(
      sizeof(kExpectedValue), MemoryKind::kCudaDevice, DeviceId{device_ordinal});
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
                                               .capacity = 8U,
                                           });
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());

  // Key setup: publication receives an ordinary device allocation plus its producer stream. It
  // must stage D2D into the HoloMQ device provider without touching pageable host memory.
  auto published =
      publish_image_tensor(**session, **binding, fixture.source, std::move(*source_tensor));
  ASSERT_TRUE(published) << runtime_error_message(published.error());
  const std::array selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = std::span<const TemporalSampleRecord>{&published->record, 1U},
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, selection));
  ASSERT_TRUE((*session)->prepare_tensor_access(VertexId{2U}, 1U, consumer_stream.get()));
  void* raw_actual_value = nullptr;
  ASSERT_EQ(cudaMallocHost(&raw_actual_value, sizeof(kExpectedValue)), cudaSuccess);
  const std::unique_ptr<std::uint64_t, test::CudaPinnedHostDeleter> actual_value(
      static_cast<std::uint64_t*>(raw_actual_value));
  *actual_value = 0U;
  {
    auto received = fixture.sink.input.receive();
    ASSERT_TRUE(received) << runtime_error_message(received.error());
    ASSERT_NE(received->data.data, nullptr);
    // Key assertion: the imported Tensor stays on the exact planned CUDA device and remains
    // immutable while consumer-stream ordering protects the D2H verification copy.
    EXPECT_EQ(received->data.data->device().device_type, kDLCUDA);
    EXPECT_EQ(received->data.data->device().device_id, device_ordinal);
    EXPECT_TRUE(received->data.data->is_read_only());
    ASSERT_EQ(cudaMemcpyAsync(actual_value.get(),
                              received->data.data->data(),
                              sizeof(kExpectedValue),
                              cudaMemcpyDeviceToHost,
                              consumer_stream.get()),
              cudaSuccess);

    // Key assertion: S1 cannot finish while S0 is held before its readiness generation record.
    const auto blocked_deadline = std::chrono::steady_clock::now() + 100ms;
    cudaError_t consumer_status = cudaErrorNotReady;
    while (std::chrono::steady_clock::now() < blocked_deadline &&
           consumer_status == cudaErrorNotReady) {
      consumer_status = cudaStreamQuery(consumer_stream.get());
      std::this_thread::yield();
    }
    EXPECT_EQ(consumer_status, cudaErrorNotReady);

    test::release_blocking_host(producer_state);
    producer_release_guard.dismiss();
    ASSERT_EQ(cudaStreamSynchronize(consumer_stream.get()), cudaSuccess);
  }
  EXPECT_EQ(*actual_value, kExpectedValue);

  ASSERT_TRUE((*binding)->retire(published->event.source_sequence));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  ASSERT_TRUE((*session)->retire_samples(
      PortId{2U}, std::span<const TemporalSampleRecord>{&published->record, 1U}));
  (*session)->cancel();
  EXPECT_TRUE((*session)->join());
}

// Descriptor-Tensor schema receive owns the same exact lossy destination while its sidecar claims
// are acquired. A selector cannot claim the incumbent after HoloMQ delivery but before the
// generated value and nested Tensor commit together.
// REQ: RUNTIME-042
// REQ: TENSOR-SCHEMA-005
// REQ: RUNTIME-044
TEST(ChannelBindingParityTest,
     DescriptorTensorSchemaIngressFencesSelectionAcrossSidecarAcquisition) {
  // Key setup: pause a transported Image after descriptor sidecar acquisition while a
  // pre-witness incumbent occupies the only drop-oldest typed destination.
  ImageSessionFixture fixture;
  fixture.base.use_one_slot_blocking_transport();
  fixture.base.use_drop_oldest_queue_policy();
  BlockingTypedIngressObserver observer;
  auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto observer_release =
      ScopeExit{[&]() noexcept { observer.release.store(true, std::memory_order_release); }};
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
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_typed_ingress_observer(
      **session, PortId{2U}, &observer, &BlockingTypedIngressObserver::observe));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_pre_witness_replacement_allowed(
      **session, PortId{2U}, true));
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());

  const auto emit_image = [&](const std::uint64_t value, const std::uint64_t generation) {
    auto owned = make_owned_tensor_sample(value);
    if (!owned) {
      return expected<void, Error>{make_unexpected(std::move(owned).error())};
    }
    schema::ImageT image{};
    image.height = 1;
    image.width = 1;
    image.encoding = schema::ImageEncoding_CUSTOM;
    image.header = std::make_shared<schema::HeaderT>();
    image.header->stamp = std::make_shared<schema::TimeT>();
    image.header->frame_id = "transaction-parity";
    image.data = std::make_shared<Tensor>(std::move(owned->tensor));
    if (auto prepared = (*session)->prepare_activation(VertexId{1U}, generation, {}); !prepared) {
      return expected<void, Error>{make_unexpected(std::move(prepared).error())};
    }
    auto emitted = fixture.source.output.emit(Sample<schema::ImageT>{
        std::move(image),
        SampleMetadata{
            .capture_timestamp_ns = static_cast<std::int64_t>(generation * 10U),
            .source_clock_id = ClockId{1U},
            .trace_id = value,
        }});
    if (!emitted) {
      return emitted;
    }
    return (*session)->retire_activation(VertexId{1U}, generation, true);
  };
  ASSERT_TRUE(emit_image(0x11U, 1U));
  const auto first_deadline = std::chrono::steady_clock::now() + 2s;
  expected<RuntimeChannelInputResetState, Error> first_state =
      make_unexpected(Error{ErrorCode::kNotReady});
  while (std::chrono::steady_clock::now() < first_deadline) {
    first_state = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
    if (first_state && first_state->occupied_samples == 1U) {
      break;
    }
    std::this_thread::yield();
  }
  ASSERT_TRUE(first_state) << runtime_error_message(first_state.error());
  ASSERT_EQ(first_state->occupied_samples, 1U);
  ASSERT_EQ(first_state->next_arrival_sequence, 2U);
  ASSERT_EQ(observer.calls.load(std::memory_order_relaxed), 1U);

  observer.entered.store(false, std::memory_order_release);
  observer.release.store(false, std::memory_order_release);
  observer.block.store(true, std::memory_order_release);
  ASSERT_TRUE(emit_image(0x22U, 2U));
  const auto observer_deadline = std::chrono::steady_clock::now() + 2s;
  while (!observer.entered.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < observer_deadline) {
    std::this_thread::yield();
  }

  // Key assertion: readiness cannot claim the reserved incumbent, and Image metadata plus its
  // nested Tensor commit atomically to that exact destination.
  const auto reservations =
      RuntimeChannelSessionTestAccess::typed_ingress_reservations(**session, PortId{2U});
  const auto retained = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
  const auto blocked_poll = (*binding)->poll(TimePoint{1, ClockId{1U}});
  RuntimeEvent blocked_event;
  const auto blocked_drain = (*binding)->drain(std::span<RuntimeEvent>{&blocked_event, 1U});
  observer.release.store(true, std::memory_order_release);
  ASSERT_TRUE(observer.entered.load(std::memory_order_acquire));
  ASSERT_TRUE(reservations) << runtime_error_message(reservations.error());
  EXPECT_EQ(*reservations, 1U);
  ASSERT_TRUE(retained) << runtime_error_message(retained.error());
  EXPECT_EQ(retained->occupied_samples, 1U);
  EXPECT_EQ(retained->next_arrival_sequence, 2U);
  ASSERT_TRUE(blocked_poll) << runtime_error_message(blocked_poll.error());
  ASSERT_TRUE(blocked_drain) << runtime_error_message(blocked_drain.error());
  EXPECT_EQ(*blocked_drain, 0U);
  EXPECT_TRUE((*binding)->poll_target().has_value());

  auto second_event = wait_for_event(**session, **binding);
  ASSERT_TRUE(second_event) << runtime_error_message(second_event.error());
  auto second_record =
      (*session)->temporal_sample(second_event->port, second_event->source_sequence);
  ASSERT_TRUE(second_record) << runtime_error_message(second_record.error());
  EXPECT_EQ(second_record->source_sequence, 2U);
  EXPECT_EQ(second_record->arrival_sequence, 2U);
  const std::array selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = std::span<const TemporalSampleRecord>{&*second_record, 1U},
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, selection));
  ASSERT_TRUE((*session)->prepare_tensor_access(VertexId{2U}, 1U, nullptr));
  auto received = fixture.sink.input.receive();
  ASSERT_TRUE(received) << runtime_error_message(received.error());
  ASSERT_NE(received->data.data, nullptr);
  ASSERT_NE(received->data.data->data_as<std::uint64_t>(), nullptr);
  EXPECT_EQ(*received->data.data->data_as<std::uint64_t>(), 0x22U);
  EXPECT_EQ(received->metadata.sequence, 2U);
  ASSERT_TRUE((*binding)->retire(second_event->source_sequence));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  ASSERT_TRUE((*session)->retire_samples(
      PortId{2U}, std::span<const TemporalSampleRecord>{&*second_record, 1U}));
  (*session)->cancel();
  EXPECT_TRUE((*session)->join());
}

// Direct retention still passes the serialized value through the complete process-local envelope
// and generated verifier before typed exposure; corrupt frames fail synchronously and retain no
// sample or readiness witness.
// REQ: CHAN-075
// REQ: CHAN-117
TEST(RuntimeChannelBinding, RejectsCorruptedDirectRetainedSchemaEnvelopeBeforeTypedExposure) {
  using channel_internal::CatalogSchemaFrameCorruption;
  constexpr std::array kCorruptions{
      CatalogSchemaFrameCorruption::kMalformedEnvelope,
      CatalogSchemaFrameCorruption::kSchemaId,
      CatalogSchemaFrameCorruption::kSchemaVersion,
      CatalogSchemaFrameCorruption::kEndpointGeneration,
      CatalogSchemaFrameCorruption::kPayloadVerifier,
  };

  for (const auto corruption : kCorruptions) {
    SCOPED_TRACE(static_cast<std::uint32_t>(corruption));
    GeneratedSchemaSessionFixture fixture;
    fixture.base.use_drop_oldest_queue_policy();
    fixture.base.use_direct_retained_delivery();
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
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::corrupt_next_process_local_schema_publication(
        **session, PortId{1U}, corruption));
    ASSERT_TRUE((*session)->refresh_connections());
    ASSERT_TRUE((*session)->arm());
    ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));

    schema::StringT value;
    value.data = "must not escape a malformed direct-retained frame";
    auto emitted = fixture.source.output.emit(std::move(value));
    ASSERT_FALSE(emitted);
    EXPECT_EQ(emitted.error().code, ErrorCode::kProtocolError);
    ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, false));

    // Key assertion: schema failure consumes neither typed storage nor readiness capacity.
    auto retained = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
    ASSERT_TRUE(retained) << runtime_error_message(retained.error());
    EXPECT_EQ(retained->occupied_samples, 0U);
    RuntimeEvent unexpected;
    auto drained = (*binding)->drain(std::span<RuntimeEvent>{&unexpected, 1U});
    ASSERT_TRUE(drained) << runtime_error_message(drained.error());
    EXPECT_EQ(*drained, 0U);

    (*session)->cancel();
    EXPECT_TRUE((*session)->join());
  }
}

// Corrupt each process-local schema-envelope component below the trusted encoder. The production
// transport and typed-retention path must fail closed before publishing a sample or readiness
// witness, including when structurally valid String bytes carry the wrong semantic identity.
// REQ: CHAN-120
// REQ: RUNTIME-020
// REQ: RUNTIME-024
TEST(RuntimeChannelBinding, RejectsCorruptedProcessLocalSchemaEnvelopeBeforeTypedExposure) {
  using channel_internal::CatalogSchemaFrameCorruption;
  constexpr std::array kCorruptions{
      CatalogSchemaFrameCorruption::kMalformedEnvelope,
      CatalogSchemaFrameCorruption::kSchemaId,
      CatalogSchemaFrameCorruption::kSchemaVersion,
      CatalogSchemaFrameCorruption::kEndpointGeneration,
      CatalogSchemaFrameCorruption::kPayloadVerifier,
  };

  for (const auto corruption : kCorruptions) {
    SCOPED_TRACE(static_cast<std::uint32_t>(corruption));
    GeneratedSchemaSessionFixture fixture;
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
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::corrupt_next_process_local_schema_publication(
        **session, PortId{1U}, corruption));
    ASSERT_TRUE((*session)->refresh_connections());
    ASSERT_TRUE((*session)->arm());
    ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));

    schema::StringT value;
    value.data = "must not escape a malformed process-local frame";
    ASSERT_TRUE(fixture.source.output.emit(std::move(value)));
    ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));

    auto event = wait_for_event(**session, **binding);
    ASSERT_FALSE(event);
    EXPECT_EQ(event.error().code, ErrorCode::kProtocolError);
    auto retained = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
    ASSERT_TRUE(retained) << runtime_error_message(retained.error());
    EXPECT_EQ(retained->occupied_samples, 0U);
    RuntimeEvent unexpected;
    auto drained = (*binding)->drain(std::span<RuntimeEvent>{&unexpected, 1U});
    ASSERT_TRUE(drained) << runtime_error_message(drained.error());
    EXPECT_EQ(*drained, 0U);

    (*session)->cancel();
    auto joined = (*session)->join();
    ASSERT_FALSE(joined);
    EXPECT_EQ(joined.error().code, ErrorCode::kProtocolError);
  }
}
}  // namespace
}  // namespace holoscan::detail

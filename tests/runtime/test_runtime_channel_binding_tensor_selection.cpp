/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/graph.hpp>
#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_state.hpp"
#include "core/source_id_internal.hpp"
#include "core/temporal_contract_internal.hpp"
#include "runtime/dispatcher.hpp"
#include "runtime/predicate_evaluator.hpp"
#include "runtime/runtime_channel_binding.hpp"

#include "runtime_channel_binding_schema_tensor_test_support.hpp"
#include "runtime_channel_binding_test_support.hpp"

namespace holoscan::detail {
namespace {

using namespace std::chrono_literals;
using namespace runtime_channel_binding_test;
using namespace runtime_channel_binding_schema_tensor_test;

// REQ: CHAN-083
// REQ: TENSOR-STORAGE-010
TEST(ChannelBindingParityTest,
     TensorLatestWithinAndReuseLastPreserveBackingIdentityAndReadOnlyClaims) {
  {
    TensorSessionFixture fixture;
    fixture.enable_latest_within();
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

    constexpr std::array<std::uint64_t, 3U> values{11U, 22U, 33U};
    constexpr std::array<std::int64_t, 3U> capture_times{10, 20, 30};
    auto script =
        publish_tensor_script(**session, **binding, fixture.source, values, capture_times);
    ASSERT_TRUE(script) << runtime_error_message(script.error());

    auto direct = RuntimeInputSelectionState::create(fixture.base.plan.input_selections.front());
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
    ASSERT_TRUE((*session)->prepare_tensor_access(VertexId{2U}, 1U, nullptr));
    {
      auto first = fixture.sink.input.receive();
      auto repeated = fixture.sink.input.receive();
      ASSERT_TRUE(first);
      ASSERT_TRUE(repeated);
      EXPECT_EQ(first->data.data(), script->data[1]);
      EXPECT_EQ(repeated->data.data(), script->data[1]);
      EXPECT_EQ(first->data.data(), repeated->data.data());
      ASSERT_NE(first->data.data_as<std::uint64_t>(), nullptr);
      EXPECT_EQ(*first->data.data_as<std::uint64_t>(), values[1]);
      auto mutable_pointer = first->data.mutable_data_as<std::uint64_t>();
      ASSERT_FALSE(mutable_pointer);
      EXPECT_EQ(mutable_pointer.error().code, ErrorCode::kPermissionDenied);
    }
    ASSERT_TRUE((*binding)->retire(script->events[1].source_sequence));
    ASSERT_TRUE(selected->finish(true));
    ASSERT_TRUE((*session)->retire_samples(
        PortId{2U}, std::span<const TemporalSampleRecord>{script->records}.subspan(1U, 1U)));
    ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));

    const std::array unselected_records{script->records[0], script->records[2]};
    ASSERT_TRUE((*binding)->retire(script->events[0].source_sequence));
    ASSERT_TRUE((*binding)->retire(script->events[2].source_sequence));
    ASSERT_TRUE((*session)->retire_samples(PortId{2U}, unselected_records));
    (*session)->cancel();
    ASSERT_TRUE((*session)->join());
    for (const auto& releases : script->releases) {
      EXPECT_EQ(releases->load(std::memory_order_relaxed), 1U);
    }
  }

  TensorSessionFixture fixture;
  fixture.enable_reuse_last();
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

  constexpr std::array<std::uint64_t, 1U> values{55U};
  constexpr std::array<std::int64_t, 1U> capture_times{10};
  auto script = publish_tensor_script(**session, **binding, fixture.source, values, capture_times);
  ASSERT_TRUE(script) << runtime_error_message(script.error());
  auto direct = RuntimeInputSelectionState::create(fixture.base.plan.input_selections.front());
  ASSERT_TRUE(direct);
  ASSERT_TRUE(direct->push(script->records.front()));

  auto fresh = direct->begin(script->records.front().capture_time,
                             script->records.front().arrival_sequence,
                             script->records.front().producer_epoch);
  ASSERT_TRUE(fresh);
  DLManagedTensorVersioned* exported = nullptr;
  const std::array fresh_selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = fresh->selection(),
      .samples = fresh->selected(),
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, fresh_selection));
  ASSERT_TRUE((*session)->prepare_tensor_access(VertexId{2U}, 1U, nullptr));
  {
    auto sample = fixture.sink.input.receive();
    ASSERT_TRUE(sample);
    EXPECT_EQ(sample->data.data(), script->data.front());
    auto dlpack = sample->data.to_dlpack_versioned();
    ASSERT_TRUE(dlpack) << runtime_error_message(dlpack.error());
    exported = *dlpack;
    ASSERT_NE(exported, nullptr);
    EXPECT_NE(exported->flags & DLPACK_FLAG_BITMASK_READ_ONLY, 0U);
  }
  ASSERT_TRUE((*binding)->retire(script->events.front().source_sequence));
  ASSERT_TRUE(fresh->finish(true));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));

  auto reused = direct->begin(TimePoint{20, ClockId{1U}},
                              script->records.front().arrival_sequence,
                              script->records.front().producer_epoch);
  ASSERT_TRUE(reused);
  EXPECT_EQ(reused->selection().state, InputState::kReused);
  const std::array reused_selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = reused->selection(),
      .samples = reused->selected(),
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 2U, reused_selection));
  ASSERT_TRUE((*session)->prepare_tensor_access(VertexId{2U}, 2U, nullptr));
  {
    auto sample = fixture.sink.input.receive();
    ASSERT_TRUE(sample);
    EXPECT_EQ(sample->data.data(), script->data.front());
    EXPECT_EQ(*sample->data.data_as<std::uint64_t>(), values.front());
  }
  ASSERT_TRUE(reused->finish(true));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 2U, true));
  auto retained = (*session)->retire_samples(PortId{2U}, script->records);
  ASSERT_FALSE(retained);
  EXPECT_EQ(retained.error().code, ErrorCode::kBackpressured);
  EXPECT_EQ(script->releases.front()->load(std::memory_order_relaxed), 0U);

  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
  ASSERT_TRUE((*session)->reset(VertexId{2U}));
  // The Runtime read/reuse claims are gone, but the external DLPack owner remains an independent
  // backing claim. Pool/storage retirement must wait for that owner rather than treating the
  // activation-scoped read lease as the final authority.
  EXPECT_EQ(script->releases.front()->load(std::memory_order_relaxed), 0U);
  ASSERT_NE(exported, nullptr);
  ASSERT_NE(exported->deleter, nullptr);
  exported->deleter(exported);
  EXPECT_EQ(script->releases.front()->load(std::memory_order_relaxed), 1U);
}

// ImageT uses the same temporal selection and activation-scoped Tensor read authority as a direct
// Tensor while its generated metadata remains one coherent sample. Predicate and compute views
// must observe the same retained backing for both fresh and ReuseLast selections.
// REQ: CHAN-084
// REQ: CHAN-119
// REQ: TENSOR-STORAGE-010
// REQ: TENSOR-SCHEMA-005
// REQ: OPERATOR-035
// REQ: RUNTIME-042
// REQ: RUNTIME-044
TEST(ChannelBindingParityTest,
     DescriptorTensorImagePreservesMetadataIdentityPredicateAndReuseClaims) {
  ImageSessionFixture fixture;
  fixture.enable_reuse_last();
  fixture.base.use_drop_oldest_queue_policy();
  // Key setup: the strict 1:1 kDropOldest edge retains verified Image metadata and its embedded
  // Tensor directly instead of publishing a descriptor token through HoloMQ.
  fixture.use_direct_retained_delivery();
  ASSERT_NE(fixture.replay[0].type_ops, nullptr);
  ASSERT_EQ(fixture.replay[0].type_ops, fixture.replay[1].type_ops);
  EXPECT_EQ(fixture.replay[0].type_ops->codec_family,
            RuntimePayloadCodecFamily::kDescriptorTensorSchema);
  EXPECT_TRUE(fixture.replay[0].type_ops->persistent_sample_storage);
  EXPECT_GT(fixture.replay[0].type_ops->wire_capacity, 0U);
  EXPECT_TRUE(fixture.base.plan.process_local_channel_services.front().tensor_payload);
  EXPECT_TRUE(fixture.base.plan.process_local_channel_services.front().tensor_pool_id.empty());

  BlockingTypedIngressObserver transport_observer;
  auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_typed_ingress_observer(
      **session, PortId{2U}, &transport_observer, &BlockingTypedIngressObserver::observe));
  auto catalog_contract = RuntimeChannelSessionTestAccess::set_external_tensor_contract(
      **session,
      PortId{2U},
      LogicalTensorRequirementsRecord{
          .memory_kind = MemoryKind::kHost,
          .max_byte_span = 1U,
          .strides = TensorStridePolicy::kContiguousOnly,
      });
  ASSERT_FALSE(catalog_contract);
  EXPECT_EQ(catalog_contract.error().code, ErrorCode::kInvalidArgument);
  const auto indexes = RuntimeChannelSessionTestAccess::pump_index_state(**session);
  ASSERT_TRUE(indexes) << runtime_error_message(indexes.error());
  // Key assertion: direct descriptor-schema retention constructs no wait attachment or pump work.
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

  constexpr std::uint64_t kExpectedValue = 0x1234'5678U;
  auto published = publish_image_sample(**session, **binding, fixture.source, kExpectedValue);
  ASSERT_TRUE(published) << runtime_error_message(published.error());
  EXPECT_EQ(transport_observer.calls.load(std::memory_order_relaxed), 0U);
  const auto direct_reservations =
      RuntimeChannelSessionTestAccess::typed_ingress_reservations(**session, PortId{2U});
  ASSERT_TRUE(direct_reservations) << runtime_error_message(direct_reservations.error());
  EXPECT_EQ(*direct_reservations, 0U);
  auto direct = RuntimeInputSelectionState::create(fixture.base.plan.input_selections.front());
  ASSERT_TRUE(direct);
  ASSERT_TRUE(direct->push(published->record));

  const auto exercise = [&](RuntimeInputSelectionState::Transaction& transaction,
                            std::uint64_t generation) {
    const std::array selection{RuntimeChannelActivationSelection{
        .port = PortId{2U},
        .selection = transaction.selection(),
        .samples = transaction.selected(),
        .destructive = true,
    }};
    ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, generation, selection));
    ASSERT_TRUE((*session)->prepare_tensor_access(VertexId{2U}, generation, nullptr));
    auto raw_view = (*session)->predicate_view(PortId{2U}, generation);
    ASSERT_TRUE(raw_view) << runtime_error_message(raw_view.error());
    const auto& predicate =
        *static_cast<const PredicateInputView<schema::ImageT>*>(raw_view.value());
    auto received = fixture.sink.input.receive();
    ASSERT_TRUE(received) << runtime_error_message(received.error());
    ASSERT_NE(predicate.sample().data.data, nullptr);
    ASSERT_NE(received->data.data, nullptr);
    EXPECT_EQ(predicate.sample().data.data->data(), published->data);
    EXPECT_EQ(received->data.data->data(), published->data);
    EXPECT_EQ(predicate.sample().data.data->data(), received->data.data->data());
    ASSERT_NE(received->data.data->data_as<std::uint64_t>(), nullptr);
    EXPECT_EQ(*received->data.data->data_as<std::uint64_t>(), kExpectedValue);

    ASSERT_NE(received->data.header, nullptr);
    ASSERT_NE(received->data.header->stamp, nullptr);
    EXPECT_EQ(received->data.height, 1);
    EXPECT_EQ(received->data.width, 1);
    EXPECT_EQ(received->data.encoding, schema::ImageEncoding_CUSTOM);
    EXPECT_EQ(received->data.header->stamp->time_ns, 123'456);
    EXPECT_EQ(received->data.header->stamp->acqtime_ns, 123'450);
    EXPECT_EQ(received->data.header->frame_id, "runtime_image");
    EXPECT_EQ(received->metadata.sequence, published->record.source_sequence);
    EXPECT_EQ(received->metadata.capture_timestamp_ns, 10);
    EXPECT_EQ(received->metadata.source_clock_id, ClockId{1U});
    EXPECT_EQ(received->metadata.trace_id, 0xABCDU);
    EXPECT_EQ(received->metadata.frame_id, 73U);
    EXPECT_EQ(received->metadata.byte_size, sizeof(std::uint64_t));
    EXPECT_EQ(received->metadata.payload_kind, PayloadKind::kDescriptorBacked);
    EXPECT_EQ(predicate.selection().state, transaction.selection().state);
    EXPECT_EQ(predicate.selection().reason, transaction.selection().reason);

    auto mutable_pointer = received->data.data->mutable_data_as<std::uint64_t>();
    ASSERT_FALSE(mutable_pointer);
    EXPECT_EQ(mutable_pointer.error().code, ErrorCode::kPermissionDenied);
  };

  auto fresh = direct->begin(published->record.capture_time,
                             published->record.arrival_sequence,
                             published->record.producer_epoch);
  ASSERT_TRUE(fresh);
  ASSERT_EQ(fresh->selection().state, InputState::kSelected);
  exercise(*fresh, 1U);
  ASSERT_TRUE((*binding)->retire(published->event.source_sequence));
  ASSERT_TRUE(fresh->finish(true));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));

  auto reused = direct->begin(TimePoint{20, ClockId{1U}},
                              published->record.arrival_sequence,
                              published->record.producer_epoch);
  ASSERT_TRUE(reused);
  ASSERT_EQ(reused->selection().state, InputState::kReused);
  exercise(*reused, 2U);
  ASSERT_TRUE(reused->finish(true));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 2U, true));
  auto retained = (*session)->retire_samples(
      PortId{2U}, std::span<const TemporalSampleRecord>{&published->record, 1U});
  ASSERT_FALSE(retained);
  EXPECT_EQ(retained.error().code, ErrorCode::kBackpressured);
  EXPECT_EQ(published->releases->load(std::memory_order_relaxed), 0U);

  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
  ASSERT_TRUE((*session)->reset(VertexId{2U}));
  EXPECT_EQ(published->releases->load(std::memory_order_relaxed), 1U);
}

// A Tensor-bearing generated value uses the same non-destructive batch and overlap authority as a
// direct Tensor payload. Reconstructing the owning Image envelope must neither copy its embedded
// Tensor nor shorten the backing lifetime shared by adjacent windows.
// REQ: CHAN-084
// REQ: TENSOR-STORAGE-010
// REQ: TENSOR-SCHEMA-005
TEST(ChannelBindingParityTest,
     DescriptorTensorImageBracketedWindowsPreserveSelectedBatchAndOverlappingBackingIdentity) {
  ImageSessionFixture fixture;
  fixture.enable_bracketed_window(5U);
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

  constexpr std::array<std::uint64_t, 7U> values{5U, 10U, 15U, 20U, 25U, 30U, 35U};
  constexpr std::array<std::int64_t, 7U> capture_times{5, 10, 15, 20, 25, 30, 35};
  auto script = publish_image_script(**session, **binding, fixture.source, values, capture_times);
  ASSERT_TRUE(script) << runtime_error_message(script.error());
  auto direct = RuntimeInputSelectionState::create(
      fixture.base.plan.input_selections.front(), &fixture.base.plan.bracketed_windows.front(), 2U);
  ASSERT_TRUE(direct) << runtime_error_message(direct.error());

  ASSERT_TRUE(direct->push(script->records[0]));
  ASSERT_TRUE(direct->push_anchor(script->records[1]));
  ASSERT_TRUE(direct->push(script->records[1]));
  ASSERT_TRUE(direct->push(script->records[2]));
  ASSERT_TRUE(direct->push_anchor(script->records[3]));
  ASSERT_TRUE(direct->push(script->records[3]));
  ASSERT_TRUE(direct->push(script->records[4]));
  ASSERT_TRUE(direct->push_anchor(script->records[5]));
  ASSERT_TRUE(direct->push(script->records[5]));
  ASSERT_TRUE(direct->push(script->records[6]));

  auto first = direct->begin(TimePoint{20, ClockId{1U}},
                             script->records[4].arrival_sequence,
                             script->records[4].producer_epoch);
  ASSERT_TRUE(first) << runtime_error_message(first.error());
  ASSERT_EQ(first->selected().size(), 5U);
  const std::array first_selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = first->selection(),
      .samples = first->selected(),
      .batch = true,
      .destructive = false,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, first_selection));
  ASSERT_TRUE((*session)->prepare_tensor_access(VertexId{2U}, 1U, nullptr));
  std::array<const void*, 5U> first_data{};
  {
    auto batch = fixture.sink.input.selected_batch();
    auto repeated = fixture.sink.input.selected_batch();
    ASSERT_TRUE(batch) << runtime_error_message(batch.error());
    ASSERT_TRUE(repeated) << runtime_error_message(repeated.error());
    ASSERT_EQ(batch->size(), first_data.size());
    ASSERT_EQ(repeated->size(), first_data.size());
    for (std::size_t index = 0U; index < first_data.size(); ++index) {
      ASSERT_NE((*batch)[index].data.data, nullptr);
      ASSERT_NE((*repeated)[index].data.data, nullptr);
      first_data[index] = (*batch)[index].data.data->data();
      EXPECT_EQ(first_data[index], script->data[index]);
      EXPECT_EQ((*repeated)[index].data.data->data(), first_data[index]);
      EXPECT_EQ(&(*batch)[index], &(*repeated)[index]);
      EXPECT_EQ((*batch)[index].metadata.capture_timestamp_ns, capture_times[index]);
      EXPECT_EQ((*batch)[index].data.header->frame_id, "runtime_image");
    }
  }
  ASSERT_TRUE(first->finish(true));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  for (const auto& releases : script->releases) {
    EXPECT_EQ(releases->load(std::memory_order_relaxed), 0U);
  }

  auto second = direct->begin(TimePoint{30, ClockId{1U}},
                              script->records[6].arrival_sequence,
                              script->records[6].producer_epoch);
  ASSERT_TRUE(second) << runtime_error_message(second.error());
  ASSERT_EQ(second->selected().size(), 5U);
  const std::array second_selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = second->selection(),
      .samples = second->selected(),
      .batch = true,
      .destructive = false,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 2U, second_selection));
  ASSERT_TRUE((*session)->prepare_tensor_access(VertexId{2U}, 2U, nullptr));
  {
    auto batch = fixture.sink.input.selected_batch();
    ASSERT_TRUE(batch) << runtime_error_message(batch.error());
    ASSERT_EQ(batch->size(), 5U);
    for (std::size_t index = 0U; index < batch->size(); ++index) {
      ASSERT_NE((*batch)[index].data.data, nullptr);
      EXPECT_EQ((*batch)[index].data.data->data(), script->data[index + 2U]);
      EXPECT_EQ((*batch)[index].metadata.capture_timestamp_ns, capture_times[index + 2U]);
    }
    EXPECT_EQ((*batch)[0].data.data->data(), first_data[2]);
    EXPECT_EQ((*batch)[1].data.data->data(), first_data[3]);
    EXPECT_EQ((*batch)[2].data.data->data(), first_data[4]);
  }
  ASSERT_TRUE(second->finish(true));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 2U, true));

  ASSERT_TRUE(direct->reset());
  for (const auto& event : script->events) {
    ASSERT_TRUE((*binding)->retire(event.source_sequence));
  }
  ASSERT_TRUE((*session)->retire_samples(PortId{2U}, script->records));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
  for (const auto& releases : script->releases) {
    EXPECT_EQ(releases->load(std::memory_order_relaxed), 1U);
  }
}

// The generated Image facade must not bypass CUDA ordering on its embedded Tensor. This test uses
// a stream-capable device-memory reference so the Runtime path can be verified without invoking a
// CUDA API on a synthetic stream handle; the device functional event path remains covered by the
// Tensor device suite.
// REQ: CHAN-084
// REQ: CHAN-119
// REQ: TENSOR-STORAGE-010
// REQ: TENSOR-SCHEMA-005
TEST(ChannelBindingParityTest,
     DescriptorTensorImageConsumerStreamRetainsBackingUntilFinalClaimRetires) {
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
                                               .capacity = 8U,
                                           });
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());

  auto tracked = make_stream_tracking_tensor();
  const void* const published_data = tracked.data;
  auto tracking = tracked.state;
  schema::ImageT image{};
  image.height = 1;
  image.width = 1;
  image.encoding = schema::ImageEncoding_CUSTOM;
  image.header = std::make_shared<schema::HeaderT>();
  image.header->frame_id = "device_image";
  image.data = std::make_shared<Tensor>(std::move(tracked.tensor));
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(Sample<schema::ImageT>{
      std::move(image),
      SampleMetadata{.capture_timestamp_ns = 10, .source_clock_id = ClockId{1U}},
  }));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));

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
  auto* const consumer_stream = reinterpret_cast<cudaStream_t>(std::uintptr_t{0xC0FFEEU});
  ASSERT_TRUE((*session)->prepare_tensor_access(VertexId{2U}, 1U, consumer_stream));
  ASSERT_TRUE((*session)->prepare_tensor_access(VertexId{2U}, 1U, consumer_stream));
  EXPECT_EQ(tracking->consumer_syncs.load(std::memory_order_relaxed), 1U);
  EXPECT_EQ(tracking->deallocation_streams.load(std::memory_order_relaxed), 1U);
  EXPECT_EQ(tracking->last_consumer_stream, consumer_stream);
  EXPECT_EQ(tracking->last_deallocation_stream, consumer_stream);
  {
    auto received = fixture.sink.input.receive();
    ASSERT_TRUE(received) << runtime_error_message(received.error());
    ASSERT_NE(received->data.data, nullptr);
    EXPECT_EQ(received->data.data->data(), published_data);
    EXPECT_EQ(received->data.data->device().device_type, kDLCUDA);
    auto mutable_pointer = received->data.data->mutable_data();
    ASSERT_FALSE(mutable_pointer);
    EXPECT_EQ(mutable_pointer.error().code, ErrorCode::kPermissionDenied);
  }
  EXPECT_GT(tracking->live_handles.load(std::memory_order_relaxed), 0U);
  ASSERT_TRUE((*binding)->retire(event->source_sequence));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  ASSERT_TRUE(
      (*session)->retire_samples(PortId{2U}, std::span<const TemporalSampleRecord>{&*record, 1U}));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
  ASSERT_TRUE((*session)->reset(VertexId{2U}));
  EXPECT_EQ(tracking->live_handles.load(std::memory_order_relaxed), 0U);
  EXPECT_EQ(tracking->destroyed.load(std::memory_order_relaxed), 1U);
}

// REQ: CHAN-083
// REQ: TENSOR-STORAGE-010
TEST(ChannelBindingParityTest, TensorPredicateViewAndComputeShareFreshAndReusedBacking) {
  TensorSessionFixture fixture;
  fixture.enable_reuse_last();
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

  constexpr std::array<std::uint64_t, 1U> values{71U};
  constexpr std::array<std::int64_t, 1U> capture_times{10};
  auto script = publish_tensor_script(**session, **binding, fixture.source, values, capture_times);
  ASSERT_TRUE(script) << runtime_error_message(script.error());
  auto direct = RuntimeInputSelectionState::create(fixture.base.plan.input_selections.front());
  ASSERT_TRUE(direct);
  ASSERT_TRUE(direct->push(script->records.front()));

  auto false_binding = predicate(
      "tensor-false",
      [](const PredicateEvaluationContext&, const PredicateInputView<Tensor>& input) noexcept {
        static_cast<void>(input.sample().data.data());
        return false;
      });
  auto false_storage = ContractTestAccess::clone_predicate_storage(false_binding);
  ASSERT_TRUE(false_storage);
  PredicateEvaluatorEngine<1U, 1U> false_evaluator{
      std::move(*false_storage), true, 0U, PredicateFiring::kOnEveryTrueEvaluation, 1U};

  auto error_binding =
      predicate("tensor-error",
                [](const PredicateEvaluationContext&,
                   const PredicateInputView<Tensor>& input) noexcept -> expected<bool, Error> {
                  static_cast<void>(input.sample().data.data());
                  return make_unexpected(Error{ErrorCode::kFailure});
                });
  auto error_storage = ContractTestAccess::clone_predicate_storage(error_binding);
  ASSERT_TRUE(error_storage);
  PredicateEvaluatorEngine<1U, 1U> error_evaluator{
      std::move(*error_storage), true, 0U, PredicateFiring::kOnEveryTrueEvaluation, 1U};

  auto true_binding = predicate(
      "tensor-true",
      [](const PredicateEvaluationContext&, const PredicateInputView<Tensor>& input) noexcept {
        return input.sample().data.data() != nullptr;
      });
  auto true_storage = ContractTestAccess::clone_predicate_storage(true_binding);
  ASSERT_TRUE(true_storage);
  PredicateEvaluatorEngine<1U, 1U> true_evaluator{
      std::move(*true_storage), true, 0U, PredicateFiring::kOnEveryTrueEvaluation, 1U};

  auto exercise = [&](auto& evaluator,
                      RuntimeInputSelectionState::Transaction& transaction,
                      std::uint64_t generation,
                      PredicateComputeSelectionRequest compute,
                      PredicateCandidateDisposition expected_disposition,
                      bool expected_firing) {
    const std::array selection{RuntimeChannelActivationSelection{
        .port = PortId{2U},
        .selection = transaction.selection(),
        .samples = transaction.selected(),
        .destructive = true,
    }};
    ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, generation, selection));
    ASSERT_TRUE((*session)->prepare_tensor_access(VertexId{2U}, generation, nullptr));
    auto raw_view = (*session)->predicate_view(PortId{2U}, generation);
    ASSERT_TRUE(raw_view) << runtime_error_message(raw_view.error());
    const auto& predicate = *static_cast<const PredicateInputView<Tensor>*>(raw_view.value());
    EXPECT_EQ(predicate.sample().data.data(), script->data.front());
    EXPECT_EQ(predicate.selection().state, transaction.selection().state);
    EXPECT_EQ(predicate.selection().reason, transaction.selection().reason);

    const std::array inputs{PredicateFrozenInput{
        .port = PortId{2U},
        .selection = transaction.selection(),
        .typed_view = &predicate,
    }};
    auto decision = evaluator.evaluate(
        PredicateFrozenSource{
            .reference_time = TimePoint{static_cast<std::int64_t>(generation), ClockId{1U}},
            .tag = Tag{.sequence = generation, .generation = 1U},
            .tag_time = TagTime{.logical_ns = static_cast<std::int64_t>(generation)},
        },
        inputs,
        compute);
    ASSERT_TRUE(decision) << runtime_error_message(decision.error());
    EXPECT_EQ(decision->disposition, expected_disposition);
    EXPECT_TRUE(decision->evaluator_called);
    EXPECT_EQ(decision->firing_accepted, expected_firing);
    ASSERT_EQ(decision->predicate_views.size(), 1U);
    EXPECT_EQ(decision->predicate_views.front(), &predicate);

    if (expected_disposition == PredicateCandidateDisposition::kEvaluatorError) {
      EXPECT_EQ(decision->evaluator_error.code, ErrorCode::kFailure);
    }
    if (expected_firing) {
      auto compute_sample = fixture.sink.input.receive();
      ASSERT_TRUE(compute_sample);
      EXPECT_EQ(compute_sample->data.data(), script->data.front());
      EXPECT_EQ(predicate.sample().data.data(), compute_sample->data.data());
    }
  };

  // Each rejection below executes its real decision path. The provisional Tensor read lease must
  // disappear while the selector transaction rolls back for an exact retry.
  auto predicate_false = direct->begin(script->records.front().capture_time,
                                       script->records.front().arrival_sequence,
                                       script->records.front().producer_epoch);
  ASSERT_TRUE(predicate_false);
  exercise(
      false_evaluator, *predicate_false, 1U, {}, PredicateCandidateDisposition::kNoFire, false);
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, false));
  ASSERT_TRUE(predicate_false->finish(false));
  EXPECT_EQ(script->releases.front()->load(std::memory_order_relaxed), 0U);

  auto evaluator_error = direct->begin(script->records.front().capture_time,
                                       script->records.front().arrival_sequence,
                                       script->records.front().producer_epoch);
  ASSERT_TRUE(evaluator_error);
  exercise(error_evaluator,
           *evaluator_error,
           2U,
           {},
           PredicateCandidateDisposition::kEvaluatorError,
           false);
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 2U, false));
  ASSERT_TRUE(evaluator_error->finish(false));
  EXPECT_EQ(script->releases.front()->load(std::memory_order_relaxed), 0U);

  InputSelectionPlanRecord missing_plan = fixture.base.plan.input_selections.front();
  missing_plan.input = PortId{3U};
  missing_plan.missing = TemporalRecordMissingKind::kSkip;
  missing_plan.reuse_max_age_ns = 0;
  auto skip_selector = RuntimeInputSelectionState::create(missing_plan);
  ASSERT_TRUE(skip_selector);
  auto skipped = skip_selector->begin(script->records.front().capture_time,
                                      script->records.front().arrival_sequence,
                                      script->records.front().producer_epoch);
  ASSERT_TRUE(skipped);
  ASSERT_EQ(skipped->selection().state, InputState::kAbsent);
  ASSERT_EQ(skipped->selection().reason, SelectionReason::kNeverSeen);
  PredicateComputeSelectionFixture<1U> skip_compute;
  skip_compute.selections.front() = skipped->selection();
  auto predicate_skip = direct->begin(script->records.front().capture_time,
                                      script->records.front().arrival_sequence,
                                      script->records.front().producer_epoch);
  ASSERT_TRUE(predicate_skip);
  exercise(true_evaluator,
           *predicate_skip,
           3U,
           skip_compute.request(),
           PredicateCandidateDisposition::kNoFire,
           false);
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 3U, false));
  ASSERT_TRUE(predicate_skip->finish(false));
  ASSERT_TRUE(skipped->finish(false));
  EXPECT_EQ(script->releases.front()->load(std::memory_order_relaxed), 0U);

  missing_plan.missing = TemporalRecordMissingKind::kSafeDefault;
  auto safe_default_selector = RuntimeInputSelectionState::create(missing_plan);
  ASSERT_TRUE(safe_default_selector);
  auto safe_default = safe_default_selector->begin(script->records.front().capture_time,
                                                   script->records.front().arrival_sequence,
                                                   script->records.front().producer_epoch);
  ASSERT_TRUE(safe_default);
  ASSERT_EQ(safe_default->selection().state, InputState::kSafeDefault);
  ASSERT_EQ(safe_default->selection().reason, SelectionReason::kNeverSeen);
  PredicateComputeSelectionFixture<1U> safe_default_compute;
  safe_default_compute.selections.front() = safe_default->selection();
  auto fresh = direct->begin(script->records.front().capture_time,
                             script->records.front().arrival_sequence,
                             script->records.front().producer_epoch);
  ASSERT_TRUE(fresh);
  exercise(true_evaluator,
           *fresh,
           4U,
           safe_default_compute.request(),
           PredicateCandidateDisposition::kDegradedCompute,
           true);
  ASSERT_TRUE((*binding)->retire(script->events.front().source_sequence));
  ASSERT_TRUE(fresh->finish(true));
  ASSERT_TRUE(safe_default->finish(true));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 4U, true));

  auto reused = direct->begin(TimePoint{20, ClockId{1U}},
                              script->records.front().arrival_sequence,
                              script->records.front().producer_epoch);
  ASSERT_TRUE(reused);
  ASSERT_EQ(reused->selection().state, InputState::kReused);
  exercise(true_evaluator, *reused, 5U, {}, PredicateCandidateDisposition::kNominalCompute, true);
  ASSERT_TRUE(reused->finish(true));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 5U, true));

  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
  ASSERT_TRUE((*session)->reset(VertexId{2U}));
  EXPECT_EQ(script->releases.front()->load(std::memory_order_relaxed), 1U);
}

// REQ: CHAN-083
// REQ: TENSOR-STORAGE-010
TEST(ChannelBindingParityTest, TensorDrainAllRollbackAndCommitPreserveBatchOrderIdentityAndClaims) {
  TensorSessionFixture fixture;
  fixture.enable_drain_all(3U);
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

  constexpr std::array<std::uint64_t, 3U> values{10U, 20U, 30U};
  constexpr std::array<std::int64_t, 3U> capture_times{10, 20, 30};
  auto script = publish_tensor_script(**session, **binding, fixture.source, values, capture_times);
  ASSERT_TRUE(script) << runtime_error_message(script.error());
  auto direct = RuntimeInputSelectionState::create(fixture.base.plan.input_selections.front());
  ASSERT_TRUE(direct);
  for (const auto& record : script->records) {
    ASSERT_TRUE(direct->push(record));
  }

  auto provisional = direct->begin(TimePoint{30, ClockId{1U}},
                                   script->records.back().arrival_sequence,
                                   script->records.back().producer_epoch);
  ASSERT_TRUE(provisional);
  const std::array provisional_selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = provisional->selection(),
      .samples = provisional->selected(),
      .batch = true,
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, provisional_selection));
  ASSERT_TRUE((*session)->prepare_tensor_access(VertexId{2U}, 1U, nullptr));
  std::optional<SelectedBatchView<Tensor>> retired_view;
  std::optional<SelectedBatchView<Tensor>::const_iterator> retired_iterator;
  {
    auto first = fixture.sink.input.selected_batch();
    auto repeated = fixture.sink.input.selected_batch();
    ASSERT_TRUE(first);
    ASSERT_TRUE(repeated);
    retired_view.emplace(*first);
    retired_iterator.emplace(retired_view->begin());
    ASSERT_EQ(first->size(), values.size());
    ASSERT_EQ(repeated->size(), values.size());
    for (std::size_t index = 0U; index < values.size(); ++index) {
      EXPECT_EQ((*first)[index].data.data(), script->data[index]);
      EXPECT_EQ((*repeated)[index].data.data(), script->data[index]);
      EXPECT_EQ(&(*first)[index], &(*repeated)[index]);
      EXPECT_EQ(*(*first)[index].data.data_as<std::uint64_t>(), values[index]);
    }
  }
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, false));
  ASSERT_TRUE(provisional->finish(false));
#if !defined(NDEBUG) && GTEST_HAS_DEATH_TEST
  ASSERT_TRUE(retired_view.has_value());
  EXPECT_DEATH(
      {
        const auto* value = retired_view->front().data.data();
        (void)value;
      },
      "");
  ASSERT_TRUE(retired_iterator.has_value());
  EXPECT_DEATH(
      {
        const auto* value = (**retired_iterator).data.data();
        (void)value;
      },
      "");
#endif
  retired_iterator.reset();
  retired_view.reset();
  EXPECT_EQ(direct->retained_count(), values.size());
  for (const auto& releases : script->releases) {
    EXPECT_EQ(releases->load(std::memory_order_relaxed), 0U);
  }

  auto accepted = direct->begin(TimePoint{30, ClockId{1U}},
                                script->records.back().arrival_sequence,
                                script->records.back().producer_epoch);
  ASSERT_TRUE(accepted);
  const std::array accepted_selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = accepted->selection(),
      .samples = accepted->selected(),
      .batch = true,
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 2U, accepted_selection));
  ASSERT_TRUE((*session)->prepare_tensor_access(VertexId{2U}, 2U, nullptr));
  {
    auto batch = fixture.sink.input.selected_batch();
    ASSERT_TRUE(batch);
    ASSERT_EQ(batch->size(), values.size());
    for (std::size_t index = 0U; index < values.size(); ++index) {
      EXPECT_EQ((*batch)[index].data.data(), script->data[index]);
    }
  }
  for (const auto& event : script->events) {
    ASSERT_TRUE((*binding)->retire(event.source_sequence));
  }
  ASSERT_TRUE(accepted->finish(true));
  ASSERT_TRUE((*session)->retire_samples(PortId{2U}, script->records));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 2U, true));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
  for (const auto& releases : script->releases) {
    EXPECT_EQ(releases->load(std::memory_order_relaxed), 1U);
  }
}

// Every Tensor in a selected batch receives the same activation-scoped consumer-stream contract.
// Releasing the readiness witness and selector claim first must not destroy any backing while the
// activation's read/CUDA claim is still live; the final activation retirement releases each one.
// REQ: CHAN-083
// REQ: TENSOR-STORAGE-010
TEST(ChannelBindingParityTest,
     TensorSelectedBatchConsumerStreamRetainsBackingUntilActivationRetires) {
  TensorSessionFixture fixture;
  fixture.enable_drain_all(3U);
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

  constexpr std::size_t kSampleCount = 3U;
  std::array<const void*, kSampleCount> published_data{};
  std::array<std::shared_ptr<StreamTrackingState>, kSampleCount> tracking{};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  for (std::size_t index = 0U; index < kSampleCount; ++index) {
    auto tensor = make_stream_tracking_tensor();
    published_data[index] = tensor.data;
    tracking[index] = tensor.state;
    ASSERT_TRUE(fixture.source.output.emit(Sample<Tensor>{
        std::move(tensor.tensor),
        SampleMetadata{.capture_timestamp_ns = static_cast<std::int64_t>(index + 1U),
                       .source_clock_id = ClockId{1U}},
    }));
  }
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));

  std::array<RuntimeEvent, kSampleCount> events{};
  std::array<TemporalSampleRecord, kSampleCount> records{};
  for (std::size_t index = 0U; index < kSampleCount; ++index) {
    auto event = wait_for_event(**session, **binding);
    ASSERT_TRUE(event) << runtime_error_message(event.error());
    events[index] = *event;
    auto record = (*session)->temporal_sample(event->port, event->source_sequence);
    ASSERT_TRUE(record) << runtime_error_message(record.error());
    records[index] = *record;
  }

  auto direct = RuntimeInputSelectionState::create(fixture.base.plan.input_selections.front());
  ASSERT_TRUE(direct);
  for (const auto& record : records) {
    ASSERT_TRUE(direct->push(record));
  }
  auto selected = direct->begin(
      records.back().capture_time, records.back().arrival_sequence, records.back().producer_epoch);
  ASSERT_TRUE(selected);
  ASSERT_EQ(selected->selected().size(), kSampleCount);
  const std::array channel_selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = selected->selection(),
      .samples = selected->selected(),
      .batch = true,
      .destructive = true,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, channel_selection));
  auto* const consumer_stream = reinterpret_cast<cudaStream_t>(std::uintptr_t{0xBAD'F00DU});
  ASSERT_TRUE((*session)->prepare_tensor_access(VertexId{2U}, 1U, consumer_stream));
  ASSERT_TRUE((*session)->prepare_tensor_access(VertexId{2U}, 1U, consumer_stream));
  {
    auto batch = fixture.sink.input.selected_batch();
    ASSERT_TRUE(batch) << runtime_error_message(batch.error());
    ASSERT_EQ(batch->size(), kSampleCount);
    for (std::size_t index = 0U; index < kSampleCount; ++index) {
      EXPECT_EQ((*batch)[index].data.data(), published_data[index]);
      EXPECT_EQ(tracking[index]->consumer_syncs.load(std::memory_order_relaxed), 1U);
      EXPECT_EQ(tracking[index]->deallocation_streams.load(std::memory_order_relaxed), 1U);
      EXPECT_EQ(tracking[index]->last_consumer_stream, consumer_stream);
      EXPECT_EQ(tracking[index]->last_deallocation_stream, consumer_stream);
    }
  }

  for (const auto& event : events) {
    ASSERT_TRUE((*binding)->retire(event.source_sequence));
  }
  ASSERT_TRUE(selected->finish(true));
  ASSERT_TRUE((*session)->retire_samples(PortId{2U}, records));
  for (const auto& state : tracking) {
    EXPECT_GT(state->live_handles.load(std::memory_order_relaxed), 0U);
    EXPECT_EQ(state->destroyed.load(std::memory_order_relaxed), 0U);
  }

  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  for (const auto& state : tracking) {
    EXPECT_EQ(state->live_handles.load(std::memory_order_relaxed), 0U);
    EXPECT_EQ(state->destroyed.load(std::memory_order_relaxed), 1U);
  }
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// REQ: CHAN-083
// REQ: TENSOR-STORAGE-010
TEST(ChannelBindingParityTest,
     TensorBracketedWindowsRetainOverlappingBackingAcrossActivationRetirement) {
  TensorSessionFixture fixture;
  fixture.enable_bracketed_window(5U);
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

  constexpr std::array<std::uint64_t, 7U> values{5U, 10U, 15U, 20U, 25U, 30U, 35U};
  constexpr std::array<std::int64_t, 7U> capture_times{5, 10, 15, 20, 25, 30, 35};
  auto script = publish_tensor_script(**session, **binding, fixture.source, values, capture_times);
  ASSERT_TRUE(script) << runtime_error_message(script.error());
  auto direct = RuntimeInputSelectionState::create(
      fixture.base.plan.input_selections.front(), &fixture.base.plan.bracketed_windows.front(), 2U);
  ASSERT_TRUE(direct) << runtime_error_message(direct.error());

  ASSERT_TRUE(direct->push(script->records[0]));
  ASSERT_TRUE(direct->push_anchor(script->records[1]));
  ASSERT_TRUE(direct->push(script->records[1]));
  ASSERT_TRUE(direct->push(script->records[2]));
  ASSERT_TRUE(direct->push_anchor(script->records[3]));
  ASSERT_TRUE(direct->push(script->records[3]));
  ASSERT_TRUE(direct->push(script->records[4]));
  ASSERT_TRUE(direct->push_anchor(script->records[5]));
  ASSERT_TRUE(direct->push(script->records[5]));
  ASSERT_TRUE(direct->push(script->records[6]));

  auto first = direct->begin(TimePoint{20, ClockId{1U}},
                             script->records[4].arrival_sequence,
                             script->records[4].producer_epoch);
  ASSERT_TRUE(first) << runtime_error_message(first.error());
  ASSERT_EQ(first->selected().size(), 5U);
  EXPECT_EQ(first->semantic_reference_time(), TimePoint(20, ClockId{1U}));
  const std::array first_selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = first->selection(),
      .samples = first->selected(),
      .batch = true,
      .destructive = false,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, first_selection));
  ASSERT_TRUE((*session)->prepare_tensor_access(VertexId{2U}, 1U, nullptr));
  std::array<const void*, 5U> first_data{};
  {
    auto batch = fixture.sink.input.selected_batch();
    ASSERT_TRUE(batch);
    ASSERT_EQ(batch->size(), first_data.size());
    for (std::size_t index = 0U; index < first_data.size(); ++index) {
      first_data[index] = (*batch)[index].data.data();
      EXPECT_EQ(first_data[index], script->data[index]);
    }
  }
  ASSERT_TRUE(first->finish(true));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  for (const auto& releases : script->releases) {
    EXPECT_EQ(releases->load(std::memory_order_relaxed), 0U);
  }

  auto second = direct->begin(TimePoint{30, ClockId{1U}},
                              script->records[6].arrival_sequence,
                              script->records[6].producer_epoch);
  ASSERT_TRUE(second) << runtime_error_message(second.error());
  ASSERT_EQ(second->selected().size(), 5U);
  EXPECT_EQ(second->semantic_reference_time(), TimePoint(30, ClockId{1U}));
  const std::array second_selection{RuntimeChannelActivationSelection{
      .port = PortId{2U},
      .selection = second->selection(),
      .samples = second->selected(),
      .batch = true,
      .destructive = false,
  }};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 2U, second_selection));
  ASSERT_TRUE((*session)->prepare_tensor_access(VertexId{2U}, 2U, nullptr));
  {
    auto batch = fixture.sink.input.selected_batch();
    ASSERT_TRUE(batch);
    ASSERT_EQ(batch->size(), 5U);
    for (std::size_t index = 0U; index < batch->size(); ++index) {
      EXPECT_EQ((*batch)[index].data.data(), script->data[index + 2U]);
    }
    EXPECT_EQ((*batch)[0].data.data(), first_data[2]);
    EXPECT_EQ((*batch)[1].data.data(), first_data[3]);
    EXPECT_EQ((*batch)[2].data.data(), first_data[4]);
  }
  ASSERT_TRUE(second->finish(true));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 2U, true));

  ASSERT_TRUE(direct->reset());
  for (const auto& event : script->events) {
    ASSERT_TRUE((*binding)->retire(event.source_sequence));
  }
  ASSERT_TRUE((*session)->retire_samples(PortId{2U}, script->records));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
  for (const auto& releases : script->releases) {
    EXPECT_EQ(releases->load(std::memory_order_relaxed), 1U);
  }
}

// The terminal source for a pending Tensor window changes, but the lifetime rule does not. An
// explicit EOS or timeout closes the window, reset/cancellation and producer-epoch transition
// discard it, and session teardown releases it. In every case overlapping historical samples and
// the pending window retire their one typed backing claim exactly once.
// REQ: CHAN-083
// REQ: TENSOR-STORAGE-010
TEST(ChannelBindingParityTest, TensorBracketedWindowTerminalPathsReleaseBackingExactlyOnce) {
  enum class TerminalPath : std::uint8_t {
    kEndOfStream,
    kTimeout,
    kCancellationReset,
    kProducerEpochReset,
    kRunTeardown,
  };
  constexpr std::array terminal_paths{TerminalPath::kEndOfStream,
                                      TerminalPath::kTimeout,
                                      TerminalPath::kCancellationReset,
                                      TerminalPath::kProducerEpochReset,
                                      TerminalPath::kRunTeardown};

  for (const auto terminal_path : terminal_paths) {
    SCOPED_TRACE(static_cast<unsigned>(terminal_path));
    std::vector<std::shared_ptr<std::atomic<std::uint32_t>>> releases;
    {
      TensorSessionFixture fixture;
      fixture.enable_bracketed_window(5U);
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

      constexpr std::array<std::uint64_t, 6U> values{5U, 10U, 15U, 20U, 25U, 30U};
      constexpr std::array<std::int64_t, 6U> capture_times{5, 10, 15, 20, 25, 30};
      auto script =
          publish_tensor_script(**session, **binding, fixture.source, values, capture_times);
      ASSERT_TRUE(script) << runtime_error_message(script.error());
      releases = script->releases;

      auto window = RuntimeInputSelectionState::create(fixture.base.plan.input_selections.front(),
                                                       &fixture.base.plan.bracketed_windows.front(),
                                                       2U);
      ASSERT_TRUE(window) << runtime_error_message(window.error());
      ASSERT_TRUE(window->push(script->records[0]));
      ASSERT_TRUE(window->push_anchor(script->records[1]));
      ASSERT_TRUE(window->push(script->records[1]));
      ASSERT_TRUE(window->push(script->records[2]));
      ASSERT_TRUE(window->push_anchor(script->records[3]));
      ASSERT_TRUE(window->push(script->records[3]));
      ASSERT_TRUE(window->push(script->records[4]));
      ASSERT_TRUE(window->push_anchor(script->records[5]));
      ASSERT_TRUE(window->push(script->records[5]));
      ASSERT_EQ(window->pending_window_count(), 1U);

      auto overlapping = window->begin(TimePoint{25, ClockId{1U}},
                                       script->records[4].arrival_sequence,
                                       script->records[4].producer_epoch);
      ASSERT_TRUE(overlapping) << runtime_error_message(overlapping.error());
      ASSERT_EQ(overlapping->selected().size(), 5U);
      const std::array overlapping_selection{RuntimeChannelActivationSelection{
          .port = PortId{2U},
          .selection = overlapping->selection(),
          .samples = overlapping->selected(),
          .batch = true,
          .destructive = false,
      }};
      ASSERT_TRUE((*session)->prepare_activation(VertexId{2U}, 1U, overlapping_selection));
      ASSERT_TRUE((*session)->prepare_tensor_access(VertexId{2U}, 1U, nullptr));
      {
        auto batch = fixture.sink.input.selected_batch();
        ASSERT_TRUE(batch) << runtime_error_message(batch.error());
        ASSERT_EQ(batch->size(), 5U);
        for (std::size_t index = 0U; index < batch->size(); ++index) {
          EXPECT_EQ((*batch)[index].data.data(), script->data[index]);
        }
      }
      ASSERT_TRUE(overlapping->finish(true));
      ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
      for (const auto& release : releases) {
        EXPECT_EQ(release->load(std::memory_order_relaxed), 0U);
      }

      switch (terminal_path) {
        case TerminalPath::kEndOfStream: {
          ASSERT_TRUE(window->close_end_of_stream());
          auto terminal = window->begin(TimePoint{30, ClockId{1U}},
                                        script->records.back().arrival_sequence,
                                        script->records.back().producer_epoch);
          ASSERT_TRUE(terminal) << runtime_error_message(terminal.error());
          EXPECT_EQ(terminal->selection().state, InputState::kAbsent);
          EXPECT_EQ(terminal->selection().reason, SelectionReason::kEndOfStream);
          ASSERT_TRUE(terminal->finish(true));
          break;
        }
        case TerminalPath::kTimeout: {
          ASSERT_TRUE(window->close_oldest(SelectionReason::kWindowTimeout));
          auto terminal = window->begin(TimePoint{30, ClockId{1U}},
                                        script->records.back().arrival_sequence,
                                        script->records.back().producer_epoch);
          ASSERT_TRUE(terminal) << runtime_error_message(terminal.error());
          EXPECT_EQ(terminal->selection().state, InputState::kAbsent);
          EXPECT_EQ(terminal->selection().reason, SelectionReason::kWindowTimeout);
          ASSERT_TRUE(terminal->finish(true));
          break;
        }
        case TerminalPath::kCancellationReset:
          ASSERT_TRUE(window->reset());
          break;
        case TerminalPath::kProducerEpochReset: {
          TemporalSampleRecord restarted = script->records.front();
          restarted.capture_time = TimePoint{40, ClockId{1U}};
          restarted.source_sequence = 1U;
          restarted.arrival_sequence = 1U;
          restarted.producer_epoch = 2U;
          ASSERT_TRUE(window->push(restarted));
          EXPECT_EQ(window->pending_window_count(), 0U);
          break;
        }
        case TerminalPath::kRunTeardown:
          break;
      }

      if (terminal_path != TerminalPath::kRunTeardown) {
        ASSERT_TRUE(window->reset());
        ASSERT_TRUE(window->reset()) << "repeated terminal cleanup must be idempotent";
        for (const auto& event : script->events) {
          ASSERT_TRUE((*binding)->retire(event.source_sequence));
        }
        ASSERT_TRUE((*session)->retire_samples(PortId{2U}, script->records));
        (*session)->cancel();
        ASSERT_TRUE((*session)->join());
      }
    }
    for (const auto& release : releases) {
      EXPECT_EQ(release->load(std::memory_order_relaxed), 1U);
    }
  }
}

// The window engine owns copied anchor coordinates, while the independent Exact selector owns
// the Tensor payload required by input(anchor, Exact{}). Closing the preceding Exact transaction
// may therefore release the historical anchor payload without invalidating the pending window;
// the current anchor remains retained until that window reaches a terminal disposition.
// REQ: CHAN-083
// REQ: TENSOR-STORAGE-010
TEST(ChannelBindingParityTest,
     TensorPendingWindowRetainsCurrentAnchorWithoutPinningHistoricalAnchorPayload) {
  TensorSessionFixture fixture;
  fixture.base.enable_scalar(InputSelectorKind::kExact, TemporalRecordMissingKind::kSkip);
  const InputSelectionPlanRecord anchor_selection = fixture.base.plan.input_selections.front();
  fixture.enable_bracketed_window(5U);
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

  constexpr std::array<std::uint64_t, 2U> values{10U, 20U};
  constexpr std::array<std::int64_t, 2U> capture_times{10, 20};
  auto script = publish_tensor_script(**session, **binding, fixture.source, values, capture_times);
  ASSERT_TRUE(script) << runtime_error_message(script.error());

  auto anchors = RuntimeInputSelectionState::create(anchor_selection);
  auto window = RuntimeInputSelectionState::create(
      fixture.base.plan.input_selections.front(), &fixture.base.plan.bracketed_windows.front(), 2U);
  ASSERT_TRUE(anchors);
  ASSERT_TRUE(window);
  ASSERT_TRUE(anchors->push(script->records[0]));
  ASSERT_TRUE(anchors->push(script->records[1]));
  ASSERT_TRUE(window->push_anchor(script->records[0]));
  ASSERT_TRUE(window->push_anchor(script->records[1]));
  ASSERT_EQ(window->pending_window_count(), 1U);

  std::vector<TemporalSampleRecord> window_claims(window->payload_claim_capacity());
  auto copied_window_claims = window->copy_payload_claims(window_claims);
  ASSERT_TRUE(copied_window_claims);
  EXPECT_EQ(*copied_window_claims, 0U)
      << "copied anchor coordinates must not create bracket-input payload claims";

  auto preceding = anchors->begin(script->records[0].capture_time,
                                  script->records[0].arrival_sequence,
                                  script->records[0].producer_epoch);
  ASSERT_TRUE(preceding);
  ASSERT_EQ(preceding->selected().size(), 1U);
  EXPECT_EQ(preceding->selected().front(), script->records[0]);
  ASSERT_TRUE(preceding->finish(true));

  std::vector<TemporalSampleRecord> anchor_claims(anchors->payload_claim_capacity());
  auto copied_anchor_claims = anchors->copy_payload_claims(anchor_claims);
  ASSERT_TRUE(copied_anchor_claims);
  ASSERT_EQ(*copied_anchor_claims, 1U);
  EXPECT_EQ(anchor_claims.front(), script->records[1]);

  ASSERT_TRUE((*binding)->retire(script->events[0].source_sequence));
  ASSERT_TRUE((*session)->retire_samples(
      PortId{2U}, std::span<const TemporalSampleRecord>{script->records}.first(1U)));
  EXPECT_EQ(script->releases[0]->load(std::memory_order_relaxed), 1U);
  EXPECT_EQ(script->releases[1]->load(std::memory_order_relaxed), 0U);
  EXPECT_EQ(window->pending_window_count(), 1U);

  ASSERT_TRUE(window->close_oldest(SelectionReason::kWindowTimeout));
  auto terminal = window->begin(TimePoint{21, ClockId{1U}},
                                script->records[1].arrival_sequence,
                                script->records[1].producer_epoch);
  ASSERT_TRUE(terminal);
  EXPECT_EQ(terminal->selection().state, InputState::kAbsent);
  EXPECT_EQ(terminal->selection().reason, SelectionReason::kWindowTimeout);
  ASSERT_TRUE(terminal->finish(true));
  EXPECT_EQ(window->pending_window_count(), 0U);

  auto current = anchors->begin(script->records[1].capture_time,
                                script->records[1].arrival_sequence,
                                script->records[1].producer_epoch);
  ASSERT_TRUE(current);
  ASSERT_EQ(current->selected().size(), 1U);
  EXPECT_EQ(current->selected().front(), script->records[1]);
  ASSERT_TRUE(current->finish(true));
  copied_anchor_claims = anchors->copy_payload_claims(anchor_claims);
  ASSERT_TRUE(copied_anchor_claims);
  EXPECT_EQ(*copied_anchor_claims, 0U);

  ASSERT_TRUE((*binding)->retire(script->events[1].source_sequence));
  ASSERT_TRUE((*session)->retire_samples(
      PortId{2U}, std::span<const TemporalSampleRecord>{script->records}.subspan(1U)));
  EXPECT_EQ(script->releases[1]->load(std::memory_order_relaxed), 1U);
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// A full bracket cache rejects the next Tensor record atomically. The rejected record can retire
// independently, while every record already owned by the selector remains backed and no partial
// SelectedBatchView becomes visible.
// REQ: CHAN-083
// REQ: TENSOR-STORAGE-010
TEST(ChannelBindingParityTest,
     TensorBracketCacheExhaustionExposesNoPartialBatchOrEarlyBackingReuse) {
  TensorSessionFixture fixture;
  fixture.enable_bracketed_window(3U);
  fixture.base.plan.input_selections.front().cache_capacity = 3U;
  fixture.base.plan.bracketed_windows.front().cache_capacity = 3U;
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

  constexpr std::array<std::uint64_t, 4U> values{5U, 10U, 15U, 20U};
  constexpr std::array<std::int64_t, 4U> capture_times{5, 10, 15, 20};
  auto script = publish_tensor_script(**session, **binding, fixture.source, values, capture_times);
  ASSERT_TRUE(script) << runtime_error_message(script.error());
  auto window = RuntimeInputSelectionState::create(
      fixture.base.plan.input_selections.front(), &fixture.base.plan.bracketed_windows.front(), 2U);
  ASSERT_TRUE(window);
  for (std::size_t index = 0U; index < 3U; ++index) {
    ASSERT_TRUE(window->push(script->records[index]));
  }
  auto exhausted = window->push(script->records[3]);
  ASSERT_FALSE(exhausted);
  EXPECT_EQ(exhausted.error().code, ErrorCode::kResourceExhausted);
  std::vector<TemporalSampleRecord> retained(window->payload_claim_capacity());
  auto retained_count = window->copy_payload_claims(retained);
  ASSERT_TRUE(retained_count);
  ASSERT_EQ(*retained_count, 3U);
  for (std::size_t index = 0U; index < 3U; ++index) {
    EXPECT_EQ(retained[index], script->records[index]);
  }
  auto no_partial = fixture.sink.input.selected_batch();
  ASSERT_FALSE(no_partial);
  EXPECT_EQ(no_partial.error().code, ErrorCode::kStaleEndpoint);

  ASSERT_TRUE((*binding)->retire(script->events[3].source_sequence));
  ASSERT_TRUE((*session)->retire_samples(
      PortId{2U}, std::span<const TemporalSampleRecord>{script->records}.subspan(3U)));
  EXPECT_EQ(script->releases[3]->load(std::memory_order_relaxed), 1U);
  for (std::size_t index = 0U; index < 3U; ++index) {
    EXPECT_EQ(script->releases[index]->load(std::memory_order_relaxed), 0U);
  }

  ASSERT_TRUE(window->reset());
  retained_count = window->copy_payload_claims(retained);
  ASSERT_TRUE(retained_count);
  EXPECT_EQ(*retained_count, 0U);
  EXPECT_EQ(window->pending_window_count(), 0U);
  for (std::size_t index = 0U; index < 3U; ++index) {
    ASSERT_TRUE((*binding)->retire(script->events[index].source_sequence));
  }
  ASSERT_TRUE((*session)->retire_samples(
      PortId{2U}, std::span<const TemporalSampleRecord>{script->records}.first(3U)));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
  for (const auto& releases : script->releases) {
    EXPECT_EQ(releases->load(std::memory_order_relaxed), 1U);
  }
}

// Ordinary Tensor forwarding accepts the complete Sample<Tensor> envelope. Runtime stamps only
// its publication-owned fields; capture/publish coordinates, trace, frame, flags, and the exact
// caller-owned backing pass through without fabricating a direct-output pool allocation.
// REQ: CHAN-083
// REQ: CHAN-118
// REQ: CHAN-102
// REQ: TENSOR-STORAGE-010
TEST(ChannelBindingParityTest, TensorSampleEmitPreservesMetadataAndExactPassThroughBacking) {
  TensorSessionFixture fixture;
  // Key setup: the strict 1:1 blocking edge retains the owning Tensor handle directly in the
  // destination slot instead of publishing a sidecar descriptor through HoloMQ.
  fixture.use_direct_retained_delivery();
  EXPECT_TRUE(fixture.base.plan.tensor_allocations.empty());
  ASSERT_EQ(fixture.base.plan.process_local_channel_services.size(), 1U);
  EXPECT_TRUE(fixture.base.plan.process_local_channel_services.front().tensor_pool_id.empty());

  auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  const auto indexes = RuntimeChannelSessionTestAccess::pump_index_state(**session);
  ASSERT_TRUE(indexes) << runtime_error_message(indexes.error());
  // Key assertion: direct Tensor retention constructs neither a WaitSet attachment nor pump work.
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

  auto owned = make_owned_tensor_sample(0xFEED'BEEFU);
  ASSERT_TRUE(owned) << runtime_error_message(owned.error());
  const void* const published_data = owned->data;
  auto releases = owned->releases;
  const SampleMetadata supplied{
      .version = SampleMetadataVersion{.major = 9U, .minor = 7U},
      .sequence = 999U,
      .capture_timestamp_ns = 123,
      .source_clock_id = ClockId{1U},
      .publish_timestamp_ns = 456,
      .publish_clock_id = ClockId{1U},
      .trace_id = 0xCAFE'BEEFU,
      .frame_id = 73U,
      .byte_size = 999U,
      .payload_kind = PayloadKind::kInlineTyped,
      .flags = SampleFlags::kDegraded,
  };
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(Sample<Tensor>{std::move(owned->tensor), supplied}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));

  auto event = wait_for_event(**session, **binding);
  ASSERT_TRUE(event) << runtime_error_message(event.error());
  auto record = (*session)->temporal_sample(event->port, event->source_sequence);
  ASSERT_TRUE(record) << runtime_error_message(record.error());
  EXPECT_EQ(record->capture_time, TimePoint(123, ClockId{1U}));
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
    // Key assertion: typed receive observes the exact backing and forwarded semantic metadata.
    EXPECT_EQ(received->data.data(), published_data);
    ASSERT_NE(received->data.data_as<std::uint64_t>(), nullptr);
    EXPECT_EQ(*received->data.data_as<std::uint64_t>(), 0xFEED'BEEFU);
    EXPECT_EQ(received->metadata.version.major, 1U);
    EXPECT_EQ(received->metadata.version.minor, 0U);
    EXPECT_EQ(received->metadata.sequence, record->source_sequence);
    EXPECT_EQ(received->metadata.capture_timestamp_ns, supplied.capture_timestamp_ns);
    EXPECT_EQ(received->metadata.source_clock_id, supplied.source_clock_id);
    EXPECT_EQ(received->metadata.publish_timestamp_ns, supplied.publish_timestamp_ns);
    EXPECT_EQ(received->metadata.publish_clock_id, supplied.publish_clock_id);
    EXPECT_EQ(received->metadata.trace_id, supplied.trace_id);
    EXPECT_EQ(received->metadata.frame_id, supplied.frame_id);
    EXPECT_EQ(received->metadata.byte_size, sizeof(std::uint64_t));
    EXPECT_EQ(received->metadata.payload_kind, PayloadKind::kDescriptorBacked);
    EXPECT_EQ(received->metadata.flags, supplied.flags);
  }
  EXPECT_EQ(releases->load(std::memory_order_relaxed), 0U);

  ASSERT_TRUE((*binding)->retire(event->source_sequence));
  // Retiring the arrival/readiness credit cannot recycle the payload while temporal selection
  // and the retained sample still own the Tensor claim.
  EXPECT_EQ(releases->load(std::memory_order_relaxed), 0U);
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  ASSERT_TRUE(
      (*session)->retire_samples(PortId{2U}, std::span<const TemporalSampleRecord>{&*record, 1U}));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
  EXPECT_EQ(releases->load(std::memory_order_relaxed), 1U);
}

// An exception before the first Tensor endpoint accepts is a proven pre-submit failure. Once one
// fanout leg accepts, the same exception has crossed the visibility boundary and consumes the
// publication sequence as indeterminate.
// REQ: CHAN-082
// REQ: RUNTIME-019
TEST(ChannelBindingParityTest, TensorFanoutExceptionsRespectSubmissionBoundary) {
#if !defined(__cpp_exceptions) && !defined(_CPPUNWIND)
  GTEST_SKIP() << "exception classification requires exception-enabled compilation";
#else
  for (const std::size_t accepted_count : {0U, 1U}) {
    SCOPED_TRACE(accepted_count);
    TensorSessionFixture fixture;
    fixture.add_blocking_fanout();
    auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
    ASSERT_TRUE(session) << runtime_error_message(session.error());
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::throw_tensor_fanout_after_accepts(
        **session, PortId{1U}, accepted_count));
    ASSERT_TRUE((*session)->refresh_connections());
    ASSERT_TRUE((*session)->arm());
    ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));

    auto owned = make_owned_tensor_sample(0xF00DU + accepted_count);
    ASSERT_TRUE(owned) << runtime_error_message(owned.error());
    auto published = fixture.source.output.emit(Sample<Tensor>{std::move(owned->tensor), {}});
    ASSERT_FALSE(published);
    EXPECT_EQ(published.error().code,
              accepted_count == 0U ? ErrorCode::kFailure : ErrorCode::kPublicationIndeterminate);

    (*session)->cancel();
    EXPECT_TRUE((*session)->join());
  }
#endif
}

// One publication owns one backing allocation even when the process-local service fans out to
// multiple typed subscribers. Each subscriber and activation contributes an independent bounded
// claim, so retiring the first leg cannot reclaim storage still named by the second leg.
// REQ: CHAN-083
// REQ: CHAN-102
// REQ: TENSOR-STORAGE-010
TEST(ChannelBindingParityTest, TensorFanoutReleasesBackingAfterTheFinalSubscriberClaim) {
  TensorSessionFixture fixture;
  fixture.add_blocking_fanout();
  auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  const auto bind = [&](VertexId owner, SourceId source) {
    return (*session)->create_readiness_binding(
        owner,
        RuntimePortBindingRequest{
            .plan_generation = fixture.base.plan.plan_generation,
            .run_generation = 1U,
            .kind = RuntimeEventKind::kInputArrival,
            .source = source,
            .source_clock = ClockId{1U},
            .reference_clock = ClockId{1U},
            .capacity = 8U,
        });
  };
  auto first_binding = bind(VertexId{2U}, make_source_id(1U));
  auto second_binding = bind(VertexId{3U}, make_source_id(2U));
  ASSERT_TRUE(first_binding) << runtime_error_message(first_binding.error());
  ASSERT_TRUE(second_binding) << runtime_error_message(second_binding.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());

  auto owned = make_owned_tensor_sample(0xCAFEU);
  ASSERT_TRUE(owned) << runtime_error_message(owned.error());
  const void* const published_data = owned->data;
  auto releases = owned->releases;
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(Sample<Tensor>{
      std::move(owned->tensor),
      SampleMetadata{.capture_timestamp_ns = 10, .source_clock_id = ClockId{1U}},
  }));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));

  auto first_event = wait_for_event(**session, **first_binding);
  auto second_event = wait_for_event(**session, **second_binding);
  ASSERT_TRUE(first_event) << runtime_error_message(first_event.error());
  ASSERT_TRUE(second_event) << runtime_error_message(second_event.error());
  auto first_record = (*session)->temporal_sample(first_event->port, first_event->source_sequence);
  auto second_record =
      (*session)->temporal_sample(second_event->port, second_event->source_sequence);
  ASSERT_TRUE(first_record) << runtime_error_message(first_record.error());
  ASSERT_TRUE(second_record) << runtime_error_message(second_record.error());

  const auto receive_leg = [&](VertexId owner,
                               PortId port,
                               std::uint64_t generation,
                               const TemporalSampleRecord& record,
                               TensorSinkOperator& sink) {
    const std::array selection{RuntimeChannelActivationSelection{
        .port = port,
        .selection = {InputState::kSelected, SelectionReason::kNone},
        .samples = std::span<const TemporalSampleRecord>{&record, 1U},
        .destructive = true,
    }};
    ASSERT_TRUE((*session)->prepare_activation(owner, generation, selection));
    ASSERT_TRUE((*session)->prepare_tensor_access(owner, generation, nullptr));
    auto sample = sink.input.receive();
    ASSERT_TRUE(sample) << runtime_error_message(sample.error());
    EXPECT_EQ(sample->data.data(), published_data);
    EXPECT_EQ(*sample->data.data_as<std::uint64_t>(), 0xCAFEU);
  };
  receive_leg(VertexId{2U}, PortId{2U}, 1U, *first_record, fixture.sink);
  receive_leg(VertexId{3U}, PortId{3U}, 1U, *second_record, fixture.second_sink);

  ASSERT_TRUE((*first_binding)->retire(first_event->source_sequence));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  ASSERT_TRUE((*session)->retire_samples(
      PortId{2U}, std::span<const TemporalSampleRecord>{&*first_record, 1U}));
  EXPECT_EQ(releases->load(std::memory_order_relaxed), 0U);

  ASSERT_TRUE((*second_binding)->retire(second_event->source_sequence));
  ASSERT_TRUE((*session)->retire_activation(VertexId{3U}, 1U, true));
  ASSERT_TRUE((*session)->retire_samples(
      PortId{3U}, std::span<const TemporalSampleRecord>{&*second_record, 1U}));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
  // Key assertion: backing is released exactly once after the last subscriber leg retires.
  EXPECT_EQ(releases->load(std::memory_order_relaxed), 1U);
}

// kDropOldest replacement retires the superseded sidecar claim rather than leaking the first Tensor
// or transferring its identity to the survivor. Arming after both submissions makes replacement
// deterministic: the pump cannot consume the first queue element before the second arrives.
// REQ: CHAN-083
// REQ: CHAN-102
// REQ: TENSOR-STORAGE-010
TEST(ChannelBindingParityTest, TensorDropOldestReplacementRetiresOnlyTheSupersededBacking) {
  TensorSessionFixture fixture;
  fixture.use_one_slot_drop_oldest_queue();
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

  auto first = make_owned_tensor_sample(11U);
  auto second = make_owned_tensor_sample(22U);
  ASSERT_TRUE(first) << runtime_error_message(first.error());
  ASSERT_TRUE(second) << runtime_error_message(second.error());
  const void* const first_data = first->data;
  const void* const second_data = second->data;
  auto first_releases = first->releases;
  auto second_releases = second->releases;
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::move(first->tensor)));
  ASSERT_TRUE(fixture.source.output.emit(std::move(second->tensor)));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  EXPECT_NE(first_data, second_data);
  // Key setup: arm after both emissions so the pump cannot consume the first before replacement.
  EXPECT_EQ(first_releases->load(std::memory_order_relaxed), 1U);
  EXPECT_EQ(second_releases->load(std::memory_order_relaxed), 0U);

  ASSERT_TRUE((*session)->arm());
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
    auto sample = fixture.sink.input.receive();
    ASSERT_TRUE(sample) << runtime_error_message(sample.error());
    EXPECT_EQ(sample->data.data(), second_data);
    EXPECT_EQ(*sample->data.data_as<std::uint64_t>(), 22U);
  }
  ASSERT_TRUE((*binding)->retire(event->source_sequence));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  ASSERT_TRUE(
      (*session)->retire_samples(PortId{2U}, std::span<const TemporalSampleRecord>{&*record, 1U}));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
  // Key assertion: only the surviving (second) Tensor's backing is released after retirement.
  EXPECT_EQ(second_releases->load(std::memory_order_relaxed), 1U);
}

// Direct kDropOldest replaces in the typed store itself. The outgoing owning Tensor must release
// before the incoming sample becomes the retained occupant, without a transient sidecar claim.
// REQ: CHAN-118
// REQ: TENSOR-STORAGE-010
TEST(ChannelBindingParityTest, DirectDropOldestTensorReleasesSupersededBackingSynchronously) {
  TensorSessionFixture fixture;
  fixture.use_one_slot_drop_oldest_queue();
  fixture.use_direct_retained_delivery();
  auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  const auto indexes = RuntimeChannelSessionTestAccess::pump_index_state(**session);
  ASSERT_TRUE(indexes) << runtime_error_message(indexes.error());
  EXPECT_EQ(indexes->wait_handle_count, 0U);
  EXPECT_EQ(indexes->drain_subscriber_count, 0U);
  ASSERT_TRUE((*session)->refresh_connections());

  auto first = make_owned_tensor_sample(11U);
  auto second = make_owned_tensor_sample(22U);
  ASSERT_TRUE(first) << runtime_error_message(first.error());
  ASSERT_TRUE(second) << runtime_error_message(second.error());
  auto first_releases = first->releases;
  auto second_releases = second->releases;
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::move(first->tensor)));
  EXPECT_EQ(first_releases->load(std::memory_order_relaxed), 0U);
  ASSERT_TRUE(fixture.source.output.emit(std::move(second->tensor)));

  // Key assertion: the synchronous direct store has already destroyed the superseded occupant,
  // while the incoming backing remains owned by the one admitted destination slot.
  EXPECT_EQ(first_releases->load(std::memory_order_relaxed), 1U);
  EXPECT_EQ(second_releases->load(std::memory_order_relaxed), 0U);
  const auto retained = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
  ASSERT_TRUE(retained) << runtime_error_message(retained.error());
  EXPECT_EQ(retained->occupied_samples, 1U);
  EXPECT_EQ(retained->next_arrival_sequence, 3U);

  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// A live consumer sample is an independent owning claim. It keeps the typed slot ineligible for
// replacement while the independent direct mailbox accepts a newer Tensor. After Runtime retires
// the typed slot, the mailbox survivor is promoted without reclaiming the backing still held by
// that consumer.
// REQ: CHAN-118
// REQ: TENSOR-STORAGE-010
TEST(ChannelBindingParityTest, DirectDropOldestTensorDoesNotRecycleBackingUnderLiveConsumer) {
  TensorSessionFixture fixture;
  fixture.use_one_slot_drop_oldest_queue();
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
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));

  auto first = make_owned_tensor_sample(11U);
  ASSERT_TRUE(first) << runtime_error_message(first.error());
  auto first_releases = first->releases;
  ASSERT_TRUE(fixture.source.output.emit(std::move(first->tensor)));
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
  std::optional<Sample<Tensor>> held;
  {
    auto received = fixture.sink.input.receive();
    ASSERT_TRUE(received) << runtime_error_message(received.error());
    ASSERT_NE(received->data.data_as<std::uint64_t>(), nullptr);
    EXPECT_EQ(*received->data.data_as<std::uint64_t>(), 11U);
    held.emplace(std::move(*received));
  }

  auto queued = make_owned_tensor_sample(22U);
  ASSERT_TRUE(queued) << runtime_error_message(queued.error());
  auto queued_releases = queued->releases;
  ASSERT_TRUE(fixture.source.output.emit(std::move(queued->tensor)));
  EXPECT_EQ(queued_releases->load(std::memory_order_relaxed), 0U);
  EXPECT_EQ(first_releases->load(std::memory_order_relaxed), 0U);
  const auto pending = RuntimeChannelSessionTestAccess::input_reset_state(**session, PortId{2U});
  ASSERT_TRUE(pending) << runtime_error_message(pending.error());
  EXPECT_EQ(pending->occupied_samples, 1U);
  EXPECT_EQ(pending->pending_direct_ingress_samples, 1U);

  ASSERT_TRUE((*binding)->retire(event->source_sequence));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  ASSERT_TRUE(
      (*session)->retire_samples(PortId{2U}, std::span<const TemporalSampleRecord>{&*record, 1U}));
  EXPECT_EQ(first_releases->load(std::memory_order_relaxed), 0U);
  EXPECT_EQ(queued_releases->load(std::memory_order_relaxed), 0U);

  auto retry = make_owned_tensor_sample(33U);
  ASSERT_TRUE(retry) << runtime_error_message(retry.error());
  auto retry_releases = retry->releases;
  ASSERT_TRUE(fixture.source.output.emit(std::move(retry->tensor)));
  ASSERT_TRUE(held.has_value());
  EXPECT_EQ(*held->data.data_as<std::uint64_t>(), 11U);
  EXPECT_EQ(first_releases->load(std::memory_order_relaxed), 0U);
  held.reset();
  EXPECT_EQ(first_releases->load(std::memory_order_relaxed), 1U);
  EXPECT_EQ(queued_releases->load(std::memory_order_relaxed), 1U);
  EXPECT_EQ(retry_releases->load(std::memory_order_relaxed), 0U);

  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// Descriptor-Tensor generated values must preserve the same fanout lifetime as direct Tensor
// payloads while reconstructing independent owning schema envelopes for each subscriber.
// REQ: CHAN-084
// REQ: CHAN-119
// REQ: TENSOR-SCHEMA-005
// REQ: TENSOR-STORAGE-010
TEST(ChannelBindingParityTest, DescriptorTensorImageFanoutSharesBackingUntilBothLegsRetire) {
  ImageSessionFixture fixture;
  fixture.add_blocking_fanout();
  // Key assertion: fanout remains on HoloMQ and retains one sidecar claim per recipient.
  EXPECT_EQ(fixture.base.plan.process_local_channel_services.front().delivery_kind,
            ProcessLocalChannelDeliveryKind::kHoloMQ);
  auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  const auto bind = [&](VertexId owner, SourceId source) {
    return (*session)->create_readiness_binding(
        owner,
        RuntimePortBindingRequest{
            .plan_generation = fixture.base.plan.plan_generation,
            .run_generation = 1U,
            .kind = RuntimeEventKind::kInputArrival,
            .source = source,
            .source_clock = ClockId{1U},
            .reference_clock = ClockId{1U},
            .capacity = 8U,
        });
  };
  auto first_binding = bind(VertexId{2U}, make_source_id(1U));
  auto second_binding = bind(VertexId{3U}, make_source_id(2U));
  ASSERT_TRUE(first_binding) << runtime_error_message(first_binding.error());
  ASSERT_TRUE(second_binding) << runtime_error_message(second_binding.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());

  auto owned = make_owned_tensor_sample(0xBEEFU);
  ASSERT_TRUE(owned) << runtime_error_message(owned.error());
  const void* const published_data = owned->data;
  auto releases = owned->releases;
  schema::ImageT image{};
  image.height = 1;
  image.width = 1;
  image.encoding = schema::ImageEncoding_CUSTOM;
  image.header = std::make_shared<schema::HeaderT>();
  image.header->frame_id = "fanout_image";
  image.data = std::make_shared<Tensor>(std::move(owned->tensor));
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(Sample<schema::ImageT>{
      std::move(image),
      SampleMetadata{.capture_timestamp_ns = 10, .source_clock_id = ClockId{1U}},
  }));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));

  auto first_event = wait_for_event(**session, **first_binding);
  auto second_event = wait_for_event(**session, **second_binding);
  ASSERT_TRUE(first_event) << runtime_error_message(first_event.error());
  ASSERT_TRUE(second_event) << runtime_error_message(second_event.error());
  auto first_record = (*session)->temporal_sample(first_event->port, first_event->source_sequence);
  auto second_record =
      (*session)->temporal_sample(second_event->port, second_event->source_sequence);
  ASSERT_TRUE(first_record) << runtime_error_message(first_record.error());
  ASSERT_TRUE(second_record) << runtime_error_message(second_record.error());

  const auto receive_leg = [&](VertexId owner,
                               PortId port,
                               std::uint64_t generation,
                               const TemporalSampleRecord& record,
                               ImageSinkOperator& sink) {
    const std::array selection{RuntimeChannelActivationSelection{
        .port = port,
        .selection = {InputState::kSelected, SelectionReason::kNone},
        .samples = std::span<const TemporalSampleRecord>{&record, 1U},
        .destructive = true,
    }};
    ASSERT_TRUE((*session)->prepare_activation(owner, generation, selection));
    ASSERT_TRUE((*session)->prepare_tensor_access(owner, generation, nullptr));
    auto sample = sink.input.receive();
    ASSERT_TRUE(sample) << runtime_error_message(sample.error());
    ASSERT_NE(sample->data.data, nullptr);
    EXPECT_EQ(sample->data.data->data(), published_data);
    EXPECT_EQ(sample->data.header->frame_id, "fanout_image");
  };
  receive_leg(VertexId{2U}, PortId{2U}, 1U, *first_record, fixture.sink);
  receive_leg(VertexId{3U}, PortId{3U}, 1U, *second_record, fixture.second_sink);

  ASSERT_TRUE((*first_binding)->retire(first_event->source_sequence));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2U}, 1U, true));
  ASSERT_TRUE((*session)->retire_samples(
      PortId{2U}, std::span<const TemporalSampleRecord>{&*first_record, 1U}));
  EXPECT_EQ(releases->load(std::memory_order_relaxed), 0U);

  ASSERT_TRUE((*second_binding)->retire(second_event->source_sequence));
  ASSERT_TRUE((*session)->retire_activation(VertexId{3U}, 1U, true));
  ASSERT_TRUE((*session)->retire_samples(
      PortId{3U}, std::span<const TemporalSampleRecord>{&*second_record, 1U}));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
  EXPECT_EQ(releases->load(std::memory_order_relaxed), 1U);
}
}  // namespace
}  // namespace holoscan::detail

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>

#include <gtest/gtest.h>

#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include "common/allocation_counter.hpp"
#include "core/operator_spec_compiler_access.hpp"
#include "core/source_id_internal.hpp"
#include "runtime_channel_binding_schema_tensor_test_support.hpp"

namespace holoscan::detail {
namespace {

using namespace runtime_channel_binding_test;

// Count on whichever lane actually assigns the received value, including the HoloMQ pump.
// No production tracing, allocator hook, or callback-only approximation is needed.
struct PreparedIngressProbe {
  std::atomic<std::size_t> assignments{};
  std::atomic<std::size_t> allocations{};
  std::atomic<std::size_t> prepared{};
  std::atomic<std::size_t> destroyed{};
  std::atomic<void*> last_prepared{};
  std::atomic<std::size_t> replacement_assignments{};
  std::size_t fail_preparation_at{};
};
PreparedIngressProbe* active_probe{};

struct ImuSessionFixture {
  SessionFixture base;
  EmptySource<schema::ImuT> source;
  EmptySink<schema::ImuT> sink;
  OperatorSpec source_spec;
  OperatorSpec sink_spec;
  std::array<RuntimeChannelPortReplayRecord, 2> replay;
  PreparedIngressProbe probe;
  RuntimePortTypeOps instrumented = *runtime_port_type_ops<schema::ImuT>();

  ImuSessionFixture() {
    active_probe = &probe;
    instrumented.prepare_sample_storage = [](void* storage) noexcept -> expected<void, Error> {
      if (active_probe->fail_preparation_at != 0 &&
          active_probe->prepared.load() + 1 == active_probe->fail_preparation_at)
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      auto result = runtime_port_type_ops<schema::ImuT>()->prepare_sample_storage(storage);
      if (result) {
        active_probe->prepared.fetch_add(1);
        active_probe->last_prepared.store(storage);
      }
      return result;
    };
    instrumented.destroy_sample_storage = [](void* storage) noexcept {
      runtime_port_type_ops<schema::ImuT>()->destroy_sample_storage(storage);
      active_probe->destroyed.fetch_add(1);
    };
    instrumented.assign_sample_storage =
        [](void* storage, std::span<const std::byte> wire, SampleMetadata metadata) noexcept {
          const test::ScopedAllocationCounter counter;
          auto result =
              runtime_port_type_ops<schema::ImuT>()->assign_sample_storage(storage, wire, metadata);
          active_probe->allocations.fetch_add(counter.count());
          active_probe->assignments.fetch_add(1);
          if (storage == active_probe->last_prepared.load())
            active_probe->replacement_assignments.fetch_add(1);
          return result;
        };
    source.setup(source_spec);
    sink.setup(sink_spec);
    const auto output = OperatorSpecCompilerAccess::port(source_spec, 0);
    const auto input = OperatorSpecCompilerAccess::port(sink_spec, 0);
    auto& plan = base.plan;
    plan.ports[0].payload_identity = output.payload_identity;
    plan.ports[0].serialized_payload_capacity_bytes = output.serialized_payload_capacity_bytes;
    plan.ports[1].payload_identity = input.payload_identity;
    plan.ports[1].serialized_payload_capacity_bytes = input.serialized_payload_capacity_bytes;
    auto& service = plan.process_local_channel_services.front();
    service.payload_identity = output.payload_identity;
    service.payload_size_bytes =
        output.serialized_payload_capacity_bytes + kSchemaEnvelopeHeaderSize;
    plan.payload_edges[0].payload_identity = output.payload_identity;
    // Imu's generated inline structs do not promise no-throw default construction, so it is not
    // eligible for the scalar fixture's generic output-loan pool. These tests publish by value.
    plan.generic_output_loans.clear();
    replay = {RuntimeChannelPortReplayRecord{.port = PortId{1},
                                             .owner = VertexId{1},
                                             .direction = PortDirection::kOutput,
                                             .declaration = &source.output,
                                             .type_ops = output.runtime_type_ops},
              RuntimeChannelPortReplayRecord{.port = PortId{2},
                                             .owner = VertexId{2},
                                             .direction = PortDirection::kInput,
                                             .declaration = &sink.input,
                                             .type_ops = &instrumented}};
  }
  ~ImuSessionFixture() { active_probe = nullptr; }
};

class PreparedSchemaIngress : public ::testing::TestWithParam<int> {};

// REQ: SCHEMA-009
// REQ: SCHEMA-010
TEST_P(PreparedSchemaIngress, DirectAndTransportMaterializeOnTheActualLaneWithoutAllocations) {
  ImuSessionFixture fixture;
  if (GetParam() != 0) {
    fixture.base.use_one_slot_blocking_transport();
    if (GetParam() == 1)
      fixture.base.use_same_partition_eventless_transport();
  } else {
    fixture.base.use_one_slot_direct_retained_delivery();
  }
  for (unsigned run = 0; run < 2; ++run) {
    auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
    ASSERT_TRUE(session) << runtime_error_message(session.error());
    auto binding = (*session)->create_readiness_binding(
        VertexId{2},
        RuntimePortBindingRequest{.plan_generation = fixture.base.plan.plan_generation,
                                  .run_generation = 1,
                                  .kind = RuntimeEventKind::kInputArrival,
                                  .source = make_source_id(1),
                                  .source_clock = ClockId{1},
                                  .reference_clock = ClockId{1},
                                  .capacity = 1});
    ASSERT_TRUE(binding) << runtime_error_message(binding.error());
    ASSERT_TRUE((*session)->refresh_connections());
    ASSERT_TRUE((*session)->arm());
    for (unsigned iteration = 0; iteration < 64; ++iteration) {
      ASSERT_TRUE((*session)->prepare_activation(VertexId{1}, 1, {}));
      schema::ImuT value;
      value.angular_velocity = schema::Vector3f{1.F, 2.F, static_cast<float>(iteration)};
      if (iteration % 2) {
        value.header = std::make_shared<schema::HeaderT>();
        value.header->frame_id.assign(128, 'i');
        value.header->stamp = std::make_shared<schema::TimeT>();
        value.header->stamp->time_ns = iteration;
      }
      if (iteration % 3)
        value.orientation_covariance = std::make_shared<schema::Matrix3f>();
      if (run == 0) {
        ASSERT_TRUE(fixture.source.output.emit(std::move(value), {.frame_id = 75}));
      } else {
        ASSERT_TRUE(fixture.source.output.emit_ref(value, {.frame_id = 75}));
        if (value.header) {
          EXPECT_EQ(value.header.use_count(), 1);
          value.header->frame_id.assign("caller has already reused this value");
          value.header->stamp->time_ns = -1;
        }
        value.angular_velocity = {};
      }
      ASSERT_TRUE((*session)->retire_activation(VertexId{1}, 1, true));
      ASSERT_TRUE((*session)->flush_process_local_activation_cut());
      auto event = wait_for_event(**session, **binding);
      ASSERT_TRUE(event) << runtime_error_message(event.error());
      auto record = (*session)->temporal_sample(event->port, event->source_sequence);
      ASSERT_TRUE(record);
      const std::array selection{RuntimeChannelActivationSelection{
          .port = PortId{2},
          .selection = {InputState::kSelected, SelectionReason::kNone},
          .samples = std::span<const TemporalSampleRecord>{&*record, 1},
          .destructive = true}};
      ASSERT_TRUE((*session)->prepare_activation(VertexId{2}, 1, selection));
      auto borrowed = fixture.sink.input.selected_sample();
      ASSERT_TRUE(borrowed);
      EXPECT_EQ(borrowed->data().angular_velocity.z(), iteration);
      EXPECT_EQ(static_cast<bool>(borrowed->data().header), iteration % 2 != 0);
      EXPECT_EQ(static_cast<bool>(borrowed->data().orientation_covariance), iteration % 3 != 0);
      EXPECT_EQ(borrowed->metadata().frame_id, 75U);
      if (iteration % 2) {
        EXPECT_EQ(borrowed->data().header->stamp->time_ns, iteration);
        EXPECT_EQ(borrowed->data().header->frame_id, std::string(128, 'i'));
      }
      ASSERT_TRUE((*binding)->retire(event->source_sequence));
      ASSERT_TRUE((*session)->retire_activation(VertexId{2}, 1, true));
      ASSERT_TRUE((*session)->retire_samples(PortId{2}, selection.front().samples));
    }
    (*session)->cancel();
    ASSERT_TRUE((*session)->join());
  }
  EXPECT_EQ(fixture.probe.assignments.load(), 128U);
  EXPECT_EQ(fixture.probe.allocations.load(), 0U);
  EXPECT_GT(fixture.probe.prepared.load(), 0U);
  EXPECT_EQ(fixture.probe.prepared.load(), fixture.probe.destroyed.load());
}

// Direct retained, eventless HoloMQ activation cut, and eventful HoloMQ pump thread.
INSTANTIATE_TEST_SUITE_P(PreparedPaths, PreparedSchemaIngress, ::testing::Values(0, 1, 2));

// REQ: SCHEMA-010
TEST(SchemaEmitRefRuntime, PressureAndBoundFailurePreserveValueAndPublicationSequence) {
  ImuSessionFixture fixture;
  fixture.base.use_one_slot_direct_retained_delivery();
  auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
  ASSERT_TRUE(session);
  auto binding = (*session)->create_readiness_binding(
      VertexId{2},
      RuntimePortBindingRequest{.plan_generation = fixture.base.plan.plan_generation,
                                .run_generation = 1,
                                .kind = RuntimeEventKind::kInputArrival,
                                .source = make_source_id(1),
                                .source_clock = ClockId{1},
                                .reference_clock = ClockId{1},
                                .capacity = 1});
  ASSERT_TRUE(binding);
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  schema::ImuT value;
  value.header = std::make_shared<schema::HeaderT>();
  value.header->frame_id.assign(128, 'r');
  const auto* header = value.header.get();
  auto inactive = fixture.source.output.emit_ref(value);
  ASSERT_FALSE(inactive);
  EXPECT_EQ(inactive.error().code, ErrorCode::kStaleEndpoint);
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1}, 1, {}));
  value.header->frame_id.assign(8192, 'x');
  auto oversized = fixture.source.output.emit_ref(value);
  ASSERT_FALSE(oversized);
  EXPECT_EQ(oversized.error().code, ErrorCode::kResourceExhausted);
  EXPECT_EQ(value.header.get(), header);
  EXPECT_EQ(value.header->frame_id.size(), 8192U);
  value.header->frame_id.assign(128, 'r');
  ASSERT_TRUE(fixture.source.output.emit_ref(value));
  for (unsigned sequence = 1; sequence <= 2; ++sequence) {
    auto blocked = fixture.source.output.emit_ref(value);
    ASSERT_FALSE(blocked);
    EXPECT_EQ(blocked.error().code, ErrorCode::kBackpressured);
    EXPECT_EQ(value.header.get(), header);
    EXPECT_EQ(value.header.use_count(), 1);
    auto event = wait_for_event(**session, **binding);
    ASSERT_TRUE(event);
    EXPECT_EQ(event->source_sequence, sequence);
    auto record = (*session)->temporal_sample(event->port, event->source_sequence);
    ASSERT_TRUE(record);
    const std::array selection{RuntimeChannelActivationSelection{
        .port = PortId{2},
        .selection = {InputState::kSelected, SelectionReason::kNone},
        .samples = std::span<const TemporalSampleRecord>{&*record, 1},
        .destructive = true}};
    ASSERT_TRUE((*session)->prepare_activation(VertexId{2}, 1, selection));
    auto borrowed = fixture.sink.input.selected_sample();
    ASSERT_TRUE(borrowed);
    EXPECT_EQ(borrowed->metadata().sequence, sequence);
    EXPECT_EQ(borrowed->data().header->frame_id, std::string(128, 'r'));
    ASSERT_TRUE((*binding)->retire(event->source_sequence));
    ASSERT_TRUE((*session)->retire_activation(VertexId{2}, 1, true));
    ASSERT_TRUE((*session)->retire_samples(PortId{2}, selection.front().samples));
    if (sequence == 1)
      ASSERT_TRUE(fixture.source.output.emit_ref(value));
  }
  ASSERT_TRUE((*session)->retire_activation(VertexId{1}, 1, true));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// REQ: SCHEMA-009
TEST(PreparedSchemaIngressFailure, PartialColdInitializationCleansUpAndCanRetry) {
  ImuSessionFixture fixture;
  fixture.probe.fail_preparation_at = 2;
  auto failed = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
  ASSERT_FALSE(failed);
  EXPECT_EQ(failed.error().code, ErrorCode::kResourceExhausted);
  EXPECT_EQ(fixture.probe.prepared.load(), 1U);
  EXPECT_EQ(fixture.probe.destroyed.load(), 1U);
  fixture.probe.fail_preparation_at = 0;
  {
    auto retried = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
    ASSERT_TRUE(retried) << runtime_error_message(retried.error());
  }
  EXPECT_EQ(fixture.probe.prepared.load(), fixture.probe.destroyed.load());
}

// REQ: SCHEMA-009
TEST(PreparedSchemaIngressReplacement, LossyReplacementUsesColdInventoryAndPreservesHeldBorrow) {
  ImuSessionFixture fixture;
  fixture.base.use_one_slot_drop_oldest_queue();
  fixture.base.use_direct_retained_delivery();
  auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto binding = (*session)->create_readiness_binding(
      VertexId{2},
      RuntimePortBindingRequest{.plan_generation = fixture.base.plan.plan_generation,
                                .run_generation = 1,
                                .kind = RuntimeEventKind::kInputArrival,
                                .source = make_source_id(1),
                                .source_clock = ClockId{1},
                                .reference_clock = ClockId{1},
                                .capacity = 1});
  ASSERT_TRUE(binding);
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1}, 1, {}));
  auto publish = [&](unsigned value) {
    schema::ImuT sample;
    sample.header = std::make_shared<schema::HeaderT>();
    sample.header->frame_id.assign(128, static_cast<char>('a' + value));
    sample.header->device_sequence = value;
    return fixture.source.output.emit(std::move(sample));
  };
  ASSERT_TRUE(publish(1));
  ASSERT_TRUE(publish(2));
  auto event = wait_for_event(**session, **binding);
  ASSERT_TRUE(event) << runtime_error_message(event.error());
  auto record = (*session)->temporal_sample(event->port, event->source_sequence);
  ASSERT_TRUE(record);
  const std::array selection{RuntimeChannelActivationSelection{
      .port = PortId{2},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .samples = std::span<const TemporalSampleRecord>{&*record, 1},
      .destructive = true}};
  ASSERT_TRUE((*session)->prepare_activation(VertexId{2}, 1, selection));
  auto borrowed = fixture.sink.input.selected_sample();
  ASSERT_TRUE(borrowed);
  EXPECT_EQ(borrowed->data().header->device_sequence, 2U);
  ASSERT_TRUE(publish(3));
  ASSERT_TRUE(publish(4));
  EXPECT_EQ(borrowed->data().header->device_sequence, 2U);
  EXPECT_EQ(borrowed->data().header->frame_id, std::string(128, 'c'));
  ASSERT_TRUE((*binding)->retire(event->source_sequence));
  ASSERT_TRUE((*session)->retire_activation(VertexId{2}, 1, true));
  ASSERT_TRUE((*session)->retire_samples(PortId{2}, selection.front().samples));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1}, 1, true));
  EXPECT_GT(fixture.probe.replacement_assignments.load(), 0U);
  EXPECT_EQ(fixture.probe.allocations.load(), 0U);
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

}  // namespace
}  // namespace holoscan::detail

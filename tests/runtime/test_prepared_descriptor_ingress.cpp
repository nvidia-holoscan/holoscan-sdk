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
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <holoscan/core/expected_macros.hpp>

#include "common/allocation_counter.hpp"
#include "core/operator_spec_compiler_access.hpp"
#include "core/source_id_internal.hpp"
#include "runtime/dispatcher.hpp"
#include "runtime_channel_binding_schema_tensor_test_support.hpp"
#include "support/prepared_descriptor_fixture.hpp"

namespace holoscan::detail {
namespace {
using namespace runtime_channel_binding_schema_tensor_test;
using test::prepared_descriptor::Packet;

struct SlotProbe {
  std::array<void*, 16> storage{};
  std::array<std::atomic<std::size_t>, 16> assignments{};
  std::size_t prepared{};
  std::atomic<std::size_t> allocations{};
  std::atomic<std::size_t> destroyed{};
  std::atomic<bool> unknown_slot{};
};
// GTest runs these fixtures serially. Keep the cloned ops alive through session destruction;
// never patch the shared generated type-ops table or install a production allocator hook.
SlotProbe* active_descriptor_probe{};

struct DescriptorSessionFixture {
  SessionFixture base;
  EmptySource<Packet> source;
  EmptySink<Packet> sink;
  OperatorSpec source_spec;
  OperatorSpec sink_spec;
  SlotProbe probe;
  RuntimePortTypeOps instrumented = *runtime_port_type_ops<Packet>();
  std::array<RuntimeChannelPortReplayRecord, 2> replay;

  DescriptorSessionFixture() {
    active_descriptor_probe = &probe;
    instrumented.prepare_sample_storage = [](void* storage) noexcept -> expected<void, Error> {
      auto& probe = *active_descriptor_probe;
      if (probe.prepared == probe.storage.size())
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      HOLOSCAN_TRY(runtime_port_type_ops<Packet>()->prepare_sample_storage(storage));
      probe.storage[probe.prepared++] = storage;
      return {};
    };
    instrumented.destroy_sample_storage = [](void* storage) noexcept {
      runtime_port_type_ops<Packet>()->destroy_sample_storage(storage);
      active_descriptor_probe->destroyed.fetch_add(1);
    };
    instrumented.assign_descriptor_tensors_sample =
        [](void* storage,
           std::span<const std::byte>
               wire,
           std::span<Tensor>
               tensors,
           DescriptorTensorWireMetadata wire_metadata,
           DescriptorTensorSampleMetadata sample_metadata) noexcept {
          const test::ScopedAllocationCounter counter;
          auto result = runtime_port_type_ops<Packet>()->assign_descriptor_tensors_sample(
              storage, wire, tensors, wire_metadata, sample_metadata);
          auto& probe = *active_descriptor_probe;
          probe.allocations.fetch_add(counter.count());
          bool found{};
          for (std::size_t i = 0; i < probe.prepared; ++i) {
            if (probe.storage[i] == storage) {
              probe.assignments[i].fetch_add(1);
              found = true;
              break;
            }
          }
          if (!found)
            probe.unknown_slot.store(true);
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
    service.payload_size_bytes = output.serialized_payload_capacity_bytes;
    service.tensor_payload = true;
    service.tensor_pool_id.clear();
    service.tensor_max_byte_span = 0;
    service.tensor_memory_kind = MemoryKind::kUnknown;
    service.tensor_device = {};
    plan.payload_edges[0].payload_identity = output.payload_identity;
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
  ~DescriptorSessionFixture() { active_descriptor_probe = nullptr; }
};

struct Script {
  std::vector<RuntimeEvent> events;
  std::vector<TemporalSampleRecord> records;
  std::vector<std::shared_ptr<std::atomic<std::uint32_t>>> releases;
};

expected<Script, Error> publish(DescriptorSessionFixture& fixture, RuntimeChannelSession& session,
                                PortBinding& binding, std::size_t count, bool maximal) {
  Script script;
  HOLOSCAN_TRY(session.prepare_activation(VertexId{1}, 1, {}));
  for (std::size_t i = 0; i < count; ++i) {
    auto value = test::prepared_descriptor::make_metadata(maximal ? 47 : 0);
    value.sequence = i + 1;
    HOLOSCAN_TRY(auto first, make_owned_tensor_sample(i + 1));
    HOLOSCAN_TRY(auto second, make_owned_tensor_sample(i + 101));
    script.releases.push_back(first.releases);
    script.releases.push_back(second.releases);
    value.first = std::make_shared<Tensor>(std::move(first.tensor));
    value.second = std::make_shared<Tensor>(std::move(second.tensor));
    HOLOSCAN_TRY(fixture.source.output.emit(
        std::move(value),
        {.capture_time = TimePoint{static_cast<std::int64_t>(5 * (i + 1)), ClockId{1}}}));
    HOLOSCAN_TRY(session.flush_process_local_activation_cut());
    HOLOSCAN_TRY(auto event, wait_for_event(session, binding));
    script.events.push_back(event);
    HOLOSCAN_TRY(auto record, session.temporal_sample(event.port, event.source_sequence));
    script.records.push_back(record);
  }
  HOLOSCAN_TRY(session.retire_activation(VertexId{1}, 1, true));
  return script;
}

expected<std::unique_ptr<PortBinding>, Error> make_binding(DescriptorSessionFixture& fixture,
                                                           RuntimeChannelSession& session) {
  return session.create_readiness_binding(
      VertexId{2},
      RuntimePortBindingRequest{.plan_generation = fixture.base.plan.plan_generation,
                                .run_generation = 1,
                                .kind = RuntimeEventKind::kInputArrival,
                                .source = make_source_id(1),
                                .source_clock = ClockId{1},
                                .reference_clock = ClockId{1},
                                .capacity = 8});
}

// Keep assertion formatting outside the allocation boundary, and never copy a generated value.
expected<bool, Error> check_batch(Input<Packet>& input, const Packet& metadata, std::size_t first,
                                  std::size_t size) {
  HOLOSCAN_TRY(auto batch, input.selected_batch());
  HOLOSCAN_TRY(auto repeated, input.selected_batch());
  if (batch.size() != size || repeated.size() != size)
    return false;
  for (std::size_t i = 0; i < size; ++i) {
    const auto& sample = batch[i];
    const auto sequence = first + i;
    if (&sample != &repeated[i] || sample.data.sequence != sequence || !sample.data.first ||
        !sample.data.second || !test::prepared_descriptor::same_metadata(sample.data, metadata) ||
        *sample.data.first->data_as<std::uint64_t>() != sequence ||
        *sample.data.second->data_as<std::uint64_t>() != sequence + 100 ||
        sample.metadata.capture_timestamp_ns != static_cast<std::int64_t>(sequence * 5))
      return false;
  }
  return true;
}

class PreparedDescriptorIngress : public ::testing::TestWithParam<bool> {};

// REQ: SCHEMA-009
TEST_P(PreparedDescriptorIngress, EveryReceiveSlotFirstUseAndBatchReuseAllocateNothing) {
  DescriptorSessionFixture fixture;
  fixture.base.enable_drain_all(8);
  if (GetParam())
    fixture.base.use_same_partition_eventless_transport();
  {
    auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
    ASSERT_TRUE(session) << runtime_error_message(session.error());
    const auto resources =
        RuntimeChannelSessionTestAccess::input_resource_state(**session, PortId{2});
    ASSERT_TRUE(resources);
    ASSERT_EQ(resources->payload_slot_count, 8U);
    // Runtime also cold-prepares one restart-attachment replacement object. It is not a
    // receive-queue slot; normal batch traffic must exercise every one of the eight real slots.
    ASSERT_EQ(fixture.probe.prepared, resources->payload_slot_count + 1);
    auto binding = make_binding(fixture, **session);
    ASSERT_TRUE(binding);
    ASSERT_TRUE((*session)->refresh_connections());
    ASSERT_TRUE((*session)->arm());
    for (std::size_t round = 0; round < 3; ++round) {
      const bool maximal = round != 1;
      const auto metadata = test::prepared_descriptor::make_metadata(maximal ? 47 : 0);
      auto script = publish(fixture, **session, **binding, 8, maximal);
      ASSERT_TRUE(script) << runtime_error_message(script.error());
      for (std::size_t slot = 0; slot < resources->payload_slot_count; ++slot)
        EXPECT_EQ(fixture.probe.assignments[slot].load(), round + 1);
      EXPECT_EQ(fixture.probe.assignments[resources->payload_slot_count].load(), 0U);
      auto selection = RuntimeInputSelectionState::create(fixture.base.plan.input_selections[0]);
      ASSERT_TRUE(selection);
      for (const auto& record : script->records)
        ASSERT_TRUE(selection->push(record));
      auto selected = selection->begin(TimePoint{40, ClockId{1}},
                                       script->records.back().arrival_sequence,
                                       script->records.back().producer_epoch);
      ASSERT_TRUE(selected);
      const std::array activation{
          RuntimeChannelActivationSelection{.port = PortId{2},
                                            .selection = selected->selection(),
                                            .samples = selected->selected(),
                                            .batch = true,
                                            .destructive = true}};
      ASSERT_TRUE((*session)->prepare_activation(VertexId{2}, round + 1, activation));
      ASSERT_TRUE((*session)->prepare_tensor_access(VertexId{2}, round + 1, nullptr));
      expected<bool, Error> valid;
      std::size_t allocations{};
      {
        const test::ScopedAllocationCounter counter;
        valid = check_batch(fixture.sink.input, metadata, 1, 8);
        allocations = counter.count();
      }
      ASSERT_TRUE(valid);
      EXPECT_TRUE(*valid);
      EXPECT_EQ(allocations, 0U);
      for (const auto& event : script->events)
        ASSERT_TRUE((*binding)->retire(event.source_sequence));
      ASSERT_TRUE(selected->finish(true));
      ASSERT_TRUE((*session)->retire_activation(VertexId{2}, round + 1, true));
      ASSERT_TRUE((*session)->retire_samples(PortId{2}, script->records));
      for (const auto& releases : script->releases)
        EXPECT_EQ(releases->load(), 1U);
    }
    (*session)->cancel();
    ASSERT_TRUE((*session)->join());
  }
  EXPECT_EQ(fixture.probe.allocations.load(), 0U);
  EXPECT_FALSE(fixture.probe.unknown_slot.load());
  EXPECT_EQ(fixture.probe.destroyed.load(), fixture.probe.prepared);
}

// REQ: SCHEMA-009
TEST_P(PreparedDescriptorIngress, OverlappingWindowsPreserveBothTensorFieldsWithoutNativeCopies) {
  DescriptorSessionFixture fixture;
  fixture.base.enable_bracketed_window(5);
  if (GetParam())
    fixture.base.use_same_partition_eventless_transport();
  auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
  ASSERT_TRUE(session);
  auto binding = make_binding(fixture, **session);
  ASSERT_TRUE(binding);
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  auto script = publish(fixture, **session, **binding, 7, true);
  ASSERT_TRUE(script);
  auto selection = RuntimeInputSelectionState::create(
      fixture.base.plan.input_selections[0], &fixture.base.plan.bracketed_windows[0], 2);
  ASSERT_TRUE(selection);
  for (std::size_t i = 0; i < 7; ++i) {
    if (i % 2)
      ASSERT_TRUE(selection->push_anchor(script->records[i]));
    ASSERT_TRUE(selection->push(script->records[i]));
  }
  const auto metadata = test::prepared_descriptor::make_metadata(47);
  std::array<const Sample<Packet>*, 5> previous{};
  for (std::size_t window = 0; window < 2; ++window) {
    auto selected =
        selection->begin(TimePoint{static_cast<std::int64_t>(20 + window * 10), ClockId{1}},
                         script->records[4 + 2 * window].arrival_sequence,
                         script->records[4 + 2 * window].producer_epoch);
    ASSERT_TRUE(selected);
    const std::array activation{
        RuntimeChannelActivationSelection{.port = PortId{2},
                                          .selection = selected->selection(),
                                          .samples = selected->selected(),
                                          .batch = true,
                                          .destructive = false}};
    ASSERT_TRUE((*session)->prepare_activation(VertexId{2}, window + 1, activation));
    ASSERT_TRUE((*session)->prepare_tensor_access(VertexId{2}, window + 1, nullptr));
    expected<bool, Error> valid;
    std::size_t allocations{};
    {
      const test::ScopedAllocationCounter counter;
      valid = check_batch(fixture.sink.input, metadata, 1 + 2 * window, 5);
      allocations = counter.count();
    }
    ASSERT_TRUE(valid);
    EXPECT_TRUE(*valid);
    EXPECT_EQ(allocations, 0U);
    auto batch = fixture.sink.input.selected_batch();
    ASSERT_TRUE(batch);
    for (std::size_t i = 0; i < 5; ++i) {
      if (window == 0)
        previous[i] = &(*batch)[i];
      else if (i < 3)
        EXPECT_EQ(&(*batch)[i], previous[i + 2]);
    }
    ASSERT_TRUE(selected->finish(true));
    ASSERT_TRUE((*session)->retire_activation(VertexId{2}, window + 1, true));
    for (const auto& releases : script->releases)
      EXPECT_EQ(releases->load(), 0U);
  }
  ASSERT_TRUE(selection->reset());
  for (const auto& event : script->events)
    ASSERT_TRUE((*binding)->retire(event.source_sequence));
  ASSERT_TRUE((*session)->retire_samples(PortId{2}, script->records));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
  for (const auto& releases : script->releases)
    EXPECT_EQ(releases->load(), 1U);
  EXPECT_EQ(fixture.probe.allocations.load(), 0U);
}

// REQ: SCHEMA-008
// REQ: SCHEMA-009
TEST_P(PreparedDescriptorIngress, ReuseLastKeepsBackingButMetadataScratchDoesNot) {
  DescriptorSessionFixture fixture;
  fixture.base.enable_reuse_last();
  if (GetParam())
    fixture.base.use_same_partition_eventless_transport();
  auto session = RuntimeChannelSession::create(fixture.base.plan, fixture.replay);
  ASSERT_TRUE(session);
  auto binding = make_binding(fixture, **session);
  ASSERT_TRUE(binding);
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  auto script = publish(fixture, **session, **binding, 1, true);
  ASSERT_TRUE(script);
  auto selection = RuntimeInputSelectionState::create(fixture.base.plan.input_selections[0]);
  ASSERT_TRUE(selection);
  ASSERT_TRUE(selection->push(script->records.front()));
  auto scratch = test::prepared_descriptor::Scratch::create();
  ASSERT_TRUE(scratch);
  const auto metadata = test::prepared_descriptor::make_metadata(47);
  const Sample<Packet>* previous{};
  for (std::size_t pass = 0; pass < 2; ++pass) {
    auto selected = selection->begin(TimePoint{pass == 0 ? 5 : 10, ClockId{1}},
                                     script->records.front().arrival_sequence,
                                     script->records.front().producer_epoch);
    ASSERT_TRUE(selected);
    EXPECT_EQ(selected->selection().state, pass == 0 ? InputState::kSelected : InputState::kReused);
    const std::array activation{
        RuntimeChannelActivationSelection{.port = PortId{2},
                                          .selection = selected->selection(),
                                          .samples = selected->selected(),
                                          .destructive = true}};
    ASSERT_TRUE((*session)->prepare_activation(VertexId{2}, pass + 1, activation));
    ASSERT_TRUE((*session)->prepare_tensor_access(VertexId{2}, pass + 1, nullptr));
    auto view = fixture.sink.input.selected_sample();
    ASSERT_TRUE(view);
    if (pass == 0)
      previous = &view->sample();
    else
      EXPECT_EQ(previous, &view->sample());
    const auto& value = view->data();
    ASSERT_TRUE(value.first);
    ASSERT_TRUE(value.second);
    const auto first_owners = value.first.use_count();
    const auto second_owners = value.second.use_count();
    std::size_t allocations{};
    expected<void, Error> copied;
    {
      const test::ScopedAllocationCounter counter;
      copied = scratch->copy_metadata_from(value);
      allocations = counter.count();
    }
    ASSERT_TRUE(copied);
    EXPECT_EQ(allocations, 0U);
    EXPECT_EQ(value.first.use_count(), first_owners);
    EXPECT_EQ(value.second.use_count(), second_owners);
    EXPECT_FALSE(scratch->descriptor().first);
    EXPECT_FALSE(scratch->descriptor().second);
    EXPECT_TRUE(test::prepared_descriptor::same_metadata(scratch->descriptor(), metadata));
    EXPECT_EQ(*value.first->data_as<std::uint64_t>(), 1U);
    EXPECT_EQ(*value.second->data_as<std::uint64_t>(), 101U);
    if (pass == 0)
      ASSERT_TRUE((*binding)->retire(script->events.front().source_sequence));
    ASSERT_TRUE(selected->finish(true));
    ASSERT_TRUE((*session)->retire_activation(VertexId{2}, pass + 1, true));
    for (const auto& releases : script->releases)
      EXPECT_EQ(releases->load(), 0U);
  }
  const auto retained = (*session)->retire_samples(PortId{2}, script->records);
  ASSERT_FALSE(retained);
  EXPECT_EQ(retained.error().code, ErrorCode::kBackpressured);
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
  ASSERT_TRUE((*session)->reset(VertexId{2}));
  for (const auto& releases : script->releases)
    EXPECT_EQ(releases->load(), 1U);
  // Ordinary copied fields remain available after both Tensor backings have been released.
  EXPECT_TRUE(test::prepared_descriptor::same_metadata(scratch->descriptor(), metadata));
  EXPECT_EQ(fixture.probe.allocations.load(), 0U);
}

INSTANTIATE_TEST_SUITE_P(PreparedPaths, PreparedDescriptorIngress, ::testing::Bool());

}  // namespace
}  // namespace holoscan::detail

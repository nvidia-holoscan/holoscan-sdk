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
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/memory/block_memory_pool.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/time/manual_clock.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "../../src/channel/channel_common.hpp"
#include "../../src/channel/channel_tensor_test_access.hpp"
#include "../../src/channel/endpoint_catalog_state.hpp"
#include "../common/cuda_test_utils.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_state.hpp"
#include "compiler/runtime_package_capabilities.hpp"
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

// Catalog Tensor ingress is driven by a publisher outside this Runtime session. Its typed receive
// cannot serve as evidence that a session-owned process-local publication has completed its cut.
// REQ: RUNTIME-029
// REQ: RUNTIME-063
// REQ: RUNTIME-078
// REQ: RUNTIME-082
TEST(ChannelBindingParityTest, CatalogTensorImportDoesNotAdvanceProcessLocalProgress) {
  const std::string endpoint_name = unique_service();
  Graph graph{"catalog-tensor-progress-isolation"};
  const ClockKey clock = graph.bind_shared_clock("runtime-clock", std::make_shared<ManualClock>(0));
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
  const auto imported = graph.import_topic<Tensor>("holomq://" + endpoint_name, std::move(options));
  const auto sink = graph.op<TensorSinkOperator>("sink");
  graph.add_flow(imported.output(), sink->input);

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  const auto* input = planned_operator_port(*state, PortDirection::kInput);
  ASSERT_NE(input, nullptr);
  const std::array replay_ports{RuntimeChannelPortReplayRecord{
      .port = input->id,
      .owner = input->owner,
      .direction = PortDirection::kInput,
      .declaration = const_cast<void*>(static_cast<const void*>(&sink->input)),
      .type_ops = runtime_port_type_ops<Tensor>(),
  }};
  auto session = RuntimeChannelSession::create(*state, replay_ports);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto client_runtime = ChannelRuntime::create();
  ASSERT_TRUE(client_runtime) << runtime_error_message(client_runtime.error());
  auto publisher = client_runtime->create_publisher<Tensor>(plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());
  ASSERT_TRUE(wait_for_catalog_subscriber(*publisher));
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  ASSERT_TRUE(wait_for_catalog_subscriber(*publisher));
  const auto pump_state = RuntimeChannelSessionTestAccess::pump_index_state(**session);
  ASSERT_TRUE(pump_state) << runtime_error_message(pump_state.error());
  EXPECT_EQ(pump_state->drain_subscriber_count, 1U);
  EXPECT_TRUE(pump_state->background_thread_required);
  EXPECT_TRUE(pump_state->background_thread_running);
  EXPECT_TRUE(pump_state->exact);

  // Key setup: let the production pump retain one externally published host Tensor.
  expected<void, Error> published = make_unexpected(Error{ErrorCode::kNotReady});
  const auto publish_deadline = std::chrono::steady_clock::now() + 2s;
  do {
    published = publish_catalog_host_tensor(*publisher, std::byte{0x31});
    if (published || (published.error().code != ErrorCode::kNotReady &&
                      published.error().code != ErrorCode::kPublicationPartial)) {
      break;
    }
    ASSERT_TRUE(publisher->update_connections());
    std::this_thread::sleep_for(1ms);
  } while (std::chrono::steady_clock::now() < publish_deadline);
  ASSERT_TRUE(published) << runtime_error_message(published.error());
  const auto deadline = std::chrono::steady_clock::now() + 2s;
  expected<RuntimeChannelInputResetState, Error> retained =
      make_unexpected(Error{ErrorCode::kNotReady});
  do {
    retained = RuntimeChannelSessionTestAccess::input_reset_state(**session, input->id);
    ASSERT_TRUE(retained) << runtime_error_message(retained.error());
    if (retained->occupied_samples == 0U) {
      std::this_thread::sleep_for(1ms);
    }
  } while (retained->occupied_samples == 0U && std::chrono::steady_clock::now() < deadline);
  ASSERT_EQ(retained->occupied_samples, 1U);

  const auto progress = RuntimeChannelSessionTestAccess::process_local_progress_state(**session);
  ASSERT_TRUE(progress) << runtime_error_message(progress.error());
  // Key assertion: the catalog receive leaves local graph delivery accounting untouched.
  EXPECT_EQ(progress->published_deliveries, 0U);
  EXPECT_EQ(progress->received_deliveries, 0U);
  EXPECT_EQ(progress->publications_in_flight, 0U);
  EXPECT_TRUE(progress->tracking_valid);
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// A host catalog Tensor kDropOldest cache materializes the replacement before destroying its
// previous retained Tensor. Runtime therefore reserves one receive-control slot beyond the typed
// cache and one readiness credit for ordered EOS. Repeated observation of that exact EOS is
// idempotent, while a conflicting terminal sequence remains a protocol error. REQ: CHAN-078 REQ:
// RUNTIME-025 REQ: SYSTEM-006
TEST(ChannelBindingParityTest, HostCatalogTensorDropOldestReplacementRetainsOrderedTerminalRecord) {
  const std::string endpoint_name = unique_service();
  Graph graph{"host-catalog-tensor-drop-oldest-replacement"};
  const ClockKey clock = graph.bind_shared_clock("runtime-clock", std::make_shared<ManualClock>(0));
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
  const auto imported = graph.import_topic<Tensor>("holomq://" + endpoint_name, std::move(options));
  const auto sink = graph.op<TensorSinkOperator>("sink");
  graph.add_flow(imported.output(),
                 sink->input,
                 ConnectionOptions{.queue_policy = QueuePolicy::kDropOldest, .queue_depth = 1U});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  const auto* input = planned_operator_port(*state, PortDirection::kInput);
  ASSERT_NE(input, nullptr);
  const std::array replay_ports{RuntimeChannelPortReplayRecord{
      .port = input->id,
      .owner = input->owner,
      .direction = PortDirection::kInput,
      .declaration = const_cast<void*>(static_cast<const void*>(&sink->input)),
      .type_ops = runtime_port_type_ops<Tensor>(),
  }};
  auto session = RuntimeChannelSession::create(*state, replay_ports);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto readiness_capacity = (*session)->readiness_capacity(input->owner);
  ASSERT_TRUE(readiness_capacity) << runtime_error_message(readiness_capacity.error());
  EXPECT_EQ(*readiness_capacity, 2U);

  auto client_runtime = ChannelRuntime::create();
  ASSERT_TRUE(client_runtime) << runtime_error_message(client_runtime.error());
  auto publisher = client_runtime->create_publisher<Tensor>(plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());
  ASSERT_TRUE(wait_for_catalog_subscriber(*publisher));

  ASSERT_TRUE(publish_catalog_host_tensor(*publisher, std::byte{0x11}));
  auto first_received = RuntimeChannelSessionTestAccess::receive_one(**session, input->id);
  ASSERT_TRUE(first_received) << runtime_error_message(first_received.error());
  ASSERT_TRUE(publish_catalog_host_tensor(*publisher, std::byte{0x22}));
  auto replacement_received = RuntimeChannelSessionTestAccess::receive_one(**session, input->id);
  ASSERT_TRUE(replacement_received) << runtime_error_message(replacement_received.error());
  auto retained = RuntimeChannelSessionTestAccess::input_reset_state(**session, input->id);
  ASSERT_TRUE(retained) << runtime_error_message(retained.error());
  EXPECT_EQ(retained->occupied_samples, 1U);
  EXPECT_EQ(retained->next_arrival_sequence, 3U);

  ASSERT_TRUE(wait_for_catalog_subscriber(*publisher));
  const auto closed = publisher->close();
  ASSERT_TRUE(closed) << runtime_error_message(closed.error());
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::receive_one(**session, input->id));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::receive_one(**session, input->id));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::store_end_of_stream(**session, input->id, 3U));
  auto conflicting = RuntimeChannelSessionTestAccess::store_end_of_stream(**session, input->id, 4U);
  ASSERT_FALSE(conflicting);
  EXPECT_EQ(conflicting.error().code, ErrorCode::kProtocolError);

  auto binding = (*session)->create_readiness_binding(input->owner,
                                                      RuntimePortBindingRequest{
                                                          .plan_generation = state->plan_generation,
                                                          .run_generation = 1U,
                                                          .kind = RuntimeEventKind::kInputArrival,
                                                          .source = make_source_id(1U),
                                                          .source_clock = ClockId{1U},
                                                          .reference_clock = ClockId{1U},
                                                          .capacity = *readiness_capacity,
                                                      });
  ASSERT_TRUE(binding) << runtime_error_message(binding.error());
  ASSERT_TRUE((*binding)->poll(TimePoint{1, ClockId{1U}}));
  std::array<RuntimeEvent, 2U> events{};
  auto drained = (*binding)->drain(events);
  ASSERT_TRUE(drained) << runtime_error_message(drained.error());
  ASSERT_EQ(*drained, 2U);
  EXPECT_EQ(events[0].kind, RuntimeEventKind::kInputArrival);
  EXPECT_EQ(events[1].kind, RuntimeEventKind::kInputEndOfStream);
  auto sample = (*session)->temporal_sample(events[0].port, events[0].source_sequence);
  ASSERT_TRUE(sample) << runtime_error_message(sample.error());
  EXPECT_EQ(sample->source_sequence, 2U);
  ASSERT_TRUE((*binding)->retire(events[0].source_sequence));
  ASSERT_TRUE((*binding)->retire(events[1].source_sequence));
  (*session)->cancel();
  EXPECT_TRUE((*session)->join());
}

// Catalog Tensor import readiness derives from the complete typed retention geometry rather than
// a capacity-one shortcut. Every finite kDropOldest cache reserves its N data candidates plus one
// independent ordered terminal witness.
// REQ: CHAN-078
// REQ: RUNTIME-025
TEST(ChannelBindingParityTest, CatalogTensorDropOldestReadinessScalesWithTypedCacheDepth) {
  constexpr std::array<std::uint32_t, 3U> kQueueDepths{1U, 2U, 3U};
  for (const std::uint32_t queue_depth : kQueueDepths) {
    SCOPED_TRACE(queue_depth);
    const std::string endpoint_name = unique_service();
    Graph graph{"catalog-tensor-drop-oldest-readiness-capacity"};
    const ClockKey clock =
        graph.bind_shared_clock("runtime-clock", std::make_shared<ManualClock>(0));
    graph.set_default_clock(clock);
    constexpr std::array axes{TensorAxisBounds{.max_extent = 8U, .max_abs_stride_bytes = 1U}};
    TopicOptions options;
    options.endpoint_name = endpoint_name;
    options.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = queue_depth + 1U};
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
    graph.add_flow(imported.output(),
                   sink->input,
                   ConnectionOptions{
                       .queue_policy = QueuePolicy::kDropOldest,
                       .queue_depth = queue_depth,
                   });

    const ExecutionPlan plan = compile(graph);
    ASSERT_TRUE(plan.ok()) << plan.json();
    const auto state = ExecutionPlanAccess::retain(plan);
    ASSERT_NE(state, nullptr);
    const auto* input = planned_operator_port(*state, PortDirection::kInput);
    ASSERT_NE(input, nullptr);
    const std::array replay_ports{RuntimeChannelPortReplayRecord{
        .port = input->id,
        .owner = input->owner,
        .direction = PortDirection::kInput,
        .declaration = const_cast<void*>(static_cast<const void*>(&sink->input)),
        .type_ops = runtime_port_type_ops<Tensor>(),
    }};
    auto session = RuntimeChannelSession::create(*state, replay_ports);
    ASSERT_TRUE(session) << runtime_error_message(session.error());
    auto readiness_capacity = (*session)->readiness_capacity(input->owner);
    ASSERT_TRUE(readiness_capacity) << runtime_error_message(readiness_capacity.error());
    EXPECT_EQ(*readiness_capacity, queue_depth + 1U);
  }
}

// An externally driven catalog data operation advances its ordered service head before Runtime
// takes the cache mutex. The exact occupied replacement destination must therefore remain reserved
// across that boundary instead of being recomputed after the irreversible receive.
// REQ: RUNTIME-026
// REQ: RUNTIME-042
// REQ: RUNTIME-044
TEST(ChannelBindingParityTest, ExternalCatalogIngressRetainsDestinationAfterOrderedReceive) {
  // Key setup: pause a one-slot drop-oldest catalog operation after ordered receive advances but
  // before its newer scalar value commits to the typed cache.
  const std::string endpoint_name = unique_service();
  Graph graph{"external-catalog-ingress-transaction"};
  const ClockKey clock = graph.bind_shared_clock("runtime-clock", std::make_shared<ManualClock>(0));
  graph.set_default_clock(clock);
  TopicOptions options;
  options.endpoint_name = endpoint_name;
  options.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = 2U};
  const auto imported =
      graph.import_topic<std::int32_t>("holomq://" + endpoint_name, std::move(options));
  const auto sink = graph.op<SinkOperator>("sink");
  auto& sink_input = const_cast<Input<std::int32_t>&>(sink->input);
  graph.add_flow(imported.output(),
                 sink->input,
                 ConnectionOptions{
                     .queue_policy = QueuePolicy::kDropOldest,
                     .queue_depth = 1U,
                 });

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
      .type_ops = runtime_port_type_ops<std::int32_t>(),
  }};
  BlockingTypedIngressObserver observer;
  auto session = RuntimeChannelSession::create(*state, replay_ports);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto client_runtime = ChannelRuntime::create();
  ASSERT_TRUE(client_runtime) << runtime_error_message(client_runtime.error());
  auto publisher =
      client_runtime->create_publisher<std::int32_t>(plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());
  ASSERT_TRUE(wait_for_catalog_subscriber(*publisher));
  const auto publish_value = [&](const std::int32_t value) -> expected<void, Error> {
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    while (std::chrono::steady_clock::now() < deadline) {
      auto published = publisher->emit(value);
      if (published) {
        return {};
      }
      if (published.error().code != ErrorCode::kNotReady) {
        return make_unexpected(std::move(published).error());
      }
      auto refreshed = publisher->update_connections();
      if (!refreshed) {
        return make_unexpected(std::move(refreshed).error());
      }
      std::this_thread::sleep_for(1ms);
    }
    return make_unexpected(Error{ErrorCode::kTimeout});
  };
  ASSERT_TRUE(publish_value(11));
  ASSERT_TRUE(wait_for_catalog_input(**session, input->id));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_pre_witness_replacement_allowed(
      **session, input->id, true));

  observer.block.store(true, std::memory_order_release);
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_typed_ingress_observer(
      **session, input->id, &observer, &BlockingTypedIngressObserver::observe));
  ASSERT_TRUE(publish_value(22));
  expected<void, Error> drain_result = make_unexpected(Error{ErrorCode::kNotReady});
  std::thread receiver([&] {
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    while (!observer.entered.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline) {
      drain_result =
          RuntimeChannelSessionTestAccess::drain_external_inputs(**session, input->owner);
      if (!drain_result) {
        return;
      }
      std::this_thread::yield();
    }
  });
  const auto observer_deadline = std::chrono::steady_clock::now() + 2s;
  while (!observer.entered.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < observer_deadline) {
    std::this_thread::yield();
  }

  // Key assertion: the ordered receive owns the exact incumbent slot through commit, and the
  // arriving value replaces it without recomputing capacity.
  const auto reservations =
      RuntimeChannelSessionTestAccess::typed_ingress_reservations(**session, input->id);
  const auto retained = RuntimeChannelSessionTestAccess::input_reset_state(**session, input->id);
  observer.release.store(true, std::memory_order_release);
  receiver.join();
  ASSERT_TRUE(observer.entered.load(std::memory_order_acquire));
  ASSERT_TRUE(reservations) << runtime_error_message(reservations.error());
  EXPECT_EQ(*reservations, 1U);
  ASSERT_TRUE(retained) << runtime_error_message(retained.error());
  EXPECT_EQ(retained->occupied_samples, 1U);
  EXPECT_EQ(retained->next_arrival_sequence, 2U);
  ASSERT_TRUE(drain_result) << runtime_error_message(drain_result.error());
  const auto released =
      RuntimeChannelSessionTestAccess::typed_ingress_reservations(**session, input->id);
  ASSERT_TRUE(released) << runtime_error_message(released.error());
  EXPECT_EQ(*released, 0U);

  auto binding = (*session)->create_readiness_binding(input->owner,
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
  auto sample = sink_input.receive();
  ASSERT_TRUE(sample) << runtime_error_message(sample.error());
  EXPECT_EQ(sample->data, 22);
  EXPECT_EQ(sample->metadata.sequence, 2U);
  ASSERT_TRUE((*binding)->retire(event.source_sequence));
  ASSERT_TRUE((*session)->retire_activation(input->owner, 1U, true));
  ASSERT_TRUE(
      (*session)->retire_samples(input->id, std::span<const TemporalSampleRecord>{&*record, 1U}));

  // Key assertion: a non-data catalog head releases the speculative data reservation before
  // retaining ordered control state; with the data slot free, close exercises that release.
  ASSERT_TRUE(publisher->close());
  expected<RuntimeChannelInputResetState, Error> closed_state =
      make_unexpected(Error{ErrorCode::kNotReady});
  const auto close_deadline = std::chrono::steady_clock::now() + 2s;
  while (std::chrono::steady_clock::now() < close_deadline) {
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::drain_external_inputs(**session, input->owner));
    closed_state = RuntimeChannelSessionTestAccess::input_reset_state(**session, input->id);
    if (closed_state && closed_state->end_of_stream_received) {
      break;
    }
    std::this_thread::yield();
  }
  ASSERT_TRUE(closed_state) << runtime_error_message(closed_state.error());
  ASSERT_TRUE(closed_state->end_of_stream_received);
  const auto control_reservations =
      RuntimeChannelSessionTestAccess::typed_ingress_reservations(**session, input->id);
  ASSERT_TRUE(control_reservations) << runtime_error_message(control_reservations.error());
  EXPECT_EQ(*control_reservations, 0U);
  (*session)->cancel();
  EXPECT_TRUE((*session)->join());
}

// One OutputState validates every fanout arm against one immutable Tensor contract. A second
// catalog endpoint with different geometry must not replace the contract established by the first.
// REQ: SYSTEM-001
TEST(ChannelBindingParityTest, CatalogTensorFanoutRejectsConflictingOutputContracts) {
  Graph graph{"catalog-tensor-contract-fanout"};
  const ClockKey default_clock =
      graph.bind_shared_clock("runtime-clock", std::make_shared<ManualClock>(0));
  graph.set_default_clock(default_clock);
  const auto source = graph.op<ContractTensorSourceOperator>("source");
  constexpr std::array first_axes{TensorAxisBounds{.max_extent = 8U, .max_abs_stride_bytes = 1U}};
  TopicOptions first_options;
  first_options.tensor = TensorBoundaryContract{
      .representation = {.memory_kind = MemoryKind::kHost,
                         .dtype = DLDataType{kDLUInt, 8U, 1U},
                         .rank = 1U,
                         .strides = TensorStridePolicy::kContiguousOnly},
      .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 8U, .axes = first_axes},
      .transfer =
          TensorTransferPolicy{.foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded},
  };
  constexpr std::array second_axes{TensorAxisBounds{.max_extent = 16U, .max_abs_stride_bytes = 1U}};
  TopicOptions second_options;
  second_options.tensor = TensorBoundaryContract{
      .representation = {.memory_kind = MemoryKind::kHost,
                         .dtype = DLDataType{kDLUInt, 8U, 1U},
                         .rank = 1U,
                         .strides = TensorStridePolicy::kContiguousOnly},
      .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 16U, .axes = second_axes},
      .transfer =
          TensorTransferPolicy{.foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded},
  };
  auto first = graph.export_topic<Tensor>("holomq://" + unique_service(), std::move(first_options));
  auto second =
      graph.export_topic<Tensor>("holomq://" + unique_service(), std::move(second_options));
  graph.add_flow(source->output, first.input());
  graph.add_flow(source->output, second.input());

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  const auto* planned_output = planned_operator_port(*state, PortDirection::kOutput);
  ASSERT_NE(planned_output, nullptr);
  const std::array replay_ports{RuntimeChannelPortReplayRecord{
      .port = planned_output->id,
      .owner = planned_output->owner,
      .direction = PortDirection::kOutput,
      .declaration = const_cast<void*>(static_cast<const void*>(&source->output)),
      .type_ops = runtime_port_type_ops<Tensor>(),
  }};

  auto session = RuntimeChannelSession::create(*state, replay_ports);
  ASSERT_FALSE(session);
  EXPECT_EQ(session.error().code, ErrorCode::kProtocolError);
}

// Runtime rejects a catalog whose service-static Tensor policy differs from the endpoint and
// source-Graph boundary contract only by memory conversion. The same mismatch is checked at both
// executor reconstruction and Channel binding startup.
// REQ: SYSTEM-023
TEST(ChannelBindingParityTest, CatalogTensorRejectsMemoryConversionProjectionMismatch) {
  Graph graph{"catalog-tensor-conversion-projection-mismatch"};
  const ClockKey default_clock =
      graph.bind_shared_clock("runtime-clock", std::make_shared<ManualClock>(0));
  graph.set_default_clock(default_clock);
  const auto source = graph.op<ContractTensorSourceOperator>("source");
  TopicOptions options;
  options.tensor = TensorBoundaryContract{
      .representation = {.memory_kind = MemoryKind::kHost,
                         .dtype = DLDataType{kDLUInt, 8U, 1U},
                         .rank = 1U,
                         .strides = TensorStridePolicy::kContiguousOnly},
      .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 16U},
      .transfer =
          TensorTransferPolicy{.foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded},
      .memory_conversion = MemoryConversionPolicy::kDisable,
  };
  auto exported = graph.export_topic<Tensor>("holomq://" + unique_service(), std::move(options));
  graph.add_flow(source->output, exported.input());

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  const auto* catalog_state = ChannelEndpointCatalogAccess::state(state->endpoint_catalog);
  ASSERT_NE(catalog_state, nullptr);
  ASSERT_EQ(catalog_state->endpoints.size(), 1U);

  auto corrupted_catalog = std::make_shared<ChannelEndpointCatalogState>(*catalog_state);
  ASSERT_TRUE(corrupted_catalog->endpoints.front().tensor);
  ASSERT_TRUE(corrupted_catalog->endpoints.front().service_static.tensor);
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  corrupted_catalog->endpoints.front().service_static.tensor->memory_conversion =
      MemoryConversionPolicy::kAllowIfSafe;
  auto corrupted_state = std::make_shared<ExecutionPlanState>(*state);
  corrupted_state->endpoint_catalog = ChannelEndpointCatalogAccess::make(corrupted_catalog);

  auto reconstructed = GraphExec::reconstruct(corrupted_state, RunGenerationId{1U});
  ASSERT_FALSE(reconstructed);
  EXPECT_EQ(reconstructed.error().message, "RUNTIME_EXTERNAL_BINDING_UNAVAILABLE");

  const auto* planned_output = planned_operator_port(*corrupted_state, PortDirection::kOutput);
  ASSERT_NE(planned_output, nullptr);
  ContractTensorSourceOperator runtime_source;
  OperatorSpec runtime_source_spec;
  runtime_source.setup(runtime_source_spec);
  const std::array replay_ports{RuntimeChannelPortReplayRecord{
      .port = planned_output->id,
      .owner = planned_output->owner,
      .direction = PortDirection::kOutput,
      .declaration = static_cast<void*>(&runtime_source.output),
      .type_ops = runtime_port_type_ops<Tensor>(),
  }};
  auto session = RuntimeChannelSession::create(*corrupted_state, replay_ports);
  ASSERT_FALSE(session);
  EXPECT_EQ(session.error().code, ErrorCode::kProtocolError);
}

// Every catalog endpoint is internally consistent, but one output cannot adopt two different
// consumer memory-conversion policies across its external fanout.
// REQ: SYSTEM-023
TEST(ChannelBindingParityTest, CatalogTensorFanoutRejectsConflictingMemoryConversion) {
  Graph graph{"catalog-tensor-conversion-fanout"};
  const ClockKey default_clock =
      graph.bind_shared_clock("runtime-clock", std::make_shared<ManualClock>(0));
  graph.set_default_clock(default_clock);
  const auto source = graph.op<ContractTensorSourceOperator>("source");
  const auto make_options = [](const MemoryConversionPolicy conversion) {
    TopicOptions options;
    options.tensor = TensorBoundaryContract{
        .representation = {.memory_kind = MemoryKind::kHost,
                           .dtype = DLDataType{kDLUInt, 8U, 1U},
                           .rank = 1U,
                           .strides = TensorStridePolicy::kContiguousOnly},
        .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 16U},
        .transfer =
            TensorTransferPolicy{.foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded},
        .memory_conversion = conversion,
    };
    return options;
  };
  auto first = graph.export_topic<Tensor>("holomq://" + unique_service(),
                                          make_options(MemoryConversionPolicy::kDisable));
  auto second = graph.export_topic<Tensor>("holomq://" + unique_service(),
                                           make_options(MemoryConversionPolicy::kAllowIfSafe));
  graph.add_flow(source->output, first.input());
  graph.add_flow(source->output, second.input());

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  const auto* planned_output = planned_operator_port(*state, PortDirection::kOutput);
  ASSERT_NE(planned_output, nullptr);
  ContractTensorSourceOperator runtime_source;
  OperatorSpec runtime_source_spec;
  runtime_source.setup(runtime_source_spec);
  const std::array replay_ports{RuntimeChannelPortReplayRecord{
      .port = planned_output->id,
      .owner = planned_output->owner,
      .direction = PortDirection::kOutput,
      .declaration = static_cast<void*>(&runtime_source.output),
      .type_ops = runtime_port_type_ops<Tensor>(),
  }};

  auto session = RuntimeChannelSession::create(*state, replay_ports);
  ASSERT_FALSE(session);
  EXPECT_EQ(session.error().code, ErrorCode::kProtocolError);
}

// Catalog fanout stages one foreign Tensor copy and shares its exportable BufferHandle across
// every external service. Re-emitting an imported HoloMQ Tensor stays on the zero-copy path.
// REQ: SYSTEM-006
// REQ: SYSTEM-007
// REQ: SYSTEM-009
TEST(ChannelBindingParityTest, CatalogTensorFanoutCopiesOnceAndReusesExportableBacking) {
  Graph graph{"catalog-tensor-shared-export"};
  const ClockKey default_clock =
      graph.bind_shared_clock("runtime-clock", std::make_shared<ManualClock>(0));
  graph.set_default_clock(default_clock);
  auto source = graph.op<ContractTensorSourceOperator>("source");
  static constexpr std::array axes{TensorAxisBounds{.max_extent = 16U, .max_abs_stride_bytes = 1U}};
  const auto make_options = [&](const std::string& endpoint_name) {
    TopicOptions options;
    options.endpoint_name = endpoint_name;
    options.bounds.subscriber = SubscriberEndpointBounds{
        .queue_depth = 2U,
        .max_borrowed_samples = 2U,
    };
    options.tensor = TensorBoundaryContract{
        .representation = {.memory_kind = MemoryKind::kHost,
                           .dtype = DLDataType{kDLUInt, 8U, 1U},
                           .rank = 1U,
                           .strides = TensorStridePolicy::kContiguousOnly},
        .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 16U, .axes = axes},
        .transfer =
            TensorTransferPolicy{.foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded},
    };
    return options;
  };
  const std::string missing_name = unique_service();
  const std::string first_name = unique_service();
  const std::string second_name = unique_service();
  auto missing = graph.export_topic<Tensor>("holomq://" + missing_name, make_options(missing_name));
  auto first = graph.export_topic<Tensor>("holomq://" + first_name, make_options(first_name));
  auto second = graph.export_topic<Tensor>("holomq://" + second_name, make_options(second_name));
  graph.add_flow(source->output, missing.input());
  graph.add_flow(source->output, first.input());
  graph.add_flow(source->output, second.input());

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  const auto* planned_output = planned_operator_port(*state, PortDirection::kOutput);
  ASSERT_NE(planned_output, nullptr);
  ContractTensorSourceOperator runtime_source;
  OperatorSpec runtime_source_spec;
  runtime_source.setup(runtime_source_spec);
  const std::array replay_ports{RuntimeChannelPortReplayRecord{
      .port = planned_output->id,
      .owner = planned_output->owner,
      .direction = PortDirection::kOutput,
      .declaration = static_cast<void*>(&runtime_source.output),
      .type_ops = runtime_port_type_ops<Tensor>(),
  }};
  auto session = RuntimeChannelSession::create(*state, replay_ports);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto client_runtime = ChannelRuntime::create();
  ASSERT_TRUE(client_runtime) << runtime_error_message(client_runtime.error());
  auto first_subscriber =
      client_runtime->create_subscriber<Tensor>(plan.endpoint_catalog(), first_name);
  auto second_subscriber =
      client_runtime->create_subscriber<Tensor>(plan.endpoint_catalog(), second_name);
  ASSERT_TRUE(first_subscriber) << runtime_error_message(first_subscriber.error());
  ASSERT_TRUE(second_subscriber) << runtime_error_message(second_subscriber.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());

  constexpr std::size_t kByteOffset = 2U;
  auto tracked = make_host_copy_tracking_tensor(kByteOffset);
  ASSERT_NE(tracked.tensor.data(), nullptr);
  ASSERT_TRUE((*session)->prepare_activation(planned_output->owner, 1U, {}));
  ASSERT_TRUE(runtime_source.output.emit(Sample<Tensor>{tracked.tensor, {}}));
  ASSERT_TRUE((*session)->retire_activation(planned_output->owner, 1U, true));
  auto first_received = receive_catalog_tensor(*first_subscriber);
  auto second_received = receive_catalog_tensor(*second_subscriber);
  ASSERT_TRUE(first_received) << runtime_error_message(first_received.error());
  ASSERT_TRUE(second_received) << runtime_error_message(second_received.error());
  EXPECT_EQ(tracked.state->host_synchronizations.load(std::memory_order_relaxed), 1U);
  const std::size_t logical_byte_size = tracked.state->data.size() - kByteOffset;
  ASSERT_EQ(first_received->data.nbytes(), static_cast<std::int64_t>(logical_byte_size));
  ASSERT_EQ(second_received->data.nbytes(), static_cast<std::int64_t>(logical_byte_size));
  EXPECT_EQ(
      std::memcmp(
          first_received->data.data(), tracked.state->data.data() + kByteOffset, logical_byte_size),
      0);
  EXPECT_EQ(std::memcmp(second_received->data.data(),
                        tracked.state->data.data() + kByteOffset,
                        logical_byte_size),
            0);
  EXPECT_EQ(first_received->data.data(), second_received->data.data());

  ASSERT_TRUE((*session)->prepare_activation(planned_output->owner, 2U, {}));
  ASSERT_TRUE(runtime_source.output.emit(Sample<Tensor>{first_received->data, {}}));
  ASSERT_TRUE((*session)->retire_activation(planned_output->owner, 2U, true));
  auto republished = receive_catalog_tensor(*first_subscriber);
  ASSERT_TRUE(republished) << runtime_error_message(republished.error());
  EXPECT_EQ(
      std::memcmp(
          republished->data.data(), tracked.state->data.data() + kByteOffset, logical_byte_size),
      0);
  EXPECT_EQ(tracked.state->host_synchronizations.load(std::memory_order_relaxed), 1U);
}

// A graph-bound device Tensor uses its retained producer stream to stage one ordered D2H copy.
// Both catalog services publish the same host-pageable backing rather than copying per endpoint.
// REQ: CHAN-091
// REQ: SYSTEM-007
TEST(ChannelBindingParityTest, CatalogTensorDeviceFanoutStagesOnceAfterProducerStream) {
  test::skip_without_cuda_device();

  int device_ordinal = -1;
  ASSERT_EQ(cudaGetDevice(&device_ordinal), cudaSuccess);

  Graph graph{"catalog-tensor-device-export"};
  const ClockKey default_clock =
      graph.bind_shared_clock("runtime-clock", std::make_shared<ManualClock>(0));
  graph.set_default_clock(default_clock);
  auto source = graph.op<DeviceContractTensorSourceOperator>("source");
  static constexpr std::array axes{TensorAxisBounds{.max_extent = 16U, .max_abs_stride_bytes = 1U}};
  const auto make_options = [&](const std::string& endpoint_name) {
    TopicOptions options;
    options.endpoint_name = endpoint_name;
    options.bounds.subscriber = SubscriberEndpointBounds{
        .queue_depth = 2U,
        .max_borrowed_samples = 2U,
    };
    options.tensor = TensorBoundaryContract{
        .representation = {.memory_kind = MemoryKind::kHost,
                           .dtype = DLDataType{kDLUInt, 8U, 1U},
                           .rank = 1U,
                           .strides = TensorStridePolicy::kContiguousOnly},
        .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 16U, .axes = axes},
        .transfer =
            TensorTransferPolicy{.foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded},
    };
    return options;
  };
  const std::string first_name = unique_service();
  const std::string second_name = unique_service();
  // Publish one device output to two independent subscribers over external catalog topics.
  auto first = graph.export_topic<Tensor>("holomq://" + first_name, make_options(first_name));
  auto second = graph.export_topic<Tensor>("holomq://" + second_name, make_options(second_name));
  graph.add_flow(source->output, first.input());
  graph.add_flow(source->output, second.input());

  // Device-visible outputs require deployment-owned device evidence; ordinary compile(graph)
  // correctly rejects an implicit device-zero assumption with IMPLICIT_DEVICE_ZERO_REJECTED.
  const auto snapshot = GraphCompilerAccess::canonical_snapshot(graph);
  const auto source_vertex =
      std::find_if(snapshot->vertices.begin(), snapshot->vertices.end(), [](const auto& vertex) {
        return vertex.kind == VertexKind::kOperator && vertex.name == "source";
      });
  const auto source_port =
      source_vertex == snapshot->vertices.end()
          ? snapshot->ports.end()
          : std::find_if(snapshot->ports.begin(), snapshot->ports.end(), [&](const auto& port) {
              return port.owner == source_vertex->id && port.direction == PortDirection::kOutput &&
                     port.name == "output";
            });
  ASSERT_NE(source_port, snapshot->ports.end());
  const std::array device_bindings{
      TensorDeviceBindingRecord{.port = PortId{source_port->id.value},
                                .device = DeviceId{device_ordinal}},
  };
  const ExecutionPlan plan =
      compile_with_runtime_evidence(graph,
                                    {},
                                    RuntimeCompileEvidenceView{
                                        .package_capabilities = runtime_package_capability_view(),
                                        .tensor_device_bindings = device_bindings,
                                    });
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  const auto* planned_output = planned_operator_port(*state, PortDirection::kOutput);
  ASSERT_NE(planned_output, nullptr);
  DeviceContractTensorSourceOperator runtime_source;
  OperatorSpec runtime_source_spec;
  runtime_source.setup(runtime_source_spec);
  const std::array replay_ports{RuntimeChannelPortReplayRecord{
      .port = planned_output->id,
      .owner = planned_output->owner,
      .direction = PortDirection::kOutput,
      .declaration = static_cast<void*>(&runtime_source.output),
      .type_ops = runtime_port_type_ops<Tensor>(),
  }};
  auto session = RuntimeChannelSession::create(*state, replay_ports);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto client_runtime = ChannelRuntime::create();
  ASSERT_TRUE(client_runtime) << runtime_error_message(client_runtime.error());
  auto first_subscriber =
      client_runtime->create_subscriber<Tensor>(plan.endpoint_catalog(), first_name);
  auto second_subscriber =
      client_runtime->create_subscriber<Tensor>(plan.endpoint_catalog(), second_name);
  ASSERT_TRUE(first_subscriber) << runtime_error_message(first_subscriber.error());
  ASSERT_TRUE(second_subscriber) << runtime_error_message(second_subscriber.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());

  // Produce bytes asynchronously in a foreign device allocation and retain that ordering stream.
  constexpr std::size_t kByteCount = 8U;
  auto pool =
      std::make_shared<BlockMemoryPool>(MemoryStorageType::kDevice, kByteCount, 1U, device_ordinal);
  ASSERT_TRUE(pool->initialize_pool());
  test::UniqueCudaStream producer_stream;
  ASSERT_EQ(test::create_non_blocking_cuda_stream(producer_stream), cudaSuccess);
  const std::array<std::int64_t, 1U> shape{static_cast<std::int64_t>(kByteCount)};
  auto device_tensor = make_tensor(shape,
                                   DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
                                   std::nullopt,
                                   MemoryStorageType::kDevice,
                                   pool,
                                   ::cuda::stream_ref{producer_stream.get()});
  ASSERT_TRUE(device_tensor) << runtime_error_message(device_tensor.error());
  auto device_data = device_tensor->mutable_data();
  ASSERT_TRUE(device_data) << runtime_error_message(device_data.error());
  const std::array<std::byte, kByteCount> expected{std::byte{0x81},
                                                   std::byte{0x72},
                                                   std::byte{0x63},
                                                   std::byte{0x54},
                                                   std::byte{0x45},
                                                   std::byte{0x36},
                                                   std::byte{0x27},
                                                   std::byte{0x18}};
  ASSERT_EQ(cudaMemcpyAsync(*device_data,
                            expected.data(),
                            expected.size(),
                            cudaMemcpyHostToDevice,
                            producer_stream.get()),
            cudaSuccess);
  ASSERT_TRUE(device_tensor->set_readiness_stream(producer_stream.get()));

  // Runtime must stage one ordered D2H copy before publishing either external descriptor.
  ASSERT_TRUE((*session)->prepare_activation(planned_output->owner, 1U, {}));
  ASSERT_TRUE(runtime_source.output.emit(Sample<Tensor>{*device_tensor, {}}));
  ASSERT_TRUE((*session)->retire_activation(planned_output->owner, 1U, true));
  auto first_received = receive_catalog_tensor(*first_subscriber);
  auto second_received = receive_catalog_tensor(*second_subscriber);
  ASSERT_TRUE(first_received) << runtime_error_message(first_received.error());
  ASSERT_TRUE(second_received) << runtime_error_message(second_received.error());
  ASSERT_EQ(first_received->data.nbytes(), static_cast<std::int64_t>(expected.size()));
  ASSERT_EQ(second_received->data.nbytes(), static_cast<std::int64_t>(expected.size()));
  EXPECT_EQ(std::memcmp(first_received->data.data(), expected.data(), expected.size()), 0);
  EXPECT_EQ(std::memcmp(second_received->data.data(), expected.data(), expected.size()), 0);
  // Pointer identity proves both subscribers retained the one shared staged host backing.
  EXPECT_EQ(first_received->data.data(), second_received->data.data());
}

// A device catalog contract materializes its HoloMQ pools on the compiler-resolved device. One
// producer-stream-ordered D2D copy supplies both services, and both imports retain the same
// backing.
//
// REQ: CHAN-092
// REQ: SYSTEM-007
TEST(ChannelBindingParityTest, CatalogTensorDeviceFanoutStagesOnceIntoHoloMQDevicePool) {
  test::skip_without_cuda_device();

  int device_ordinal = -1;
  ASSERT_EQ(cudaGetDevice(&device_ordinal), cudaSuccess);
  test::UniqueCudaStream producer_stream;
  ASSERT_EQ(test::create_non_blocking_cuda_stream(producer_stream), cudaSuccess);

  Graph graph{"catalog-tensor-device-pool-export"};
  const ClockKey default_clock =
      graph.bind_shared_clock("runtime-clock", std::make_shared<ManualClock>(0));
  graph.set_default_clock(default_clock);
  auto source = graph.op<DeviceContractTensorSourceOperator>("source");
  static constexpr std::array axes{TensorAxisBounds{.max_extent = 16U, .max_abs_stride_bytes = 1U}};
  const auto make_options = [&](const std::string& endpoint_name) {
    TopicOptions options;
    options.endpoint_name = endpoint_name;
    options.bounds.subscriber = SubscriberEndpointBounds{
        .queue_depth = 2U,
        .max_borrowed_samples = 2U,
    };
    options.tensor = TensorBoundaryContract{
        .representation = {.memory_kind = MemoryKind::kCudaDevice,
                           .dtype = DLDataType{kDLUInt, 8U, 1U},
                           .rank = 1U,
                           .strides = TensorStridePolicy::kContiguousOnly},
        .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 16U, .axes = axes},
        .transfer =
            TensorTransferPolicy{.foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded},
    };
    return options;
  };
  const std::string first_name = unique_service();
  const std::string second_name = unique_service();
  auto first = graph.export_topic<Tensor>("holomq://" + first_name, make_options(first_name));
  auto second = graph.export_topic<Tensor>("holomq://" + second_name, make_options(second_name));
  graph.add_flow(source->output, first.input());
  graph.add_flow(source->output, second.input());

  const auto snapshot = GraphCompilerAccess::canonical_snapshot(graph);
  const auto source_vertex =
      std::find_if(snapshot->vertices.begin(), snapshot->vertices.end(), [](const auto& vertex) {
        return vertex.kind == VertexKind::kOperator && vertex.name == "source";
      });
  const auto source_port =
      source_vertex == snapshot->vertices.end()
          ? snapshot->ports.end()
          : std::find_if(snapshot->ports.begin(), snapshot->ports.end(), [&](const auto& port) {
              return port.owner == source_vertex->id && port.direction == PortDirection::kOutput &&
                     port.name == "output";
            });
  ASSERT_NE(source_port, snapshot->ports.end());
  std::vector<TensorDeviceBindingRecord> device_bindings;
  device_bindings.reserve(snapshot->ports.size());
  for (const auto& port : snapshot->ports) {
    ASSERT_EQ(port.payload_identity.kind, PayloadIdentityKind::kTensor);
    device_bindings.push_back(TensorDeviceBindingRecord{.port = PortId{port.id.value},
                                                        .device = DeviceId{device_ordinal}});
  }
  const ExecutionPlan plan =
      compile_with_runtime_evidence(graph,
                                    {},
                                    RuntimeCompileEvidenceView{
                                        .package_capabilities = runtime_package_capability_view(),
                                        .tensor_device_bindings = device_bindings,
                                    });
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->tensor_interface_pools.size(), 2U);
  EXPECT_TRUE(std::all_of(state->tensor_interface_pools.begin(),
                          state->tensor_interface_pools.end(),
                          [&](const auto& pool) {
                            return pool.memory_kind == MemoryKind::kCudaDevice &&
                                   pool.device == DeviceId{device_ordinal};
                          }));
  const auto* planned_output = planned_operator_port(*state, PortDirection::kOutput);
  ASSERT_NE(planned_output, nullptr);
  DeviceContractTensorSourceOperator runtime_source;
  OperatorSpec runtime_source_spec;
  runtime_source.setup(runtime_source_spec);
  const std::array replay_ports{RuntimeChannelPortReplayRecord{
      .port = planned_output->id,
      .owner = planned_output->owner,
      .direction = PortDirection::kOutput,
      .declaration = static_cast<void*>(&runtime_source.output),
      .type_ops = runtime_port_type_ops<Tensor>(),
  }};
  auto session = RuntimeChannelSession::create(*state, replay_ports);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto client_runtime = ChannelRuntime::create();
  ASSERT_TRUE(client_runtime) << runtime_error_message(client_runtime.error());
  auto first_subscriber =
      client_runtime->create_subscriber<Tensor>(plan.endpoint_catalog(), first_name);
  auto second_subscriber =
      client_runtime->create_subscriber<Tensor>(plan.endpoint_catalog(), second_name);
  ASSERT_TRUE(first_subscriber) << runtime_error_message(first_subscriber.error());
  ASSERT_TRUE(second_subscriber) << runtime_error_message(second_subscriber.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());

  constexpr std::size_t kByteCount = 8U;
  auto source_pool =
      std::make_shared<BlockMemoryPool>(MemoryStorageType::kDevice, kByteCount, 1U, device_ordinal);
  ASSERT_TRUE(source_pool->initialize_pool());
  const std::array<std::int64_t, 1U> shape{static_cast<std::int64_t>(kByteCount)};
  auto device_tensor = make_tensor(shape,
                                   DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
                                   std::nullopt,
                                   MemoryStorageType::kDevice,
                                   source_pool,
                                   ::cuda::stream_ref{producer_stream.get()});
  ASSERT_TRUE(device_tensor) << runtime_error_message(device_tensor.error());
  auto device_data = device_tensor->mutable_data();
  ASSERT_TRUE(device_data) << runtime_error_message(device_data.error());
  const std::array<std::byte, kByteCount> expected{std::byte{0x91},
                                                   std::byte{0x82},
                                                   std::byte{0x73},
                                                   std::byte{0x64},
                                                   std::byte{0x55},
                                                   std::byte{0x46},
                                                   std::byte{0x37},
                                                   std::byte{0x28}};
  ASSERT_EQ(cudaMemcpyAsync(*device_data,
                            expected.data(),
                            expected.size(),
                            cudaMemcpyHostToDevice,
                            producer_stream.get()),
            cudaSuccess);
  ASSERT_TRUE(device_tensor->set_readiness_stream(producer_stream.get()));

  ASSERT_TRUE((*session)->prepare_activation(planned_output->owner, 1U, {}));
  ASSERT_TRUE(runtime_source.output.emit(Sample<Tensor>{*device_tensor, {}}));
  ASSERT_TRUE((*session)->retire_activation(planned_output->owner, 1U, true));

  test::UniqueCudaStream consumer_stream;
  ASSERT_EQ(test::create_non_blocking_cuda_stream(consumer_stream), cudaSuccess);
  std::array<std::byte, kByteCount> first_actual{};
  std::array<std::byte, kByteCount> second_actual{};
  {
    // This inner scope destroys both imported Tensors on scope exit, before consumer_stream is
    // destroyed, so their deferred releases retain a valid CUDA stream through teardown.
    auto first_received = receive_catalog_tensor(*first_subscriber);
    auto second_received = receive_catalog_tensor(*second_subscriber);
    ASSERT_TRUE(first_received) << runtime_error_message(first_received.error());
    ASSERT_TRUE(second_received) << runtime_error_message(second_received.error());
    EXPECT_EQ(first_received->data.device().device_type, kDLCUDA);
    EXPECT_EQ(second_received->data.device().device_type, kDLCUDA);
    EXPECT_EQ(first_received->data.data(), second_received->data.data());
    ASSERT_TRUE(first_received->data.synchronize_for_consumer_stream(consumer_stream.get()));
    ASSERT_TRUE(second_received->data.synchronize_for_consumer_stream(consumer_stream.get()));
    ASSERT_EQ(cudaMemcpyAsync(first_actual.data(),
                              first_received->data.data(),
                              first_actual.size(),
                              cudaMemcpyDeviceToHost,
                              consumer_stream.get()),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(second_actual.data(),
                              second_received->data.data(),
                              second_actual.size(),
                              cudaMemcpyDeviceToHost,
                              consumer_stream.get()),
              cudaSuccess);
    ASSERT_TRUE(first_received->data.set_deallocation_stream(consumer_stream.get()));
    ASSERT_TRUE(second_received->data.set_deallocation_stream(consumer_stream.get()));
    ASSERT_EQ(cudaStreamSynchronize(consumer_stream.get()), cudaSuccess);
    // NB: this sync runs BEFORE the received Tensors are destroyed at scope exit below, so it does
    // NOT drain their deferred HoloMQ ref release. set_deallocation_stream arms a cuLaunchHostFunc
    // release_ref that the ConstBuffer destructor enqueues on consumer_stream only at `}`, after
    // this sync. The test is safe solely because it never publishes again after the import. If a
    // re-publication on a finite device pool is ever added here, await slot recovery first (see
    // wait_for_recycled_tensor_loan in channel_tensor_loan.cpp) rather than relying on this sync.
  }
  EXPECT_EQ(first_actual, expected);
  EXPECT_EQ(second_actual, expected);
}

// Public output placement supplies the private device for a pinned catalog staging pool. The
// producer copy remains stream ordered, while the received public Tensor keeps DLPack's fixed
// kDLCUDAHost/device-zero identity.
// REQ: CHAN-095
// REQ: EXTERNAL-007
TEST(ChannelBindingParityTest, CatalogTensorPinnedExportUsesPlacedOutputDevice) {
  test::skip_without_cuda_device();

  int device_ordinal = -1;
  ASSERT_EQ(cudaGetDevice(&device_ordinal), cudaSuccess);
  const std::string endpoint_name = unique_service();
  Graph graph{"catalog-tensor-pinned-export"};
  const ClockKey default_clock =
      graph.bind_shared_clock("runtime-clock", std::make_shared<ManualClock>(0));
  graph.set_default_clock(default_clock);
  auto source = graph.op<DeviceContractTensorSourceOperator>("source");
  static constexpr std::array axes{TensorAxisBounds{.max_extent = 16U, .max_abs_stride_bytes = 1U}};
  TopicOptions topic_options;
  topic_options.endpoint_name = endpoint_name;
  topic_options.bounds.subscriber = SubscriberEndpointBounds{
      .queue_depth = 1U,
      .max_borrowed_samples = 1U,
  };
  topic_options.tensor = TensorBoundaryContract{
      .representation = {.memory_kind = MemoryKind::kPinnedHost,
                         .dtype = DLDataType{kDLUInt, 8U, 1U},
                         .rank = 1U,
                         .strides = TensorStridePolicy::kContiguousOnly},
      .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 16U, .axes = axes},
      .transfer =
          TensorTransferPolicy{.foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded},
  };
  auto exported = graph.export_topic<Tensor>("holomq://" + endpoint_name, topic_options);
  graph.add_flow(source->output, exported.input());

  CompileOptions compile_options;
  compile_options.deployment.bind_tensor_output_device(TensorOutputDevicePlacement{
      .operator_path = "source",
      .output_port = "output",
      .device = DeviceId{device_ordinal},
  });
  const ExecutionPlan plan = compile(graph, std::move(compile_options));
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->tensor_interface_pools.size(), 1U);
  EXPECT_EQ(state->tensor_interface_pools.front().memory_kind, MemoryKind::kPinnedHost);
  EXPECT_EQ(state->tensor_interface_pools.front().device, DeviceId{device_ordinal});

  // Service-stream owner evidence is optional when no Green Context is active. Keep this ordinary
  // pinned export independent of that unused evidence while retaining the planned pool device.
  auto runtime_state = std::make_shared<ExecutionPlanState>(*state);
  ASSERT_TRUE(runtime_state->cuda_green_context_pools.empty());
  for (auto& transfer : runtime_state->tensor_transfer_plans) {
    transfer.service_owner_id.clear();
  }

  const auto* planned_output = planned_operator_port(*runtime_state, PortDirection::kOutput);
  ASSERT_NE(planned_output, nullptr);
  DeviceContractTensorSourceOperator runtime_source;
  OperatorSpec runtime_source_spec;
  runtime_source.setup(runtime_source_spec);
  const std::array replay_ports{RuntimeChannelPortReplayRecord{
      .port = planned_output->id,
      .owner = planned_output->owner,
      .direction = PortDirection::kOutput,
      .declaration = static_cast<void*>(&runtime_source.output),
      .type_ops = runtime_port_type_ops<Tensor>(),
  }};
  auto session = RuntimeChannelSession::create(*runtime_state, replay_ports);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto client_runtime = ChannelRuntime::create();
  ASSERT_TRUE(client_runtime) << runtime_error_message(client_runtime.error());
  auto subscriber =
      client_runtime->create_subscriber<Tensor>(plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());

  constexpr std::size_t kByteCount = 8U;
  auto source_pool =
      std::make_shared<BlockMemoryPool>(MemoryStorageType::kDevice, kByteCount, 1U, device_ordinal);
  ASSERT_TRUE(source_pool->initialize_pool());
  test::UniqueCudaStream producer_stream;
  ASSERT_EQ(test::create_non_blocking_cuda_stream(producer_stream), cudaSuccess);
  const std::array<std::int64_t, 1U> shape{static_cast<std::int64_t>(kByteCount)};
  auto device_tensor = make_tensor(shape,
                                   DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
                                   std::nullopt,
                                   MemoryStorageType::kDevice,
                                   source_pool,
                                   ::cuda::stream_ref{producer_stream.get()});
  ASSERT_TRUE(device_tensor) << runtime_error_message(device_tensor.error());
  auto device_data = device_tensor->mutable_data();
  ASSERT_TRUE(device_data) << runtime_error_message(device_data.error());
  const std::array<std::byte, kByteCount> expected{std::byte{0xA1},
                                                   std::byte{0xB2},
                                                   std::byte{0xC3},
                                                   std::byte{0xD4},
                                                   std::byte{0xE5},
                                                   std::byte{0xF6},
                                                   std::byte{0x17},
                                                   std::byte{0x28}};
  ASSERT_EQ(cudaMemcpyAsync(*device_data,
                            expected.data(),
                            expected.size(),
                            cudaMemcpyHostToDevice,
                            producer_stream.get()),
            cudaSuccess);
  ASSERT_TRUE(device_tensor->set_readiness_stream(producer_stream.get()));

  ASSERT_TRUE((*session)->prepare_activation(planned_output->owner, 1U, {}));
  ASSERT_TRUE(runtime_source.output.emit(Sample<Tensor>{*device_tensor, {}}));
  ASSERT_TRUE((*session)->retire_activation(planned_output->owner, 1U, true));
  {
    auto received = subscriber->receive_sample();
    ASSERT_TRUE(received) << runtime_error_message(received.error());
    EXPECT_EQ(received->data.device().device_type, kDLCUDAHost);
    EXPECT_EQ(received->data.device().device_id, 0);
    EXPECT_TRUE(received->data.is_read_only());
    ASSERT_TRUE(received->data.synchronize_for_host());
    EXPECT_EQ(std::memcmp(received->data.data(), expected.data(), expected.size()), 0);
  }
}

// The require-exportable policy rejects foreign storage before any catalog endpoint submits or
// consumes the foreign Tensor's host synchronization contract.
// REQ: SYSTEM-006
// REQ: SYSTEM-007
// REQ: SYSTEM-009
TEST(ChannelBindingParityTest, CatalogTensorRequireExportableRejectsBeforeFanoutSubmission) {
  Graph graph{"catalog-tensor-require-exportable"};
  const ClockKey default_clock =
      graph.bind_shared_clock("runtime-clock", std::make_shared<ManualClock>(0));
  graph.set_default_clock(default_clock);
  auto source = graph.op<RequireExportableTensorSourceOperator>("source");
  static constexpr std::array axes{TensorAxisBounds{.max_extent = 16U, .max_abs_stride_bytes = 1U}};
  const auto make_options = [&](const std::string& endpoint_name) {
    TopicOptions options;
    options.endpoint_name = endpoint_name;
    options.bounds.subscriber = SubscriberEndpointBounds{
        .queue_depth = 2U,
        .max_borrowed_samples = 2U,
    };
    options.tensor = TensorBoundaryContract{
        .representation = {.memory_kind = MemoryKind::kHost,
                           .dtype = DLDataType{kDLUInt, 8U, 1U},
                           .rank = 1U,
                           .strides = TensorStridePolicy::kContiguousOnly},
        .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 16U, .axes = axes},
        .transfer = TensorTransferPolicy{.foreign_emit_policy =
                                             InterfaceTensorEmitPolicy::kRequireExportable},
    };
    return options;
  };
  const std::string first_name = unique_service();
  const std::string second_name = unique_service();
  auto first = graph.export_topic<Tensor>("holomq://" + first_name, make_options(first_name));
  auto second = graph.export_topic<Tensor>("holomq://" + second_name, make_options(second_name));
  graph.add_flow(source->output, first.input());
  graph.add_flow(source->output, second.input());

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  const auto* planned_output = planned_operator_port(*state, PortDirection::kOutput);
  ASSERT_NE(planned_output, nullptr);
  RequireExportableTensorSourceOperator runtime_source;
  OperatorSpec runtime_source_spec;
  runtime_source.setup(runtime_source_spec);
  const std::array replay_ports{RuntimeChannelPortReplayRecord{
      .port = planned_output->id,
      .owner = planned_output->owner,
      .direction = PortDirection::kOutput,
      .declaration = static_cast<void*>(&runtime_source.output),
      .type_ops = runtime_port_type_ops<Tensor>(),
  }};
  auto session = RuntimeChannelSession::create(*state, replay_ports);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  auto client_runtime = ChannelRuntime::create();
  ASSERT_TRUE(client_runtime) << runtime_error_message(client_runtime.error());
  auto first_subscriber =
      client_runtime->create_subscriber<Tensor>(plan.endpoint_catalog(), first_name);
  auto second_subscriber =
      client_runtime->create_subscriber<Tensor>(plan.endpoint_catalog(), second_name);
  ASSERT_TRUE(first_subscriber) << runtime_error_message(first_subscriber.error());
  ASSERT_TRUE(second_subscriber) << runtime_error_message(second_subscriber.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());

  auto foreign = make_host_copy_tracking_tensor(2U);
  ASSERT_TRUE((*session)->prepare_activation(planned_output->owner, 1U, {}));
  auto rejected = runtime_source.output.emit(Sample<Tensor>{foreign.tensor, {}});
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(foreign.state->host_synchronizations.load(std::memory_order_relaxed), 0U);
  auto first_empty = first_subscriber->receive_sample();
  auto second_empty = second_subscriber->receive_sample();
  ASSERT_FALSE(first_empty);
  ASSERT_FALSE(second_empty);
  EXPECT_EQ(first_empty.error().code, ErrorCode::kNotReady);
  EXPECT_EQ(second_empty.error().code, ErrorCode::kNotReady);
  EXPECT_EQ(foreign.state->host_synchronizations.load(std::memory_order_relaxed), 0U);
  ASSERT_TRUE((*session)->retire_activation(planned_output->owner, 1U, true));
}

}  // namespace
}  // namespace holoscan::detail

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <memory>
#include <mutex>
#include <semaphore>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "compiler/process_local_channel_plan_validation.hpp"
#include "core/operator_spec_compiler_access.hpp"
#include "core/source_id_internal.hpp"
#include "runtime/runtime_fanout_test_control.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "runtime_channel_binding_test_support.hpp"

namespace holoscan::detail {
namespace {

using namespace runtime_channel_binding_test;

// Give the source and both sinks (and any second producer) their own partitions and select delivery
// inside emit().
void use_experimental_partitions(SessionFixture& fixture, const std::uint32_t operators = 3U) {
  fixture.use_same_partition_eventless_transport();
  fixture.plan.execution_partitions.clear();
  for (std::uint32_t index = 0U; index < operators; ++index) {
    const auto vertex = VertexId{index + 1U};
    const auto partition = ExecutionPartitionId{index + 1U};
    fixture.plan.execution_partitions.push_back(
        ExecutionPartitionRecord{.id = partition, .members = {vertex}, .execution_lane = index});
    fixture.plan.vertex_process_placements.push_back(
        VertexProcessPlacementPlanRecord{.vertex = vertex, .process_group = ProcessGroupId{1U}});
    fixture.plan.partition_runners.push_back(
        PartitionRunnerPlanRecord{.ordinal = index,
                                  .partition = partition,
                                  .execution_lane = index,
                                  .process_group = ProcessGroupId{1U}});
  }
  fixture.plan.fanout_experiment = ProcessLocalFanoutExperiment::kEmitRecipients;
  for (auto& service : fixture.plan.process_local_channel_services) {
    service.emit_admitted = service.delivery_kind == ProcessLocalChannelDeliveryKind::kHoloMQ;
  }
}

void use_experimental_fanout(SessionFixture& fixture) {
  fixture.use_one_slot_blocking_fanout();
  use_experimental_partitions(fixture);
}

void use_experimental_drop_oldest_fanout(SessionFixture& fixture) {
  fixture.use_one_slot_drop_oldest_fanout();
  use_experimental_partitions(fixture);
}

// Hold one receive open inside the endpoint and record any readiness probe that reaches the
// endpoint meanwhile. Backend ports and their staging are not thread-safe, so such a probe races
// the receive even when both return plausible results.
class ProbeObservingInlineSubscriber final : public channel_internal::InlinePayloadSubscriber {
 public:
  [[nodiscard]] expected<bool, Error> has_samples() const override {
    if (receive_active.load(std::memory_order_acquire)) {
      probed_during_receive.store(true, std::memory_order_release);
    }
    return remaining.load(std::memory_order_acquire) != 0U;
  }

  [[nodiscard]] bool event_wake_enabled() const noexcept override { return false; }

  [[nodiscard]] void* wait_context_runtime() noexcept override { return nullptr; }

  [[nodiscard]] bool has_publishers() const noexcept override { return true; }

  [[nodiscard]] expected<channel_internal::InlinePayloadReceiveResult, Error> receive(
      const std::span<std::byte> payload) override {
    if (remaining.load(std::memory_order_acquire) == 0U) {
      return make_unexpected(Error{ErrorCode::kNotReady});
    }
    receive_active.store(true, std::memory_order_release);
    receive_entered.release();
    release_receive.acquire();
    receive_active.store(false, std::memory_order_release);
    remaining.fetch_sub(1U, std::memory_order_acq_rel);
    constexpr std::int32_t kValue = 41;
    if (payload.size() != sizeof(kValue)) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    std::memcpy(payload.data(), &kValue, sizeof(kValue));
    return channel_internal::InlinePayloadReceiveResult{
        .metadata =
            SampleMetadata{
                .sequence = 1U,
                .byte_size = sizeof(kValue),
                .payload_kind = PayloadKind::kInlineTyped,
            },
        .payload_size = sizeof(kValue),
    };
  }

  [[nodiscard]] expected<std::uint64_t, Error> attach_wait_set(void*) override {
    return make_unexpected(Error{ErrorCode::kNotSupported});
  }

  [[nodiscard]] expected<void, Error> detach_wait_set() noexcept override { return {}; }

  std::binary_semaphore receive_entered{0};
  std::binary_semaphore release_receive{0};
  std::atomic<bool> receive_active{};
  mutable std::atomic<bool> probed_during_receive{};
  std::atomic<std::uint32_t> remaining{1U};
};

// Both recipients retain their only typed slot while a second publication enters transport.
// Releasing one recipient must admit only its next sample; the other recipient's retry obligation
// must survive until its own release. No subsequent publication or explicit cut supplies progress.
// Concurrent admission skips both known-blocked heads, so each sample-retirement release must
// retry exactly its own subscriber and clear the blocked-head signal only when both are admitted.
// REQ: RUNTIME-033
// REQ: CHAN-136
class RuntimeFanoutWakeOrder : public ::testing::TestWithParam<bool> {};

TEST_P(RuntimeFanoutWakeOrder, RetiresBlockedHeadsWithoutPumpOrAnotherPublication) {
  const bool concurrent_admission = GetParam();
  SessionFixture fixture;
  use_experimental_fanout(fixture);
  auto control = std::make_shared<RuntimeFanoutTestControl>();
  control->concurrent_admission = concurrent_admission;
  fixture.plan.executable_factories.test_fanout_control = std::move(control);
  ASSERT_TRUE(valid_process_local_channel_plan(fixture.plan));
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  std::array<std::unique_ptr<PortBinding>, 2U> bindings;
  for (std::uint32_t index = 0U; index < 2U; ++index) {
    auto binding = (*session)->create_readiness_binding(
        VertexId{index + 2U},
        RuntimePortBindingRequest{.plan_generation = fixture.plan.plan_generation,
                                  .run_generation = 1U,
                                  .kind = RuntimeEventKind::kInputArrival,
                                  .source = make_source_id(index + 1U),
                                  .source_clock = ClockId{1U},
                                  .reference_clock = ClockId{1U},
                                  .capacity = 2U});
    ASSERT_TRUE(binding) << runtime_error_message(binding.error());
    bindings[index] = std::move(*binding);
  }
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{41}));
  std::array<RuntimeEvent, 2U> first_events;
  std::array<TemporalSampleRecord, 2U> first_records;
  for (std::size_t index = 0U; index < bindings.size(); ++index) {
    auto event = wait_for_event(**session, *bindings[index]);
    ASSERT_TRUE(event) << runtime_error_message(event.error());
    first_events[index] = *event;
    auto sample = (*session)->temporal_sample(event->port, event->source_sequence);
    ASSERT_TRUE(sample);
    first_records[index] = *sample;
  }
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{42}));
  auto progress = RuntimeChannelSessionTestAccess::process_local_progress_state(**session);
  ASSERT_TRUE(progress);
  EXPECT_EQ(progress->published_deliveries, 4U);
  EXPECT_EQ(progress->received_deliveries, 2U);
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  if (concurrent_admission) {
    // Key assertion: both second-publication heads are known blocked.
    EXPECT_TRUE(RuntimeChannelSessionTestAccess::blocked_transport_retry_pending(**session));
  }

  for (std::size_t index = 0U; index < bindings.size(); ++index) {
    ASSERT_TRUE(bindings[index]->retire(first_events[index].source_sequence));
    const std::array released{first_records[index]};
    ASSERT_TRUE((*session)->retire_samples(first_events[index].port, released));
    progress = RuntimeChannelSessionTestAccess::process_local_progress_state(**session);
    ASSERT_TRUE(progress);
    EXPECT_EQ(progress->received_deliveries, 3U + index);
    if (concurrent_admission) {
      // Key assertion: the other recipient's head stays blocked until its own release.
      EXPECT_EQ(RuntimeChannelSessionTestAccess::blocked_transport_retry_pending(**session),
                index == 0U);
    }
    auto next = wait_for_event(**session, *bindings[index]);
    ASSERT_TRUE(next) << runtime_error_message(next.error());
    EXPECT_GT(next->source_sequence, first_events[index].source_sequence);
    auto sample = (*session)->temporal_sample(next->port, next->source_sequence);
    ASSERT_TRUE(sample);
    EXPECT_EQ(sample->source_sequence, 2U);
    ASSERT_TRUE(bindings[index]->retire(next->source_sequence));
    const std::array consumed{*sample};
    ASSERT_TRUE((*session)->retire_samples(next->port, consumed));
  }
  ASSERT_TRUE((*session)->status());
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

INSTANTIATE_TEST_SUITE_P(RecipientWakeOrder, RuntimeFanoutWakeOrder, ::testing::Bool());

// A second producer (vertex 4) that feeds the second sink through its own blocking service.
struct SecondProducer {
  SourceOperator source;
  OperatorSpec spec;
};

// Feed the second sink from `producer` instead of from the fixture's source. Returns the replay
// records for all four ports.
std::vector<RuntimeChannelPortReplayRecord> add_second_producer(SessionFixture& fixture,
                                                                SecondProducer& producer) {
  producer.source.setup(producer.spec);
  const auto output = OperatorSpecCompilerAccess::port(producer.spec, 0U);
  auto& plan = fixture.plan;
  const PayloadIdentityRecord payload = plan.ports.front().payload_identity;
  const std::string service = "process-local-channel." + std::to_string(plan.plan_generation) +
                              ".source.4.policy." +
                              std::to_string(static_cast<std::uint32_t>(QueuePolicy::kBlocking));
  plan.ports.push_back(PortPlanRecord{
      .id = PortId{3U},
      .owner = VertexId{3U},
      .direction = PortDirection::kInput,
      .local_name = "input",
      .payload_identity = payload,
      .queue_depth = plan.payload_edges.front().queue_depth,
      .effective_profile = Profile::kStandard,
  });
  plan.ports.push_back(PortPlanRecord{
      .id = PortId{4U},
      .owner = VertexId{4U},
      .direction = PortDirection::kOutput,
      .local_name = "output",
      .payload_identity = payload,
      .max_emits_per_compute = 8U,
      .effective_profile = Profile::kStandard,
  });
  auto service_record = plan.process_local_channel_services.front();
  service_record.source = PortId{4U};
  service_record.service_id = service;
  plan.process_local_channel_services.push_back(std::move(service_record));
  plan.process_local_channel_publishers.push_back(ProcessLocalChannelPublisherPlanRecord{
      .source = PortId{4U},
      .service_id = service,
      .max_loaned_samples = 8U,
  });
  auto subscriber = plan.process_local_channel_subscribers.front();
  subscriber.edge = EdgeId{2U};
  subscriber.source = PortId{4U};
  subscriber.target = PortId{3U};
  subscriber.service_id = service;
  plan.process_local_channel_subscribers.push_back(std::move(subscriber));
  auto edge = plan.payload_edges.front();
  edge.edge = EdgeId{2U};
  edge.source = PortId{4U};
  edge.target = PortId{3U};
  plan.payload_edges.push_back(std::move(edge));
  auto locality = plan.channel_locality.front();
  locality.edge = EdgeId{2U};
  plan.channel_locality.push_back(std::move(locality));
  auto loan = plan.generic_output_loans.front();
  loan.output = PortId{4U};
  plan.generic_output_loans.push_back(std::move(loan));
  std::vector<RuntimeChannelPortReplayRecord> replay(fixture.replay.begin(), fixture.replay.end());
  replay.push_back(RuntimeChannelPortReplayRecord{
      .port = PortId{4U},
      .owner = VertexId{4U},
      .direction = PortDirection::kOutput,
      .declaration = const_cast<void*>(output.declaration),
      .type_ops = output.runtime_type_ops,
  });
  return replay;
}

// The pump and delivery inside emit() serve different producers of one session. The second
// producer's consumer can be restarted, so the pump serves that producer, while the first
// producer's blocking service is admitted inside emit(). The pump never scans the emit-admitted
// subscriber, so a retirement that frees that subscriber's capacity must retry its blocked head
// itself.
// REQ: RUNTIME-029
// REQ: CHAN-136
// REQ: RUNTIME-083
TEST(RuntimeFanoutExperiment, EmitAdmissionCoexistsWithAPumpServedProducer) {
  SessionFixture fixture;
  fixture.use_one_slot_blocking_transport();
  SecondProducer second;
  const auto replay = add_second_producer(fixture, second);
  use_experimental_partitions(fixture, 4U);
  fixture.plan.lifecycle_restart_targets.push_back(LifecycleRestartTargetPlanRecord{
      .target = OperatorRef{.id = 3U, .plan_generation = fixture.plan.plan_generation},
      .self_allowed = true,
      .disposition = LifecycleRestartTargetDisposition::kAdmitted,
  });
  for (auto& service : fixture.plan.process_local_channel_services) {
    service.emit_admitted = service.source == PortId{1U};
    service.event_wake = !service.emit_admitted;
  }
  for (auto& subscriber : fixture.plan.process_local_channel_subscribers) {
    subscriber.event_wake = subscriber.source != PortId{1U};
  }
  ASSERT_TRUE(valid_process_local_channel_plan(fixture.plan));
  auto session = RuntimeChannelSession::create(fixture.plan, replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  std::array<std::unique_ptr<PortBinding>, 2U> bindings;
  for (std::uint32_t index = 0U; index < 2U; ++index) {
    auto binding = (*session)->create_readiness_binding(
        VertexId{index + 2U},
        RuntimePortBindingRequest{.plan_generation = fixture.plan.plan_generation,
                                  .run_generation = 1U,
                                  .kind = RuntimeEventKind::kInputArrival,
                                  .source = make_source_id(index + 1U),
                                  .source_clock = ClockId{1U},
                                  .reference_clock = ClockId{1U},
                                  .capacity = 4U});
    ASSERT_TRUE(binding) << runtime_error_message(binding.error());
    bindings[index] = std::move(*binding);
  }
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  auto pump = RuntimeChannelSessionTestAccess::pump_index_state(**session);
  ASSERT_TRUE(pump) << runtime_error_message(pump.error());
  EXPECT_TRUE(pump->background_thread_running);

  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{41}));
  auto blocking_first = wait_for_event(**session, *bindings[0]);
  ASSERT_TRUE(blocking_first) << runtime_error_message(blocking_first.error());
  auto blocking_record =
      (*session)->temporal_sample(blocking_first->port, blocking_first->source_sequence);
  ASSERT_TRUE(blocking_record);
  // The blocking consumer still holds its only typed slot, so the next head stays blocked.
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{42}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  // The pump serves the second producer's consumer.
  ASSERT_TRUE((*session)->prepare_activation(VertexId{4U}, 1U, {}));
  ASSERT_TRUE(second.source.output.emit(std::int32_t{7}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{4U}, 1U, true));
  auto pumped = wait_for_event(**session, *bindings[1]);
  ASSERT_TRUE(pumped) << runtime_error_message(pumped.error());

  ASSERT_TRUE(bindings[0]->retire(blocking_first->source_sequence));
  const std::array released{*blocking_record};
  ASSERT_TRUE((*session)->retire_samples(blocking_first->port, released));
  // Key assertion: the retirement retried the emit-admitted head although a pump runs.
  auto blocking_second = wait_for_event(**session, *bindings[0]);
  ASSERT_TRUE(blocking_second) << runtime_error_message(blocking_second.error());
  auto second_record =
      (*session)->temporal_sample(blocking_second->port, blocking_second->source_sequence);
  ASSERT_TRUE(second_record);
  EXPECT_EQ(second_record->source_sequence, 2U);
  ASSERT_TRUE(bindings[0]->retire(blocking_second->source_sequence));
  ASSERT_TRUE(bindings[1]->retire(pumped->source_sequence));
  ASSERT_TRUE((*session)->status());
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// A producer may mix the two only across different consumers. Here the pump serves the
// restartable drop-oldest consumer and the blocking sibling feeds another consumer, so no
// consumer can see an emit-admitted arrival overtake an earlier pump-served one. The plan
// validator replays that admission and rejects a plan that also withholds the blocking sibling.
// REQ: RUNTIME-016
// REQ: RUNTIME-083
TEST(RuntimeFanoutExperiment, AdmitsAnEmitServiceBesideAPumpServedSiblingForAnotherConsumer) {
  SessionFixture fixture;
  fixture.use_one_slot_blocking_transport();
  fixture.add_mixed_policy_fanout();
  fixture.plan.payload_edges.front().fanout = 2U;
  use_experimental_partitions(fixture);
  ASSERT_TRUE(valid_process_local_channel_plan(fixture.plan));
  fixture.plan.lifecycle_restart_targets.push_back(LifecycleRestartTargetPlanRecord{
      .target = OperatorRef{.id = 3U, .plan_generation = fixture.plan.plan_generation},
      .self_allowed = true,
      .disposition = LifecycleRestartTargetDisposition::kAdmitted,
  });
  std::string pumped_service;
  for (auto& service : fixture.plan.process_local_channel_services) {
    if (service.queue_policy == QueuePolicy::kDropOldest) {
      service.emit_admitted = false;
      service.event_wake = true;
      pumped_service = service.service_id;
    }
  }
  for (auto& subscriber : fixture.plan.process_local_channel_subscribers) {
    subscriber.event_wake = subscriber.service_id == pumped_service;
  }
  // Key assertion: the blocking sibling feeds another consumer, so it stays inside emit().
  EXPECT_TRUE(valid_process_local_channel_plan(fixture.plan));
  for (auto& service : fixture.plan.process_local_channel_services) {
    service.emit_admitted = false;
    service.event_wake = true;
  }
  for (auto& subscriber : fixture.plan.process_local_channel_subscribers) {
    subscriber.event_wake = true;
  }
  EXPECT_FALSE(valid_process_local_channel_plan(fixture.plan));
}

// Delivery inside emit() also serves drop-oldest services: every recipient's typed store holds the
// sample before emit() returns, with no pump. A recipient that still holds its only typed slot
// stages the next sample instead. Staging publishes no readiness change, so that admission posts
// no owner hint, as direct delivery does; retiring the slot promotes the staged sample.
// REQ: CHAN-128
// REQ: RUNTIME-029
// REQ: RUNTIME-083
TEST(RuntimeFanoutExperiment, AdmitsDropOldestRecipientsInsideEmit) {
  SessionFixture fixture;
  use_experimental_drop_oldest_fanout(fixture);
  const auto counters = std::make_shared<RuntimePhaseCounters>();
  fixture.plan.executable_factories.phase_counters = counters;
  ASSERT_TRUE(valid_process_local_channel_plan(fixture.plan));
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  std::array<std::unique_ptr<PortBinding>, 2U> bindings;
  for (std::uint32_t index = 0U; index < 2U; ++index) {
    auto binding = (*session)->create_readiness_binding(
        VertexId{index + 2U},
        RuntimePortBindingRequest{.plan_generation = fixture.plan.plan_generation,
                                  .run_generation = 1U,
                                  .kind = RuntimeEventKind::kInputArrival,
                                  .source = make_source_id(index + 1U),
                                  .source_clock = ClockId{1U},
                                  .reference_clock = ClockId{1U},
                                  .capacity = 2U});
    ASSERT_TRUE(binding) << runtime_error_message(binding.error());
    bindings[index] = std::move(*binding);
  }
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  auto pump = RuntimeChannelSessionTestAccess::pump_index_state(**session);
  ASSERT_TRUE(pump) << runtime_error_message(pump.error());
  EXPECT_FALSE(pump->background_thread_required);
  const auto phase = [&](const RuntimePhaseCounter counter) {
    return counters->snapshot()[RuntimePhaseCounters::index(counter)];
  };

  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{41}));
  // Key assertion: both drop-oldest recipients received the sample inside emit().
  auto progress = RuntimeChannelSessionTestAccess::process_local_progress_state(**session);
  ASSERT_TRUE(progress);
  EXPECT_EQ(progress->published_deliveries, 2U);
  EXPECT_EQ(progress->received_deliveries, 2U);
  EXPECT_EQ(phase(RuntimePhaseCounter::kChannelEmitStagedHintsElided), 0U);
  std::array<RuntimeEvent, 2U> first_events;
  std::array<TemporalSampleRecord, 2U> first_records;
  for (std::size_t index = 0U; index < bindings.size(); ++index) {
    auto event = wait_for_event(**session, *bindings[index]);
    ASSERT_TRUE(event) << runtime_error_message(event.error());
    first_events[index] = *event;
    auto sample = (*session)->temporal_sample(event->port, event->source_sequence);
    ASSERT_TRUE(sample);
    EXPECT_EQ(sample->source_sequence, 1U);
    first_records[index] = *sample;
  }

  // Each witnessed first sample pins its recipient's only typed slot, so the second one stages.
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{42}));
  progress = RuntimeChannelSessionTestAccess::process_local_progress_state(**session);
  ASSERT_TRUE(progress);
  EXPECT_EQ(progress->received_deliveries, 4U);
  EXPECT_EQ(progress->replaced_deliveries, 0U);
  // Key assertion: the staged admissions changed no readiness and posted no hint.
  EXPECT_EQ(phase(RuntimePhaseCounter::kChannelEmitStagedHintsElided), 2U);
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));

  for (std::size_t index = 0U; index < bindings.size(); ++index) {
    ASSERT_TRUE(bindings[index]->retire(first_events[index].source_sequence));
    const std::array released{first_records[index]};
    ASSERT_TRUE((*session)->retire_samples(first_events[index].port, released));
    auto next = wait_for_event(**session, *bindings[index]);
    ASSERT_TRUE(next) << runtime_error_message(next.error());
    auto sample = (*session)->temporal_sample(next->port, next->source_sequence);
    ASSERT_TRUE(sample);
    EXPECT_EQ(sample->source_sequence, 2U);
    ASSERT_TRUE(bindings[index]->retire(next->source_sequence));
    const std::array consumed{*sample};
    ASSERT_TRUE((*session)->retire_samples(next->port, consumed));
  }
  ASSERT_TRUE((*session)->status());
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// Under concurrent admission, a receive's post-lock observations must report the sample that
// receive admitted, even when a later receive on the same subscriber has ingested EOS before they
// run. REQ: CHAN-136
TEST(RuntimeFanoutExperiment, AdmissionObservationsUseTheirOwnReceiveReceipt) {
  using namespace std::chrono_literals;
  SessionFixture fixture;
  use_experimental_fanout(fixture);
  struct Record {
    PortId target;
    std::uint64_t sequence{};
    RuntimeFanoutObservationPoint point{};
  };
  struct Observations {
    PortId paused_target;
    std::atomic<bool> pause_pending{true};
    std::binary_semaphore paused{0};
    std::binary_semaphore resume{0};
    std::mutex mutex;
    std::array<Record, 16U> records{};
    std::size_t count{};
  } observations;
  observations.paused_target = fixture.plan.process_local_channel_subscribers[0U].target;
  auto control = std::make_shared<RuntimeFanoutTestControl>();
  control->concurrent_admission = true;
  control->context = &observations;
  control->observe = [](void* opaque,
                        const PortId target,
                        const std::uint64_t sequence,
                        const RuntimeFanoutObservationPoint point) noexcept {
    auto& observed = *static_cast<Observations*>(opaque);
    {
      const std::scoped_lock lock{observed.mutex};
      if (observed.count < observed.records.size()) {
        observed.records[observed.count] = Record{target, sequence, point};
      }
      ++observed.count;
    }
    if (point == RuntimeFanoutObservationPoint::kAdmissionCompleted &&
        target == observed.paused_target && observed.pause_pending.exchange(false)) {
      observed.paused.release();
      static_cast<void>(observed.resume.try_acquire_for(2s));
    }
  };
  fixture.plan.executable_factories.test_fanout_control = std::move(control);
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));

  // Key setup: pause the first admission's observations, then complete an EOS receive on the same
  // subscriber before they resume.
  auto publication =
      std::async(std::launch::async, [&] { return fixture.source.output.emit(std::int32_t{41}); });
  const bool paused = observations.paused.try_acquire_for(2s);
  const auto end_of_stream = RuntimeChannelSessionTestAccess::receive_end_of_stream_concurrently(
      **session, observations.paused_target, 2U);
  observations.resume.release();
  const auto emitted = publication.get();
  ASSERT_TRUE(paused);
  ASSERT_TRUE(end_of_stream) << runtime_error_message(end_of_stream.error());
  ASSERT_TRUE(emitted) << runtime_error_message(emitted.error());

  // Key assertion: every observation of the paused receive still names publication 1.
  std::vector<RuntimeFanoutObservationPoint> points;
  {
    const std::scoped_lock lock{observations.mutex};
    ASSERT_LE(observations.count, observations.records.size());
    for (std::size_t index = 0U; index < observations.count; ++index) {
      const Record& record = observations.records[index];
      if (record.target == observations.paused_target) {
        EXPECT_EQ(record.sequence, 1U);
        points.push_back(record.point);
      }
    }
  }
  EXPECT_EQ(points,
            (std::vector{RuntimeFanoutObservationPoint::kAdmissionCompleted,
                         RuntimeFanoutObservationPoint::kReadyHintStarted,
                         RuntimeFanoutObservationPoint::kReadyHintReturned}));
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// Under concurrent admission, a released list of up to 16 entries retries from private stack
// scratch without the channel scan lock; a 17-entry list retries the same entries under the scan
// lock, still scoped to the released recipients. The entries alternate the fixture's two blocked
// subscribers, so this checks the list-length boundary, not 17 distinct recipients.
// REQ: CHAN-136
TEST(RuntimeFanoutExperiment, ConcurrentAdmissionRetryListBoundaryKeepsBothPaths) {
  SessionFixture fixture;
  use_experimental_fanout(fixture);
  const auto counters = std::make_shared<RuntimePhaseCounters>();
  fixture.plan.executable_factories.phase_counters = counters;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::enable_concurrent_admission(**session));
  const PortId first = fixture.plan.process_local_channel_subscribers[0U].target;
  const PortId second = fixture.plan.process_local_channel_subscribers[1U].target;
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_transport_head_blocked(**session, first, true));
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_transport_head_blocked(**session, second, true));
  const auto count = [&](const RuntimePhaseCounter counter) {
    return counters->snapshot()[RuntimePhaseCounters::index(counter)];
  };
  for (const std::size_t length : {std::size_t{16U}, std::size_t{17U}}) {
    SCOPED_TRACE(::testing::Message() << "released=" << length);
    std::vector<PortId> released(length);
    for (std::size_t index = 0U; index < length; ++index) {
      released[index] = index % 2U == 0U ? first : second;
    }
    const auto locks_before = count(RuntimePhaseCounter::kChannelScanLockAcquisitionsCut);
    const auto retries_before = count(RuntimePhaseCounter::kChannelActivationCutScopedRetries);
    const auto probes_before = count(RuntimePhaseCounter::kChannelActivationCutSubscriberProbes);
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::retry_released_inputs(**session, released));
    // Key assertion: one scoped retry probes every released entry on both sides.
    EXPECT_EQ(count(RuntimePhaseCounter::kChannelActivationCutScopedRetries), retries_before + 1U);
    EXPECT_EQ(count(RuntimePhaseCounter::kChannelActivationCutSubscriberProbes) - probes_before,
              length);
    // Key assertion: only the list beyond the private scratch capacity takes the scan lock.
    EXPECT_EQ(count(RuntimePhaseCounter::kChannelScanLockAcquisitionsCut) - locks_before,
              length > 16U ? 1U : 0U);
  }
  EXPECT_TRUE(RuntimeChannelSessionTestAccess::blocked_transport_retry_pending(**session));
}

// Record whether a receive follows its own thread's readiness probe with no other probe between
// them. The endpoint never has data, so it can be probed and received indefinitely.
class ProbeReceiveOrderInlineSubscriber final : public channel_internal::InlinePayloadSubscriber {
 public:
  [[nodiscard]] expected<bool, Error> has_samples() const override {
    last_prober.store(thread_token(), std::memory_order_relaxed);
    return true;
  }

  [[nodiscard]] bool event_wake_enabled() const noexcept override { return false; }

  [[nodiscard]] void* wait_context_runtime() noexcept override { return nullptr; }

  [[nodiscard]] bool has_publishers() const noexcept override { return true; }

  [[nodiscard]] expected<channel_internal::InlinePayloadReceiveResult, Error> receive(
      std::span<std::byte>) override {
    return make_unexpected(Error{ErrorCode::kNotReady});
  }

  [[nodiscard]] expected<channel_internal::InlinePayloadReceiveGuard, Error>
  receive_borrowed_if_data_accepted(bool) override {
    receives.fetch_add(1U, std::memory_order_relaxed);
    if (last_prober.load(std::memory_order_relaxed) != thread_token()) {
      interleaved.fetch_add(1U, std::memory_order_relaxed);
    }
    return make_unexpected(Error{ErrorCode::kNotReady});
  }

  [[nodiscard]] expected<std::uint64_t, Error> attach_wait_set(void*) override {
    return make_unexpected(Error{ErrorCode::kNotSupported});
  }

  [[nodiscard]] expected<void, Error> detach_wait_set() noexcept override { return {}; }

  mutable std::atomic<std::uint64_t> last_prober{};
  std::atomic<std::uint64_t> receives{};
  std::atomic<std::uint64_t> interleaved{};

 private:
  [[nodiscard]] static std::uint64_t thread_token() noexcept {
    static std::atomic<std::uint64_t> next{1U};
    thread_local const std::uint64_t token = next.fetch_add(1U, std::memory_order_relaxed);
    return token;
  }
};

// Under concurrent admission, several lanes drain one subscriber without the scan lock. Each must
// probe and receive in one transport transaction: a head another lane consumes between them would
// otherwise be received again, and an accepted EOS is re-reported as a new delivery, so the session
// counts more deliveries than were published and never completes.
// REQ: CHAN-136
TEST(RuntimeFanoutExperiment, ConcurrentAdmissionProbesAndReceivesInOneTransaction) {
  SessionFixture fixture;
  use_experimental_fanout(fixture);
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::enable_concurrent_admission(**session));
  const PortId input = fixture.plan.process_local_channel_subscribers[0U].target;
  auto replacement = std::make_unique<ProbeReceiveOrderInlineSubscriber>();
  auto* const endpoint = replacement.get();
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_input_subscriber(
      **session, input, std::move(replacement)));

  // Key setup: two lanes repeatedly admit to the same subscriber at once.
  constexpr int kAdmissions = 20'000;
  std::atomic<bool> start{false};
  std::atomic<int> failures{0};
  const auto admit = [&] {
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    const std::array<PortId, 1U> recipients{input};
    for (int admission = 0; admission < kAdmissions; ++admission) {
      if (!RuntimeChannelSessionTestAccess::admit_recipients(**session, recipients)) {
        failures.fetch_add(1, std::memory_order_relaxed);
      }
    }
  };
  std::thread first(admit);
  std::thread second(admit);
  start.store(true, std::memory_order_release);
  first.join();
  second.join();

  // Key assertion: every receive directly follows its own lane's probe.
  EXPECT_EQ(failures.load(std::memory_order_relaxed), 0);
  EXPECT_EQ(endpoint->receives.load(std::memory_order_relaxed), 2U * kAdmissions);
  EXPECT_EQ(endpoint->interleaved.load(std::memory_order_relaxed), 0U);
}

// Under concurrent admission, emit-time admission reaches a subscriber without the channel scan
// lock while another lane may be receiving from it. The admission's readiness probe must wait for
// that receive's transport mutex, because the backend port and its staging are not thread-safe.
// REQ: CHAN-136
TEST(RuntimeFanoutExperiment, ConcurrentAdmissionProbesTransportUnderItsMutex) {
  using namespace std::chrono_literals;
  SessionFixture fixture;
  use_experimental_fanout(fixture);
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::enable_concurrent_admission(**session));
  const PortId input = fixture.plan.process_local_channel_subscribers[0U].target;
  auto replacement = std::make_unique<ProbeObservingInlineSubscriber>();
  auto* const endpoint = replacement.get();
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_input_subscriber(
      **session, input, std::move(replacement)));

  // Key setup: one receive holds the subscriber's transport mutex inside the endpoint while an
  // emit-time admission to the same subscriber starts.
  auto receive = std::async(std::launch::async, [&] {
    return RuntimeChannelSessionTestAccess::receive_one(**session, input);
  });
  if (!endpoint->receive_entered.try_acquire_for(2s)) {
    // Release and finish the receive before failing, or its future would block the test forever.
    endpoint->release_receive.release();
    static_cast<void>(receive.get());
    FAIL() << "the receive never reached the endpoint";
  }
  std::atomic<bool> admission_started{};
  auto admission = std::async(std::launch::async, [&] {
    admission_started.store(true, std::memory_order_release);
    const std::array<PortId, 1U> recipients{input};
    return RuntimeChannelSessionTestAccess::admit_recipients(**session, recipients);
  });
  while (!admission_started.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  // Give an unserialized probe time to reach the endpoint before the receive completes.
  std::this_thread::sleep_for(20ms);
  endpoint->release_receive.release();
  const auto received = receive.get();
  const auto admitted = admission.get();

  // Key assertion: the admission probed the endpoint only after the receive released it.
  EXPECT_FALSE(endpoint->probed_during_receive.load(std::memory_order_acquire));
  ASSERT_TRUE(received) << runtime_error_message(received.error());
  ASSERT_TRUE(admitted) << runtime_error_message(admitted.error());
  EXPECT_EQ(endpoint->remaining.load(std::memory_order_acquire), 0U);
}

// Diagnostic isolation control for the private emit experiment. A descheduled cache owner can
// still delay an accepted publication and its siblings. This intentionally records a limitation;
// an ownership redesign should replace this control with a successful-isolation assertion.
// REQ: RUNTIME-017
TEST(RuntimeFanoutExperiment, DocumentsProducerDependenceOnDestinationCacheOwner) {
  using namespace std::chrono_literals;
  SessionFixture fixture;
  use_experimental_fanout(fixture);
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  struct HeldLock {
    std::binary_semaphore entered{0};
    std::binary_semaphore release{0};
  } held;
  auto holder = std::async(std::launch::async, [&] {
    return RuntimeChannelSessionTestAccess::with_input_lock(
        **session, PortId{2U}, &held, [](void* opaque) noexcept {
          auto& state = *static_cast<HeldLock*>(opaque);
          state.entered.release();
          static_cast<void>(state.release.try_acquire_for(2s));
        });
  });
  const bool entered = held.entered.try_acquire_for(1s);
  auto publication =
      std::async(std::launch::async, [&] { return fixture.source.output.emit(std::int32_t{41}); });
  const auto deadline = std::chrono::steady_clock::now() + 1s;
  bool accepted = false;
  while (!accepted && std::chrono::steady_clock::now() < deadline) {
    const auto progress = RuntimeChannelSessionTestAccess::process_local_progress_state(**session);
    accepted = progress && progress->published_deliveries == 2U;
    std::this_thread::yield();
  }
  const bool waiting_on_admission = publication.wait_for(0ns) == std::future_status::timeout;
  held.release.release();
  const auto lock_result = holder.get();
  const auto emitted = publication.get();
  ASSERT_TRUE(entered);
  ASSERT_TRUE(lock_result);
  ASSERT_TRUE(accepted);
  EXPECT_TRUE(waiting_on_admission);
  ASSERT_TRUE(emitted) << runtime_error_message(emitted.error());
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// Fault-injected backend outcomes exercise authority/sequence accounting through the new admission
// hook. These publishers do not deliver data; real transport/backlog delivery is tested above.
// REQ: CHAN-108
TEST(RuntimeFanoutExperiment, EmptyExactRecipientProofFailsInsteadOfScanningInventory) {
  SessionFixture fixture;
  use_experimental_fanout(fixture);
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session);
  std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
  publishers.push_back(std::make_unique<ScriptedInlinePublisher>(
      std::vector<channel_internal::InlinePublicationResult>{
          {.phase = channel_internal::InlinePublicationPhase::kAccepted,
           .recipient_count = 3U,
           .error = Error{ErrorCode::kSuccess}}}));
  // Deliberately corrupt the two-recipient shape: the replacement has no exact recipient proof.
  const std::array expected_recipients{3U};
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_output_publishers(
      **session, PortId{1U}, std::move(publishers), expected_recipients));
  ASSERT_TRUE((*session)->arm());
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{41}));
  const auto status = (*session)->status();
  ASSERT_FALSE(status);
  EXPECT_EQ(status.error().code, ErrorCode::kProtocolError);
  (*session)->cancel();
  EXPECT_FALSE((*session)->join());
}

// REQ: CHAN-136
// REQ: CHAN-108
TEST(RuntimeFanoutExperiment, PreservesRejectedAcceptedPartialAndIndeterminateOutcomes) {
  using Phase = channel_internal::InlinePublicationPhase;
  struct Case {
    Phase phase;
    std::uint32_t recipients;
    ErrorCode backend_error;
    ErrorCode expected_error;
    std::uint64_t next_sequence;
  };
  for (const Case& test_case :
       {Case{Phase::kPreSubmitRejected,
             0U,
             ErrorCode::kBackpressured,
             ErrorCode::kBackpressured,
             1U},
        Case{Phase::kAccepted, 2U, ErrorCode::kSuccess, ErrorCode::kSuccess, 2U},
        Case{Phase::kAccepted, 1U, ErrorCode::kSuccess, ErrorCode::kPublicationPartial, 2U},
        Case{Phase::kPostSubmitUncertain,
             1U,
             ErrorCode::kPublicationPartial,
             ErrorCode::kPublicationPartial,
             2U},
        Case{Phase::kPostSubmitUncertain,
             0U,
             ErrorCode::kPublicationIndeterminate,
             ErrorCode::kPublicationIndeterminate,
             2U}}) {
    SCOPED_TRACE(static_cast<unsigned>(test_case.expected_error));
    SessionFixture fixture;
    use_experimental_fanout(fixture);
    ASSERT_TRUE(valid_process_local_channel_plan(fixture.plan));
    auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
    ASSERT_TRUE(session) << runtime_error_message(session.error());
    std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
    publishers.push_back(std::make_unique<ScriptedInlinePublisher>(
        std::vector<channel_internal::InlinePublicationResult>{
            {.phase = test_case.phase,
             .recipient_count = test_case.recipients,
             .error = Error{test_case.backend_error}}}));
    const std::array expected_recipients{2U};
    ASSERT_TRUE(RuntimeChannelSessionTestAccess::replace_output_publishers(
        **session, PortId{1U}, std::move(publishers), expected_recipients));
    ASSERT_TRUE((*session)->refresh_connections());
    ASSERT_TRUE((*session)->arm());
    ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
    auto published = fixture.source.output.emit(std::int32_t{41});
    EXPECT_EQ(published ? ErrorCode::kSuccess : published.error().code, test_case.expected_error);
    const auto progress = RuntimeChannelSessionTestAccess::process_local_progress_state(**session);
    ASSERT_TRUE(progress);
    EXPECT_EQ(progress->publications_in_flight, 0U);
    ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, published.has_value()));
    (*session)->cancel();
    ASSERT_TRUE((*session)->join());
    auto sequence = RuntimeChannelSessionTestAccess::output_next_sequence(**session, PortId{1U});
    ASSERT_TRUE(sequence);
    EXPECT_EQ(*sequence, test_case.next_sequence);
  }
}

// Delivery inside emit() takes, for each recipient, one transport-mutex section and two cache-mutex
// sections: the destination reservation, then the store and commit. Nothing contends here.
// REQ: RUNTIME-083
TEST(RuntimeFanoutExperiment, EmitAdmissionLockBudget) {
  SessionFixture fixture;
  use_experimental_drop_oldest_fanout(fixture);
  const auto counters = std::make_shared<RuntimePhaseCounters>();
  fixture.plan.executable_factories.phase_counters = counters;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  std::array<std::unique_ptr<PortBinding>, 2U> bindings;
  for (std::uint32_t index = 0U; index < 2U; ++index) {
    auto binding = (*session)->create_readiness_binding(
        VertexId{index + 2U},
        RuntimePortBindingRequest{.plan_generation = fixture.plan.plan_generation,
                                  .run_generation = 1U,
                                  .kind = RuntimeEventKind::kInputArrival,
                                  .source = make_source_id(index + 1U),
                                  .source_clock = ClockId{1U},
                                  .reference_clock = ClockId{1U},
                                  .capacity = 2U});
    ASSERT_TRUE(binding) << runtime_error_message(binding.error());
    bindings[index] = std::move(*binding);
  }
  ASSERT_TRUE((*session)->refresh_connections());
  ASSERT_TRUE((*session)->arm());
  const auto phase = [&](const RuntimePhaseCounter counter) {
    return counters->snapshot()[RuntimePhaseCounters::index(counter)];
  };
  ASSERT_TRUE((*session)->prepare_activation(VertexId{1U}, 1U, {}));
  const auto transport_before = phase(RuntimePhaseCounter::kChannelTransportMutexAcquisitions);
  const auto cache_before = phase(RuntimePhaseCounter::kChannelCacheMutexAcquisitions);

  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{41}));

  // Key assertion: two recipients cost two transport sections and four cache sections.
  EXPECT_EQ(phase(RuntimePhaseCounter::kChannelTransportMutexAcquisitions) - transport_before, 2U);
  EXPECT_EQ(phase(RuntimePhaseCounter::kChannelCacheMutexAcquisitions) - cache_before, 4U);
  EXPECT_EQ(phase(RuntimePhaseCounter::kChannelTransportMutexContended), 0U);
  EXPECT_EQ(phase(RuntimePhaseCounter::kChannelCacheMutexContended), 0U);
  ASSERT_TRUE((*session)->retire_activation(VertexId{1U}, 1U, true));
  (*session)->cancel();
  ASSERT_TRUE((*session)->join());
}

// A direct-retained subscriber never receives through a transport endpoint, so its head can never
// be known-blocked. Under concurrent admission, a retirement retry that releases it must not read
// that flag under its transport mutex.
// REQ: CHAN-136
TEST(RuntimeFanoutExperiment, RetirementSkipsBlockedHeadProbesOfDirectSubscribers) {
  SessionFixture fixture;
  fixture.use_direct_retained_delivery();
  const auto counters = std::make_shared<RuntimePhaseCounters>();
  fixture.plan.executable_factories.phase_counters = counters;
  auto session = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(session) << runtime_error_message(session.error());
  // Arming recomputes concurrent admission from the plan, so this retry runs unarmed.
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::enable_concurrent_admission(**session));
  const PortId direct_input = fixture.plan.process_local_channel_subscribers.front().target;
  // The test hook refuses a flag that production can never set on this subscriber.
  EXPECT_FALSE(
      RuntimeChannelSessionTestAccess::set_transport_head_blocked(**session, direct_input, true));
  const auto transport_mutex_acquisitions = [&] {
    return counters->snapshot()[RuntimePhaseCounters::index(
        RuntimePhaseCounter::kChannelTransportMutexAcquisitions)];
  };
  const auto before = transport_mutex_acquisitions();

  const std::array released{direct_input};
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::retry_released_inputs(**session, released));

  // Key assertion: the retry read no blocked flag under the transport mutex.
  EXPECT_EQ(transport_mutex_acquisitions(), before);
}

}  // namespace
}  // namespace holoscan::detail

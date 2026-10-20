/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <string>
#include <utility>
#include <vector>

#include "core/diagnostic_registry.hpp"
#include "physical_plan_test_support.hpp"

// GTest fatal assertions establish optional preconditions that clang-tidy does not model.
// NOLINTBEGIN(bugprone-unchecked-optional-access)
namespace holoscan::detail {
namespace {

using namespace physical_plan_test;

// Build a local-enum identity with an explicit native width for physical-plan fixtures.
// Representative test: CompilePhysicalPlan.LowersNativeWidthLocalEnumForDirectAndHoloMQDelivery.
[[nodiscard]] PayloadIdentityRecord local_enum_payload(std::uint8_t native_object_size = 1U) {
  return PayloadIdentityRecord{
      .kind = PayloadIdentityKind::kScalar,
      .payload_kind = PayloadKind::kInlineTyped,
      .scalar =
          ScalarIdentityRecord{
              .kind = ScalarIdentityKind::kLocalEnum,
              .native_object_size = native_object_size,
          },
  };
}

// Two edges with different queue policies leave source fanout unmerged; two services
// are produced. Generic loan pool slots must cover the maximum fanout inflight demand.
//
//   source --[blocking, depth=4, maxinflight=6, cache=5]--> left
//          --[drop-oldest, depth=2, maxinflight=3, cache=2]--> right
//
// REQ: COMPILE-009
// REQ: COMPILE-095
// REQ: COMPILE-040
TEST(CompilePhysicalPlan, ResolvesCanonicalPortsEdgesFanoutCachesAndGenericLoans) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{7U};
  snapshot.vertices = {vertex(1U, "source"), vertex(2U, "left"), vertex(3U, "right")};
  snapshot.ports = {
      output_port(1U, 1U, "out", scalar_payload(), 3U),
      input_port(2U, 2U, "in", scalar_payload(), 4U),
      input_port(3U, 3U, "in", scalar_payload(), 2U),
  };
  // Key setup: split one source's complete fanout across blocking and drop-oldest services on one
  // execution partition.
  snapshot.edges = {edge(1U, 1U, 2U, 4U), edge(2U, 1U, 3U, 2U, QueuePolicy::kDropOldest)};
  snapshot.edges.front().add_flow_options.max_inflight_samples = 6U;
  snapshot.edges.back().add_flow_options.max_inflight_samples = 3U;
  const std::array<InputSelectionPlanRecord, 2> selections{
      InputSelectionPlanRecord{.input = PortId{2U}, .cache_capacity = 5U},
      InputSelectionPlanRecord{.input = PortId{3U}, .cache_capacity = 2U},
  };
  const auto profiles = standard_profiles(snapshot.vertices.size());
  const auto partitions = default_execution_partitions(snapshot);

  auto result = lower_for_test(snapshot,
                               41U,
                               profiles,
                               selections,
                               {},
                               {},
                               nullptr,
                               {},
                               partitions,
                               kForceHoloMQProcessLocalFanout);

  for (const auto& error : result.errors) {
    ADD_FAILURE() << error.token;
  }
  EXPECT_TRUE(result.errors.empty());
  ASSERT_EQ(result.ports.size(), 3U);
  EXPECT_EQ(result.ports[0].max_emits_per_compute, 3U);
  ASSERT_EQ(result.payload_edges.size(), 2U);
  EXPECT_EQ(result.payload_edges[0].fanout, 2U);
  EXPECT_EQ(result.payload_edges[0].cache_capacity, 5U);
  EXPECT_EQ(result.payload_edges[0].max_inflight_samples, 6U);
  EXPECT_EQ(result.payload_edges[1].queue_policy, QueuePolicy::kDropOldest);
  ASSERT_EQ(result.process_local_channel_services.size(), 2U);
  ASSERT_EQ(result.process_local_channel_publishers.size(), 2U);
  ASSERT_EQ(result.process_local_channel_subscribers.size(), 2U);
  EXPECT_EQ(result.process_local_channel_services[0].source, PortId{1U});
  EXPECT_EQ(result.process_local_channel_services[0].queue_policy, QueuePolicy::kBlocking);
  EXPECT_EQ(result.process_local_channel_services[0].payload_size_bytes, 4U);
  EXPECT_EQ(result.process_local_channel_services[0].max_subscribers, 1U);
  EXPECT_EQ(result.process_local_channel_services[0].max_subscriber_queue_depth, 4U);
  EXPECT_EQ(result.process_local_channel_services[0].event_wake_reason,
            ProcessLocalHoloMQEventWakeReason::kSuppressed);
  EXPECT_EQ(result.process_local_channel_services[1].queue_policy, QueuePolicy::kDropOldest);
  EXPECT_EQ(result.process_local_channel_services[1].max_subscriber_queue_depth, 2U);
  EXPECT_EQ(result.process_local_channel_services[1].event_wake_reason,
            ProcessLocalHoloMQEventWakeReason::kSuppressed);
  EXPECT_EQ(result.process_local_channel_subscribers[0].edge, EdgeId{1U});
  EXPECT_EQ(result.process_local_channel_subscribers[0].cache_capacity, 5U);
  EXPECT_EQ(result.process_local_channel_subscribers[1].edge, EdgeId{2U});
  EXPECT_EQ(result.process_local_channel_subscribers[1].cache_capacity, 2U);
  EXPECT_EQ(result.process_local_channel_subscribers[0].readiness_capacity, 5U);
  // Key assertion: source-family completeness suppresses both policy-specific services together;
  // replay rejects a wake fact widened on only one sibling service.
  EXPECT_TRUE(std::ranges::none_of(result.process_local_channel_services,
                                   [](const auto& service) { return service.event_wake; }));
  EXPECT_TRUE(std::ranges::none_of(result.process_local_channel_subscribers,
                                   [](const auto& subscriber) { return subscriber.event_wake; }));
  // Key assertion: one loan pool; slot_count covers burst(3) + all subscriber queues + caches.
  ASSERT_EQ(result.generic_output_loans.size(), 1U);
  EXPECT_EQ(result.generic_output_loans.front().max_live_loans, 3U);
  EXPECT_EQ(result.generic_output_loans.front().slot_count, 19U);
  EXPECT_EQ(result.generic_output_loans.front().slot_size_bytes, 4U);
  EXPECT_EQ(result.generic_output_loans.front().aggregate_byte_capacity, 76U);
  EXPECT_EQ(result.generic_output_loans.front().plan_generation, 41U);
  EXPECT_EQ(result.generic_output_loans.front().run_generation, 0U);
  ASSERT_EQ(result.publication_outcomes.size(), 1U);
  EXPECT_EQ(result.publication_outcomes.front().pre_submit_authority,
            PreSubmitAuthorityDisposition::kRetainedSameActivation);
  EXPECT_FALSE(result.publication_outcomes.front().runtime_retry_enabled);
  ExecutionPlanState state;
  state.plan_generation = 41U;
  state.execution_partitions = partitions;
  detail::publish_physical_plan_records(std::move(result), state);
  EXPECT_TRUE(valid_process_local_channel_plan(state));
  state.process_local_channel_services.back().event_wake = true;
  state.process_local_channel_services.back().event_wake_reason =
      ProcessLocalHoloMQEventWakeReason::kForcedControl;
  state.process_local_channel_subscribers.back().event_wake = true;
  EXPECT_FALSE(valid_process_local_channel_plan(state));
}

// By default each eligible HoloMQ service of a family is delivered inside the producer's emit(), on
// its own. The drop-oldest service with a split target is emit-admitted and eventless; its blocking
// sibling, whose target shares the source's partition, gains nothing and stays eventless under the
// post-compute cut. A restartable target keeps its service on the pump, and replay rejects a
// changed admission fact either way.
// REQ: RUNTIME-083
TEST(CompilePhysicalPlan, SelectsDeliveryInsideEmitPerServiceByDefault) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{7U};
  snapshot.vertices = {vertex(1U, "source"), vertex(2U, "blocking"), vertex(3U, "drop-oldest")};
  snapshot.ports = {
      output_port(1U, 1U, "out", scalar_payload()),
      input_port(2U, 2U, "in", scalar_payload()),
      input_port(3U, 3U, "in", scalar_payload()),
  };
  snapshot.edges = {
      edge(1U, 1U, 2U, 1U),
      edge(2U, 1U, 3U, 1U, QueuePolicy::kDropOldest),
  };
  const auto profiles = standard_profiles(snapshot.vertices.size());
  const std::array partitions{
      ExecutionPartitionRecord{
          .id = ExecutionPartitionId{1U},
          .members = {VertexId{1U}, VertexId{2U}},
          .execution_lane = 0U,
      },
      ExecutionPartitionRecord{
          .id = ExecutionPartitionId{2U}, .members = {VertexId{3U}}, .execution_lane = 1U},
  };
  const auto service_with = [](const auto& services, const QueuePolicy policy) {
    return *std::ranges::find_if(
        services, [&](const auto& service) { return service.queue_policy == policy; });
  };

  auto result = lower_for_test(
      snapshot, 41U, profiles, {}, {}, {}, nullptr, {}, partitions, kForceHoloMQProcessLocalFanout);
  ASSERT_TRUE(result.errors.empty());
  ASSERT_EQ(result.process_local_channel_services.size(), 2U);
  const auto admitted =
      service_with(result.process_local_channel_services, QueuePolicy::kDropOldest);
  const auto colocated =
      service_with(result.process_local_channel_services, QueuePolicy::kBlocking);
  // Key assertion: only the split drop-oldest service is emit-admitted, and neither needs a wake.
  EXPECT_TRUE(admitted.emit_admitted);
  EXPECT_FALSE(admitted.event_wake);
  EXPECT_EQ(admitted.event_wake_reason, ProcessLocalHoloMQEventWakeReason::kSuppressed);
  EXPECT_FALSE(colocated.emit_admitted);
  EXPECT_FALSE(colocated.event_wake);
  EXPECT_TRUE(std::ranges::none_of(result.process_local_channel_subscribers,
                                   [](const auto& subscriber) { return subscriber.event_wake; }));
  ExecutionPlanState state;
  state.plan_generation = 41U;
  state.execution_partitions.assign(partitions.begin(), partitions.end());
  detail::publish_physical_plan_records(std::move(result), state);
  EXPECT_TRUE(valid_process_local_channel_plan(state));
  // A tampered plan that admits the colocated service, or withdraws the split one, is rejected.
  auto tampered = state;
  for (auto& service : tampered.process_local_channel_services) {
    service.emit_admitted = service.queue_policy == QueuePolicy::kBlocking;
  }
  EXPECT_FALSE(valid_process_local_channel_plan(tampered));
  tampered = state;
  for (auto& service : tampered.process_local_channel_services) {
    service.emit_admitted = false;
  }
  EXPECT_FALSE(valid_process_local_channel_plan(tampered));
  // The pump-delivery control admits nothing, so the same records no longer replay.
  tampered = state;
  tampered.force_process_local_pump_delivery = true;
  EXPECT_FALSE(valid_process_local_channel_plan(tampered));

  // A restartable drop-oldest target keeps its service eventful, and the family wake then covers
  // the blocking sibling too.
  const std::array restart_targets{LifecycleRestartTargetPlanRecord{
      .target = OperatorRef{.id = 3U, .plan_generation = 41U},
      .self_allowed = true,
      .disposition = LifecycleRestartTargetDisposition::kAdmitted,
  }};
  auto restartable = lower_for_test(snapshot,
                                    41U,
                                    profiles,
                                    {},
                                    {},
                                    restart_targets,
                                    nullptr,
                                    {},
                                    partitions,
                                    kForceHoloMQProcessLocalFanout);
  ASSERT_TRUE(restartable.errors.empty());
  EXPECT_TRUE(
      std::ranges::all_of(restartable.process_local_channel_services, [](const auto& service) {
        return !service.emit_admitted && service.event_wake &&
               service.event_wake_reason ==
                   ProcessLocalHoloMQEventWakeReason::kSplitExecutionPartition;
      }));
  ExecutionPlanState restartable_state;
  restartable_state.plan_generation = 41U;
  restartable_state.execution_partitions.assign(partitions.begin(), partitions.end());
  restartable_state.lifecycle_restart_targets.assign(restart_targets.begin(),
                                                     restart_targets.end());
  detail::publish_physical_plan_records(std::move(restartable), restartable_state);
  EXPECT_TRUE(valid_process_local_channel_plan(restartable_state));
}

// An Operator without restart authority of its own is still reconstructed when it belongs to the
// restart closure of an admitted target, so delivery inside emit() must not serve its services.
// Restart target R closes over a dependent that either publishes an inline fanout (the source
// case) or receives one (the recipient case). That fanout stays on the pump, an unrelated service
// from another producer is still emit-admitted, and replay rejects a plan that admits the
// dependent's service.
// REQ: RUNTIME-083
TEST(CompilePhysicalPlan, KeepsRestartClosureDependentsOffDeliveryInsideEmit) {
  // Vertices: R(1) restart target; S(2) fanout source; T(3), U(4) its recipients; P(5) -> Q(6) an
  // unrelated pair. Each Operator runs in its own partition.
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{7U};
  snapshot.vertices = {vertex(1U, "restart-target"),
                       vertex(2U, "source"),
                       vertex(3U, "first"),
                       vertex(4U, "second"),
                       vertex(5U, "other-source"),
                       vertex(6U, "other-sink")};
  snapshot.ports = {
      output_port(1U, 2U, "out", scalar_payload()),
      input_port(2U, 3U, "in", scalar_payload()),
      input_port(3U, 4U, "in", scalar_payload()),
      output_port(4U, 5U, "out", scalar_payload()),
      input_port(5U, 6U, "in", scalar_payload()),
  };
  snapshot.edges = {edge(1U, 1U, 2U), edge(2U, 1U, 3U), edge(3U, 4U, 5U)};
  const auto profiles = standard_profiles(snapshot.vertices.size());
  std::vector<ExecutionPartitionRecord> partitions;
  for (std::uint32_t index = 1U; index <= 6U; ++index) {
    partitions.push_back(ExecutionPartitionRecord{.id = ExecutionPartitionId{index},
                                                  .members = {VertexId{index}},
                                                  .execution_lane = index - 1U});
  }
  // R is an admitted target; the dependent has a target record that no origin requests.
  const auto targets_for = [](std::uint32_t dependent) {
    return std::array{
        LifecycleRestartTargetPlanRecord{
            .target = OperatorRef{.id = 1U, .plan_generation = 41U},
            .self_allowed = true,
            .disposition = LifecycleRestartTargetDisposition::kAdmitted,
        },
        LifecycleRestartTargetPlanRecord{
            .target = OperatorRef{.id = dependent, .plan_generation = 41U},
            .disposition = LifecycleRestartTargetDisposition::kNotRequested,
        },
    };
  };
  for (const std::uint32_t dependent : {2U, 3U}) {
    SCOPED_TRACE(dependent == 2U ? "source dependent" : "recipient dependent");
    const auto targets = targets_for(dependent);
    const std::array closures{LifecycleRestartClosurePlanRecord{
        .target = OperatorRef{.id = 1U, .plan_generation = 41U},
        .closure = {OperatorRef{.id = 1U, .plan_generation = 41U},
                    OperatorRef{.id = dependent, .plan_generation = 41U}},
    }};
    // Fanout-1 edges would be retained directly; force HoloMQ so each output has a service.
    LowerForTestOptions options = kForceHoloMQProcessLocalInline;
    options.lifecycle_closures = closures;
    auto result =
        lower_for_test(snapshot, 41U, profiles, {}, {}, targets, nullptr, {}, partitions, options);
    ASSERT_TRUE(result.errors.empty());
    const auto service_from = [&](PortId source) {
      return std::ranges::find_if(result.process_local_channel_services,
                                  [&](const auto& service) { return service.source == source; });
    };
    const auto fanout = service_from(PortId{1U});
    const auto unrelated = service_from(PortId{4U});
    ASSERT_NE(fanout, result.process_local_channel_services.end());
    ASSERT_NE(unrelated, result.process_local_channel_services.end());
    // Key assertion: the dependent's fanout stays on the pump; the unrelated service does not.
    EXPECT_FALSE(fanout->emit_admitted);
    EXPECT_TRUE(fanout->event_wake);
    EXPECT_TRUE(unrelated->emit_admitted);

    ExecutionPlanState state;
    state.plan_generation = 41U;
    state.execution_partitions = partitions;
    state.lifecycle_restart_targets.assign(targets.begin(), targets.end());
    state.lifecycle_restart_closures.assign(closures.begin(), closures.end());
    detail::publish_physical_plan_records(std::move(result), state);
    EXPECT_TRUE(valid_process_local_channel_plan(state));
    // Key assertion: replay rederives closure membership and rejects an admitted dependent.
    auto tampered = state;
    for (auto& service : tampered.process_local_channel_services) {
      if (service.source == PortId{1U}) {
        service.emit_admitted = true;
        service.event_wake = false;
        service.event_wake_reason = ProcessLocalHoloMQEventWakeReason::kSuppressed;
      }
    }
    for (auto& subscriber : tampered.process_local_channel_subscribers) {
      if (subscriber.source == PortId{1U}) {
        subscriber.event_wake = false;
      }
    }
    EXPECT_FALSE(valid_process_local_channel_plan(tampered));
  }
}

// A drop-oldest Tensor fanout family kept on HoloMQ, here by the source-private fanout control
// (CHAN-162 makes such a family direct by default), has subscribers that stage like inline ones
// (CHAN-161), so delivery inside emit() can serve it. By default that admits a host-memory family;
// a device family moves descriptor import onto the producer's lane and a blocking one is
// unmeasured, so both stay on the pump unless the private experiment admits them. Replay rejects a
// changed admission fact.
//
//   source [partition 1] --[Tensor]--> first  [partition 2]
//                        \--[Tensor]--> second [partition 3]
//
// REQ: RUNTIME-085
TEST(CompilePhysicalPlan, SelectsDeliveryInsideEmitForHostDropOldestTensorFamilies) {
  const auto lower = [](const QueuePolicy policy,
                        const MemoryKind memory,
                        const ProcessLocalEmitAdmissionScope scope) {
    GraphSnapshotData snapshot;
    snapshot.generation = GraphGeneration{7U};
    snapshot.vertices = {vertex(1U, "source"), vertex(2U, "first"), vertex(3U, "second")};
    snapshot.ports = {
        output_port(1U, 1U, "out", tensor_payload()),
        input_port(2U, 2U, "in", tensor_payload(), 1U),
        input_port(3U, 3U, "in", tensor_payload(), 1U),
    };
    snapshot.ports.front().setup.tensor_layout = TensorRepresentation{
        .memory_kind = memory,
        .dtype = DLDataType{kDLUInt, 8U, 1U},
        .rank = 1U,
    };
    snapshot.ports.front().setup.tensor_allocation =
        TensorPoolBounds{.max_rank = 1U, .max_byte_span = 64U};
    snapshot.edges = {edge(1U, 1U, 2U, 1U, policy), edge(2U, 1U, 3U, 1U, policy)};
    const std::array partitions{
        ExecutionPartitionRecord{
            .id = ExecutionPartitionId{1U}, .members = {VertexId{1U}}, .execution_lane = 0U},
        ExecutionPartitionRecord{
            .id = ExecutionPartitionId{2U}, .members = {VertexId{2U}}, .execution_lane = 1U},
        ExecutionPartitionRecord{
            .id = ExecutionPartitionId{3U}, .members = {VertexId{3U}}, .execution_lane = 2U},
    };
    std::vector<TensorDeviceBindingRecord> bindings;
    if (memory == MemoryKind::kCudaDevice) {
      bindings.push_back(TensorDeviceBindingRecord{.port = PortId{1U}, .device = DeviceId{0}});
    }
    auto result = lower_for_test(
        snapshot,
        41U,
        standard_profiles(snapshot.vertices.size()),
        {},
        {},
        {},
        nullptr,
        bindings,
        partitions,
        LowerForTestOptions{.force_holomq_process_local_fanout = true, .emit_admission = scope});
    EXPECT_TRUE(result.errors.empty())
        << (result.errors.empty() ? std::string{} : result.errors.front().token);
    EXPECT_EQ(result.process_local_channel_services.size(), 1U);
    ExecutionPlanState state;
    state.plan_generation = 41U;
    state.execution_partitions.assign(partitions.begin(), partitions.end());
    if (scope == ProcessLocalEmitAdmissionScope::kAllSupported) {
      state.fanout_experiment = ProcessLocalFanoutExperiment::kEmitRecipients;
    }
    detail::publish_physical_plan_records(std::move(result), state);
    return state;
  };
  const auto admitted = [](const ExecutionPlanState& state) {
    return !state.process_local_channel_services.empty() &&
           state.process_local_channel_services.front().delivery_kind ==
               ProcessLocalChannelDeliveryKind::kHoloMQ &&
           state.process_local_channel_services.front().emit_admitted;
  };

  // Key assertion: the default admits only the host-memory drop-oldest family.
  const auto host = lower(
      QueuePolicy::kDropOldest, MemoryKind::kHost, ProcessLocalEmitAdmissionScope::kInlinePayloads);
  EXPECT_TRUE(admitted(host));
  EXPECT_FALSE(host.process_local_channel_services.front().event_wake);
  EXPECT_TRUE(valid_process_local_channel_plan(host));
  const auto device = lower(QueuePolicy::kDropOldest,
                            MemoryKind::kCudaDevice,
                            ProcessLocalEmitAdmissionScope::kInlinePayloads);
  EXPECT_FALSE(admitted(device));
  EXPECT_TRUE(valid_process_local_channel_plan(device));
  const auto blocking = lower(
      QueuePolicy::kBlocking, MemoryKind::kHost, ProcessLocalEmitAdmissionScope::kInlinePayloads);
  EXPECT_FALSE(admitted(blocking));
  EXPECT_TRUE(valid_process_local_channel_plan(blocking));

  // The private experiment admits the device and blocking families too.
  for (const auto& experiment : {lower(QueuePolicy::kDropOldest,
                                       MemoryKind::kCudaDevice,
                                       ProcessLocalEmitAdmissionScope::kAllSupported),
                                 lower(QueuePolicy::kBlocking,
                                       MemoryKind::kHost,
                                       ProcessLocalEmitAdmissionScope::kAllSupported)}) {
    EXPECT_TRUE(admitted(experiment));
    EXPECT_TRUE(valid_process_local_channel_plan(experiment));
  }

  // Replay rejects a withdrawn host admission and a device admission without the experiment.
  auto tampered = host;
  tampered.process_local_channel_services.front().emit_admitted = false;
  EXPECT_FALSE(valid_process_local_channel_plan(tampered));
  tampered = device;
  tampered.process_local_channel_services.front().emit_admitted = true;
  EXPECT_FALSE(valid_process_local_channel_plan(tampered));
}

// Delivery inside emit() keeps a consumer's arrivals from one producer in publication order across
// that producer's outputs only while the pump serves none of the services between the two: the
// pair rule direct retention also follows. Two outputs of one source feed split targets;
// both are emit-admitted until the status target becomes restartable. The pump then serves the
// status output, but the data output feeds a different consumer and stays admitted. When the
// data output also feeds the restartable target, the pump serves the pair it shares with the data
// sink, so the data sink's status output is withheld too. Replay rejects a plan that admits a
// withheld service or withholds an admitted one.
// REQ: RUNTIME-016
// REQ: RUNTIME-083
TEST(CompilePhysicalPlan, WithholdsDeliveryInsideEmitOnlyFromPairsThePumpServes) {
  const auto profiles = standard_profiles(3U);
  const std::array partitions{
      ExecutionPartitionRecord{
          .id = ExecutionPartitionId{1U}, .members = {VertexId{1U}}, .execution_lane = 0U},
      ExecutionPartitionRecord{
          .id = ExecutionPartitionId{2U}, .members = {VertexId{2U}}, .execution_lane = 1U},
      ExecutionPartitionRecord{
          .id = ExecutionPartitionId{3U}, .members = {VertexId{3U}}, .execution_lane = 2U},
  };
  const std::array restart_targets{LifecycleRestartTargetPlanRecord{
      .target = OperatorRef{.id = 3U, .plan_generation = 41U},
      .self_allowed = true,
      .disposition = LifecycleRestartTargetDisposition::kAdmitted,
  }};
  const auto publish = [&](PhysicalPlanLoweringResult result,
                           std::span<const LifecycleRestartTargetPlanRecord>
                               targets) {
    ExecutionPlanState state;
    state.plan_generation = 41U;
    state.execution_partitions.assign(partitions.begin(), partitions.end());
    state.lifecycle_restart_targets.assign(targets.begin(), targets.end());
    detail::publish_physical_plan_records(std::move(result), state);
    return state;
  };
  // Fanout-1 edges would be retained directly; force HoloMQ so each output has a service.
  const auto lower = [&](const GraphSnapshotData& snapshot,
                         std::span<const LifecycleRestartTargetPlanRecord>
                             targets) {
    return lower_for_test(snapshot,
                          41U,
                          profiles,
                          {},
                          {},
                          targets,
                          nullptr,
                          {},
                          partitions,
                          kForceHoloMQProcessLocalInline);
  };
  const auto service_from = [](const auto& services, PortId source) {
    return std::ranges::find_if(services,
                                [&](const auto& service) { return service.source == source; });
  };

  GraphSnapshotData separate;
  separate.generation = GraphGeneration{7U};
  separate.vertices = {vertex(1U, "source"), vertex(2U, "data-sink"), vertex(3U, "status-sink")};
  separate.ports = {
      output_port(1U, 1U, "data", scalar_payload()),
      output_port(2U, 1U, "status", scalar_payload()),
      input_port(3U, 2U, "in", scalar_payload()),
      input_port(4U, 3U, "in", scalar_payload()),
  };
  separate.edges = {edge(1U, 1U, 3U), edge(2U, 2U, 4U)};

  auto admitted = lower(separate, {});
  ASSERT_TRUE(admitted.errors.empty());
  ASSERT_EQ(admitted.process_local_channel_services.size(), 2U);
  EXPECT_TRUE(std::ranges::all_of(admitted.process_local_channel_services, [](const auto& service) {
    return service.emit_admitted && !service.event_wake;
  }));
  EXPECT_TRUE(valid_process_local_channel_plan(publish(std::move(admitted), {})));

  auto separate_pumped = lower(separate, restart_targets);
  ASSERT_TRUE(separate_pumped.errors.empty());
  const auto& separate_services = separate_pumped.process_local_channel_services;
  const auto data = service_from(separate_services, PortId{1U});
  const auto status = service_from(separate_services, PortId{2U});
  ASSERT_NE(data, separate_services.end());
  ASSERT_NE(status, separate_services.end());
  // Key assertion: the pump serves only the restartable consumer's pair.
  EXPECT_TRUE(data->emit_admitted);
  EXPECT_FALSE(data->event_wake);
  EXPECT_FALSE(status->emit_admitted);
  EXPECT_TRUE(status->event_wake);
  const ExecutionPlanState separate_state = publish(std::move(separate_pumped), restart_targets);
  EXPECT_TRUE(valid_process_local_channel_plan(separate_state));
  auto tampered = separate_state;
  for (auto& service : tampered.process_local_channel_services) {
    if (service.source == PortId{1U}) {
      service.emit_admitted = false;
      service.event_wake = true;
      service.event_wake_reason = ProcessLocalHoloMQEventWakeReason::kSplitExecutionPartition;
    }
  }
  EXPECT_FALSE(valid_process_local_channel_plan(tampered));

  GraphSnapshotData shared = separate;
  shared.ports.push_back(input_port(5U, 2U, "status-in", scalar_payload()));
  shared.edges = {edge(1U, 1U, 3U), edge(2U, 1U, 4U), edge(3U, 2U, 5U)};
  auto shared_pumped = lower(shared, restart_targets);
  ASSERT_TRUE(shared_pumped.errors.empty());
  // Key assertion: the data output's pumped family reaches the data sink, so the data sink's
  // status output from the same producer stays on the pump with it.
  EXPECT_TRUE(std::ranges::all_of(
      shared_pumped.process_local_channel_services,
      [](const auto& service) { return !service.emit_admitted && service.event_wake; }));
  const ExecutionPlanState shared_state = publish(std::move(shared_pumped), restart_targets);
  EXPECT_TRUE(valid_process_local_channel_plan(shared_state));
  tampered = shared_state;
  for (auto& service : tampered.process_local_channel_services) {
    if (service.source == PortId{2U}) {
      service.emit_admitted = true;
      service.event_wake = false;
      service.event_wake_reason = ProcessLocalHoloMQEventWakeReason::kSuppressed;
    }
  }
  for (auto& subscriber : tampered.process_local_channel_subscribers) {
    if (subscriber.source == PortId{2U}) {
      subscriber.event_wake = false;
    }
  }
  EXPECT_FALSE(valid_process_local_channel_plan(tampered));
}

// An eventless HoloMQ service is deferred too: the producer's activation cut admits it after the
// callback, so a consumer that also receives an emit-admitted output of the same producer would see
// a later publication first. The local sink receives the status output on the producer's own lane,
// eventless, and the data output beside a sink in another partition; the data output is withheld.
// Without the status edge it is admitted. Replay rejects a plan that admits it beside the status
// service.
// REQ: RUNTIME-016
// REQ: RUNTIME-083
TEST(CompilePhysicalPlan, WithholdsDeliveryInsideEmitBesideAnEventlessSibling) {
  const auto profiles = standard_profiles(3U);
  const std::array partitions{
      ExecutionPartitionRecord{.id = ExecutionPartitionId{1U},
                               .members = {VertexId{1U}, VertexId{2U}},
                               .execution_lane = 0U},
      ExecutionPartitionRecord{
          .id = ExecutionPartitionId{2U}, .members = {VertexId{3U}}, .execution_lane = 1U},
  };
  // Fanout-1 edges would be retained directly; force HoloMQ so each output has a service.
  const auto lower = [&](const GraphSnapshotData& snapshot) {
    return lower_for_test(snapshot,
                          41U,
                          profiles,
                          {},
                          {},
                          {},
                          nullptr,
                          {},
                          partitions,
                          kForceHoloMQProcessLocalInline);
  };
  const auto publish = [&](PhysicalPlanLoweringResult result) {
    ExecutionPlanState state;
    state.plan_generation = 41U;
    state.execution_partitions.assign(partitions.begin(), partitions.end());
    detail::publish_physical_plan_records(std::move(result), state);
    return state;
  };
  const auto service_from = [](const auto& services, PortId source) {
    return std::ranges::find_if(services,
                                [&](const auto& service) { return service.source == source; });
  };

  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{7U};
  snapshot.vertices = {vertex(1U, "source"), vertex(2U, "local-sink"), vertex(3U, "remote-sink")};
  snapshot.ports = {
      output_port(1U, 1U, "status", scalar_payload()),
      output_port(2U, 1U, "data", scalar_payload()),
      input_port(3U, 2U, "status-in", scalar_payload()),
      input_port(4U, 2U, "data-in", scalar_payload()),
      input_port(5U, 3U, "in", scalar_payload()),
  };
  snapshot.edges = {edge(1U, 1U, 3U), edge(2U, 2U, 4U), edge(3U, 2U, 5U)};

  GraphSnapshotData without_status = snapshot;
  without_status.edges = {edge(2U, 2U, 4U), edge(3U, 2U, 5U)};
  auto admitted = lower(without_status);
  ASSERT_TRUE(admitted.errors.empty());
  ASSERT_EQ(admitted.process_local_channel_services.size(), 1U);
  EXPECT_TRUE(admitted.process_local_channel_services.front().emit_admitted);

  auto withheld = lower(snapshot);
  ASSERT_TRUE(withheld.errors.empty());
  const auto& services = withheld.process_local_channel_services;
  const auto status = service_from(services, PortId{1U});
  const auto data = service_from(services, PortId{2U});
  ASSERT_NE(status, services.end());
  ASSERT_NE(data, services.end());
  EXPECT_FALSE(status->emit_admitted);
  EXPECT_FALSE(status->event_wake);
  // Key assertion: the local sink's eventless status service keeps the data output off emit
  // admission, so the data output wakes its consumers like any deferred service.
  EXPECT_FALSE(data->emit_admitted);
  EXPECT_TRUE(data->event_wake);

  const ExecutionPlanState state = publish(std::move(withheld));
  EXPECT_TRUE(valid_process_local_channel_plan(state));
  auto tampered = state;
  for (auto& service : tampered.process_local_channel_services) {
    if (service.source == PortId{2U}) {
      service.emit_admitted = true;
      service.event_wake = false;
      service.event_wake_reason = ProcessLocalHoloMQEventWakeReason::kSuppressed;
    }
  }
  for (auto& subscriber : tampered.process_local_channel_subscribers) {
    if (subscriber.source == PortId{2U}) {
      subscriber.event_wake = false;
    }
  }
  EXPECT_FALSE(valid_process_local_channel_plan(tampered));
}

// A mixed-policy source family is one event-wake decision. One split target keeps wake attachment
// on both policy-specific services even though the blocking sibling is otherwise eligible.
// REQ: CHAN-101
// REQ: RUNTIME-019
TEST(CompilePhysicalPlan, KeepsMixedPolicySiblingEventWakesWhenOneTargetUsesAnotherPartition) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{7U};
  snapshot.vertices = {vertex(1U, "source"), vertex(2U, "blocking"), vertex(3U, "drop-oldest")};
  snapshot.ports = {
      output_port(1U, 1U, "out", scalar_payload()),
      input_port(2U, 2U, "in", scalar_payload()),
      input_port(3U, 3U, "in", scalar_payload()),
  };
  snapshot.edges = {
      edge(1U, 1U, 2U, 1U),
      edge(2U, 1U, 3U, 1U, QueuePolicy::kDropOldest),
  };
  const auto profiles = standard_profiles(snapshot.vertices.size());
  const std::array partitions{
      ExecutionPartitionRecord{
          .id = ExecutionPartitionId{1U},
          .members = {VertexId{1U}, VertexId{2U}},
          .execution_lane = 0U,
      },
      ExecutionPartitionRecord{
          .id = ExecutionPartitionId{2U}, .members = {VertexId{3U}}, .execution_lane = 1U},
  };

  // Key setup: lower one complete mixed-policy family with only its drop-oldest target split from
  // the source and blocking target.
  // Pump delivery keeps the family-wide wake contract; compile() admits the split service
  // inside emit() instead (SelectsDeliveryInsideEmitPerServiceByDefault).
  auto result =
      lower_for_test(snapshot, 41U, profiles, {}, {}, {}, nullptr, {}, partitions, kPumpDelivery);

  // Key assertion: the split sibling's retained-wake reason is projected onto both services, and
  // replay rejects an attempted family-wide suppression.
  ASSERT_TRUE(result.errors.empty());
  ASSERT_EQ(result.process_local_channel_services.size(), 2U);
  ASSERT_EQ(result.process_local_channel_subscribers.size(), 2U);
  EXPECT_TRUE(std::ranges::all_of(result.process_local_channel_services, [](const auto& service) {
    return service.event_wake &&
           service.event_wake_reason == ProcessLocalHoloMQEventWakeReason::kSplitExecutionPartition;
  }));
  EXPECT_TRUE(std::ranges::all_of(result.process_local_channel_subscribers,
                                  [](const auto& subscriber) { return subscriber.event_wake; }));
  ExecutionPlanState state;
  state.plan_generation = 41U;
  state.force_process_local_pump_delivery = true;
  state.execution_partitions.assign(partitions.begin(), partitions.end());
  detail::publish_physical_plan_records(std::move(result), state);
  EXPECT_TRUE(valid_process_local_channel_plan(state));
  for (auto& service : state.process_local_channel_services) {
    service.event_wake = false;
    service.event_wake_reason = ProcessLocalHoloMQEventWakeReason::kSuppressed;
  }
  for (auto& subscriber : state.process_local_channel_subscribers) {
    subscriber.event_wake = false;
  }
  EXPECT_FALSE(valid_process_local_channel_plan(state));
}

// Both edges use the same queue policy (blocking), so the compiler merges them into
// a single immutable service with max_subscribers=2. Contrast with
// ResolvesCanonicalPortsEdgesFanoutCachesAndGenericLoans where differing policies
// prevent merging.
//
//   source --[blocking, depth=4]--> left
//          --[blocking, depth=2]--> right
//
// REQ: COMPILE-009
// REQ: COMPILE-095
// REQ: COMPILE-040
// REQ: CHAN-065
// REQ: CHAN-071
// REQ: CHAN-096
TEST(CompilePhysicalPlan, GroupsCompatibleProcessLocalFanoutIntoOneImmutableService) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{7U};
  snapshot.vertices = {vertex(1U, "source"), vertex(2U, "left"), vertex(3U, "right")};
  snapshot.ports = {
      output_port(1U, 1U, "out", scalar_payload(), 3U),
      input_port(2U, 2U, "in", scalar_payload(), 4U),
      input_port(3U, 3U, "in", scalar_payload(), 2U),
  };
  // Both edges are blocking: same policy allows merging into one service.
  snapshot.edges = {edge(1U, 1U, 2U, 4U), edge(2U, 1U, 3U, 2U)};
  snapshot.edges.front().add_flow_options.max_inflight_samples = 5U;
  snapshot.edges.back().add_flow_options.max_inflight_samples = 3U;
  const std::array<InputSelectionPlanRecord, 2> selections{
      InputSelectionPlanRecord{.input = PortId{2U}, .cache_capacity = 5U},
      InputSelectionPlanRecord{.input = PortId{3U}, .cache_capacity = 2U},
  };
  const auto profiles = standard_profiles(snapshot.vertices.size());
  const auto partitions = default_execution_partitions(snapshot);

  auto result = lower_for_test(snapshot,
                               41U,
                               profiles,
                               selections,
                               {},
                               {},
                               nullptr,
                               {},
                               partitions,
                               kForceHoloMQProcessLocalFanout);

  EXPECT_TRUE(result.errors.empty());
  // Key assertion: same policy → one shared immutable service covering both subscribers.
  ASSERT_EQ(result.process_local_channel_services.size(), 1U);
  ASSERT_EQ(result.process_local_channel_publishers.size(), 1U);
  ASSERT_EQ(result.process_local_channel_subscribers.size(), 2U);
  const auto& service = result.process_local_channel_services.front();
  EXPECT_EQ(service.source, PortId{1U});
  EXPECT_EQ(service.service_id, "process-local-channel.41.source.1.policy.0");
  EXPECT_EQ(service.payload_size_bytes, 4U);
  EXPECT_EQ(service.max_subscribers, 2U);
  EXPECT_EQ(service.max_subscriber_queue_depth, 4U);
  EXPECT_EQ(service.max_subscriber_borrowed_samples, 4U);
  EXPECT_EQ(service.publisher_history_depth, 0U);
  EXPECT_EQ(service.delivery_kind, ProcessLocalChannelDeliveryKind::kHoloMQ);
  EXPECT_FALSE(service.event_wake);
  EXPECT_EQ(service.event_wake_reason, ProcessLocalHoloMQEventWakeReason::kSuppressed);
  EXPECT_EQ(result.process_local_channel_publishers.front().service_id, service.service_id);
  EXPECT_EQ(result.process_local_channel_publishers.front().max_loaned_samples, 3U);
  EXPECT_EQ(result.process_local_channel_subscribers[0].service_id, service.service_id);
  EXPECT_EQ(result.process_local_channel_subscribers[0].requested_queue_depth, 4U);
  EXPECT_EQ(result.process_local_channel_subscribers[0].max_borrowed_samples, 4U);
  EXPECT_EQ(result.process_local_channel_subscribers[0].cache_capacity, 5U);
  EXPECT_EQ(result.process_local_channel_subscribers[1].service_id, service.service_id);
  EXPECT_EQ(result.process_local_channel_subscribers[1].requested_queue_depth, 2U);
  EXPECT_EQ(result.process_local_channel_subscribers[1].max_borrowed_samples, 2U);
  EXPECT_EQ(result.process_local_channel_subscribers[1].cache_capacity, 2U);
  EXPECT_TRUE(std::ranges::none_of(result.process_local_channel_subscribers,
                                   [](const auto& subscriber) { return subscriber.event_wake; }));

  ExecutionPlanState state;
  state.plan_generation = 41U;
  state.execution_partitions = partitions;
  detail::publish_physical_plan_records(std::move(result), state);
  EXPECT_TRUE(valid_process_local_channel_plan(state));
  const auto applicability = inspect_process_local_holomq_event_wake_applicability(state);
  ASSERT_EQ(applicability.size(), 1U);
  EXPECT_EQ(applicability.front().source, PortId{1U});
  const std::string applicability_service_id = applicability.front().service_id;
  EXPECT_EQ(applicability_service_id, state.process_local_channel_services.front().service_id);
  EXPECT_EQ(applicability.front().planned_subscriber_count, 2U);
  EXPECT_EQ(applicability.front().subscriber_record_count, 2U);
  EXPECT_EQ(applicability.front().payload_edge_count, 2U);
  EXPECT_FALSE(applicability.front().event_wake);
  EXPECT_EQ(applicability.front().reason, ProcessLocalHoloMQEventWakeReason::kSuppressed);
  EXPECT_EQ(process_local_holomq_event_wake_reason_token(applicability.front().reason),
            "suppressed");
  // Key assertion: the cold inspection result owns service identity independently of plan state.
  state.process_local_channel_services.front().service_id = "mutated-after-inspection";
  EXPECT_EQ(applicability.front().service_id, applicability_service_id);
  state.process_local_channel_services.front().service_id = applicability_service_id;
  state.process_local_channel_services.front().event_wake_reason =
      ProcessLocalHoloMQEventWakeReason::kReplayAmbiguity;
  EXPECT_FALSE(valid_process_local_channel_plan(state));
  state.process_local_channel_services.front().event_wake_reason =
      ProcessLocalHoloMQEventWakeReason::kSuppressed;
  state.process_local_channel_subscribers.back().event_wake = true;
  EXPECT_FALSE(valid_process_local_channel_plan(state));
}

// A shared scalar service is eventless only when every consumer belongs to the producer's exact
// execution partition. One split target keeps wake attachment uniform across the whole service.
// REQ: CHAN-101
// REQ: RUNTIME-019
TEST(CompilePhysicalPlan, KeepsScalarFanoutEventWakeWhenAnyTargetUsesAnotherPartition) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{7U};
  snapshot.vertices = {vertex(1U, "source"), vertex(2U, "left"), vertex(3U, "right")};
  snapshot.ports = {
      output_port(1U, 1U, "out", scalar_payload()),
      input_port(2U, 2U, "in", scalar_payload()),
      input_port(3U, 3U, "in", scalar_payload()),
  };
  snapshot.edges = {edge(1U, 1U, 2U), edge(2U, 1U, 3U)};
  const auto profiles = standard_profiles(snapshot.vertices.size());
  const std::array partitions{
      ExecutionPartitionRecord{
          .id = ExecutionPartitionId{1U},
          .members = {VertexId{1U}, VertexId{2U}},
          .execution_lane = 0U,
      },
      ExecutionPartitionRecord{
          .id = ExecutionPartitionId{2U}, .members = {VertexId{3U}}, .execution_lane = 1U},
  };

  // Key setup: lower one otherwise-compatible scalar fanout service with one split target.
  // Pump delivery keeps the family-wide wake contract; compile() admits the split service
  // inside emit() instead (SelectsDeliveryInsideEmitPerServiceByDefault).
  auto result =
      lower_for_test(snapshot, 41U, profiles, {}, {}, {}, nullptr, {}, partitions, kPumpDelivery);

  // Key assertion: the service and both subscribers retain event wakes, and replay rejects a
  // partial or service-wide attempt to suppress them.
  ASSERT_TRUE(result.errors.empty());
  ASSERT_EQ(result.process_local_channel_services.size(), 1U);
  ASSERT_EQ(result.process_local_channel_subscribers.size(), 2U);
  EXPECT_TRUE(result.process_local_channel_services.front().event_wake);
  EXPECT_EQ(result.process_local_channel_services.front().event_wake_reason,
            ProcessLocalHoloMQEventWakeReason::kSplitExecutionPartition);
  EXPECT_TRUE(std::ranges::all_of(result.process_local_channel_subscribers,
                                  [](const auto& subscriber) { return subscriber.event_wake; }));
  ExecutionPlanState state;
  state.plan_generation = 41U;
  state.force_process_local_pump_delivery = true;
  state.execution_partitions.assign(partitions.begin(), partitions.end());
  detail::publish_physical_plan_records(std::move(result), state);
  EXPECT_TRUE(valid_process_local_channel_plan(state));
  state.process_local_channel_services.front().event_wake = false;
  for (auto& subscriber : state.process_local_channel_subscribers) {
    subscriber.event_wake = false;
  }
  EXPECT_FALSE(valid_process_local_channel_plan(state));
}

// A complete lossy fanout service uses the same synchronous activation-cut proof as blocking;
// replacement attribution remains enabled on its one publisher. compile() delivers this family by
// direct retention (CHAN-160), so the fanout seam keeps it on HoloMQ here.
// REQ: CHAN-101
// REQ: CHAN-115
// REQ: RUNTIME-019
TEST(CompilePhysicalPlan, SuppressesDropOldestScalarFanoutEventWakeServiceWide) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{7U};
  snapshot.vertices = {vertex(1U, "source"), vertex(2U, "left"), vertex(3U, "right")};
  snapshot.ports = {
      output_port(1U, 1U, "out", scalar_payload()),
      input_port(2U, 2U, "in", scalar_payload()),
      input_port(3U, 3U, "in", scalar_payload()),
  };
  snapshot.edges = {
      edge(1U, 1U, 2U, 1U, QueuePolicy::kDropOldest),
      edge(2U, 1U, 3U, 1U, QueuePolicy::kDropOldest),
  };
  const auto profiles = standard_profiles(snapshot.vertices.size());
  const auto partitions = default_execution_partitions(snapshot);

  // Key setup: lower one complete two-target drop-oldest scalar service on a shared partition.
  auto result = lower_for_test(
      snapshot, 41U, profiles, {}, {}, {}, nullptr, {}, partitions, kForceHoloMQProcessLocalFanout);

  // Key assertion: wake suppression is uniform and the publisher retains lossy replacement
  // attribution; replay rejects a wake fact changed on only one subscriber.
  ASSERT_TRUE(result.errors.empty());
  ASSERT_EQ(result.process_local_channel_services.size(), 1U);
  ASSERT_EQ(result.process_local_channel_publishers.size(), 1U);
  ASSERT_EQ(result.process_local_channel_subscribers.size(), 2U);
  EXPECT_EQ(result.process_local_channel_services.front().queue_policy, QueuePolicy::kDropOldest);
  EXPECT_FALSE(result.process_local_channel_services.front().event_wake);
  EXPECT_EQ(result.process_local_channel_services.front().event_wake_reason,
            ProcessLocalHoloMQEventWakeReason::kSuppressed);
  EXPECT_TRUE(result.process_local_channel_publishers.front().replacement_attribution_required);
  EXPECT_TRUE(std::ranges::none_of(result.process_local_channel_subscribers,
                                   [](const auto& subscriber) { return subscriber.event_wake; }));
  ExecutionPlanState state;
  state.plan_generation = 41U;
  state.execution_partitions = partitions;
  detail::publish_physical_plan_records(std::move(result), state);
  EXPECT_TRUE(valid_process_local_channel_plan(state));
  state.process_local_channel_subscribers.front().event_wake = true;
  EXPECT_FALSE(valid_process_local_channel_plan(state));
}

// REQ: COMPILE-009
// REQ: COMPILE-095
// REQ: COMPILE-040
// REQ: CHAN-065
// REQ: CHAN-096
// REQ: CHAN-116
// REQ: CHAN-128
TEST(CompilePhysicalPlan, UsesNativeBooleanWidthForPrivateProcessLocalStorage) {
  PayloadIdentityRecord boolean_payload{
      .kind = PayloadIdentityKind::kScalar,
      .payload_kind = PayloadKind::kInlineTyped,
      .scalar = ScalarIdentityRecord{.kind = ScalarIdentityKind::kBoolean},
  };
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{7U};
  snapshot.vertices = {vertex(1U, "source"), vertex(2U, "sink")};
  snapshot.ports = {
      output_port(1U, 1U, "out", boolean_payload),
      input_port(2U, 2U, "in", std::move(boolean_payload)),
  };
  snapshot.edges = {edge(1U, 1U, 2U)};
  const auto profiles = standard_profiles(snapshot.vertices.size());

  const auto result = lower_for_test(snapshot, 41U, profiles);

  EXPECT_TRUE(result.errors.empty());
  ASSERT_EQ(result.process_local_channel_services.size(), 1U);
  const auto& service = result.process_local_channel_services.front();
  EXPECT_EQ(service.payload_size_bytes, sizeof(bool));
  EXPECT_EQ(service.delivery_kind, ProcessLocalChannelDeliveryKind::kDirectRetained);
  EXPECT_FALSE(service.event_wake);
  EXPECT_EQ(service.event_wake_reason, ProcessLocalHoloMQEventWakeReason::kNotHoloMQ);
  ASSERT_EQ(result.process_local_channel_subscribers.size(), 1U);
  EXPECT_FALSE(result.process_local_channel_subscribers.front().event_wake);
  ASSERT_EQ(result.generic_output_loans.size(), 1U);
  EXPECT_EQ(result.generic_output_loans.front().slot_size_bytes, sizeof(bool));

  snapshot.edges.front().add_flow_options.queue_policy = QueuePolicy::kDropOldest;
  snapshot.edges.front().add_flow_options.queue_depth = 4U;
  snapshot.edges.front().add_flow_options.max_inflight_samples = 4U;
  snapshot.ports[1].setup.queue_depth = 4U;
  const auto drop_oldest = lower_for_test(snapshot, 41U, profiles);
  ASSERT_TRUE(drop_oldest.errors.empty());
  ASSERT_EQ(drop_oldest.process_local_channel_services.size(), 1U);
  EXPECT_EQ(drop_oldest.process_local_channel_services.front().delivery_kind,
            ProcessLocalChannelDeliveryKind::kDirectRetained);
  EXPECT_FALSE(drop_oldest.process_local_channel_services.front().event_wake);
  ASSERT_EQ(drop_oldest.process_local_channel_publishers.size(), 1U);
  EXPECT_TRUE(
      drop_oldest.process_local_channel_publishers.front().replacement_attribution_required);

  const auto fallback = lower_for_test(
      snapshot, 41U, profiles, {}, {}, {}, nullptr, {}, {}, kForceHoloMQProcessLocalInline);
  EXPECT_TRUE(fallback.errors.empty());
  ASSERT_EQ(fallback.process_local_channel_services.size(), 1U);
  EXPECT_EQ(fallback.process_local_channel_services.front().delivery_kind,
            ProcessLocalChannelDeliveryKind::kHoloMQ);
  EXPECT_FALSE(fallback.process_local_channel_services.front().event_wake);
  EXPECT_EQ(fallback.process_local_channel_services.front().event_wake_reason,
            ProcessLocalHoloMQEventWakeReason::kSuppressed);
}

// REQ: GRAPH-042
// REQ: COMPILE-009
// REQ: COMPILE-095
// REQ: CHAN-065
// REQ: CHAN-096
// REQ: CHAN-116
// REQ: CHAN-147
TEST(CompilePhysicalPlan, LowersNativeWidthLocalEnumForDirectAndHoloMQDelivery) {
  // Key setup: lower a two-byte local enum first to one sink, then to a second fanout sink.
  // Graph topology: source.out --> left.in; source.out --[fanout case only]--> right.in
  // Key assertion: direct delivery needs no generic output loans, and both paths retain native
  // width; the published fanout plan rejects a payload identity whose width no longer matches
  // storage.
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{7U};
  snapshot.vertices = {
      vertex(1U, "source"),
      vertex(2U, "left"),
      vertex(3U, "right"),
  };
  snapshot.ports = {
      output_port(1U, 1U, "out", local_enum_payload(2U)),
      input_port(2U, 2U, "in", local_enum_payload(2U)),
      input_port(3U, 3U, "in", local_enum_payload(2U)),
  };
  snapshot.edges = {edge(1U, 1U, 2U)};
  const auto profiles = standard_profiles(snapshot.vertices.size());
  const auto partitions = default_execution_partitions(snapshot);

  const auto direct = lower_for_test(snapshot, 41U, profiles, {}, {}, {}, nullptr, {}, partitions);

  ASSERT_TRUE(direct.errors.empty());
  ASSERT_EQ(direct.process_local_channel_services.size(), 1U);
  EXPECT_EQ(direct.process_local_channel_services.front().payload_size_bytes, 2U);
  EXPECT_EQ(direct.process_local_channel_services.front().delivery_kind,
            ProcessLocalChannelDeliveryKind::kDirectRetained);
  EXPECT_TRUE(direct.generic_output_loans.empty());

  snapshot.edges.push_back(edge(2U, 1U, 3U));
  auto fanout = lower_for_test(
      snapshot, 42U, profiles, {}, {}, {}, nullptr, {}, partitions, kForceHoloMQProcessLocalFanout);

  ASSERT_TRUE(fanout.errors.empty());
  ASSERT_EQ(fanout.process_local_channel_services.size(), 1U);
  EXPECT_EQ(fanout.process_local_channel_services.front().payload_size_bytes, 2U);
  EXPECT_EQ(fanout.process_local_channel_services.front().max_subscribers, 2U);
  EXPECT_EQ(fanout.process_local_channel_services.front().delivery_kind,
            ProcessLocalChannelDeliveryKind::kHoloMQ);
  ASSERT_EQ(fanout.process_local_channel_subscribers.size(), 2U);
  ExecutionPlanState state;
  state.plan_generation = 42U;
  state.execution_partitions = partitions;
  detail::publish_physical_plan_records(std::move(fanout), state);
  EXPECT_TRUE(valid_process_local_channel_plan(state));
  state.process_local_channel_services.front().payload_identity.scalar->native_object_size = 4U;
  EXPECT_FALSE(valid_process_local_channel_plan(state));
}

// REQ: GRAPH-042
// REQ: COMPILE-009
// REQ: COMPILE-095
// REQ: CHAN-147
TEST(CompilePhysicalPlan, RejectsLocalEnumAcrossProcessGroups) {
  // Key setup: connect a local-enum output and input placed in different process groups.
  // Graph topology: source.out [process A] --> sink.in [process B]
  // Key assertion: lowering emits the local-enum cross-process diagnostic and creates no service.
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{7U};
  snapshot.vertices = {vertex(1U, "source"), vertex(2U, "sink")};
  snapshot.ports = {
      output_port(1U, 1U, "out", local_enum_payload()),
      input_port(2U, 2U, "in", local_enum_payload()),
  };
  snapshot.edges = {edge(1U, 1U, 2U)};
  const auto profiles = standard_profiles(snapshot.vertices.size());
  const auto placements = split_process_placements(snapshot);
  const auto partitions = default_execution_partitions(snapshot);

  const auto result = lower_physical_plan(snapshot,
                                          41U,
                                          profiles,
                                          runtime_package_capability_view(),
                                          {},
                                          {},
                                          {},
                                          placements,
                                          {},
                                          partitions);

  const auto* diagnostic =
      find_structural_diagnostic(StructuralDiagnosticId::kChannelLocalEnumCrossProcessUnsupported);
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(diagnostic->token, "CHANNEL_LOCAL_ENUM_CROSS_PROCESS_UNSUPPORTED");
  EXPECT_TRUE(has_error(result, "CHANNEL_LOCAL_ENUM_CROSS_PROCESS_UNSUPPORTED"));
  EXPECT_TRUE(result.process_local_channel_services.empty());
  EXPECT_TRUE(result.cross_process_channel_services.empty());
}

// A copied plan must not turn a process-local C++ enum into a transport identity.
// REQ: CHAN-147
TEST(CompilePhysicalPlan, RejectsLocalEnumInCopiedCrossProcessPlan) {
  // Key setup: lower a valid cross-process scalar edge, then replace its published identities
  // with same-width local enums in a copied plan.
  // Graph topology: source.out [process A] --> sink.in [process B]
  // Key assertion: cross-process validation and artifact finalization both reject the enum copy.
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{7U};
  snapshot.vertices = {vertex(1U, "source"), vertex(2U, "sink")};
  snapshot.ports = {output_port(1U, 1U, "out", scalar_payload()),
                    input_port(2U, 2U, "in", scalar_payload())};
  snapshot.edges = {edge(1U, 1U, 2U)};
  const auto profiles = standard_profiles(snapshot.vertices.size());
  const auto placements = split_process_placements(snapshot);
  const auto partitions = default_execution_partitions(snapshot);
  auto lowered = lower_physical_plan(snapshot,
                                     41U,
                                     profiles,
                                     runtime_package_capability_view(),
                                     {},
                                     {},
                                     {},
                                     placements,
                                     {},
                                     partitions,
                                     {});
  ASSERT_TRUE(lowered.errors.empty());
  auto state = published_physical_plan_state(std::move(lowered), 41U, placements);
  ASSERT_EQ(state.payload_edges.size(), 1U);
  ASSERT_EQ(state.channel_locality.size(), 1U);
  ASSERT_EQ(state.cross_process_channel_services.size(), 1U);
  // Mirror the compiler's partition-derived locality reconciliation for this physical fixture.
  state.payload_edges.front().locality = ChannelLocalityKind::kCrossProcess;
  state.channel_locality.front().requested = ChannelLocalityKind::kCrossProcess;
  state.channel_locality.front().accepted = ChannelLocalityKind::kCrossProcess;
  state.channel_locality.front().requires_runtime_attach = true;
  ASSERT_TRUE(valid_cross_process_channel_plan(state));
  state.payload_edges.front().payload_identity = local_enum_payload(4U);
  state.cross_process_channel_services.front().payload_identity = local_enum_payload(4U);
  EXPECT_FALSE(valid_cross_process_channel_plan(state));
  state.admitted = true;
  const auto finalized = finalize_plan_artifact(state);
  ASSERT_FALSE(finalized);
  EXPECT_EQ(finalized.error().message, "CROSS_PROCESS_CHANNEL_PLAN_INVALID");
}

// REQ: COMPILE-009
// REQ: COMPILE-095
// REQ: CHAN-115
// REQ: CHAN-116
TEST(CompilePhysicalPlan, PreservesDropOldestDepthOnHoloMQ) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{7U};
  snapshot.vertices = {vertex(1U, "source"), vertex(2U, "sink")};
  snapshot.ports = {
      output_port(1U, 1U, "out", scalar_payload()),
      input_port(2U, 2U, "in", scalar_payload()),
  };
  snapshot.edges = {edge(1U, 1U, 2U, 4U, QueuePolicy::kDropOldest)};
  const auto profiles = standard_profiles(snapshot.vertices.size());

  const auto result = lower_for_test(
      snapshot, 41U, profiles, {}, {}, {}, nullptr, {}, {}, kForceHoloMQProcessLocalInline);

  ASSERT_TRUE(result.errors.empty());
  ASSERT_EQ(result.process_local_channel_services.size(), 1U);
  EXPECT_EQ(result.process_local_channel_services.front().queue_policy, QueuePolicy::kDropOldest);
  EXPECT_EQ(result.process_local_channel_services.front().max_subscriber_queue_depth, 4U);
  EXPECT_EQ(result.process_local_channel_services.front().delivery_kind,
            ProcessLocalChannelDeliveryKind::kHoloMQ);
  EXPECT_FALSE(result.process_local_channel_services.front().event_wake);
  ASSERT_EQ(result.process_local_channel_publishers.size(), 1U);
  EXPECT_TRUE(result.process_local_channel_publishers.front().replacement_attribution_required);

  PayloadEdgePlanRecord unknown_policy = result.payload_edges.front();
  // This test deliberately constructs an invalid closed-enum value to verify conservative
  // fallback behavior.
  // NOLINTBEGIN(clang-analyzer-optin.core.EnumCastOutOfRange)
  unknown_policy.queue_policy =
      static_cast<QueuePolicy>(static_cast<std::uint32_t>(QueuePolicy::kDropOldest) + 1U);
  // NOLINTEND(clang-analyzer-optin.core.EnumCastOutOfRange)
  EXPECT_EQ(process_local_channel_delivery_kind(unknown_policy),
            ProcessLocalChannelDeliveryKind::kHoloMQ);
}

// REQ: CHAN-101
// REQ: RUNTIME-019
TEST(CompilePhysicalPlan, SuppressesHoloMQEventWakeOnlyForCertifiedSamePartitionService) {
  // Key setup: lower the same forced-HoloMQ scalar edge with colocated, lifecycle-capable,
  // private forced-wake, and split-partition plans.
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{7U};
  snapshot.vertices = {vertex(1U, "source"), vertex(2U, "sink")};
  snapshot.ports = {
      output_port(1U, 1U, "out", scalar_payload()),
      input_port(2U, 2U, "in", scalar_payload()),
  };
  snapshot.edges = {edge(1U, 1U, 2U)};
  const auto profiles = standard_profiles(snapshot.vertices.size());
  const auto same_partitions = default_execution_partitions(snapshot);
  auto colocated = lower_for_test(snapshot,
                                  41U,
                                  profiles,
                                  {},
                                  {},
                                  {},
                                  nullptr,
                                  {},
                                  same_partitions,
                                  kForceHoloMQProcessLocalInline);
  const std::array lifecycle_targets{LifecycleRestartTargetPlanRecord{
      .target = OperatorRef{.id = 2U, .plan_generation = 41U, .run_generation = 0U},
      .effective_profile = Profile::kStandard,
      .disposition = LifecycleRestartTargetDisposition::kAdmitted,
  }};
  const auto lifecycle_capable = lower_for_test(snapshot,
                                                41U,
                                                profiles,
                                                {},
                                                {},
                                                lifecycle_targets,
                                                nullptr,
                                                {},
                                                same_partitions,
                                                kForceHoloMQProcessLocalInline);
  auto forced_wake = lower_for_test(snapshot,
                                    41U,
                                    profiles,
                                    {},
                                    {},
                                    {},
                                    nullptr,
                                    {},
                                    same_partitions,
                                    kForceHoloMQProcessLocalInlineAndEventWake);
  const std::array split_partitions{
      ExecutionPartitionRecord{
          .id = ExecutionPartitionId{1U}, .members = {VertexId{1U}}, .execution_lane = 0U},
      ExecutionPartitionRecord{
          .id = ExecutionPartitionId{2U}, .members = {VertexId{2U}}, .execution_lane = 1U},
  };
  auto split = lower_for_test(snapshot,
                              41U,
                              profiles,
                              {},
                              {},
                              {},
                              nullptr,
                              {},
                              split_partitions,
                              LowerForTestOptions{
                                  .force_holomq_process_local_inline = true,
                                  .emit_admission = ProcessLocalEmitAdmissionScope::kNone,
                              });

  // Key assertion: exact colocated plans disable wake regardless of lifecycle capability, the
  // private A/B control retains it, and replay rejects a changed control or wake fact.
  ASSERT_TRUE(colocated.errors.empty());
  ASSERT_EQ(colocated.process_local_channel_services.size(), 1U);
  ASSERT_EQ(colocated.process_local_channel_subscribers.size(), 1U);
  EXPECT_FALSE(colocated.process_local_channel_services.front().event_wake);
  EXPECT_EQ(colocated.process_local_channel_services.front().event_wake_reason,
            ProcessLocalHoloMQEventWakeReason::kSuppressed);
  EXPECT_FALSE(colocated.process_local_channel_subscribers.front().event_wake);
  ExecutionPlanState colocated_state;
  colocated_state.plan_generation = 41U;
  colocated_state.execution_partitions = same_partitions;
  detail::publish_physical_plan_records(std::move(colocated), colocated_state);
  EXPECT_TRUE(valid_process_local_channel_plan(colocated_state));
  colocated_state.process_local_channel_services.front().event_wake = true;
  colocated_state.process_local_channel_subscribers.front().event_wake = true;
  EXPECT_FALSE(valid_process_local_channel_plan(colocated_state));

  ASSERT_TRUE(lifecycle_capable.errors.empty());
  ASSERT_EQ(lifecycle_capable.process_local_channel_services.size(), 1U);
  EXPECT_FALSE(lifecycle_capable.process_local_channel_services.front().event_wake);
  EXPECT_EQ(lifecycle_capable.process_local_channel_services.front().event_wake_reason,
            ProcessLocalHoloMQEventWakeReason::kSuppressed);

  ASSERT_TRUE(forced_wake.errors.empty());
  ASSERT_EQ(forced_wake.process_local_channel_services.size(), 1U);
  ASSERT_EQ(forced_wake.process_local_channel_subscribers.size(), 1U);
  EXPECT_TRUE(forced_wake.process_local_channel_services.front().event_wake);
  EXPECT_EQ(forced_wake.process_local_channel_services.front().event_wake_reason,
            ProcessLocalHoloMQEventWakeReason::kForcedControl);
  EXPECT_TRUE(forced_wake.process_local_channel_subscribers.front().event_wake);
  ExecutionPlanState forced_wake_state;
  forced_wake_state.plan_generation = 41U;
  forced_wake_state.execution_partitions = same_partitions;
  forced_wake_state.force_process_local_holomq_event_wake = true;
  detail::publish_physical_plan_records(std::move(forced_wake), forced_wake_state);
  EXPECT_TRUE(valid_process_local_channel_plan(forced_wake_state));
  forced_wake_state.force_process_local_holomq_event_wake = false;
  EXPECT_FALSE(valid_process_local_channel_plan(forced_wake_state));

  ASSERT_TRUE(split.errors.empty());
  ASSERT_EQ(split.process_local_channel_services.size(), 1U);
  ASSERT_EQ(split.process_local_channel_subscribers.size(), 1U);
  EXPECT_TRUE(split.process_local_channel_services.front().event_wake);
  EXPECT_EQ(split.process_local_channel_services.front().event_wake_reason,
            ProcessLocalHoloMQEventWakeReason::kSplitExecutionPartition);
  EXPECT_TRUE(split.process_local_channel_subscribers.front().event_wake);
  ExecutionPlanState split_state;
  split_state.plan_generation = 41U;
  split_state.force_process_local_pump_delivery = true;
  split_state.execution_partitions.assign(split_partitions.begin(), split_partitions.end());
  detail::publish_physical_plan_records(std::move(split), split_state);
  EXPECT_TRUE(valid_process_local_channel_plan(split_state));
  split_state.process_local_channel_services.front().event_wake = false;
  split_state.process_local_channel_subscribers.front().event_wake = false;
  EXPECT_FALSE(valid_process_local_channel_plan(split_state));
}

// A fixed-width compatibility payload remains transportable but is not one of the explicitly
// admitted typed direct-retention families.
// REQ: CHAN-096
// REQ: CHAN-116
TEST(CompilePhysicalPlan, KeepsUnsupportedDropOldestPayloadOnHoloMQ) {
  auto payload = scalar_payload();
  payload.payload_kind = PayloadKind::kCompatibilityPayload;
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{7U};
  snapshot.vertices = {vertex(1U, "source"), vertex(2U, "sink")};
  snapshot.ports = {
      output_port(1U, 1U, "out", payload),
      input_port(2U, 2U, "in", payload),
  };
  snapshot.edges = {edge(1U, 1U, 2U, 1U, QueuePolicy::kDropOldest)};
  const auto profiles = standard_profiles(snapshot.vertices.size());

  auto result = lower_for_test(snapshot, 41U, profiles);

  ASSERT_TRUE(result.errors.empty());
  ASSERT_EQ(result.process_local_channel_services.size(), 1U);
  EXPECT_EQ(result.process_local_channel_services.front().delivery_kind,
            ProcessLocalChannelDeliveryKind::kHoloMQ);
  EXPECT_TRUE(result.process_local_channel_services.front().event_wake);
  EXPECT_EQ(result.process_local_channel_services.front().event_wake_reason,
            ProcessLocalHoloMQEventWakeReason::kUnsupportedPayloadContract);
  ASSERT_EQ(result.process_local_channel_publishers.size(), 1U);
  EXPECT_TRUE(result.process_local_channel_publishers.front().replacement_attribution_required);

  auto state = published_physical_plan_state(std::move(result), 41U);
  EXPECT_TRUE(valid_process_local_channel_plan(state));
}

// Sibling publications from one callback must not be reordered by mixing HoloMQ and direct
// materialization before the same downstream activation.
//
//   source.candidate --[drop-oldest]---> sink.candidate
//                    `--[blocking]-----> decoy.candidate
//         .driver    --[blocking]-> sink.driver
//
// REQ: CHAN-100
TEST(CompilePhysicalPlan, KeepsMixedSiblingDeliveriesOnHoloMQ) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{7U};
  snapshot.vertices = {vertex(1U, "source"), vertex(2U, "sink"), vertex(3U, "decoy")};
  snapshot.ports = {
      output_port(1U, 1U, "candidate", scalar_payload()),
      output_port(2U, 1U, "driver", scalar_payload()),
      input_port(3U, 2U, "candidate", scalar_payload()),
      input_port(4U, 2U, "driver", scalar_payload()),
      input_port(5U, 3U, "candidate", scalar_payload()),
  };
  // Key setup: the fanout control keeps the mixed-policy candidate fanout on HoloMQ (compile()
  // would deliver it directly) while the blocking driver is individually direct-eligible. Pairwise
  // coherence keeps both source-to-sink edges on one delivery family.
  snapshot.edges = {
      edge(1U, 1U, 3U, 1U, QueuePolicy::kDropOldest),
      edge(2U, 2U, 4U),
      edge(3U, 1U, 5U),
  };
  const auto profiles = standard_profiles(snapshot.vertices.size());

  const auto result = lower_for_test(
      snapshot, 41U, profiles, {}, {}, {}, nullptr, {}, {}, kForceHoloMQProcessLocalFanout);

  EXPECT_TRUE(result.errors.empty());
  ASSERT_EQ(result.process_local_channel_services.size(), 3U);
  ASSERT_EQ(result.process_local_channel_subscribers.size(), 3U);
  // Key assertion: the complete producer/consumer pair uses one materialization path.
  EXPECT_TRUE(std::ranges::all_of(result.process_local_channel_services, [](const auto& service) {
    return service.delivery_kind == ProcessLocalChannelDeliveryKind::kHoloMQ && !service.event_wake;
  }));
  EXPECT_TRUE(std::ranges::all_of(result.process_local_channel_services, [](const auto& service) {
    return service.event_wake_reason == ProcessLocalHoloMQEventWakeReason::kSuppressed;
  }));
  EXPECT_TRUE(std::ranges::none_of(result.process_local_channel_subscribers,
                                   [](const auto& subscriber) { return subscriber.event_wake; }));
}

// A whole drop-oldest fanout family of inline or generated-schema payloads is direct-retained:
// one service with one subscriber per recipient and no HoloMQ wake.
//
//   source.out --[drop-oldest]--> left.in
//              `-[drop-oldest]--> right.in
//
// REQ: CHAN-160
TEST(CompilePhysicalPlan, DeliversADropOldestFanoutFamilyByDirectRetention) {
  for (const auto& payload : {scalar_payload(), schema_payload()}) {
    SCOPED_TRACE(payload.kind == PayloadIdentityKind::kSchema ? "schema" : "scalar");
    GraphSnapshotData snapshot;
    snapshot.generation = GraphGeneration{7U};
    snapshot.vertices = {vertex(1U, "source"), vertex(2U, "left"), vertex(3U, "right")};
    snapshot.ports = {
        output_port(1U, 1U, "out", payload),
        input_port(2U, 2U, "in", payload),
        input_port(3U, 3U, "in", payload),
    };
    if (payload.kind == PayloadIdentityKind::kSchema) {
      const RuntimePortTypeOps* const type_ops = runtime_port_type_ops<schema::StringT>();
      ASSERT_NE(type_ops, nullptr);
      for (auto& port : snapshot.ports) {
        port.serialized_payload_capacity_bytes = type_ops->wire_capacity;
      }
    }
    snapshot.edges = {
        edge(1U, 1U, 2U, 1U, QueuePolicy::kDropOldest),
        edge(2U, 1U, 3U, 1U, QueuePolicy::kDropOldest),
    };
    const auto profiles = standard_profiles(snapshot.vertices.size());
    const auto partitions = default_execution_partitions(snapshot);

    auto result = lower_for_test(snapshot, 41U, profiles, {}, {}, {}, nullptr, {}, partitions);

    ASSERT_TRUE(result.errors.empty());
    ASSERT_EQ(result.process_local_channel_services.size(), 1U);
    ASSERT_EQ(result.process_local_channel_subscribers.size(), 2U);
    const auto& service = result.process_local_channel_services.front();
    // Key assertion: the family is one direct service that stores into both recipients.
    EXPECT_EQ(service.delivery_kind, ProcessLocalChannelDeliveryKind::kDirectRetained);
    EXPECT_EQ(service.max_subscribers, 2U);
    EXPECT_FALSE(service.event_wake);
    EXPECT_FALSE(service.emit_admitted);
    EXPECT_TRUE(std::ranges::none_of(result.process_local_channel_subscribers,
                                     [](const auto& subscriber) { return subscriber.event_wake; }));
    ExecutionPlanState state;
    state.plan_generation = 41U;
    state.execution_partitions = partitions;
    detail::publish_physical_plan_records(std::move(result), state);
    EXPECT_TRUE(valid_process_local_channel_plan(state));
    // A family that no longer covers the source's whole fanout (here, a third arm the plan does
    // not list) cannot stay direct.
    for (auto& edge : state.payload_edges) {
      edge.fanout = 3U;
    }
    EXPECT_FALSE(valid_process_local_channel_plan(state));
  }
}

// A blocking or mixed-policy family of inline or generated-schema payloads is direct-retained too,
// with one direct service per queue policy and no HoloMQ wake. A blocking member with a restartable
// endpoint keeps its family on HoloMQ, including an endpoint that is restartable only as a member
// of another Operator's restart closure.
//
//   source.out --[blocking]------------------> left.in
//              `-[blocking or drop-oldest]---> right.in
//   restart-target (no ports; its restart closure may include right)
//
// REQ: CHAN-167
TEST(CompilePhysicalPlan, DeliversBlockingAndMixedPolicyFanoutFamiliesByDirectRetention) {
  const auto lower = [](const QueuePolicy second_policy,
                        std::span<const LifecycleRestartTargetPlanRecord>
                            restart_targets,
                        std::span<const LifecycleRestartClosurePlanRecord>
                            restart_closures) {
    GraphSnapshotData snapshot;
    snapshot.generation = GraphGeneration{7U};
    snapshot.vertices = {vertex(1U, "source"),
                         vertex(2U, "left"),
                         vertex(3U, "right"),
                         vertex(4U, "restart-target")};
    snapshot.ports = {
        output_port(1U, 1U, "out", scalar_payload()),
        input_port(2U, 2U, "in", scalar_payload()),
        input_port(3U, 3U, "in", scalar_payload()),
    };
    snapshot.edges = {edge(1U, 1U, 2U, 1U), edge(2U, 1U, 3U, 1U, second_policy)};
    const auto profiles = standard_profiles(snapshot.vertices.size());
    const auto partitions = default_execution_partitions(snapshot);
    LowerForTestOptions options;
    options.lifecycle_closures = restart_closures;
    auto result = lower_for_test(
        snapshot, 41U, profiles, {}, {}, restart_targets, nullptr, {}, partitions, options);
    ExecutionPlanState state;
    state.plan_generation = 41U;
    state.execution_partitions = partitions;
    state.lifecycle_restart_targets.assign(restart_targets.begin(), restart_targets.end());
    state.lifecycle_restart_closures.assign(restart_closures.begin(), restart_closures.end());
    EXPECT_TRUE(result.errors.empty());
    detail::publish_physical_plan_records(std::move(result), state);
    return state;
  };
  for (const QueuePolicy second_policy : {QueuePolicy::kBlocking, QueuePolicy::kDropOldest}) {
    SCOPED_TRACE(second_policy == QueuePolicy::kBlocking ? "blocking" : "mixed policy");
    const auto state = lower(second_policy, {}, {});
    ASSERT_EQ(state.process_local_channel_subscribers.size(), 2U);
    // Key assertion: every service of the family is direct, one per queue policy, without a wake.
    ASSERT_EQ(state.process_local_channel_services.size(),
              second_policy == QueuePolicy::kBlocking ? 1U : 2U);
    std::uint32_t subscribers = 0U;
    for (const auto& service : state.process_local_channel_services) {
      EXPECT_EQ(service.delivery_kind, ProcessLocalChannelDeliveryKind::kDirectRetained);
      EXPECT_FALSE(service.event_wake);
      subscribers += service.max_subscribers;
    }
    EXPECT_EQ(subscribers, 2U);
    EXPECT_TRUE(std::ranges::none_of(state.process_local_channel_subscribers,
                                     [](const auto& subscriber) { return subscriber.event_wake; }));
    EXPECT_TRUE(valid_process_local_channel_plan(state));
  }
  // Key assertion: a restartable recipient keeps the blocking family on HoloMQ.
  const std::array restart_targets{LifecycleRestartTargetPlanRecord{
      .target = OperatorRef{.id = 3U, .plan_generation = 41U},
      .self_allowed = true,
      .disposition = LifecycleRestartTargetDisposition::kAdmitted,
  }};
  const auto restartable = lower(QueuePolicy::kBlocking, restart_targets, {});
  ASSERT_FALSE(restartable.process_local_channel_services.empty());
  EXPECT_TRUE(
      std::ranges::all_of(restartable.process_local_channel_services, [](const auto& service) {
        return service.delivery_kind == ProcessLocalChannelDeliveryKind::kHoloMQ;
      }));
  EXPECT_TRUE(valid_process_local_channel_plan(restartable));
  // Key assertion: so does a recipient restarted only with the closure of another admitted target,
  // and plan validation applies the same rule.
  const std::array closure_targets{LifecycleRestartTargetPlanRecord{
      .target = OperatorRef{.id = 4U, .plan_generation = 41U},
      .self_allowed = true,
      .disposition = LifecycleRestartTargetDisposition::kAdmitted,
  }};
  const std::array closures{LifecycleRestartClosurePlanRecord{
      .target = OperatorRef{.id = 4U, .plan_generation = 41U},
      .closure = {OperatorRef{.id = 4U, .plan_generation = 41U},
                  OperatorRef{.id = 3U, .plan_generation = 41U}},
  }};
  const auto closure_member = lower(QueuePolicy::kBlocking, closure_targets, closures);
  ASSERT_FALSE(closure_member.process_local_channel_services.empty());
  EXPECT_TRUE(
      std::ranges::all_of(closure_member.process_local_channel_services, [](const auto& service) {
        return service.delivery_kind == ProcessLocalChannelDeliveryKind::kHoloMQ;
      }));
  EXPECT_TRUE(valid_process_local_channel_plan(closure_member));
}

// A family that the fanout seam or pump delivery pins stays on HoloMQ.
// REQ: CHAN-160
TEST(CompilePhysicalPlan, KeepsIneligibleFanoutFamiliesOnHoloMQ) {
  const auto lower = [](const QueuePolicy second_policy, const LowerForTestOptions options) {
    GraphSnapshotData snapshot;
    snapshot.generation = GraphGeneration{7U};
    snapshot.vertices = {vertex(1U, "source"), vertex(2U, "left"), vertex(3U, "right")};
    snapshot.ports = {
        output_port(1U, 1U, "out", scalar_payload()),
        input_port(2U, 2U, "in", scalar_payload()),
        input_port(3U, 3U, "in", scalar_payload()),
    };
    snapshot.edges = {
        edge(1U, 1U, 2U, 1U, QueuePolicy::kDropOldest),
        edge(2U, 1U, 3U, 1U, second_policy),
    };
    const auto profiles = standard_profiles(snapshot.vertices.size());
    const auto partitions = default_execution_partitions(snapshot);
    return lower_for_test(snapshot, 41U, profiles, {}, {}, {}, nullptr, {}, partitions, options);
  };
  const auto all_holomq = [](const PhysicalPlanLoweringResult& result) {
    return !result.process_local_channel_services.empty() &&
           std::ranges::all_of(result.process_local_channel_services, [](const auto& service) {
             return service.delivery_kind == ProcessLocalChannelDeliveryKind::kHoloMQ;
           });
  };

  const auto pinned = lower(QueuePolicy::kBlocking, kForceHoloMQProcessLocalFanout);
  ASSERT_TRUE(pinned.errors.empty());
  EXPECT_TRUE(all_holomq(pinned));
  const auto pumped = lower(QueuePolicy::kBlocking, kPumpDelivery);
  ASSERT_TRUE(pumped.errors.empty());
  EXPECT_TRUE(all_holomq(pumped));
}

// Forcing one member of a direct family onto HoloMQ forces the whole family, and the pairs it then
// adds force further direct edges: lowering iterates to a fixed point.
//
//   source.candidate --[drop-oldest]--> sink.candidate     (direct family, until escalated)
//                    `-[drop-oldest]--> decoy.candidate
//         .blocked   --[blocking]-----> sink.blocked       (blocking Tensor fanout: HoloMQ)
//                    `-[blocking]-----> other.blocked
//         .extra     --[blocking]-----> decoy.extra        (direct-eligible fanout one)
//
// REQ: CHAN-100
// REQ: CHAN-160
TEST(CompilePhysicalPlan, EscalatesADirectFanoutFamilyThroughItsOperatorPairs) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{7U};
  snapshot.vertices = {
      vertex(1U, "source"), vertex(2U, "sink"), vertex(3U, "decoy"), vertex(4U, "other")};
  snapshot.ports = {
      output_port(1U, 1U, "candidate", scalar_payload()),
      output_port(2U, 1U, "blocked", tensor_payload()),
      output_port(3U, 1U, "extra", scalar_payload()),
      input_port(4U, 2U, "candidate", scalar_payload()),
      input_port(5U, 2U, "blocked", tensor_payload()),
      input_port(6U, 3U, "candidate", scalar_payload()),
      input_port(7U, 3U, "extra", scalar_payload()),
      input_port(8U, 4U, "blocked", tensor_payload()),
  };
  // A blocking family is direct only for inline and generated-schema payloads (CHAN-167), so this
  // raw Tensor family stays on HoloMQ and starts the escalation.
  snapshot.ports[1].setup.tensor_layout = TensorRepresentation{
      .memory_kind = MemoryKind::kHost,
      .dtype = DLDataType{kDLUInt, 8U, 1U},
      .rank = 1U,
  };
  snapshot.ports[1].setup.tensor_allocation =
      TensorPoolBounds{.max_rank = 1U, .max_byte_span = 64U};
  snapshot.edges = {
      edge(1U, 1U, 4U, 1U, QueuePolicy::kDropOldest),
      edge(2U, 1U, 6U, 1U, QueuePolicy::kDropOldest),
      edge(3U, 2U, 5U),
      edge(4U, 2U, 8U),
      edge(5U, 3U, 7U),
  };
  const auto profiles = standard_profiles(snapshot.vertices.size());

  const auto result = lower_for_test(snapshot, 41U, profiles);

  ASSERT_TRUE(result.errors.empty());
  ASSERT_EQ(result.process_local_channel_services.size(), 3U);
  // Key assertion: source->sink forces the candidate family, whose decoy arm then forces
  // source.extra -> decoy.extra, although that edge alone is direct-eligible.
  EXPECT_TRUE(std::ranges::all_of(result.process_local_channel_services, [](const auto& service) {
    return service.delivery_kind == ProcessLocalChannelDeliveryKind::kHoloMQ;
  }));
}

// A family with cross-process or exported members keeps its process-local members direct when every
// remote member is an accepted, unprojected drop-oldest inline, generated-schema, raw Tensor, or
// descriptor-backed schema leg. A blocking or projected remote member, a family with no
// process-local member, or a family missing an edge keeps the whole family on HoloMQ.
//
// REQ: CHAN-165
// REQ: CHAN-168
// REQ: CHAN-169
TEST(CompilePhysicalPlan, ClassifiesMixedLocalityFanoutFamilies) {
  struct Member {
    bool remote{};
    QueuePolicy policy{QueuePolicy::kDropOldest};
    bool projected{};
  };
  const auto qualifies = [](const PayloadIdentityRecord& payload,
                            const std::vector<Member>& members,
                            const std::uint32_t fanout) {
    std::vector<std::pair<PayloadEdgePlanRecord, bool>> edges;
    for (std::size_t index = 0U; index < members.size(); ++index) {
      PayloadEdgePlanRecord edge{
          .edge = EdgeId{index + 1U},
          .source = PortId{1U},
          .target = PortId{index + 2U},
          .payload_identity = payload,
          .queue_policy = members[index].policy,
          .queue_depth = 1U,
          .fanout = fanout,
          .locality = ChannelLocalityKind::kProcessLocal,
          .disposition = PlanDisposition::kAccepted,
      };
      if (members[index].projected) {
        edge.tensor_projection = TensorProjectionRecord{};
      }
      edges.emplace_back(std::move(edge), members[index].remote);
    }
    return process_local_direct_fanout_sources(
               edges,
               [](const auto& item) -> const PayloadEdgePlanRecord& { return item.first; },
               [](const auto& item) { return item.second; })
        .contains(1U);
  };
  const Member local{};
  const Member remote{.remote = true};

  // Key assertion: only complete families with a local member and drop-oldest inline, schema, raw
  // Tensor, or descriptor-backed schema remote legs qualify.
  EXPECT_TRUE(qualifies(scalar_payload(), {local, local}, 2U));
  EXPECT_TRUE(qualifies(scalar_payload(), {local, remote}, 2U));
  EXPECT_TRUE(qualifies(schema_payload(), {local, local, remote}, 3U));
  EXPECT_TRUE(qualifies(scalar_payload(), {local, remote, remote}, 3U));
  EXPECT_FALSE(qualifies(scalar_payload(), {remote, remote}, 2U));
  EXPECT_FALSE(qualifies(scalar_payload(), {local, remote}, 3U));
  EXPECT_FALSE(qualifies(
      scalar_payload(), {local, Member{.remote = true, .policy = QueuePolicy::kBlocking}}, 2U));
  EXPECT_FALSE(qualifies(scalar_payload(), {local, Member{.remote = true, .projected = true}}, 2U));
  EXPECT_TRUE(qualifies(tensor_payload(), {local, local}, 2U));
  EXPECT_TRUE(qualifies(tensor_payload(), {local, remote}, 2U));
  EXPECT_TRUE(qualifies(tensor_payload(), {local, local, remote, remote}, 4U));
  EXPECT_FALSE(qualifies(tensor_payload(), {remote, remote}, 2U));
  EXPECT_FALSE(qualifies(tensor_payload(), {local, Member{.remote = true, .projected = true}}, 2U));
  EXPECT_TRUE(qualifies(image_payload(), {local, remote}, 2U));
  EXPECT_FALSE(qualifies(
      image_payload(), {local, Member{.remote = true, .policy = QueuePolicy::kBlocking}}, 2U));
}

// REQ: CHAN-101
// REQ: RUNTIME-019
TEST(CompilePhysicalPlan, ReportsRepresentativeEventWakeApplicabilityTopologies) {
  struct Census {
    std::size_t services{};
    std::size_t edges{};
    std::vector<ProcessLocalHoloMQEventWakeReason> reasons;
  };
  const auto collect = [](const GraphSnapshotData& snapshot) {
    const auto profiles = standard_profiles(snapshot.vertices.size());
    const auto partitions = default_execution_partitions(snapshot);
    auto physical = lower_for_test(snapshot,
                                   41U,
                                   profiles,
                                   {},
                                   {},
                                   {},
                                   nullptr,
                                   {},
                                   partitions,
                                   kForceHoloMQProcessLocalInline);
    EXPECT_TRUE(physical.errors.empty());
    ExecutionPlanState state;
    state.plan_generation = 41U;
    state.execution_partitions = partitions;
    detail::publish_physical_plan_records(std::move(physical), state);
    EXPECT_TRUE(valid_process_local_channel_plan(state));
    const auto applicability = inspect_process_local_holomq_event_wake_applicability(state);
    Census census{.services = applicability.size()};
    for (const auto& service : applicability) {
      census.edges += service.payload_edge_count;
      census.reasons.push_back(service.reason);
    }
    return census;
  };

  // Key setup: collect cold reason rows for multi-output, multi-input, chain, and owner-loop
  // process-local HoloMQ controls.
  GraphSnapshotData multi_output;
  multi_output.generation = GraphGeneration{7U};
  multi_output.vertices = {vertex(1U, "source"), vertex(2U, "left"), vertex(3U, "right")};
  multi_output.ports = {
      output_port(1U, 1U, "left", scalar_payload()),
      output_port(2U, 1U, "right", scalar_payload()),
      input_port(3U, 2U, "in", scalar_payload()),
      input_port(4U, 3U, "in", scalar_payload()),
  };
  multi_output.edges = {edge(1U, 1U, 3U), edge(2U, 2U, 4U)};
  const Census multi_output_census = collect(std::move(multi_output));

  GraphSnapshotData multi_input;
  multi_input.generation = GraphGeneration{7U};
  multi_input.vertices = {vertex(1U, "left"), vertex(2U, "right"), vertex(3U, "sink")};
  multi_input.ports = {
      output_port(1U, 1U, "out", scalar_payload()),
      output_port(2U, 2U, "out", scalar_payload()),
      input_port(3U, 3U, "left", scalar_payload()),
      input_port(4U, 3U, "right", scalar_payload()),
  };
  multi_input.edges = {edge(1U, 1U, 3U), edge(2U, 2U, 4U)};
  const Census multi_input_census = collect(std::move(multi_input));

  GraphSnapshotData chain;
  chain.generation = GraphGeneration{7U};
  chain.vertices = {vertex(1U, "source"), vertex(2U, "middle"), vertex(3U, "sink")};
  chain.ports = {
      output_port(1U, 1U, "out", scalar_payload()),
      input_port(2U, 2U, "in", scalar_payload()),
      output_port(3U, 2U, "out", scalar_payload()),
      input_port(4U, 3U, "in", scalar_payload()),
  };
  chain.edges = {edge(1U, 1U, 2U), edge(2U, 3U, 4U)};
  const Census chain_census = collect(std::move(chain));

  GraphSnapshotData owner_loop;
  owner_loop.generation = GraphGeneration{7U};
  owner_loop.vertices = {vertex(1U, "loop")};
  owner_loop.ports = {
      output_port(1U, 1U, "out", scalar_payload()),
      input_port(2U, 1U, "in", scalar_payload()),
  };
  owner_loop.edges = {edge(1U, 1U, 2U)};
  const Census owner_loop_census = collect(std::move(owner_loop));

  // Key assertion: multi-output, multi-input, and chain services suppress wakes, while the
  // same-owner edge retains its distinct reason.
  EXPECT_EQ(multi_output_census.services, 2U);
  EXPECT_EQ(multi_output_census.edges, 2U);
  EXPECT_TRUE(std::ranges::all_of(multi_output_census.reasons, [](const auto reason) {
    return reason == ProcessLocalHoloMQEventWakeReason::kSuppressed;
  }));
  EXPECT_EQ(multi_input_census.services, 2U);
  EXPECT_EQ(multi_input_census.edges, 2U);
  EXPECT_TRUE(std::ranges::all_of(multi_input_census.reasons, [](const auto reason) {
    return reason == ProcessLocalHoloMQEventWakeReason::kSuppressed;
  }));
  EXPECT_EQ(chain_census.services, 2U);
  EXPECT_EQ(chain_census.edges, 2U);
  EXPECT_TRUE(std::ranges::all_of(chain_census.reasons, [](const auto reason) {
    return reason == ProcessLocalHoloMQEventWakeReason::kSuppressed;
  }));
  ASSERT_EQ(owner_loop_census.reasons.size(), 1U);
  EXPECT_EQ(owner_loop_census.services, 1U);
  EXPECT_EQ(owner_loop_census.edges, 1U);
  EXPECT_EQ(owner_loop_census.reasons.front(), ProcessLocalHoloMQEventWakeReason::kOwnerLoop);
}

// An ordinary generated owning schema keeps its bounded encoder, versioned envelope, verifier,
// and materializer while avoiding the intermediate transport queue for one local edge.
// REQ: CHAN-097
// REQ: CHAN-117
// REQ: CHAN-128
TEST(CompilePhysicalPlan, SelectsDirectRetainedGeneratedSchemaWithHoloMQFallback) {
  const auto payload = schema_payload();
  const RuntimePortTypeOps* const type_ops = runtime_port_type_ops<schema::StringT>();
  ASSERT_NE(type_ops, nullptr);
  ASSERT_EQ(type_ops->codec_family, RuntimePayloadCodecFamily::kGeneratedSchema);

  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{7U};
  snapshot.vertices = {vertex(1U, "source"), vertex(2U, "sink")};
  snapshot.ports = {
      output_port(1U, 1U, "out", payload),
      input_port(2U, 2U, "in", payload),
  };
  snapshot.ports[0].serialized_payload_capacity_bytes = type_ops->wire_capacity;
  snapshot.ports[1].serialized_payload_capacity_bytes = type_ops->wire_capacity;
  snapshot.edges = {edge(1U, 1U, 2U)};
  const auto profiles = standard_profiles(snapshot.vertices.size());

  const auto result = lower_for_test(snapshot, 42U, profiles);

  ASSERT_TRUE(result.errors.empty());
  ASSERT_EQ(result.process_local_channel_services.size(), 1U);
  const auto& service = result.process_local_channel_services.front();
  EXPECT_EQ(service.payload_size_bytes, type_ops->wire_capacity + kSchemaEnvelopeHeaderSize);
  EXPECT_EQ(service.delivery_kind, ProcessLocalChannelDeliveryKind::kDirectRetained);
  EXPECT_FALSE(service.event_wake);
  ASSERT_EQ(result.process_local_channel_subscribers.size(), 1U);
  EXPECT_FALSE(result.process_local_channel_subscribers.front().event_wake);

  snapshot.edges.front().add_flow_options.queue_policy = QueuePolicy::kDropOldest;
  snapshot.edges.front().add_flow_options.queue_depth = 4U;
  snapshot.edges.front().add_flow_options.max_inflight_samples = 4U;
  snapshot.ports[1].setup.queue_depth = 4U;
  const auto drop_oldest = lower_for_test(snapshot, 42U, profiles);
  ASSERT_TRUE(drop_oldest.errors.empty());
  ASSERT_EQ(drop_oldest.process_local_channel_services.size(), 1U);
  EXPECT_EQ(drop_oldest.process_local_channel_services.front().delivery_kind,
            ProcessLocalChannelDeliveryKind::kDirectRetained);
  EXPECT_FALSE(drop_oldest.process_local_channel_services.front().event_wake);
  ASSERT_EQ(drop_oldest.process_local_channel_publishers.size(), 1U);
  EXPECT_TRUE(
      drop_oldest.process_local_channel_publishers.front().replacement_attribution_required);

  const auto fallback = lower_for_test(
      snapshot, 42U, profiles, {}, {}, {}, nullptr, {}, {}, kForceHoloMQProcessLocalInline);
  ASSERT_TRUE(fallback.errors.empty());
  ASSERT_EQ(fallback.process_local_channel_services.size(), 1U);
  EXPECT_EQ(fallback.process_local_channel_services.front().delivery_kind,
            ProcessLocalChannelDeliveryKind::kHoloMQ);
  EXPECT_FALSE(fallback.process_local_channel_services.front().event_wake);
}

// A HoloMQ service is sized exactly for the writer's frame. A reader whose own limit is larger
// still attaches, because the service capacity, not the reader bound, sizes every endpoint.
// REQ: SCHEMA-007
TEST(CompilePhysicalPlan, SizesHoloMQSchemaServiceForWriterFrameAndAdmitsLargerReader) {
  constexpr std::uint64_t kWriterMaximum = 1024U;
  constexpr std::uint64_t kReaderLimit = 1500U;
  constexpr std::uint64_t kRequiredFrame = kWriterMaximum + kSchemaEnvelopeHeaderSize;
  static_assert(kRequiredFrame == 1064U);

  const auto payload = schema_payload();
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{7U};
  snapshot.vertices = {vertex(1U, "source"), vertex(2U, "sink")};
  snapshot.ports = {
      output_port(1U, 1U, "out", payload),
      input_port(2U, 2U, "in", payload),
  };
  snapshot.ports[0].serialized_payload_capacity_bytes = kWriterMaximum;
  snapshot.ports[1].serialized_payload_capacity_bytes = kReaderLimit;
  snapshot.edges = {edge(1U, 1U, 2U)};

  auto result = lower_for_test(snapshot,
                               42U,
                               standard_profiles(snapshot.vertices.size()),
                               {},
                               {},
                               {},
                               nullptr,
                               {},
                               {},
                               kForceHoloMQProcessLocalInline);
  ASSERT_TRUE(result.errors.empty());
  ASSERT_EQ(result.process_local_channel_services.size(), 1U);
  const auto& service = result.process_local_channel_services.front();
  EXPECT_EQ(service.payload_size_bytes, kRequiredFrame);
  auto state = published_physical_plan_state(std::move(result), 42U);
  state.execution_partitions = default_execution_partitions(snapshot);
  EXPECT_TRUE(valid_process_local_channel_plan(state));
  ++state.process_local_channel_services.front().payload_size_bytes;
  EXPECT_FALSE(valid_process_local_channel_plan(state));

  snapshot.ports[1].serialized_payload_capacity_bytes = kWriterMaximum - 1U;
  const auto rejected = lower_for_test(snapshot,
                                       42U,
                                       standard_profiles(snapshot.vertices.size()),
                                       {},
                                       {},
                                       {},
                                       nullptr,
                                       {},
                                       {},
                                       kForceHoloMQProcessLocalInline);
  EXPECT_FALSE(rejected.errors.empty());
}

// A complete generated-schema fanout uses the same synchronous activation cut as scalar fanout;
// each subscriber still verifies and materializes its own owning value from the bounded envelope.
// REQ: CHAN-101
// REQ: CHAN-117
// REQ: RUNTIME-019
TEST(CompilePhysicalPlan, SuppressesGeneratedSchemaFanoutEventWakeServiceWide) {
  const auto payload = schema_payload();
  const RuntimePortTypeOps* const type_ops = runtime_port_type_ops<schema::StringT>();
  ASSERT_NE(type_ops, nullptr);

  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{7U};
  snapshot.vertices = {vertex(1U, "source"), vertex(2U, "left"), vertex(3U, "right")};
  snapshot.ports = {
      output_port(1U, 1U, "out", payload),
      input_port(2U, 2U, "in", payload),
      input_port(3U, 3U, "in", payload),
  };
  for (auto& port : snapshot.ports) {
    port.serialized_payload_capacity_bytes = type_ops->wire_capacity;
  }
  snapshot.edges = {edge(1U, 1U, 2U), edge(2U, 1U, 3U)};
  const auto profiles = standard_profiles(snapshot.vertices.size());
  const auto partitions = default_execution_partitions(snapshot);

  // Key setup: lower one complete two-target bounded generated-schema service on a shared
  // execution partition.
  auto result = lower_for_test(
      snapshot, 42U, profiles, {}, {}, {}, nullptr, {}, partitions, kForceHoloMQProcessLocalFanout);

  // Key assertion: HoloMQ wake suppression is uniform across the service and replay rejects a
  // changed subscriber wake fact.
  ASSERT_TRUE(result.errors.empty());
  ASSERT_EQ(result.process_local_channel_services.size(), 1U);
  ASSERT_EQ(result.process_local_channel_subscribers.size(), 2U);
  EXPECT_EQ(result.process_local_channel_services.front().delivery_kind,
            ProcessLocalChannelDeliveryKind::kHoloMQ);
  EXPECT_EQ(result.process_local_channel_services.front().payload_size_bytes,
            type_ops->wire_capacity + kSchemaEnvelopeHeaderSize);
  EXPECT_FALSE(result.process_local_channel_services.front().event_wake);
  EXPECT_TRUE(std::ranges::none_of(result.process_local_channel_subscribers,
                                   [](const auto& subscriber) { return subscriber.event_wake; }));
  ExecutionPlanState state;
  state.plan_generation = 42U;
  state.execution_partitions = partitions;
  detail::publish_physical_plan_records(std::move(result), state);
  EXPECT_TRUE(valid_process_local_channel_plan(state));
  state.process_local_channel_subscribers.back().event_wake = true;
  EXPECT_FALSE(valid_process_local_channel_plan(state));
}

// Cross-process fanout identity uses the minimum accepted edge in each compatible family, not
// the snapshot's incidental edge order.
//
//   source [process A] --edge 1--> left  [process B]
//                      --edge 2--> right [process C]
//
// REQ: CHAN-085
TEST(CompilePhysicalPlan, CrossProcessFanoutUsesMinimumEdgeAnchorIndependentOfSnapshotOrder) {
  const auto make_snapshot = [](bool minimum_first) {
    GraphSnapshotData snapshot;
    snapshot.generation = GraphGeneration{7U};
    snapshot.vertices = {vertex(1U, "source"), vertex(2U, "left"), vertex(3U, "right")};
    snapshot.ports = {
        output_port(1U, 1U, "out", scalar_payload()),
        input_port(2U, 2U, "in", scalar_payload()),
        input_port(3U, 3U, "in", scalar_payload()),
    };
    const EdgeRecord minimum = edge(1U, 1U, 2U);
    const EdgeRecord other = edge(2U, 1U, 3U);
    snapshot.edges = minimum_first ? std::vector{minimum, other} : std::vector{other, minimum};
    return snapshot;
  };
  const auto lower = [](const GraphSnapshotData& snapshot) {
    const auto profiles = standard_profiles(snapshot.vertices.size());
    const auto placements = split_process_placements(snapshot);
    const auto partitions = default_execution_partitions(snapshot);
    return lower_physical_plan(snapshot,
                               42U,
                               profiles,
                               runtime_package_capability_view(),
                               {},
                               {},
                               {},
                               placements,
                               {},
                               partitions,
                               {});
  };

  const auto reordered = lower(make_snapshot(false));
  const auto ordered = lower(make_snapshot(true));
  ASSERT_TRUE(reordered.errors.empty())
      << (reordered.errors.empty() ? std::string{} : reordered.errors.front().token);
  ASSERT_TRUE(ordered.errors.empty())
      << (ordered.errors.empty() ? std::string{} : ordered.errors.front().token);
  ASSERT_EQ(reordered.cross_process_channel_services.size(), 1U);
  ASSERT_EQ(reordered.cross_process_channel_publishers.size(), 1U);
  ASSERT_EQ(reordered.cross_process_channel_subscribers.size(), 2U);
  ASSERT_EQ(ordered.cross_process_channel_services.size(), 1U);

  const auto& service = reordered.cross_process_channel_services.front();
  EXPECT_EQ(service.edge, EdgeId{1U});
  EXPECT_EQ(service.target, PortId{2U});
  EXPECT_EQ(service.target_process_group, ProcessGroupId{2U});
  EXPECT_NE(service.service_id.find(".anchor.1"), std::string::npos);
  EXPECT_EQ(reordered.cross_process_channel_publishers.front().edge, EdgeId{1U});
  EXPECT_EQ(service.service_id, ordered.cross_process_channel_services.front().service_id);
  EXPECT_EQ(service.edge, ordered.cross_process_channel_services.front().edge);
  EXPECT_EQ(service.target, ordered.cross_process_channel_services.front().target);
}
// REQ: COMPILE-040
TEST(CompilePhysicalPlan, RejectsMalformedPortEdgeAndQueueContractsWithStableDiagnostics) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{1U};
  snapshot.vertices = {vertex(1U, "source"), vertex(2U, "sink")};
  snapshot.ports = {
      output_port(1U, 1U, "out", scalar_payload(), 2U),
      input_port(2U, 2U, "in", scalar_payload(), 3U),
  };
  snapshot.edges = {edge(1U, 1U, 2U, 3U)};
  const auto profiles = standard_profiles(snapshot.vertices.size());

  auto malformed_port = snapshot;
  malformed_port.ports[1].setup.queue_depth = 0U;
  EXPECT_TRUE(
      has_error(lower_for_test(malformed_port, 43U, profiles), "PHYSICAL_PORT_CONTRACT_INVALID"));

  auto malformed_edge = snapshot;
  malformed_edge.edges[0].source = GraphPortId{2U};
  malformed_edge.edges[0].target = GraphPortId{1U};
  EXPECT_TRUE(
      has_error(lower_for_test(malformed_edge, 43U, profiles), "PHYSICAL_EDGE_CONTRACT_INVALID"));

  auto malformed_queue = snapshot;
  malformed_queue.edges[0].add_flow_options.max_inflight_samples = 1U;
  EXPECT_TRUE(
      has_error(lower_for_test(malformed_queue, 43U, profiles), "PHYSICAL_QUEUE_CAPACITY_INVALID"));
}

// max_inflight_samples is metadata on the edge that Runtime uses for flow control, but the
// compiler must not grow the loan pool or tensor allocation pool to match it.
//
//   scalar-source ---> scalar-sink
//   tensor-source ---> tensor-sink
//
// REQ: COMPILE-040
TEST(CompilePhysicalPlan, MaxInflightEvidenceDoesNotProvisionAdditionalPayloadSlots) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{1U};
  snapshot.vertices = {vertex(1U, "scalar-source"),
                       vertex(2U, "tensor-source"),
                       vertex(3U, "scalar-sink"),
                       vertex(4U, "tensor-sink")};
  snapshot.ports = {
      output_port(1U, 1U, "scalar", scalar_payload(), 2U),
      output_port(2U, 2U, "tensor", tensor_payload(), 2U),
      input_port(3U, 3U, "scalar", scalar_payload(), 3U),
      input_port(4U, 4U, "tensor", tensor_payload(), 3U),
  };
  snapshot.ports[1].setup.tensor_layout = TensorRepresentation{
      .memory_kind = MemoryKind::kHost,
      .dtype = DLDataType{kDLFloat, 32U, 1U},
      .rank = 1U,
  };
  snapshot.ports[1].setup.tensor_allocation =
      TensorPoolBounds{.max_rank = 1U, .max_byte_span = 1024U};
  snapshot.edges = {edge(1U, 1U, 3U, 3U), edge(2U, 2U, 4U, 3U)};
  const auto profiles = standard_profiles(snapshot.vertices.size());

  const auto baseline = lower_for_test(snapshot, 44U, profiles);
  EXPECT_TRUE(baseline.errors.empty());
  ASSERT_EQ(baseline.payload_edges.size(), 2U);
  ASSERT_EQ(baseline.generic_output_loans.size(), 1U);
  ASSERT_EQ(baseline.tensor_allocations.size(), 1U);

  // Key setup: inflate max_inflight_samples well above queue depth on both edges.
  snapshot.edges[0].add_flow_options.max_inflight_samples = 9U;
  snapshot.edges[1].add_flow_options.max_inflight_samples = 9U;
  const auto enlarged = lower_for_test(snapshot, 44U, profiles);
  EXPECT_TRUE(enlarged.errors.empty());
  ASSERT_EQ(enlarged.payload_edges.size(), 2U);
  ASSERT_EQ(enlarged.generic_output_loans.size(), 1U);
  ASSERT_EQ(enlarged.tensor_allocations.size(), 1U);

  EXPECT_EQ(baseline.payload_edges[0].max_inflight_samples, 3U);
  EXPECT_EQ(baseline.payload_edges[1].max_inflight_samples, 3U);
  EXPECT_EQ(enlarged.payload_edges[0].max_inflight_samples, 9U);
  EXPECT_EQ(enlarged.payload_edges[1].max_inflight_samples, 9U);
  // Key assertion: pool slot counts and byte capacities are unchanged despite enlarged inflight.
  EXPECT_EQ(enlarged.generic_output_loans[0].slot_count,
            baseline.generic_output_loans[0].slot_count);
  EXPECT_EQ(enlarged.generic_output_loans[0].aggregate_byte_capacity,
            baseline.generic_output_loans[0].aggregate_byte_capacity);
  EXPECT_EQ(enlarged.tensor_allocations[0].slot_count, baseline.tensor_allocations[0].slot_count);
  EXPECT_EQ(enlarged.tensor_allocations[0].aggregate_byte_capacity,
            baseline.tensor_allocations[0].aggregate_byte_capacity);
}

// REQ: COMPILE-047
TEST(CompilePhysicalPlan, OmitsGenericLoanWhenQualifiedBehaviorEvidenceIsAbsent) {
  auto pod = scalar_payload();
  pod.kind = PayloadIdentityKind::kPod;
  pod.payload_kind = PayloadKind::kInlineTyped;
  pod.scalar.reset();
  pod.pod = PodIdentityRecord{
      .canonical_name = "example.fixed-layout",
      .version = 1U,
      .object_size = 4U,
      .object_alignment = 4U,
      .fields = {{.name = "value",
                  .byte_offset = 0U,
                  .scalar_kind = PodScalarKind::kUInt32,
                  .element_count = 1U}},
  };
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{1U};
  snapshot.vertices = {vertex(1U, "pod-source")};
  snapshot.ports = {output_port(1U, 1U, "out", std::move(pod))};
  const auto profiles = standard_profiles(snapshot.vertices.size());

  const auto result = lower_for_test(snapshot, 42U, profiles);

  EXPECT_TRUE(result.errors.empty());
  EXPECT_TRUE(result.generic_output_loans.empty());
  ASSERT_EQ(result.publication_outcomes.size(), 1U);
  EXPECT_EQ(result.publication_outcomes.front().output, PortId{1U});
}

// A graph-interface vertex exposes a public port that is bound to an internal operator port.
// Physical lowering must project the binding with stable PortIds (not declaration pointers).
//
//   [interface: public (port 1)] --binding--> operator: internal (port 2)
//
// REQ: COMPILE-009
// REQ: COMPILE-095
TEST(CompilePhysicalPlan, ProjectsValidatedInterfaceBindingsWithoutDeclarationAddresses) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{1U};
  snapshot.vertices = {vertex(1U, "interface", VertexKind::kGraphInterface),
                       vertex(2U, "operator")};
  snapshot.ports = {
      input_port(1U, 1U, "public", scalar_payload(), 1U, PortOwnerKind::kGraphInterface),
      input_port(2U, 2U, "internal", scalar_payload()),
  };
  // Key setup: one interface binding from public port to internal operator port.
  snapshot.interface_bindings.push_back(InterfaceBindingRecord{
      .public_port = GraphPortId{1U},
      .internal_port = GraphPortId{2U},
      .direction = PortDirection::kInput,
  });
  const auto profiles = standard_profiles(snapshot.vertices.size());

  const auto result = lower_for_test(snapshot, 3U, profiles);

  EXPECT_TRUE(result.errors.empty());
  // Key assertion: binding is projected with the resolved PortId values, not GraphPortId aliases.
  ASSERT_EQ(result.interface_bindings.size(), 1U);
  EXPECT_EQ(result.interface_bindings.front().public_port, PortId{1U});
  EXPECT_EQ(result.interface_bindings.front().internal_port, PortId{2U});
}

}  // namespace
}  // namespace holoscan::detail
// NOLINTEND(bugprone-unchecked-optional-access)

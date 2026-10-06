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

  auto result =
      lower_for_test(snapshot, 41U, profiles, selections, {}, {}, nullptr, {}, partitions);

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
  auto result = lower_for_test(snapshot, 41U, profiles, {}, {}, {}, nullptr, {}, partitions);

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

  auto result =
      lower_for_test(snapshot, 41U, profiles, selections, {}, {}, nullptr, {}, partitions);

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
  auto result = lower_for_test(snapshot, 41U, profiles, {}, {}, {}, nullptr, {}, partitions);

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
// replacement attribution remains enabled on its one publisher.
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
  auto result = lower_for_test(snapshot, 41U, profiles, {}, {}, {}, nullptr, {}, partitions);

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
  auto fanout = lower_for_test(snapshot, 42U, profiles, {}, {}, {}, nullptr, {}, partitions);

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
                              kForceHoloMQProcessLocalInline);

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
//                    `------------> decoy.candidate
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
  // Key setup: candidate fanout requires HoloMQ while the blocking sibling is individually
  // direct-eligible. Pairwise coherence keeps both source-to-sink edges on one delivery family.
  snapshot.edges = {
      edge(1U, 1U, 3U, 1U, QueuePolicy::kDropOldest),
      edge(2U, 2U, 4U),
      edge(3U, 1U, 5U, 1U, QueuePolicy::kDropOldest),
  };
  const auto profiles = standard_profiles(snapshot.vertices.size());

  const auto result = lower_for_test(snapshot, 41U, profiles);

  EXPECT_TRUE(result.errors.empty());
  ASSERT_EQ(result.process_local_channel_services.size(), 2U);
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

// REQ: CHAN-101
// REQ: RUNTIME-019
TEST(CompilePhysicalPlan, ReportsRepresentativeEventWakeApplicabilityTopologies) {
  struct Census {
    std::size_t services{};
    std::size_t edges{};
    std::vector<ProcessLocalHoloMQEventWakeReason> reasons;
  };
  const auto collect = [](GraphSnapshotData snapshot) {
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

// REQ: SCHEMA-003
TEST(CompilePhysicalPlan, AdmitsDifferentSchemaBoundsWithinOneSelectedHoloMQBucket) {
  constexpr std::uint64_t kWriterMaximum = 1024U;
  constexpr std::uint64_t kReaderLimit = 1500U;
  constexpr std::uint64_t kRequiredFrame = kWriterMaximum + kSchemaEnvelopeHeaderSize;
  constexpr std::uint64_t kSelectedBucket = 2048U;
  constexpr std::uint64_t kBucketUsableSerialized = kSelectedBucket - kSchemaEnvelopeHeaderSize;
  static_assert(kRequiredFrame == 1064U);
  static_assert(kBucketUsableSerialized == 2008U);

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
  EXPECT_EQ(service.selected_inline_bucket_bytes, kSelectedBucket);
  auto state = published_physical_plan_state(std::move(result), 42U);
  state.execution_partitions = default_execution_partitions(snapshot);
  EXPECT_TRUE(valid_process_local_channel_plan(state));
  ++state.process_local_channel_services.front().selected_inline_bucket_bytes;
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
  auto result = lower_for_test(snapshot, 42U, profiles, {}, {}, {}, nullptr, {}, partitions);

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

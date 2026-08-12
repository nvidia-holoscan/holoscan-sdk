/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <string>
#include <utility>

#include "physical_plan_test_support.hpp"

// GTest fatal assertions establish optional preconditions that clang-tidy does not model.
// NOLINTBEGIN(bugprone-unchecked-optional-access)
namespace holoscan::detail {
namespace {

using namespace physical_plan_test;

// Two edges with different queue policies leave source fanout unmerged; two services
// are produced. Generic loan pool slots must cover the maximum fanout inflight demand.
//
//   source --[blocking, depth=4, maxinflight=6, cache=5]--> left
//          --[drop-oldest, depth=2, maxinflight=3, cache=2]--> right
//
// REQ: COMPILE-009
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
  // Different queue policies on the two edges: blocking vs drop-oldest.
  snapshot.edges = {edge(1U, 1U, 2U, 4U), edge(2U, 1U, 3U, 2U, QueuePolicy::kDropOldest)};
  snapshot.edges.front().add_flow_options.max_inflight_samples = 6U;
  snapshot.edges.back().add_flow_options.max_inflight_samples = 3U;
  const std::array<InputSelectionPlanRecord, 2> selections{
      InputSelectionPlanRecord{.input = PortId{2U}, .cache_capacity = 5U},
      InputSelectionPlanRecord{.input = PortId{3U}, .cache_capacity = 2U},
  };
  const auto profiles = standard_profiles(snapshot.vertices.size());

  const auto result = lower_for_test(snapshot, 41U, profiles, selections);

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
  EXPECT_EQ(result.process_local_channel_services[1].queue_policy, QueuePolicy::kDropOldest);
  EXPECT_EQ(result.process_local_channel_services[1].max_subscriber_queue_depth, 2U);
  EXPECT_EQ(result.process_local_channel_subscribers[0].edge, EdgeId{1U});
  EXPECT_EQ(result.process_local_channel_subscribers[0].cache_capacity, 5U);
  EXPECT_EQ(result.process_local_channel_subscribers[1].edge, EdgeId{2U});
  EXPECT_EQ(result.process_local_channel_subscribers[1].cache_capacity, 2U);
  EXPECT_EQ(result.process_local_channel_subscribers[0].readiness_capacity, 5U);
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

  const auto result = lower_for_test(snapshot, 41U, profiles, selections);

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
  EXPECT_TRUE(service.event_wake);
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
}

// REQ: COMPILE-009
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

  const auto fallback = lower_for_test(snapshot, 41U, profiles, {}, {}, {}, nullptr, {}, {}, true);
  EXPECT_TRUE(fallback.errors.empty());
  ASSERT_EQ(fallback.process_local_channel_services.size(), 1U);
  EXPECT_EQ(fallback.process_local_channel_services.front().delivery_kind,
            ProcessLocalChannelDeliveryKind::kHoloMQ);
  EXPECT_TRUE(fallback.process_local_channel_services.front().event_wake);
}

// REQ: COMPILE-009
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

  const auto result = lower_for_test(snapshot, 41U, profiles, {}, {}, {}, nullptr, {}, {}, true);

  ASSERT_TRUE(result.errors.empty());
  ASSERT_EQ(result.process_local_channel_services.size(), 1U);
  EXPECT_EQ(result.process_local_channel_services.front().queue_policy, QueuePolicy::kDropOldest);
  EXPECT_EQ(result.process_local_channel_services.front().max_subscriber_queue_depth, 4U);
  EXPECT_EQ(result.process_local_channel_services.front().delivery_kind,
            ProcessLocalChannelDeliveryKind::kHoloMQ);
  EXPECT_TRUE(result.process_local_channel_services.front().event_wake);
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
  ASSERT_EQ(result.process_local_channel_publishers.size(), 1U);
  EXPECT_TRUE(result.process_local_channel_publishers.front().replacement_attribution_required);

  ExecutionPlanState state;
  state.plan_generation = 41U;
  detail::publish_physical_plan_records(std::move(result), state);
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
    return service.delivery_kind == ProcessLocalChannelDeliveryKind::kHoloMQ && service.event_wake;
  }));
  EXPECT_TRUE(std::ranges::all_of(result.process_local_channel_subscribers,
                                  [](const auto& subscriber) { return subscriber.event_wake; }));
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

  const auto fallback = lower_for_test(snapshot, 42U, profiles, {}, {}, {}, nullptr, {}, {}, true);
  ASSERT_TRUE(fallback.errors.empty());
  ASSERT_EQ(fallback.process_local_channel_services.size(), 1U);
  EXPECT_EQ(fallback.process_local_channel_services.front().delivery_kind,
            ProcessLocalChannelDeliveryKind::kHoloMQ);
  EXPECT_TRUE(fallback.process_local_channel_services.front().event_wake);
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
  snapshot.ports[1].setup.tensor_layout = TensorPortLayout{
      .memory_kind = MemoryKind::kHost,
      .dtype = DLDataType{kDLFloat, 32U, 1U},
      .rank = 1U,
  };
  snapshot.ports[1].setup.tensor_allocation =
      TensorAllocationBounds{.max_rank = 1U, .max_byte_span = 1024U};
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

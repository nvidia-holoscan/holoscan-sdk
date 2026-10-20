/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "compiler/execution_plan_lookup.hpp"
#include "compiler/snapshot_lookup.hpp"
#include "core/graph_compiler_access.hpp"

namespace holoscan::detail {
namespace {

// Shared lookups use typed absence and do not fabricate placeholder records for an empty plan.
// REQ: COMPILE-082
// REQ: COMPILE-085
TEST(ExecutionPlanLookup, EmptyPlanReturnsNullForEveryRelation) {
  const ExecutionPlanState plan;

  // Key setup: query every supported relation on one empty final-plan state.
  // Key assertion: every lookup reports absence as nullptr without allocation or diagnostics.
  EXPECT_EQ(find_execution_partition_by_member(plan, VertexId{1U}), nullptr);
  EXPECT_EQ(find_unique_execution_partition_by_member(plan, VertexId{1U}), nullptr);
  EXPECT_EQ(find_unique_cuda_graph_capture_plan_by_region(plan, GraphId{1U}, RegionId{1U}),
            nullptr);
  EXPECT_FALSE(cuda_graph_uses_fixed_output_address(plan, ProcessGroupId{1U}, PortId{1U}));
  EXPECT_FALSE(cuda_graph_fixed_output_device(plan, ProcessGroupId{1U}, PortId{1U}).valid());
  EXPECT_EQ(find_unique_cuda_graph_execution_region_by_member(plan, VertexId{1U}), nullptr);
  EXPECT_EQ(find_execution_partition_by_id(plan, ExecutionPartitionId{1U}), nullptr);
  EXPECT_EQ(find_vertex_process_placement_by_vertex(plan, VertexId{1U}), nullptr);
  EXPECT_EQ(find_unique_vertex_process_placement_by_vertex(plan, VertexId{1U}), nullptr);
  EXPECT_EQ(find_vertex_by_id(plan, VertexId{}), nullptr);
  EXPECT_EQ(find_vertex_by_id(plan, VertexId{1U}), nullptr);
  EXPECT_EQ(find_port_by_id(plan, PortId{1U}), nullptr);
  EXPECT_EQ(find_port_by_id_and_direction(plan, PortId{1U}, PortDirection::kInput), nullptr);
  EXPECT_EQ(find_operator_parameters_by_owner(plan, OperatorRef{1U, 1U, 0U}), nullptr);
  EXPECT_TRUE(parameter_fields_by_owner(plan, OperatorRef{1U, 1U, 0U}).empty());
  EXPECT_TRUE(parameter_derived_terms_by_owner(plan, OperatorRef{1U, 1U, 0U}).empty());
  EXPECT_EQ(find_input_selection_by_input(plan, PortId{1U}), nullptr);
  EXPECT_EQ(find_clock_conversion_by_id(plan, 1U), nullptr);
  EXPECT_EQ(find_clock_by_id(plan, ClockId{1U}), nullptr);
  EXPECT_EQ(find_tensor_interface_pool_by_topic(plan, ExternalTopicId{1U}), nullptr);
  EXPECT_EQ(find_tensor_interface_pool_by_topic_field(plan, ExternalTopicId{1U}, 1U, 2U), nullptr);
  EXPECT_EQ(find_tensor_allocation_by_output(plan, PortId{1U}), nullptr);
  EXPECT_TRUE(tensor_field_allocations_by_output(plan, PortId{1U}).empty());
  EXPECT_EQ(find_tensor_device_resource(plan, ProcessGroupId{1U}, DeviceId{0}), nullptr);
  EXPECT_EQ(find_cuda_compute_stream_by_partition(
                plan, ProcessGroupId{1U}, DeviceId{0}, ExecutionPartitionId{1U}),
            nullptr);
  EXPECT_EQ(
      find_cuda_compute_stream_by_fused_edge(plan, ProcessGroupId{1U}, DeviceId{0}, EdgeId{1U}),
      nullptr);
  EXPECT_FALSE(has_cuda_compute_stream_for_device(plan, ProcessGroupId{1U}, DeviceId{0}));
  EXPECT_EQ(find_process_local_channel_service_by_id(plan, "missing"), nullptr);
  EXPECT_EQ(find_process_local_channel_publisher_by_service_id(plan, "missing"), nullptr);
  EXPECT_EQ(find_cross_process_channel_service_by_id(plan, "missing"), nullptr);
  EXPECT_EQ(find_external_channel_service_by_topic(plan, ExternalTopicId{1U}), nullptr);
  EXPECT_EQ(find_external_channel_publisher_by_topic(plan, ExternalTopicId{1U}), nullptr);
  EXPECT_EQ(find_external_channel_subscriber_by_topic(plan, ExternalTopicId{1U}), nullptr);
  EXPECT_EQ(find_readiness_by_owner(plan, VertexId{1U}), nullptr);
  EXPECT_EQ(find_temporal_trigger_by_owner(plan, OperatorInstanceId{1U}), nullptr);
  EXPECT_EQ(find_unique_temporal_trigger_by_owner(plan, OperatorInstanceId{1U}), nullptr);
  EXPECT_EQ(find_clock_trigger_by_owner(plan, OperatorRef{1U, 1U, 0U}), nullptr);
  EXPECT_EQ(find_predicate_trigger_by_owner(plan, OperatorRef{1U, 1U, 0U}), nullptr);
  EXPECT_EQ(find_unique_predicate_trigger_by_owner(plan, OperatorInstanceId{1U}), nullptr);
  EXPECT_EQ(find_predicate_callable_by_id(plan, "missing"), nullptr);
}

// The shared seam preserves each existing relation's exact key and insertion-order pointer.
// REQ: COMPILE-082
TEST(ExecutionPlanLookup, FindsExactKeysIndependentOfRecordPosition) {
  constexpr VertexId kFirstVertex{11U};
  constexpr VertexId kSecondVertex{22U};
  constexpr OperatorRef kFirstOperator{101U, 7U, 0U};
  constexpr OperatorRef kSecondOperator{202U, 7U, 0U};
  ExecutionPlanState plan;
  plan.execution_partitions = {
      ExecutionPartitionRecord{.id = ExecutionPartitionId{1U}, .members = {kFirstVertex}},
      ExecutionPartitionRecord{.id = ExecutionPartitionId{2U}, .members = {kSecondVertex}},
  };
  plan.vertex_process_placements = {
      VertexProcessPlacementPlanRecord{.vertex = kSecondVertex,
                                       .process_group = ProcessGroupId{2U}},
      VertexProcessPlacementPlanRecord{.vertex = kFirstVertex, .process_group = ProcessGroupId{1U}},
  };
  auto graph = std::make_shared<GraphSnapshotData>();
  graph->vertices = {
      VertexRecord{.id = kSecondVertex, .graph = GraphId{1U}, .name = "second"},
      VertexRecord{.id = kFirstVertex, .graph = GraphId{1U}, .name = "first"},
  };
  plan.graph = GraphCompilerAccess::canonical_view(std::move(graph));
  plan.ports = {
      PortPlanRecord{.id = PortId{2U}, .owner = kSecondVertex, .direction = PortDirection::kOutput},
      PortPlanRecord{.id = PortId{1U}, .owner = kFirstVertex, .direction = PortDirection::kInput},
  };
  plan.operator_parameters = {
      OperatorParameterPlanRecord{.owner = kSecondOperator},
      OperatorParameterPlanRecord{.owner = kFirstOperator},
  };
  plan.input_selections = {
      InputSelectionPlanRecord{.input = PortId{2U}},
      InputSelectionPlanRecord{.input = PortId{1U}},
  };
  plan.clock_conversions = {
      ClockConversionRecord{.id = 0U},
      ClockConversionRecord{.id = 2U},
      ClockConversionRecord{.id = 1U},
  };
  plan.clocks = {
      ClockPlanRecord{.id = ClockId{1U}, .graph = GraphId{1U}, .key = ClockKey{"first"}},
      ClockPlanRecord{.id = ClockId{2U}, .graph = GraphId{1U}, .key = ClockKey{"second"}},
  };
  plan.tensor_interface_pools = {
      TensorInterfacePoolPlanRecord{.topic = ExternalTopicId{2U}},
      TensorInterfacePoolPlanRecord{.topic = ExternalTopicId{1U}},
      TensorInterfacePoolPlanRecord{
          .topic = ExternalTopicId{1U}, .field_id = 1U, .tensor_index = 0U},
  };
  plan.tensor_allocations = {
      TensorAllocationPlanRecord{.output = PortId{2U}},
      TensorAllocationPlanRecord{.output = PortId{1U}},
  };
  plan.tensor_field_allocations = {
      TensorFieldAllocationPlanRecord{.output = PortId{1U}, .field_id = 1U},
      TensorFieldAllocationPlanRecord{.output = PortId{1U}, .field_id = 2U},
      TensorFieldAllocationPlanRecord{.output = PortId{2U}, .field_id = 1U},
  };
  plan.tensor_device_resources = {
      TensorDeviceResourcePlanRecord{.process_group = ProcessGroupId{2U}, .device = DeviceId{1}},
      TensorDeviceResourcePlanRecord{.process_group = ProcessGroupId{1U}, .device = DeviceId{0}},
  };
  plan.cuda_compute_streams = {
      CudaComputeStreamPlanRecord{.process_group = ProcessGroupId{2U},
                                  .device = DeviceId{1},
                                  .partition = ExecutionPartitionId{2U},
                                  .fused_linear_edges = {EdgeId{2U}}},
      CudaComputeStreamPlanRecord{.process_group = ProcessGroupId{1U},
                                  .device = DeviceId{0},
                                  .partition = ExecutionPartitionId{1U},
                                  .fused_linear_edges = {EdgeId{1U}}},
  };
  plan.process_local_channel_services = {
      ProcessLocalChannelServicePlanRecord{.service_id = "second"},
      ProcessLocalChannelServicePlanRecord{.service_id = "first"},
  };
  plan.process_local_channel_publishers = {
      ProcessLocalChannelPublisherPlanRecord{.service_id = "second"},
      ProcessLocalChannelPublisherPlanRecord{.service_id = "first"},
  };
  plan.cross_process_channel_services = {
      CrossProcessChannelServicePlanRecord{.service_id = "second"},
      CrossProcessChannelServicePlanRecord{.service_id = "first"},
  };
  plan.channel_services = {
      ChannelServiceStaticPlanRecord{.topic = ExternalTopicId{2U}},
      ChannelServiceStaticPlanRecord{.topic = ExternalTopicId{1U}},
  };
  plan.channel_publishers = {
      ChannelPublisherEndpointPlanRecord{.topic = ExternalTopicId{2U}},
      ChannelPublisherEndpointPlanRecord{.topic = ExternalTopicId{1U}},
  };
  plan.channel_subscribers = {
      ChannelSubscriberEndpointPlanRecord{.topic = ExternalTopicId{2U}},
      ChannelSubscriberEndpointPlanRecord{.topic = ExternalTopicId{1U}},
  };
  plan.readiness = {
      ReadinessPlanRecord{.owner = kSecondVertex},
      ReadinessPlanRecord{.owner = kFirstVertex},
  };
  plan.temporal_triggers = {
      TemporalTriggerPlanRecord{.owner = kSecondOperator},
      TemporalTriggerPlanRecord{.owner = kFirstOperator},
  };
  plan.clock_triggers = {
      ClockTriggerPlanRecord{.owner = kSecondOperator},
      ClockTriggerPlanRecord{.owner = kFirstOperator},
  };
  plan.predicate_triggers = {
      PredicateTriggerPlanRecord{.id = "second", .owner = kSecondOperator},
      PredicateTriggerPlanRecord{.id = "first", .owner = kFirstOperator},
  };
  plan.predicate_callables = {
      PredicateCallablePlanRecord{.predicate_id = "second"},
      PredicateCallablePlanRecord{.predicate_id = "first"},
  };

  // Key setup: keep the dense partition family canonical, insert the second key before the first
  // in insertion-order keyed record families, and retain one malformed zero conversion identity.
  // Key assertion: each lookup uses its documented dense or explicit key contract and rejects the
  // default-initialized conversion identity before scanning records.
  EXPECT_EQ(find_execution_partition_by_member(plan, kFirstVertex), &plan.execution_partitions[0]);
  EXPECT_EQ(find_unique_execution_partition_by_member(plan, kFirstVertex),
            &plan.execution_partitions[0]);
  EXPECT_EQ(find_execution_partition_by_id(plan, ExecutionPartitionId{1U}),
            &plan.execution_partitions[0]);
  EXPECT_EQ(find_vertex_process_placement_by_vertex(plan, kFirstVertex),
            &plan.vertex_process_placements[1]);
  EXPECT_EQ(find_unique_vertex_process_placement_by_vertex(plan, kFirstVertex),
            &plan.vertex_process_placements[1]);
  EXPECT_EQ(find_vertex_by_id(plan, kFirstVertex), &plan.graph.vertices()[1]);
  EXPECT_EQ(find_port_by_id(plan, PortId{1U}), &plan.ports[1]);
  EXPECT_EQ(find_port_by_id_and_direction(plan, PortId{1U}, PortDirection::kInput), &plan.ports[1]);
  EXPECT_EQ(find_port_by_id_and_direction(plan, PortId{1U}, PortDirection::kOutput), nullptr);
  EXPECT_EQ(find_operator_parameters_by_owner(plan, kFirstOperator), &plan.operator_parameters[1]);
  EXPECT_EQ(find_input_selection_by_input(plan, PortId{1U}), &plan.input_selections[1]);
  EXPECT_EQ(find_clock_conversion_by_id(plan, 0U), nullptr);
  EXPECT_EQ(find_clock_conversion_by_id(plan, 1U), &plan.clock_conversions[2]);
  EXPECT_EQ(find_clock_by_id(plan, ClockId{2U}), &plan.clocks[1]);
  EXPECT_EQ(find_tensor_interface_pool_by_topic(plan, ExternalTopicId{1U}),
            &plan.tensor_interface_pools[1]);
  EXPECT_EQ(find_tensor_interface_pool_by_topic_field(plan, ExternalTopicId{1U}, 1U, 0U),
            &plan.tensor_interface_pools[2]);
  EXPECT_EQ(find_tensor_interface_pool_by_topic_field(plan, ExternalTopicId{1U}, 0U, 0U), nullptr);
  EXPECT_EQ(find_tensor_allocation_by_output(plan, PortId{1U}), &plan.tensor_allocations[1]);
  const auto field_allocations = tensor_field_allocations_by_output(plan, PortId{1U});
  ASSERT_EQ(field_allocations.size(), 2U);
  EXPECT_EQ(&field_allocations.front(), &plan.tensor_field_allocations[0]);
  EXPECT_EQ(&field_allocations.back(), &plan.tensor_field_allocations[1]);
  EXPECT_EQ(find_tensor_device_resource(plan, ProcessGroupId{1U}, DeviceId{0}),
            &plan.tensor_device_resources[1]);
  EXPECT_EQ(find_cuda_compute_stream_by_partition(
                plan, ProcessGroupId{1U}, DeviceId{0}, ExecutionPartitionId{1U}),
            &plan.cuda_compute_streams[1]);
  EXPECT_EQ(
      find_cuda_compute_stream_by_fused_edge(plan, ProcessGroupId{1U}, DeviceId{0}, EdgeId{1U}),
      &plan.cuda_compute_streams[1]);
  EXPECT_TRUE(has_cuda_compute_stream_for_device(plan, ProcessGroupId{1U}, DeviceId{0}));
  EXPECT_EQ(find_process_local_channel_service_by_id(plan, "first"),
            &plan.process_local_channel_services[1]);
  EXPECT_EQ(find_process_local_channel_publisher_by_service_id(plan, "first"),
            &plan.process_local_channel_publishers[1]);
  EXPECT_EQ(find_cross_process_channel_service_by_id(plan, "first"),
            &plan.cross_process_channel_services[1]);
  EXPECT_EQ(find_external_channel_service_by_topic(plan, ExternalTopicId{1U}),
            &plan.channel_services[1]);
  EXPECT_EQ(find_external_channel_publisher_by_topic(plan, ExternalTopicId{1U}),
            &plan.channel_publishers[1]);
  EXPECT_EQ(find_external_channel_subscriber_by_topic(plan, ExternalTopicId{1U}),
            &plan.channel_subscribers[1]);
  EXPECT_EQ(find_readiness_by_owner(plan, kFirstVertex), &plan.readiness[1]);
  EXPECT_EQ(find_temporal_trigger_by_owner(plan, kFirstOperator.id), &plan.temporal_triggers[1]);
  EXPECT_EQ(find_unique_temporal_trigger_by_owner(plan, kFirstOperator.id),
            &plan.temporal_triggers[1]);
  EXPECT_EQ(find_clock_trigger_by_owner(plan, kFirstOperator), &plan.clock_triggers[1]);
  EXPECT_EQ(find_predicate_trigger_by_owner(plan, kFirstOperator), &plan.predicate_triggers[1]);
  EXPECT_EQ(find_unique_predicate_trigger_by_owner(plan, kFirstOperator.id),
            &plan.predicate_triggers[1]);
  EXPECT_EQ(find_predicate_callable_by_id(plan, "first"), &plan.predicate_callables[1]);
}

// Execution partition IDs are a dense final-plan identity, unlike insertion-order keyed
// intermediate record families.
TEST(ExecutionPlanLookup, DensePartitionLookupFailsClosedForMalformedIdentity) {
  ExecutionPlanState plan;
  plan.execution_partitions = {
      ExecutionPartitionRecord{.id = ExecutionPartitionId{1U}},
      ExecutionPartitionRecord{.id = ExecutionPartitionId{2U}},
  };

  // Key setup: begin with canonical dense IDs, then exchange the IDs at their canonical slots.
  ASSERT_EQ(find_execution_partition_by_id(plan, ExecutionPartitionId{1U}),
            &plan.execution_partitions[0]);
  ASSERT_EQ(find_execution_partition_by_id(plan, ExecutionPartitionId{2U}),
            &plan.execution_partitions[1]);
  EXPECT_EQ(find_execution_partition_by_id(plan, ExecutionPartitionId{}), nullptr);
  EXPECT_EQ(find_execution_partition_by_id(plan, ExecutionPartitionId{3U}), nullptr);
  plan.execution_partitions[0].id = ExecutionPartitionId{2U};
  plan.execution_partitions[1].id = ExecutionPartitionId{1U};

  // Key assertion: malformed ordering cannot redirect either lookup to a nearby record.
  EXPECT_EQ(find_execution_partition_by_id(plan, ExecutionPartitionId{1U}), nullptr);
  EXPECT_EQ(find_execution_partition_by_id(plan, ExecutionPartitionId{2U}), nullptr);
}

// Intermediate derivation must reject ambiguous trigger owners even though final-plan accessors
// deliberately remain policy-free first-match lookups.
TEST(ExecutionPlanLookup, UniqueTriggerLookupsRejectDuplicateOwners) {
  constexpr OperatorRef kOwner{7U, 3U, 0U};
  ExecutionPlanState plan;
  plan.temporal_triggers = {
      TemporalTriggerPlanRecord{.owner = kOwner},
      TemporalTriggerPlanRecord{.owner = kOwner},
  };
  plan.predicate_triggers = {
      PredicateTriggerPlanRecord{.id = "first", .owner = kOwner},
      PredicateTriggerPlanRecord{.id = "second", .owner = kOwner},
  };

  // Key setup: provide two intermediate records for each exact Operator owner.
  ASSERT_EQ(find_temporal_trigger_by_owner(plan, kOwner.id), &plan.temporal_triggers.front());
  ASSERT_EQ(find_predicate_trigger_by_owner(plan, kOwner), &plan.predicate_triggers.front());
  // Key assertion: unique derivation lookups fail closed instead of choosing the first record.
  EXPECT_EQ(find_unique_temporal_trigger_by_owner(plan, kOwner.id), nullptr);
  EXPECT_EQ(find_unique_predicate_trigger_by_owner(plan, kOwner.id), nullptr);
}

// Temporal triggers are keyed by Operator instance ID; clock and predicate triggers retain the
// complete plan/run-qualified OperatorRef key used by their existing consumers.
TEST(ExecutionPlanLookup, DistinguishesCompleteOperatorReferenceKeys) {
  constexpr OperatorRef kOwner{9U, 4U, 0U};
  constexpr OperatorRef kDifferentPlan{9U, 5U, 0U};
  ExecutionPlanState plan;
  plan.execution_partitions = {
      ExecutionPartitionRecord{.id = ExecutionPartitionId{3U}, .members = {VertexId{3U}}},
  };
  plan.vertex_process_placements = {
      VertexProcessPlacementPlanRecord{.vertex = VertexId{3U}, .process_group = ProcessGroupId{1U}},
  };
  plan.ports = {
      PortPlanRecord{.id = PortId{3U}, .owner = VertexId{3U}, .direction = PortDirection::kInput}};
  plan.temporal_triggers = {TemporalTriggerPlanRecord{.owner = kOwner}};
  plan.clock_triggers = {ClockTriggerPlanRecord{.owner = kOwner}};
  plan.predicate_triggers = {PredicateTriggerPlanRecord{.id = "predicate", .owner = kOwner}};
  plan.operator_parameters = {OperatorParameterPlanRecord{.owner = kOwner}};

  // Key setup: query the same Operator instance ID under a different plan generation.
  EXPECT_EQ(find_temporal_trigger_by_owner(plan, kDifferentPlan.id), &plan.temporal_triggers[0]);

  // Key assertion: only the temporal relation intentionally ignores the remaining OperatorRef
  // fields; clock and predicate relations require the complete key.
  EXPECT_EQ(find_clock_trigger_by_owner(plan, kDifferentPlan), nullptr);
  EXPECT_EQ(find_predicate_trigger_by_owner(plan, kDifferentPlan), nullptr);
  EXPECT_EQ(find_operator_parameters_by_owner(plan, kDifferentPlan), nullptr);
}

// Parameter subrecords retain the same complete OperatorRef identity as their owning inventory
// record and preserve the compiler's canonical within-owner ordering.
TEST(ExecutionPlanLookup, ParameterOwnerSpansUseCompleteKeyAndPreserveOrder) {
  constexpr OperatorRef kFirstOwner{1U, 7U, 0U};
  constexpr OperatorRef kSecondOwner{2U, 7U, 0U};
  constexpr OperatorRef kDifferentPlan{1U, 8U, 0U};
  ExecutionPlanState plan;
  plan.parameter_fields = {
      ParameterFieldPlanRecord{.owner = kFirstOwner, .canonical_name = "first.alpha"},
      ParameterFieldPlanRecord{.owner = kFirstOwner, .canonical_name = "first.beta"},
      ParameterFieldPlanRecord{.owner = kSecondOwner, .canonical_name = "second.alpha"},
  };
  plan.parameter_derived_terms = {
      ParameterDerivedTermPlanRecord{.term_id = "first.max_start_delay", .owner = kFirstOwner},
      ParameterDerivedTermPlanRecord{.term_id = "first.max-age", .owner = kFirstOwner},
      ParameterDerivedTermPlanRecord{.term_id = "second.max_start_delay", .owner = kSecondOwner},
  };

  // Key setup: give one owner two contiguous records and a neighboring owner one record.
  const auto fields = parameter_fields_by_owner(plan, kFirstOwner);
  const auto terms = parameter_derived_terms_by_owner(plan, kFirstOwner);

  // Key assertion: the exact plan-qualified owner selects only its canonical contiguous range.
  ASSERT_EQ(fields.size(), 2U);
  EXPECT_EQ(&fields.front(), &plan.parameter_fields[0]);
  EXPECT_EQ(fields[0].canonical_name, "first.alpha");
  EXPECT_EQ(fields[1].canonical_name, "first.beta");
  ASSERT_EQ(terms.size(), 2U);
  EXPECT_EQ(&terms.front(), &plan.parameter_derived_terms[0]);
  EXPECT_EQ(terms[0].term_id, "first.max_start_delay");
  EXPECT_EQ(terms[1].term_id, "first.max-age");
  EXPECT_TRUE(parameter_fields_by_owner(plan, kDifferentPlan).empty());
  EXPECT_TRUE(parameter_derived_terms_by_owner(plan, kDifferentPlan).empty());
}

// A populated accepted plan still reports a missing exact key without falling back to a nearby
// record or changing caller-owned diagnostic policy.
TEST(ExecutionPlanLookup, PopulatedPlanReturnsNullForMissingKeys) {
  constexpr OperatorRef kOwner{3U, 6U, 0U};
  ExecutionPlanState plan;
  plan.readiness = {ReadinessPlanRecord{.owner = VertexId{3U}}};
  plan.temporal_triggers = {TemporalTriggerPlanRecord{.owner = kOwner}};
  plan.clock_triggers = {ClockTriggerPlanRecord{.owner = kOwner}};
  plan.predicate_triggers = {PredicateTriggerPlanRecord{.id = "known", .owner = kOwner}};
  plan.predicate_callables = {PredicateCallablePlanRecord{.predicate_id = "known"}};
  plan.operator_parameters = {OperatorParameterPlanRecord{.owner = kOwner}};
  plan.input_selections = {InputSelectionPlanRecord{.input = PortId{3U}}};
  plan.clock_conversions = {ClockConversionRecord{.id = 3U}};
  // A non-dense corrupted clock record must not be mistaken for dense ID one.
  plan.clocks = {ClockPlanRecord{.id = ClockId{3U}}};
  plan.tensor_interface_pools = {TensorInterfacePoolPlanRecord{.topic = ExternalTopicId{3U}}};
  plan.tensor_allocations = {TensorAllocationPlanRecord{.output = PortId{3U}}};
  plan.tensor_field_allocations = {
      TensorFieldAllocationPlanRecord{.output = PortId{3U}, .field_id = 1U}};
  plan.tensor_device_resources = {
      TensorDeviceResourcePlanRecord{.process_group = ProcessGroupId{3U}, .device = DeviceId{0}}};
  plan.cuda_compute_streams = {CudaComputeStreamPlanRecord{.process_group = ProcessGroupId{3U},
                                                           .device = DeviceId{0},
                                                           .partition = ExecutionPartitionId{3U},
                                                           .fused_linear_edges = {EdgeId{3U}}}};
  plan.process_local_channel_services = {
      ProcessLocalChannelServicePlanRecord{.service_id = "known"}};
  plan.process_local_channel_publishers = {
      ProcessLocalChannelPublisherPlanRecord{.service_id = "known"}};
  plan.cross_process_channel_services = {
      CrossProcessChannelServicePlanRecord{.service_id = "known"}};
  plan.channel_services = {ChannelServiceStaticPlanRecord{.topic = ExternalTopicId{3U}}};
  plan.channel_publishers = {ChannelPublisherEndpointPlanRecord{.topic = ExternalTopicId{3U}}};
  plan.channel_subscribers = {ChannelSubscriberEndpointPlanRecord{.topic = ExternalTopicId{3U}}};

  // Key setup: query absent keys adjacent to populated records.
  // Key assertion: absence remains nullptr for every relation.
  EXPECT_EQ(find_execution_partition_by_member(plan, VertexId{4U}), nullptr);
  EXPECT_EQ(find_execution_partition_by_id(plan, ExecutionPartitionId{4U}), nullptr);
  EXPECT_EQ(find_vertex_process_placement_by_vertex(plan, VertexId{4U}), nullptr);
  EXPECT_EQ(find_port_by_id(plan, PortId{4U}), nullptr);
  EXPECT_EQ(find_port_by_id_and_direction(plan, PortId{3U}, PortDirection::kOutput), nullptr);
  EXPECT_EQ(find_operator_parameters_by_owner(plan, OperatorRef{4U, 6U, 0U}), nullptr);
  EXPECT_TRUE(parameter_fields_by_owner(plan, OperatorRef{4U, 6U, 0U}).empty());
  EXPECT_TRUE(parameter_derived_terms_by_owner(plan, OperatorRef{4U, 6U, 0U}).empty());
  EXPECT_EQ(find_input_selection_by_input(plan, PortId{4U}), nullptr);
  EXPECT_EQ(find_clock_conversion_by_id(plan, 4U), nullptr);
  EXPECT_EQ(find_clock_by_id(plan, ClockId{1U}), nullptr);
  EXPECT_EQ(find_tensor_interface_pool_by_topic(plan, ExternalTopicId{4U}), nullptr);
  EXPECT_EQ(find_tensor_interface_pool_by_topic_field(plan, ExternalTopicId{3U}, 1U, 0U), nullptr);
  EXPECT_EQ(find_tensor_allocation_by_output(plan, PortId{4U}), nullptr);
  EXPECT_TRUE(tensor_field_allocations_by_output(plan, PortId{4U}).empty());
  EXPECT_EQ(find_tensor_device_resource(plan, ProcessGroupId{4U}, DeviceId{0}), nullptr);
  EXPECT_EQ(find_tensor_device_resource(plan, ProcessGroupId{3U}, DeviceId{1}), nullptr);
  EXPECT_EQ(find_cuda_compute_stream_by_partition(
                plan, ProcessGroupId{3U}, DeviceId{0}, ExecutionPartitionId{4U}),
            nullptr);
  EXPECT_EQ(
      find_cuda_compute_stream_by_fused_edge(plan, ProcessGroupId{3U}, DeviceId{0}, EdgeId{4U}),
      nullptr);
  EXPECT_FALSE(has_cuda_compute_stream_for_device(plan, ProcessGroupId{4U}, DeviceId{0}));
  EXPECT_EQ(find_process_local_channel_service_by_id(plan, "missing"), nullptr);
  EXPECT_EQ(find_process_local_channel_publisher_by_service_id(plan, "missing"), nullptr);
  EXPECT_EQ(find_cross_process_channel_service_by_id(plan, "missing"), nullptr);
  EXPECT_EQ(find_external_channel_service_by_topic(plan, ExternalTopicId{4U}), nullptr);
  EXPECT_EQ(find_external_channel_publisher_by_topic(plan, ExternalTopicId{4U}), nullptr);
  EXPECT_EQ(find_external_channel_subscriber_by_topic(plan, ExternalTopicId{4U}), nullptr);
  EXPECT_EQ(find_readiness_by_owner(plan, VertexId{4U}), nullptr);
  EXPECT_EQ(find_temporal_trigger_by_owner(plan, OperatorInstanceId{4U}), nullptr);
  EXPECT_EQ(find_clock_trigger_by_owner(plan, OperatorRef{4U, 6U, 0U}), nullptr);
  EXPECT_EQ(find_predicate_trigger_by_owner(plan, OperatorRef{4U, 6U, 0U}), nullptr);
  EXPECT_EQ(find_predicate_callable_by_id(plan, "unknown"), nullptr);
}

// REQ: COMPILE-082
TEST(ExecutionPlanLookup, UniqueMembershipLookupsRejectDuplicateRecords) {
  constexpr VertexId kMember{7U};
  ExecutionPlanState plan;
  plan.execution_partitions = {
      ExecutionPartitionRecord{.id = ExecutionPartitionId{1U}, .members = {kMember}},
      ExecutionPartitionRecord{.id = ExecutionPartitionId{2U}, .members = {kMember}},
  };
  plan.vertex_process_placements = {
      VertexProcessPlacementPlanRecord{.vertex = kMember, .process_group = ProcessGroupId{1U}},
      VertexProcessPlacementPlanRecord{.vertex = kMember, .process_group = ProcessGroupId{2U}},
  };

  // General lookups preserve their first-match behavior. Admission checks use the unique variants
  // so malformed retained records cannot silently choose one partition or process placement.
  EXPECT_EQ(find_execution_partition_by_member(plan, kMember), &plan.execution_partitions[0]);
  EXPECT_EQ(find_unique_execution_partition_by_member(plan, kMember), nullptr);
  EXPECT_EQ(find_vertex_process_placement_by_vertex(plan, kMember),
            &plan.vertex_process_placements[0]);
  EXPECT_EQ(find_unique_vertex_process_placement_by_vertex(plan, kMember), nullptr);
}

// REQ: COMPILE-085
TEST(ExecutionPlanLookup, CUDAGraphLookupsRequireUniqueRegionRelations) {
  constexpr GraphId kGraph{1U};
  constexpr RegionId kRegion{2U};
  constexpr VertexId kMember{3U};
  ExecutionPlanState plan;
  plan.cuda_graph_capture_plans = {
      CUDAGraphCapturePlanRecord{
          .graph = kGraph,
          .region = kRegion,
          .members = {CUDAGraphCaptureMemberRecord{.vertex = kMember}},
      },
  };
  plan.cuda_graph_execution_regions = {
      CUDAGraphExecutionRegionPlanRecord{
          .graph = kGraph,
          .region = kRegion,
          .fused_operator = kMember,
      },
  };

  EXPECT_EQ(find_unique_cuda_graph_capture_plan_by_region(plan, kGraph, kRegion),
            &plan.cuda_graph_capture_plans.front());
  EXPECT_EQ(find_unique_cuda_graph_execution_region_by_member(plan, kMember),
            &plan.cuda_graph_execution_regions.front());
  EXPECT_EQ(find_unique_cuda_graph_capture_plan_by_region(plan, kGraph, RegionId{3U}), nullptr);
  EXPECT_EQ(find_unique_cuda_graph_execution_region_by_member(plan, VertexId{4U}), nullptr);

  // Duplicate capture plans make the execution-region membership relation ambiguous as well.
  plan.cuda_graph_capture_plans.push_back(plan.cuda_graph_capture_plans.front());
  EXPECT_EQ(find_unique_cuda_graph_capture_plan_by_region(plan, kGraph, kRegion), nullptr);
  EXPECT_EQ(find_unique_cuda_graph_execution_region_by_member(plan, kMember), nullptr);

  plan.cuda_graph_capture_plans.pop_back();
  plan.cuda_graph_execution_regions.push_back(plan.cuda_graph_execution_regions.front());
  EXPECT_EQ(find_unique_cuda_graph_execution_region_by_member(plan, kMember), nullptr);
}

// REQ: COMPILE-088
TEST(ExecutionPlanLookup, CUDAGraphFixedOutputLookupsAreScopedToOneProcessGroup) {
  constexpr PortId kOutput{7U};
  constexpr ProcessGroupId kFirstGroup{1U};
  constexpr ProcessGroupId kSecondGroup{2U};
  constexpr ProcessGroupId kMissingGroup{3U};
  ExecutionPlanState plan;
  plan.cuda_graph_execution_regions = {
      CUDAGraphExecutionRegionPlanRecord{
          .process_group = kFirstGroup,
          .device = DeviceId{0},
          .buffers = {CUDAGraphCaptureBufferPlanRecord{.source_port = kOutput}},
      },
      CUDAGraphExecutionRegionPlanRecord{
          .process_group = kSecondGroup,
          .device = DeviceId{1},
          .buffers = {CUDAGraphCaptureBufferPlanRecord{.source_port = kOutput}},
      },
  };

  EXPECT_TRUE(cuda_graph_uses_fixed_output_address(plan, kFirstGroup, kOutput));
  EXPECT_TRUE(cuda_graph_uses_fixed_output_address(plan, kSecondGroup, kOutput));
  EXPECT_FALSE(cuda_graph_uses_fixed_output_address(plan, kMissingGroup, kOutput));
  EXPECT_EQ(cuda_graph_fixed_output_device(plan, kFirstGroup, kOutput), DeviceId{0});
  EXPECT_EQ(cuda_graph_fixed_output_device(plan, kSecondGroup, kOutput), DeviceId{1});
  EXPECT_FALSE(cuda_graph_fixed_output_device(plan, kMissingGroup, kOutput).valid());
}

// REQ: COMPILE-084
TEST(ExecutionPlanLookup, IdentifiesOnlyEdgesInternalToAdmittedCUDAGraphRegions) {
  constexpr GraphId kGraph{1U};
  constexpr RegionId kRegion{2U};
  constexpr EdgeId kInternalEdge{3U};
  constexpr EdgeId kBoundaryEdge{4U};
  ExecutionPlanState plan;
  plan.cuda_graph_capture_plans = {
      CUDAGraphCapturePlanRecord{
          .graph = kGraph,
          .region = kRegion,
          .internal_edges = {CUDAGraphCaptureEdgeRecord{.edge = kInternalEdge}},
      },
  };
  plan.cuda_graph_execution_regions = {
      CUDAGraphExecutionRegionPlanRecord{.graph = kGraph, .region = kRegion},
  };

  EXPECT_TRUE(is_cuda_graph_internal_execution_edge(plan, kInternalEdge));
  EXPECT_FALSE(is_cuda_graph_internal_execution_edge(plan, kBoundaryEdge));

  // Duplicate capture plans make the execution-region relation ambiguous.
  plan.cuda_graph_capture_plans.push_back(plan.cuda_graph_capture_plans.front());
  EXPECT_FALSE(is_cuda_graph_internal_execution_edge(plan, kInternalEdge));

  // Two otherwise valid regions cannot both own the same internal edge.
  plan.cuda_graph_capture_plans.pop_back();
  plan.cuda_graph_capture_plans.push_back(CUDAGraphCapturePlanRecord{
      .graph = GraphId{5U},
      .region = RegionId{6U},
      .internal_edges = {CUDAGraphCaptureEdgeRecord{.edge = kInternalEdge}},
  });
  plan.cuda_graph_execution_regions.push_back(
      CUDAGraphExecutionRegionPlanRecord{.graph = GraphId{5U}, .region = RegionId{6U}});
  EXPECT_FALSE(is_cuda_graph_internal_execution_edge(plan, kInternalEdge));
}

TEST(SnapshotLookup, DenseRecordsAndProfilesFailClosedForInvalidIdentity) {
  GraphSnapshotData snapshot;
  snapshot.graphs = {
      GraphRecord{.id = GraphId{1U}, .name = "root"},
      GraphRecord{.id = GraphId{2U}, .parent = GraphId{1U}, .name = "child"},
  };
  snapshot.vertices = {
      VertexRecord{.id = VertexId{1U}, .graph = GraphId{1U}, .name = "first"},
      VertexRecord{.id = VertexId{2U}, .graph = GraphId{2U}, .name = "second"},
  };
  snapshot.ports = {
      PortRecord{.id = GraphPortId{1U}, .owner = VertexId{1U}},
      PortRecord{.id = GraphPortId{2U}, .owner = VertexId{2U}},
  };
  snapshot.edges = {
      EdgeRecord{.id = EdgeId{1U}, .source = GraphPortId{1U}, .target = GraphPortId{2U}},
      EdgeRecord{.id = EdgeId{2U}, .source = GraphPortId{1U}, .target = GraphPortId{2U}},
  };
  const std::vector<std::optional<Profile>> profiles{Profile::kStandard, Profile::kRealTime};

  // Key setup: start with canonical dense records, then corrupt every record family by swapping
  // the IDs at the two dense positions.
  // Key assertion: canonical identities resolve in constant time and malformed order, zero IDs,
  // out-of-range IDs, and invalid profile IDs all fail closed.
  EXPECT_EQ(find_root_graph(snapshot), &snapshot.graphs.front());
  EXPECT_EQ(find_graph(snapshot, GraphId{2U}), &snapshot.graphs[1]);
  EXPECT_EQ(find_vertex(snapshot, VertexId{2U}), &snapshot.vertices[1]);
  EXPECT_EQ(find_port(snapshot, GraphPortId{2U}), &snapshot.ports[1]);
  EXPECT_EQ(find_edge(snapshot, EdgeId{2U}), &snapshot.edges[1]);
  EXPECT_EQ(effective_profile_for_vertex(profiles, VertexId{2U}), Profile::kRealTime);
  EXPECT_FALSE(effective_profile_for_vertex(profiles, VertexId{}));
  EXPECT_FALSE(effective_profile_for_vertex(profiles, VertexId{3U}));

  snapshot.graphs[0].id = GraphId{2U};
  snapshot.graphs[1].id = GraphId{1U};
  snapshot.vertices[0].id = VertexId{2U};
  snapshot.vertices[1].id = VertexId{1U};
  snapshot.ports[0].id = GraphPortId{2U};
  snapshot.ports[1].id = GraphPortId{1U};
  snapshot.edges[0].id = EdgeId{2U};
  snapshot.edges[1].id = EdgeId{1U};
  EXPECT_EQ(find_graph(snapshot, GraphId{1U}), nullptr);
  EXPECT_EQ(find_vertex(snapshot, VertexId{1U}), nullptr);
  EXPECT_EQ(find_port(snapshot, GraphPortId{1U}), nullptr);
  EXPECT_EQ(find_edge(snapshot, EdgeId{1U}), nullptr);

  snapshot.graphs[0].parent = GraphId{1U};
  snapshot.graphs[1].parent = GraphId{1U};
  EXPECT_EQ(find_root_graph(snapshot), nullptr);
}

TEST(SnapshotLookup, AliasComponentTerminatesOnCyclesAndContainsEachPortOnce) {
  GraphSnapshotData snapshot;
  snapshot.interface_bindings = {
      InterfaceBindingRecord{.public_port = GraphPortId{1U},
                             .internal_port = GraphPortId{2U},
                             .direction = PortDirection::kInput},
      InterfaceBindingRecord{.public_port = GraphPortId{2U},
                             .internal_port = GraphPortId{3U},
                             .direction = PortDirection::kInput},
      InterfaceBindingRecord{.public_port = GraphPortId{3U},
                             .internal_port = GraphPortId{1U},
                             .direction = PortDirection::kInput},
  };

  // Key setup: form a malformed three-port alias cycle.
  // Key assertion: undirected traversal terminates with each reachable port exactly once, while
  // directed public-to-internal resolution rejects the cycle with an invalid identity.
  EXPECT_EQ(interface_alias_component(snapshot, GraphPortId{1U}),
            (std::vector<GraphPortId>{GraphPortId{1U}, GraphPortId{2U}, GraphPortId{3U}}));
  EXPECT_FALSE(resolve_interface_port(snapshot, GraphPortId{1U}));
}

}  // namespace
}  // namespace holoscan::detail

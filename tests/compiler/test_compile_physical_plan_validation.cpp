/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>

#include "physical_plan_test_support.hpp"

// GTest fatal assertions establish optional preconditions that clang-tidy does not model.
// NOLINTBEGIN(bugprone-unchecked-optional-access)
namespace holoscan::detail {
namespace {

using namespace physical_plan_test;

// REQ: COMPILE-002
// REQ: COMPILE-007
// REQ: COMPILE-040
// REQ: CHAN-065
TEST(CompilePhysicalPlanValidation,
     ExternalImportPreservesGraphQueueGeometryAcrossCatalogRoundTrip) {
  Graph graph{"external-plan-validation"};
  test::add_compile_test_default_clock(graph);
  TopicOptions options;
  options.endpoint_name = "validated-import";
  options.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = 3U};
  const auto imported =
      graph.import_topic<std::uint32_t>("holomq://validated-import", std::move(options));
  const auto sink = graph.op<ExternalScalarSinkOperator>("sink");
  graph.add_flow(imported.output(),
                 sink->input,
                 ConnectionOptions{
                     .queue_policy = QueuePolicy::kDropOldest,
                     .queue_depth = 5U,
                 });

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_TRUE(valid_external_channel_plan(*state));
  const auto* catalog = ChannelEndpointCatalogAccess::state(state->endpoint_catalog);
  ASSERT_NE(catalog, nullptr);
  ASSERT_EQ(catalog->endpoints.size(), 1U);
  EXPECT_EQ(catalog->endpoints.front().service_static.queue_policy, QueuePolicy::kDropOldest);
  EXPECT_EQ(catalog->endpoints.front().service_static.max_subscriber_depth, 5U);
  EXPECT_EQ(catalog->endpoints.front().service_static.max_subscriber_borrowed, 5U);

  auto parsed = ChannelEndpointCatalog::parse_json(state->endpoint_catalog.json());
  ASSERT_TRUE(parsed) << parsed.error().message;
  const auto* parsed_state = ChannelEndpointCatalogAccess::state(*parsed);
  ASSERT_NE(parsed_state, nullptr);
  ASSERT_EQ(parsed_state->endpoints.size(), 1U);
  EXPECT_EQ(parsed_state->endpoints.front().service_static.queue_policy, QueuePolicy::kDropOldest);
  EXPECT_EQ(parsed_state->endpoints.front().service_static.max_subscriber_depth, 5U);
  EXPECT_EQ(parsed_state->endpoints.front().service_static.max_subscriber_borrowed, 5U);
}

// REQ: COMPILE-002
// REQ: COMPILE-040
// REQ: CHAN-065
// REQ: RUNTIME-015
TEST(CompilePhysicalPlanValidation, RejectsExternalChannelClosedRelationCorruption) {
  Graph graph{"external-plan-corruption"};
  test::add_compile_test_default_clock(graph);
  TopicOptions options;
  options.endpoint_name = "validated-import";
  options.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = 3U};
  const auto imported =
      graph.import_topic<std::uint32_t>("holomq://validated-import", std::move(options));
  const auto sink = graph.op<ExternalScalarSinkOperator>("sink");
  const auto decoy_sink = graph.op<ExternalScalarSinkOperator>("decoy-sink");
  const auto decoy_source = graph.op<ExternalScalarSourceOperator>("decoy-source");
  graph.add_flow(imported.output(),
                 sink->input,
                 ConnectionOptions{
                     .queue_policy = QueuePolicy::kDropOldest,
                     .queue_depth = 5U,
                 });
  graph.add_flow(decoy_source->output, decoy_sink->input);
  graph.partition("import-path").add(sink);
  graph.partition("decoy-path").add(decoy_source).add(decoy_sink);

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto accepted = ExecutionPlanAccess::retain(plan);
  ASSERT_NE(accepted, nullptr);
  ASSERT_TRUE(valid_external_channel_plan(*accepted));
  const auto external_edge = std::find_if(
      accepted->payload_edges.begin(), accepted->payload_edges.end(), [](const auto& edge) {
        return edge.locality == ChannelLocalityKind::kExternalImport;
      });
  ASSERT_NE(external_edge, accepted->payload_edges.end());
  const auto decoy_input =
      std::find_if(accepted->ports.begin(), accepted->ports.end(), [&](const auto& port) {
        return port.direction == PortDirection::kInput && port.id != external_edge->target;
      });
  const auto decoy_output =
      std::find_if(accepted->ports.begin(), accepted->ports.end(), [&](const auto& port) {
        return port.owner_kind == PortOwnerKind::kOperator &&
               port.direction == PortDirection::kOutput;
      });
  ASSERT_NE(decoy_input, accepted->ports.end());
  ASSERT_NE(decoy_output, accepted->ports.end());
  const auto expect_rejection = [&](auto mutate) {
    ExecutionPlanState candidate = *accepted;
    mutate(candidate);
    EXPECT_FALSE(valid_external_channel_plan(candidate));
  };

  expect_rejection([](auto& state) {
    ASSERT_TRUE(state.channel_services.front().payload_identity.scalar);
    state.channel_services.front().payload_identity.scalar->kind = ScalarIdentityKind::kInt32;
  });
  expect_rejection(
      [](auto& state) { state.channel_publishers.front().topic = ExternalTopicId{2U}; });
  expect_rejection(
      [](auto& state) { state.channel_publishers.push_back(state.channel_publishers.front()); });
  expect_rejection(
      [](auto& state) { state.channel_subscribers.push_back(state.channel_subscribers.front()); });
  expect_rejection([](auto& state) { ++state.channel_services.front().publisher_history_depth; });
  const std::size_t external_edge_index =
      static_cast<std::size_t>(std::distance(accepted->payload_edges.begin(), external_edge));
  expect_rejection([&](auto& state) {
    state.ports[static_cast<std::size_t>(external_edge->target.value() - 1U)].owner =
        decoy_input->owner;
  });
  expect_rejection([&](auto& state) {
    state.ports[static_cast<std::size_t>(external_edge->target.value() - 1U)].direction =
        PortDirection::kOutput;
  });
  expect_rejection(
      [&](auto& state) { state.payload_edges[external_edge_index].target = decoy_input->id; });
  expect_rejection(
      [&](auto& state) { state.payload_edges[external_edge_index].target = decoy_output->id; });
  expect_rejection([&](auto& state) { ++state.payload_edges[external_edge_index].edge.value; });
  expect_rejection([&](auto& state) {
    state.interface_port_bindings.push_back(InterfacePortBindingPlanRecord{
        .public_port = PortId{state.graph.external_topics().front().graph_port.value},
        .internal_port = decoy_input->id,
        .direction = PortDirection::kInput,
    });
  });
  expect_rejection([](auto& state) {
    auto catalog = std::make_shared<ChannelEndpointCatalogState>(
        *ChannelEndpointCatalogAccess::state(state.endpoint_catalog));
    catalog->endpoints.push_back(catalog->endpoints.front());
    state.endpoint_catalog = ChannelEndpointCatalogAccess::make(std::move(catalog));
  });
  expect_rejection([](auto& state) {
    auto catalog = std::make_shared<ChannelEndpointCatalogState>(
        *ChannelEndpointCatalogAccess::state(state.endpoint_catalog));
    catalog->endpoints.front().service_static.queue_policy = QueuePolicy::kBlocking;
    state.endpoint_catalog = ChannelEndpointCatalogAccess::make(std::move(catalog));
  });

  ExecutionPlanState invalid_artifact = *accepted;
  ++invalid_artifact.channel_services.front().publisher_history_depth;
  const auto finalized = finalize_plan_artifact(invalid_artifact);
  ASSERT_FALSE(finalized);
  EXPECT_EQ(finalized.error().message, "EXTERNAL_CHANNEL_PLAN_INVALID");
}

// External Tensor exports contribute queue depth to imported-retention sizing. Keep the
// nonzero-depth invariant explicit so existence-based Runtime checks remain equivalent to the
// Compiler's accumulated-depth predicate.
// REQ: COMPILE-002
// REQ: COMPILE-040
// REQ: CHAN-065
TEST(CompilePhysicalPlanValidation, RejectsZeroDepthExternalTensorExport) {
  Graph graph{"zero-depth-external-tensor-export"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<ManifestTensorOperator>("source");
  constexpr std::array axes{TensorAxisBounds{.max_extent = 16U},
                            TensorAxisBounds{.max_extent = 16U}};
  TopicOptions options;
  options.tensor = LogicalTensorRequirements{
      .memory_kind = MemoryKind::kHost,
      .dtype = DLDataType{kDLFloat, 32U, 1U},
      .rank = 2U,
      .logical_bounds = LogicalTensorBounds{.max_rank = 2U, .max_byte_span = 4096U, .axes = axes},
      .foreign_emit_policy = InterfaceTensorEmitPolicy::kInlineSmallHostThenCopyIfNeeded,
      .max_inline_bytes = 1024U,
  };
  auto exported = graph.export_topic<Tensor>("holomq://zero-depth-tensor", options);
  graph.add_flow(source->tensor, exported.input());

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto accepted = ExecutionPlanAccess::retain(plan);
  ASSERT_NE(accepted, nullptr);
  ASSERT_TRUE(valid_external_channel_plan(*accepted));
  const auto external_edge = std::find_if(
      accepted->payload_edges.begin(), accepted->payload_edges.end(), [](const auto& edge) {
        return edge.locality == ChannelLocalityKind::kExternalExport &&
               edge.payload_identity.kind == PayloadIdentityKind::kTensor;
      });
  ASSERT_NE(external_edge, accepted->payload_edges.end());
  ASSERT_GT(external_edge->queue_depth, 0U);

  ExecutionPlanState invalid = *accepted;
  const auto edge_index =
      static_cast<std::size_t>(std::distance(accepted->payload_edges.begin(), external_edge));
  invalid.payload_edges[edge_index].queue_depth = 0U;
  EXPECT_FALSE(valid_external_channel_plan(invalid));
}

// REQ: COMPILE-040
TEST(CompilePhysicalPlan, RejectsAServiceIdentityThatDoesNotJoinTheCanonicalCatalog) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{1U};
  snapshot.vertices = {vertex(1U, "boundary", VertexKind::kImportedTopic)};
  snapshot.ports = {
      output_port(1U, 1U, "topic", scalar_payload(), 1U, PortOwnerKind::kImportedTopic),
  };
  snapshot.external_topics.push_back(ExternalTopicRecord{
      .id = ExternalTopicId{1U},
      .vertex = VertexId{1U},
      .graph_port = GraphPortId{1U},
      .imported = true,
      .uri = "holomq://canonical",
      .payload_identity = scalar_payload(),
  });
  const auto profiles = standard_profiles(snapshot.vertices.size());
  auto endpoints = catalog_endpoints(snapshot);
  endpoints.front().uri = "holomq://foreign";
  const auto placements = default_process_placements(snapshot);
  const auto package = runtime_package_capability_view();

  const auto result =
      lower_physical_plan(snapshot, 16U, profiles, package, {}, {}, endpoints, placements, {});

  EXPECT_TRUE(has_error(result, "PHYSICAL_SERVICE_CONTRACT_INVALID"));
  EXPECT_TRUE(result.channel_services.empty());
}

// REQ: COMPILE-040
TEST(CompilePhysicalPlan, JoinsCatalogEndpointsByTheCompleteBoundaryContract) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{1U};
  snapshot.vertices = {vertex(1U, "boundary", VertexKind::kImportedTopic)};
  snapshot.ports = {
      output_port(1U, 1U, "topic", scalar_payload(), 1U, PortOwnerKind::kImportedTopic),
  };
  snapshot.external_topics.push_back(ExternalTopicRecord{
      .id = ExternalTopicId{1U},
      .vertex = VertexId{1U},
      .graph_port = GraphPortId{1U},
      .imported = true,
      .uri = "holomq://canonical",
      .payload_identity = scalar_payload(),
  });
  const auto profiles = standard_profiles(snapshot.vertices.size());
  auto endpoints = catalog_endpoints(snapshot);
  auto foreign = endpoints.front();
  foreign.uri = "holomq://foreign";
  foreign.service = "foreign-service";
  endpoints.insert(endpoints.begin(), std::move(foreign));
  const auto placements = default_process_placements(snapshot);
  const auto package = runtime_package_capability_view();

  const auto result =
      lower_physical_plan(snapshot, 16U, profiles, package, {}, {}, endpoints, placements, {});

  EXPECT_TRUE(result.errors.empty())
      << (result.errors.empty() ? std::string{} : result.errors.front().token);
  ASSERT_EQ(result.channel_services.size(), 1U);
  EXPECT_EQ(result.channel_services.front().service_id, "catalog-service-1");
}

// REQ: COMPILE-040
// REQ: CHAN-075
// REQ: CHAN-120
TEST(CompilePhysicalPlan, RejectsCatalogSchemaCapacityMismatch) {
  constexpr std::uint64_t kCapacity = generated_schema_max_serialized_size_v<schema::StringT>;
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{1U};
  snapshot.vertices = {vertex(1U, "boundary", VertexKind::kImportedTopic)};
  snapshot.ports = {
      output_port(1U, 1U, "topic", schema_payload(), 1U, PortOwnerKind::kImportedTopic),
  };
  snapshot.ports.front().serialized_payload_capacity_bytes = kCapacity;
  const PayloadIdentityRecord payload = schema_payload();
  snapshot.external_topics.push_back(ExternalTopicRecord{
      .id = ExternalTopicId{1U},
      .vertex = VertexId{1U},
      .graph_port = GraphPortId{1U},
      .imported = true,
      .uri = "holomq://canonical-schema",
      .payload_identity = payload,
      .serialized_payload_capacity_bytes = kCapacity,
      .schema = payload.schema,
  });
  const auto profiles = standard_profiles(snapshot.vertices.size());
  auto endpoints = catalog_endpoints(snapshot);
  ++endpoints.front().serialized_payload_capacity_bytes;
  const auto placements = default_process_placements(snapshot);
  const auto package = runtime_package_capability_view();

  const auto result =
      lower_physical_plan(snapshot, 16U, profiles, package, {}, {}, endpoints, placements, {});

  EXPECT_TRUE(has_error(result, "PHYSICAL_SERVICE_CONTRACT_INVALID"));
  EXPECT_TRUE(result.channel_services.empty());
}

// REQ: COMPILE-002
// REQ: COMPILE-040
// REQ: CHAN-075
// REQ: CHAN-120
TEST(CompilePhysicalPlanValidation, RejectsExternalSchemaCapacityCorruption) {
  TopicOptions options;
  options.endpoint_name = "validated-schema-import";

  Graph graph{"external-schema-capacity-corruption"};
  test::add_compile_test_default_clock(graph);
  const auto topic =
      graph.import_topic<schema::StringT>("holomq://validated-schema-import", std::move(options));
  const auto sink = graph.op<ExternalSchemaSinkOperator>("sink");
  graph.add_flow(topic.output(), sink->input);
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto accepted = ExecutionPlanAccess::retain(plan);
  ASSERT_NE(accepted, nullptr);
  ASSERT_TRUE(valid_external_channel_plan(*accepted));

  const auto expect_rejection = [&](auto mutate) {
    ExecutionPlanState candidate = *accepted;
    mutate(candidate);
    EXPECT_FALSE(valid_external_channel_plan(candidate));
  };
  expect_rejection([](auto& state) {
    ++state
          .ports[static_cast<std::size_t>(state.graph.external_topics().front().graph_port.value -
                                          1U)]
          .serialized_payload_capacity_bytes;
  });
  expect_rejection([](auto& state) {
    auto catalog = std::make_shared<ChannelEndpointCatalogState>(
        *ChannelEndpointCatalogAccess::state(state.endpoint_catalog));
    ++catalog->endpoints.front().serialized_payload_capacity_bytes;
    state.endpoint_catalog = ChannelEndpointCatalogAccess::make(std::move(catalog));
  });
  expect_rejection([](auto& state) {
    auto catalog = std::make_shared<ChannelEndpointCatalogState>(
        *ChannelEndpointCatalogAccess::state(state.endpoint_catalog));
    ++catalog->endpoints.front().service_static.serialized_payload_capacity_bytes;
    state.endpoint_catalog = ChannelEndpointCatalogAccess::make(std::move(catalog));
  });
}

// REQ: COMPILE-040
TEST(CompilePhysicalPlan, RejectsTensorTransferAdmissionWithoutFiniteLogicalBounds) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{1U};
  snapshot.vertices = {vertex(1U, "boundary", VertexKind::kImportedTopic)};
  snapshot.ports = {
      output_port(1U, 1U, "topic", tensor_payload(), 1U, PortOwnerKind::kImportedTopic),
  };
  snapshot.external_topics.push_back(ExternalTopicRecord{
      .id = ExternalTopicId{1U},
      .vertex = VertexId{1U},
      .graph_port = GraphPortId{1U},
      .imported = true,
      .uri = "holomq://unbounded-tensor",
      .payload_identity = tensor_payload(),
  });
  const auto profiles = standard_profiles(snapshot.vertices.size());

  const auto result = lower_for_test(snapshot, 17U, profiles);

  EXPECT_TRUE(has_error(result, "PHYSICAL_TENSOR_LOGICAL_REQUIREMENTS_INVALID"));
  EXPECT_TRUE(result.tensor_transfer_plans.empty());
  EXPECT_TRUE(result.tensor_interface_pools.empty());
  EXPECT_TRUE(result.imported_generation_capacities.empty());
}

// A host Tensor boundary with rank=0 and a zero-extent axis must still be admitted and
// sized correctly by max_byte_span (slot_count = 8, aggregate = 8 × 64 B).
//
//   [holomq://tensor] --> boundary --[depth=2]--> sink
//
// REQ: COMPILE-009
// REQ: COMPILE-040
TEST(CompilePhysicalPlan, TensorBoundaryPoolAdmitsRankZeroAndZeroExtentWithFiniteBounds) {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{1U};
  snapshot.vertices = {vertex(1U, "boundary", VertexKind::kImportedTopic), vertex(2U, "sink")};
  snapshot.ports = {
      output_port(1U, 1U, "topic", tensor_payload(), 1U, PortOwnerKind::kImportedTopic),
      input_port(2U, 2U, "in", tensor_payload(), 2U),
  };
  snapshot.edges = {edge(1U, 1U, 2U, 2U)};
  ExternalTopicRecord topic{
      .id = ExternalTopicId{1U},
      .vertex = VertexId{1U},
      .graph_port = GraphPortId{1U},
      .imported = true,
      .uri = "holomq://tensor",
      .payload_identity = tensor_payload(),
  };
  topic.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = 2U};
  topic.tensor = LogicalTensorRequirementsRecord{
      .memory_kind = MemoryKind::kHost,
      .dtype = DLDataType{kDLUInt, 8U, 1U},
      .rank = 0U,
      .max_rank = 2U,
      .max_byte_span = 64U,
      .axes = {{.max_extent = 0U}, {.max_extent = 8U}},
      .strides = TensorStridePolicy::kContiguousOnly,
      .foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
  };
  snapshot.external_topics.push_back(std::move(topic));
  const auto profiles = standard_profiles(snapshot.vertices.size());

  const auto result = lower_for_test(snapshot, 22U, profiles);

  EXPECT_TRUE(result.errors.empty());
  // Key assertion: pool is admitted with rank 0 and slot_count = 8 (driven by max_byte_span).
  ASSERT_EQ(result.tensor_interface_pools.size(), 1U);
  EXPECT_EQ(result.tensor_interface_pools.front().logical.rank, 0U);
  EXPECT_EQ(result.tensor_interface_pools.front().emit_policy,
            InterfaceTensorEmitPolicy::kCopyIfNeeded);
  EXPECT_EQ(result.tensor_interface_pools.front().plan_generation, 22U);
  EXPECT_EQ(result.tensor_interface_pools.front().run_generation, 0U);
  EXPECT_FALSE(result.tensor_interface_pools.front().device.valid());
  EXPECT_EQ(result.tensor_interface_pools.front().allocator_process_group, ProcessGroupId{1U});
  EXPECT_EQ(result.tensor_interface_pools.front().slot_count, 8U);
  EXPECT_EQ(result.tensor_interface_pools.front().aggregate_byte_capacity,
            result.tensor_interface_pools.front().slot_count * 64U);
  ASSERT_EQ(result.imported_generation_capacities.size(), 1U);
  EXPECT_EQ(result.imported_generation_capacities.front().endpoint_max_generations, 1U);
  EXPECT_EQ(result.imported_generation_capacities.front().candidate_reservation_capacity, 0U);
  EXPECT_EQ(result.imported_generation_capacities.front().release_record_capacity, 1U);
  EXPECT_TRUE(result.imported_generation_capacities.front().no_wrap_aba_generation);

  std::array<bool, 17> reasons{};
  std::uint32_t producer_capacity = 0U;
  std::uint32_t delivery_capacity = 0U;
  std::uint32_t retention_capacity = 0U;
  std::uint32_t retirement_capacity = 0U;
  for (const auto& contributor : result.pool_capacity_contributors) {
    if (contributor.pool_id != result.tensor_interface_pools.front().pool_id) {
      continue;
    }
    if (contributor.contributor_id.ends_with(".producer")) {
      producer_capacity = contributor.maximum_distinct_slots;
    } else if (contributor.contributor_id.ends_with(".delivery")) {
      delivery_capacity = contributor.maximum_distinct_slots;
    } else if (contributor.contributor_id.ends_with(".retention")) {
      retention_capacity = contributor.maximum_distinct_slots;
    } else if (contributor.contributor_id.ends_with(".retirement")) {
      retirement_capacity = contributor.maximum_distinct_slots;
    }
    for (const auto reason : contributor.reasons) {
      reasons[static_cast<std::size_t>(reason)] = true;
    }
  }
  EXPECT_EQ(producer_capacity, 2U);
  EXPECT_EQ(delivery_capacity, 2U);
  EXPECT_EQ(retention_capacity, 2U);
  EXPECT_EQ(retirement_capacity, 2U);
  EXPECT_TRUE(reasons[static_cast<std::size_t>(TensorSlotClaimReason::kProducerLoan)]);
  EXPECT_TRUE(reasons[static_cast<std::size_t>(TensorSlotClaimReason::kProducerWriter)]);
  EXPECT_TRUE(reasons[static_cast<std::size_t>(TensorSlotClaimReason::kAcceptedPublication)]);
  EXPECT_TRUE(reasons[static_cast<std::size_t>(TensorSlotClaimReason::kSubscriberBorrowOrImport)]);
  EXPECT_TRUE(reasons[static_cast<std::size_t>(TensorSlotClaimReason::kRetirementService)]);
  EXPECT_FALSE(reasons[static_cast<std::size_t>(TensorSlotClaimReason::kConsumerCudaRelease)]);
  EXPECT_FALSE(reasons[static_cast<std::size_t>(TensorSlotClaimReason::kReuseLast)]);
  EXPECT_TRUE(reasons[static_cast<std::size_t>(TensorSlotClaimReason::kDlpackOrFrameworkBorrow)]);

  ASSERT_EQ(result.tensor_transfer_plans.size(), 7U);
  const auto copied_host =
      std::find_if(result.tensor_transfer_plans.begin(),
                   result.tensor_transfer_plans.end(),
                   [](const auto& transfer) {
                     return transfer.source_class == TensorSourceClass::kOrdinaryHostContiguous;
                   });
  ASSERT_NE(copied_host, result.tensor_transfer_plans.end());
  ASSERT_TRUE(copied_host->realization.has_value());
  EXPECT_EQ(*copied_host->realization, TensorTransferRealization::kCopiedToExportablePool);
  EXPECT_EQ(copied_host->min_logical_bytes, 0U);
  EXPECT_EQ(copied_host->max_logical_bytes, 64U);
  EXPECT_TRUE(copied_host->delivered_read_only);
  const auto pinned = std::find_if(
      result.tensor_transfer_plans.begin(),
      result.tensor_transfer_plans.end(),
      [](const auto& transfer) { return transfer.source_class == TensorSourceClass::kPinnedHost; });
  ASSERT_NE(pinned, result.tensor_transfer_plans.end());
  EXPECT_EQ(pinned->disposition, PlanDisposition::kRejected);
  EXPECT_FALSE(pinned->realization.has_value());
}

// Boundary vertices are non-executable; pool ownership follows the adjacent Operator.
// Tested for both directions in a single parameterized lambda:
//
//   imported:  [holomq://adjacent-import] --> boundary (pg=9) ---> operator (pg=4)
//              pool allocator_process_group must be pg=4 (the consuming Operator)
//
//   exported:  operator (pg=4) ---> boundary (pg=9) --> [holomq://adjacent-export]
//              pool allocator_process_group must be pg=4 (the producing Operator)
//
// REQ: COMPILE-009
// REQ: COMPILE-040
TEST(CompilePhysicalPlan, TensorBoundaryPoolUsesAdjacentExecutableForAllocationOwnership) {
  const auto lower_boundary = [](bool imported) {
    GraphSnapshotData snapshot;
    snapshot.generation = GraphGeneration{1U};
    snapshot.vertices = {
        vertex(1U, "boundary", imported ? VertexKind::kImportedTopic : VertexKind::kExportedTopic),
        vertex(2U, "operator"),
    };
    if (imported) {
      snapshot.ports = {
          output_port(1U, 1U, "topic", tensor_payload(), 1U, PortOwnerKind::kImportedTopic),
          input_port(2U, 2U, "in", tensor_payload(), 2U),
      };
      snapshot.edges = {edge(1U, 1U, 2U, 2U)};
    } else {
      snapshot.ports = {
          input_port(1U, 1U, "topic", tensor_payload(), 2U, PortOwnerKind::kExportedTopic),
          output_port(2U, 2U, "out", tensor_payload(), 2U),
      };
      snapshot.edges = {edge(1U, 2U, 1U, 2U)};
    }
    ExternalTopicRecord topic{
        .id = ExternalTopicId{1U},
        .vertex = VertexId{1U},
        .graph_port = GraphPortId{1U},
        .imported = imported,
        .uri = imported ? "holomq://adjacent-import" : "holomq://adjacent-export",
        .payload_identity = tensor_payload(),
    };
    if (imported) {
      topic.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = 2U};
    } else {
      topic.bounds.subscriber =
          SubscriberEndpointBounds{.queue_depth = 2U, .max_borrowed_samples = 2U};
    }
    topic.tensor = LogicalTensorRequirementsRecord{
        .memory_kind = MemoryKind::kHost,
        .dtype = DLDataType{kDLUInt, 8U, 1U},
        .rank = 1U,
        .max_rank = 1U,
        .max_byte_span = 64U,
        .axes = {{.max_extent = 64U}},
        .foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
    };
    snapshot.external_topics.push_back(std::move(topic));
    const auto profiles = standard_profiles(snapshot.vertices.size());
    const auto endpoints = catalog_endpoints(snapshot);
    const std::array placements{
        VertexProcessPlacementPlanRecord{
            .vertex = VertexId{1U},
            .process_group = ProcessGroupId{9U},
            .isolation = ProcessIsolation::kSameProcess,
            .launch_group = "boundary",
        },
        VertexProcessPlacementPlanRecord{
            .vertex = VertexId{2U},
            .process_group = ProcessGroupId{4U},
            .isolation = ProcessIsolation::kSameProcess,
            .launch_group = "operator",
        },
    };
    const std::array partitions{ExecutionPartitionRecord{
        .id = ExecutionPartitionId{1U},
        .members = {VertexId{2U}},
        .execution_lane = 0U,
    }};
    const RuntimePackageCapabilityView package = runtime_package_capability_view();
    return lower_physical_plan(
        snapshot, 23U, profiles, package, {}, {}, endpoints, placements, {}, partitions, {});
  };

  for (const bool imported : {true, false}) {
    const auto result = lower_boundary(imported);
    EXPECT_TRUE(result.errors.empty());
    ASSERT_EQ(result.tensor_interface_pools.size(), 1U);
    // Key assertion: pool owner is always the Operator process group (pg=4), not the boundary
    // (pg=9).
    EXPECT_EQ(result.tensor_interface_pools.front().allocator_process_group, ProcessGroupId{4U});
  }
}

}  // namespace
}  // namespace holoscan::detail
// NOLINTEND(bugprone-unchecked-optional-access)

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "physical_plan_test_support.hpp"

#include <string>
#include <utility>
#include <vector>

namespace holoscan::detail::physical_plan_test {
[[nodiscard]] PayloadIdentityRecord scalar_payload() {
  return PayloadIdentityRecord{
      .kind = PayloadIdentityKind::kScalar,
      .payload_kind = PayloadKind::kInlineTyped,
      .scalar = ScalarIdentityRecord{.kind = ScalarIdentityKind::kUInt32},
  };
}

[[nodiscard]] PayloadIdentityRecord tensor_payload() {
  return PayloadIdentityRecord{
      .kind = PayloadIdentityKind::kTensor,
      .payload_kind = PayloadKind::kDescriptorBacked,
  };
}

[[nodiscard]] PayloadIdentityRecord schema_payload() {
  return operator_spec_payload_identity<schema::StringT>();
}

[[nodiscard]] PayloadIdentityRecord image_payload() {
  return operator_spec_payload_identity<schema::ImageT>();
}

[[nodiscard]] PayloadIdentityRecord tensor_packet_payload() {
  return operator_spec_payload_identity<schema::TensorPacketT>();
}

[[nodiscard]] std::uint64_t scalar_generic_loan_slot_size(const PayloadIdentityRecord& payload) {
  if (payload.kind != PayloadIdentityKind::kScalar || !payload.scalar) {
    return 0U;
  }
  switch (payload.scalar->kind) {
    case ScalarIdentityKind::kBoolean:
      return sizeof(bool);
    case ScalarIdentityKind::kInt8:
    case ScalarIdentityKind::kUInt8:
      return 1U;
    case ScalarIdentityKind::kInt16:
    case ScalarIdentityKind::kUInt16:
      return 2U;
    case ScalarIdentityKind::kInt32:
    case ScalarIdentityKind::kUInt32:
    case ScalarIdentityKind::kFloat32:
      return 4U;
    case ScalarIdentityKind::kInt64:
    case ScalarIdentityKind::kUInt64:
    case ScalarIdentityKind::kFloat64:
      return 8U;
    case ScalarIdentityKind::kLocalEnum:
      return 0U;
  }
  return 0U;
}

[[nodiscard]] VertexRecord vertex(std::uint64_t id, std::string_view name, VertexKind kind) {
  return VertexRecord{
      .id = VertexId{id},
      .graph = GraphId{1},
      .kind = kind,
      .name = std::string(name),
  };
}

[[nodiscard]] PortRecord output_port(std::uint64_t id, std::uint64_t owner, std::string_view name,
                                     PayloadIdentityRecord payload, std::uint32_t burst,
                                     PortOwnerKind owner_kind) {
  PortConfigurationRecord setup;
  setup.max_emits_per_compute = burst;
  setup.late_joiner_history_depth = 0U;
  const std::uint64_t generic_loan_slot_size =
      owner_kind == PortOwnerKind::kOperator ? scalar_generic_loan_slot_size(payload) : 0U;
  return PortRecord{
      .id = GraphPortId{id},
      .owner = VertexId{owner},
      .owner_kind = owner_kind,
      .direction = PortDirection::kOutput,
      .name = std::string(name),
      .payload_identity = std::move(payload),
      .generic_loan_supported = generic_loan_slot_size != 0U,
      .generic_loan_slot_size_bytes = generic_loan_slot_size,
      .setup = std::move(setup),
  };
}

[[nodiscard]] PortRecord input_port(std::uint64_t id, std::uint64_t owner, std::string_view name,
                                    PayloadIdentityRecord payload, std::uint32_t depth,
                                    PortOwnerKind owner_kind) {
  PortConfigurationRecord setup;
  setup.queue_depth = depth;
  return PortRecord{
      .id = GraphPortId{id},
      .owner = VertexId{owner},
      .owner_kind = owner_kind,
      .direction = PortDirection::kInput,
      .name = std::string(name),
      .payload_identity = std::move(payload),
      .setup = std::move(setup),
  };
}

[[nodiscard]] EdgeRecord edge(std::uint64_t id, std::uint64_t source, std::uint64_t target,
                              std::uint32_t depth, QueuePolicy policy) {
  ConnectionOptionsRecord options;
  options.queue_policy = policy;
  options.queue_depth = depth;
  options.max_inflight_samples = depth;
  return EdgeRecord{
      .id = EdgeId{id},
      .source = GraphPortId{source},
      .target = GraphPortId{target},
      .add_flow_options = std::move(options),
  };
}

[[nodiscard]] TensorFieldConfigurationRecord tensor_field(
    std::uint16_t field_id, std::uint16_t tensor_index, MemoryKind memory_kind, std::uint32_t rank,
    std::uint64_t max_byte_span, InterfaceTensorEmitPolicy emit_policy) {
  return TensorFieldConfigurationRecord{
      .field_id = field_id,
      .tensor_index = tensor_index,
      .tensor_layout =
          TensorRepresentation{
              .memory_kind = memory_kind,
              .dtype = DLDataType{kDLUInt, 8U, 1U},
              .rank = rank,
          },
      .tensor_allocation = TensorPoolBounds{.max_rank = rank, .max_byte_span = max_byte_span},
      .payload_options =
          TensorFieldTransferRecord{
              .max_tensor_byte_span = max_byte_span,
              .interface_tensor_emit_policy = emit_policy,
          },
  };
}

[[nodiscard]] std::vector<std::optional<Profile>> standard_profiles(std::size_t count) {
  return std::vector<std::optional<Profile>>(count, Profile::kStandard);
}

[[nodiscard]] std::vector<VertexProcessPlacementPlanRecord> default_process_placements(
    const GraphSnapshotData& snapshot) {
  std::vector<VertexProcessPlacementPlanRecord> result;
  result.reserve(snapshot.vertices.size());
  for (const auto& item : snapshot.vertices) {
    result.push_back(VertexProcessPlacementPlanRecord{
        .vertex = item.id,
        .process_group = ProcessGroupId{1U},
        .isolation = ProcessIsolation::kSameProcess,
        .launch_group = "core.default",
    });
  }
  return result;
}

[[nodiscard]] std::vector<VertexProcessPlacementPlanRecord> split_process_placements(
    const GraphSnapshotData& snapshot) {
  std::vector<VertexProcessPlacementPlanRecord> result;
  result.reserve(snapshot.vertices.size());
  for (std::size_t index = 0U; index < snapshot.vertices.size(); ++index) {
    const ProcessGroupId process_group{static_cast<std::uint32_t>(index + 1U)};
    result.push_back(VertexProcessPlacementPlanRecord{
        .vertex = snapshot.vertices[index].id,
        .process_group = process_group,
        .isolation = ProcessIsolation::kSeparateProcess,
        .launch_group = "test.process." + std::to_string(process_group.value),
    });
  }
  return result;
}

[[nodiscard]] std::vector<ExecutionPartitionRecord> default_execution_partitions(
    const GraphSnapshotData& snapshot) {
  ExecutionPartitionRecord partition{
      .id = ExecutionPartitionId{1U},
      .execution_lane = 0U,
  };
  for (const auto& item : snapshot.vertices) {
    if (item.kind == VertexKind::kOperator) {
      partition.members.push_back(item.id);
    }
  }
  if (partition.members.empty()) {
    return {};
  }
  return {std::move(partition)};
}

[[nodiscard]] std::vector<ChannelEndpointRecord> catalog_endpoints(
    const GraphSnapshotData& snapshot) {
  std::vector<ChannelEndpointRecord> result;
  result.reserve(snapshot.external_topics.size());
  for (const auto& topic : snapshot.external_topics) {
    result.push_back(ChannelEndpointRecord{
        .name = "endpoint-" + std::to_string(topic.id.value),
        .uri = topic.uri,
        .direction = topic.imported ? TopicDirection::kImport : TopicDirection::kExport,
        .boundary_port = PortId{topic.graph_port.value},
        .service = "catalog-service-" + std::to_string(topic.id.value),
        .payload_identity = topic.payload_identity,
        .schema_writer_max_serialized_bytes = topic.serialized_payload_capacity_bytes,
        .attach = topic.attach,
        .tensor = topic.tensor,
        .tensor_fields = topic.tensor_fields,
        .bounds = topic.bounds,
    });
  }
  return result;
}

[[nodiscard]] PhysicalPlanLoweringResult lower_for_test(
    const GraphSnapshotData& snapshot, PlanGenerationId plan_generation,
    std::span<const std::optional<Profile>> profiles,
    std::span<const InputSelectionPlanRecord> selections,
    std::span<const BracketedWindowPlanRecord> windows,
    std::span<const LifecycleRestartTargetPlanRecord> lifecycle_targets,
    const RuntimePackageCapabilityView* package_override,
    std::span<const TensorDeviceBindingRecord> tensor_device_bindings,
    std::span<const ExecutionPartitionRecord> execution_partitions,
    const LowerForTestOptions options) {
  const auto endpoints = catalog_endpoints(snapshot);
  const auto placements = default_process_placements(snapshot);
  const auto default_partitions = default_execution_partitions(snapshot);
  const auto accepted_partitions =
      execution_partitions.empty() ? std::span<const ExecutionPartitionRecord>{default_partitions}
                                   : execution_partitions;
  const RuntimePackageCapabilityView generated = runtime_package_capability_view();
  const RuntimePackageCapabilityView& package =
      package_override != nullptr ? *package_override : generated;
  return lower_physical_plan(snapshot,
                             plan_generation,
                             profiles,
                             package,
                             selections,
                             windows,
                             endpoints,
                             placements,
                             lifecycle_targets,
                             accepted_partitions,
                             tensor_device_bindings,
                             options.force_holomq_process_local_inline,
                             options.force_holomq_process_local_event_wake,
                             {},
                             options.resolved_queue_sizings,
                             options.queue_sizing_recipes,
                             options.cuda_compute_stream_policy,
                             false,
                             options.force_holomq_process_local_event_wake
                                 ? ProcessLocalEmitAdmissionScope::kNone
                                 : options.emit_admission,
                             options.force_holomq_process_local_fanout,
                             options.lifecycle_closures);
}

[[nodiscard]] ExecutionPlanState published_physical_plan_state(
    PhysicalPlanLoweringResult result, PlanGenerationId plan_generation,
    std::span<const VertexProcessPlacementPlanRecord> placements,
    const RatePlanLoweringResult* exact_rate_plan) {
  ExecutionPlanState state;
  state.plan_generation = plan_generation;
  // lower_for_test() deliberately preserves the original shared-stream fixture default. Publish
  // that explicit test policy instead of relying on ExecutionPlanState's production default.
  state.cuda_compute_stream_policy = CudaComputeStreamPolicy::kSharedProcessDevice;
  state.vertex_process_placements.assign(placements.begin(), placements.end());
  detail::publish_physical_plan_records(std::move(result), state);

  if (exact_rate_plan != nullptr) {
    state.port_rates = exact_rate_plan->port_rates;
    state.accumulation_horizons = exact_rate_plan->accumulation_horizons;
    state.hard_floor_contributors = exact_rate_plan->hard_floor_contributors;
    state.queue_sizing_recipes = exact_rate_plan->queue_sizing_recipes;
    return state;
  }

  // Physical-only fixture callers do not run rate lowering. Keep their minimal artifact
  // scaffolding separate from callers that supply the exact semantic rate records above.
  state.port_rates.reserve(state.ports.size());
  for (const auto& port : state.ports) {
    state.port_rates.push_back(PortRatePlanRecord{
        .port = port.id,
        .kind = PortRateKind::kUnspecified,
        .disposition = PlanDisposition::kAccepted,
    });
  }
  state.hard_floor_contributors.reserve(state.payload_edges.size());
  state.queue_sizing_recipes.reserve(state.payload_edges.size());
  for (std::size_t index = 0U; index < state.payload_edges.size(); ++index) {
    const auto& payload_edge = state.payload_edges[index];
    const std::uint32_t id = static_cast<std::uint32_t>(index + 1U);
    state.hard_floor_contributors.push_back(HardFloorContributorPlanRecord{
        .id = HardFloorContributorId{id},
        .edge = payload_edge.edge,
        .origin = HardFloorOrigin::kMinimumQueue,
        .required_slots = 1U,
        .scenario = HardFloorScenarioId{id},
        .alternative_group = HardFloorAlternativeGroupId{id},
    });
    state.queue_sizing_recipes.push_back(QueueSizingRecipePlanRecord{
        .id = QueueSizingRecipeId{id},
        .edge = payload_edge.edge,
        .origin = QueueDepthOrigin::kAddFlowOption,
        .explicit_depth = payload_edge.queue_depth,
        .headroom_ratio = RateRatio{2U, 1U},
        .reuse_policy = ProviderReusePolicy::kCompatibleExactOrLarger,
        .geometry_mode = AllocationGeometryMode::kCompileConcrete,
        .disposition = PlanDisposition::kAccepted,
    });
  }
  return state;
}

[[nodiscard]] bool has_error(const PhysicalPlanLoweringResult& result, std::string_view token) {
  return std::any_of(result.errors.begin(), result.errors.end(), [&](const auto& error) {
    return error.token == token;
  });
}

[[nodiscard]] const PortPlanRecord* find_port(const PhysicalPlanLoweringResult& result, PortId id) {
  const auto found = std::ranges::find(result.ports, id, &PortPlanRecord::id);
  return found == result.ports.end() ? nullptr : &*found;
}

[[nodiscard]] const TensorAllocationPlanRecord* find_allocation(
    const PhysicalPlanLoweringResult& result, PortId output) {
  const auto found =
      std::ranges::find(result.tensor_allocations, output, &TensorAllocationPlanRecord::output);
  return found == result.tensor_allocations.end() ? nullptr : &*found;
}

[[nodiscard]] const PoolCapacityContributorRecord* find_contributor(
    const PhysicalPlanLoweringResult& result, std::string_view pool_id, std::string_view suffix) {
  const auto found =
      std::ranges::find_if(result.pool_capacity_contributors, [&](const auto& contributor) {
        return contributor.pool_id == pool_id && contributor.contributor_id.ends_with(suffix);
      });
  return found == result.pool_capacity_contributors.end() ? nullptr : &*found;
}

[[nodiscard]] MultiTensorSnapshot make_multi_tensor_snapshot(
    std::uint64_t serialized_payload_capacity_bytes) {
  GraphSnapshotData graph;
  graph.generation = GraphGeneration{7U};
  graph.vertices = {vertex(1U, "multi-source"), vertex(2U, "multi-sink")};
  graph.ports = {
      output_port(1U, 1U, "frames", image_payload(), 2U),
      input_port(2U, 2U, "frames", image_payload(), 2U),
  };
  for (auto& port : graph.ports) {
    port.serialized_payload_capacity_bytes = serialized_payload_capacity_bytes;
    port.descriptor_tensor_field_count = 3U;
  }
  graph.ports[0].setup.tensor_fields = {
      tensor_field(11U, 2U, MemoryKind::kHost, 1U, 256U),
      tensor_field(2U, 0U, MemoryKind::kHost, 1U, 128U),
      tensor_field(7U, 1U, MemoryKind::kHost, 2U, 512U),
  };
  graph.ports[1].setup.tensor_fields = graph.ports[0].setup.tensor_fields;
  graph.edges = {edge(1U, 1U, 2U, 2U)};

  MultiTensorSnapshot result;
  result.graph = std::move(graph);
  result.profiles = standard_profiles(result.graph.vertices.size());
  result.placements = split_process_placements(result.graph);
  result.partitions = default_execution_partitions(result.graph);
  return result;
}

[[nodiscard]] ExecutionPlanState physical_artifact_state() {
  GraphSnapshotData snapshot;
  snapshot.generation = GraphGeneration{9U};
  snapshot.vertices = {
      vertex(1U, "scalar-source"),
      vertex(2U, "tensor-source"),
      vertex(3U, "scalar-sink"),
      vertex(4U, "tensor-boundary", VertexKind::kImportedTopic),
      vertex(5U, "tensor-sink"),
      vertex(6U, "interface", VertexKind::kGraphInterface),
      vertex(7U, "internal"),
  };
  snapshot.ports = {
      output_port(1U, 1U, "scalar", scalar_payload(), 2U),
      output_port(2U, 2U, "allocated", tensor_payload(), 2U),
      input_port(3U, 3U, "scalar", scalar_payload(), 3U),
      output_port(4U, 4U, "boundary", tensor_payload(), 1U, PortOwnerKind::kImportedTopic),
      input_port(5U, 5U, "tensor", tensor_payload(), 2U),
      input_port(6U, 6U, "public", scalar_payload(), 1U, PortOwnerKind::kGraphInterface),
      input_port(7U, 7U, "internal", scalar_payload(), 1U),
  };
  snapshot.ports[1].setup.tensor_layout = TensorRepresentation{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = DLDataType{kDLFloat, 32U, 1U},
      .rank = 2U,
  };
  snapshot.ports[1].setup.tensor_allocation =
      TensorPoolBounds{.max_rank = 3U, .max_byte_span = 4096U};
  snapshot.edges = {edge(1U, 1U, 3U, 3U), edge(2U, 4U, 5U, 2U)};
  snapshot.interface_bindings.push_back(InterfaceBindingRecord{
      .public_port = GraphPortId{6U},
      .internal_port = GraphPortId{7U},
      .direction = PortDirection::kInput,
  });
  ExternalTopicRecord topic{
      .id = ExternalTopicId{1U},
      .vertex = VertexId{4U},
      .graph_port = GraphPortId{4U},
      .imported = true,
      .uri = "holomq://artifact-tensor",
      .payload_identity = tensor_payload(),
  };
  topic.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = 2U};
  topic.tensor = TensorBoundaryContractRecord{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = DLDataType{kDLFloat, 32U, 1U},
      .rank = 2U,
      .max_rank = 2U,
      .max_byte_span = 4096U,
      .axes = {{.max_extent = 16U}, {.max_extent = 16U}},
      .foreign_emit_policy = InterfaceTensorEmitPolicy::kInlineSmallHostThenCopyIfNeeded,
      .max_inline_bytes = 1024U,
      .max_staging_byte_span = 4096U,
      .memory_conversion = MemoryConversionPolicy::kDisable,
  };
  topic.memory_conversion = MemoryConversionPolicy::kDisable;
  snapshot.external_topics.push_back(std::move(topic));
  const auto profiles = standard_profiles(snapshot.vertices.size());
  const std::array device_bindings{
      TensorDeviceBindingRecord{.port = PortId{2U}, .device = DeviceId{0}},
      TensorDeviceBindingRecord{.port = PortId{4U}, .device = DeviceId{0}},
  };
  const auto placements = default_process_placements(snapshot);
  const auto partitions = default_execution_partitions(snapshot);
  auto physical =
      lower_for_test(snapshot, 71U, profiles, {}, {}, {}, nullptr, device_bindings, partitions);
  auto state = published_physical_plan_state(std::move(physical), 71U, placements);
  state.execution_partitions = partitions;
  const OperatorRef owner{.id = 1U, .plan_generation = 71U, .run_generation = 0U};
  state.aligned_triggers.push_back(AlignedTriggerPlanRecord{
      .owner = owner,
      .members = {{.input = PortId{3U},
                   .event_clock = ClockId{1U},
                   .queue_capacity = 3U,
                   .cache_capacity = 3U}},
      .inclusive_window_ns = 10U,
      .comparison_clock = ClockId{1U},
      .closure_proofs = {AlignedClosureProofKind::kStreamAdvancement},
  });
  state.clock_triggers.push_back(ClockTriggerPlanRecord{
      .owner = owner,
      .clock = ClockId{1U},
      .period_ns = 1000U,
  });
  state.max_start_delays.push_back(MaxStartDelayPlanRecord{
      .owner = owner,
      .budget_ns = 2000U,
      .source_clock = ClockId{1U},
  });
  state.readable_triggers.push_back(ReadableTriggerPlanRecord{
      .owner = owner,
      .source = make_source_id(1U),
      .plan_generation = 71U,
      .witness_clock = ClockId{1U},
  });
  state.missing_aggregates.push_back(MissingAggregatePlanRecord{
      .owner = owner,
      .inputs = {{.input = PortId{3U},
                  .selector = InputSelectorKind::kExact,
                  .missing = TemporalRecordMissingKind::kSkip}},
  });
  return state;
}

// Emits a host Tensor output (with allocation bounds) and a scalar output.

}  // namespace holoscan::detail::physical_plan_test

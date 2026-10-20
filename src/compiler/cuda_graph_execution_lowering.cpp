/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cuda_graph_execution_lowering.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/core/domain/dlpack_utils.hpp>

#include "compiler/execution_plan_identity.hpp"
#include "compiler/execution_plan_lookup.hpp"
#include "core/canonical_text_encoding.hpp"
#include "core/plan_record_contracts.hpp"

namespace holoscan::detail {
namespace {

template <typename Records>
[[nodiscard]] bool has_unique_region_keys(const Records& records) {
  std::vector<std::pair<std::uint64_t, std::uint64_t>> keys;
  keys.reserve(records.size());
  for (const auto& record : records) {
    keys.emplace_back(record.graph.value, record.region.value);
  }
  std::sort(keys.begin(), keys.end());
  return std::adjacent_find(keys.begin(), keys.end()) == keys.end();
}

[[nodiscard]] std::string region_pointer(GraphId graph, RegionId region) {
  return "/graphs/" + std::to_string(graph.value - 1U) + "/regions/" +
         std::to_string(region.value - 1U) + "/constraints/cuda_graph";
}

[[nodiscard]] bool published_execution_errors_exactly_match(
    const ExecutionPlanState& state, std::span<const PlanLoweringDiagnostic> expected) {
  std::vector<PlanLoweringDiagnostic> published;
  for (const auto& diagnostic : state.diagnostics) {
    if (diagnostic.severity != DiagnosticSeverity::kError ||
        diagnostic.token != "CUDA_GRAPH_EXECUTION_REGION_INVALID") {
      continue;
    }
    published.push_back(PlanLoweringDiagnostic{
        .token = diagnostic.token,
        .source = diagnostic.source,
        .json_pointer = diagnostic.json_pointer,
        .required = diagnostic.required,
        .provided = diagnostic.provided,
    });
  }
  canonicalize_plan_lowering_diagnostics(published);
  return published.size() == expected.size() && std::equal(published.begin(),
                                                           published.end(),
                                                           expected.begin(),
                                                           plan_lowering_diagnostic_exactly_equal);
}

[[nodiscard]] bool error_belongs_to_capture_plan(const PlanLoweringDiagnostic& error,
                                                 const CUDAGraphCapturePlanRecord& capture_plan) {
  const std::string expected_pointer = region_pointer(capture_plan.graph, capture_plan.region);
  return error.source.graph == capture_plan.graph && error.json_pointer == expected_pointer;
}

// Records why a validated region cannot be converted into a fused execution operator.
void reject(CUDAGraphExecutionLoweringResult& result, GraphGeneration graph_generation,
            const CUDAGraphCapturePlanRecord& capture_plan, std::string_view rule,
            std::string provided, VertexId vertex = {}, PortId port = {}) {
  result.errors.push_back(PlanLoweringDiagnostic{
      .token = "CUDA_GRAPH_EXECUTION_REGION_INVALID",
      .source = DiagnosticSource{.graph_generation = graph_generation,
                                 .graph = capture_plan.graph,
                                 .vertex = vertex,
                                 .port = port},
      .json_pointer = region_pointer(capture_plan.graph, capture_plan.region),
      .required = std::string(rule),
      .provided = "region_id=" + std::to_string(capture_plan.region.value) +
                  ",region_name=" + percent_encode_uri_component(capture_plan.region_name) + "," +
                  std::move(provided),
  });
}

[[nodiscard]] std::string member_partition_evidence(
    const ExecutionPlanState& state, const CUDAGraphCapturePlanRecord& capture_plan) {
  std::string evidence{"member_partitions="};
  for (const auto& member : capture_plan.members) {
    if (evidence.back() != '=') {
      evidence += ";";
    }
    evidence += std::to_string(member.vertex.value) + ":";
    bool found{};
    for (const auto& partition : state.execution_partitions) {
      const bool contains_member =
          std::binary_search(partition.members.begin(),
                             partition.members.end(),
                             member.vertex,
                             [](VertexId lhs, VertexId rhs) { return lhs.value < rhs.value; });
      if (!contains_member) {
        continue;
      }
      if (found) {
        evidence += "|";
      }
      evidence += std::to_string(partition.id.value);
      found = true;
    }
    if (!found) {
      evidence += "none";
    }
  }
  return evidence;
}

[[nodiscard]] std::string member_process_group_evidence(
    const ExecutionPlanState& state, const CUDAGraphCapturePlanRecord& capture_plan) {
  std::string evidence{"reason=members_span_process_groups,member_process_groups="};
  for (const auto& member : capture_plan.members) {
    if (evidence.back() != '=') {
      evidence += ";";
    }
    evidence += std::to_string(member.vertex.value) + ":";
    bool found{};
    for (const auto& placement : state.vertex_process_placements) {
      if (placement.vertex != member.vertex) {
        continue;
      }
      if (found) {
        evidence += "|";
      }
      evidence += placement.process_group ? std::to_string(placement.process_group.value) : "none";
      found = true;
    }
    if (!found) {
      evidence += "none";
    }
  }
  return evidence;
}

struct BufferLoweringResult {
  std::optional<CUDAGraphCaptureBufferPlanRecord> buffer;
  std::string_view rule{cuda_graph_execution_rule::kFixedTensorBuffer};
  std::string failure;
  VertexId offending_vertex;
  PortId offending_port;
};

[[nodiscard]] constexpr std::string_view disposition_name(PlanDisposition disposition) noexcept {
  switch (disposition) {
    case PlanDisposition::kAccepted:
      return "kAccepted";
    case PlanDisposition::kRejected:
      return "kRejected";
    case PlanDisposition::kExplicitFallback:
      return "kExplicitFallback";
  }
  return "invalid";
}

[[nodiscard]] constexpr std::string_view tensor_output_storage_name(
    TensorOutputStorage storage) noexcept {
  switch (storage) {
    case TensorOutputStorage::kPassThrough:
      return "kPassThrough";
    case TensorOutputStorage::kRuntimePool:
      return "kRuntimePool";
  }
  return "invalid";
}

[[nodiscard]] constexpr std::string_view capture_buffer_kind_name(
    CUDAGraphCaptureBufferKind kind) noexcept {
  switch (kind) {
    case CUDAGraphCaptureBufferKind::kIngress:
      return "ingress";
    case CUDAGraphCaptureBufferKind::kInternal:
      return "internal";
    case CUDAGraphCaptureBufferKind::kEgress:
      return "egress";
  }
  return "invalid";
}

// A complete maximum shape becomes exact only for an output feeding a CUDA Graph capture edge.
// Ordinary Tensor lowering continues to interpret the same contract axes as upper bounds.
[[nodiscard]] std::optional<std::vector<std::uint64_t>> cuda_graph_fixed_shape(
    const PortPlanRecord& source) {
  if (!source.tensor_contract || !source.tensor_contract->representation.rank ||
      source.tensor_contract->axes.size() != *source.tensor_contract->representation.rank) {
    return std::nullopt;
  }
  std::vector<std::uint64_t> result;
  result.reserve(source.tensor_contract->axes.size());
  for (const TensorAxisBounds& axis : source.tensor_contract->axes) {
    if (axis.max_extent == 0U ||
        axis.max_extent > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
      return std::nullopt;
    }
    result.push_back(axis.max_extent);
  }
  return result;
}

[[nodiscard]] std::optional<std::uint64_t> cuda_graph_fixed_byte_span(
    DLDataType dtype, std::span<const std::uint64_t> fixed_shape) noexcept {
  std::uint64_t byte_span = dlpack_dtype_item_size_bytes(dtype);
  if (byte_span == 0U) {
    return std::nullopt;
  }
  for (const std::uint64_t extent : fixed_shape) {
    if (extent > std::numeric_limits<std::uint64_t>::max() / byte_span) {
      return std::nullopt;
    }
    byte_span *= extent;
  }
  return byte_span;
}

// Return evidence when the canonical contiguous layout of a fixed shape exceeds an authored
// per-axis stride limit. A zero limit is unconstrained, matching ordinary Tensor allocation.
[[nodiscard]] std::optional<std::string> cuda_graph_fixed_stride_failure(
    DLDataType dtype, std::span<const std::uint64_t> fixed_shape,
    std::span<const TensorAxisBounds> axes) {
  if (axes.size() < fixed_shape.size()) {
    return "reason=fixed_shape_stride_bounds_incomplete,fixed_rank=" +
           std::to_string(fixed_shape.size()) + ",axis_bounds=" + std::to_string(axes.size());
  }
  std::uint64_t stride = dlpack_dtype_item_size_bytes(dtype);
  if (stride == 0U) {
    return "reason=fixed_shape_stride_invalid_dtype";
  }
  for (std::size_t reverse = fixed_shape.size(); reverse > 0U; --reverse) {
    const std::size_t axis = reverse - 1U;
    const std::uint64_t maximum = axes[axis].max_abs_stride_bytes;
    if (maximum != 0U && stride > maximum) {
      return "reason=fixed_shape_exceeds_stride_bound,axis=" + std::to_string(axis) +
             ",required_byte_stride=" + std::to_string(stride) +
             ",max_abs_stride_bytes=" + std::to_string(maximum);
    }
    if (fixed_shape[axis] > std::numeric_limits<std::uint64_t>::max() / stride) {
      return "reason=fixed_shape_stride_overflow,axis=" + std::to_string(axis);
    }
    stride *= fixed_shape[axis];
  }
  return std::nullopt;
}

// Resolve one region edge to the ordinary allocation plan whose size and placement will be used
// to materialize the fixed CUDA Graph allocation.
// Keep the ordered rejection checklist beside the resulting buffer record so source and
// target diagnostics retain their precedence and the admitted allocation can be audited locally.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] BufferLoweringResult lower_buffer(const ExecutionPlanState& state,
                                                const CUDAGraphCapturePlanRecord& capture_plan,
                                                const CUDAGraphCaptureEdgeRecord& edge,
                                                CUDAGraphCaptureBufferKind kind,
                                                ProcessGroupId region_process_group) {
  const auto source_failure = [&](std::string failure,
                                  std::string_view rule =
                                      cuda_graph_execution_rule::kFixedTensorBuffer) {
    return BufferLoweringResult{.rule = rule,
                                .failure = std::move(failure),
                                .offending_vertex = edge.source,
                                .offending_port = edge.source_port};
  };
  const auto target_failure = [&](std::string failure,
                                  std::string_view rule =
                                      cuda_graph_execution_rule::kFixedTensorBuffer) {
    return BufferLoweringResult{.rule = rule,
                                .failure = std::move(failure),
                                .offending_vertex = edge.target,
                                .offending_port = edge.target_port};
  };
  const PortPlanRecord* source = find_port_by_id(state, edge.source_port);
  if (source == nullptr) {
    return source_failure("reason=source_port_missing");
  }
  const PortPlanRecord* target = find_port_by_id(state, edge.target_port);
  if (target == nullptr) {
    return target_failure("reason=target_port_missing");
  }
  const auto physical_edge = std::find_if(state.payload_edges.begin(),
                                          state.payload_edges.end(),
                                          [&](const auto& item) { return item.edge == edge.edge; });
  if (physical_edge == state.payload_edges.end()) {
    return source_failure("reason=payload_edge_missing");
  }
  if (physical_edge->source != edge.source_port || physical_edge->target != edge.target_port) {
    return source_failure("reason=payload_edge_ports_mismatch,physical_source=" +
                          std::to_string(physical_edge->source.value()) +
                          ",physical_target=" + std::to_string(physical_edge->target.value()));
  }
  if (physical_edge->disposition != PlanDisposition::kAccepted) {
    return source_failure("reason=payload_edge_not_accepted,disposition=" +
                          std::string(disposition_name(physical_edge->disposition)));
  }

  // A captured address cannot come from a pass-through Tensor. Runtime will replace this output's
  // ordinary pool binding with the fixed allocation named in the returned record.
  if (!source->tensor_contract ||
      source->tensor_contract->storage != TensorOutputStorage::kRuntimePool) {
    const auto storage = source->tensor_contract
                             ? tensor_output_storage_name(source->tensor_contract->storage)
                             : std::string_view{"missing"};
    return source_failure("reason=producer_output_not_runtime_pool,storage=" + std::string(storage),
                          cuda_graph_execution_rule::kRuntimePoolTensorOutput);
  }
  if (source->max_emits_per_compute != 1U) {
    return source_failure("reason=producer_emit_bound_not_one,max_emits_per_compute=" +
                              std::to_string(source->max_emits_per_compute),
                          cuda_graph_execution_rule::kSingleTensorEmit);
  }
  // A fixed-address pool serializes producer, delivery, and retirement on one slot. Admit only
  // edges whose logical contract allows that serialization.
  if (physical_edge->queue_depth != 1U || physical_edge->max_inflight_samples != 1U ||
      physical_edge->cache_capacity != 0U || source->late_joiner_history_depth != 0U) {
    return source_failure(
        "reason=fixed_address_requires_single_live_sample,queue_depth=" +
            std::to_string(physical_edge->queue_depth) +
            ",max_inflight_samples=" + std::to_string(physical_edge->max_inflight_samples) +
            ",cache_capacity=" + std::to_string(physical_edge->cache_capacity) +
            ",history_depth=" + std::to_string(source->late_joiner_history_depth),
        cuda_graph_execution_rule::kSingleFixedTensorSlot);
  }

  // Region topology validation already rejects side outputs from internal members and multiple
  // egress edges from the sink, so those captured outputs each have one consumer. The ingress
  // producer is outside the region and is not covered by those checks.
  // Here, we enforce its specific output port to feed only the region source before replacing the
  // pool allocation with fixed CUDA Graph input storage. Support for sharing
  // that ingress output with other consumers will be explored later.
  if (kind == CUDAGraphCaptureBufferKind::kIngress) {
    const std::uint32_t ingress_output_consumer_count = physical_edge->fanout;
    if (ingress_output_consumer_count != 1U) {
      return source_failure("reason=ingress_output_has_other_consumers,fanout=" +
                                std::to_string(ingress_output_consumer_count),
                            cuda_graph_execution_rule::kExclusiveIngressOutput);
    }
  }

  const TensorAllocationPlanRecord* allocation =
      find_tensor_allocation_by_output(state, edge.source_port);
  if (allocation == nullptr) {
    return source_failure("reason=allocation_missing");
  }
  // The producer and its allocation must agree exactly. The consumer may leave properties
  // unconstrained, but every property it declares must be satisfied by that allocation.
  if (allocation->disposition != PlanDisposition::kAccepted) {
    return source_failure("reason=allocation_not_accepted,disposition=" +
                          std::string(disposition_name(allocation->disposition)));
  }
  if (allocation->bounds.max_rank == 0U || allocation->bounds.max_byte_span == 0U) {
    return source_failure("reason=allocation_bounds_not_fixed,max_rank=" +
                          std::to_string(allocation->bounds.max_rank) +
                          ",max_byte_span=" + std::to_string(allocation->bounds.max_byte_span));
  }
  if (!allocation->layout || !allocation->layout->memory_kind || !allocation->layout->dtype ||
      !allocation->layout->rank || !allocation->layout->strides) {
    return source_failure("reason=allocation_layout_incomplete");
  }
  if (*allocation->layout->strides != TensorStridePolicy::kContiguousOnly) {
    return source_failure("reason=capture_requires_contiguous_tensor_layout");
  }
  if (kind == CUDAGraphCaptureBufferKind::kIngress && *allocation->layout->rank == 0U) {
    return source_failure("reason=scalar_capture_ingress_unsupported",
                          cuda_graph_execution_rule::kExactTensorShape);
  }
  const auto fixed_shape = cuda_graph_fixed_shape(*source);
  if (!fixed_shape) {
    return source_failure(
        "reason=producer_exact_shape_missing,buffer_kind=" +
            std::string(capture_buffer_kind_name(kind)) +
            ",rank=" + std::to_string(*allocation->layout->rank) +
            ",axis_bounds=" + std::to_string(source->tensor_contract->axes.size()),
        cuda_graph_execution_rule::kExactTensorShape);
  }
  const bool allocation_covers_fixed_shape =
      allocation->axes.size() >= fixed_shape->size() &&
      std::equal(fixed_shape->begin(),
                 fixed_shape->end(),
                 allocation->axes.begin(),
                 [](std::uint64_t extent, const TensorAxisBounds& axis) {
                   return extent <= axis.max_extent;
                 });
  if (!allocation_covers_fixed_shape) {
    return source_failure("reason=allocation_does_not_cover_exact_shape,buffer_kind=" +
                              std::string(capture_buffer_kind_name(kind)) +
                              ",fixed_rank=" + std::to_string(fixed_shape->size()) +
                              ",axis_bounds=" + std::to_string(allocation->axes.size()),
                          cuda_graph_execution_rule::kExactTensorShape);
  }
  if (allocation->axes.size() != *allocation->layout->rank ||
      std::ranges::any_of(allocation->axes,
                          [](const TensorAxisBounds& axis) { return axis.max_extent == 0U; })) {
    return source_failure(
        "reason=capture_shape_missing,buffer_kind=" + std::string(capture_buffer_kind_name(kind)) +
            ",rank=" + std::to_string(*allocation->layout->rank) +
            ",axis_bounds=" + std::to_string(allocation->axes.size()),
        cuda_graph_execution_rule::kCompleteTensorShapeBounds);
  }
  const auto fixed_byte_span = cuda_graph_fixed_byte_span(*allocation->layout->dtype, *fixed_shape);
  if (!fixed_byte_span) {
    return source_failure("reason=fixed_shape_byte_span_overflow_or_invalid_dtype,buffer_kind=" +
                              std::string(capture_buffer_kind_name(kind)),
                          cuda_graph_execution_rule::kExactTensorShape);
  }
  if (*fixed_byte_span > allocation->bounds.max_byte_span) {
    return source_failure("reason=fixed_shape_exceeds_byte_capacity,buffer_kind=" +
                              std::string(capture_buffer_kind_name(kind)) +
                              ",required_byte_span=" + std::to_string(*fixed_byte_span) +
                              ",max_byte_span=" + std::to_string(allocation->bounds.max_byte_span),
                          cuda_graph_execution_rule::kExactTensorShape);
  }
  if (const auto failure = cuda_graph_fixed_stride_failure(
          *allocation->layout->dtype, *fixed_shape, source->tensor_contract->axes)) {
    return source_failure(*failure + ",buffer_kind=" + std::string(capture_buffer_kind_name(kind)),
                          cuda_graph_execution_rule::kExactTensorShape);
  }
  if (const auto failure = cuda_graph_fixed_stride_failure(
          *allocation->layout->dtype, *fixed_shape, allocation->axes)) {
    return source_failure(*failure + ",buffer_kind=" + std::string(capture_buffer_kind_name(kind)),
                          cuda_graph_execution_rule::kExactTensorShape);
  }
  if (allocation->axes != source->tensor_contract->axes) {
    return source_failure("reason=allocation_source_axis_bounds_mismatch,producer_axis_bounds=" +
                              std::to_string(source->tensor_contract->axes.size()) +
                              ",allocation_axis_bounds=" + std::to_string(allocation->axes.size()),
                          cuda_graph_execution_rule::kExactTensorShape);
  }
  if (allocation->layout->memory_kind != allocation->memory_kind) {
    return source_failure(
        "reason=allocation_memory_mismatch,allocation_memory=" +
        std::string(memory_kind_name(allocation->memory_kind)) +
        ",layout_memory=" + std::string(memory_kind_name(*allocation->layout->memory_kind)));
  }
  if (allocation->pool_id.empty()) {
    return source_failure("reason=pool_id_missing");
  }
  if (allocation->allocator_root_id.empty()) {
    return source_failure("reason=allocator_root_id_missing");
  }
  // Runtime materializes and accesses the fixed allocation in the fused operator's process. This
  // also rejects an ingress allocation owned by an upstream operator in another process.
  if (allocation->allocator_process_group != region_process_group) {
    return source_failure("reason=allocation_process_group_mismatch,allocator_process_group=" +
                              std::to_string(allocation->allocator_process_group.value) +
                              ",region_process_group=" + std::to_string(region_process_group.value),
                          cuda_graph_execution_rule::kSingleProcessGroup);
  }
  if (kind == CUDAGraphCaptureBufferKind::kEgress) {
    const auto* consumer_placement =
        find_unique_vertex_process_placement_by_vertex(state, target->owner);
    if (consumer_placement == nullptr) {
      return target_failure("reason=egress_consumer_process_group_missing_or_ambiguous",
                            cuda_graph_execution_rule::kSingleProcessGroup);
    }
    if (consumer_placement->process_group != region_process_group) {
      return target_failure(
          "reason=egress_consumer_process_group_mismatch,consumer_process_group=" +
              std::to_string(consumer_placement->process_group.value) +
              ",region_process_group=" + std::to_string(region_process_group.value),
          cuda_graph_execution_rule::kSingleProcessGroup);
    }
  }
  if (!payload_identity_exactly_equal(source->payload_identity, target->payload_identity)) {
    return target_failure("reason=payload_identity_mismatch");
  }
  if (!tensor_port_layout_exactly_equal(allocation->layout, source->tensor_layout)) {
    return source_failure("reason=allocation_source_layout_mismatch");
  }
  if (!tensor_allocation_satisfies_input(*allocation, target->tensor_layout)) {
    return target_failure("reason=allocation_does_not_satisfy_input");
  }
  if (target->tensor_contract && !target->tensor_contract->axes.empty()) {
    const bool consumer_accepts_fixed_shape =
        target->tensor_contract->axes.size() >= fixed_shape->size() &&
        std::equal(fixed_shape->begin(),
                   fixed_shape->end(),
                   target->tensor_contract->axes.begin(),
                   [](std::uint64_t extent, const TensorAxisBounds& axis) {
                     return extent <= axis.max_extent;
                   });
    if (!consumer_accepts_fixed_shape) {
      return target_failure(
          "reason=consumer_does_not_accept_exact_shape,buffer_kind=" +
              std::string(capture_buffer_kind_name(kind)) +
              ",fixed_rank=" + std::to_string(fixed_shape->size()) +
              ",axis_bounds=" + std::to_string(target->tensor_contract->axes.size()),
          cuda_graph_execution_rule::kExactTensorShape);
    }
    if (const auto failure = cuda_graph_fixed_stride_failure(
            *allocation->layout->dtype, *fixed_shape, target->tensor_contract->axes)) {
      return target_failure(
          *failure + ",buffer_kind=" + std::string(capture_buffer_kind_name(kind)),
          cuda_graph_execution_rule::kExactTensorShape);
    }
  }
  return {
      .buffer =
          CUDAGraphCaptureBufferPlanRecord{
              .kind = kind,
              .edge = edge.edge,
              .source_port = edge.source_port,
              .target_port = edge.target_port,
              .payload_identity = source->payload_identity,
              .layout = *allocation->layout,
              .bounds = allocation->bounds,
              .axes = allocation->axes,
              .fixed_shape = *fixed_shape,
              .active_slot_count = 1U,
              .aggregate_byte_capacity = allocation->bounds.max_byte_span,
              .memory_kind = allocation->memory_kind,
              .device = allocation->device,
              .fixed_allocation_id = cuda_graph_fixed_allocation_id(
                  state.plan_generation, capture_plan.graph, edge.edge),
              .pool_id = allocation->pool_id,
              .allocator_root_id = allocation->allocator_root_id,
          },
  };
}

[[nodiscard]] std::string device_list(std::vector<DeviceId> devices) {
  std::sort(devices.begin(), devices.end(), [](DeviceId lhs, DeviceId rhs) {
    return lhs.value < rhs.value;
  });
  devices.erase(std::unique(devices.begin(), devices.end()), devices.end());
  if (devices.empty()) {
    return "none";
  }
  std::string result;
  for (const DeviceId device : devices) {
    if (!result.empty()) {
      result += "|";
    }
    result += std::to_string(device.value);
  }
  return result;
}

struct SourceStreamResolution {
  const CudaComputeStreamPlanRecord* stream{};
  std::string failure;
};

// Resolve and validate the edge-owned stream for a region with internal edges.
[[nodiscard]] SourceStreamResolution internal_source_stream_for(
    const ExecutionPlanState& state, const CUDAGraphCapturePlanRecord& capture_plan,
    const ExecutionPartitionRecord& partition, ProcessGroupId process_group, DeviceId device) {
  // The compiler stream plan owns the edge-to-stream relation. The region does not create a new
  // stream or infer one from device placement alone.
  const auto* selected = find_cuda_compute_stream_by_fused_edge(
      state, process_group, device, capture_plan.internal_edges.front().edge);
  if (selected == nullptr) {
    return {.failure = "reason=internal_edge_stream_missing,edge=" +
                       std::to_string(capture_plan.internal_edges.front().edge.value)};
  }
  if (selected->stream_id.empty()) {
    return {.failure = "reason=internal_edge_stream_id_empty,edge=" +
                       std::to_string(capture_plan.internal_edges.front().edge.value)};
  }
  if (selected->partition && selected->partition != partition.id) {
    return {.failure = "reason=edge_stream_partition_mismatch,edge=" +
                       std::to_string(capture_plan.internal_edges.front().edge.value) +
                       ",stream_partition=" + std::to_string(selected->partition->value) +
                       ",region_partition=" + std::to_string(partition.id.value)};
  }
  // Compare the remaining edges directly with the source edge's stream. Edge IDs provide stable
  // rejection evidence because stream IDs change when the plan generation is rebound.
  std::string mismatched_edges = std::to_string(capture_plan.internal_edges.front().edge.value);
  bool stream_mismatch{};
  for (std::size_t index = 1U; index < capture_plan.internal_edges.size(); ++index) {
    const auto& edge = capture_plan.internal_edges[index];
    const auto* edge_stream =
        find_cuda_compute_stream_by_fused_edge(state, process_group, device, edge.edge);
    if (edge_stream == selected) {
      continue;
    }
    stream_mismatch = true;
    mismatched_edges += "|" + std::to_string(edge.edge.value);
  }
  if (stream_mismatch) {
    return {.failure =
                "reason=internal_edge_streams_do_not_match,edges=" + std::move(mismatched_edges) +
                ",requirement=all_internal_edges_use_region_source_stream"};
  }
  return {.stream = selected};
}

// Select the stream for the fused operator. For a multi-member region, the first internal edge
// identifies the stream and every remaining internal edge must resolve to the same stream.
// Boundary-edge streams synchronize ingress and egress; they do not select the capture stream.
[[nodiscard]] SourceStreamResolution source_stream_for(
    const ExecutionPlanState& state, const CUDAGraphCapturePlanRecord& capture_plan,
    const ExecutionPartitionRecord& partition, ProcessGroupId process_group, DeviceId device) {
  if (!capture_plan.internal_edges.empty()) {
    return internal_source_stream_for(state, capture_plan, partition, process_group, device);
  }

  // A single-member region has no internal edge-to-stream relation. Use its partition-owned
  // operator stream when one exists.
  if (const auto* partition_stream =
          find_cuda_compute_stream_by_partition(state, process_group, device, partition.id)) {
    if (partition_stream->stream_id.empty()) {
      return {.failure = "reason=partition_operator_stream_id_empty,partition=" +
                         std::to_string(partition.id.value)};
    }
    return {.stream = partition_stream};
  }

  // Without a partition stream, only the process/device Tensor-resource stream can serve the
  // operator. This partitionless ordinary stream is not operator-owned: CUDA plan validation
  // requires its owner to be the process/device allocator root and admits at most one such owner.
  // A partitionless Green Context stream belongs to a compiler service, not operator execution.
  std::vector<const CudaComputeStreamPlanRecord*> ordinary_streams;
  std::size_t green_context_service_streams{};
  for (const auto& stream : state.cuda_compute_streams) {
    if (stream.process_group != process_group || stream.device != device || stream.partition) {
      continue;
    }
    if (stream.green_context_id) {
      ++green_context_service_streams;
      continue;
    }
    if (!stream.stream_id.empty()) {
      ordinary_streams.push_back(&stream);
    }
  }
  if (ordinary_streams.size() == 1U) {
    return {.stream = ordinary_streams.front()};
  }
  if (ordinary_streams.empty()) {
    return {.failure =
                "reason=operator_stream_missing,partition_stream=none,ordinary_streams=none,"
                "green_context_service_streams=" +
                std::to_string(green_context_service_streams)};
  }
  return {.failure = "reason=multiple_ordinary_operator_streams,stream_count=" +
                     std::to_string(ordinary_streams.size())};
}

struct RegionDeviceResolution {
  std::optional<DeviceId> device;
  std::string failure;
};

// Require device-backed Tensor buffers and partition-owned operator streams to resolve to one CUDA
// device. Conflicting devices reject the region.
[[nodiscard]] RegionDeviceResolution resolved_region_device(
    std::span<const CUDAGraphCaptureBufferPlanRecord> buffers, const ExecutionPlanState& state,
    const ExecutionPartitionRecord& partition, ProcessGroupId process_group) {
  std::vector<DeviceId> buffer_devices;
  for (const auto& buffer : buffers) {
    if (buffer.device.valid()) {
      buffer_devices.push_back(buffer.device);
    }
  }

  // Partition-owned streams confirm buffer placement and establish placement for an edge-less
  // operator. Ordinary process/device streams do not establish operator placement.
  std::vector<DeviceId> partition_stream_devices;
  for (const auto& stream : state.cuda_compute_streams) {
    if (stream.process_group != process_group || stream.partition != partition.id ||
        stream.stream_id.empty() || !stream.device.valid()) {
      continue;
    }
    partition_stream_devices.push_back(stream.device);
  }

  std::vector<DeviceId> all_devices = buffer_devices;
  all_devices.insert(
      all_devices.end(), partition_stream_devices.begin(), partition_stream_devices.end());
  std::sort(all_devices.begin(), all_devices.end(), [](DeviceId lhs, DeviceId rhs) {
    return lhs.value < rhs.value;
  });
  all_devices.erase(std::unique(all_devices.begin(), all_devices.end()), all_devices.end());
  if (all_devices.size() == 1U) {
    return {.device = all_devices.front()};
  }

  const std::string evidence =
      "buffer_devices=" + device_list(std::move(buffer_devices)) +
      ",partition_stream_devices=" + device_list(std::move(partition_stream_devices));
  if (all_devices.empty()) {
    return {.failure = "reason=no_cuda_device_evidence," + evidence};
  }
  return {.failure = "reason=conflicting_cuda_devices," + evidence};
}

// Retained artifacts are validated independently, so enforce the nonempty plan invariant even
// though topology validation establishes it during normal compilation.
[[nodiscard]] bool validate_capture_plan_has_members(
    CUDAGraphExecutionLoweringResult& result, GraphGeneration graph_generation,
    const CUDAGraphCapturePlanRecord& capture_plan) {
  if (!capture_plan.members.empty()) {
    return true;
  }

  reject(result,
         graph_generation,
         capture_plan,
         cuda_graph_execution_rule::kNonemptyCapturePlan,
         "members=0");
  return false;
}

// Captured callbacks cannot observe a new Parameter generation or be reconstructed by a lifecycle
// restart. Require immutable member state before producing a reusable executable graph plan.
[[nodiscard]] bool validate_immutable_capture_state(
    CUDAGraphExecutionLoweringResult& result, GraphGeneration graph_generation,
    const ExecutionPlanState& state, const CUDAGraphCapturePlanRecord& capture_plan) {
  for (const auto& member : capture_plan.members) {
    const auto* parameters = find_operator_parameters_by_owner(state, member.operator_ref);
    const bool updates_enabled = parameters != nullptr && (parameters->runtime_updates_enabled ||
                                                           parameters->lifecycle_updates_enabled);
    const bool restart_enabled =
        std::ranges::any_of(state.lifecycle_restart_targets,
                            [&](const auto& target) {
                              return target.target == member.operator_ref &&
                                     target.disposition ==
                                         LifecycleRestartTargetDisposition::kAdmitted;
                            }) ||
        std::ranges::any_of(state.lifecycle_restart_closures, [&](const auto& closure) {
          const auto target = std::ranges::find(state.lifecycle_restart_targets,
                                                closure.target,
                                                &LifecycleRestartTargetPlanRecord::target);
          return target != state.lifecycle_restart_targets.end() &&
                 target->disposition == LifecycleRestartTargetDisposition::kAdmitted &&
                 std::ranges::find(closure.closure, member.operator_ref) != closure.closure.end();
        });
    if (parameters != nullptr && !updates_enabled && !restart_enabled) {
      continue;
    }
    reject(result,
           graph_generation,
           capture_plan,
           cuda_graph_execution_rule::kImmutableCaptureState,
           parameters == nullptr ? "reason=parameter_plan_missing"
           : updates_enabled     ? "reason=parameter_updates_enabled"
                                 : "reason=lifecycle_restart_enabled",
           member.vertex);
    return false;
  }
  return true;
}

// Every region member must have exactly one membership in the same execution partition. The fused
// operator inherits the source member's partition and execution lane.
[[nodiscard]] const ExecutionPartitionRecord* resolve_shared_execution_partition(
    CUDAGraphExecutionLoweringResult& result, GraphGeneration graph_generation,
    const ExecutionPlanState& state, const CUDAGraphCapturePlanRecord& capture_plan,
    VertexId source) {
  const ExecutionPartitionRecord* partition =
      find_unique_execution_partition_by_member(state, source);
  const bool all_members_share_partition =
      partition != nullptr &&
      std::all_of(
          capture_plan.members.begin(), capture_plan.members.end(), [&](const auto& member) {
            return find_unique_execution_partition_by_member(state, member.vertex) == partition;
          });
  if (all_members_share_partition) {
    return partition;
  }

  reject(result,
         graph_generation,
         capture_plan,
         cuda_graph_execution_rule::kSingleExecutionPartition,
         member_partition_evidence(state, capture_plan),
         source);
  return nullptr;
}

// The fused operator executes in one Runtime process, so all region members must resolve to the
// same process group.
[[nodiscard]] std::optional<ProcessGroupId> resolve_shared_process_group(
    CUDAGraphExecutionLoweringResult& result, GraphGeneration graph_generation,
    const ExecutionPlanState& state, const CUDAGraphCapturePlanRecord& capture_plan,
    VertexId source) {
  const auto* source_placement = find_unique_vertex_process_placement_by_vertex(state, source);
  std::optional<ProcessGroupId> process_group;
  if (source_placement != nullptr && source_placement->process_group) {
    process_group = source_placement->process_group;
  }
  const bool all_members_share_process =
      process_group &&
      std::all_of(
          capture_plan.members.begin(), capture_plan.members.end(), [&](const auto& member) {
            const auto* placement =
                find_unique_vertex_process_placement_by_vertex(state, member.vertex);
            return placement != nullptr && placement->process_group == process_group;
          });
  if (all_members_share_process) {
    return process_group;
  }

  reject(result,
         graph_generation,
         capture_plan,
         cuda_graph_execution_rule::kSingleProcessGroup,
         member_process_group_evidence(state, capture_plan),
         source);
  return std::nullopt;
}

// Resolve the region's logical edges to fixed Tensor allocations. Buffer order is ingress,
// source-to-sink internal edges, then egress, and is retained in the immutable execution plan.
[[nodiscard]] std::optional<std::vector<CUDAGraphCaptureBufferPlanRecord>>
lower_fixed_tensor_buffers(CUDAGraphExecutionLoweringResult& result,
                           GraphGeneration graph_generation, const ExecutionPlanState& state,
                           const CUDAGraphCapturePlanRecord& capture_plan,
                           ProcessGroupId region_process_group) {
  std::vector<CUDAGraphCaptureBufferPlanRecord> buffers;
  buffers.reserve(capture_plan.internal_edges.size() +
                  static_cast<std::size_t>(capture_plan.ingress.has_value()) +
                  static_cast<std::size_t>(capture_plan.egress.has_value()));
  bool buffer_invalid{};
  const auto append_buffer = [&](const CUDAGraphCaptureEdgeRecord& edge,
                                 CUDAGraphCaptureBufferKind kind) {
    auto lowered = lower_buffer(state, capture_plan, edge, kind, region_process_group);
    if (!lowered.buffer) {
      reject(result,
             graph_generation,
             capture_plan,
             lowered.rule,
             "edge=" + std::to_string(edge.edge.value) + "," + std::move(lowered.failure),
             lowered.offending_vertex,
             lowered.offending_port);
      buffer_invalid = true;
      return;
    }
    buffers.push_back(std::move(*lowered.buffer));
  };
  if (capture_plan.ingress) {
    append_buffer(*capture_plan.ingress, CUDAGraphCaptureBufferKind::kIngress);
  }
  for (const auto& edge : capture_plan.internal_edges) {
    append_buffer(edge, CUDAGraphCaptureBufferKind::kInternal);
  }
  if (capture_plan.egress) {
    append_buffer(*capture_plan.egress, CUDAGraphCaptureBufferKind::kEgress);
  }
  if (buffer_invalid) {
    return std::nullopt;
  }
  return buffers;
}

// All physical buffers and the partition-owned stream must establish one CUDA device. An edge-less
// region without a partition stream has no authoritative device and is rejected.
[[nodiscard]] std::optional<DeviceId> resolve_single_cuda_device(
    CUDAGraphExecutionLoweringResult& result, GraphGeneration graph_generation,
    const ExecutionPlanState& state, const CUDAGraphCapturePlanRecord& capture_plan,
    const ExecutionPartitionRecord& partition, ProcessGroupId process_group,
    std::span<const CUDAGraphCaptureBufferPlanRecord> buffers, VertexId source) {
  auto resolution = resolved_region_device(buffers, state, partition, process_group);
  if (resolution.device) {
    return resolution.device;
  }

  reject(result,
         graph_generation,
         capture_plan,
         cuda_graph_execution_rule::kSingleCUDADevice,
         std::move(resolution.failure),
         source);
  return std::nullopt;
}

// A Tensor passed between captured operators must stay in CUDA device memory.
[[nodiscard]] bool validate_internal_buffer_memory(CUDAGraphExecutionLoweringResult& result,
                                                   GraphGeneration graph_generation,
                                                   const ExecutionPlanState& state,
                                                   const CUDAGraphCapturePlanRecord& capture_plan,
                                                   const CUDAGraphCaptureBufferPlanRecord& buffer) {
  if (buffer.memory_kind == MemoryKind::kCudaDevice) {
    return true;
  }

  const auto* source_port = find_port_by_id(state, buffer.source_port);
  reject(result,
         graph_generation,
         capture_plan,
         cuda_graph_execution_rule::kInternalCUDADeviceBuffer,
         "edge=" + std::to_string(buffer.edge.value) +
             ",memory_kind=" + std::string(memory_kind_name(buffer.memory_kind)) +
             ",device=" + std::to_string(buffer.device.value),
         source_port != nullptr ? source_port->owner : VertexId{},
         buffer.source_port);
  return false;
}

// A region boundary accepts CUDA device memory or pinned host memory.
[[nodiscard]] bool validate_boundary_buffer_memory(CUDAGraphExecutionLoweringResult& result,
                                                   GraphGeneration graph_generation,
                                                   const ExecutionPlanState& state,
                                                   const CUDAGraphCapturePlanRecord& capture_plan,
                                                   const CUDAGraphCaptureBufferPlanRecord& buffer) {
  const bool capture_compatible_memory = buffer.memory_kind == MemoryKind::kCudaDevice ||
                                         buffer.memory_kind == MemoryKind::kPinnedHost;
  if (capture_compatible_memory) {
    return true;
  }

  const auto* source_port = find_port_by_id(state, buffer.source_port);
  reject(result,
         graph_generation,
         capture_plan,
         cuda_graph_execution_rule::kCaptureCompatibleBoundaryBuffer,
         "edge=" + std::to_string(buffer.edge.value) +
             ",memory_kind=" + std::string(memory_kind_name(buffer.memory_kind)) +
             ",device=" + std::to_string(buffer.device.value),
         source_port != nullptr ? source_port->owner : VertexId{},
         buffer.source_port);
  return false;
}

// Require CUDA device memory for internal buffers. Boundary buffers may additionally use pinned
// host memory for capture-compatible ingress and egress.
[[nodiscard]] bool validate_capture_buffer_memory(
    CUDAGraphExecutionLoweringResult& result, GraphGeneration graph_generation,
    const ExecutionPlanState& state, const CUDAGraphCapturePlanRecord& capture_plan,
    std::span<const CUDAGraphCaptureBufferPlanRecord> buffers) {
  for (const auto& buffer : buffers) {
    bool valid{};
    if (buffer.kind == CUDAGraphCaptureBufferKind::kInternal) {
      valid =
          validate_internal_buffer_memory(result, graph_generation, state, capture_plan, buffer);
    } else {
      valid =
          validate_boundary_buffer_memory(result, graph_generation, state, capture_plan, buffer);
    }
    if (!valid) {
      return false;
    }
  }
  return true;
}

// The source member's planned stream becomes the fused operator's launch stream. Every internal
// edge must already be assigned to that stream.
[[nodiscard]] const CudaComputeStreamPlanRecord* resolve_source_cuda_stream(
    CUDAGraphExecutionLoweringResult& result, GraphGeneration graph_generation,
    const ExecutionPlanState& state, const CUDAGraphCapturePlanRecord& capture_plan,
    const ExecutionPartitionRecord& partition, ProcessGroupId process_group, DeviceId device,
    VertexId source) {
  auto resolution = source_stream_for(state, capture_plan, partition, process_group, device);
  if (resolution.stream != nullptr) {
    return resolution.stream;
  }

  reject(result,
         graph_generation,
         capture_plan,
         cuda_graph_execution_rule::kSourceCUDAStream,
         std::move(resolution.failure),
         source);
  return nullptr;
}

}  // namespace

CUDAGraphExecutionLoweringResult finalize_cuda_graph_execution_regions(
    GraphGeneration graph_generation, const ExecutionPlanState& state) {
  CUDAGraphExecutionLoweringResult result;
  result.regions.reserve(state.cuda_graph_capture_plans.size());
  for (const auto& capture_plan : state.cuda_graph_capture_plans) {
    // Require a nonempty capture plan before resolving its source member.
    if (!validate_capture_plan_has_members(result, graph_generation, capture_plan)) {
      continue;
    }

    // Require state used during capture to remain unchanged for the RunSession.
    if (!validate_immutable_capture_state(result, graph_generation, state, capture_plan)) {
      continue;
    }

    // Require every capture member to share one execution partition.
    const VertexId source = capture_plan.members.front().vertex;
    const auto* partition =
        resolve_shared_execution_partition(result, graph_generation, state, capture_plan, source);
    if (partition == nullptr) {
      continue;
    }

    // Require every capture member to share one Runtime process group.
    const auto process_group =
        resolve_shared_process_group(result, graph_generation, state, capture_plan, source);
    if (!process_group) {
      continue;
    }

    // Resolve every region edge to one fixed Tensor allocation.
    auto buffers =
        lower_fixed_tensor_buffers(result, graph_generation, state, capture_plan, *process_group);
    if (!buffers) {
      continue;
    }

    // Require internal and boundary buffers to use capture-compatible memory kinds.
    if (!validate_capture_buffer_memory(result, graph_generation, state, capture_plan, *buffers)) {
      continue;
    }

    // Require the region buffers and partition stream to identify one CUDA device.
    const auto device = resolve_single_cuda_device(result,
                                                   graph_generation,
                                                   state,
                                                   capture_plan,
                                                   *partition,
                                                   *process_group,
                                                   *buffers,
                                                   source);
    if (!device) {
      continue;
    }

    // Require one planned CUDA stream to carry every internal region edge.
    const auto* stream = resolve_source_cuda_stream(
        result, graph_generation, state, capture_plan, *partition, *process_group, *device, source);
    if (stream == nullptr) {
      continue;
    }

    // The accepted record contains every physical identity needed to construct the fused operator
    // during RunSession startup.
    result.regions.push_back(CUDAGraphExecutionRegionPlanRecord{
        .graph = capture_plan.graph,
        .region = capture_plan.region,
        .fused_operator = source,
        .partition = partition->id,
        .execution_lane = partition->execution_lane,
        .process_group = *process_group,
        .device = *device,
        .stream_id = stream->stream_id,
        .green_context_id = stream->green_context_id,
        .ingress = capture_plan.ingress,
        .egress = capture_plan.egress,
        .buffers = std::move(*buffers),
    });
  }
  canonicalize_plan_lowering_diagnostics(result.errors);
  return result;
}

bool valid_cuda_graph_execution_regions(const ExecutionPlanState& state) {
  // Re-run physical admission from immutable plan records instead of trusting serialized
  // execution-region records individually. Graph generation supplies diagnostic provenance only.
  const auto expected = finalize_cuda_graph_execution_regions(state.graph_identity.version, state);
  if (!has_unique_region_keys(state.cuda_graph_capture_plans) ||
      !has_unique_region_keys(state.cuda_graph_execution_regions)) {
    return false;
  }
  if (expected.regions.size() != state.cuda_graph_execution_regions.size() ||
      !std::equal(expected.regions.begin(),
                  expected.regions.end(),
                  state.cuda_graph_execution_regions.begin())) {
    return false;
  }
  // Every capture plan must result in one fused region or at least one published rejection. A
  // single rejected plan may report more than one invalid buffer.
  for (const auto& capture_plan : state.cuda_graph_capture_plans) {
    const bool has_region =
        std::any_of(expected.regions.begin(), expected.regions.end(), [&](const auto& region) {
          return region.graph == capture_plan.graph && region.region == capture_plan.region;
        });
    const bool has_error =
        std::any_of(expected.errors.begin(), expected.errors.end(), [&](const auto& error) {
          return error_belongs_to_capture_plan(error, capture_plan);
        });
    if (has_region == has_error) {
      return false;
    }
  }
  return published_execution_errors_exactly_match(state, expected.errors);
}

std::optional<CUDAGraphRuntimeChannelPlanBackup> remove_cuda_graph_internal_runtime_channels(
    ExecutionPlanState& state) {
  if (state.cuda_graph_execution_regions.empty()) {
    return std::nullopt;  // Ordinary graphs incur no channel backup or filtering work.
  }
  // Identify the services owned by captured internal edges before changing the channel plan.
  std::vector<std::string> removed_service_ids;
  for (const auto& subscriber : state.process_local_channel_subscribers) {
    if (is_cuda_graph_internal_execution_edge(state, subscriber.edge)) {
      removed_service_ids.push_back(subscriber.service_id);
    }
  }
  std::sort(removed_service_ids.begin(), removed_service_ids.end());
  removed_service_ids.erase(std::unique(removed_service_ids.begin(), removed_service_ids.end()),
                            removed_service_ids.end());
  const auto service_was_internal = [&](std::string_view service_id) {
    return std::binary_search(removed_service_ids.begin(), removed_service_ids.end(), service_id);
  };

  // Region topology gives each internal source port one consumer. Leave the channel plan intact if
  // retained compiler records violate that invariant; artifact validation will reject the plan.
  const bool shared_internal_service =
      std::any_of(state.process_local_channel_subscribers.begin(),
                  state.process_local_channel_subscribers.end(),
                  [&](const auto& subscriber) {
                    return service_was_internal(subscriber.service_id) &&
                           !is_cuda_graph_internal_execution_edge(state, subscriber.edge);
                  });
  if (shared_internal_service) {
    return std::nullopt;
  }

  // Preserve the complete ordinary channel plan so rejected-artifact recovery can undo fusion.
  CUDAGraphRuntimeChannelPlanBackup backup{
      .services = state.process_local_channel_services,
      .publishers = state.process_local_channel_publishers,
      .subscribers = state.process_local_channel_subscribers,
  };
  std::erase_if(state.process_local_channel_subscribers, [&](const auto& subscriber) {
    return is_cuda_graph_internal_execution_edge(state, subscriber.edge);
  });
  // Region validation gives every internal source port exactly one consumer. Removing that
  // consumer therefore removes the complete source-port service and publisher.
  std::erase_if(state.process_local_channel_publishers,
                [&](const auto& publisher) { return service_was_internal(publisher.service_id); });
  std::erase_if(state.process_local_channel_services,
                [&](const auto& service) { return service_was_internal(service.service_id); });
  return backup;
}

void discard_cuda_graph_capture_and_execution_plan(
    ExecutionPlanState& state, std::optional<CUDAGraphRuntimeChannelPlanBackup>&& channel_backup) {
  state.cuda_graph_capture_plans.clear();
  state.cuda_graph_execution_regions.clear();
  if (channel_backup) {
    state.process_local_channel_services = std::move(channel_backup->services);
    state.process_local_channel_publishers = std::move(channel_backup->publishers);
    state.process_local_channel_subscribers = std::move(channel_backup->subscribers);
  }
}

}  // namespace holoscan::detail

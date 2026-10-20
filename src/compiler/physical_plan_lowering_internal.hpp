/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "compiler/detail/checked_arithmetic.hpp"
#include "compiler/physical_plan_lowering.hpp"
#include "compiler/snapshot_lookup.hpp"
#include "core/tensor_contract_resolution_internal.hpp"

namespace holoscan::detail {

inline constexpr std::string_view kTensorOverflow{"PHYSICAL_TENSOR_CAPACITY_OVERFLOW"};
inline constexpr std::string_view kTensorLayoutMismatch{"PHYSICAL_TENSOR_LAYOUT_MISMATCH"};
inline constexpr std::string_view kQueueInvalid{"PHYSICAL_QUEUE_CAPACITY_INVALID"};
inline constexpr std::string_view kServiceInvalid{"PHYSICAL_SERVICE_CONTRACT_INVALID"};
inline constexpr std::string_view kTensorDeviceBindingInvalid{
    "PHYSICAL_TENSOR_DEVICE_BINDING_INVALID"};
inline constexpr std::string_view kTensorProviderUnsupported{
    "PHYSICAL_TENSOR_PROVIDER_UNSUPPORTED"};
inline constexpr std::string_view kSchemaTensorFieldUnbounded{
    "CHANNEL_SCHEMA_TENSOR_FIELD_UNBOUNDED"};
inline constexpr std::string_view kSchemaTensorFieldUnsupportedMemory{
    "CHANNEL_SCHEMA_TENSOR_FIELD_UNSUPPORTED_MEMORY"};
inline constexpr std::string_view kProcessLocalChannelInvalid{"PROCESS_LOCAL_CHANNEL_PLAN_INVALID"};
inline constexpr std::string_view kInlineProviderUnsupported{
    "PHYSICAL_TENSOR_INLINE_PROVIDER_UNSUPPORTED"};
inline constexpr std::string_view kTensorMemoryKindUnresolved{
    "PHYSICAL_TENSOR_MEMORY_KIND_UNRESOLVED"};
inline constexpr std::string_view kTensorProcessGroupUnresolved{
    "PHYSICAL_TENSOR_PROCESS_GROUP_UNRESOLVED"};
inline constexpr std::string_view kResourceCoverInvalid{"PARTITION_RESOURCE_REALIZATION_FAILED"};

inline void reject_physical_plan(PhysicalPlanLoweringResult& output, std::string_view token,
                                 DiagnosticSource source = {}, std::string json_pointer = {},
                                 std::string required = {}, std::string provided = {},
                                 std::optional<DiagnosticDetail> detail = std::nullopt) {
  output.errors.push_back(PhysicalPlanDiagnostic{
      .token = std::string(token),
      .source = source,
      .json_pointer = std::move(json_pointer),
      .required = std::move(required),
      .provided = std::move(provided),
      .detail = std::move(detail),
  });
}

inline DiagnosticDetail physical_edge_detail(const GraphSnapshotData& snapshot,
                                             const PortRecord* source, const PortRecord* target,
                                             std::vector<DiagnosticMismatch> mismatches = {}) {
  DiagnosticDetail detail{
      .operation = DiagnosticOperation::kCompile,
      .error = Error{ErrorCode::kIncompatible, "incompatible connection contract"}};
  const auto path = [&](const PortRecord* port) {
    const auto* owner = port != nullptr ? find_vertex(snapshot, port->owner) : nullptr;
    std::string result;
    if (owner == nullptr) {
      return result;
    }
    const auto* graph = find_graph(snapshot, owner->graph);
    if (graph != nullptr && graph->path) {
      for (const auto& segment : graph->path->segments()) {
        append_percent_encoded_uri_path_segment(result, segment);
      }
    }
    append_percent_encoded_uri_path_segment(result, owner->name);
    return result;
  };
  detail.operator_path = path(target);
  detail.related_operator_path = path(source);
  if (target != nullptr) {
    detail.port_name = target->name;
  }
  if (source != nullptr) {
    const auto* owner = find_vertex(snapshot, source->owner);
    detail.related_source = DiagnosticSource{.graph_generation = snapshot.generation,
                                             .graph = owner != nullptr ? owner->graph : GraphId{},
                                             .vertex = source->owner,
                                             .port = PortId{source->id.value}};
    detail.related_port_name = source->name;
  }
  detail.truncated = mismatches.size() > 16U;
  if (detail.truncated) {
    mismatches.resize(16U);
  }
  detail.mismatches = std::move(mismatches);
  return detail;
}

template <typename Enum>
[[nodiscard]] constexpr bool enum_at_most(Enum value, Enum maximum) noexcept {
  using Underlying = std::underlying_type_t<Enum>;
  return static_cast<Underlying>(value) <= static_cast<Underlying>(maximum);
}

[[nodiscard]] inline DiagnosticSource source_for(const GraphSnapshotData& snapshot,
                                                 VertexId vertex = {}, PortId port = {}) noexcept {
  return DiagnosticSource{.graph_generation = snapshot.generation, .vertex = vertex, .port = port};
}

[[nodiscard]] inline std::optional<TensorContractRecord> resolved_tensor_contract(
    const PortRecord& port) {
  return resolve_output_tensor_contract(port.setup.tensor_contract,
                                        port.graph_override.tensor_contract);
}

[[nodiscard]] inline std::optional<TensorRepresentation> resolved_layout(const PortRecord& port) {
  const auto contract = resolved_tensor_contract(port);
  if (contract) {
    return contract->representation;
  }
  return port.graph_override.tensor_layout ? port.graph_override.tensor_layout
                                           : port.setup.tensor_layout;
}

[[nodiscard]] inline bool is_tensor_backed_payload(const PayloadIdentityRecord& payload) noexcept {
  return payload.kind == PayloadIdentityKind::kTensor ||
         (payload.kind == PayloadIdentityKind::kSchema &&
          payload.payload_kind == PayloadKind::kDescriptorBacked);
}

[[nodiscard]] inline std::optional<ProcessGroupId> process_group_for(
    std::span<const VertexProcessPlacementPlanRecord> placements, VertexId vertex) noexcept {
  const auto found = std::find_if(placements.begin(), placements.end(), [&](const auto& placement) {
    return placement.vertex == vertex;
  });
  if (found == placements.end() || !found->process_group) {
    return std::nullopt;
  }
  return found->process_group;
}

[[nodiscard]] inline bool requires_explicit_device(MemoryKind memory) noexcept {
  return memory == MemoryKind::kCudaDevice || memory == MemoryKind::kCudaManaged;
}

[[nodiscard]] inline const TensorByteSpanDerivationPlanRecord* find_tensor_byte_span_derivation(
    std::span<const TensorByteSpanDerivationPlanRecord> derivations, PortId output,
    std::optional<std::uint16_t> field_id = std::nullopt) noexcept {
  const auto found =
      std::find_if(derivations.begin(), derivations.end(), [&](const auto& candidate) {
        return candidate.output == output && candidate.field_id == field_id;
      });
  return found != derivations.end() ? &*found : nullptr;
}

[[nodiscard]] inline const TensorByteSpanDerivationPlanRecord*
find_port_tensor_byte_span_derivation(
    std::span<const TensorByteSpanDerivationPlanRecord> derivations,
    const PortRecord& port) noexcept {
  const bool singular_descriptor_tensor =
      port.payload_identity.kind == PayloadIdentityKind::kSchema &&
      port.payload_identity.payload_kind == PayloadKind::kDescriptorBacked &&
      port.descriptor_tensor_field_count == 1U;
  if (singular_descriptor_tensor) {
    const auto found =
        std::find_if(derivations.begin(), derivations.end(), [&](const auto& candidate) {
          return candidate.output.value() == port.id.value && candidate.field_id.has_value();
        });
    return found != derivations.end() ? &*found : nullptr;
  }
  return find_tensor_byte_span_derivation(derivations, PortId{port.id.value});
}

[[nodiscard]] inline std::string port_pointer(PortId port, std::string_view suffix) {
  return "/ports/" + std::to_string(port.value()) + std::string(suffix);
}

[[nodiscard]] inline std::vector<const TensorFieldConfigurationRecord*> canonical_tensor_fields(
    const std::vector<TensorFieldConfigurationRecord>& fields) {
  std::vector<const TensorFieldConfigurationRecord*> canonical;
  canonical.reserve(fields.size());
  for (const auto& field : fields) {
    canonical.push_back(&field);
  }
  std::sort(canonical.begin(), canonical.end(), [](const auto* lhs, const auto* rhs) {
    return lhs->field_id < rhs->field_id;
  });
  return canonical;
}

[[nodiscard]] constexpr bool is_graph_internal_locality(
    const ChannelLocalityKind locality) noexcept {
  // Process-group identity remains authoritative after placement reconciliation.
  return locality == ChannelLocalityKind::kProcessLocal ||
         locality == ChannelLocalityKind::kCrossProcess;
}

/// Return the Tensor-backed outputs with an edge into another process group of this application,
/// or exported from a placed process. HoloMQ's pinned-host generation protocol uses CUDA stream
/// memory operations, so such an output's pinned-host storage carries a private synchronization
/// device on every service, local ones included.
[[nodiscard]] inline std::unordered_set<std::uint64_t> cross_process_tensor_sources(
    std::span<const PayloadEdgePlanRecord> edges, const GraphSnapshotData& snapshot,
    std::span<const VertexProcessPlacementPlanRecord> process_placements) {
  std::unordered_set<std::uint64_t> sources;
  sources.reserve(edges.size());
  for (const auto& edge : edges) {
    if (!is_tensor_backed_payload(edge.payload_identity)) {
      continue;
    }
    const auto* source = find_port(snapshot, GraphPortId{edge.source.value()});
    if (edge.locality == ChannelLocalityKind::kExternalExport) {
      if (source != nullptr && process_group_for(process_placements, source->owner)) {
        sources.insert(edge.source.value());
      }
      continue;
    }
    if (!is_graph_internal_locality(edge.locality)) {
      continue;
    }
    const auto* target = find_port(snapshot, GraphPortId{edge.target.value()});
    if (source == nullptr || target == nullptr) {
      continue;
    }
    const auto source_group = process_group_for(process_placements, source->owner);
    const auto target_group = process_group_for(process_placements, target->owner);
    if (source_group && target_group && source_group != target_group) {
      sources.insert(edge.source.value());
    }
  }
  return sources;
}

struct TensorClaimContributor {
  std::string_view name;
  std::uint32_t maximum{};
  std::vector<TensorSlotClaimReason> reasons;
};

[[nodiscard]] inline std::optional<std::uint32_t> append_tensor_claim_contributors(
    std::string_view pool_id, std::span<const TensorClaimContributor> contributors,
    std::vector<PoolCapacityContributorRecord>& output) {
  std::uint32_t total = 0U;
  for (const auto& contributor : contributors) {
    if (contributor.maximum == 0U || contributor.reasons.empty()) {
      continue;
    }
    std::uint32_t next = 0U;
    if (!checked_add(total, contributor.maximum, next)) {
      return std::nullopt;
    }
    total = next;
    output.push_back(PoolCapacityContributorRecord{
        .pool_id = std::string(pool_id),
        .contributor_id = std::string(pool_id) + "." + std::string(contributor.name),
        .maximum_distinct_slots = contributor.maximum,
        .reasons = contributor.reasons,
    });
  }
  return total;
}

[[nodiscard]] inline std::size_t accepted_cover_count(
    std::span<const ExecutionPartitionRecord> partitions, VertexId owner) noexcept {
  std::size_t count = 0U;
  for (const auto& partition : partitions) {
    count += static_cast<std::size_t>(
        std::count(partition.members.begin(), partition.members.end(), owner));
  }
  return count;
}

}  // namespace holoscan::detail

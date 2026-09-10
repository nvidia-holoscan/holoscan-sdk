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
#include <utility>
#include <vector>

#include "compiler/detail/checked_arithmetic.hpp"
#include "compiler/physical_plan_lowering.hpp"
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
                                 std::string required = {}, std::string provided = {}) {
  output.errors.push_back(PhysicalPlanDiagnostic{
      .token = std::string(token),
      .source = source,
      .json_pointer = std::move(json_pointer),
      .required = std::move(required),
      .provided = std::move(provided),
  });
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

[[nodiscard]] constexpr std::string_view memory_kind_name(MemoryKind memory) noexcept {
  switch (memory) {
    case MemoryKind::kUnknown:
      return "kUnknown";
    case MemoryKind::kHost:
      return "kHost";
    case MemoryKind::kPinnedHost:
      return "kPinnedHost";
    case MemoryKind::kCudaDevice:
      return "kCudaDevice";
    case MemoryKind::kCudaManaged:
      return "kCudaManaged";
  }
  return "invalid";
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

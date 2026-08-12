/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/graph.hpp>

namespace holoscan::detail::partitioning {

using VertexIndex = std::uint32_t;
using EdgeIndex = std::uint32_t;

/// Identify one candidate partition within an untrusted planner proposal.
struct CandidatePartitionId {
  std::uint32_t value{};
  [[nodiscard]] explicit constexpr operator bool() const noexcept { return value != 0U; }
  friend constexpr bool operator==(CandidatePartitionId, CandidatePartitionId) noexcept = default;
};

/// Identify one compiler-normalized mandatory execution domain.
struct ExecutionDomainKey {
  std::uint64_t value{};
  friend constexpr bool operator==(ExecutionDomainKey, ExecutionDomainKey) noexcept = default;
};

/// Describe the finite normalized facts that force an edge to cross a partition boundary.
enum class PartitionBoundary : std::uint32_t {
  kNone = 0U,
  kProcess = 1U << 0U,
  kExternalEndpoint = 1U << 1U,
  kDeviceTransfer = 1U << 2U,
  kRegionBackend = 1U << 3U,
  kExplicitThreadPlacement = 1U << 4U,
  kLifecycle = 1U << 5U,
  kReadiness = 1U << 6U,
};

/// Carry a normalized set of mandatory partition-boundary reasons.
struct PartitionBoundaryMask {
  std::uint32_t bits{};

  [[nodiscard]] constexpr bool empty() const noexcept { return bits == 0U; }
  [[nodiscard]] constexpr bool contains(PartitionBoundary boundary) const noexcept {
    return (bits & static_cast<std::uint32_t>(boundary)) != 0U;
  }
  constexpr void add(PartitionBoundary boundary) noexcept {
    bits |= static_cast<std::uint32_t>(boundary);
  }
  constexpr void add(PartitionBoundaryMask other) noexcept { bits |= other.bits; }
  friend constexpr bool operator==(PartitionBoundaryMask, PartitionBoundaryMask) noexcept = default;
};

/// Enumerate normalized compiler facts that a private planner may require.
enum class PartitionPlannerCapability : std::uint32_t {
  kExecutionDomains = 1U << 0U,
  kBoundaryMasks = 1U << 1U,
  kCollapseLegality = 1U << 2U,
  kIndependentReadinessRoots = 1U << 3U,
};

/// Carry a finite capability set for planner registration and invocation checks.
struct PartitionPlannerCapabilities {
  std::uint32_t bits{};

  [[nodiscard]] constexpr bool contains(PartitionPlannerCapability capability) const noexcept {
    return (bits & static_cast<std::uint32_t>(capability)) != 0U;
  }
  [[nodiscard]] constexpr bool contains(PartitionPlannerCapabilities required) const noexcept {
    return (bits & required.bits) == required.bits;
  }
  constexpr void add(PartitionPlannerCapability capability) noexcept {
    bits |= static_cast<std::uint32_t>(capability);
  }
  friend constexpr bool operator==(PartitionPlannerCapabilities,
                                   PartitionPlannerCapabilities) noexcept = default;
};

inline constexpr std::uint32_t kPartitionPlannerInterfaceVersion = 1U;
inline constexpr std::string_view kSerialBaselineAlgorithmId = "holoscan.partition.serial_baseline";
inline constexpr std::string_view kSourceChainAlgorithmId = "holoscan.partition.source_chain";
inline constexpr std::uint32_t kPartitionAlgorithmVersion = 1U;

/// Return the one exact EA1 developer-preview environment variable name.
[[nodiscard]] constexpr std::string_view partition_environment_variable() noexcept {
  return "HOLOSCAN_EXPERIMENTAL_PARTITION_ALGORITHM";
}

/// Return every normalized fact implemented by the EA1 compiler.
[[nodiscard]] constexpr PartitionPlannerCapabilities all_partition_planner_capabilities() noexcept {
  PartitionPlannerCapabilities capabilities;
  capabilities.add(PartitionPlannerCapability::kExecutionDomains);
  capabilities.add(PartitionPlannerCapability::kBoundaryMasks);
  capabilities.add(PartitionPlannerCapability::kCollapseLegality);
  capabilities.add(PartitionPlannerCapability::kIndependentReadinessRoots);
  return capabilities;
}

inline constexpr auto kAllPartitionPlannerCapabilities = all_partition_planner_capabilities();

/// Return the owning public evidence key for the EA1 production baseline.
[[nodiscard]] inline AlgorithmKey serial_baseline_algorithm() {
  return AlgorithmKey{std::string(kSerialBaselineAlgorithmId), kPartitionAlgorithmVersion};
}

/// Return the owning public evidence key for the EA1 experimental source-chain planner.
[[nodiscard]] inline AlgorithmKey source_chain_algorithm() {
  return AlgorithmKey{std::string(kSourceChainAlgorithmId), kPartitionAlgorithmVersion};
}

/// Describe one private planner implementation and the normalized facts it consumes.
struct PartitionPlannerDescriptor {
  AlgorithmKey algorithm;
  std::uint32_t interface_version{};
  AlgorithmMaturity maturity{AlgorithmMaturity::kProduction};
  PartitionPlannerCapabilities required_capabilities;
};

/// Describe one executable vertex in the compiler-normalized partition problem.
struct PartitionVertexView {
  VertexId id{};
  std::uint32_t canonical_order{};
  std::span<const EdgeIndex> incoming_edges;
  std::span<const EdgeIndex> outgoing_edges;
  bool starts_execution_chain{};
  ExecutionDomainKey execution_domain;
  PartitionBoundaryMask boundary_before;
  std::span<const RegionId> source_regions;
};

/// Describe one compiler-normalized logical edge and its collapse proof.
struct PartitionEdgeView {
  EdgeId id{};
  VertexIndex source{};
  VertexIndex target{};
  PartitionBoundaryMask mandatory_boundaries;
  bool same_partition_collapse_legal{};
  bool same_stream_fifo_legal{};
};

/// Supply a bounded, pointer-free, immutable partition-planning problem.
struct PartitionProblem {
  std::span<const PartitionVertexView> vertices;
  std::span<const PartitionEdgeView> edges;
  std::uint32_t max_partitions{};
  PartitionPlannerCapabilities available_capabilities;
};

/// Assign one normalized vertex to one candidate partition.
struct VertexPartitionAssignment {
  VertexIndex vertex{};
  CandidatePartitionId partition;
};

/// Explain one candidate assignment without assigning final public plan identities.
struct PartitionDecisionProposal {
  VertexIndex vertex{};
  CandidatePartitionId partition;
  PartitionDecisionKind decision{PartitionDecisionKind::kStart};
  std::vector<PartitionDecisionReason> reasons;
};

/// Carry an untrusted assignment-only proposal from one private planner.
struct PartitionProposal {
  AlgorithmKey algorithm;
  std::vector<VertexPartitionAssignment> assignments;
  std::vector<PartitionDecisionProposal> decisions;
};

/// Carry one validator-assigned canonical partition without exposing runtime authority.
struct CanonicalPartition {
  std::uint32_t canonical_id{};
  std::vector<VertexId> members;
};

/// Carry one canonicalized decision for later translation into public plan evidence.
struct CanonicalPartitionDecision {
  VertexId vertex{};
  std::uint32_t canonical_partition_id{};
  PartitionDecisionKind decision{PartitionDecisionKind::kStart};
  std::vector<PartitionDecisionReason> reasons;
};

/// Carry the validated canonical result consumed by later compiler lowering.
struct ValidatedPartitionPlan {
  AlgorithmKey algorithm;
  std::vector<CanonicalPartition> partitions;
  std::vector<CanonicalPartitionDecision> decisions;
};

}  // namespace holoscan::detail::partitioning

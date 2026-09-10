/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <holoscan/core/execution_plan.hpp>

#include "core/graph_snapshot.hpp"

namespace holoscan::detail {

/// @brief Classify active Parameter mutation admitted by profile policy v1.
enum class ParameterMutationProfileDisposition : std::uint8_t {
  kRuntimeAndLifecycle = 0,  ///< Runtime and lifecycle candidates may be admitted.
  kFrozenOnly = 1,           ///< Only immutable generation-zero values are admitted.
};

/// @brief State whether rate-aware sizing needs bounded-arrival evidence for one vertex.
enum class RateBoundedEvidenceRequirement : std::uint8_t {
  kNotRequired = 0,
  kRequired = 1,
};

inline constexpr std::string_view kProfileValueUnknownToken = "PROFILE_VALUE_UNKNOWN";
inline constexpr std::string_view kProfileRegionLoosensGraphToken = "PROFILE_REGION_LOOSENS_GRAPH";
inline constexpr std::string_view kProfileRealTimeEvidenceMissingToken =
    "PROFILE_CAPABILITY_UNSATISFIED";
inline constexpr std::string_view kProfileRegulatedEvidenceMissingToken =
    "PROFILE_EVIDENCE_UNSATISFIED";

/// Hold the total result of private policy-v1 profile admission.
///
/// Vectors indexed by vertex use `VertexId::value - 1` from the canonical
/// compiler snapshot. An absent profile means that an invalid root or region
/// request prevented resolution for that vertex. Error tokens are sorted and
/// deduplicated stable registry spellings; they contain no diagnostic prose.
struct ProfileAdmissionResult {
  ProfileManifest manifest;
  std::vector<std::optional<Profile>> effective_vertex_profiles;
  std::vector<std::vector<RegionId>> contributing_regions_by_vertex;
  std::vector<RateBoundedEvidenceRequirement> rate_evidence_requirements_by_vertex;
  std::vector<std::string> error_tokens;

  [[nodiscard]] bool accepted() const noexcept {
    return manifest.disposition == ProfileAdmissionDisposition::kAccepted;
  }

  /// Return the canonical per-vertex profile view for downstream lowerers.
  ///
  /// Entry `VertexId::value - 1` belongs to that canonical vertex. An empty
  /// entry means profile resolution for that vertex failed; downstream passes
  /// must not substitute the root request for it.
  [[nodiscard]] std::span<const std::optional<Profile>> effective_profiles() const noexcept {
    return std::span<const std::optional<Profile>>{effective_vertex_profiles};
  }

  /// Return the effective profile for one canonical vertex, when resolved.
  [[nodiscard]] std::optional<Profile> effective_profile(VertexId vertex) const noexcept {
    if (vertex.value == 0U || vertex.value > effective_vertex_profiles.size()) {
      return std::nullopt;
    }
    return effective_vertex_profiles[static_cast<std::size_t>(vertex.value - 1U)];
  }

  /// Return the canonical per-vertex bounded-rate-evidence policy for downstream lowerers.
  [[nodiscard]] std::span<const RateBoundedEvidenceRequirement> rate_evidence_requirements()
      const noexcept {
    return std::span<const RateBoundedEvidenceRequirement>{rate_evidence_requirements_by_vertex};
  }
};

/// Return the exact canonical policy-v1 bytes covered by ProfilePolicyId.
///
/// These private compact JSON bytes are stable compiler-test evidence, not an
/// installed profile-selection or registration surface.
[[nodiscard]] std::string_view profile_policy_canonical_json_v1() noexcept;

/// @brief Derive an identity from owner-approved canonical policy bytes and version.
/// @param canonical_policy Exact canonical UTF-8 policy table, excluding its digest field.
/// @param major Owner-classified policy major version.
/// @param minor Owner-classified policy minor version.
/// @return Version plus deterministic digest of exactly `canonical_policy`.
/// @note This private helper does not infer compatibility from two policy tables. Policy
/// evolution remains an explicit owner decision: compatible rule changes retain the
/// version, additive compatible changes increment minor, and incompatible changes
/// increment major and reset minor.
[[nodiscard]] ProfilePolicyId profile_policy_id_from_canonical(std::string_view canonical_policy,
                                                               std::uint32_t major,
                                                               std::uint32_t minor) noexcept;

/// @brief Return the exact identity of the source-controlled policy-v1 table.
[[nodiscard]] ProfilePolicyId profile_policy_id_v1() noexcept;

/// @brief Return the policy-v1 active Parameter mutation disposition.
/// @param profile Valid closed-set profile value.
/// @return Frozen policy outcome independent of runtime implementation state.
[[nodiscard]] ParameterMutationProfileDisposition parameter_mutation_profile_disposition(
    Profile profile) noexcept;

/// Resolve the closed profile policy over one canonical Graph snapshot.
///
/// This cold, deterministic pass does not choose or rewrite topology,
/// partition, readiness, placement, transport, or runtime policy. It assumes
/// GraphHandle flattening and canonical identity assignment already completed.
[[nodiscard]] ProfileAdmissionResult resolve_profile_admission(const GraphSnapshotData& snapshot,
                                                               Profile requested_profile);

}  // namespace holoscan::detail

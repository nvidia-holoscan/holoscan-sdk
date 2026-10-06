/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compiler/profile_admission.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/core/parameter_schema.hpp>

#include "compiler/detail/canonical_utils.hpp"
#include "compiler/execution_plan_contracts.hpp"
#include "compiler/snapshot_lookup.hpp"

namespace holoscan::detail {
namespace {

// Object keys, rule IDs, profile tables, and scope kinds are in their policy-v1
// canonical orders. Hashing covers these exact UTF-8 bytes without the
// StableHashBuilder string-field delimiter.
constexpr std::string_view kCanonicalPolicyV1 =
    R"({"dominance":["research","standard","robotics","real_time","regulated"],"major":1,"minor":0,"profiles":[{"capabilities":[],"dispositions":["named_relaxations"],"name":"research"},{"capabilities":[],"dispositions":["baseline"],"name":"standard"},{"capabilities":["holoscan.profile.robotics.finite_evidence/v1"],"dispositions":["finite_evidence"],"name":"robotics"},{"capabilities":["holoscan.profile.real_time.bounded_hot_path/v1"],"dispositions":["fail_closed"],"name":"real_time"},{"capabilities":["holoscan.profile.regulated.complete_evidence/v1"],"dispositions":["fail_closed"],"name":"regulated"}],"rules":["holoscan.profile.closed_set/v1","holoscan.profile.contributors.region_only/v1","holoscan.profile.region.tighten_only/v1","holoscan.profile.scope.strictest_max/v1"],"schema":"holoscan.profile_policy.canonical_json.v1","scope_kinds":["boundary","bridge","channel","converter","edge","feature","helper","operator","port"]})";

struct Scope {
  std::optional<Profile> profile;
  std::vector<RegionId> contributors;
};

[[nodiscard]] bool tightens_or_equals(Profile candidate, Profile baseline) noexcept {
  // REQ: COMPILE-003 - semantic dominance is policy data, never enum ordinal order.
  const auto candidate_rank = profile_semantic_rank(candidate);
  const auto baseline_rank = profile_semantic_rank(baseline);
  return candidate_rank && baseline_rank && *candidate_rank >= *baseline_rank;
}

[[nodiscard]] std::optional<Profile> strictest(Profile lhs, Profile rhs) noexcept {
  const auto lhs_rank = profile_semantic_rank(lhs);
  const auto rhs_rank = profile_semantic_rank(rhs);
  if (!lhs_rank || !rhs_rank) {
    return std::nullopt;
  }
  return *lhs_rank < *rhs_rank ? std::optional<Profile>{rhs} : std::optional<Profile>{lhs};
}

void canonicalize_contributors(std::vector<RegionId>& contributors) {
  std::sort(contributors.begin(), contributors.end(), [](RegionId lhs, RegionId rhs) {
    return lhs.value < rhs.value;
  });
  contributors.erase(std::unique(contributors.begin(), contributors.end()), contributors.end());
}

[[nodiscard]] Scope join_scopes(const Scope& lhs, const Scope& rhs) {
  Scope result;
  if (lhs.profile && rhs.profile) {
    result.profile = strictest(*lhs.profile, *rhs.profile);
  }
  result.contributors = lhs.contributors;
  result.contributors.insert(
      result.contributors.end(), rhs.contributors.begin(), rhs.contributors.end());
  canonicalize_contributors(result.contributors);
  return result;
}

[[nodiscard]] bool region_contains_vertex(const GraphSnapshotData& snapshot,
                                          const RegionRecord& region, const VertexRecord& vertex) {
  if (std::find(region.members.begin(), region.members.end(), vertex.id) != region.members.end()) {
    return true;
  }
  return std::any_of(
      region.graph_members.begin(), region.graph_members.end(), [&](GraphId graph_member) {
        return is_descendant_or_self(snapshot, vertex.graph, graph_member);
      });
}

[[nodiscard]] std::optional<std::size_t> vertex_index(const GraphSnapshotData& snapshot,
                                                      VertexId id) noexcept {
  return find_vertex(snapshot, id) == nullptr
             ? std::nullopt
             : std::optional<std::size_t>{static_cast<std::size_t>(id.value - 1U)};
}

[[nodiscard]] std::optional<std::size_t> port_index(const GraphSnapshotData& snapshot,
                                                    GraphPortId id) noexcept {
  return find_port(snapshot, id) == nullptr
             ? std::nullopt
             : std::optional<std::size_t>{static_cast<std::size_t>(id.value - 1U)};
}

[[nodiscard]] std::optional<VertexId> resolved_port_owner(const GraphSnapshotData& snapshot,
                                                          GraphPortId port) {
  const GraphPortId resolved = resolve_interface_port(snapshot, port);
  const auto index = port_index(snapshot, resolved);
  return index ? std::optional<VertexId>{snapshot.ports[*index].owner} : std::nullopt;
}

[[nodiscard]] std::string object_id(std::string_view kind, std::uint64_t value) {
  std::string result{kind};
  result.push_back(':');
  result += std::to_string(value);
  return result;
}

void append_scope(ProfileManifest& manifest, std::string kind, std::string id, const Scope& scope) {
  if (!scope.profile) {
    return;
  }
  manifest.scope_resolutions.push_back(ProfileScopeResolutionRecord{
      .object_id = std::move(id),
      .object_kind = std::move(kind),
      .effective_profile = *scope.profile,
      .contributing_regions = scope.contributors,
  });
}

[[nodiscard]] ProfilePolicyId make_policy_id() noexcept {
  return profile_policy_id_from_canonical(kCanonicalPolicyV1, 1U, 0U);
}

void append_error(std::vector<std::string>& errors, std::string_view token) {
  errors.emplace_back(token);
}

[[nodiscard]] std::vector<std::optional<Profile>> resolve_regions(
    const GraphSnapshotData& snapshot, std::optional<Profile> root_profile,
    ProfileManifest& manifest, std::vector<std::string>& errors) {
  std::vector<std::optional<Profile>> effective_profiles(snapshot.regions.size());
  for (std::size_t index = 0; index < snapshot.regions.size(); ++index) {
    const RegionRecord& region = snapshot.regions[index];
    RegionProfileRecord record;
    record.region_id = region.id;
    record.origin = region.profile ? RegionProfileOrigin::kExplicitRegion
                                   : RegionProfileOrigin::kInheritedGraph;

    const bool request_is_valid =
        !region.profile || profile_semantic_rank(*region.profile).has_value();
    if (region.profile && request_is_valid) {
      record.requested_profile = region.profile;
    } else if (!request_is_valid) {
      append_error(errors, kProfileValueUnknownToken);
    }

    if (root_profile && !region.profile) {
      record.effective_profile = root_profile;
    } else if (root_profile && request_is_valid &&
               tightens_or_equals(*region.profile, *root_profile)) {
      record.effective_profile = region.profile;
    } else if (root_profile && request_is_valid) {
      append_error(errors, kProfileRegionLoosensGraphToken);
    }

    effective_profiles[index] = record.effective_profile;
    manifest.effective_regions.push_back(std::move(record));
  }
  return effective_profiles;
}

[[nodiscard]] std::vector<Scope> make_vertex_scopes(const GraphSnapshotData& snapshot,
                                                    std::optional<Profile> root_profile) {
  std::vector<Scope> scopes(snapshot.vertices.size());
  if (root_profile) {
    for (auto& scope : scopes) {
      scope.profile = root_profile;
    }
  }
  return scopes;
}

void apply_operator_region_scopes(const GraphSnapshotData& snapshot,
                                  std::span<const std::optional<Profile>> effective_region_profiles,
                                  std::optional<Profile> root_profile,
                                  std::vector<Scope>& vertex_scopes) {
  // REQ: COMPILE-003 - Graph-member regions propagate through descendants,
  // every authored region is independently tighten-only, and overlaps use the
  // strictest semantic join without changing canonical topology.
  for (std::size_t vertex_pos = 0; vertex_pos < snapshot.vertices.size(); ++vertex_pos) {
    const VertexRecord& vertex = snapshot.vertices[vertex_pos];
    if (vertex.kind != VertexKind::kOperator || !root_profile) {
      continue;
    }
    Scope& scope = vertex_scopes[vertex_pos];
    for (std::size_t region_pos = 0; region_pos < snapshot.regions.size(); ++region_pos) {
      const RegionRecord& region = snapshot.regions[region_pos];
      if (!region_contains_vertex(snapshot, region, vertex)) {
        continue;
      }
      scope.contributors.push_back(region.id);
      if (!effective_region_profiles[region_pos]) {
        scope.profile.reset();
      } else if (scope.profile) {
        scope.profile = strictest(*scope.profile, *effective_region_profiles[region_pos]);
      }
    }
    canonicalize_contributors(scope.contributors);
  }
}

void apply_alias_scopes(const GraphSnapshotData& snapshot, std::optional<Profile> root_profile,
                        std::vector<Scope>& vertex_scopes) {
  for (std::size_t index = 0; index < snapshot.vertices.size(); ++index) {
    const VertexRecord& vertex = snapshot.vertices[index];
    if (vertex.kind != VertexKind::kGraphInterface) {
      continue;
    }
    Scope alias_scope{.profile = root_profile};
    bool observed_binding = false;
    for (const auto& binding : snapshot.interface_bindings) {
      const auto public_index = port_index(snapshot, binding.public_port);
      if (!public_index || snapshot.ports[*public_index].owner != vertex.id) {
        continue;
      }
      const auto owner = resolved_port_owner(snapshot, binding.internal_port);
      const auto owner_index = owner ? vertex_index(snapshot, *owner) : std::nullopt;
      if (!owner_index) {
        alias_scope.profile.reset();
        observed_binding = true;
      } else {
        alias_scope = observed_binding ? join_scopes(alias_scope, vertex_scopes[*owner_index])
                                       : vertex_scopes[*owner_index];
        observed_binding = true;
      }
    }
    vertex_scopes[index] = std::move(alias_scope);
  }
}

[[nodiscard]] std::optional<GraphPortId> opposite_boundary_port(const GraphSnapshotData& snapshot,
                                                                const EdgeRecord& edge,
                                                                GraphPortId boundary_port) {
  const GraphPortId resolved_boundary = resolve_interface_port(snapshot, boundary_port);
  if (resolve_interface_port(snapshot, edge.source) == resolved_boundary) {
    return edge.target;
  }
  if (resolve_interface_port(snapshot, edge.target) == resolved_boundary) {
    return edge.source;
  }
  return std::nullopt;
}

void apply_boundary_scopes(const GraphSnapshotData& snapshot, std::optional<Profile> root_profile,
                           std::vector<Scope>& vertex_scopes) {
  // Boundary scope is the strictest union of the canonical objects incident
  // to its intrinsic port. A disconnected boundary remains root-only.
  for (const auto& topic : snapshot.external_topics) {
    const auto boundary_index = vertex_index(snapshot, topic.vertex);
    if (!boundary_index) {
      continue;
    }
    Scope boundary_scope{.profile = root_profile};
    for (const auto& edge : snapshot.edges) {
      const auto opposite = opposite_boundary_port(snapshot, edge, topic.graph_port);
      const auto owner = opposite ? resolved_port_owner(snapshot, *opposite) : std::nullopt;
      const auto owner_index = owner ? vertex_index(snapshot, *owner) : std::nullopt;
      if (!owner_index || snapshot.vertices[*owner_index].kind == VertexKind::kImportedTopic ||
          snapshot.vertices[*owner_index].kind == VertexKind::kExportedTopic) {
        continue;
      }
      boundary_scope = join_scopes(boundary_scope, vertex_scopes[*owner_index]);
    }
    vertex_scopes[*boundary_index] = std::move(boundary_scope);
  }
}

void emit_scope_rows(const GraphSnapshotData& snapshot, std::span<const Scope> vertex_scopes,
                     ProfileAdmissionResult& result) {
  ProfileManifest& manifest = result.manifest;
  for (std::size_t index = 0; index < snapshot.vertices.size(); ++index) {
    result.effective_vertex_profiles[index] = vertex_scopes[index].profile;
    result.contributing_regions_by_vertex[index] = vertex_scopes[index].contributors;
    const VertexRecord& vertex = snapshot.vertices[index];
    if (vertex.kind == VertexKind::kOperator) {
      append_scope(
          manifest, "operator", object_id("operator", vertex.id.value), vertex_scopes[index]);
    }
  }

  std::vector<Scope> port_scopes(snapshot.ports.size());
  for (std::size_t index = 0; index < snapshot.ports.size(); ++index) {
    const PortRecord& port = snapshot.ports[index];
    const auto owner = resolved_port_owner(snapshot, port.id);
    const auto owner_index = owner ? vertex_index(snapshot, *owner) : std::nullopt;
    if (owner_index) {
      port_scopes[index] = vertex_scopes[*owner_index];
    }
    append_scope(manifest, "port", object_id("port", port.id.value), port_scopes[index]);
  }

  for (const auto& edge : snapshot.edges) {
    const auto source_index = port_index(snapshot, resolve_interface_port(snapshot, edge.source));
    const auto target_index = port_index(snapshot, resolve_interface_port(snapshot, edge.target));
    if (source_index && target_index) {
      const Scope scope = join_scopes(port_scopes[*source_index], port_scopes[*target_index]);
      append_scope(manifest, "edge", object_id("edge", edge.id.value), scope);
    }
  }

  for (const auto& topic : snapshot.external_topics) {
    const auto index = vertex_index(snapshot, topic.vertex);
    if (index) {
      append_scope(
          manifest, "boundary", object_id("boundary", topic.id.value), vertex_scopes[*index]);
    }
  }

  std::sort(manifest.scope_resolutions.begin(),
            manifest.scope_resolutions.end(),
            [](const auto& lhs, const auto& rhs) {
              return std::pair{lhs.object_kind, lhs.object_id} <
                     std::pair{rhs.object_kind, rhs.object_id};
            });
}

void emit_robotics_critical_records(const GraphSnapshotData& snapshot,
                                    std::span<const std::optional<Profile>> region_profiles,
                                    ProfileManifest& manifest) {
  for (std::size_t index = 0; index < snapshot.regions.size(); ++index) {
    const RegionRecord& region = snapshot.regions[index];
    if (region_profiles[index] != Profile::kRobotics) {
      continue;
    }
    for (const RegionConstraint& constraint : region.constraints) {
      std::string_view suffix;
      switch (constraint.kind()) {
        case RegionConstraintKind::kLatencyBudget:
          suffix = "latency_budget";
          break;
        case RegionConstraintKind::kBoundedMemory:
          suffix = "bounded_memory";
          break;
        case RegionConstraintKind::kStaticTopology:
        case RegionConstraintKind::kDeviceCompatibleOperators:
        case RegionConstraintKind::kCudaGraph:
          continue;
      }
      const std::string reason = "holoscan.profile.robotics_critical." + std::string{suffix};
      const std::string feature = "region:" + std::to_string(region.id.value) + ':' + reason;
      const std::string evidence = reason + "/v1";
      manifest.feature_dispositions.push_back(ProfileFeatureDispositionRecord{
          .feature_id = feature,
          .disposition = ProfileFeatureDisposition::kRestricted,
          .rule_id = reason,
          .evidence_id = evidence,
      });
      manifest.scope_resolutions.push_back(ProfileScopeResolutionRecord{
          .object_id = feature,
          .object_kind = "feature",
          .effective_profile = Profile::kRobotics,
          .contributing_regions = {region.id},
      });
      if (constraint.kind() == RegionConstraintKind::kLatencyBudget) {
        manifest.derived_capacities.push_back(ProfileDerivedCapacityRecord{
            .object_id = "region:" + std::to_string(region.id.value),
            .capacity_id = "latency_budget",
            .limit = static_cast<std::uint64_t>(constraint.latency().count()),
            .unit = "nanoseconds",
        });
      }
    }
  }
  std::sort(manifest.scope_resolutions.begin(),
            manifest.scope_resolutions.end(),
            [](const auto& lhs, const auto& rhs) {
              return std::pair{lhs.object_kind, lhs.object_id} <
                     std::pair{rhs.object_kind, rhs.object_id};
            });
}

[[nodiscard]] bool robotics_region_requires_bounded_rate_evidence(
    const RegionRecord& region) noexcept {
  return std::ranges::any_of(region.constraints, [](const RegionConstraint& constraint) {
    return constraint.kind() == RegionConstraintKind::kLatencyBudget ||
           constraint.kind() == RegionConstraintKind::kBoundedMemory;
  });
}

void resolve_rate_evidence_requirements(const GraphSnapshotData& snapshot,
                                        ProfileAdmissionResult& result) {
  result.rate_evidence_requirements_by_vertex.assign(snapshot.vertices.size(),
                                                     RateBoundedEvidenceRequirement::kNotRequired);
  for (std::size_t index = 0U; index < snapshot.vertices.size(); ++index) {
    const auto profile = result.effective_vertex_profiles[index];
    bool required = profile == Profile::kRealTime || profile == Profile::kRegulated;
    if (!required && profile == Profile::kRobotics) {
      for (const RegionId region_id : result.contributing_regions_by_vertex[index]) {
        const auto region = std::ranges::find(snapshot.regions, region_id, &RegionRecord::id);
        if (region != snapshot.regions.end() &&
            robotics_region_requires_bounded_rate_evidence(*region)) {
          required = true;
          break;
        }
      }
    }
    result.rate_evidence_requirements_by_vertex[index] =
        required ? RateBoundedEvidenceRequirement::kRequired
                 : RateBoundedEvidenceRequirement::kNotRequired;
  }
}

}  // namespace

std::string_view profile_policy_canonical_json_v1() noexcept {
  return kCanonicalPolicyV1;
}

ProfilePolicyId profile_policy_id_from_canonical(std::string_view canonical_policy,
                                                 std::uint32_t major,
                                                 std::uint32_t minor) noexcept {
  StableHashBuilder hash;
  for (const char byte : canonical_policy) {
    hash.add_byte(static_cast<std::uint8_t>(byte));
  }
  const ParamSchemaHash digest = hash.finish();
  ProfilePolicyId result{.major = major, .minor = minor};
  result.digest.value = digest.value;
  return result;
}

ProfilePolicyId profile_policy_id_v1() noexcept {
  return make_policy_id();
}

ParameterMutationProfileDisposition parameter_mutation_profile_disposition(
    Profile profile) noexcept {
  // Policy-v1 admits active mutation only where bounded dynamic control-plane
  // behavior is explicitly part of the profile. Robotics requires additional
  // region-specific evidence that this component does not infer; constrained
  // profiles therefore remain generation-zero-only here.
  switch (profile) {
    case Profile::kStandard:
    case Profile::kResearch:
      return ParameterMutationProfileDisposition::kRuntimeAndLifecycle;
    case Profile::kRobotics:
    case Profile::kRealTime:
    case Profile::kRegulated:
      return ParameterMutationProfileDisposition::kFrozenOnly;
  }
  return ParameterMutationProfileDisposition::kFrozenOnly;
}

ProfileAdmissionResult resolve_profile_admission(const GraphSnapshotData& snapshot,
                                                 Profile requested_profile) {
  ProfileAdmissionResult result;
  ProfileManifest& manifest = result.manifest;
  manifest.profile_policy_id = make_policy_id();
  manifest.plan_schema_version = "holoscan.execution_plan.private.v8";
  result.effective_vertex_profiles.resize(snapshot.vertices.size());
  result.contributing_regions_by_vertex.resize(snapshot.vertices.size());

  std::optional<Profile> root_profile;
  if (!profile_semantic_rank(requested_profile)) {
    append_error(result.error_tokens, kProfileValueUnknownToken);
  } else {
    root_profile = requested_profile;
    manifest.requested_profile = requested_profile;
    manifest.effective_graph_profile = requested_profile;
  }

  const auto effective_region_profiles =
      resolve_regions(snapshot, root_profile, manifest, result.error_tokens);
  std::vector<Scope> vertex_scopes = make_vertex_scopes(snapshot, root_profile);
  apply_operator_region_scopes(snapshot, effective_region_profiles, root_profile, vertex_scopes);
  apply_alias_scopes(snapshot, root_profile, vertex_scopes);
  apply_boundary_scopes(snapshot, root_profile, vertex_scopes);
  emit_scope_rows(snapshot, vertex_scopes, result);
  emit_robotics_critical_records(snapshot, effective_region_profiles, manifest);
  resolve_rate_evidence_requirements(snapshot, result);

  canonicalize_tokens(result.error_tokens);
  manifest.disposition = result.error_tokens.empty() ? ProfileAdmissionDisposition::kAccepted
                                                     : ProfileAdmissionDisposition::kRejected;
  return result;
}

}  // namespace holoscan::detail

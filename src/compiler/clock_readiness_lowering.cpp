/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compiler/clock_readiness_lowering.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "compiler/clock_plan_resolution.hpp"
#include "compiler/snapshot_lookup.hpp"
#include "core/canonical_text_encoding.hpp"
#include "core/diagnostic_registry.hpp"
#include "core/source_id_internal.hpp"

namespace holoscan::detail {
namespace {

[[nodiscard]] Error artifact_error(ErrorCode code, std::string_view token) {
  return Error{.code = code, .message = std::string(token)};
}

[[nodiscard]] constexpr std::string_view readiness_kind_token(ReadinessSourceKind kind) noexcept {
  switch (kind) {
    case ReadinessSourceKind::kReadable:
      return "readable";
    case ReadinessSourceKind::kNotification:
      return "notification";
    case ReadinessSourceKind::kDoorbell:
      return "doorbell";
    case ReadinessSourceKind::kSpin:
      return "spin";
  }
  return "invalid";
}

struct CanonicalReadinessDeclaration {
  ReadinessSourceDeclarationRecord declaration;
  GraphId owner_graph{};
  std::vector<std::string> graph_path;
  std::string owner_name;
  std::string owner_path;
};

[[nodiscard]] expected<CanonicalReadinessDeclaration, Error> canonical_readiness_declaration(
    const GraphSnapshotData& snapshot, const ReadinessSourceDeclarationRecord& declaration) {
  const VertexRecord* owner = find_vertex(snapshot, declaration.owner);
  if (owner == nullptr) {
    return make_unexpected(artifact_error(ErrorCode::kNotFound, "READINESS_OWNER_UNDECLARED"));
  }
  const GraphRecord* graph = find_graph(snapshot, owner->graph);
  if (graph == nullptr) {
    return make_unexpected(artifact_error(ErrorCode::kNotFound, "READINESS_OWNER_UNDECLARED"));
  }

  CanonicalReadinessDeclaration result{
      .declaration = declaration,
      .owner_graph = owner->graph,
      .owner_name = owner->name,
  };
  result.owner_path =
      graph->path.has_value() ? std::string{} : std::string{kRootGraphPathComponent};
  if (graph->path.has_value()) {
    result.graph_path.assign(graph->path->segments().begin(), graph->path->segments().end());
    for (const auto& segment : result.graph_path) {
      if (!result.owner_path.empty()) {
        result.owner_path.push_back('/');
      }
      append_percent_encoded_uri_component(result.owner_path, segment);
    }
  }
  result.owner_path += "/operator/";
  append_percent_encoded_uri_component(result.owner_path, owner->name);
  return result;
}

}  // namespace

namespace {

[[nodiscard]] expected<void, Error> validate_clock_binding_sources(
    const GraphSnapshotData& snapshot) {
  for (const auto& binding : snapshot.clock_bindings) {
    const auto imported = std::find_if(
        snapshot.clock_imports.begin(), snapshot.clock_imports.end(), [&](const auto& row) {
          return row.graph == binding.child_graph && row.key == binding.child_clock;
        });
    if (imported == snapshot.clock_imports.end()) {
      return make_unexpected(
          artifact_error(ErrorCode::kInvalidArgument, "CLOCK_IMPORT_BINDING_INVALID"));
    }
  }
  return {};
}

[[nodiscard]] expected<void, Error> validate_default_clock_inventory(
    const GraphSnapshotData& snapshot) {
  std::vector<GraphId> defaults;
  defaults.reserve(snapshot.clocks.size() + snapshot.clock_imports.size());
  for (const auto& clock : snapshot.clocks) {
    if (clock.is_default) {
      defaults.push_back(clock.graph);
    }
  }
  for (const auto& clock_import : snapshot.clock_imports) {
    if (clock_import.is_default) {
      defaults.push_back(clock_import.graph);
    }
  }
  std::ranges::sort(defaults, {}, &GraphId::value);
  if (std::ranges::adjacent_find(defaults) != defaults.end()) {
    return make_unexpected(artifact_error(ErrorCode::kDuplicateName, "CLOCK_DEFAULT_DUPLICATE"));
  }
  return {};
}

[[nodiscard]] expected<void, Error> validate_clock_import_declarations(
    const GraphSnapshotData& snapshot) {
  for (std::size_t index = 0U; index < snapshot.clock_imports.size(); ++index) {
    const auto& clock_import = snapshot.clock_imports[index];
    if (!clock_import.graph || clock_import.key.is_default() || clock_import.key.name().empty()) {
      return make_unexpected(
          artifact_error(ErrorCode::kInvalidArgument, "CLOCK_IMPORT_DECLARATION_INVALID"));
    }
    const auto duplicate = std::find_if(
        std::next(snapshot.clock_imports.begin(), static_cast<std::ptrdiff_t>(index + 1U)),
        snapshot.clock_imports.end(),
        [&](const auto& candidate) {
          return candidate.graph == clock_import.graph && candidate.key == clock_import.key;
        });
    const bool concrete_collision =
        std::any_of(snapshot.clocks.begin(), snapshot.clocks.end(), [&](const auto& candidate) {
          return candidate.graph == clock_import.graph && candidate.key == clock_import.key;
        });
    if (duplicate != snapshot.clock_imports.end() || concrete_collision) {
      return make_unexpected(artifact_error(ErrorCode::kDuplicateName, "CLOCK_IMPORT_DUPLICATE"));
    }
  }
  return {};
}

[[nodiscard]] expected<void, Error> validate_clock_import_bindings(
    const GraphSnapshotData& snapshot) {
  // Import identity is more specific than default selection. Validate it first so a corrupted
  // record that violates both contracts retains the exact import/collision diagnostic.
  if (auto declarations = validate_clock_import_declarations(snapshot); !declarations) {
    return declarations;
  }
  if (auto defaults = validate_default_clock_inventory(snapshot); !defaults) {
    return defaults;
  }
  for (const auto& clock_import : snapshot.clock_imports) {
    const auto binding_count = static_cast<std::size_t>(std::count_if(
        snapshot.clock_bindings.begin(), snapshot.clock_bindings.end(), [&](const auto& candidate) {
          return candidate.child_graph == clock_import.graph &&
                 candidate.child_clock == clock_import.key;
        }));
    if (binding_count != 1U) {
      return make_unexpected(artifact_error(
          binding_count == 0U ? ErrorCode::kNotFound : ErrorCode::kInvalidArgument,
          binding_count == 0U ? "CLOCK_IMPORT_UNBOUND" : "CLOCK_IMPORT_BINDING_INVALID"));
    }
    const AuthoredClockResolution resolved = resolve_authored_clock(snapshot.graphs,
                                                                    snapshot.clocks,
                                                                    snapshot.clock_imports,
                                                                    snapshot.clock_bindings,
                                                                    clock_import.graph,
                                                                    clock_import.key);
    if (resolved.status == AuthoredClockResolutionStatus::kImportUnbound) {
      return make_unexpected(artifact_error(ErrorCode::kNotFound, "CLOCK_IMPORT_UNBOUND"));
    }
    if (!resolved.resolved()) {
      return make_unexpected(
          artifact_error(ErrorCode::kInvalidArgument, "CLOCK_IMPORT_BINDING_INVALID"));
    }
  }
  return validate_clock_binding_sources(snapshot);
}

[[nodiscard]] expected<void, Error> append_clock_declarations(
    const GraphSnapshotData& snapshot, ClockAndReadinessLoweringResult& result) {
  result.clocks.reserve(snapshot.clocks.size());
  for (std::size_t index = 0U; index < snapshot.clocks.size(); ++index) {
    const auto& declaration = snapshot.clocks[index];
    const auto capability = std::find_if(snapshot.clock_capabilities.begin(),
                                         snapshot.clock_capabilities.end(),
                                         [&](const auto& candidate) {
                                           return candidate.graph == declaration.graph &&
                                                  candidate.key == declaration.key;
                                         });
    if (declaration.key.is_default() || declaration.key.name().empty()) {
      return make_unexpected(
          artifact_error(ErrorCode::kInvalidArgument, "CLOCK_DECLARATION_INVALID"));
    }
    if (find_clock_plan_id(result.clocks, declaration.graph, declaration.key).valid()) {
      return make_unexpected(artifact_error(ErrorCode::kDuplicateName, "CLOCK_CATALOG_DUPLICATE"));
    }
    if (declaration.is_default &&
        find_default_clock_plan_id(result.clocks, declaration.graph).valid()) {
      return make_unexpected(artifact_error(ErrorCode::kDuplicateName, "CLOCK_DEFAULT_DUPLICATE"));
    }
    result.clocks.push_back(ClockPlanRecord{
        .id = ClockId{static_cast<std::uint64_t>(index + 1U)},
        .graph = declaration.graph,
        .key = declaration.key,
        .shared = declaration.shared,
        .is_default = declaration.is_default,
        .can_read_now = capability != snapshot.clock_capabilities.end() && capability->can_read_now,
        .can_wait_until =
            capability != snapshot.clock_capabilities.end() && capability->can_wait_until,
        .steady = capability != snapshot.clock_capabilities.end() && capability->steady,
        .independent_progress =
            capability != snapshot.clock_capabilities.end() && capability->independent_progress,
        .finite_deadline =
            capability != snapshot.clock_capabilities.end() && capability->finite_deadline,
        .external_driver_admitted =
            capability != snapshot.clock_capabilities.end() && capability->external_driver_admitted,
        .capability_evidence_id = capability == snapshot.clock_capabilities.end()
                                      ? "holoscan.clock.capability.missing.v1"
                                      : capability->evidence_id,
    });
  }
  return {};
}

[[nodiscard]] expected<void, Error> append_clock_import_resolutions(
    const GraphSnapshotData& snapshot, ClockAndReadinessLoweringResult& result) {
  result.clock_import_resolutions.reserve(snapshot.clock_imports.size());
  for (const auto& clock_import : snapshot.clock_imports) {
    const auto binding = std::find_if(
        snapshot.clock_bindings.begin(), snapshot.clock_bindings.end(), [&](const auto& candidate) {
          return candidate.child_graph == clock_import.graph &&
                 candidate.child_clock == clock_import.key;
        });
    const AuthoredClockResolution resolved = resolve_authored_clock(snapshot.graphs,
                                                                    snapshot.clocks,
                                                                    snapshot.clock_imports,
                                                                    snapshot.clock_bindings,
                                                                    clock_import.graph,
                                                                    clock_import.key);
    if (binding == snapshot.clock_bindings.end() || !resolved.resolved()) {
      return make_unexpected(
          artifact_error(ErrorCode::kInvalidArgument, "CLOCK_IMPORT_BINDING_INVALID"));
    }
    const ClockId resolved_clock =
        find_clock_plan_id(result.clocks, resolved.declaration->graph, resolved.declaration->key);
    if (!resolved_clock.valid()) {
      return make_unexpected(
          artifact_error(ErrorCode::kInvalidArgument, "CLOCK_IMPORT_BINDING_INVALID"));
    }
    result.clock_import_resolutions.push_back(ClockImportResolutionPlanRecord{
        .child_graph = clock_import.graph,
        .child_clock = clock_import.key,
        .is_default = clock_import.is_default,
        .parent_graph = binding->parent_graph,
        .parent_clock = binding->parent_clock,
        .resolved_clock = resolved_clock,
    });
  }
  return {};
}

[[nodiscard]] expected<ClockAndReadinessLoweringResult, Error> lower_clock_catalog(
    const GraphSnapshotData& snapshot, PlanGenerationId plan_generation) {
  if (plan_generation == 0U) {
    return make_unexpected(artifact_error(ErrorCode::kInvalidArgument, "PLAN_GENERATION_INVALID"));
  }
  if (auto bindings = validate_clock_import_bindings(snapshot); !bindings) {
    return make_unexpected(std::move(bindings).error());
  }

  ClockAndReadinessLoweringResult result;
  result.plan_generation = plan_generation;
  if (auto declarations = append_clock_declarations(snapshot, result); !declarations) {
    return make_unexpected(std::move(declarations).error());
  }
  if (auto imports = append_clock_import_resolutions(snapshot, result); !imports) {
    return make_unexpected(std::move(imports).error());
  }

  const GraphRecord* root = find_root_graph(snapshot);
  if (root != nullptr) {
    result.root_default_clock = find_default_clock_plan_id(result.clocks, root->id);
  }
  return result;
}

[[nodiscard]] expected<void, Error> append_clock_conversions(
    const GraphSnapshotData& snapshot, ClockAndReadinessLoweringResult& result) {
  result.clock_conversions.reserve(snapshot.clock_conversions.size());
  for (std::size_t index = 0U; index < snapshot.clock_conversions.size(); ++index) {
    const auto& declaration = snapshot.clock_conversions[index];
    const ClockId from =
        resolve_clock_plan_id(snapshot, result.clocks, declaration.graph, declaration.from);
    const ClockId to =
        resolve_clock_plan_id(snapshot, result.clocks, declaration.graph, declaration.to);
    if (!from.valid() || !to.valid()) {
      return make_unexpected(artifact_error(ErrorCode::kNotFound, "CLOCK_CONVERSION_UNDECLARED"));
    }
    if (from == to || declaration.conversion.scale.numerator <= 0 ||
        declaration.conversion.scale.denominator == 0U) {
      return make_unexpected(
          artifact_error(ErrorCode::kInvalidArgument, "CLOCK_CONVERSION_INVALID"));
    }
    const auto existing = std::find_if(
        result.clock_conversions.begin(),
        result.clock_conversions.end(),
        [&](const auto& candidate) { return candidate.from == from && candidate.to == to; });
    if (existing != result.clock_conversions.end()) {
      if (existing->scale != declaration.conversion.scale ||
          existing->offset_ns != declaration.conversion.offset_ns) {
        return make_unexpected(
            artifact_error(ErrorCode::kIncompatible, "CLOCK_CONVERSION_CONFLICT"));
      }
      // Embedded graphs can independently declare the same relation after their imports resolve
      // into one parent clock pair. Publish one canonical record so every pair lookup is unique.
      continue;
    }
    result.clock_conversions.push_back(ClockConversionRecord{
        .id = static_cast<std::uint64_t>(result.clock_conversions.size() + 1U),
        .plan_generation = result.plan_generation,
        .graph = declaration.graph,
        .from = from,
        .to = to,
        .scale = declaration.conversion.scale,
        .offset_ns = declaration.conversion.offset_ns,
        .rounding = ClockRoundingMode::kNearestTiesToEven,
        .overflow = ClockOverflowPolicy::kError,
    });
  }
  return {};
}

[[nodiscard]] expected<std::vector<CanonicalReadinessDeclaration>, Error>
canonical_readiness_declarations(const GraphSnapshotData& snapshot) {
  std::vector<CanonicalReadinessDeclaration> declarations;
  declarations.reserve(snapshot.readiness_sources.size());
  for (const auto& declaration : snapshot.readiness_sources) {
    auto canonical = canonical_readiness_declaration(snapshot, declaration);
    if (!canonical) {
      return make_unexpected(std::move(canonical).error());
    }
    declarations.push_back(std::move(canonical).value());
  }
  std::sort(declarations.begin(), declarations.end(), [](const auto& lhs, const auto& rhs) {
    return std::tuple{lhs.graph_path,
                      lhs.owner_name,
                      readiness_kind_token(lhs.declaration.kind),
                      lhs.declaration.name,
                      lhs.declaration.declaration_ordinal} <
           std::tuple{rhs.graph_path,
                      rhs.owner_name,
                      readiness_kind_token(rhs.declaration.kind),
                      rhs.declaration.name,
                      rhs.declaration.declaration_ordinal};
  });
  if (declarations.size() > std::numeric_limits<std::uint32_t>::max()) {
    return make_unexpected(
        artifact_error(ErrorCode::kResourceExhausted, "READINESS_SOURCE_ID_EXHAUSTED"));
  }
  return declarations;
}

[[nodiscard]] bool readiness_capacity_valid(
    const ReadinessSourceDeclarationRecord& source) noexcept {
  const bool capacity_one =
      source.kind == ReadinessSourceKind::kReadable || source.kind == ReadinessSourceKind::kSpin;
  if (capacity_one) {
    return source.requested_capacity == 1U;
  }
  if (source.requested_capacity == 0U) {
    return false;
  }
  return source.kind != ReadinessSourceKind::kNotification ||
         source.requested_sender_reference_capacity != 0U;
}

[[nodiscard]] expected<ClockId, Error> resolve_readiness_clock(
    const GraphSnapshotData& snapshot, std::span<const ClockPlanRecord> clocks,
    const CanonicalReadinessDeclaration& candidate) {
  const auto& source = candidate.declaration;
  const ClockId resolved =
      resolve_clock_plan_id(snapshot, clocks, candidate.owner_graph, source.requested_clock);
  if (resolved.valid()) {
    return resolved;
  }
  return make_unexpected(artifact_error(ErrorCode::kNotFound,
                                        source.requested_clock.is_default()
                                            ? "READINESS_DEFAULT_CLOCK_UNDECLARED"
                                            : "READINESS_CLOCK_UNDECLARED"));
}

[[nodiscard]] expected<std::vector<ClockId>, Error> validate_and_resolve_readiness(
    const GraphSnapshotData& snapshot, std::span<const CanonicalReadinessDeclaration> declarations,
    std::span<const ClockPlanRecord> clocks) {
  std::vector<ClockId> resolved_clocks;
  resolved_clocks.reserve(declarations.size());
  for (const auto& candidate : declarations) {
    const auto& source = candidate.declaration;
    if (!readiness_capacity_valid(source)) {
      return make_unexpected(
          artifact_error(ErrorCode::kInvalidArgument, "READINESS_CAPACITY_INVALID"));
    }
    auto resolved = resolve_readiness_clock(snapshot, clocks, candidate);
    if (!resolved) {
      return make_unexpected(std::move(resolved).error());
    }
    resolved_clocks.push_back(std::move(resolved).value());
  }
  return resolved_clocks;
}

void append_readiness_records(std::span<const CanonicalReadinessDeclaration> declarations,
                              std::span<const ClockId> resolved_clocks,
                              std::span<const std::optional<Profile>> effective_vertex_profiles,
                              ClockAndReadinessLoweringResult& result) {
  result.readiness.reserve(declarations.size());
  for (std::size_t index = 0U; index < declarations.size(); ++index) {
    const auto& candidate = declarations[index];
    const auto& source = candidate.declaration;
    result.readiness.push_back(ReadinessPlanRecord{
        .source_id = make_source_id(static_cast<std::uint32_t>(index + 1U)),
        .plan_generation = result.plan_generation,
        .owner = source.owner,
        .owner_path = candidate.owner_path,
        .declaration_ordinal = source.declaration_ordinal,
        .kind = source.kind,
        .name = source.name,
        .effective_profile = effective_profile_for_vertex(effective_vertex_profiles, source.owner),
        .requested_clock = source.requested_clock,
        .resolved_clock = resolved_clocks[index],
        .requested_capacity = source.requested_capacity,
        .requested_sender_reference_capacity = source.requested_sender_reference_capacity,
        .wake_requirement = source.wake_requirement,
        .accepted_capacity = source.requested_capacity,
        .accepted_sender_reference_capacity = source.requested_sender_reference_capacity,
        .referenced = source.referenced,
        .materialized = source.referenced,
        .initial_source_generation = 1U,
        .initial_source_sequence = 0U,
        .disposition_reason =
            source.referenced ? DiagnosticId{} : diagnostic_id("READINESS_SOURCE_UNUSED"),
    });
    if (!source.referenced) {
      result.unused_source_warnings.push_back(DiagnosticSource{
          .graph = candidate.owner_graph,
          .vertex = source.owner,
      });
    }
  }
}

}  // namespace

ClockId find_plan_default_clock_id(std::span<const ClockPlanRecord> clocks,
                                   GraphId graph) noexcept {
  return find_default_clock_plan_id(clocks, graph);
}

ClockId resolve_plan_clock_id(const GraphSnapshotData& snapshot,
                              std::span<const ClockPlanRecord> clocks, GraphId graph,
                              const ClockKey& key) noexcept {
  return resolve_clock_plan_id(snapshot, clocks, graph, key);
}

expected<ClockAndReadinessLoweringResult, Error> lower_clocks_and_readiness(
    const GraphSnapshotData& snapshot, PlanGenerationId plan_generation,
    std::span<const std::optional<Profile>> effective_vertex_profiles) {
  // REQ: COMPILE-007 - validate the whole clock/conversion/source table before
  // publishing any row. Numeric ClockId values never escape without the same
  // nonzero plan-generation provenance.
  auto result = lower_clock_catalog(snapshot, plan_generation);
  if (!result) {
    return make_unexpected(std::move(result).error());
  }
  auto conversions = append_clock_conversions(snapshot, result.value());
  if (!conversions) {
    return make_unexpected(std::move(conversions).error());
  }
  auto declarations = canonical_readiness_declarations(snapshot);
  if (!declarations) {
    return make_unexpected(std::move(declarations).error());
  }
  auto resolved_clocks =
      validate_and_resolve_readiness(snapshot, declarations.value(), result.value().clocks);
  if (!resolved_clocks) {
    return make_unexpected(std::move(resolved_clocks).error());
  }
  append_readiness_records(
      declarations.value(), resolved_clocks.value(), effective_vertex_profiles, result.value());
  return result;
}
}  // namespace holoscan::detail

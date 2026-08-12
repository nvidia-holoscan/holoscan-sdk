/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/parameter_catalog.hpp>

#include "../core/graph_snapshot.hpp"
#include "../runtime/executable_factory_inventory.hpp"
#include "parameter_generation_zero.hpp"

namespace holoscan::detail {

struct LifecyclePlanLowering;
struct DeploymentPlanLowering;

using ParameterFieldProjectionIdentity = std::pair<ParamFieldId, ParamSchemaHash>;
using ParameterInstanceProjectionIdentity = std::pair<ParamInstanceId, ParamSchemaHash>;

/// @brief Classify a compact Parameter projection collision without relying on hash accidents.
enum class ParameterProjectionCollision : std::uint8_t {
  kNone = 0,        ///< The candidate projection is new and valid.
  kFieldId = 1,     ///< A field identity or repeated field projection conflicts.
  kInstanceId = 2,  ///< Distinct full instance identities share one compact instance id.
};

/// @brief Classify one field projection against prior fields in the same schema.
[[nodiscard]] ParameterProjectionCollision classify_parameter_field_projection(
    std::span<const ParameterFieldProjectionIdentity> existing, ParamFieldId field,
    ParamSchemaHash full_field) noexcept;

/// @brief Classify one instance projection against prior fields in the complete plan.
[[nodiscard]] ParameterProjectionCollision classify_parameter_instance_projection(
    std::span<const ParameterInstanceProjectionIdentity> existing, ParamInstanceId instance,
    ParamSchemaHash full_instance) noexcept;

/// @brief Return the stable structural token for one collision classification.
[[nodiscard]] std::string_view parameter_projection_collision_token(
    ParameterProjectionCollision collision) noexcept;

/// @brief Retain one bounded Parameter admission diagnostic before publication.
struct ParameterPlanDiagnostic {
  std::string token;        ///< Registered structural diagnostic token.
  DiagnosticSource source;  ///< Canonical plan-local source, when applicable.
};

/// @brief Own the complete pointer-free Parameter plan-lowering result.
struct ParameterPlanLowering {
  std::vector<OperatorParameterPlanRecord> operators;  ///< Dense Operator inventory.
  std::vector<ParameterFieldPlanRecord> fields;        ///< Canonically ordered fields.
  ParameterCatalogView catalog;                        ///< Public policy-filtered catalog.
  std::vector<ParameterGenerationZeroOperatorRecord>
      generation_zero_operators;  ///< Privileged aggregate provenance and digests.
  std::vector<ParameterGenerationZeroFieldRecord>
      generation_zero_fields;  ///< Privileged per-field provenance and digests.
  std::vector<LifecycleMutableFieldPlanRecord>
      lifecycle_mutable_fields;                      ///< Admitted restart-driving fields.
  ParameterRuntimeCapacityPlanRecord capacity{};     ///< Finite runtime capacities.
  std::vector<ParameterPlanDiagnostic> diagnostics;  ///< Deterministic rejections.
};

/// @brief Lower local factory metadata into immutable Parameter plan records.
/// @param snapshot Canonical pointer-free graph identity and path inventory.
/// @param factories Canonical process-local executable factory inventory, updated atomically with
/// isolated resolved clones when an accepted override set is present.
/// @param effective_vertex_profiles Profile indexed by canonical VertexId minus one.
/// @param plan_generation Nonzero identity for every produced OperatorRef.
/// @param options Owned Parameter compile policy and finite capacity request.
/// @param lifecycle_options Active transaction options, or empty when lifecycle mutation is off.
/// @param admission_context_digest Canonical profile/deployment/package/migration context identity.
/// @param parameter_overrides Optional complete owning explicit override set.
/// @return Total records and stable bounded diagnostics.
/// @note The lowerer copies no aggregate value, operation pointer, or type token into public plan
/// state. Typed generation zero remains solely in the private factory inventory.
[[nodiscard]] ParameterPlanLowering lower_parameter_plan(
    const GraphSnapshotData& snapshot, ExecutableFactoryInventory& factories,
    std::span<const std::optional<Profile>> effective_vertex_profiles,
    PlanGenerationId plan_generation, const ParameterCompileOptions& options,
    const std::optional<LifecycleTransactionOptions>& lifecycle_options,
    ParamSchemaHash admission_context_digest,
    const std::optional<ParamOverrideSet>& parameter_overrides);

/// @brief Rebind one immutable Parameter catalog to the final plan generation.
/// @param catalog Catalog produced while canonical plan identity was being derived.
/// @param plan_generation Final nonzero plan-generation identity.
/// @return Deep-owned catalog with every Operator reference and compatibility digest rebound.
/// @note This cold compiler operation changes generation provenance only; all
/// normalized Parameter semantics and compact identities remain unchanged.
[[nodiscard]] ParameterCatalogView rebind_parameter_catalog_plan_generation(
    const ParameterCatalogView& catalog, PlanGenerationId plan_generation);

/// @brief Return the versioned identity of the empty EA1 migration catalog.
/// @return Stable nonzero digest used until EA1-018 supplies an admitted catalog digest.
[[nodiscard]] ParamSchemaHash empty_parameter_migration_catalog_digest() noexcept;

/// @brief Digest the complete external Parameter admission-input context.
/// @param profile Resolved profile/policy plus plan schema, runtime ABI, target, and admitted
/// package/evidence inputs. Diagnostic-only producer build identity is excluded.
/// @param deployment Final normalized role, placement, probe, and lane evidence.
/// @param migration_catalog_digest Exact immutable migration-catalog identity.
/// @return Canonical context digest excluding overrides, authoring order, and graph-derived plan
/// obligations.
/// @note EA1-018 must use this same builder while capturing a rebind target. It may replace only
/// the explicit empty migration-catalog digest after qualified migrations are implemented and
/// must add any separately admitted package-offer input once that surface exists. Final profile
/// capabilities, materialization, fallback, and generated-endpoint records remain covered by the
/// complete PlanDigest because they depend on graph lowering and cannot be admission inputs.
[[nodiscard]] ParamSchemaHash parameter_admission_context_digest(
    const ProfileManifest& profile, const DeploymentPlanLowering& deployment,
    ParamSchemaHash migration_catalog_digest) noexcept;

/// @brief Compute one protected semantic value digest for compile validation and tests.
/// @param type Exact process-local aggregate descriptor.
/// @param schema Declared schema, or null for opaque mode.
/// @param value Live exact aggregate matching \p type.
/// @param no_params Whether this aggregate is exactly \ref NoParams.
/// @return Total digest disposition and protected bytes when canonical scalar projection exists.
[[nodiscard]] HOLOSCAN_CORE_EXPORT ParameterProtectedValueDigest
parameter_value_digest(const ParamTypeDescriptor& type, const ParamSchemaDescriptor* schema,
                       const void* value, bool no_params) noexcept;

/// @brief Finalize Parameter lifecycle capacity after restart closures exist.
/// @param result Structural Parameter lowering to complete in place.
/// @param lifecycle First-pass lifecycle lowering containing canonical closures and checkpoints.
/// @param lifecycle_options Configured lifecycle capacities, or empty when transactions are off.
/// @note This second phase is required because lifecycle-mutable fields may
/// restart dependency-closure members that do not themselves declare mutable
/// fields. It validates all nine lifecycle resources against the complete
/// union closure and disables admitted fields when capacity is insufficient.
void finalize_parameter_lifecycle_capacity(
    ParameterPlanLowering& result, const LifecyclePlanLowering& lifecycle,
    const std::optional<LifecycleTransactionOptions>& lifecycle_options);

}  // namespace holoscan::detail

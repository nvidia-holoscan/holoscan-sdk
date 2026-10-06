/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>
#include <span>
#include <string>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/parameter_catalog.hpp>

#include "../core/graph_snapshot.hpp"
#include "../runtime/executable_factory_inventory.hpp"
#include "parameter_generation_zero.hpp"
#include "parameter_identity.hpp"

namespace holoscan::detail {

struct LifecyclePlanLowering;

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
/// isolated resolved clones when an accepted graph-owned parameter input is present.
/// @param effective_vertex_profiles Profile indexed by canonical VertexId minus one.
/// @param plan_generation Nonzero identity for every produced OperatorRef.
/// @param options Owned Parameter compile policy and finite capacity request.
/// @param lifecycle_options Active transaction options, or empty when lifecycle mutation is off.
/// @return Total records and stable bounded diagnostics.
/// @note The lowerer copies no aggregate value, operation pointer, or type token into public plan
/// state. Typed generation zero remains solely in the private factory inventory.
[[nodiscard]] ParameterPlanLowering lower_parameter_plan(
    const GraphSnapshotData& snapshot, ExecutableFactoryInventory& factories,
    std::span<const std::optional<Profile>> effective_vertex_profiles,
    PlanGenerationId plan_generation, const ParameterCompileOptions& options,
    const std::optional<LifecycleTransactionOptions>& lifecycle_options);

/// @brief Rebind one immutable Parameter catalog to the final plan generation.
/// @param catalog Catalog produced while canonical plan identity was being derived.
/// @param plan_generation Final nonzero plan-generation identity.
/// @return Deep-owned catalog with every Operator reference and compatibility digest rebound.
/// @note This cold compiler operation changes generation provenance only; all
/// normalized Parameter semantics and compact identities remain unchanged.
[[nodiscard]] ParameterCatalogView rebind_parameter_catalog_plan_generation(
    const ParameterCatalogView& catalog, PlanGenerationId plan_generation);

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

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "parameter_catalog_projection.hpp"

#include <algorithm>
#include <utility>
#include <vector>

#include "../core/parameter_catalog_digest_internal.hpp"
#include "../core/parameter_catalog_internal.hpp"

namespace holoscan::detail {

ParamCatalogDigest runtime_parameter_catalog_digest(
    PlanGenerationId plan_generation, RunGenerationId run_generation,
    std::span<const OperatorParamRecord> operators,
    std::span<const ParamFieldCatalogRecord> fields) noexcept {
  StableHashBuilder hash;
  hash.add("holoscan.parameter_catalog.policy_view.v1");
  // Bind even an empty policy projection to its exact source plan/run without
  // seeding the digest from hidden source records.
  hash.add(plan_generation);
  hash.add(run_generation);
  add_parameter_catalog_records(hash, operators, fields);
  return ParamCatalogDigest{.value = hash.finish().value};
}

expected<ParameterCatalogView, Error> filter_runtime_parameter_catalog(
    const ParameterCatalogView& source, const ParamAccessPolicy& policy,
    PlanGenerationId plan_generation, RunGenerationId run_generation) noexcept {
  try {
    std::vector<ParamFieldCatalogRecord> fields;
    if (!has_param_permission(policy.permissions, ParamPermission::kRead)) {
      std::vector<OperatorParamRecord> operators;
      const ParamCatalogDigest digest =
          runtime_parameter_catalog_digest(plan_generation, run_generation, operators, fields);
      return ParameterCatalogAccess::make(digest, std::move(operators), std::move(fields));
    }
    fields.reserve(source.fields().size());
    for (const ParamFieldCatalogRecord& field : source.fields()) {
      const ParamScopeFilter scope = policy.allowed_scope;
      if (scope != ParamScopeFilter::run() &&
          (scope.operator_instance != field.address.operator_instance ||
           (scope.field != 0U && scope.field != field.address.field))) {
        continue;
      }
      // Sensitive catalog metadata remains discoverable with its compile-frozen
      // redacted projection. include_sensitive governs value/read/event access;
      // it does not remove the value-free field identity promised by the
      // public catalog contract.
      if ((field.visibility == ParamVisibility::kInternal && !policy.include_internal) ||
          (policy.origin == ParamControlOrigin::kExternalAdapter &&
           field.external_access == ParamExternalAccess::kNone)) {
        continue;
      }
      fields.push_back(field);
    }

    std::vector<OperatorParamRecord> operators;
    operators.reserve(source.operators().size());
    for (const OperatorParamRecord& op : source.operators()) {
      const ParamScopeFilter scope = policy.allowed_scope;
      if (scope != ParamScopeFilter::run() && scope.operator_instance != op.operator_instance) {
        continue;
      }
      const bool source_has_fields =
          std::any_of(source.fields().begin(), source.fields().end(), [&](const auto& field) {
            return field.address.operator_instance == op.operator_instance;
          });
      const bool visible_field = std::any_of(fields.begin(), fields.end(), [&](const auto& field) {
        return field.address.operator_instance == op.operator_instance;
      });
      if (!source_has_fields || visible_field) {
        operators.push_back(op);
      }
    }
    const ParamCatalogDigest digest =
        runtime_parameter_catalog_digest(plan_generation, run_generation, operators, fields);
    return ParameterCatalogAccess::make(digest, std::move(operators), std::move(fields));
  } catch (...) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
}

}  // namespace holoscan::detail

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <span>
#include <string_view>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/parameter_catalog.hpp>

#include "parameter_generation_zero.hpp"

namespace holoscan::detail {

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

/// Canonicalize one portable scalar carrier before stable hashing or comparison.
[[nodiscard]] bool canonicalize_parameter_scalar(ParamScalarValue& value) noexcept;

/// Compute one protected digest for a schema field read from an aggregate.
[[nodiscard]] ParameterProtectedValueDigest parameter_field_value_digest(
    const ParamSchemaDescriptor& schema, const ParamFieldDescriptor& field,
    const void* aggregate) noexcept;

/// Compute the full stable identity of one Parameter field instance.
[[nodiscard]] ParamSchemaHash parameter_instance_identity(const OperatorPath& path,
                                                          ParamSchemaHash field_full_id) noexcept;

/// Project a full Parameter instance identity into the compact runtime identifier.
[[nodiscard]] ParamInstanceId compact_parameter_instance_id(ParamSchemaHash identity) noexcept;

/// Compute the canonical digest for an ordered Parameter catalog projection.
[[nodiscard]] ParamCatalogDigest parameter_catalog_digest(
    std::span<const OperatorParamRecord> operators,
    std::span<const ParamFieldCatalogRecord> fields) noexcept;

/// @brief Return the versioned identity of the empty baseline migration catalog.
/// @return Stable nonzero digest used until HSDK5-018 supplies an admitted catalog digest.
[[nodiscard]] ParamSchemaHash empty_parameter_migration_catalog_digest() noexcept;

/// @brief Digest the complete external Parameter admission-input context.
/// @param profile Resolved profile/policy plus plan schema, runtime ABI, target, and admitted
/// package/evidence inputs. Diagnostic-only producer build identity is excluded.
/// @param deployment Canonical cold deployment input, before partition-name resolution.
/// @param migration_catalog_digest Exact immutable migration-catalog identity.
/// @return Canonical context digest excluding overrides, authoring order, and graph-derived plan
/// obligations.
/// @note HSDK5-018 must use this same builder while capturing a rebind target. It may replace only
/// the explicit empty migration-catalog digest after qualified migrations are implemented and
/// must add any separately admitted package-offer input once that surface exists. Final profile
/// capabilities, materialization, fallback, and generated-endpoint records remain covered by the
/// complete PlanDigest because they depend on graph lowering and cannot be admission inputs.
[[nodiscard]] ParamSchemaHash parameter_admission_context_digest(
    const ProfileManifest& profile, const DeploymentBinding& deployment,
    ParamSchemaHash migration_catalog_digest);

/// @brief Compute one protected semantic value digest for compile validation and tests.
/// @param type Exact process-local aggregate descriptor.
/// @param schema Declared schema, or null for opaque mode.
/// @param value Live exact aggregate matching \p type.
/// @param no_params Whether this aggregate is exactly \ref NoParams.
/// @return Total digest disposition and protected bytes when canonical scalar projection exists.
[[nodiscard]] HOLOSCAN_CORE_EXPORT ParameterProtectedValueDigest
parameter_value_digest(const ParamTypeDescriptor& type, const ParamSchemaDescriptor* schema,
                       const void* value, bool no_params) noexcept;

}  // namespace holoscan::detail

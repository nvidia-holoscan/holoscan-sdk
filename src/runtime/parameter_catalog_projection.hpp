/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <span>

#include <holoscan/core/parameter_control.hpp>

namespace holoscan::detail {

/// Return the canonical digest for one exact runtime Parameter catalog projection.
[[nodiscard]] ParamCatalogDigest runtime_parameter_catalog_digest(
    PlanGenerationId plan_generation, RunGenerationId run_generation,
    std::span<const OperatorParamRecord> operators,
    std::span<const ParamFieldCatalogRecord> fields) noexcept;

/// Copy one policy-visible catalog projection and recompute its exact digest.
[[nodiscard]] expected<ParameterCatalogView, Error> filter_runtime_parameter_catalog(
    const ParameterCatalogView& source, const ParamAccessPolicy& policy,
    PlanGenerationId plan_generation, RunGenerationId run_generation) noexcept;

}  // namespace holoscan::detail

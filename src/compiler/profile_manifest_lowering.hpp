/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoscan/core/compile.hpp>

#include "execution_plan_state.hpp"
#include "runtime_package_capabilities.hpp"

namespace holoscan::detail {

/// Populate the complete canonical profile requirement and materialization inventory.
///
/// This pass records only plan requirements and compile evidence. Executing-package
/// offers, provider selection, and startup match outcomes never enter plan identity.
[[nodiscard]] expected<void, Error> finalize_profile_manifest(const CompileOptions& options,
                                                              ExecutionPlanState& state);

/// Populate and match the canonical profile inventory against an injected immutable package view.
[[nodiscard]] expected<void, Error> finalize_profile_manifest(
    const CompileOptions& options, ExecutionPlanState& state,
    const RuntimePackageCapabilityView& package_capabilities);

}  // namespace holoscan::detail

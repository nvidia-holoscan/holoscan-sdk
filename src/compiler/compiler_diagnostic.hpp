/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>
#include <string_view>

#include <holoscan/core/diagnostic.hpp>

namespace holoscan::detail {

struct ExecutionPlanState;
struct PlanLoweringDiagnostic;

/// Retain a structural compiler diagnostic before attempting optional cause enrichment.
Diagnostic& append_compiler_diagnostic(
    ExecutionPlanState& state, std::string_view token, DiagnosticSource source = {},
    std::optional<holoscan::DiagnosticSeverity> severity_override = std::nullopt,
    std::string_view json_pointer = {}, std::string_view required = {},
    std::string_view provided = {}, const DiagnosticDetail* evidence = nullptr);

/// Publish derivation evidence using names from the same canonical graph as its source IDs.
void publish_derivation_diagnostic(ExecutionPlanState& state,
                                   const PlanLoweringDiagnostic& diagnostic);

}  // namespace holoscan::detail

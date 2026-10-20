/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

#include "graph_exec.hpp"
#include "runtime_diagnostic_sink.hpp"

namespace holoscan::detail {

/// Own one complete, authenticated process-local Operator reconstruction transaction.
struct RuntimeOperatorReconstruction {
  std::vector<RuntimeOperatorInstance> operators;
  std::vector<std::size_t> receiver_by_vertex;
  std::vector<std::size_t> receiver_by_operator_id;
  std::optional<RuntimePackageCapabilityMatchReport> package_capability_report;
  std::unique_ptr<ParameterUpdateCoordinator> parameter_updates;
};

/// Validate and reconstruct all process-local Operators and their Parameter bindings.
[[nodiscard]] expected<RuntimeOperatorReconstruction, Error> reconstruct_runtime_operators(
    const ExecutionPlanState& plan, RunGenerationId run_generation,
    ProcessGroupId process_group = {}, RuntimeReplayMismatchDetail* replay_mismatch = nullptr,
    RuntimeDiagnosticDetail* failure_detail = nullptr);

}  // namespace holoscan::detail

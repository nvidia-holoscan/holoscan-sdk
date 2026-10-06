/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/graph.hpp>

namespace holoscan::detail {

struct ExecutionPlanState;

/// Construct one cold Operator-reconstruction diagnostic.
[[nodiscard]] inline Error operator_reconstruction_error(ErrorCode code, const char* token) {
  return Error{code, token};
}

/// Construct one cold Operator-reconstruction diagnostic with human-readable failure detail.
[[nodiscard]] inline Error operator_reconstruction_error(ErrorCode code, std::string_view token,
                                                         std::string_view detail) {
  std::string message{token};
  if (!detail.empty()) {
    message.append(": ");
    message.append(detail);
  }
  return Error{code, std::move(message)};
}

/// Validate the complete frozen lifecycle inventory.
[[nodiscard]] bool lifecycle_plan_well_formed(const ExecutionPlanState& plan,
                                              std::size_t operator_count);

/// Resolve one Operator's canonical nested graph path.
[[nodiscard]] expected<std::vector<std::string>, Error> planned_operator_path(
    const ExecutionPlanState& plan, VertexId owner);

/// Validate the grouped Parameter field and derived-term inventories.
[[nodiscard]] bool parameter_owner_inventories_well_formed(const ExecutionPlanState& plan,
                                                           std::size_t operator_count);

}  // namespace holoscan::detail

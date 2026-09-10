/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

#include "compiler/execution_plan_state.hpp"

namespace holoscan::detail {

/// Serialize normalized plan semantics and set the exact SHA-256 plan digest.
///
/// This operation performs no filesystem, environment, transport, runtime,
/// thread, CUDA, logging, or lifecycle mutation.
[[nodiscard]] expected<void, Error> finalize_plan_artifact(ExecutionPlanState& state);

}  // namespace holoscan::detail

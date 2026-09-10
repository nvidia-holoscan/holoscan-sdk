/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

#include "graph_exec.hpp"

namespace holoscan::detail {

/// Validate Runtime foundations consumed by per-Operator replay.
[[nodiscard]] expected<void, Error> validate_foundation_plan(const ExecutionPlanState& plan,
                                                             std::size_t operator_count);

/// Replay and authenticate one reconstructed Operator declaration.
[[nodiscard]] expected<void, Error> validate_replayed_operator(
    const ExecutionPlanState& plan, const ExecutableOperatorFactoryRecord& record, OperatorBase& op,
    const OperatorSpec& replay, std::uint32_t factory_slot,
    RuntimeReplayMismatchDetail* replay_mismatch);

}  // namespace holoscan::detail

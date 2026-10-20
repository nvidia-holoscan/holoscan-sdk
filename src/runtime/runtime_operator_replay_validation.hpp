/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>

#include "../compiler/execution_plan_state.hpp"
#include "executable_factory_inventory.hpp"
#include "runtime_diagnostic_sink.hpp"
#include "runtime_replay_mismatch.hpp"

namespace holoscan::detail {

/// Validate Runtime foundations consumed by per-Operator replay.
[[nodiscard]] expected<void, Error> validate_foundation_plan(const ExecutionPlanState& plan,
                                                             std::size_t operator_count);

/// Replay and authenticate one reconstructed Operator declaration.
[[nodiscard]] expected<void, Error> validate_replayed_operator(
    const ExecutionPlanState& plan, const ExecutableOperatorFactoryRecord& record, OperatorBase& op,
    const OperatorSpec& replay, std::uint32_t factory_slot,
    RuntimeReplayMismatchDetail* replay_mismatch,
    RuntimeDiagnosticDetail* failure_detail = nullptr);

}  // namespace holoscan::detail

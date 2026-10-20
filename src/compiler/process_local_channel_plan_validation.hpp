/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace holoscan::detail {

struct ExecutionPlanState;

/// Check the narrow all-local blocking scalar/POD envelope of the private fanout experiment.
[[nodiscard]] bool valid_process_local_fanout_experiment(const ExecutionPlanState& state);

/// Validate the exact private process-local Channel construction facts derived by the compiler.
///
/// This cold-path check rejects drift between the public physical plan and the
/// private service, publisher, and subscriber records consumed by Runtime. It
/// performs no Runtime or backend mutation. The caller must retain the plan
/// state for the duration of the call.
[[nodiscard]] bool valid_process_local_channel_plan(const ExecutionPlanState& state);

/// Validate the exact external-topic Channel and endpoint-catalog relation.
///
/// This cold-path check proves that every canonical external topic has exactly
/// one matching service, publisher, subscriber, and catalog endpoint record,
/// with no duplicate or extra construction authority.
[[nodiscard]] bool valid_external_channel_plan(const ExecutionPlanState& state);

/// @brief Validate the exact private cross-process Channel construction facts.
///
/// Every accepted cross-process payload edge eligible for an inline, Tensor,
/// or descriptor-Tensor schema realization must own exactly one deterministic
/// service with one publisher-local and one subscriber-local request; repeated
/// logical edges never share a service. The check performs no Runtime or
/// backend mutation, and the caller must retain \p state for the duration of
/// the call.
///
/// @param state Immutable compiler-owned execution-plan state to validate.
/// @return `true` when the complete cross-process construction inventory is
/// internally consistent; otherwise `false`.
[[nodiscard]] bool valid_cross_process_channel_plan(const ExecutionPlanState& state);

}  // namespace holoscan::detail

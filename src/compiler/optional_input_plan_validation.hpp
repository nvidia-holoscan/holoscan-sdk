/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
namespace holoscan::detail {
struct ExecutionPlanState;
/// Authenticate optional-input permission, canonical wiring, and frozen temporal admission.
[[nodiscard]] bool valid_optional_input_plan(const ExecutionPlanState& state);
}  // namespace holoscan::detail

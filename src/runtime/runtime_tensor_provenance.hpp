/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <string_view>
namespace holoscan::detail {
// Inspect existing backing while its Runtime pool is alive. No new Tensor authority is granted.
[[nodiscard]] HOLOSCAN_CORE_EXPORT bool runtime_tensor_backing_matches(
    const Tensor& tensor, PortId output, PlanGenerationId plan, RunGenerationId run,
    std::string_view pool_id) noexcept;
}  // namespace holoscan::detail

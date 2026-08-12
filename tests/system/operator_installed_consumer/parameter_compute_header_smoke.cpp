/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <type_traits>

#include <holoscan/core/execution_context.hpp>

static_assert(std::is_standard_layout_v<holoscan::ParamComputeControl>);
static_assert(std::is_trivially_copyable_v<holoscan::ParamComputeControl>);

int parameter_compute_header_smoke() {
  const holoscan::ParamComputeControl control;
  const std::array<holoscan::ParamInlinePatch, 1U> patches{};
  const holoscan::ParamSubmitResult result = control.try_submit_from_compute(patches);
  return result.status == holoscan::ParamUpdateStatus::kUpdatesDisabled &&
                 result.state == holoscan::ParamUpdateState::kRejected
             ? 0
             : 1;
}

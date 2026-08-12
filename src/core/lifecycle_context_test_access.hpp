/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoscan/core/lifecycle.hpp>

#include "lifecycle_context_internal.hpp"

namespace holoscan::detail {

/// Expose the private callback-lifetime lifecycle binding to source-tree tests only.
class LifecycleContextTestAccess {
 public:
  [[nodiscard]] static LifecycleContext context(LifecycleContextState& state) noexcept {
    return LifecycleContext{&state};
  }
};

}  // namespace holoscan::detail

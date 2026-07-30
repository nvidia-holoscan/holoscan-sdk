/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/profiler/profiler.hpp>

namespace holoscan::profiler {

static bool trace_enabled_ = false;

void trace(bool enable) {
  trace_enabled_ = enable;
}

bool trace_enabled() {
  return trace_enabled_;
}

}  // namespace holoscan::profiler

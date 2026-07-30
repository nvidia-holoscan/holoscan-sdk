/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_PROFILER_PROFILER_HPP
#define HOLOSCAN_PROFILER_PROFILER_HPP

namespace holoscan::profiler {

/**
 * @brief Enables or disables tracing.
 *
 * @param enable Whether or not to enable tracing.
 */
void trace(bool enable);

/**
 * @brief Returns whether or not tracing is enabled.
 *
 * @returns true if tracing is enabled, false otherwise.
 */
bool trace_enabled();

}  // namespace holoscan::profiler

#include <holoscan/profiler/nvtx3.hpp>

#endif  // HOLOSCAN_PROFILER_PROFILER_HPP

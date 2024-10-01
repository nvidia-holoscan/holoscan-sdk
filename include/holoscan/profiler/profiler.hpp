/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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

#include "holoscan/profiler/nvtx3.hpp"

#endif  // HOLOSCAN_PROFILER_PROFILER_HPP

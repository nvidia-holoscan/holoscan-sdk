/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_CORE_SYSTEM_CPU_INFO_HPP
#define HOLOSCAN_CORE_SYSTEM_CPU_INFO_HPP

#include <memory>

namespace holoscan {

namespace CPUMetricFlag {
enum : uint64_t {
  DEFAULT = 0x00,
  CORE_COUNT = 0x01,
  CPU_COUNT = 0x02,
  AVAILABLE_PROCESSOR_COUNT = 0x04,
  COUNT = CORE_COUNT | CPU_COUNT | AVAILABLE_PROCESSOR_COUNT,
  CPU_USAGE = 0x8,
  MEMORY_USAGE = 0x10,
  SHARED_MEMORY_USAGE = 0x20,
  ALL = COUNT | CPU_USAGE | MEMORY_USAGE | SHARED_MEMORY_USAGE,
};
}  // namespace CPUMetricFlag

/**
 * @brief CPUInfo struct
 *
 * This struct is responsible for holding the CPU information.
 */
struct CPUInfo {
  uint64_t metric_flags = 0;             ///< The metric flags
  int32_t num_cores = 0;                 ///< The number of cores
  int32_t num_cpus = 0;                  ///< The number of CPUs
  int32_t num_processors = 0;            ///< The number of available processors
  float cpu_usage = 0.0f;                ///< The CPU usage (in percent)
  uint64_t memory_total = 0;             ///< The total memory (in bytes)
  uint64_t memory_free = 0;              ///< The free memory (in bytes)
  uint64_t memory_available = 0;         ///< The available memory (in bytes)
  float memory_usage = 0.0f;             ///< The memory usage (in percent)
  uint64_t shared_memory_total = 0;      ///< The total shared memory (in bytes)
  uint64_t shared_memory_free = 0;       ///< The free shared memory (in bytes)
  uint64_t shared_memory_available = 0;  ///< The available shared memory (in bytes)
  float shared_memory_usage = 0.0f;      ///< The shared memory usage (in percent)
};
}  // namespace holoscan

#endif /* HOLOSCAN_CORE_SYSTEM_CPU_INFO_HPP */

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

#ifndef HOLOSCAN_CORE_SYSTEM_GPU_INFO_HPP
#define HOLOSCAN_CORE_SYSTEM_GPU_INFO_HPP

#include <memory>

#include "nvml_wrapper.h"

namespace holoscan {

namespace GPUMetricFlag {
enum : uint64_t {
  DEFAULT = 0x00,
  GPU_DEVICE_ID = 0x01,
  GPU_UTILIZATION = 0x02,
  MEMORY_USAGE = 0x04,
  POWER_LIMIT = 0x08,
  POWER_USAGE = 0x10,
  TEMPERATURE = 0x20,
  ALL = GPU_DEVICE_ID | GPU_UTILIZATION | MEMORY_USAGE | POWER_LIMIT | POWER_USAGE | TEMPERATURE,
};
}  // namespace GPUMetricFlag

/**
 * @brief GPUInfo struct
 *
 * This struct is responsible for holding the GPU information.
 */
struct GPUInfo {
  uint64_t metric_flags = 0;                         ///< The metric flags
  uint32_t index = 0;                                ///< The GPU index
  char name[NVML_DEVICE_NAME_BUFFER_SIZE] = {};      ///< The GPU name
  bool is_integrated = false;                        ///< The GPU is integrated
  nvml::nvmlPciInfo_st pci = {};                     ///< The GPU PCI information
  char serial[NVML_DEVICE_SERIAL_BUFFER_SIZE] = {};  ///< The GPU serial number
  char uuid[NVML_DEVICE_UUID_BUFFER_SIZE] = {};      ///< The GPU UUID
  /// The GPU utilization. Percent of time over the past sample period during which one or more
  /// kernels was executing on the GPU.
  uint32_t gpu_utilization = 0;
  /// The memory utilization. Percent of time over the past sample period during which global
  /// (device) memory was being read or written.
  uint32_t memory_utilization = 0;
  uint64_t memory_total = 0;  ///< The total memory (in bytes)
  uint64_t memory_free = 0;   ///< The free memory (in bytes)
  uint64_t memory_used = 0;   ///< The used memory (in bytes)
  float memory_usage = 0.0f;  ///< The memory usage (in percent)
  uint32_t power_limit = 0;   ///< The power limit (in milliwatts)
  uint32_t power_usage = 0;   ///< The power usage (in milliwatts)
  uint32_t temperature = 0;   ///< The temperature (in degrees Celsius)
};
}  // namespace holoscan

#endif /* HOLOSCAN_CORE_SYSTEM_GPU_INFO_HPP */

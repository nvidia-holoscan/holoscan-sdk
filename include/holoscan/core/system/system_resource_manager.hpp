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
#ifndef HOLOSCAN_CORE_SYSTEM_SYSTEM_RESOURCE_MANAGER_HPP
#define HOLOSCAN_CORE_SYSTEM_SYSTEM_RESOURCE_MANAGER_HPP

#include <memory>

#include "holoscan/core/system/cpu_resource_monitor.hpp"
#include "holoscan/core/system/gpu_resource_monitor.hpp"
#include "holoscan/core/system/topology.hpp"

namespace holoscan {

/**
 * @brief SystemResourceManager class
 *
 * This class is responsible for monitoring the system resources.
 * It provides the information about the topology of the system and the system resources such as
 * CPU, GPU, etc. This information is collected by the AppWorker and passed to the AppDriver for
 * scheduling in the distributed application.
 */
class SystemResourceManager {
 public:
  SystemResourceManager();
  virtual ~SystemResourceManager() = default;

  /**
   * @brief Get CPU resource monitor.
   *
   * @return The pointer to the CPU resource monitor.
   */
  CPUResourceMonitor* cpu_monitor();

  /**
   * @brief Get GPU resource monitor.
   *
   * @return The pointer to the GPU resource monitor.
   */
  GPUResourceMonitor* gpu_monitor();

 protected:
  std::shared_ptr<Topology> topology_;                        ///< The topology of the system
  std::shared_ptr<CPUResourceMonitor> cpu_resource_monitor_;  ///< The CPU resource monitor
  std::shared_ptr<GPUResourceMonitor> gpu_resource_monitor_;  ///< The GPU resource monitor
};
}  // namespace holoscan

#endif /* HOLOSCAN_CORE_SYSTEM_SYSTEM_RESOURCE_MANAGER_HPP */

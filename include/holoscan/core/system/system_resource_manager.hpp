/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef HOLOSCAN_CORE_SYSTEM_SYSTEM_RESOURCE_MANAGER_HPP
#define HOLOSCAN_CORE_SYSTEM_SYSTEM_RESOURCE_MANAGER_HPP

#include <memory>

#include <holoscan/core/system/cpu_resource_monitor.hpp>
#include <holoscan/core/system/gpu_resource_monitor.hpp>
#include <holoscan/core/system/topology.hpp>

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

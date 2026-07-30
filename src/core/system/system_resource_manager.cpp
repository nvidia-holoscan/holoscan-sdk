/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <holoscan/core/system/system_resource_manager.hpp>

#include <hwloc.h>

#include <iostream>
#include <memory>

#include <holoscan/logger/logger.hpp>

namespace holoscan {

SystemResourceManager::SystemResourceManager() {
  topology_ = std::make_shared<Topology>();
  topology_->load();
  cpu_resource_monitor_ = std::make_shared<CPUResourceMonitor>(topology_->context());
  gpu_resource_monitor_ = std::make_shared<GPUResourceMonitor>();
}

CPUResourceMonitor* SystemResourceManager::cpu_monitor() {
  return cpu_resource_monitor_.get();
}

GPUResourceMonitor* SystemResourceManager::gpu_monitor() {
  return gpu_resource_monitor_.get();
}

}  // namespace holoscan

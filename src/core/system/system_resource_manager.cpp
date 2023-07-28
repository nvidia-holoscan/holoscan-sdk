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
#include "holoscan/core/system/system_resource_manager.hpp"

#include <hwloc.h>

#include <iostream>
#include <memory>

#include "holoscan/logger/logger.hpp"

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

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

#include "holoscan/core/resources/gxf/rmm_allocator.hpp"

#include <cstdint>
#include <string>

#include "gxf/std/resources.hpp"  // for GPUDevice

#include "holoscan/core/component_spec.hpp"

namespace holoscan {

namespace {

// kPoolInitialSize, kPoolMaxSize copied from rmm_allocator.cpp
#ifdef __aarch64__
constexpr const char* kPoolInitialSize = "8MB";  // 8 MB initial pool size
constexpr const char* kPoolMaxSize = "16MB";
#else
constexpr const char* kPoolInitialSize = "16MB";  // 16 MB initial pool size
constexpr const char* kPoolMaxSize = "32MB";
#endif
constexpr int32_t kDefaultDeviceId = 0;

}  // namespace

RMMAllocator::RMMAllocator(const std::string& name, nvidia::gxf::RMMAllocator* component)
    : CudaAllocator(name, component) {
  auto maybe_device_initial = component->getParameter<std::string>("device_memory_initial_size");
  if (!maybe_device_initial) {
    throw std::runtime_error("Failed to get device_memory_initial_size");
  }
  device_memory_initial_size_ = maybe_device_initial.value();

  auto maybe_device_max = component->getParameter<std::string>("device_memory_max_size");
  if (!maybe_device_max) { throw std::runtime_error("Failed to get device_memory_max_size"); }
  device_memory_max_size_ = maybe_device_max.value();

  auto maybe_host_initial = component->getParameter<std::string>("host_memory_initial_size");
  if (!maybe_host_initial) { throw std::runtime_error("Failed to get host_memory_initial_size"); }
  host_memory_initial_size_ = maybe_host_initial.value();

  auto maybe_host_max = component->getParameter<std::string>("host_memory_max_size");
  if (!maybe_host_max) { throw std::runtime_error("Failed to get host_memory_max_size"); }
  host_memory_max_size_ = maybe_host_max.value();

  auto maybe_gpu_device =
      component->getParameter<nvidia::gxf::Handle<nvidia::gxf::GPUDevice>>("dev_id");
  if (!maybe_gpu_device) { throw std::runtime_error("Failed to get dev_id"); }
  auto gpu_device_handle = maybe_gpu_device.value();
  dev_id_ = gpu_device_handle->device_id();
}

nvidia::gxf::RMMAllocator* RMMAllocator::get() const {
  return static_cast<nvidia::gxf::RMMAllocator*>(gxf_cptr_);
}

void RMMAllocator::setup(ComponentSpec& spec) {
  spec.param(device_memory_initial_size_,
             "device_memory_initial_size",
             "Device Memory Pool Initial Size",
             "The initial memory pool size used by this device. Examples of valid values are "
             "'512MB', '256 KB', '1 GB'. The format is a non-negative integer value followed by "
             "an optional space and then a suffix representing the units. Supported units are "
             "B, KB, MB, GB and TB where the values are powers of 1024 bytes.",
             std::string(kPoolInitialSize));
  spec.param(device_memory_max_size_,
             "device_memory_max_size",
             "Device Memory Pool Maximum Size",
             "The max memory pool size used by this device. Examples of valid values are "
             "'512MB', '256 KB', '1 GB'. The format is a non-negative integer value followed by "
             "an optional space and then a suffix representing the units. Supported units are "
             "B, KB, MB, GB and TB where the values are powers of 1024 bytes.",
             std::string(kPoolMaxSize));
  spec.param(host_memory_initial_size_,
             "host_memory_initial_size",
             "Host Memory Pool Initial Size",
             "The initial memory pool size used by the host. Examples of valid values are "
             "'512MB', '256 KB', '1 GB'. The format is a non-negative integer value followed by "
             "an optional space and then a suffix representing the units. Supported units are "
             "B, KB, MB, GB and TB where the values are powers of 1024 bytes.",
             std::string(kPoolInitialSize));
  spec.param(host_memory_max_size_,
             "host_memory_max_size",
             "Host Memory Pool Maximum Size",
             "The max memory pool size used by the host. Examples of valid values are "
             "'512MB', '256 KB', '1 GB'. The format is a non-negative integer value followed by "
             "an optional space and then a suffix representing the units. Supported units are "
             "B, KB, MB, GB and TB where the values are powers of 1024 bytes.",
             std::string(kPoolMaxSize));
  spec.param(dev_id_,
             "dev_id",
             "Device Id",
             "Device on which to create the memory pool.",
             kDefaultDeviceId);
}

}  // namespace holoscan

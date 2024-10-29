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

#include "holoscan/core/resources/gxf/stream_ordered_allocator.hpp"

#include <string>

#include "gxf/std/resources.hpp"  // for GPUDevice

#include "holoscan/core/component_spec.hpp"

namespace holoscan {

namespace {
// default values copied from gxf/cuda/stream_ordered_allocator.cpp
#ifdef __aarch64__
constexpr const char* kPoolInitialSize = "8MB";  // 8 MB initial pool size
constexpr const char* kPoolMaxSize = "16MB";
#else
constexpr const char* kPoolInitialSize = "16MB";  // 16 MB initial pool size
constexpr const char* kPoolMaxSize = "32MB";
#endif
constexpr const char* kReleaseThreshold = "4MB";  // 4MB release threshold
constexpr int32_t kDefaultDeviceId = 0;

}  // namespace

StreamOrderedAllocator::StreamOrderedAllocator(const std::string& name,
                                               nvidia::gxf::StreamOrderedAllocator* component)
    : CudaAllocator(name, component) {
  auto maybe_device_initial = component->getParameter<std::string>("device_memory_initial_size");
  if (!maybe_device_initial) {
    throw std::runtime_error("Failed to get device_memory_initial_size");
  }
  device_memory_initial_size_ = maybe_device_initial.value();

  auto maybe_device_max = component->getParameter<std::string>("device_memory_max_size");
  if (!maybe_device_max) { throw std::runtime_error("Failed to get device_memory_max_size"); }
  device_memory_max_size_ = maybe_device_max.value();

  auto maybe_release_threshold = component->getParameter<std::string>("release_threshold");
  if (!maybe_release_threshold) { throw std::runtime_error("Failed to get release_threshold"); }
  release_threshold_ = maybe_release_threshold.value();

  auto maybe_gpu_device =
      component->getParameter<nvidia::gxf::Handle<nvidia::gxf::GPUDevice>>("dev_id");
  if (!maybe_gpu_device) { throw std::runtime_error("Failed to get dev_id"); }
  auto gpu_device_handle = maybe_gpu_device.value();
  dev_id_ = gpu_device_handle->device_id();
}

void StreamOrderedAllocator::setup(ComponentSpec& spec) {
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
  spec.param(release_threshold_,
             "release_threadhold",
             "Amount of reserved memory to hold onto before trying to release memory back "
             "to the OS",
             "The release threshold specifies the maximum amount of memory the pool caches. "
             "Examples of valid values are '512MB', '256 KB', '1 GB'. The format is a "
             "non-negative integer value followed by an optional space and then a suffix "
             "representing the units. Supported units are B, KB, MB, GB and TB where the values "
             "are powers of 1024 bytes.",
             std::string(kReleaseThreshold));
  spec.param(dev_id_,
             "dev_id",
             "Device Id",
             "Device on which to create the memory pool.",
             static_cast<int32_t>(0));
}

nvidia::gxf::StreamOrderedAllocator* StreamOrderedAllocator::get() const {
  return static_cast<nvidia::gxf::StreamOrderedAllocator*>(gxf_cptr_);
}

}  // namespace holoscan

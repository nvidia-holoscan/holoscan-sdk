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

#ifndef HOLOSCAN_CORE_RESOURCES_GXF_RMM_ALLOCATOR_HPP
#define HOLOSCAN_CORE_RESOURCES_GXF_RMM_ALLOCATOR_HPP

#include <cstdint>
#include <string>

#include "gxf/rmm/rmm_allocator.hpp"
#include "gxf/std/allocator.hpp"

#include "./cuda_allocator.hpp"

namespace holoscan {

/**
 * @brief RMM (RAPIDS memory manager) allocator.
 *
 * This is a memory pool which provides a user-specified number of equally sized blocks of memory.
 */
class RMMAllocator : public CudaAllocator {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(RMMAllocator, CudaAllocator)

  RMMAllocator() = default;
  RMMAllocator(const std::string& device_memory_initial_size,
               const std::string& device_memory_max_size,
               const std::string& host_memory_initial_size, const std::string& host_memory_max_size,
               int32_t dev_id = 0)
      : device_memory_initial_size_(device_memory_initial_size),
        device_memory_max_size_(device_memory_max_size),
        host_memory_initial_size_(host_memory_initial_size),
        host_memory_max_size_(host_memory_max_size),
        dev_id_(dev_id) {}
  RMMAllocator(const std::string& name, nvidia::gxf::RMMAllocator* component);

  const char* gxf_typename() const override { return "nvidia::gxf::RMMAllocator"; }

  void setup(ComponentSpec& spec) override;

  nvidia::gxf::RMMAllocator* get() const;

  // pool_size method implemented on the parent CudaAllocator class

 private:
  Parameter<std::string> device_memory_initial_size_;
  Parameter<std::string> device_memory_max_size_;
  Parameter<std::string> host_memory_initial_size_;
  Parameter<std::string> host_memory_max_size_;
  Parameter<int32_t> dev_id_;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCES_GXF_RMM_ALLOCATOR_HPP */

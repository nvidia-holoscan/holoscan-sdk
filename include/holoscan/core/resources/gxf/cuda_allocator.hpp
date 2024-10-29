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

#ifndef HOLOSCAN_CORE_RESOURCES_GXF_CUDA_ALLOCATOR_HPP
#define HOLOSCAN_CORE_RESOURCES_GXF_CUDA_ALLOCATOR_HPP

#include <cstdint>
#include <string>

#include <gxf/cuda/cuda_allocator.hpp>

#include "../../gxf/gxf_resource.hpp"
#include "./allocator.hpp"

namespace holoscan {

/**
 * @brief Base class for all CUDA allocators.
 *
 * CudaAllocators are allocators for CUDA memory that support asynchronous allocation.
 */
class CudaAllocator : public Allocator {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(CudaAllocator, Allocator)

  CudaAllocator() = default;
  CudaAllocator(const std::string& name, nvidia::gxf::CudaAllocator* component);

  const char* gxf_typename() const override { return "nvidia::gxf::CudaAllocator"; }

  // the following async functions and get_pool_size are specific to CudaAllocator
  nvidia::byte* allocate_async(uint64_t size, cudaStream_t stream);
  void free_async(byte* pointer, cudaStream_t stream);
  size_t pool_size(MemoryStorageType type) const;

  nvidia::gxf::CudaAllocator* get() const;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCES_GXF_CUDA_ALLOCATOR_HPP */

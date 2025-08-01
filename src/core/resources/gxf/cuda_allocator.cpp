/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/core/resources/gxf/cuda_allocator.hpp"

#include <string>

namespace holoscan {

CudaAllocator::CudaAllocator(const std::string& name, nvidia::gxf::CudaAllocator* component)
    : Allocator(name, component) {}

nvidia::gxf::CudaAllocator* CudaAllocator::get() const {
  return static_cast<nvidia::gxf::CudaAllocator*>(gxf_cptr_);
}

nvidia::byte* CudaAllocator::allocate_async(uint64_t size, cudaStream_t stream) {
  auto allocator = get();
  if (allocator) {
    auto result = allocator->allocate_async(size, stream);
    if (result) {
      return result.value();
    }
  }

  HOLOSCAN_LOG_ERROR("Failed to asynchronously allocate memory of size {}", size);

  return nullptr;
}

void CudaAllocator::free_async(nvidia::byte* pointer, cudaStream_t stream) {
  auto allocator = get();
  if (allocator) {
    auto result = allocator->free_async(pointer, stream);
    if (!result) {
      HOLOSCAN_LOG_ERROR("Failed to asynchronously free memory at {}", static_cast<void*>(pointer));
    }
  }
}

size_t CudaAllocator::pool_size(MemoryStorageType type) const {
  auto allocator = get();
  if (!allocator) {
    throw std::runtime_error("null GXF component pointer");
  }
  auto maybe_size = allocator->get_pool_size(static_cast<nvidia::gxf::MemoryStorageType>(type));
  if (!maybe_size) {
    throw std::runtime_error("failed to get pool size");
  }
  return maybe_size.value();
}

}  // namespace holoscan

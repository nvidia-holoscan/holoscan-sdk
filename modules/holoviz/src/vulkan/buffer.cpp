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

#include "buffer.hpp"

#include <memory>

namespace holoscan::viz {

Buffer::Buffer(vk::Device device, nvvk::ResourceAllocator* alloc, size_t size)
    : Resource(device, alloc), size_(size) {}

Buffer::~Buffer() {
  destroy();

  // check if this buffer had been imported to CUDA
  if (device_ptr_) {
    const CudaService::ScopedPush cuda_context = cuda_service_->PushContext();
    device_ptr_.reset();
  }
  alloc_->destroy(buffer_);
}

void Buffer::import_to_cuda(const std::unique_ptr<CudaService>& cuda_service) {
  const CudaService::ScopedPush cuda_context = cuda_service->PushContext();

  const nvvk::MemAllocator::MemInfo mem_info =
      alloc_->getMemoryAllocator()->getMemoryInfo(buffer_.memHandle);

  // call the base class for creating the external mem and the semaphores
  Resource::import_to_cuda(cuda_service, mem_info);

  CUDA_EXTERNAL_MEMORY_BUFFER_DESC buffer_desc{};
  buffer_desc.size = size_;
  buffer_desc.offset = mem_info.offset;

  device_ptr_.reset([external_mem = external_mem_.get(), &buffer_desc] {
    CUdeviceptr device_ptr;
    CudaCheck(cuExternalMemoryGetMappedBuffer(&device_ptr, external_mem, &buffer_desc));
    return device_ptr;
  }());
}

}  // namespace holoscan::viz

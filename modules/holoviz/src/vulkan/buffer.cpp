/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "buffer.hpp"

#include <memory>

#include "vulkan_app.hpp"

namespace holoscan::viz {

Buffer::Buffer(Vulkan* vulkan, nvvk::ResourceAllocator* alloc, size_t size)
    : Resource(vulkan, alloc), size_(size) {}

Buffer::~Buffer() {
  try {
    wait();

    // check if this buffer had been imported to CUDA
    if (device_ptr_) {
      const CudaService::ScopedPush cuda_context = vulkan_->get_cuda_service()->PushContext();
      device_ptr_.reset();
    }
    alloc_->destroy(buffer_);
  } catch (const std::exception& e) {
  }  // ignore potential exceptions
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

  device_ptr_.reset([external_mem = external_mems_.front().get(), &buffer_desc] {
    CUdeviceptr device_ptr;
    CudaCheck(cuExternalMemoryGetMappedBuffer(&device_ptr, external_mem, &buffer_desc));
    return device_ptr;
  }());
}

}  // namespace holoscan::viz

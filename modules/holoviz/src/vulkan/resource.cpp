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

#include "resource.hpp"

#include <unistd.h>

#include <memory>

#include <holoscan/logger/logger.hpp>

namespace holoscan::viz {

Resource::Resource(vk::Device device, nvvk::ResourceAllocator* alloc)
    : device_(device), alloc_(alloc) {}

Resource::~Resource() {
  destroy();
}

void Resource::access_with_vulkan(nvvk::BatchSubmission& batch_submission) {
  if (external_mem_) {
    if (state_ == AccessState::CUDA) {
      // enqueue the semaphore signalled by CUDA to be waited on by rendering
      batch_submission.enqueueWait(cuda_access_wait_semaphore_.get(),
                                   VkPipelineStageFlagBits::VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    }

    // also signal the render semapore which will be waited on by CUDA
    batch_submission.enqueueSignal(vulkan_access_signal_semaphore_.get());
    state_ = AccessState::VULKAN;
  }
}

void Resource::begin_access_with_cuda(CUstream stream) {
  if (state_ == AccessState::VULKAN) {
    CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS ext_wait_params{};
    const CUexternalSemaphore external_wait_semaphore = vulkan_access_wait_semaphore_.get();
    CudaCheck(cuWaitExternalSemaphoresAsync(&external_wait_semaphore, &ext_wait_params, 1, stream));
    state_ = AccessState::UNKNOWN;
  }
}

void Resource::end_access_with_cuda(CUstream stream) {
  // signal the semaphore for the CUDA operation on the buffer
  CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS ext_signal_params{};
  const CUexternalSemaphore external_signal_semaphore = cuda_access_signal_semaphore_.get();
  CudaCheck(
      cuSignalExternalSemaphoresAsync(&external_signal_semaphore, &ext_signal_params, 1, stream));
  state_ = AccessState::CUDA;
}

void Resource::destroy() {
  if (fence_) {
    // if the resource had been tagged with a fence, wait for it before freeing the memory
    const vk::Result result = device_.waitForFences(fence_, true, 100'000'000);
    if (result != vk::Result::eSuccess) {
      HOLOSCAN_LOG_WARN("Waiting for texture fence failed with {}", vk::to_string(result));
    }
    fence_ = nullptr;
  }

  // check if this resource had been imported to CUDA
  if (external_mem_) {
    const CudaService::ScopedPush cuda_context = cuda_service_->PushContext();

    external_mem_.reset();
    cuda_access_signal_semaphore_.reset();
    vulkan_access_wait_semaphore_.reset();

    cuda_access_wait_semaphore_.reset();
    vulkan_access_signal_semaphore_.reset();
  }
}

void Resource::import_to_cuda(const std::unique_ptr<CudaService>& cuda_service,
                    const nvvk::MemAllocator::MemInfo& mem_info) {
  cuda_service_ = cuda_service.get();

  vk::MemoryGetFdInfoKHR memory_get_fd_info;
  memory_get_fd_info.memory = mem_info.memory;
  memory_get_fd_info.handleType = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd;
  UniqueValue<int, decltype(&close), &close> file_handle;
  file_handle.reset(device_.getMemoryFdKHR(memory_get_fd_info));

  CUDA_EXTERNAL_MEMORY_HANDLE_DESC memory_handle_desc{};
  memory_handle_desc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
  memory_handle_desc.handle.fd = file_handle.get();
  memory_handle_desc.size = mem_info.offset + mem_info.size;

  external_mem_.reset([&memory_handle_desc] {
    CUexternalMemory external_mem;
    CudaCheck(cuImportExternalMemory(&external_mem, &memory_handle_desc));
    return external_mem;
  }());
  // don't need to close the file handle if it had been successfully imported
  file_handle.release();

  // create the semaphores, one for waiting after CUDA access and one for signalling
  // Vulkan access
  vk::StructureChain<vk::SemaphoreCreateInfo, vk::ExportSemaphoreCreateInfoKHR> chain;
  vk::SemaphoreCreateInfo& semaphore_create_info = chain.get<vk::SemaphoreCreateInfo>();
  chain.get<vk::ExportSemaphoreCreateInfoKHR>().handleTypes =
      vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd;
  cuda_access_wait_semaphore_ = device_.createSemaphoreUnique(semaphore_create_info);
  vulkan_access_signal_semaphore_ = device_.createSemaphoreUnique(semaphore_create_info);

  // import the semaphore to CUDA
  cuda_access_signal_semaphore_ = import_semaphore_to_cuda(cuda_access_wait_semaphore_.get());
  vulkan_access_wait_semaphore_ = import_semaphore_to_cuda(vulkan_access_signal_semaphore_.get());
}

UniqueCUexternalSemaphore Resource::import_semaphore_to_cuda(vk::Semaphore semaphore) {
  vk::SemaphoreGetFdInfoKHR semaphore_get_fd_info;
  semaphore_get_fd_info.semaphore = semaphore;
  semaphore_get_fd_info.handleType = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd;

  UniqueValue<int, decltype(&close), &close> file_handle;
  file_handle.reset(device_.getSemaphoreFdKHR(semaphore_get_fd_info));

  CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC semaphore_handle_desc{};
  semaphore_handle_desc.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD;
  semaphore_handle_desc.handle.fd = file_handle.get();

  UniqueCUexternalSemaphore cuda_semaphore;
  cuda_semaphore.reset([&semaphore_handle_desc] {
    CUexternalSemaphore ext_semaphore;
    CudaCheck(cuImportExternalSemaphore(&ext_semaphore, &semaphore_handle_desc));
    return ext_semaphore;
  }());

  // don't need to close the file handle if it had been successfully imported
  file_handle.release();

  return cuda_semaphore;
}

}  // namespace holoscan::viz

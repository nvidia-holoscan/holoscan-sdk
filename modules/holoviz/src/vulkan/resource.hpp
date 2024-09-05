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

#ifndef MODULES_HOLOVIZ_SRC_VULKAN_RESOURCE_HPP
#define MODULES_HOLOVIZ_SRC_VULKAN_RESOURCE_HPP

#include <memory>
#include <vector>

#include <nvvk/commands_vk.hpp>
#include <nvvk/resourceallocator_vk.hpp>

#include <vulkan/vulkan.hpp>

#include "../cuda/cuda_service.hpp"

namespace holoscan::viz {

class Vulkan;

/// Resource base class. Can be shared between CUDA and Vulkan. Access to the resource is
/// synchronized with semaphores.
class Resource {
 public:
  explicit Resource(Vulkan* vulkan, nvvk::ResourceAllocator* alloc);
  Resource() = delete;
  virtual ~Resource();

  /**
   * Synchronize access to the resource before using it with Vulkan
   *
   * @param batch_submission command buffer to use for synchronization
   */
  void access_with_vulkan(nvvk::BatchSubmission& batch_submission);

  /**
   * Start accessing the resource with CUDA
   *
   * @param stream CUDA stream to use for synchronization
   */
  void begin_access_with_cuda(CUstream stream);

  /**
   * End of resource access from CUDA
   *
   * @param stream CUDA stream to use for synchronization
   */
  void end_access_with_cuda(CUstream stream);

  /**
   * Wait for the access fence to be triggered.
   */
  void wait();

  /// access state
  enum class AccessState {
    /// not accessed yet
    UNKNOWN,
    /// last accessed by CUDA
    CUDA,
    /// last accessed by VULKAN
    VULKAN
  };
  AccessState state_ = AccessState::UNKNOWN;

  /// last usage of the resource, need to sync before destroying memory
  vk::Fence fence_ = nullptr;

 protected:
  std::vector<UniqueCUexternalMemory> external_mems_;

  Vulkan* const vulkan_;
  nvvk::ResourceAllocator* const alloc_;

  void import_to_cuda(const std::unique_ptr<CudaService>& cuda_service,
                      const nvvk::MemAllocator::MemInfo& mem_info);

 private:
  /// this semaphore is used to synchronize CUDA operations on the texture, it's signaled by CUDA
  /// after accessing the texture (for upload) and waited on by Vulkan before accessing (rendering)
  vk::UniqueSemaphore cuda_access_wait_semaphore_;
  UniqueCUexternalSemaphore cuda_access_signal_semaphore_;

  /// this semaphore is used to synchronize Vulkan operations on the texture, it's signaled by
  /// Vulkan after accessing the texture (for rendering) and waited on by CUDA before accessing
  /// (upload)
  vk::UniqueSemaphore vulkan_access_signal_semaphore_;
  UniqueCUexternalSemaphore vulkan_access_wait_semaphore_;

  UniqueCUexternalSemaphore import_semaphore_to_cuda(vk::Semaphore semaphore);
};

}  // namespace holoscan::viz

#endif /* MODULES_HOLOVIZ_SRC_VULKAN_RESOURCE_HPP */

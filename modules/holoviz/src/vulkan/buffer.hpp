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

#ifndef MODULES_HOLOVIZ_SRC_VULKAN_BUFFER_HPP
#define MODULES_HOLOVIZ_SRC_VULKAN_BUFFER_HPP

#include <memory>

#include "resource.hpp"

#include "../holoviz/image_format.hpp"

namespace holoscan::viz {

class Buffer : public Resource {
 public:
  explicit Buffer(vk::Device device, nvvk::ResourceAllocator* alloc, size_t size);
  Buffer() = delete;

  virtual ~Buffer();

  void import_to_cuda(const std::unique_ptr<CudaService>& cuda_service);

  const size_t size_;

  nvvk::Buffer buffer_{};
  UniqueCUdeviceptr device_ptr_;
};

}  // namespace holoscan::viz

#endif /* MODULES_HOLOVIZ_SRC_VULKAN_BUFFER_HPP */

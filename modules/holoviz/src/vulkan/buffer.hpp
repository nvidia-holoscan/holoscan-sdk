/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef MODULES_HOLOVIZ_SRC_VULKAN_BUFFER_HPP
#define MODULES_HOLOVIZ_SRC_VULKAN_BUFFER_HPP

#include <memory>

#include "resource.hpp"

#include "../holoviz/image_format.hpp"

namespace holoscan::viz {

class Buffer : public Resource {
 public:
  explicit Buffer(Vulkan* vulkan, nvvk::ResourceAllocator* alloc, size_t size);
  Buffer() = delete;

  virtual ~Buffer();

  void import_to_cuda(const std::unique_ptr<CudaService>& cuda_service);

  const size_t size_;

  nvvk::Buffer buffer_{};
  UniqueCUdeviceptr device_ptr_;
};

}  // namespace holoscan::viz

#endif /* MODULES_HOLOVIZ_SRC_VULKAN_BUFFER_HPP */

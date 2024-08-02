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

#ifndef MODULES_HOLOVIZ_SRC_VULKAN_TEXTURE_HPP
#define MODULES_HOLOVIZ_SRC_VULKAN_TEXTURE_HPP

#include <memory>

#include "resource.hpp"

#include "../holoviz/image_format.hpp"

namespace holoscan::viz {

class Texture : public Resource {
 public:
  explicit Texture(vk::Device device, nvvk::ResourceAllocator* alloc, uint32_t width,
                   uint32_t height, ImageFormat format);
  Texture() = delete;
  virtual ~Texture();

  void import_to_cuda(const std::unique_ptr<CudaService>& cuda_service);

  const uint32_t width_;
  const uint32_t height_;
  const ImageFormat format_;

  nvvk::Texture texture_{};
  UniqueCUmipmappedArray mipmap_;
};

}  // namespace holoscan::viz

#endif /* MODULES_HOLOVIZ_SRC_VULKAN_TEXTURE_HPP */

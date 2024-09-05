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
#include <vector>

#include <nvvk/descriptorsets_vk.hpp>

#include "resource.hpp"

#include "../holoviz/image_format.hpp"
#include "buffer.hpp"

namespace holoscan::viz {

class Texture : public Resource {
 public:
  explicit Texture(Vulkan* vulkan, nvvk::ResourceAllocator* alloc, uint32_t width, uint32_t height,
                   ImageFormat format);
  Texture() = delete;
  virtual ~Texture();

  void import_to_cuda(const std::unique_ptr<CudaService>& cuda_service);

  /**
   * Upload data from CUDA device memory to a texture which had been imported to CUDA with
   * ::import_to_cuda.
   *
   * @param ext_stream    CUDA stream to use for operations
   * @param device_ptr    Cuda device memory pointer for the planes
   * @param row_pitch     the number of bytes between each row for the planes, if zero then data is
   * assumed to be contiguous in memory
   */
  void upload(CUstream ext_stream, const std::array<CUdeviceptr, 3>& device_ptr,
              const std::array<size_t, 3>& row_pitch);

  const uint32_t width_;
  const uint32_t height_;
  const ImageFormat format_;

  nvvk::Texture texture_{};
  std::vector<UniqueCUmipmappedArray> mipmaps_;
  std::vector<std::unique_ptr<Buffer>> upload_buffers_;

  vk::UniqueSamplerYcbcrConversion sampler_ycbcr_conversion_;

  nvvk::DescriptorSetBindings desc_set_layout_bind_;
  vk::UniqueDescriptorSetLayout desc_set_layout_;
  vk::UniquePipelineLayout pipeline_layout_;
  vk::UniquePipeline pipeline_;
};

}  // namespace holoscan::viz

#endif /* MODULES_HOLOVIZ_SRC_VULKAN_TEXTURE_HPP */

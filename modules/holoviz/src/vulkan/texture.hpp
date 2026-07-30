/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

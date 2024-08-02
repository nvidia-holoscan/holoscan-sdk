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

#include "texture.hpp"

#include <memory>

#include "format_util.hpp"

namespace holoscan::viz {

Texture::Texture(vk::Device device, nvvk::ResourceAllocator* alloc, uint32_t width,
                 uint32_t height, ImageFormat format)
    : Resource(device, alloc), width_(width), height_(height), format_(format) {}

Texture::~Texture() {
  destroy();

  // check if this texture had been imported to CUDA
  if (mipmap_) {
    const CudaService::ScopedPush cuda_context = cuda_service_->PushContext();
    mipmap_.reset();
  }
  alloc_->destroy(texture_);
}

void Texture::import_to_cuda(const std::unique_ptr<CudaService>& cuda_service) {
  const CudaService::ScopedPush cuda_context = cuda_service->PushContext();

  const nvvk::MemAllocator::MemInfo mem_info =
      alloc_->getMemoryAllocator()->getMemoryInfo(texture_.memHandle);

  // call the base class for creating the external mem and the semaphores
  Resource::import_to_cuda(cuda_service, mem_info);

  uint32_t src_channels, dst_channels, component_size;
  format_info(format_, &src_channels, &dst_channels, &component_size);

  CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC mipmapped_array_desc{};
  mipmapped_array_desc.arrayDesc.Width = width_;
  mipmapped_array_desc.arrayDesc.Height = height_;
  mipmapped_array_desc.arrayDesc.Depth = 0;
  switch (component_size) {
    case 1:
      mipmapped_array_desc.arrayDesc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
      break;
    case 2:
      mipmapped_array_desc.arrayDesc.Format = CU_AD_FORMAT_UNSIGNED_INT16;
      break;
    case 4:
      mipmapped_array_desc.arrayDesc.Format = CU_AD_FORMAT_UNSIGNED_INT32;
      break;
    default:
      throw std::runtime_error("Unhandled component size");
  }
  mipmapped_array_desc.arrayDesc.NumChannels = dst_channels;
  mipmapped_array_desc.arrayDesc.Flags = CUDA_ARRAY3D_SURFACE_LDST;

  mipmapped_array_desc.numLevels = 1;
  mipmapped_array_desc.offset = mem_info.offset;

  mipmap_.reset([external_mem = external_mem_.get(), &mipmapped_array_desc] {
    CUmipmappedArray mipmaped_array;
    CudaCheck(cuExternalMemoryGetMappedMipmappedArray(
        &mipmaped_array, external_mem, &mipmapped_array_desc));
    return mipmaped_array;
  }());
}

}  // namespace holoscan::viz

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
#include <utility>

#include "../cuda/convert.hpp"
#include "format_util.hpp"
#include "vulkan_app.hpp"

namespace holoscan::viz {

Texture::Texture(Vulkan* vulkan, nvvk::ResourceAllocator* alloc, uint32_t width, uint32_t height,
                 ImageFormat format)
    : Resource(vulkan, alloc), width_(width), height_(height), format_(format) {}

Texture::~Texture() {
  try {
    wait();

    // check if this texture had been imported to CUDA
    if (!mipmaps_.empty()) {
      const CudaService::ScopedPush cuda_context = vulkan_->get_cuda_service()->PushContext();
      mipmaps_.clear();
    }
    alloc_->destroy(texture_);
  } catch (const std::exception& e) {}  // ignore potential exceptions
}

void Texture::import_to_cuda(const std::unique_ptr<CudaService>& cuda_service) {
  const CudaService::ScopedPush cuda_context = cuda_service->PushContext();

  if (is_yuv_format(format_)) {
    // can't upload directly to YUV textures. Create a buffer, import it to CUDA and when
    // uploading, copy to that buffer and update the texture from the buffer using Vulkan upload.
    for (uint32_t plane = 0; plane < texture_.memHandles.size(); ++plane) {
      uint32_t channels, hw_channels, component_size, width_divisor, height_divisior;
      format_info(format_,
                  &channels,
                  &hw_channels,
                  &component_size,
                  &width_divisor,
                  &height_divisior,
                  plane);

      const size_t size =
          channels * (width_ / width_divisor) * (height_ / height_divisior) * component_size;
      upload_buffers_.emplace_back(
          vulkan_->create_buffer_for_cuda_interop(size, vk::BufferUsageFlagBits::eTransferSrc));
      upload_buffers_.back()->import_to_cuda(cuda_service);
    }
  } else {
    for (uint32_t plane = 0; plane < texture_.memHandles.size(); ++plane) {
      const nvvk::MemAllocator::MemInfo mem_info =
          alloc_->getMemoryAllocator()->getMemoryInfo(texture_.memHandles[plane]);

      // call the base class for creating the external mem and the semaphores
      Resource::import_to_cuda(cuda_service, mem_info);

      uint32_t channels, hw_channels, component_size, width_divisor, height_divisior;
      format_info(format_,
                  &channels,
                  &hw_channels,
                  &component_size,
                  &width_divisor,
                  &height_divisior,
                  plane);

      CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC mipmapped_array_desc{};
      mipmapped_array_desc.arrayDesc.Width = width_ / width_divisor;
      mipmapped_array_desc.arrayDesc.Height = height_ / height_divisior;
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
      mipmapped_array_desc.arrayDesc.NumChannels = hw_channels;
      mipmapped_array_desc.arrayDesc.Flags = 0;
      if (channels != hw_channels) {
        // need surface LDST for RGB to RGBA conversion kernel
        mipmapped_array_desc.arrayDesc.Flags |= CUDA_ARRAY3D_SURFACE_LDST;
      }

      mipmapped_array_desc.numLevels = 1;
      mipmapped_array_desc.offset = mem_info.offset;

      UniqueCUmipmappedArray mipmap(
          [external_mem = external_mems_.back().get(), &mipmapped_array_desc] {
            CUmipmappedArray mipmaped_array;
            CudaCheck(cuExternalMemoryGetMappedMipmappedArray(
                &mipmaped_array, external_mem, &mipmapped_array_desc));
            return mipmaped_array;
          }());
      mipmaps_.push_back(std::move(mipmap));
    }
  }
}

void Texture::upload(CUstream ext_stream, const std::array<CUdeviceptr, 3>& device_ptr,
                     const std::array<size_t, 3>& row_pitch) {
  assert(device_ptr.size() == row_pitch.size());

  if (mipmaps_.empty() && upload_buffers_.empty()) {
    throw std::runtime_error("Texture had not been imported to CUDA, can't upload data.");
  }

  const CudaService::ScopedPush cuda_context = vulkan_->get_cuda_service()->PushContext();

  // select the stream to be used by CUDA operations
  const CUstream stream = vulkan_->get_cuda_service()->select_cuda_stream(ext_stream);

  if (!mipmaps_.empty()) {
    // start accessing the texture with CUDA
    begin_access_with_cuda(stream);
  }

  std::array<Buffer*, 3> buffers{};
  for (uint32_t plane = 0; plane < device_ptr.size(); ++plane) {
    if (!device_ptr[plane]) { break; }

    uint32_t channels, hw_channels, component_size, width_divisor, height_divisior;
    format_info(
        format_, &channels, &hw_channels, &component_size, &width_divisor, &height_divisior, plane);

    // the width and height might be different for each plane for Y'CbCr formats
    const uint32_t width = width_ / width_divisor;
    const uint32_t height = height_ / height_divisior;

    size_t src_pitch = row_pitch[plane] != 0 ? row_pitch[plane] : width * channels * component_size;

    if (!mipmaps_.empty()) {
      // direct upload to CUDA imported Vulkan texture by copying to CUDA array
      CUarray array;
      CudaCheck(cuMipmappedArrayGetLevel(&array, mipmaps_[plane].get(), 0));

      if (channels != hw_channels) {
        // three channel texture data is not hardware natively supported, convert to four channel
        if ((channels != 3) || (hw_channels != 4) || (component_size != 1)) {
          throw std::runtime_error("Unhandled conversion.");
        }

        // if the source CUDA memory is on a different device, allocate temporary memory, copy from
        // the source memory to the temporary memory and start the convert kernel using the
        // temporary memory
        UniqueAsyncCUdeviceptr tmp_device_ptr;
        if (!vulkan_->get_cuda_service()->IsMemOnDevice(device_ptr[plane])) {
          const size_t tmp_pitch = width * channels * component_size;

          // allocate temporary memory, note this is using the stream ordered memory allocator which
          // is not syncing globally like the normal `cuMemAlloc`
          tmp_device_ptr.reset([size = tmp_pitch * height, stream] {
            CUdeviceptr dev_ptr;
            CudaCheck(cuMemAllocAsync(&dev_ptr, size, stream));
            return std::pair<CUdeviceptr, CUstream>(dev_ptr, stream);
          }());

          CUDA_MEMCPY2D memcpy_2d{};
          memcpy_2d.srcMemoryType = CU_MEMORYTYPE_DEVICE;
          memcpy_2d.srcDevice = device_ptr[plane];
          memcpy_2d.srcPitch = src_pitch;
          memcpy_2d.dstMemoryType = CU_MEMORYTYPE_DEVICE;
          memcpy_2d.dstDevice = tmp_device_ptr.get().first;
          memcpy_2d.dstPitch = tmp_pitch;
          memcpy_2d.WidthInBytes = tmp_pitch;
          memcpy_2d.Height = height;
          CudaCheck(cuMemcpy2DAsync(&memcpy_2d, stream));

          src_pitch = tmp_pitch;
        }

        uint8_t alpha;
        switch (format_) {
          case ImageFormat::R8G8B8_UNORM:
          case ImageFormat::R8G8B8_SRGB:
            alpha = 0xFf;
            break;
          case ImageFormat::R8G8B8_SNORM:
            alpha = 0x7f;
            break;
          default:
            throw std::runtime_error("Unhandled format.");
        }

        ConvertR8G8B8ToR8G8B8A8(width,
                                height,
                                tmp_device_ptr ? tmp_device_ptr.get().first : device_ptr[plane],
                                src_pitch,
                                array,
                                stream,
                                alpha);
      } else {
        // else just copy
        CUDA_MEMCPY2D memcpy_2d{};
        memcpy_2d.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        memcpy_2d.srcDevice = device_ptr[plane];
        memcpy_2d.srcPitch = src_pitch;
        memcpy_2d.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        memcpy_2d.dstArray = array;
        memcpy_2d.WidthInBytes = width * hw_channels * component_size;
        memcpy_2d.Height = height;
        CudaCheck(cuMemcpy2DAsync(&memcpy_2d, stream));
      }
    } else {
      // copy to Vulkan buffer which had been imported to CUDA and the use Vulkan to upload to
      // texture
      upload_buffers_[plane]->begin_access_with_cuda(stream);

      CUDA_MEMCPY2D memcpy_2d{};
      memcpy_2d.srcMemoryType = CU_MEMORYTYPE_DEVICE;
      memcpy_2d.srcDevice = device_ptr[plane];
      memcpy_2d.srcPitch = src_pitch;
      memcpy_2d.dstMemoryType = CU_MEMORYTYPE_DEVICE;
      memcpy_2d.dstDevice = upload_buffers_[plane]->device_ptr_.get();
      memcpy_2d.dstPitch = memcpy_2d.WidthInBytes = width * hw_channels * component_size;
      memcpy_2d.Height = height;
      CudaCheck(cuMemcpy2DAsync(&memcpy_2d, stream));

      upload_buffers_[plane]->end_access_with_cuda(stream);
      buffers[plane] = upload_buffers_[plane].get();
    }
  }

  if (!mipmaps_.empty()) {
    // indicate that the texture had been used by CUDA
    end_access_with_cuda(stream);
  } else {
    vulkan_->upload_to_texture(this, buffers);
  }

  CudaService::sync_with_selected_stream(ext_stream, stream);
}

}  // namespace holoscan::viz

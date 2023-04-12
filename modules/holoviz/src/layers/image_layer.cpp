/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "image_layer.hpp"

#include <stdexcept>
#include <vector>

#include <nvvk/resourceallocator_vk.hpp>

#include "../context.hpp"
#include "../vulkan/vulkan_app.hpp"

namespace holoscan::viz {

struct ImageLayer::Impl {
  bool can_be_reused(Impl& other) const {
    // we can reuse if the format/size and LUT match and
    //  if we did not switch from host to device memory and vice versa
    if ((format_ == other.format_) && (width_ == other.width_) && (height_ == other.height_) &&
        (lut_size_ == other.lut_size_) && (lut_format_ == other.lut_format_) &&
        (lut_data_ == other.lut_data_) && (lut_normalized_ == other.lut_normalized_) &&
        ((host_ptr_ != nullptr) == (other.device_ptr_ == 0)) &&
        ((device_ptr_ != 0) == (other.host_ptr_ == nullptr))) {
      // update the host pointer, Cuda device pointer and the cuda stream.
      // Data will be uploaded when drawing regardless if the layer is reused or not
      /// @todo this should be made explicit, first check if the layer can be reused and then
      ///     update the reused layer with these properties below which don't prevent reusing
      other.host_ptr_ = host_ptr_;
      other.device_ptr_ = device_ptr_;
      other.cuda_stream_ = cuda_stream_;
      return true;
    }

    return false;
  }

  // user provided state
  ImageFormat format_ = ImageFormat(-1);
  uint32_t width_ = 0;
  uint32_t height_ = 0;
  CUdeviceptr device_ptr_ = 0;
  const void* host_ptr_ = nullptr;
  CUstream cuda_stream_ = 0;

  uint32_t lut_size_ = 0;
  ImageFormat lut_format_ = ImageFormat(-1);
  std::vector<uint8_t> lut_data_;
  bool lut_normalized_ = false;

  // internal state
  Vulkan* vulkan_ = nullptr;
  Vulkan::Texture* texture_ = nullptr;
  Vulkan::Texture* lut_texture_ = nullptr;
};

ImageLayer::ImageLayer() : Layer(Type::Image), impl_(new ImageLayer::Impl) {}

ImageLayer::~ImageLayer() {
  if (impl_->vulkan_) {
    if (impl_->texture_) { impl_->vulkan_->destroy_texture(impl_->texture_); }
    if (impl_->lut_texture_) { impl_->vulkan_->destroy_texture(impl_->lut_texture_); }
  }
}

void ImageLayer::image_cuda_device(uint32_t width, uint32_t height, ImageFormat fmt,
                                   CUdeviceptr device_ptr) {
  if (impl_->host_ptr_) {
    throw std::runtime_error("Can't simultaneously specify device and host image for a layer.");
  }
  impl_->width_ = width;
  impl_->height_ = height;
  impl_->format_ = fmt;
  impl_->device_ptr_ = device_ptr;
  impl_->cuda_stream_ = Context::get().get_cuda_stream();
}

void ImageLayer::image_cuda_array(ImageFormat fmt, CUarray array) {
  throw std::runtime_error("Not implemented");
}

void ImageLayer::image_host(uint32_t width, uint32_t height, ImageFormat fmt, const void* data) {
  if (impl_->device_ptr_) {
    throw std::runtime_error("Can't simultaneously specify device and host image for a layer.");
  }
  impl_->width_ = width;
  impl_->height_ = height;
  impl_->format_ = fmt;
  impl_->host_ptr_ = data;
}

void ImageLayer::lut(uint32_t size, ImageFormat fmt, size_t data_size, const void* data,
                     bool normalized) {
  impl_->lut_size_ = size;
  impl_->lut_format_ = fmt;
  impl_->lut_data_.assign(reinterpret_cast<const uint8_t*>(data),
                          reinterpret_cast<const uint8_t*>(data) + data_size);
  impl_->lut_normalized_ = normalized;
}

bool ImageLayer::can_be_reused(Layer& other) const {
  return Layer::can_be_reused(other) &&
         impl_->can_be_reused(*static_cast<const ImageLayer&>(other).impl_.get());
}

void ImageLayer::end(Vulkan* vulkan) {
  if (impl_->device_ptr_) {
    // check if this is a reused layer, in this case
    //  we just have to upload the data to the texture
    if (!impl_->texture_) {
      /// @todo need to remember Vulkan instance for destroying texture,
      ///       destroy should probably be handled by Vulkan class
      impl_->vulkan_ = vulkan;

      // check if we have a lut, if yes, the texture needs to
      //  be nearest sampled since it has index values
      const bool has_lut = !impl_->lut_data_.empty();

      // create a texture to which we can upload from CUDA
      impl_->texture_ = vulkan->create_texture_for_cuda_interop(
          impl_->width_,
          impl_->height_,
          impl_->format_,
          has_lut ? vk::Filter::eNearest : vk::Filter::eLinear);
    }
    vulkan->upload_to_texture(impl_->device_ptr_, impl_->texture_, impl_->cuda_stream_);
  } else if (impl_->host_ptr_) {
    // check if this is a reused layer,
    //  in this case we just have to upload the data to the texture
    if (!impl_->texture_) {
      /// @todo need to remember Vulkan instance for destroying texture,
      ///       destroy should probably be handled by Vulkan class
      impl_->vulkan_ = vulkan;

      // check if we have a lut, if yes, the texture needs to be
      //  nearest sampled since it has index values
      const bool has_lut = !impl_->lut_data_.empty();

      // create a texture to which we can upload from CUDA
      impl_->texture_ =
          vulkan->create_texture(impl_->width_,
                                 impl_->height_,
                                 impl_->format_,
                                 0,
                                 nullptr,
                                 has_lut ? vk::Filter::eNearest : vk::Filter::eLinear);
    }
    vulkan->upload_to_texture(impl_->host_ptr_, impl_->texture_);
  }

  if (!impl_->lut_data_.empty() && !impl_->lut_texture_) {
    // create LUT texture
    impl_->lut_texture_ =
        vulkan->create_texture(impl_->lut_size_,
                               1,
                               impl_->lut_format_,
                               impl_->lut_data_.size(),
                               impl_->lut_data_.data(),
                               impl_->lut_normalized_ ? vk::Filter::eLinear : vk::Filter::eNearest,
                               impl_->lut_normalized_);
  }
}

void ImageLayer::render(Vulkan* vulkan) {
  if (impl_->texture_) {
    // draw
    vulkan->draw_texture(impl_->texture_, impl_->lut_texture_, get_opacity());
  }
}

}  // namespace holoscan::viz

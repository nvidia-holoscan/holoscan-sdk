/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <array>
#include <memory>
#include <stdexcept>
#include <vector>

#include <nvvk/resourceallocator_vk.hpp>

#include "../context.hpp"
#include "../vulkan/format_util.hpp"
#include "../vulkan/texture.hpp"
#include "../vulkan/vulkan_app.hpp"

namespace holoscan::viz {

class SourceData {
 public:
  SourceData() = default;

  /**
   * @returns true if the type of source (device/host) is the same (or all sources are null)
   */
  bool same_type(const SourceData& other) const {
    return (!has_host_memory() && !has_device_memory()) ||
           ((has_host_memory() == !other.has_device_memory()) &&
            (has_device_memory() == !other.has_host_memory()));
  }

  bool has_host_memory() const { return (host_ptr_[0] != nullptr); }

  bool has_device_memory() const { return (device_ptr_[0] != 0); }

  std::array<CUdeviceptr, 3> device_ptr_{};
  std::array<const void*, 3> host_ptr_{};
  std::array<size_t, 3> row_pitch_{};
};

struct ImageLayer::Impl {
  bool can_be_reused(Impl& other) const {
    // we can reuse if the format/component mapping/size and LUT match and
    //  if we did not switch from host to device memory and vice versa
    if ((format_ == other.format_) && (component_mapping_ == other.component_mapping_) &&
        (ycbcr_model_conversion_ == other.ycbcr_model_conversion_) &&
        (ycbcr_range_ == other.ycbcr_range_) && (x_chroma_location_ == other.x_chroma_location_) &&
        (y_chroma_location_ == other.y_chroma_location_) && (width_ == other.width_) &&
        (height_ == other.height_) && source_data_.same_type(other.source_data_) &&
        (depth_format_ == other.depth_format_) && (depth_width_ == other.depth_width_) &&
        (depth_height_ == other.depth_height_) &&
        ((!depth_host_ptr_ && !depth_device_ptr_) ||
         (((depth_host_ptr_ != nullptr) == (other.depth_device_ptr_ == 0)) &&
          ((depth_device_ptr_ != 0) == (other.depth_host_ptr_ == nullptr)))) &&
        (lut_size_ == other.lut_size_) && (lut_format_ == other.lut_format_) &&
        (lut_data_ == other.lut_data_) && (lut_normalized_ == other.lut_normalized_)) {
      // update the host pointer, Cuda device pointer, row pitch and the cuda stream.
      // Data will be uploaded when drawing regardless if the layer is reused or not
      /// @todo this should be made explicit, first check if the layer can be reused and then
      ///     update the reused layer with these properties below which don't prevent reusing
      other.source_data_ = source_data_;
      other.cuda_stream_ = cuda_stream_;

      other.depth_host_ptr_ = depth_host_ptr_;
      other.depth_device_ptr_ = depth_device_ptr_;
      other.depth_row_pitch_ = depth_row_pitch_;
      other.depth_cuda_stream_ = depth_cuda_stream_;
      return true;
    }

    return false;
  }

  // user provided state
  ImageFormat format_ = ImageFormat(-1);
  vk::ComponentMapping component_mapping_;
  vk::SamplerYcbcrModelConversion ycbcr_model_conversion_ =
      vk::SamplerYcbcrModelConversion::eYcbcr601;
  vk::SamplerYcbcrRange ycbcr_range_ = vk::SamplerYcbcrRange::eItuFull;
  vk::ChromaLocation x_chroma_location_ = vk::ChromaLocation::eCositedEven;
  vk::ChromaLocation y_chroma_location_ = vk::ChromaLocation::eCositedEven;
  uint32_t width_ = 0;
  uint32_t height_ = 0;
  SourceData source_data_;
  CUstream cuda_stream_ = 0;

  ImageFormat depth_format_ = ImageFormat(-1);
  uint32_t depth_width_ = 0;
  uint32_t depth_height_ = 0;
  CUdeviceptr depth_device_ptr_ = 0;
  const void* depth_host_ptr_ = nullptr;
  size_t depth_row_pitch_ = 0;
  CUstream depth_cuda_stream_ = 0;

  uint32_t lut_size_ = 0;
  ImageFormat lut_format_ = ImageFormat(-1);
  std::vector<uint8_t> lut_data_;
  bool lut_normalized_ = false;

  // internal state
  std::unique_ptr<Texture> texture_;
  std::unique_ptr<Texture> depth_texture_;
  std::unique_ptr<Texture> lut_texture_;
};

ImageLayer::ImageLayer() : Layer(Type::Image), impl_(new ImageLayer::Impl) {}

ImageLayer::~ImageLayer() {}

void ImageLayer::image_cuda_device(uint32_t width, uint32_t height, ImageFormat fmt,
                                   CUdeviceptr device_ptr_plane_0, size_t row_pitch_plane_0,
                                   CUdeviceptr device_ptr_plane_1, size_t row_pitch_plane_1,
                                   CUdeviceptr device_ptr_plane_2, size_t row_pitch_plane_2) {
  // If a depth format is specified, use this image to write depth for the color image.
  if (is_depth_format(fmt)) {
    if (impl_->depth_host_ptr_) {
      throw std::runtime_error("Can't simultaneously specify device and host image for a layer.");
    }

    impl_->depth_width_ = width;
    impl_->depth_height_ = height;
    impl_->depth_format_ = ImageFormat::R32_SFLOAT;  // use a color format for sampling
    impl_->depth_device_ptr_ = device_ptr_plane_0;
    impl_->depth_cuda_stream_ = Context::get().get_cuda_stream();
    impl_->depth_row_pitch_ = row_pitch_plane_0;
    return;
  }

  if (impl_->source_data_.has_host_memory()) {
    throw std::runtime_error("Can't simultaneously specify device and host image for a layer.");
  }

  impl_->width_ = width;
  impl_->height_ = height;
  impl_->format_ = fmt;
  impl_->source_data_.device_ptr_[0] = device_ptr_plane_0;
  impl_->source_data_.row_pitch_[0] = row_pitch_plane_0;
  impl_->source_data_.device_ptr_[1] = device_ptr_plane_1;
  impl_->source_data_.row_pitch_[1] = row_pitch_plane_1;
  impl_->source_data_.device_ptr_[2] = device_ptr_plane_2;
  impl_->source_data_.row_pitch_[2] = row_pitch_plane_2;
  impl_->cuda_stream_ = Context::get().get_cuda_stream();
}

void ImageLayer::image_cuda_array(ImageFormat fmt, CUarray array) {
  throw std::runtime_error("Not implemented");
}

void ImageLayer::image_host(uint32_t width, uint32_t height, ImageFormat fmt,
                            const void* data_plane_0, size_t row_pitch_plane_0,
                            const void* data_plane_1, size_t row_pitch_plane_1,
                            const void* data_plane_2, size_t row_pitch_plane_2) {
  // If a depth format is specified, use this image to write depth for the color image.
  if (is_depth_format(fmt)) {
    if (impl_->depth_device_ptr_) {
      throw std::runtime_error("Can't simultaneously specify device and host image for a layer.");
    }

    impl_->depth_width_ = width;
    impl_->depth_height_ = height;
    impl_->depth_format_ = ImageFormat::R32_SFLOAT;  // use a color format for sampling
    impl_->depth_host_ptr_ = data_plane_0;
    impl_->depth_cuda_stream_ = Context::get().get_cuda_stream();
    impl_->depth_row_pitch_ = row_pitch_plane_0;
    return;
  }

  if (impl_->source_data_.has_device_memory()) {
    throw std::runtime_error("Can't simultaneously specify device and host image for a layer.");
  }

  impl_->width_ = width;
  impl_->height_ = height;
  impl_->format_ = fmt;
  impl_->source_data_.host_ptr_[0] = data_plane_0;
  impl_->source_data_.row_pitch_[0] = row_pitch_plane_0;
  impl_->source_data_.host_ptr_[1] = data_plane_1;
  impl_->source_data_.row_pitch_[1] = row_pitch_plane_1;
  impl_->source_data_.host_ptr_[2] = data_plane_2;
  impl_->source_data_.row_pitch_[2] = row_pitch_plane_2;
}

void ImageLayer::lut(uint32_t size, ImageFormat fmt, size_t data_size, const void* data,
                     bool normalized) {
  impl_->lut_size_ = size;
  impl_->lut_format_ = fmt;
  impl_->lut_data_.assign(reinterpret_cast<const uint8_t*>(data),
                          reinterpret_cast<const uint8_t*>(data) + data_size);
  impl_->lut_normalized_ = normalized;
}

void ImageLayer::image_component_mapping(ComponentSwizzle r, ComponentSwizzle g, ComponentSwizzle b,
                                         ComponentSwizzle a) {
  auto to_vk_swizzle = [](ComponentSwizzle in) -> vk::ComponentSwizzle {
    switch (in) {
      case ComponentSwizzle::IDENTITY:
        return vk::ComponentSwizzle::eIdentity;
      case ComponentSwizzle::ZERO:
        return vk::ComponentSwizzle::eZero;
      case ComponentSwizzle::ONE:
        return vk::ComponentSwizzle::eOne;
      case ComponentSwizzle::R:
        return vk::ComponentSwizzle::eR;
      case ComponentSwizzle::G:
        return vk::ComponentSwizzle::eG;
      case ComponentSwizzle::B:
        return vk::ComponentSwizzle::eB;
      case ComponentSwizzle::A:
        return vk::ComponentSwizzle::eA;
      default:
        throw std::runtime_error("Unhandled component swizzle.");
    }
  };
  impl_->component_mapping_.r = to_vk_swizzle(r);
  impl_->component_mapping_.g = to_vk_swizzle(g);
  impl_->component_mapping_.b = to_vk_swizzle(b);
  impl_->component_mapping_.a = to_vk_swizzle(a);
}

void ImageLayer::image_yuv_model_conversion(YuvModelConversion yuv_model_conversion) {
  switch (yuv_model_conversion) {
    case YuvModelConversion::YUV_601:
      impl_->ycbcr_model_conversion_ = vk::SamplerYcbcrModelConversion::eYcbcr601;
      break;
    case YuvModelConversion::YUV_709:
      impl_->ycbcr_model_conversion_ = vk::SamplerYcbcrModelConversion::eYcbcr709;
      break;
    case YuvModelConversion::YUV_2020:
      impl_->ycbcr_model_conversion_ = vk::SamplerYcbcrModelConversion::eYcbcr2020;
      break;
    default:
      throw std::runtime_error("Unhandled yuv model conversion.");
  }
}

void ImageLayer::image_yuv_range(YuvRange yuv_range) {
  switch (yuv_range) {
    case YuvRange::ITU_FULL:
      impl_->ycbcr_range_ = vk::SamplerYcbcrRange::eItuFull;
      break;
    case YuvRange::ITU_NARROW:
      impl_->ycbcr_range_ = vk::SamplerYcbcrRange::eItuNarrow;
      break;
    default:
      throw std::runtime_error("Unhandled yuv range.");
  }
}

void ImageLayer::image_chroma_location(ChromaLocation x_chroma_location,
                                       ChromaLocation y_chroma_location) {
  switch (x_chroma_location) {
    case ChromaLocation::COSITED_EVEN:
      impl_->x_chroma_location_ = vk::ChromaLocation::eCositedEven;
      break;
    case ChromaLocation::MIDPOINT:
      impl_->x_chroma_location_ = vk::ChromaLocation::eMidpoint;
      break;
    default:
      throw std::runtime_error("Unhandled chroma location.");
  }

  switch (y_chroma_location) {
    case ChromaLocation::COSITED_EVEN:
      impl_->y_chroma_location_ = vk::ChromaLocation::eCositedEven;
      break;
    case ChromaLocation::MIDPOINT:
      impl_->y_chroma_location_ = vk::ChromaLocation::eMidpoint;
      break;
    default:
      throw std::runtime_error("Unhandled chroma location.");
  }
}

bool ImageLayer::can_be_reused(Layer& other) const {
  return Layer::can_be_reused(other) &&
         impl_->can_be_reused(*static_cast<const ImageLayer&>(other).impl_.get());
}

void ImageLayer::end(Vulkan* vulkan) {
  if (impl_->source_data_.has_device_memory() || impl_->source_data_.has_host_memory()) {
    // check if this is a reused layer, in this case
    //  we just have to upload the data to the texture
    if (!impl_->texture_) {
      // check if we have a lut, if yes, the texture needs to
      //  be nearest sampled since it has index values
      const bool has_lut = !impl_->lut_data_.empty();

      Vulkan::CreateTextureArgs args;
      args.width_ = impl_->width_;
      args.height_ = impl_->height_;
      args.format_ = impl_->format_;
      args.component_mapping_ = impl_->component_mapping_;
      args.filter_ = has_lut ? vk::Filter::eNearest : vk::Filter::eLinear;
      args.cuda_interop_ = impl_->source_data_.has_device_memory();
      args.ycbcr_model_conversion_ = impl_->ycbcr_model_conversion_;
      args.x_chroma_location_ = impl_->x_chroma_location_;
      args.y_chroma_location_ = impl_->y_chroma_location_;
      args.ycbcr_range_ = impl_->ycbcr_range_;

      // create a texture to which we can upload from CUDA
      impl_->texture_ = vulkan->create_texture(args);
    }

    if (impl_->source_data_.has_device_memory()) {
      impl_->texture_->upload(
          impl_->cuda_stream_, impl_->source_data_.device_ptr_, impl_->source_data_.row_pitch_);
    } else {
      assert(impl_->source_data_.has_host_memory());
      vulkan->upload_to_texture(
          impl_->texture_.get(), impl_->source_data_.host_ptr_, impl_->source_data_.row_pitch_);
    }
  }

  if (impl_->depth_device_ptr_ || impl_->depth_host_ptr_) {
    // check if this is a reused layer, in this case
    //  we just have to upload the data to the texture
    if (!impl_->depth_texture_) {
      // create a texture to which we can upload from CUDA
      Vulkan::CreateTextureArgs args;
      args.width_ = impl_->depth_width_;
      args.height_ = impl_->depth_height_;
      args.format_ = impl_->depth_format_;
      args.filter_ = vk::Filter::eLinear;
      args.cuda_interop_ = impl_->depth_device_ptr_ != 0;

      impl_->depth_texture_ = vulkan->create_texture(args);
    }

    if (impl_->depth_device_ptr_) {
      impl_->depth_texture_->upload(
          impl_->depth_cuda_stream_, {impl_->depth_device_ptr_}, {impl_->depth_row_pitch_});
    } else {
      assert(impl_->depth_host_ptr_);
      vulkan->upload_to_texture(
          impl_->depth_texture_.get(), {impl_->depth_host_ptr_}, {impl_->depth_row_pitch_});
    }
  }

  if (!impl_->lut_data_.empty() && !impl_->lut_texture_) {
    // create LUT texture
    Vulkan::CreateTextureArgs args;
    args.width_ = impl_->lut_size_;
    args.height_ = 1;
    args.format_ = impl_->lut_format_;
    args.filter_ = impl_->lut_normalized_ ? vk::Filter::eLinear : vk::Filter::eNearest;
    args.normalized_ = impl_->lut_normalized_;

    impl_->lut_texture_ = vulkan->create_texture(args);

    vulkan->upload_to_texture(
        impl_->lut_texture_.get(), {impl_->lut_data_.data()}, {impl_->lut_data_.size()});
  }
}

void ImageLayer::render(Vulkan* vulkan) {
  if (impl_->texture_) {
    std::vector<Layer::View> views = get_views();
    if (views.empty()) {
      views.push_back(Layer::View());
    }

    for (const View& view : views) {
      vulkan->set_viewport(view.offset_x, view.offset_y, view.width, view.height);
      nvmath::mat4f view_matrix;
      if (view.matrix.has_value()) {
        view_matrix = view.matrix.value();
      } else {
        view_matrix = nvmath::mat4f(1);
      }
      // draw
      vulkan->draw_texture(impl_->texture_.get(),
                           impl_->depth_texture_.get(),
                           impl_->lut_texture_.get(),
                           get_opacity(),
                           view_matrix);
    }
  }
}

}  // namespace holoscan::viz

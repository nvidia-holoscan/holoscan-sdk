/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/operators/holoviz/buffer_info.hpp"

#include <tuple>

#include "holoscan/logger/logger.hpp"

#include "gxf/multimedia/video.hpp"

namespace viz = holoscan::viz;

namespace holoscan::ops {

static std::tuple<uint32_t, viz::ComponentSwizzle, viz::ComponentSwizzle, viz::ComponentSwizzle,
                  viz::ComponentSwizzle>
component_and_swizzle(HolovizOp::ImageFormat image_format) {
  uint32_t components = 0;
  viz::ComponentSwizzle component_swizzle[4]{};

  switch (image_format) {
    case HolovizOp::ImageFormat::R8_UINT:
    case HolovizOp::ImageFormat::R8_SINT:
    case HolovizOp::ImageFormat::R8_UNORM:
    case HolovizOp::ImageFormat::R8_SNORM:
    case HolovizOp::ImageFormat::R8_SRGB:
    case HolovizOp::ImageFormat::R16_UINT:
    case HolovizOp::ImageFormat::R16_SINT:
    case HolovizOp::ImageFormat::R16_UNORM:
    case HolovizOp::ImageFormat::R16_SNORM:
    case HolovizOp::ImageFormat::R32_UINT:
    case HolovizOp::ImageFormat::R32_SINT:
    case HolovizOp::ImageFormat::R32_SFLOAT:
    case HolovizOp::ImageFormat::D32_SFLOAT:
    case HolovizOp::ImageFormat::D16_UNORM:
      components = 1;
      component_swizzle[0] = viz::ComponentSwizzle::R;
      component_swizzle[1] = viz::ComponentSwizzle::R;
      component_swizzle[2] = viz::ComponentSwizzle::R;
      component_swizzle[3] = viz::ComponentSwizzle::ONE;
      break;
    case HolovizOp::ImageFormat::X8_D24_UNORM:
      components = 1;
      component_swizzle[0] = viz::ComponentSwizzle::G;
      component_swizzle[1] = viz::ComponentSwizzle::G;
      component_swizzle[2] = viz::ComponentSwizzle::G;
      component_swizzle[3] = viz::ComponentSwizzle::ONE;
      break;
    case HolovizOp::ImageFormat::R8G8B8_UNORM:
    case HolovizOp::ImageFormat::R8G8B8_SNORM:
    case HolovizOp::ImageFormat::R8G8B8_SRGB:
    case HolovizOp::ImageFormat::Y8U8Y8V8_422_UNORM:
    case HolovizOp::ImageFormat::U8Y8V8Y8_422_UNORM:
    case HolovizOp::ImageFormat::Y8_U8V8_2PLANE_420_UNORM:
    case HolovizOp::ImageFormat::Y8_U8V8_2PLANE_422_UNORM:
    case HolovizOp::ImageFormat::Y8_U8_V8_3PLANE_420_UNORM:
    case HolovizOp::ImageFormat::Y8_U8_V8_3PLANE_422_UNORM:
    case HolovizOp::ImageFormat::Y16_U16V16_2PLANE_420_UNORM:
    case HolovizOp::ImageFormat::Y16_U16V16_2PLANE_422_UNORM:
    case HolovizOp::ImageFormat::Y16_U16_V16_3PLANE_420_UNORM:
    case HolovizOp::ImageFormat::Y16_U16_V16_3PLANE_422_UNORM:
      components = 3;
      component_swizzle[0] = viz::ComponentSwizzle::IDENTITY;
      component_swizzle[1] = viz::ComponentSwizzle::IDENTITY;
      component_swizzle[2] = viz::ComponentSwizzle::IDENTITY;
      component_swizzle[3] = viz::ComponentSwizzle::ONE;
      break;
    case HolovizOp::ImageFormat::R8G8B8A8_UNORM:
    case HolovizOp::ImageFormat::R8G8B8A8_SNORM:
    case HolovizOp::ImageFormat::R8G8B8A8_SRGB:
    case HolovizOp::ImageFormat::R16G16B16A16_UNORM:
    case HolovizOp::ImageFormat::R16G16B16A16_SNORM:
    case HolovizOp::ImageFormat::R16G16B16A16_SFLOAT:
    case HolovizOp::ImageFormat::R32G32B32A32_SFLOAT:
    case HolovizOp::ImageFormat::B8G8R8A8_UNORM:
    case HolovizOp::ImageFormat::B8G8R8A8_SRGB:
      components = 4;
      component_swizzle[0] = viz::ComponentSwizzle::IDENTITY;
      component_swizzle[1] = viz::ComponentSwizzle::IDENTITY;
      component_swizzle[2] = viz::ComponentSwizzle::IDENTITY;
      component_swizzle[3] = viz::ComponentSwizzle::IDENTITY;
      break;
    case HolovizOp::ImageFormat::A2B10G10R10_UNORM_PACK32:
    case HolovizOp::ImageFormat::A2R10G10B10_UNORM_PACK32:
    case HolovizOp::ImageFormat::A8B8G8R8_UNORM_PACK32:
    case HolovizOp::ImageFormat::A8B8G8R8_SRGB_PACK32:
      components = 1;
      component_swizzle[0] = viz::ComponentSwizzle::IDENTITY;
      component_swizzle[1] = viz::ComponentSwizzle::IDENTITY;
      component_swizzle[2] = viz::ComponentSwizzle::IDENTITY;
      component_swizzle[3] = viz::ComponentSwizzle::IDENTITY;
      break;
    default:
      throw std::runtime_error(fmt::format("Unhandled image format {}", int(image_format)));
  }

  return {components,
          component_swizzle[0],
          component_swizzle[1],
          component_swizzle[2],
          component_swizzle[3]};
}

gxf_result_t BufferInfo::init(const nvidia::gxf::Handle<nvidia::gxf::Tensor>& tensor,
                              HolovizOp::ImageFormat input_image_format) {
  rank = tensor->rank();
  // validate row-major memory layout
  int32_t last_axis = rank - 1;
  for (auto axis = last_axis; axis > 0; --axis) {
    if (tensor->stride(axis) > tensor->stride(axis - 1)) {
      HOLOSCAN_LOG_ERROR("Tensor must have a row-major memory layout (C-contiguous memory order).");
      return GXF_INVALID_DATA_FORMAT;
    }
  }

  /// @todo this is assuming HWC, should either auto-detect (if possible) or make user
  /// configurable
  components = tensor->shape().dimension(tensor->rank() - 1);
  stride[2] = tensor->stride(tensor->rank() - 1);
  if (tensor->rank() > 1) {
    width = tensor->shape().dimension(tensor->rank() - 2);
    stride[1] = tensor->stride(tensor->rank() - 2);
    if (tensor->rank() > 2) {
      height = tensor->shape().dimension(tensor->rank() - 3);
      stride[0] = tensor->stride(tensor->rank() - 3);
    } else {
      height = 1;
      stride[0] = stride[1];
    }
  } else {
    width = 1;
    stride[1] = stride[2];
  }
  element_type = tensor->element_type();
  name = tensor.name();
  buffer_ptr = tensor->pointer();
  storage_type = tensor->storage_type();
  bytes_size = tensor->bytes_size();

  // get image format for 2D tensors
  if (input_image_format == HolovizOp::ImageFormat::AUTO_DETECT) {
    if (rank == 3) {
      struct Format {
        nvidia::gxf::PrimitiveType type_;
        int32_t channels_;
        HolovizOp::ImageFormat format_;
      };
      constexpr Format kGFXToHolovizFormats[] = {
          {nvidia::gxf::PrimitiveType::kUnsigned8, 1, HolovizOp::ImageFormat::R8_UNORM},
          {nvidia::gxf::PrimitiveType::kInt8, 1, HolovizOp::ImageFormat::R8_SNORM},
          {nvidia::gxf::PrimitiveType::kUnsigned16, 1, HolovizOp::ImageFormat::R16_UNORM},
          {nvidia::gxf::PrimitiveType::kInt16, 1, HolovizOp::ImageFormat::R16_SNORM},
          {nvidia::gxf::PrimitiveType::kUnsigned32, 1, HolovizOp::ImageFormat::R32_UINT},
          {nvidia::gxf::PrimitiveType::kInt32, 1, HolovizOp::ImageFormat::R32_SINT},
          {nvidia::gxf::PrimitiveType::kFloat32, 1, HolovizOp::ImageFormat::R32_SFLOAT},
          {nvidia::gxf::PrimitiveType::kUnsigned8, 3, HolovizOp::ImageFormat::R8G8B8_UNORM},
          {nvidia::gxf::PrimitiveType::kInt8, 3, HolovizOp::ImageFormat::R8G8B8_SNORM},
          {nvidia::gxf::PrimitiveType::kUnsigned8, 4, HolovizOp::ImageFormat::R8G8B8A8_UNORM},
          {nvidia::gxf::PrimitiveType::kInt8, 4, HolovizOp::ImageFormat::R8G8B8A8_SNORM},
          {nvidia::gxf::PrimitiveType::kUnsigned16, 4, HolovizOp::ImageFormat::R16G16B16A16_UNORM},
          {nvidia::gxf::PrimitiveType::kInt16, 4, HolovizOp::ImageFormat::R16G16B16A16_SNORM},
          {nvidia::gxf::PrimitiveType::kFloat32, 4, HolovizOp::ImageFormat::R32G32B32A32_SFLOAT}};

      for (auto&& format : kGFXToHolovizFormats) {
        if ((format.type_ == element_type) && (format.channels_ == int32_t(components))) {
          image_format = format.format_;
          break;
        }
      }
    }
    if (image_format == HolovizOp::ImageFormat::AUTO_DETECT) {
      // It's not an error if we can't auto-detect the image format of a tensor, this can be
      // a tensor used for drawing a graphics primitive.
      return GXF_SUCCESS;
    }
  } else {
    image_format = input_image_format;
  }

  auto result = component_and_swizzle(image_format);
  if (components != std::get<0>(result)) {
    HOLOSCAN_LOG_WARN(
        "Image format '{}' with component count '{}' mismatches tensor '{}' with component count "
        "'{}'",
        int(image_format),
        std::get<0>(result),
        tensor.name(),
        components);
  }
  component_swizzle[0] = std::get<1>(result);
  component_swizzle[1] = std::get<2>(result);
  component_swizzle[2] = std::get<3>(result);
  component_swizzle[3] = std::get<4>(result);

  return GXF_SUCCESS;
}

gxf_result_t BufferInfo::init(const nvidia::gxf::Handle<nvidia::gxf::VideoBuffer>& video,
                              HolovizOp::ImageFormat input_image_format) {
  // NOTE: VideoBuffer::moveToTensor() converts VideoBuffer instance to the Tensor instance
  // with an unexpected shape:  [width, height] or [width, height, num_planes].
  // And, if we use moveToTensor() to convert VideoBuffer to Tensor, we may lose the original
  // video buffer when the VideoBuffer instance is used in other places. For that reason, we
  // directly access internal data of VideoBuffer instance to access Tensor data.
  const auto& buffer_info = video->video_frame_info();

  // here auto detect means to use the image format of the video buffer
  if (input_image_format == HolovizOp::ImageFormat::AUTO_DETECT) {
    struct Format {
      nvidia::gxf::VideoFormat color_format_;
      nvidia::gxf::PrimitiveType element_type_;
      int32_t channels_;
      HolovizOp::ImageFormat format_;
      viz::ComponentSwizzle component_swizzle[4];
    };
    constexpr Format kVideoToHolovizFormats[] = {
        {nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY,
         nvidia::gxf::PrimitiveType::kUnsigned8,
         1,
         HolovizOp::ImageFormat::R8_UNORM,
         {viz::ComponentSwizzle::R,
          viz::ComponentSwizzle::R,
          viz::ComponentSwizzle::R,
          viz::ComponentSwizzle::ONE}},
        {nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY16,
         nvidia::gxf::PrimitiveType::kUnsigned16,
         1,
         HolovizOp::ImageFormat::R16_UNORM,
         {viz::ComponentSwizzle::R,
          viz::ComponentSwizzle::R,
          viz::ComponentSwizzle::R,
          viz::ComponentSwizzle::ONE}},
        {nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY32F,
         nvidia::gxf::PrimitiveType::kFloat32,
         1,
         HolovizOp::ImageFormat::R32_SFLOAT,
         {viz::ComponentSwizzle::R,
          viz::ComponentSwizzle::R,
          viz::ComponentSwizzle::R,
          viz::ComponentSwizzle::ONE}},
        {nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_D32F,
         nvidia::gxf::PrimitiveType::kFloat32,
         1,
         HolovizOp::ImageFormat::D32_SFLOAT,
         {viz::ComponentSwizzle::IDENTITY,
          viz::ComponentSwizzle::IDENTITY,
          viz::ComponentSwizzle::IDENTITY,
          viz::ComponentSwizzle::IDENTITY}},
        {nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB,
         nvidia::gxf::PrimitiveType::kUnsigned8,
         3,
         HolovizOp::ImageFormat::R8G8B8_UNORM,
         {viz::ComponentSwizzle::IDENTITY,
          viz::ComponentSwizzle::IDENTITY,
          viz::ComponentSwizzle::IDENTITY,
          viz::ComponentSwizzle::IDENTITY}},
        {nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_BGR,
         nvidia::gxf::PrimitiveType::kUnsigned8,
         3,
         HolovizOp::ImageFormat::R8G8B8_UNORM,
         {viz::ComponentSwizzle::B,
          viz::ComponentSwizzle::G,
          viz::ComponentSwizzle::R,
          viz::ComponentSwizzle::IDENTITY}},
        {nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA,
         nvidia::gxf::PrimitiveType::kUnsigned8,
         4,
         HolovizOp::ImageFormat::R8G8B8A8_UNORM,
         {viz::ComponentSwizzle::IDENTITY,
          viz::ComponentSwizzle::IDENTITY,
          viz::ComponentSwizzle::IDENTITY,
          viz::ComponentSwizzle::IDENTITY}},
        {nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_BGRA,
         nvidia::gxf::PrimitiveType::kUnsigned8,
         4,
         HolovizOp::ImageFormat::R8G8B8A8_UNORM,
         {viz::ComponentSwizzle::B,
          viz::ComponentSwizzle::G,
          viz::ComponentSwizzle::R,
          viz::ComponentSwizzle::A}},
        {nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_ARGB,
         nvidia::gxf::PrimitiveType::kUnsigned8,
         4,
         HolovizOp::ImageFormat::R8G8B8A8_UNORM,
         {viz::ComponentSwizzle::A,
          viz::ComponentSwizzle::R,
          viz::ComponentSwizzle::G,
          viz::ComponentSwizzle::B}},
        {nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_ABGR,
         nvidia::gxf::PrimitiveType::kUnsigned8,
         4,
         HolovizOp::ImageFormat::R8G8B8A8_UNORM,
         {viz::ComponentSwizzle::A,
          viz::ComponentSwizzle::B,
          viz::ComponentSwizzle::G,
          viz::ComponentSwizzle::R}},
        {nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBX,
         nvidia::gxf::PrimitiveType::kUnsigned8,
         4,
         HolovizOp::ImageFormat::R8G8B8A8_UNORM,
         {viz::ComponentSwizzle::R,
          viz::ComponentSwizzle::G,
          viz::ComponentSwizzle::B,
          viz::ComponentSwizzle::ONE}},
        {nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_BGRX,
         nvidia::gxf::PrimitiveType::kUnsigned8,
         4,
         HolovizOp::ImageFormat::R8G8B8A8_UNORM,
         {viz::ComponentSwizzle::B,
          viz::ComponentSwizzle::G,
          viz::ComponentSwizzle::R,
          viz::ComponentSwizzle::ONE}},
        {nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_XRGB,
         nvidia::gxf::PrimitiveType::kUnsigned8,
         4,
         HolovizOp::ImageFormat::R8G8B8A8_UNORM,
         {viz::ComponentSwizzle::G,
          viz::ComponentSwizzle::B,
          viz::ComponentSwizzle::A,
          viz::ComponentSwizzle::ONE}},
        {nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_XBGR,
         nvidia::gxf::PrimitiveType::kUnsigned8,
         4,
         HolovizOp::ImageFormat::R8G8B8A8_UNORM,
         {viz::ComponentSwizzle::A,
          viz::ComponentSwizzle::B,
          viz::ComponentSwizzle::G,
          viz::ComponentSwizzle::ONE}},
    };

    for (auto&& format : kVideoToHolovizFormats) {
      if (format.color_format_ == buffer_info.color_format) {
        element_type = format.element_type_;
        components = format.channels_;
        image_format = format.format_;
        component_swizzle[0] = format.component_swizzle[0];
        component_swizzle[1] = format.component_swizzle[1];
        component_swizzle[2] = format.component_swizzle[2];
        component_swizzle[3] = format.component_swizzle[3];
        break;
      }
    }

    if (image_format == HolovizOp::ImageFormat::AUTO_DETECT) {
      struct YuvFormat {
        nvidia::gxf::VideoFormat color_format_;
        HolovizOp::ImageFormat format_;
        HolovizOp::YuvModelConversion yuv_model_conversion_;
        HolovizOp::YuvRange yuv_range_;
      };
      constexpr YuvFormat kYuvVideoToHolovizFormats[] = {
          {nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_YUV420,
          HolovizOp::ImageFormat::Y8_U8_V8_3PLANE_420_UNORM,
          HolovizOp::YuvModelConversion::YUV_601,
          HolovizOp::YuvRange::ITU_NARROW},
          {nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_YUV420_ER,
          HolovizOp::ImageFormat::Y8_U8_V8_3PLANE_420_UNORM,
          HolovizOp::YuvModelConversion::YUV_601,
          HolovizOp::YuvRange::ITU_FULL},
          {nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_YUV420_709,
          HolovizOp::ImageFormat::Y8_U8_V8_3PLANE_420_UNORM,
          HolovizOp::YuvModelConversion::YUV_709,
          HolovizOp::YuvRange::ITU_NARROW},
          {nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_YUV420_709_ER,
          HolovizOp::ImageFormat::Y8_U8_V8_3PLANE_420_UNORM,
          HolovizOp::YuvModelConversion::YUV_709,
          HolovizOp::YuvRange::ITU_FULL},
          {nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12,
          HolovizOp::ImageFormat::Y8_U8V8_2PLANE_420_UNORM,
          HolovizOp::YuvModelConversion::YUV_601,
          HolovizOp::YuvRange::ITU_NARROW},
          {nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12_ER,
          HolovizOp::ImageFormat::Y8_U8V8_2PLANE_420_UNORM,
          HolovizOp::YuvModelConversion::YUV_601,
          HolovizOp::YuvRange::ITU_FULL},
          {nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12_709,
          HolovizOp::ImageFormat::Y8_U8V8_2PLANE_420_UNORM,
          HolovizOp::YuvModelConversion::YUV_709,
          HolovizOp::YuvRange::ITU_NARROW},
          {nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12_709_ER,
          HolovizOp::ImageFormat::Y8_U8V8_2PLANE_420_UNORM,
          HolovizOp::YuvModelConversion::YUV_709,
          HolovizOp::YuvRange::ITU_FULL},
      };

      for (auto&& format : kYuvVideoToHolovizFormats) {
        if (format.color_format_ == buffer_info.color_format) {
          element_type = nvidia::gxf::PrimitiveType::kUnsigned8;
          components = 3;
          image_format = format.format_;
          component_swizzle[0] = viz::ComponentSwizzle::IDENTITY;
          component_swizzle[1] = viz::ComponentSwizzle::IDENTITY;
          component_swizzle[2] = viz::ComponentSwizzle::IDENTITY;
          component_swizzle[3] = viz::ComponentSwizzle::ONE;
          yuv_model_conversion = format.yuv_model_conversion_;
          yuv_range = format.yuv_range_;
          break;
        }
      }
    }

    if (image_format == HolovizOp::ImageFormat::AUTO_DETECT) {
      HOLOSCAN_LOG_ERROR("Video buffer '{}': unsupported input format: '{}'\n",
                         video.name(),
                         static_cast<int64_t>(buffer_info.color_format));
      return GXF_FAILURE;
    }
  } else {
    image_format = input_image_format;
    auto result = component_and_swizzle(image_format);
    components = std::get<0>(result);
    component_swizzle[0] = std::get<1>(result);
    component_swizzle[1] = std::get<2>(result);
    component_swizzle[2] = std::get<3>(result);
    component_swizzle[3] = std::get<4>(result);
  }

  rank = 3;
  width = buffer_info.width;
  height = buffer_info.height;
  name = video.name();
  buffer_ptr = video->pointer();
  storage_type = video->storage_type();
  bytes_size = video->size();
  stride[0] = buffer_info.color_planes[0].stride;
  stride[1] = components;
  stride[2] = PrimitiveTypeSize(element_type);

  color_planes = buffer_info.color_planes;

  return GXF_SUCCESS;
}

}  // namespace holoscan::ops

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

#include "holoscan/operators/holoviz/buffer_info.hpp"

#include "holoscan/logger/logger.hpp"

#include "gxf/multimedia/video.hpp"

namespace viz = holoscan::viz;

namespace holoscan::ops {

gxf_result_t BufferInfo::init(const nvidia::gxf::Handle<nvidia::gxf::Tensor>& tensor) {
  rank = tensor->rank();
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
  if (rank == 3) {
    struct Format {
      nvidia::gxf::PrimitiveType type_;
      int32_t channels_;
      viz::ImageFormat format_;
      viz::ComponentSwizzle component_swizzle[4];
    };
    constexpr Format kGFXToHolovizFormats[] = {{nvidia::gxf::PrimitiveType::kUnsigned8,
                                                1,
                                                viz::ImageFormat::R8_UNORM,
                                                {viz::ComponentSwizzle::R,
                                                 viz::ComponentSwizzle::R,
                                                 viz::ComponentSwizzle::R,
                                                 viz::ComponentSwizzle::ONE}},
                                               {nvidia::gxf::PrimitiveType::kInt8,
                                                1,
                                                viz::ImageFormat::R8_SNORM,
                                                {viz::ComponentSwizzle::R,
                                                 viz::ComponentSwizzle::R,
                                                 viz::ComponentSwizzle::R,
                                                 viz::ComponentSwizzle::ONE}},
                                               {nvidia::gxf::PrimitiveType::kUnsigned16,
                                                1,
                                                viz::ImageFormat::R16_UNORM,
                                                {viz::ComponentSwizzle::R,
                                                 viz::ComponentSwizzle::R,
                                                 viz::ComponentSwizzle::R,
                                                 viz::ComponentSwizzle::ONE}},
                                               {nvidia::gxf::PrimitiveType::kInt16,
                                                1,
                                                viz::ImageFormat::R16_SNORM,
                                                {viz::ComponentSwizzle::R,
                                                 viz::ComponentSwizzle::R,
                                                 viz::ComponentSwizzle::R,
                                                 viz::ComponentSwizzle::ONE}},
                                               {nvidia::gxf::PrimitiveType::kUnsigned32,
                                                1,
                                                viz::ImageFormat::R32_UINT,
                                                {viz::ComponentSwizzle::R,
                                                 viz::ComponentSwizzle::R,
                                                 viz::ComponentSwizzle::R,
                                                 viz::ComponentSwizzle::ONE}},
                                               {nvidia::gxf::PrimitiveType::kInt32,
                                                1,
                                                viz::ImageFormat::R32_SINT,
                                                {viz::ComponentSwizzle::R,
                                                 viz::ComponentSwizzle::R,
                                                 viz::ComponentSwizzle::R,
                                                 viz::ComponentSwizzle::ONE}},
                                               {nvidia::gxf::PrimitiveType::kFloat32,
                                                1,
                                                viz::ImageFormat::R32_SFLOAT,
                                                {viz::ComponentSwizzle::R,
                                                 viz::ComponentSwizzle::R,
                                                 viz::ComponentSwizzle::R,
                                                 viz::ComponentSwizzle::ONE}},
                                               {nvidia::gxf::PrimitiveType::kUnsigned8,
                                                3,
                                                viz::ImageFormat::R8G8B8_UNORM,
                                                {viz::ComponentSwizzle::IDENTITY,
                                                 viz::ComponentSwizzle::IDENTITY,
                                                 viz::ComponentSwizzle::IDENTITY,
                                                 viz::ComponentSwizzle::ONE}},
                                               {nvidia::gxf::PrimitiveType::kInt8,
                                                3,
                                                viz::ImageFormat::R8G8B8_SNORM,
                                                {viz::ComponentSwizzle::IDENTITY,
                                                 viz::ComponentSwizzle::IDENTITY,
                                                 viz::ComponentSwizzle::IDENTITY,
                                                 viz::ComponentSwizzle::ONE}},
                                               {nvidia::gxf::PrimitiveType::kUnsigned8,
                                                4,
                                                viz::ImageFormat::R8G8B8A8_UNORM,
                                                {viz::ComponentSwizzle::IDENTITY,
                                                 viz::ComponentSwizzle::IDENTITY,
                                                 viz::ComponentSwizzle::IDENTITY,
                                                 viz::ComponentSwizzle::IDENTITY}},
                                               {nvidia::gxf::PrimitiveType::kInt8,
                                                4,
                                                viz::ImageFormat::R8G8B8A8_SNORM,
                                                {viz::ComponentSwizzle::IDENTITY,
                                                 viz::ComponentSwizzle::IDENTITY,
                                                 viz::ComponentSwizzle::IDENTITY,
                                                 viz::ComponentSwizzle::IDENTITY}},
                                               {nvidia::gxf::PrimitiveType::kUnsigned16,
                                                4,
                                                viz::ImageFormat::R16G16B16A16_UNORM,
                                                {viz::ComponentSwizzle::IDENTITY,
                                                 viz::ComponentSwizzle::IDENTITY,
                                                 viz::ComponentSwizzle::IDENTITY,
                                                 viz::ComponentSwizzle::IDENTITY}},
                                               {nvidia::gxf::PrimitiveType::kInt16,
                                                4,
                                                viz::ImageFormat::R16G16B16A16_SNORM,
                                                {viz::ComponentSwizzle::IDENTITY,
                                                 viz::ComponentSwizzle::IDENTITY,
                                                 viz::ComponentSwizzle::IDENTITY,
                                                 viz::ComponentSwizzle::IDENTITY}},
                                               {nvidia::gxf::PrimitiveType::kFloat32,
                                                4,
                                                viz::ImageFormat::R32G32B32A32_SFLOAT,
                                                {viz::ComponentSwizzle::IDENTITY,
                                                 viz::ComponentSwizzle::IDENTITY,
                                                 viz::ComponentSwizzle::IDENTITY,
                                                 viz::ComponentSwizzle::IDENTITY}}};

    for (auto&& format : kGFXToHolovizFormats) {
      if ((format.type_ == element_type) && (format.channels_ == int32_t(components))) {
        image_format = format.format_;
        component_swizzle[0] = format.component_swizzle[0];
        component_swizzle[1] = format.component_swizzle[1];
        component_swizzle[2] = format.component_swizzle[2];
        component_swizzle[3] = format.component_swizzle[3];
        break;
      }
    }
  }

  return GXF_SUCCESS;
}

gxf_result_t BufferInfo::init(const nvidia::gxf::Handle<nvidia::gxf::VideoBuffer>& video) {
  // NOTE: VideoBuffer::moveToTensor() converts VideoBuffer instance to the Tensor instance
  // with an unexpected shape:  [width, height] or [width, height, num_planes].
  // And, if we use moveToTensor() to convert VideoBuffer to Tensor, we may lose the original
  // video buffer when the VideoBuffer instance is used in other places. For that reason, we
  // directly access internal data of VideoBuffer instance to access Tensor data.
  const auto& buffer_info = video->video_frame_info();

  struct Format {
    nvidia::gxf::VideoFormat color_format_;
    nvidia::gxf::PrimitiveType element_type_;
    int32_t channels_;
    viz::ImageFormat format_;
    viz::ComponentSwizzle component_swizzle[4];
  };
  constexpr Format kVideoToHolovizFormats[] = {
      {nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY,
       nvidia::gxf::PrimitiveType::kUnsigned8,
       1,
       viz::ImageFormat::R8_UNORM,
       {viz::ComponentSwizzle::R,
        viz::ComponentSwizzle::R,
        viz::ComponentSwizzle::R,
        viz::ComponentSwizzle::ONE}},
      {nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY16,
       nvidia::gxf::PrimitiveType::kUnsigned16,
       1,
       viz::ImageFormat::R16_UNORM,
       {viz::ComponentSwizzle::R,
        viz::ComponentSwizzle::R,
        viz::ComponentSwizzle::R,
        viz::ComponentSwizzle::ONE}},
      {nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY32F,
       nvidia::gxf::PrimitiveType::kFloat32,
       1,
       viz::ImageFormat::R32_SFLOAT,
       {viz::ComponentSwizzle::R,
        viz::ComponentSwizzle::R,
        viz::ComponentSwizzle::R,
        viz::ComponentSwizzle::ONE}},
      {nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_D32F,
       nvidia::gxf::PrimitiveType::kFloat32,
       1,
       viz::ImageFormat::D32_SFLOAT,
       {viz::ComponentSwizzle::IDENTITY,
        viz::ComponentSwizzle::IDENTITY,
        viz::ComponentSwizzle::IDENTITY,
        viz::ComponentSwizzle::IDENTITY}},
      {nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB,
       nvidia::gxf::PrimitiveType::kUnsigned8,
       3,
       viz::ImageFormat::R8G8B8_UNORM,
       {viz::ComponentSwizzle::IDENTITY,
        viz::ComponentSwizzle::IDENTITY,
        viz::ComponentSwizzle::IDENTITY,
        viz::ComponentSwizzle::IDENTITY}},
      {nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_BGR,
       nvidia::gxf::PrimitiveType::kUnsigned8,
       3,
       viz::ImageFormat::R8G8B8_UNORM,
       {viz::ComponentSwizzle::B,
        viz::ComponentSwizzle::G,
        viz::ComponentSwizzle::R,
        viz::ComponentSwizzle::IDENTITY}},
      {nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA,
       nvidia::gxf::PrimitiveType::kUnsigned8,
       4,
       viz::ImageFormat::R8G8B8A8_UNORM,
       {viz::ComponentSwizzle::IDENTITY,
        viz::ComponentSwizzle::IDENTITY,
        viz::ComponentSwizzle::IDENTITY,
        viz::ComponentSwizzle::IDENTITY}},
      {nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_BGRA,
       nvidia::gxf::PrimitiveType::kUnsigned8,
       4,
       viz::ImageFormat::R8G8B8A8_UNORM,
       {viz::ComponentSwizzle::B,
        viz::ComponentSwizzle::G,
        viz::ComponentSwizzle::R,
        viz::ComponentSwizzle::A}},
      {nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_ARGB,
       nvidia::gxf::PrimitiveType::kUnsigned8,
       4,
       viz::ImageFormat::R8G8B8A8_UNORM,
       {viz::ComponentSwizzle::A,
        viz::ComponentSwizzle::R,
        viz::ComponentSwizzle::G,
        viz::ComponentSwizzle::B}},
      {nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_ABGR,
       nvidia::gxf::PrimitiveType::kUnsigned8,
       4,
       viz::ImageFormat::R8G8B8A8_UNORM,
       {viz::ComponentSwizzle::A,
        viz::ComponentSwizzle::G,
        viz::ComponentSwizzle::G,
        viz::ComponentSwizzle::R}},
      {nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBX,
       nvidia::gxf::PrimitiveType::kUnsigned8,
       4,
       viz::ImageFormat::R8G8B8A8_UNORM,
       {viz::ComponentSwizzle::R,
        viz::ComponentSwizzle::G,
        viz::ComponentSwizzle::B,
        viz::ComponentSwizzle::ONE}},
      {nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_BGRX,
       nvidia::gxf::PrimitiveType::kUnsigned8,
       4,
       viz::ImageFormat::R8G8B8A8_UNORM,
       {viz::ComponentSwizzle::B,
        viz::ComponentSwizzle::G,
        viz::ComponentSwizzle::R,
        viz::ComponentSwizzle::ONE}},
      {nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_XRGB,
       nvidia::gxf::PrimitiveType::kUnsigned8,
       4,
       viz::ImageFormat::R8G8B8A8_UNORM,
       {viz::ComponentSwizzle::G,
        viz::ComponentSwizzle::B,
        viz::ComponentSwizzle::A,
        viz::ComponentSwizzle::ONE}},
      {nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_XBGR,
       nvidia::gxf::PrimitiveType::kUnsigned8,
       4,
       viz::ImageFormat::R8G8B8A8_UNORM,
       {viz::ComponentSwizzle::A,
        viz::ComponentSwizzle::B,
        viz::ComponentSwizzle::G,
        viz::ComponentSwizzle::ONE}},
  };

  int32_t channels = 0;
  for (auto&& format : kVideoToHolovizFormats) {
    if (format.color_format_ == buffer_info.color_format) {
      element_type = format.element_type_;
      channels = format.channels_;
      image_format = format.format_;
      component_swizzle[0] = format.component_swizzle[0];
      component_swizzle[1] = format.component_swizzle[1];
      component_swizzle[2] = format.component_swizzle[2];
      component_swizzle[3] = format.component_swizzle[3];
      break;
    }
  }

  if (!channels) {
    HOLOSCAN_LOG_ERROR("Unsupported input format: {}\n",
                       static_cast<int64_t>(buffer_info.color_format));
    return GXF_FAILURE;
  }

  rank = 3;
  components = channels;
  width = buffer_info.width;
  height = buffer_info.height;
  name = video.name();
  buffer_ptr = video->pointer();
  storage_type = video->storage_type();
  bytes_size = video->size();
  stride[0] = buffer_info.color_planes[0].stride;
  stride[1] = channels;
  stride[2] = PrimitiveTypeSize(element_type);

  return GXF_SUCCESS;
}

}  // namespace holoscan::ops

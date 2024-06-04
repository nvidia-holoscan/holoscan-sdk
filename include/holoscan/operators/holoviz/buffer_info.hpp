/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef INCLUDE_HOLOSCAN_OPERATORS_HOLOVIZ_BUFFER_INFO_HPP
#define INCLUDE_HOLOSCAN_OPERATORS_HOLOVIZ_BUFFER_INFO_HPP

#include <string>

#include "holoviz/holoviz.hpp"  // holoviz module

#include "gxf/multimedia/video.hpp"

namespace holoscan::ops {

/// Buffer information, can be initialized either with a tensor or a video buffer
struct BufferInfo {
  /**
   * Initialize with tensor
   *
   * @return error code
   */
  gxf_result_t init(const nvidia::gxf::Handle<nvidia::gxf::Tensor>& tensor);

  /**
   * Initialize with video buffer
   *
   * @return error code
   */
  gxf_result_t init(const nvidia::gxf::Handle<nvidia::gxf::VideoBuffer>& video);

  uint32_t rank;
  uint32_t components, width, height;
  nvidia::gxf::PrimitiveType element_type;
  viz::ImageFormat image_format = static_cast<viz::ImageFormat>(-1);
  viz::ComponentSwizzle component_swizzle[4] = {viz::ComponentSwizzle::IDENTITY,
                                                viz::ComponentSwizzle::IDENTITY,
                                                viz::ComponentSwizzle::IDENTITY,
                                                viz::ComponentSwizzle::IDENTITY};
  std::string name;
  const nvidia::byte* buffer_ptr;
  nvidia::gxf::MemoryStorageType storage_type;
  uint64_t bytes_size;
  nvidia::gxf::Tensor::stride_array_t stride;
};

}  // namespace holoscan::ops

#endif /* INCLUDE_HOLOSCAN_OPERATORS_HOLOVIZ_BUFFER_INFO_HPP */

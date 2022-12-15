/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef NVIDIA_CLARA_HOLOSCAN_GXF_V4L2_SOURCE_HPP_
#define NVIDIA_CLARA_HOLOSCAN_GXF_V4L2_SOURCE_HPP_

#include <string>
#include <vector>

#include "gxf/std/allocator.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {
namespace holoscan {

/// @brief Input codelet for common V4L2 camera sources.
///
/// Provides a codelet for a realtime V4L2 source supporting USB cameras and other media inputs
/// on Linux.
/// The output is a VideoBuffer object.
class V4L2Source : public gxf::Codelet {
 public:
  gxf_result_t registerInterface(gxf::Registrar* registrar) override;

  gxf_result_t start() override;
  gxf_result_t tick() override;
  gxf_result_t stop() override;

 private:
  struct Buffer {
    Buffer(void* _ptr, size_t _length);

    void* ptr;
    size_t length;
  };

  void YUYVToRGBA(const void* yuyv, void* rgba, size_t width, size_t height);

  gxf::Parameter<gxf::Handle<gxf::Transmitter>> signal_;
  gxf::Parameter<gxf::Handle<gxf::Allocator>> allocator_;
  gxf::Parameter<std::string> device_;
  gxf::Parameter<uint32_t> width_;
  gxf::Parameter<uint32_t> height_;
  gxf::Parameter<uint32_t> num_buffers_;

  int fd_;
  std::vector<Buffer> buffers_;
};

}  // namespace holoscan
}  // namespace nvidia
#endif  // NVIDIA_CLARA_HOLOSCAN_GXF_V4L2_SOURCE_HPP_

/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef MOCKS_VIDEO_BUFFER_MOCK_HPP
#define MOCKS_VIDEO_BUFFER_MOCK_HPP

#include <cinttypes>
#include <string>
#include <vector>

#include "gxf/std/allocator.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/memory_buffer.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {
namespace holoscan {
namespace mocks {

/// @brief Mock video input codelet for testing.
class VideoBufferMock : public gxf::Codelet {
 public:
  gxf_result_t start() override;
  gxf_result_t tick() override;
  gxf_result_t stop() override;

  gxf_result_t registerInterface(gxf::Registrar* registrar) override;

 private:
  gxf::Parameter<int32_t> in_width_;
  gxf::Parameter<int32_t> in_height_;
  gxf::Parameter<int16_t> in_channels_;
  gxf::Parameter<uint8_t> in_bytes_per_pixel_;
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> out_;
  gxf::Parameter<std::string> out_tensor_name_;
  gxf::Parameter<gxf::Handle<gxf::Allocator>> pool_;

  // Tick counter
  gxf::MemoryBuffer buffer_;
  uint64_t count_ = 0;
};

}  // namespace mocks
}  // namespace holoscan
}  // namespace nvidia

#endif /* MOCKS_VIDEO_BUFFER_MOCK_HPP */

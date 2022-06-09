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
#ifndef NVIDIA_CLARA_HOLOSCAN_GXF_FRAME_DATA_HPP_
#define NVIDIA_CLARA_HOLOSCAN_GXF_FRAME_DATA_HPP_

#include <vector>

namespace nvidia {
namespace holoscan {
namespace visualizer_tool_tracking {

struct FrameData {
  // host memory buffers
  std::vector<float> confidence_host_;
  std::vector<float> position_host_;
  // OpenGL buffers
  GLuint confidence_;
  GLuint position_;
};

}  // namespace visualizer_tool_tracking
}  // namespace holoscan
}  // namespace nvidia
#endif  // NVIDIA_CLARA_HOLOSCAN_GXF_FRAME_DATA_HPP_

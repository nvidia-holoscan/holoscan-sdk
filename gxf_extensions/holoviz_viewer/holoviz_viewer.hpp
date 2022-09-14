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

#ifndef NVIDIA_CLARA_HOLOSCAN_GXF_EXTENSIONS_HOLOVIZ_VIEWER_HPP_
#define NVIDIA_CLARA_HOLOSCAN_GXF_EXTENSIONS_HOLOVIZ_VIEWER_HPP_

#include <string>

#include "gxf/core/handle.hpp"
#include "gxf/cuda/cuda_stream_pool.hpp"
#include "gxf/std/allocator.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/memory_buffer.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/tensor.hpp"
#include "gxf/std/transmitter.hpp"

#include "holoviz/Holoviz.h"

namespace viz = clara::holoviz;

namespace nvidia::holoscan {

constexpr uint32_t kDefaultWidth = 2560;
constexpr uint32_t kDefaultHeight = 1440;
constexpr uint32_t kDefaultFramerate = 240;
constexpr bool kDefaultExclusiveDisplay = false;

class HolovizViewer : public gxf::Codelet {
 public:
  gxf_result_t registerInterface(gxf::Registrar* registrar) override;
  gxf_result_t initialize() override;
  gxf_result_t deinitialize() override;

  gxf_result_t start() override;
  gxf_result_t tick() override;
  gxf_result_t stop() override;

 private:
  gxf::Parameter<gxf::Handle<gxf::Receiver>> receiver_;
  gxf::Parameter<std::string> input_image_name_;
  gxf::Parameter<std::string> window_title_;
  gxf::Parameter<std::string> display_name_;
  gxf::Parameter<uint32_t> width_;
  gxf::Parameter<uint32_t> height_;
  gxf::Parameter<uint32_t> framerate_;
  gxf::Parameter<bool> use_exclusive_display_;

  viz::ImageFormat image_format_ = viz::ImageFormat::R8G8B8_UNORM;
  bool is_initialized_ = false;
};

}

#endif  // NVIDIA_CLARA_HOLOSCAN_GXF_EXTENSIONS_HOLOVIZ_VIEWER_HPP_

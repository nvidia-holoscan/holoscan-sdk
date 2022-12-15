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
#ifndef NVIDIA_CLARA_HOLOSCAN_GXF_EXTENSIONS_TOOL_TRACKING_POSTPROCESSOR_HPP_
#define NVIDIA_CLARA_HOLOSCAN_GXF_EXTENSIONS_TOOL_TRACKING_POSTPROCESSOR_HPP_

#include <string>
#include <vector>

#include "gxf/std/allocator.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/transmitter.hpp"

#include "tool_tracking_postprocessor.cu.hpp"

namespace nvidia {
namespace holoscan {
namespace tool_tracking_postprocessor {

/// @brief tool tracking model postproceessing Codelet converting inference output.
///
/// This Codelet performs tool tracking model postprocessing in CUDA.
class Postprocessor : public gxf::Codelet {
 public:
  gxf_result_t start() override;
  gxf_result_t tick() override;
  gxf_result_t stop() override;
  gxf_result_t registerInterface(gxf::Registrar* registrar) override;

 private:
  gxf::Parameter<gxf::Handle<gxf::Receiver>> in_;
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> out_;

  gxf::Parameter<float> min_prob_;
  gxf::Parameter<std::vector<std::vector<float>>> overlay_img_colors_;

  gxf::Parameter<gxf::Handle<gxf::Allocator>> host_allocator_;
  gxf::Parameter<gxf::Handle<gxf::Allocator>> device_allocator_;
};
}  // namespace tool_tracking_postprocessor
}  // namespace holoscan
}  // namespace nvidia

#endif  // NVIDIA_CLARA_HOLOSCAN_GXF_EXTENSIONS_TOOL_TRACKING_POSTPROCESSOR_HPP_

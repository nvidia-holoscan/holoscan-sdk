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
#ifndef NVIDIA_GXF_EXTENSIONS_VISUALIZER_ICARDIO_HPP_
#define NVIDIA_GXF_EXTENSIONS_VISUALIZER_ICARDIO_HPP_

#include <fstream>
#include <map>
#include <string>
#include <vector>

#include <holoinfer_utils.hpp>

namespace HoloInfer = holoscan::inference;

namespace nvidia {
namespace holoscan {
namespace multiai {

class VisualizerICardio : public gxf::Codelet {
 public:
  gxf_result_t start() override;
  gxf_result_t tick() override;
  gxf_result_t stop() override;
  gxf_result_t registerInterface(gxf::Registrar* registrar) override;

 private:
  gxf::Parameter<std::vector<std::string>> in_tensor_names_;
  gxf::Parameter<std::vector<std::string>> out_tensor_names_;
  gxf::Parameter<gxf::Handle<gxf::Allocator>> allocator_;

  gxf::Parameter<HoloInfer::GXFReceivers> receivers_;
  gxf::Parameter<HoloInfer::GXFTransmitters> transmitters_;

  HoloInfer::DataMap data_per_tensor;
  std::map<std::string, std::vector<int>> tensor_size_map_;

  const std::string module_{"Visualizer icardio Codelet"};
  const std::string pc_tensor_name_{"plax_chamber_processed"};

  gxf::Parameter<bool> input_on_cuda_;

  const std::map<std::string, std::vector<int>> tensor_to_shape_ = {{"keypoints", {1, 5, 3}},
                                                                    {"keyarea_1", {1, 1, 4}},
                                                                    {"keyarea_2", {1, 1, 4}},
                                                                    {"keyarea_3", {1, 1, 4}},
                                                                    {"keyarea_4", {1, 1, 4}},
                                                                    {"keyarea_5", {1, 1, 4}},
                                                                    {"lines", {1, 5, 2}},
                                                                    {"logo", {320, 320, 4}}};
  const std::map<std::string, int> tensor_to_index_ = {
      {"keyarea_1", 1}, {"keyarea_2", 2}, {"keyarea_3", 3}, {"keyarea_4", 4}, {"keyarea_5", 5}};

  const std::string path_to_logo_file_ = "../data/multiai_ultrasound/logo.txt";
  std::vector<int> logo_image_;
};

}  // namespace multiai
}  // namespace holoscan
}  // namespace nvidia

#endif  // NVIDIA_GXF_EXTENSIONS_VISUALIZER_ICARDIO_HPP_

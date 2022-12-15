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

#ifndef HOLOSCAN_OPERATORS_VISUALIZER_ICARDIO_HPP
#define HOLOSCAN_OPERATORS_VISUALIZER_ICARDIO_HPP

#include <memory>
#include <string>
#include <vector>

#include "../../core/gxf/gxf_operator.hpp"

namespace holoscan::ops {
/**
 * @brief Visualizer iCardio Operator class to generate data for visualization
 *
 * Class wraps a GXF Codelet(`nvidia::holoscan::multiai::VisualizerICardio`).
 */
class VisualizerICardioOp : public holoscan::ops::GXFOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(VisualizerICardioOp, holoscan::ops::GXFOperator)

  VisualizerICardioOp() = default;

  const char* gxf_typename() const override {
    return "nvidia::holoscan::multiai::VisualizerICardio";
  }

  void setup(OperatorSpec& spec) override;

 private:
  Parameter<std::vector<std::string>> in_tensor_names_;
  Parameter<std::vector<std::string>> out_tensor_names_;
  Parameter<std::shared_ptr<Allocator>> allocator_;
  Parameter<std::vector<IOSpec*>> receivers_;
  Parameter<std::vector<IOSpec*>> transmitters_;
  Parameter<bool> input_on_cuda_;
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_VISUALIZER_ICARDIO_HPP */

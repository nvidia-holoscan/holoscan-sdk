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

#include "holoscan/operators/multiai_inference/multiai_inference.hpp"
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"

/**
 * Custom YAML parser for DataMap class
 */
template <>
struct YAML::convert<holoscan::ops::MultiAIInferenceOp::DataMap> {
  static Node encode(const holoscan::ops::MultiAIInferenceOp::DataMap& datamap) {
    Node node;
    auto mappings = datamap.get_map();
    for (const auto& dm : mappings) { node[dm.first] = dm.second; }
    return node;
  }

  static bool decode(const Node& node, holoscan::ops::MultiAIInferenceOp::DataMap& datamap) {
    if (!node.IsMap()) {
      HOLOSCAN_LOG_ERROR("InputSpec: expected a map");
      return false;
    }
    try {
      for (YAML::const_iterator it = node.begin(); it != node.end(); ++it) {
        std::string key = it->first.as<std::string>();
        std::string value = it->second.as<std::string>();
        datamap.insert(key, value);
      }
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR(e.what());
      return false;
    }
    return true;
  }
};

/**
 * Custom YAML parser for DataVecMap class
 */
template <>
struct YAML::convert<holoscan::ops::MultiAIInferenceOp::DataVecMap> {
  static Node encode(const holoscan::ops::MultiAIInferenceOp::DataVecMap& datavmap) {
    Node node;
    auto mappings = datavmap.get_map();
    for (const auto& dm : mappings) {
      auto vec_of_values = dm.second;
      for (const auto& value : vec_of_values) node[dm.first].push_back(value);
    }
    return node;
  }

  static bool decode(const Node& node, holoscan::ops::MultiAIInferenceOp::DataVecMap& datavmap) {
    if (!node.IsMap()) {
      HOLOSCAN_LOG_ERROR("InputSpec: expected a map");
      return false;
    }

    try {
      for (YAML::const_iterator it = node.begin(); it != node.end(); ++it) {
        std::string key = it->first.as<std::string>();
        std::vector<std::string> value = it->second.as<std::vector<std::string>>();
        datavmap.insert(key, value);
      }
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR(e.what());
      return false;
    }
    return true;
  }
};

namespace holoscan::ops {

void MultiAIInferenceOp::setup(OperatorSpec& spec) {
  auto& transmitter = spec.output<gxf::Entity>("transmitter");
  spec.param(backend_, "backend", "Supported backend");
  spec.param(model_path_map_,
             "model_path_map",
             "Model Keyword with File Path",
             "Path to ONNX model to be loaded.",
             DataMap());
  spec.param(pre_processor_map_,
             "pre_processor_map",
             "Pre processor setting per model",
             "Pre processed data to model map.",
             DataVecMap());
  spec.param(inference_map_,
             "inference_map",
             "Inferred tensor per model",
             "Tensor to model map.",
             DataMap());
  spec.param(
      in_tensor_names_, "in_tensor_names", "Input Tensors", "Input tensors", {std::string("")});
  spec.param(
      out_tensor_names_, "out_tensor_names", "Output Tensors", "Output tensors", {std::string("")});
  spec.param(allocator_, "allocator", "Allocator", "Output Allocator");
  spec.param(infer_on_cpu_, "infer_on_cpu", "Inference on CPU", "Use CPU.", false);
  spec.param(is_engine_path_, "is_engine_path", "Input path is engine file", "", false);

  spec.param(enable_fp16_, "enable_fp16", "Use fp16", "Use fp16.", false);
  spec.param(input_on_cuda_, "input_on_cuda", "Input buffer on CUDA", "", true);
  spec.param(output_on_cuda_, "output_on_cuda", "Output buffer on CUDA", "", true);
  spec.param(transmit_on_cuda_, "transmit_on_cuda", "Transmit message on CUDA", "", true);

  spec.param(parallel_inference_, "parallel_inference", "Parallel inference", "", true);
  spec.param(receivers_, "receivers", "Receivers", "List of receivers", {});
  spec.param(transmitter_, "transmitter", "Transmitter", "Transmitter", {&transmitter});
}

void MultiAIInferenceOp::initialize() {
  register_converter<DataMap>();
  register_converter<DataVecMap>();
  GXFOperator::initialize();
}

}  // namespace holoscan::ops

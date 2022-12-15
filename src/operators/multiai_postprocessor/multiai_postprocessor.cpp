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

#include "holoscan/operators/multiai_postprocessor/multiai_postprocessor.hpp"
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"

template <>
struct YAML::convert<holoscan::ops::MultiAIPostprocessorOp::DataMap> {
  static Node encode(const holoscan::ops::MultiAIPostprocessorOp::DataMap& datamap) {
    Node node;
    auto mappings = datamap.get_map();
    for (const auto& dm : mappings) { node[dm.first] = dm.second; }
    return node;
  }

  static bool decode(const Node& node, holoscan::ops::MultiAIPostprocessorOp::DataMap& datamap) {
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
      return true;
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR(e.what());
      return false;
    }
  }
};

/**
 * Custom YAML parser for DataVecMap class
 */
template <>
struct YAML::convert<holoscan::ops::MultiAIPostprocessorOp::DataVecMap> {
  static Node encode(const holoscan::ops::MultiAIPostprocessorOp::DataVecMap& datavmap) {
    Node node;
    auto mappings = datavmap.get_map();
    for (const auto& dm : mappings) {
      auto vec_of_values = dm.second;
      for (const auto& value : vec_of_values) node[dm.first].push_back(value);
    }
    return node;
  }

  static bool decode(const Node& node,
                     holoscan::ops::MultiAIPostprocessorOp::DataVecMap& datavmap) {
    if (!node.IsMap()) {
      HOLOSCAN_LOG_ERROR("DataVecMap: expected a map");
      return false;
    }

    try {
      for (YAML::const_iterator it = node.begin(); it != node.end(); ++it) {
        std::string key = it->first.as<std::string>();
        std::vector<std::string> value = it->second.as<std::vector<std::string>>();
        datavmap.insert(key, value);
      }
      return true;
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR(e.what());
      return false;
    }
  }
};

namespace holoscan::ops {

void MultiAIPostprocessorOp::setup(OperatorSpec& spec) {
  auto& transmitter = spec.output<gxf::Entity>("transmitter");

  spec.param(process_operations_,
             "process_operations",
             "Operations per tensor",
             "Operations in sequence on tensors.");
  spec.param(processed_map_, "processed_map", "In to out tensor", "Input-output tensor mapping.");
  spec.param(
      in_tensor_names_, "in_tensor_names", "Input Tensors", "Input tensors", {std::string("")});
  spec.param(
      out_tensor_names_, "out_tensor_names", "Output Tensors", "Output tensors", {std::string("")});
  spec.param(input_on_cuda_, "input_on_cuda", "Input buffer on CUDA", "", false);
  spec.param(output_on_cuda_, "output_on_cuda", "Output buffer on CUDA", "", false);
  spec.param(transmit_on_cuda_, "transmit_on_cuda", "Transmit message on CUDA", "", false);
  spec.param(allocator_, "allocator", "Allocator", "Output Allocator");
  spec.param(receivers_, "receivers", "Receivers", "List of receivers", {});
  spec.param(transmitter_, "transmitter", "Transmitter", "Transmitter", {&transmitter});
}

void MultiAIPostprocessorOp::initialize() {
  register_converter<DataVecMap>();
  register_converter<DataMap>();
  GXFOperator::initialize();
}

}  // namespace holoscan::ops

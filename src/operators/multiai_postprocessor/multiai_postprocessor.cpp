/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "holoscan/core/execution_context.hpp"
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/io_context.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"
#include "holoscan/utils/holoinfer_utils.hpp"

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
             "Operations in sequence on tensors.",
             DataVecMap());
  spec.param(processed_map_,
             "processed_map",
             "In to out tensor",
             "Input-output tensor mapping.",
             DataMap());
  spec.param(in_tensor_names_, "in_tensor_names", "Input Tensors", "Input tensors", {});
  spec.param(out_tensor_names_, "out_tensor_names", "Output Tensors", "Output tensors", {});
  spec.param(input_on_cuda_, "input_on_cuda", "Input buffer on CUDA", "", false);
  spec.param(output_on_cuda_, "output_on_cuda", "Output buffer on CUDA", "", false);
  spec.param(transmit_on_cuda_, "transmit_on_cuda", "Transmit message on CUDA", "", false);
  spec.param(allocator_, "allocator", "Allocator", "Output Allocator");
  spec.param(receivers_, "receivers", "Receivers", "List of receivers", {});
  spec.param(transmitter_, "transmitter", "Transmitter", "Transmitter", {&transmitter});
}

void MultiAIPostprocessorOp::conditional_disable_output_port(const std::string& port_name) {
  const std::string disable_port_name = std::string("disable_") + port_name;

  // Check if the boolean argument with the name "disable_(port_name)" is present.
  auto disable_port =
      std::find_if(args().begin(), args().end(), [&disable_port_name](const auto& arg) {
        return (arg.name() == disable_port_name);
      });

  // If the argument exists without value or is set to true, we unset(nullify) the port parameter.
  const bool need_disabled =
      (disable_port != args().end() &&
       (!disable_port->has_value() ||
        (disable_port->has_value() && (std::any_cast<bool>(disable_port->value()) == true))));

  if (disable_port != args().end()) {
    // If the argument is present, we just remove it from the arguments.
    args().erase(disable_port);
  }

  if (need_disabled) {
    // Set the condition of the port to kNone, which means that the port doesn't have any condition.
    spec()->outputs()[port_name]->condition(ConditionType::kNone);
    add_arg(Arg(port_name) = std::vector<holoscan::IOSpec*>());
  }
}

void MultiAIPostprocessorOp::initialize() {
  register_converter<DataVecMap>();
  register_converter<DataMap>();

  // Output port is conditionally disabled using a boolean argument
  conditional_disable_output_port("transmitter");

  Operator::initialize();
}

void MultiAIPostprocessorOp::start() {
  try {
    // Check for the validity of parameters from configuration
    if (input_on_cuda_.get() || output_on_cuda_.get() || transmit_on_cuda_.get()) {
      HoloInfer::raise_error(module_, "CUDA based data not supported in Multi AI post processor");
    }
    auto status = HoloInfer::multiai_processor_validity_check(
        processed_map_.get().get_map(), in_tensor_names_.get(), out_tensor_names_.get());
    if (status.get_code() != HoloInfer::holoinfer_code::H_SUCCESS) {
      status.display_message();
      HoloInfer::raise_error(module_, "Parameter Validation failed: " + status.get_message());
    }

    // Create holoscan processing context
    holoscan_postprocess_context_ = std::make_unique<HoloInfer::ProcessorContext>();
  } catch (const std::bad_alloc& b_) {
    HoloInfer::raise_error(module_, "Start, Memory allocation, Message: " + std::string(b_.what()));
  } catch (const std::runtime_error& rt_) {
    HOLOSCAN_LOG_ERROR(rt_.what());
    throw;
  } catch (...) { HoloInfer::raise_error(module_, "Start, Unknown exception"); }

  // Initialize holoscan processing context
  auto status = holoscan_postprocess_context_->initialize(process_operations_.get().get_map());
  if (status.get_code() != HoloInfer::holoinfer_code::H_SUCCESS) {
    status.display_message();
    HoloInfer::raise_error(module_, "Start, Out data setup");
  }
}

void MultiAIPostprocessorOp::compute(InputContext& op_input, OutputContext& op_output,
                                     ExecutionContext& context) {
  // get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
  auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(),
                                                                       allocator_.get()->gxf_cid());

  try {
    // Extract relevant data from input GXF Receivers, and update multiai specifications
    gxf_result_t stat = holoscan::utils::multiai_get_data_per_model(op_input,
                                                                    in_tensor_names_.get(),
                                                                    data_per_tensor_,
                                                                    dims_per_tensor_,
                                                                    input_on_cuda_.get(),
                                                                    module_);

    if (stat != GXF_SUCCESS) { HoloInfer::raise_error(module_, "Tick, Data extraction"); }

    // Execute processing
    auto status = holoscan_postprocess_context_->process(process_operations_.get().get_map(),
                                                         processed_map_.get().get_map(),
                                                         data_per_tensor_,
                                                         dims_per_tensor_);
    if (status.get_code() != HoloInfer::holoinfer_code::H_SUCCESS) {
      status.display_message();
      HoloInfer::report_error(module_, "Tick, post_process");
    }

    // Get processed data and dimensions (currently only on host)
    auto processed_data_map = holoscan_postprocess_context_->get_processed_data();
    auto processed_dims_map = holoscan_postprocess_context_->get_processed_data_dims();

    if (out_tensor_names_.get().size() != 0) {
      auto cont = context.context();
      // Transmit output buffers via a single GXF transmitter
      stat = holoscan::utils::multiai_transmit_data_per_model(cont,
                                                              processed_map_.get().get_map(),
                                                              processed_data_map,
                                                              op_output,
                                                              out_tensor_names_.get(),
                                                              processed_dims_map,
                                                              output_on_cuda_.get(),
                                                              transmit_on_cuda_.get(),
                                                              nvidia::gxf::PrimitiveType::kFloat32,
                                                              allocator.value(),
                                                              module_);

      if (stat != GXF_SUCCESS) { HoloInfer::report_error(module_, "Tick, Data Transmission"); }
    }
  } catch (const std::runtime_error& r_) {
    HoloInfer::report_error(module_, "Tick, Message->" + std::string(r_.what()));
  } catch (...) { HoloInfer::report_error(module_, "Tick, unknown exception"); }
}

}  // namespace holoscan::ops

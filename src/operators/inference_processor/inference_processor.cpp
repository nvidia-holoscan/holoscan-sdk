/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/operators/inference_processor/inference_processor.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "holoscan/core/execution_context.hpp"
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/io_context.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"
#include "holoscan/utils/cuda_macros.hpp"
#include "holoscan/utils/holoinfer_utils.hpp"

template <>
struct YAML::convert<holoscan::ops::InferenceProcessorOp::DataMap> {
  static Node encode(const holoscan::ops::InferenceProcessorOp::DataMap& datamap) {
    Node node;
    auto mappings = datamap.get_map();
    for (const auto& [key, value] : mappings) { node[key] = value; }
    return node;
  }

  static bool decode(const Node& node, holoscan::ops::InferenceProcessorOp::DataMap& datamap) {
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
struct YAML::convert<holoscan::ops::InferenceProcessorOp::DataVecMap> {
  static Node encode(const holoscan::ops::InferenceProcessorOp::DataVecMap& datavmap) {
    Node node;
    auto mappings = datavmap.get_map();
    for (const auto& [key, vec_of_values] : mappings) {
      for (const auto& value : vec_of_values) node[key].push_back(value);
    }
    return node;
  }

  static bool decode(const Node& node, holoscan::ops::InferenceProcessorOp::DataVecMap& datavmap) {
    if (!node.IsMap()) {
      HOLOSCAN_LOG_ERROR("DataVecMap: expected a map");
      return false;
    }

    try {
      for (YAML::const_iterator it = node.begin(); it != node.end(); ++it) {
        std::string key = it->first.as<std::string>();
        switch (it->second.Type()) {
          case YAML::NodeType::Scalar: {  // For backward compatibility v0.5 and lower
            HOLOSCAN_LOG_INFO("Values for Tensor {} not in a vector form.", key);
            HOLOSCAN_LOG_INFO(
                "HoloInfer in Holoscan SDK 0.6 onwards expects mapped tensor names in a vector "
                "form.");
            HOLOSCAN_LOG_INFO(
                "Converting mappings for tensor {} to vector for backward compatibility.", key);
            std::string value = it->second.as<std::string>();
            datavmap.insert(key, {std::move(value)});
          } break;
          case YAML::NodeType::Sequence: {
            std::vector<std::string> value = it->second.as<std::vector<std::string>>();
            datavmap.insert(key, value);
          } break;
          default: {
            HOLOSCAN_LOG_ERROR("Unsupported entry in parameter set for model {}", key);
            return false;
          }
        }
      }
      return true;
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR(e.what());
      return false;
    }
  }
};

namespace holoscan::ops {

void InferenceProcessorOp::setup(OperatorSpec& spec) {
  spec.input<std::vector<gxf::Entity>>("receivers", IOSpec::kAnySize);
  spec.output<gxf::Entity>("transmitter");

  spec.param(process_operations_,
             "process_operations",
             "Operations per tensor",
             "Operations in sequence on tensors.",
             DataVecMap());
  spec.param(processed_map_,
             "processed_map",
             "In to out tensor",
             "Input-output tensor mapping.",
             DataVecMap());
  spec.param(in_tensor_names_, "in_tensor_names", "Input Tensors", "Input tensors", {});
  spec.param(config_path_, "config_path", "Path to config file", "Config File", {});
  spec.param(custom_kernels_, "custom_kernels", "Custom Cuda Kernel", "Custom kernel", DataMap());
  spec.param(out_tensor_names_, "out_tensor_names", "Output Tensors", "Output tensors", {});
  spec.param(input_on_cuda_, "input_on_cuda", "Input buffer on CUDA", "", false);
  spec.param(output_on_cuda_, "output_on_cuda", "Output buffer on CUDA", "", false);
  spec.param(transmit_on_cuda_, "transmit_on_cuda", "Transmit message on CUDA", "", false);
  spec.param(use_cuda_graphs_, "use_cuda_graphs", "Use CUDA Graphs for custom kernels", "", false);
  spec.param(allocator_, "allocator", "Allocator", "Output Allocator");
  spec.param(cuda_stream_pool_,
             "cuda_stream_pool",
             "CUDA Stream Pool",
             "Instance of gxf::CudaStreamPool.",
             ParameterFlag::kOptional);
}

void InferenceProcessorOp::conditional_disable_output_port(const std::string& port_name) {
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

void InferenceProcessorOp::initialize() {
  register_converter<DataVecMap>();
  register_converter<DataMap>();

  // Output port is conditionally disabled using a boolean argument
  conditional_disable_output_port("transmitter");

  Operator::initialize();
}

void InferenceProcessorOp::start() {
  try {
    // Check for the validity of parameters from configuration

    auto status = HoloInfer::processor_validity_check(
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
  auto status = holoscan_postprocess_context_->initialize(process_operations_.get().get_map(),
                                                          custom_kernels_.get().get_map(),
                                                          use_cuda_graphs_.get(),
                                                          config_path_.get());
  if (status.get_code() != HoloInfer::holoinfer_code::H_SUCCESS) {
    status.display_message();
    HoloInfer::raise_error(module_, "Start, Out data setup");
  }
}

void InferenceProcessorOp::stop() {
  data_per_tensor_.clear();
  holoscan_postprocess_context_.reset();
}

void InferenceProcessorOp::compute(InputContext& op_input, OutputContext& op_output,
                                   ExecutionContext& context) {
  // get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
  auto allocator =
      nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), allocator_->gxf_cid());
  auto cont = context.context();

  // process with CUDA if input is on CUDA
  const bool process_with_cuda = input_on_cuda_.get();

  try {
    // Extract relevant data from input GXF Receivers, and update inference specifications
    // (cuda_stream will be set by get_data_per_model)
    cudaStream_t cuda_stream{};
    gxf_result_t stat = holoscan::utils::get_data_per_model(op_input,
                                                            in_tensor_names_.get(),
                                                            data_per_tensor_,
                                                            dims_per_tensor_,
                                                            input_on_cuda_.get(),
                                                            module_,
                                                            cuda_stream);

    if (stat != GXF_SUCCESS) { HoloInfer::raise_error(module_, "Tick, Data extraction"); }

    // Transmit this stream on the output port if needed
    if (cuda_stream != cudaStreamDefault && output_on_cuda_.get()) {
      HOLOSCAN_LOG_TRACE("InferenceProcessorOp: sending CUDA stream to output");
      op_output.set_cuda_stream(cuda_stream, "transmitter");
    }

    HoloInfer::TimePoint s_time, e_time;
    HoloInfer::timer_init(s_time);
    // Execute processing
    auto status = holoscan_postprocess_context_->process(process_operations_.get().get_map(),
                                                         processed_map_.get().get_map(),
                                                         data_per_tensor_,
                                                         dims_per_tensor_,
                                                         process_with_cuda,
                                                         cuda_stream);
    if (status.get_code() != HoloInfer::holoinfer_code::H_SUCCESS) {
      status.display_message();
      HoloInfer::report_error(module_, "Tick, post_process");
    }
    data_per_tensor_.clear();
    dims_per_tensor_.clear();
    HoloInfer::timer_init(e_time);
    HoloInfer::timer_check(s_time, e_time, "Process operations");
    // Get processed data and dimensions (currently only on host)
    auto processed_data_map = holoscan_postprocess_context_->get_processed_data();
    auto processed_dims_map = holoscan_postprocess_context_->get_processed_data_dims();

    if (processed_data_map.size() != 0) {
      // Transmit output buffers via a single GXF transmitter
      stat = holoscan::utils::transmit_data_per_model(cont,
                                                      processed_map_.get().get_map(),
                                                      processed_data_map,
                                                      op_output,
                                                      out_tensor_names_.get(),
                                                      processed_dims_map,
                                                      process_with_cuda,
                                                      transmit_on_cuda_.get(),
                                                      allocator.value(),
                                                      module_,
                                                      cuda_stream);
      if (stat != GXF_SUCCESS) { HoloInfer::report_error(module_, "Tick, Data Transmission"); }
    }
  } catch (const std::runtime_error& r_) {
    HoloInfer::report_error(module_, "Tick, Message->" + std::string(r_.what()));
  } catch (...) { HoloInfer::report_error(module_, "Tick, unknown exception"); }
}

}  // namespace holoscan::ops

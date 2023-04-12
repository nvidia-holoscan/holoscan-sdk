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

#include "holoscan/operators/multiai_inference/multiai_inference.hpp"

#include "holoscan/core/execution_context.hpp"
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/io_context.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"
#include "holoscan/utils/holoinfer_utils.hpp"

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
  register_converter<DataMap>();
  register_converter<DataVecMap>();
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
  Operator::initialize();
}

void MultiAIInferenceOp::start() {
  try {
    // Check for the validity of parameters from configuration
    auto status = HoloInfer::multiai_inference_validity_check(model_path_map_.get().get_map(),
                                                              pre_processor_map_.get().get_map(),
                                                              inference_map_.get().get_map(),
                                                              in_tensor_names_.get(),
                                                              out_tensor_names_.get());
    if (status.get_code() != HoloInfer::holoinfer_code::H_SUCCESS) {
      HoloInfer::raise_error(module_, "Parameter Validation failed: " + status.get_message());
    }

    bool is_aarch64 = HoloInfer::is_platform_aarch64();
    if (is_aarch64 && backend_.get().compare("onnxrt") == 0 && !infer_on_cpu_.get()) {
      HoloInfer::raise_error(module_, "Onnxruntime with CUDA not supported on aarch64.");
    }

    // Create multiai specification structure
    multiai_specs_ = std::make_shared<HoloInfer::MultiAISpecs>(backend_.get(),
                                                               model_path_map_.get().get_map(),
                                                               inference_map_.get().get_map(),
                                                               is_engine_path_.get(),
                                                               infer_on_cpu_.get(),
                                                               parallel_inference_.get(),
                                                               enable_fp16_.get(),
                                                               input_on_cuda_.get(),
                                                               output_on_cuda_.get());

    // Create holoscan inference context
    holoscan_infer_context_ = std::make_unique<HoloInfer::InferContext>();

    // Set and transfer inference specification to inference context
    // Multi AI specifications are updated with memory allocations
    status = holoscan_infer_context_->set_inference_params(multiai_specs_);
    if (status.get_code() != HoloInfer::holoinfer_code::H_SUCCESS) {
      HoloInfer::raise_error(module_, "Start, Parameters setup, " + status.get_message());
    }
  } catch (const std::bad_alloc& b_) {
    HoloInfer::raise_error(module_, "Start, Memory allocation, Message: " + std::string(b_.what()));
  } catch (const std::runtime_error& rt_) {
    HOLOSCAN_LOG_ERROR(rt_.what());
    throw;
  } catch (...) { HoloInfer::raise_error(module_, "Start, Unknown exception"); }
}

void MultiAIInferenceOp::stop() {
  holoscan_infer_context_.reset();
}

gxf_result_t timer_check(HoloInfer::TimePoint& start, HoloInfer::TimePoint& end,
                         const std::string& module) {
  HoloInfer::timer_init(end);
  int64_t delta = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  HOLOSCAN_LOG_DEBUG("{} : {} ms", module, delta);
  return GXF_SUCCESS;
}

void MultiAIInferenceOp::compute(InputContext& op_input, OutputContext& op_output,
                                 ExecutionContext& context) {
  // get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
  auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(),
                                                                       allocator_.get()->gxf_cid());

  try {
    // Extract relevant data from input GXF Receivers, and update multiai specifications
    gxf_result_t stat =
        holoscan::utils::multiai_get_data_per_model(op_input,
                                                    in_tensor_names_.get(),
                                                    multiai_specs_->data_per_tensor_,
                                                    dims_per_tensor_,
                                                    input_on_cuda_.get(),
                                                    module_);

    if (stat != GXF_SUCCESS) { HoloInfer::raise_error(module_, "Tick, Data extraction"); }

    auto status = HoloInfer::map_data_to_model_from_tensor(pre_processor_map_.get().get_map(),
                                                           multiai_specs_->data_per_model_,
                                                           multiai_specs_->data_per_tensor_);
    if (status.get_code() != HoloInfer::holoinfer_code::H_SUCCESS) {
      HoloInfer::raise_error(module_, "Tick, Data mapping, " + status.get_message());
    }

    // Execute inference and populate output buffer in multiai specifications
    status = holoscan_infer_context_->execute_inference(multiai_specs_->data_per_model_,
                                                        multiai_specs_->output_per_model_);
    if (status.get_code() != HoloInfer::holoinfer_code::H_SUCCESS) {
      HoloInfer::raise_error(module_, "Tick, Inference execution, " + status.get_message());
    }
    HOLOSCAN_LOG_DEBUG(status.get_message());

    // Get output dimensions
    auto model_out_dims_map = holoscan_infer_context_->get_output_dimensions();

    auto cont = context.context();

    // Transmit output buffers via a single GXF transmitter
    stat = holoscan::utils::multiai_transmit_data_per_model(cont,
                                                            inference_map_.get().get_map(),
                                                            multiai_specs_->output_per_model_,
                                                            op_output,
                                                            out_tensor_names_.get(),
                                                            model_out_dims_map,
                                                            output_on_cuda_.get(),
                                                            transmit_on_cuda_.get(),
                                                            data_type_,
                                                            allocator.value(),
                                                            module_);
    if (stat != GXF_SUCCESS) { HoloInfer::raise_error(module_, "Tick, Data Transmission"); }
  } catch (const std::runtime_error& r_) {
    HoloInfer::raise_error(module_,
                           "Tick, Inference execution, Message->" + std::string(r_.what()));
  } catch (...) { HoloInfer::raise_error(module_, "Tick, unknown exception"); }
}

}  // namespace holoscan::ops

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

#include "holoscan/operators/inference/inference.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "holoscan/core/execution_context.hpp"
#include "holoscan/core/executors/gxf/gxf_executor.hpp"
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/io_context.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"
#include "holoscan/operators/inference/codecs.hpp"
#include "holoscan/utils/holoinfer_utils.hpp"

/**
 * Custom YAML parser for DataMap class
 */
template <>
struct YAML::convert<holoscan::ops::InferenceOp::DataMap> {
  static Node encode(const holoscan::ops::InferenceOp::DataMap& datamap) {
    Node node;
    auto mappings = datamap.get_map();
    for (const auto& [key, value] : mappings) {
      node[key] = value;
    }
    return node;
  }

  static bool decode(const Node& node, holoscan::ops::InferenceOp::DataMap& datamap) {
    if (!node.IsMap()) {
      HOLOSCAN_LOG_ERROR("InputSpec: expected a map");
      return false;
    }
    try {
      for (YAML::const_iterator it = node.begin(); it != node.end(); ++it) {
        std::string key = it->first.as<std::string>();
        std::string value = it->second.as<std::string>();
        datamap.insert(key, std::move(value));
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
struct YAML::convert<holoscan::ops::InferenceOp::DataVecMap> {
  static Node encode(const holoscan::ops::InferenceOp::DataVecMap& datavmap) {
    Node node;
    auto mappings = datavmap.get_map();
    for (const auto& [key, vec_of_values] : mappings) {
      for (const auto& value : vec_of_values)
        node[key].push_back(value);
    }
    return node;
  }

  static bool decode(const Node& node, holoscan::ops::InferenceOp::DataVecMap& datavmap) {
    if (!node.IsMap()) {
      HOLOSCAN_LOG_ERROR("InputSpec: expected a map");
      return false;
    }

    try {
      for (YAML::const_iterator it = node.begin(); it != node.end(); ++it) {
        std::string key = it->first.as<std::string>();

        switch (it->second.Type()) {
          case YAML::NodeType::Scalar: {  // For backward compatibility v0.5 and lower
            HOLOSCAN_LOG_WARN("Values for model {} not in a vector form.", key);
            HOLOSCAN_LOG_INFO(
                "HoloInfer in Holoscan SDK 0.6 onwards expects tensor names for models in a "
                "vector "
                "form in the parameter set.");
            HOLOSCAN_LOG_INFO(
                "Converting input tensor names for model {} to vector form for backward "
                "compatibility.",
                key);
            HOLOSCAN_LOG_WARN("Single I/O per model supported in backward compatibility mode.");
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
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR(e.what());
      return false;
    }
    return true;
  }
};

namespace holoscan::ops {

void InferenceOp::setup(OperatorSpec& spec) {
  register_converter<DataMap>();
  register_converter<DataVecMap>();
  spec.input<std::vector<gxf::Entity>>("receivers", IOSpec::kAnySize);
  spec.input<std::any>("model_activation_specs").condition(ConditionType::kNone);

  spec.output<gxf::Entity>("transmitter");

  spec.param(backend_, "backend", "Supported backend", "backend", {});
  spec.param(backend_map_, "backend_map", "Supported backend map", "", DataMap());
  spec.param(model_path_map_,
             "model_path_map",
             "Model Keyword with File Path",
             "Path to ONNX model to be loaded.",
             DataMap());
  spec.param(device_map_,
             "device_map",
             "Model Keyword with associated device",
             "Device ID on which model will do inference.",
             DataMap());
  spec.param(dla_core_map_,
             "dla_core_map",
             "Model Keyword with associated DLA core index",
             "The DLA core index on which model will do inference, starts at 0. Set to -1 to "
             "disable DLA.",
             DataMap());
  spec.param(temporal_map_,
             "temporal_map",
             "Model Keyword with associated frame execution delay",
             "Frame delay for model inference.",
             DataMap());
  spec.param(activation_map_,
             "activation_map",
             "Model Keyword with associated model inference activation",
             "Activation of model inference (1 = active, 0 = inactive).",
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
             DataVecMap());
  spec.param(in_tensor_names_, "in_tensor_names", "Input Tensors", "Input tensors", {});
  spec.param(out_tensor_names_, "out_tensor_names", "Output Tensors", "Output tensors", {});
  spec.param(trt_opt_profile_,
             "trt_opt_profile",
             "TensorRT Opt Profile",
             "Optimization profile for input tensors",
             {1, 1, 1});
  spec.param(allocator_, "allocator", "Allocator", "Output Allocator");
  spec.param(infer_on_cpu_, "infer_on_cpu", "Inference on CPU", "Use CPU.", false);
  spec.param(is_engine_path_, "is_engine_path", "Input path is engine file", "", false);

  spec.param(enable_fp16_, "enable_fp16", "Use fp16", "Use fp16.", false);
  spec.param(enable_cuda_graphs_,
             "enable_cuda_graphs",
             "Use CUDA graphs",
             "Enable usage of CUDA Graphs for backends which support it.",
             true);

  spec.param(dla_core_,
             "dla_core",
             "DLA core index",
             "The DLA core index to execute the engine on, starts at 0. Set to -1 (the default) to "
             "disable DLA.",
             -1);

  spec.param(
      dla_gpu_fallback_,
      "dla_gpu_fallback",
      "Enable DLA GPU fallback",
      "If DLA is enabled, use the GPU if a layer cannot be executed on DLA. If the fallback is "
      "disabled, engine creation will fail if a layer cannot executed on DLA.",
      true);

  spec.param(input_on_cuda_, "input_on_cuda", "Input buffer on CUDA", "", true);
  spec.param(output_on_cuda_, "output_on_cuda", "Output buffer on CUDA", "", true);
  spec.param(transmit_on_cuda_, "transmit_on_cuda", "Transmit message on CUDA", "", true);

  spec.param(parallel_inference_, "parallel_inference", "Parallel inference", "", true);
  spec.param(cuda_stream_pool_,
             "cuda_stream_pool",
             "CUDA Stream Pool",
             "Instance of gxf::CudaStreamPool.",
             ParameterFlag::kOptional);
}

void InferenceOp::initialize() {
  register_converter<DataMap>();
  register_converter<DataVecMap>();
  holoscan::gxf::GXFExecutor::register_codec<std::vector<ActivationSpec>>(
      "std::vector<holoscan::ops::InferenceOp::InputSpec>", true);
  Operator::initialize();
}

void InferenceOp::start() {
  try {
    // Check for the validity of parameters from configuration
    auto status = HoloInfer::inference_validity_check(model_path_map_.get().get_map(),
                                                      pre_processor_map_.get().get_map(),
                                                      inference_map_.get().get_map(),
                                                      in_tensor_names_.get(),
                                                      out_tensor_names_.get());
    if (status.get_code() != HoloInfer::holoinfer_code::H_SUCCESS) {
      status.display_message();
      HoloInfer::raise_error(module_, "Parameter Validation failed: " + status.get_message());
    }

    std::function<cudaStream_t(int32_t device_id)> allocate_cuda_stream;
    // If a CUDA stream pool is provided, use it to allocate a CUDA stream
    if (cuda_stream_pool_.try_get()) {
      allocate_cuda_stream = [this](int32_t device_id) -> cudaStream_t {
        if (cuda_stream_pool_->get_dev_id() == device_id) {
          auto maybe_stream = cuda_stream_pool_->get()->allocateStream();
          if (!maybe_stream) {
            throw std::runtime_error("Failed to allocate CUDA stream");
          }
          return maybe_stream.value()->stream().value();
        }
        return nullptr;
      };
    }

    // Create inference specification structure
    inference_specs_ =
        std::make_shared<HoloInfer::InferenceSpecs>(backend_.get(),
                                                    backend_map_.get().get_map(),
                                                    model_path_map_.get().get_map(),
                                                    pre_processor_map_.get().get_map(),
                                                    inference_map_.get().get_map(),
                                                    device_map_.get().get_map(),
                                                    dla_core_map_.get().get_map(),
                                                    temporal_map_.get().get_map(),
                                                    activation_map_.get().get_map(),
                                                    trt_opt_profile_.get(),
                                                    is_engine_path_.get(),
                                                    infer_on_cpu_.get(),
                                                    parallel_inference_.get(),
                                                    enable_fp16_.get(),
                                                    input_on_cuda_.get(),
                                                    output_on_cuda_.get(),
                                                    enable_cuda_graphs_.get(),
                                                    dla_core_.get(),
                                                    dla_gpu_fallback_.get(),
                                                    allocate_cuda_stream);
    HOLOSCAN_LOG_INFO("Inference Specifications created");
    // Create holoscan inference context
    holoscan_infer_context_ = std::make_unique<HoloInfer::InferContext>();

    // Set and transfer inference specification to inference context
    // inference specifications are updated with memory allocations
    status = holoscan_infer_context_->set_inference_params(inference_specs_);
    if (status.get_code() != HoloInfer::holoinfer_code::H_SUCCESS) {
      status.display_message();
      HoloInfer::raise_error(module_, "Start, Parameters setup, " + status.get_message());
    }
    HOLOSCAN_LOG_INFO("Inference context setup complete");
  } catch (const std::bad_alloc& b_) {
    HoloInfer::raise_error(module_, "Start, Memory allocation, Message: " + std::string(b_.what()));
  } catch (const std::runtime_error& rt_) {
    HOLOSCAN_LOG_ERROR(rt_.what());
    throw;
  } catch (...) {
    HoloInfer::raise_error(module_, "Start, Unknown exception");
  }
}

void InferenceOp::stop() {
  inference_specs_.reset();
  holoscan_infer_context_.reset();
}

void InferenceOp::compute(InputContext& op_input, OutputContext& op_output,
                          ExecutionContext& context) {
  // get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
  auto allocator =
      nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), allocator_->gxf_cid());
  auto cont = context.context();
  try {
    // Get activation maps spec from user, if any, build filters on inference specs
    const auto act_map_specs_message =
        op_input.receive<std::vector<ActivationSpec>>("model_activation_specs");
    std::vector<ActivationSpec> maybe_act_map_specs;

    // Update activation_map by specs
    if (act_map_specs_message) {
      maybe_act_map_specs = act_map_specs_message.value();
    }

    // Extract relevant data from input GXF Receivers, and update inference specifications
    // (cuda_stream will be set by get_data_per_model)
    cudaStream_t cuda_stream{};
    gxf_result_t stat = holoscan::utils::get_data_per_model(op_input,
                                                            in_tensor_names_.get(),
                                                            inference_specs_->data_per_tensor_,
                                                            dims_per_tensor_,
                                                            input_on_cuda_.get(),
                                                            module_,
                                                            cuda_stream);

    if (stat != GXF_SUCCESS) {
      HoloInfer::raise_error(module_, "Tick, Data extraction");
    }

    // Transmit this stream on the output port if needed
    if (cuda_stream != cudaStreamDefault && output_on_cuda_.get()) {
      HOLOSCAN_LOG_TRACE("InferenceOp: forwarding CUDA stream from receivers input to output");
      op_output.set_cuda_stream(cuda_stream, "transmitter");
    }

    // check for tensor validity the first time
    if (validate_tensor_dimensions_) {
      validate_tensor_dimensions_ = false;
      auto model_in_dims_map = holoscan_infer_context_->get_input_dimensions();

      auto dim_status = HoloInfer::tensor_dimension_check(
          pre_processor_map_.get().get_map(), model_in_dims_map, dims_per_tensor_);
      if (dim_status.get_code() != HoloInfer::holoinfer_code::H_SUCCESS) {
        HoloInfer::raise_error(module_,
                               "Compute, Inference execution, " + dim_status.get_message());
      }
    }
    // Execute inference and populate output buffer in inference specifications
    HoloInfer::TimePoint s_time, e_time;
    HoloInfer::timer_init(s_time);

    // Set activation map for inference specifications
    stat = holoscan::utils::set_activation_per_model(
        inference_specs_, activation_map_.get().get_map(), maybe_act_map_specs, module_);
    if (stat != GXF_SUCCESS) {
      HoloInfer::raise_error(module_, "Compute, Inference activation specification");
    }
    auto status = holoscan_infer_context_->execute_inference(inference_specs_, cuda_stream);
    HoloInfer::timer_init(e_time);
    HoloInfer::timer_check(s_time, e_time, "Inference Operator: Inference execution");
    if (status.get_code() != HoloInfer::holoinfer_code::H_SUCCESS) {
      status.display_message();
      HoloInfer::raise_error(module_, "Compute, Inference execution, " + status.get_message());
    }
    HOLOSCAN_LOG_DEBUG(status.get_message());

    // Get output dimensions
    auto model_out_dims_map = holoscan_infer_context_->get_output_dimensions();

    // Transmit output buffers via a single GXF transmitter
    stat = holoscan::utils::transmit_data_per_model(cont,
                                                    inference_map_.get().get_map(),
                                                    inference_specs_->output_per_model_,
                                                    op_output,
                                                    out_tensor_names_.get(),
                                                    model_out_dims_map,
                                                    output_on_cuda_.get(),
                                                    transmit_on_cuda_.get(),
                                                    allocator.value(),
                                                    module_,
                                                    cuda_stream);
    if (stat != GXF_SUCCESS) {
      HoloInfer::raise_error(module_, "Compute, Data Transmission");
    }
  } catch (const std::runtime_error& r_) {
    HoloInfer::raise_error(module_,
                           "Compute, Inference execution, Message->" + std::string(r_.what()));
  } catch (...) {
    HoloInfer::raise_error(module_, "Compute, unknown exception");
  }
}

}  // namespace holoscan::ops

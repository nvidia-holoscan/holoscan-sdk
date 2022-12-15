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
#include "multiai_inference.hpp"

#include <sys/stat.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace nvidia {
namespace holoscan {
namespace multiai {

gxf_result_t MultiAIInference::start() {
  try {
    // Check for the validity of parameters from configuration
    auto status = HoloInfer::multiai_inference_validity_check(model_path_map_.get(),
                                                              pre_processor_map_.get(),
                                                              inference_map_.get(),
                                                              in_tensor_names_.get(),
                                                              out_tensor_names_.get());
    if (status.get_code() != HoloInfer::holoinfer_code::H_SUCCESS) {
      return HoloInfer::report_error(module_,
                                     "Parameter Validation failed: " + status.get_message());
    }

    // Create multiai specification structure
    multiai_specs_ = std::make_shared<HoloInfer::MultiAISpecs>(backend_.get(),
                                                               model_path_map_.get(),
                                                               inference_map_.get(),
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
      return HoloInfer::report_error(module_, "Start, Parameters setup, " + status.get_message());
    }
  } catch (const std::bad_alloc& b_) {
    return HoloInfer::report_error(module_,
                                   "Start, Memory allocation, Message: " + std::string(b_.what()));
  } catch (...) { return HoloInfer::report_error(module_, "Start, Unknown exception"); }

  return GXF_SUCCESS;
}

gxf_result_t MultiAIInference::stop() {
  holoscan_infer_context_.reset();
  return GXF_SUCCESS;
}

gxf_result_t MultiAIInference::tick() {
  try {
    // Extract relevant data from input GXF Receivers, and update multiai specifications
    gxf_result_t stat = HoloInfer::multiai_get_data_per_model(receivers_.get(),
                                                              in_tensor_names_.get(),
                                                              multiai_specs_->data_per_tensor_,
                                                              dims_per_tensor_,
                                                              input_on_cuda_.get(),
                                                              module_);

    if (stat != GXF_SUCCESS) { return HoloInfer::report_error(module_, "Tick, Data extraction"); }

    auto status = HoloInfer::map_data_to_model_from_tensor(pre_processor_map_.get(),
                                                           multiai_specs_->data_per_model_,
                                                           multiai_specs_->data_per_tensor_);
    if (status.get_code() != HoloInfer::holoinfer_code::H_SUCCESS) {
      return HoloInfer::report_error(module_, "Tick, Data mapping, " + status.get_message());
    }

    // Execute inference and populate output buffer in multiai specifications
    status = holoscan_infer_context_->execute_inference(multiai_specs_->data_per_model_,
                                                        multiai_specs_->output_per_model_);
    if (status.get_code() != HoloInfer::holoinfer_code::H_SUCCESS) {
      return HoloInfer::report_error(module_, "Tick, Inference execution, " + status.get_message());
    }
    GXF_LOG_DEBUG("%s", status.get_message().c_str());

    // Get output dimensions
    auto model_out_dims_map = holoscan_infer_context_->get_output_dimensions();

    auto cont = context();

    // Transmit output buffers via a single GXF transmitter
    stat = HoloInfer::multiai_transmit_data_per_model(cont,
                                                      inference_map_.get(),
                                                      multiai_specs_->output_per_model_,
                                                      transmitter_.get(),
                                                      out_tensor_names_.get(),
                                                      model_out_dims_map,
                                                      output_on_cuda_.get(),
                                                      transmit_on_cuda_.get(),
                                                      data_type_,
                                                      module_,
                                                      allocator_.get());
    if (stat != GXF_SUCCESS) { return HoloInfer::report_error(module_, "Tick, Data Transmission"); }
  } catch (const std::runtime_error& r_) {
    return HoloInfer::report_error(module_,
                                   "Tick, Inference execution, Message->" + std::string(r_.what()));
  } catch (...) { return HoloInfer::report_error(module_, "Tick, unknown exception"); }
  return GXF_SUCCESS;
}

gxf_result_t MultiAIInference::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;

  result &= registrar->parameter(backend_, "backend", "Supported backend");

  result &= registrar->parameter(model_path_map_,
                                 "model_path_map",
                                 "Model Keyword with File Path",
                                 "Path to ONNX model to be loaded.");
  result &= registrar->parameter(pre_processor_map_,
                                 "pre_processor_map",
                                 "Pre processor setting per model",
                                 "Pre processed data to model map.");
  result &= registrar->parameter(
      inference_map_, "inference_map", "Inferred tensor per model", "Tensor to model map.");
  result &= registrar->parameter(
      in_tensor_names_, "in_tensor_names", "Input Tensors", "Input tensors", {std::string("")});
  result &= registrar->parameter(
      out_tensor_names_, "out_tensor_names", "Output Tensors", "Output tensors", {std::string("")});
  result &= registrar->parameter(allocator_, "allocator", "Allocator", "Output Allocator");
  result &= registrar->parameter(
      is_engine_path_, "is_engine_path", "Input file paths are trt engine files", "", false);
  result &=
      registrar->parameter(infer_on_cpu_, "infer_on_cpu", "Inference on CPU", "Use CPU.", false);
  result &=
      registrar->parameter(enable_fp16_, "enable_fp16", "use FP16 engine", "Use FP16.", false);
  result &= registrar->parameter(
      input_on_cuda_, "input_on_cuda", "Input for inference on cuda", "", true);
  result &=
      registrar->parameter(output_on_cuda_, "output_on_cuda", "Inferred output on cuda", "", true);
  result &= registrar->parameter(
      transmit_on_cuda_, "transmit_on_cuda", "Transmit message on cuda", "", true);

  result &= registrar->parameter(parallel_inference_, "parallel_inference", "Parallel inference");

  result &= registrar->parameter(
      receivers_, "receivers", "Receivers", "List of receivers to take input tensors");
  result &= registrar->parameter(transmitter_, "transmitter", "Transmitter", "Transmitter");

  return gxf::ToResultCode(result);
}

}  // namespace multiai
}  // namespace holoscan
}  // namespace nvidia

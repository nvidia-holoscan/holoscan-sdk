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
#include "multiai_postprocessor.hpp"

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

gxf_result_t MultiAIPostprocessor::start() {
  try {
    // Check for the validity of parameters from configuration
    if (input_on_cuda_.get() || output_on_cuda_.get() || transmit_on_cuda_.get()) {
      return HoloInfer::report_error(module_,
                                     "CUDA based data not supported in Multi AI post processor");
    }
    auto status = HoloInfer::multiai_processor_validity_check(
        processed_map_.get(), in_tensor_names_.get(), out_tensor_names_.get());
    if (status.get_code() != HoloInfer::holoinfer_code::H_SUCCESS) {
      return HoloInfer::report_error(module_,
                                     "Parameter Validation failed: " + status.get_message());
    }

    // Create holoscan processing context
    holoscan_postprocess_context_ = std::make_unique<HoloInfer::ProcessorContext>();
  } catch (const std::bad_alloc& b_) {
    return HoloInfer::report_error(module_,
                                   "Start, Memory allocation, Message: " + std::string(b_.what()));
  } catch (...) { return HoloInfer::report_error(module_, "Start, Unknown exception"); }

  // Initialize holoscan processing context
  auto status = holoscan_postprocess_context_->initialize(process_operations_.get());
  if (status.get_code() != HoloInfer::holoinfer_code::H_SUCCESS) {
    return HoloInfer::report_error(module_, "Start, Out data setup");
  }

  return GXF_SUCCESS;
}

gxf_result_t MultiAIPostprocessor::stop() {
  return GXF_SUCCESS;
}

gxf_result_t MultiAIPostprocessor::tick() {
  try {
    // Extract relevant data from input GXF Receivers, and update data per model
    gxf_result_t stat = HoloInfer::multiai_get_data_per_model(receivers_.get(),
                                                              in_tensor_names_.get(),
                                                              data_per_tensor_,
                                                              dims_per_tensor_,
                                                              input_on_cuda_.get(),
                                                              module_);
    if (stat != GXF_SUCCESS) { return HoloInfer::report_error(module_, "Tick, Data extraction"); }

    // Execute processing
    auto status = holoscan_postprocess_context_->process(
        process_operations_.get(), processed_map_.get(), data_per_tensor_, dims_per_tensor_);
    if (status.get_code() != HoloInfer::holoinfer_code::H_SUCCESS) {
      return HoloInfer::report_error(module_, "Tick, post_process");
    }

    // Get processed data and dimensions (currently only on host)
    auto processed_data = holoscan_postprocess_context_->get_processed_data();
    auto processed_dims = holoscan_postprocess_context_->get_processed_data_dims();

    auto cont = context();
    // Transmit output buffers via a single GXF transmitter
    stat = HoloInfer::multiai_transmit_data_per_model(cont,
                                                      processed_map_.get(),
                                                      processed_data,
                                                      transmitter_.get(),
                                                      out_tensor_names_.get(),
                                                      processed_dims,
                                                      output_on_cuda_.get(),
                                                      transmit_on_cuda_.get(),
                                                      gxf::PrimitiveType::kFloat32,
                                                      module_,
                                                      allocator_.get());
    if (stat != GXF_SUCCESS) { return HoloInfer::report_error(module_, "Tick, Data Transmission"); }
  } catch (const std::runtime_error& r_) {
    return HoloInfer::report_error(module_, "Tick, Message->" + std::string(r_.what()));
  } catch (...) { return HoloInfer::report_error(module_, "Tick, unknown exception"); }
  return GXF_SUCCESS;
}

gxf_result_t MultiAIPostprocessor::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;
  result &= registrar->parameter(process_operations_,
                                 "process_operations",
                                 "Operations per tensor",
                                 "Operations in sequence on tensors.");
  result &= registrar->parameter(processed_map_, "processed_map", "Processed tensor map");
  result &= registrar->parameter(
      in_tensor_names_, "in_tensor_names", "Input Tensors", "Input tensors", {std::string("")});
  result &= registrar->parameter(
      out_tensor_names_, "out_tensor_names", "Output Tensors", "Output tensors", {std::string("")});
  result &= registrar->parameter(
      input_on_cuda_, "input_on_cuda", "Input for processing on cuda", "", false);
  result &= registrar->parameter(
      output_on_cuda_, "output_on_cuda", "Processed output on cuda", "", false);
  result &= registrar->parameter(
      transmit_on_cuda_, "transmit_on_cuda", "Transmit message on cuda", "", false);
  result &= registrar->parameter(allocator_, "allocator", "Allocator", "Output Allocator");
  result &= registrar->parameter(
      receivers_, "receivers", "Receivers", "List of receivers to take input tensors");
  result &= registrar->parameter(transmitter_, "transmitter", "Transmitter", "Transmitter");

  return gxf::ToResultCode(result);
}

}  // namespace multiai
}  // namespace holoscan
}  // namespace nvidia

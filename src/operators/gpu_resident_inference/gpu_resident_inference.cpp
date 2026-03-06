/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/operators/gpu_resident_inference/gpu_resident_inference.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "holoscan/core/execution_context.hpp"
#include "holoscan/core/io_context.hpp"
#include "holoscan/core/operator_spec.hpp"

#include <holoinfer_utils.hpp>

namespace holoscan::ops {

GPUResidentInferenceOp::GPUResidentInferenceOp(const std::string& config_file)
    : config_file_(config_file) {}

void GPUResidentInferenceOp::setup(OperatorSpec& spec) {
  HOLOSCAN_LOG_INFO("Loading config file: {}", config_file_);
  auto status = HoloInfer::load_yaml(config_file_,
                                     model_path_map_,
                                     pre_processor_map_,
                                     inference_map_,
                                     batch_sizes_,
                                     in_tensor_names_,
                                     out_tensor_names_,
                                     tensor_to_buffersize_,
                                     tensor_to_datatype_);

  if (status.get_code() != HoloInfer::holoinfer_code::H_SUCCESS) {
    status.display_message();
    HoloInfer::raise_error(module_, "Setup: Loading yaml config failed: " + status.get_message());
  }

  // In current release, GPU Resident Inference operator supports:
  // - single model per inference instance
  // - single input and single output to/from the model
  // - Multiple instances of GPU resident inference operator in a Holoscan application is not tested
  // in the current release

  auto in_tensor_it = tensor_to_buffersize_.find(in_tensor_names_[0]);
  auto in_datatype_it = tensor_to_datatype_.find(in_tensor_names_[0]);
  auto out_tensor_it = tensor_to_buffersize_.find(out_tensor_names_[0]);
  auto out_datatype_it = tensor_to_datatype_.find(out_tensor_names_[0]);

  if (in_tensor_it == tensor_to_buffersize_.end() || in_datatype_it == tensor_to_datatype_.end() ||
      out_tensor_it == tensor_to_buffersize_.end() ||
      out_datatype_it == tensor_to_datatype_.end()) {
    HoloInfer::raise_error(module_, "Setup: Required tensor configurations not found");
  }

  auto in_buffer_size = in_tensor_it->second * HoloInfer::get_element_size(in_datatype_it->second);
  auto out_buffer_size = out_tensor_it->second *
                         holoscan::inference::get_element_size(out_datatype_it->second);

  HOLOSCAN_LOG_INFO("input buffer size {}, output buffer size {}", in_buffer_size, out_buffer_size);
  spec.device_input("in", in_buffer_size);
  spec.device_output("out", out_buffer_size);
}

void GPUResidentInferenceOp::start() {
  try {
    // Check for the validity of parameters from configuration
    auto status = HoloInfer::inference_validity_check(
        model_path_map_, pre_processor_map_, inference_map_, in_tensor_names_, out_tensor_names_);
    if (status.get_code() != HoloInfer::holoinfer_code::H_SUCCESS) {
      status.display_message();
      HoloInfer::raise_error(module_, "Parameter Validation failed: " + status.get_message());
    }

    std::function<cudaStream_t(int32_t)> allocate_cuda_stream;
    // If a CUDA stream pool is provided, use it to allocate a CUDA stream
    allocate_cuda_stream = [this](int32_t device_id) -> cudaStream_t {
      cudaStream_t stream;
      // Set the device context before creating the stream
      cudaSetDevice(device_id);
      // Create a new stream on the specified device
      cudaError_t err = cudaStreamCreate(&stream);
      if (err != cudaSuccess) {
        HOLOSCAN_LOG_ERROR("Failed to create CUDA stream: {}", cudaGetErrorString(err));
        throw std::runtime_error("CUDA stream creation failed");
      }
      return stream;
    };
    // Create inference specification structure
    inference_specs_ = std::make_shared<HoloInfer::InferenceSpecs>(backend_,
                                                                   backend_map_,
                                                                   model_path_map_,
                                                                   pre_processor_map_,
                                                                   inference_map_,
                                                                   device_map_,
                                                                   dla_core_map_,
                                                                   temporal_map_,
                                                                   activation_map_,
                                                                   batch_sizes_,
                                                                   dynamic_input_dims_,
                                                                   is_engine_path_,
                                                                   infer_on_cpu_,
                                                                   parallel_inference_,
                                                                   enable_fp16_,
                                                                   input_on_cuda_,
                                                                   output_on_cuda_,
                                                                   use_cuda_graphs_,
                                                                   dla_core_,
                                                                   dla_gpu_fallback_,
                                                                   true,
                                                                   allocate_cuda_stream);

    HOLOSCAN_LOG_INFO("Inference Specifications created");

    // Create holoscan inference context
    holoscan_infer_context_ = std::make_unique<HoloInfer::InferContext>();

    if (device_memory("in") == nullptr) {
      HOLOSCAN_LOG_ERROR("Could not find input device memory");
      throw std::runtime_error("Input device memory not found in GPUResidentInferenceOp::start");
    }

    if (device_memory("out") == nullptr) {
      HOLOSCAN_LOG_ERROR("Could not find output device memory");
      throw std::runtime_error("Output device memory not found in GPUResidentInferenceOp::start");
    }

    inference_specs_->gpu_resident_input_ = device_memory("in");
    inference_specs_->gpu_resident_output_ = device_memory("out");

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

void GPUResidentInferenceOp::stop() {
  inference_specs_.reset();
  holoscan_infer_context_.reset();
}

void GPUResidentInferenceOp::compute([[maybe_unused]] InputContext& op_input,
                                     [[maybe_unused]] OutputContext& op_output,
                                     [[maybe_unused]] ExecutionContext& context) {
  auto cuda_stream_ptr = cuda_stream();

  HOLOSCAN_LOG_DEBUG("GPUResidentInferenceOp::compute() -- {} -- Input at: {}",
                     name(),
                     inference_specs_->gpu_resident_input_);
  HOLOSCAN_LOG_DEBUG("GPUResidentInferenceOp::compute() -- {} -- Output at: {}",
                     name(),
                     inference_specs_->gpu_resident_output_);

  if (!cuda_stream_ptr) {
      HoloInfer::raise_error(module_, "Compute, Invalid CUDA stream pointer");
  }
  auto status = holoscan_infer_context_->execute_inference(inference_specs_, *cuda_stream_ptr);

  if (status.get_code() != HoloInfer::holoinfer_code::H_SUCCESS) {
    status.display_message();
    HoloInfer::raise_error(module_, "Compute, Inference execution, " + status.get_message());
  }
}

}  // namespace holoscan::ops

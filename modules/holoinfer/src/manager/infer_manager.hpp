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
#ifndef _HOLOSCAN_INFER_MANAGER_H
#define _HOLOSCAN_INFER_MANAGER_H

#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>

#include <holoinfer.hpp>
#include <holoinfer_buffer.hpp>
#include <holoinfer_constants.hpp>
#include <holoinfer_utils.hpp>
#include <infer/infer.hpp>

#if __has_include(<onnxruntime_c_api.h>)
#define use_onnxruntime 1
#include <infer/onnx/core.hpp>
#endif

#ifdef use_torch
#include <infer/torch/core.hpp>
#endif

#include <infer/trt/core.hpp>
#include <params/infer_param.hpp>

namespace holoscan {
namespace inference {
/**
 * @brief Manager class for inference
 */
class ManagerInfer {
 public:
  /**
   * @brief Default Constructor
   */
  ManagerInfer();

  /**
   * @brief Destructor
   */
  ~ManagerInfer();

  /**
   * @brief Create inference settings and memory
   *
   * @param inference_specs specifications for inference
   *
   * @returns InferStatus with appropriate code and message
   */
  InferStatus set_inference_params(std::shared_ptr<InferenceSpecs>& inference_specs);

  /**
   * @brief Prepares and launches single/multiple inference
   *
   * @param preprocess_data_map Input DataMap with model name as key and DataBuffer as value
   * @param output_data_map Output DataMap with tensor name as key and DataBuffer as value
   *
   * @returns InferStatus with appropriate code and message
   */
  InferStatus execute_inference(DataMap& preprocess_data_map, DataMap& output_data_map);

  /**
   * @brief Executes Core inference for a particular model and generates inferred data
   *
   * @param model_name Input model to do the inference on
   * @param permodel_preprocess_data Input DataMap with model name as key and DataBuffer as value
   * @param permodel_output_data Output DataMap with tensor name as key and DataBuffer as value
   *
   * @returns InferStatus with appropriate code and message
   */
  InferStatus run_core_inference(const std::string& model_name, DataMap& permodel_preprocess_data,
                                 DataMap& permodel_output_data);
  /**
   * @brief Cleans up internal context per model
   *
   */
  void cleanup();

  /**
   * @brief Get input dimension per model
   *
   * @returns Map with model name as key and dimension as value
   */
  DimType get_input_dimensions() const;

  /**
   * @brief Get output dimension per tensor
   *
   * @returns Map with tensor name as key and dimension as value
   */
  DimType get_output_dimensions() const;

 private:
  /// Flag to infer models in parallel. Defaults to False
  bool parallel_processing_ = false;

  /// Flag to demonstrate if input data buffer is on cuda
  bool cuda_buffer_in_ = false;

  /// Flag to demonstrate if output data buffer will be on cuda
  bool cuda_buffer_out_ = false;

  /// @brief Flag to demonstrate if multi-GPU feature has Peer to Peer transfer enabled.
  bool mgpu_p2p_transfer = true;

  /// @brief Map to store cuda streams associated with each input tensor in each model on GPU-dt.
  /// Will be used with Multi-GPU feature.
  std::map<std::string, std::map<std::string, cudaStream_t>> input_streams_gpudt;

  /// @brief Map to store cuda streams associated with each output tensor in each model on GPU-dt.
  /// Will be used with Multi-GPU feature.
  std::map<std::string, std::map<std::string, cudaStream_t>> output_streams_gpudt;

  /// @brief Map to store cuda streams associated with each input tensor in each model on the
  /// inference device.  Will be used with Multi-GPU feature.
  std::map<std::string, std::map<std::string, cudaStream_t>> input_streams_device;

  /// @brief Map to store cuda streams associated with each output tensor in each model on the
  /// inference device. Will be used with Multi-GPU feature.
  std::map<std::string, std::map<std::string, cudaStream_t>> output_streams_device;

  /// Map storing parameters per model
  std::map<std::string, std::unique_ptr<Params>> infer_param_;

  /// Map storing Inference context per model
  std::map<std::string, std::unique_ptr<InferBase>> holo_infer_context_;

  /// Map storing input dimension per model
  DimType models_input_dims_;

  /// Output buffer for multi-GPU inference
  std::map<std::string, DataMap> mgpu_output_buffer_;

  /// Input buffer for multi-gpu inference
  std::map<std::string, DataMap> mgpu_input_buffer_;

  /// Data transfer GPU. Default: 0. Not configurable in this release.
  int device_gpu_dt = 0;

  /// Map storing inferred output dimension per tensor
  DimType models_output_dims_;

  /// Map storing Backends supported with holoinfer mapping
  inline static std::map<std::string, holoinfer_backend> supported_backend_{
      {"onnxrt", holoinfer_backend::h_onnx},
      {"trt", holoinfer_backend::h_trt},
      {"torch", holoinfer_backend::h_torch}};
};

/// Pointer to manager class for inference
std::unique_ptr<ManagerInfer> manager;

}  // namespace inference
}  // namespace holoscan

#endif

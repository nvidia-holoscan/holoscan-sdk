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

#ifndef HOLOSCAN_OPERATORS_GPU_RESIDENT_INFERENCE_GPU_RESIDENT_INFERENCE_HPP
#define HOLOSCAN_OPERATORS_GPU_RESIDENT_INFERENCE_GPU_RESIDENT_INFERENCE_HPP

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <holoscan/core/gpu_resident_operator.hpp>
#include <holoscan/core/io_context.hpp>
#include <holoscan/core/io_spec.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>

#include <holoinfer.hpp>
#include <holoinfer_buffer.hpp>

namespace HoloInfer = holoscan::inference;

namespace holoscan::ops {
/**
 * @brief GPU Resident Inference Operator class to perform model inference.
 *
 * ==Parameters==
 *
 * - **backend**: Backend to use for inference. Only TensorRT supported. `"trt"` for TensorRT
 *   (default: `"trt"`).
 * - **config_file**: Path to the config file.
 * - **in_tensor_names**: Input tensors.
 * - **out_tensor_names**: Output tensors.
 * - **batch_sizes**: Batch sizes per model. Key is model name and value is the batch size.
 * - **model_path_map**: Model to path map. Key is model name and value is the model path.
 * - **pre_processor_map**: Model to Pre processed input data (to the model) map. Key is model name
 * and value is the input tensor names to the model.
 * - **inference_map**: Model to output tensor map. Key is model name and value is the output tensor
 * names from the model.
 * - **tensor_to_buffersize**: Map holding buffer size per tensor. Key is tensor name and value is
 * the buffer size.
 * - **tensor_to_datatype**: Map holding data type per tensor. Key is tensor name and value is the
 * data type.
 * - **device_map**: Device map. Key is model name and value is the GPU ID for inference.
 * - **dla_core_map**: DLA core map. Key is model name and value is the DLA core index for
 * inference.
 * - **temporal_map**: Temporal map. Key is model name and value is the frame delay for model
 * inference.
 * - **activation_map**: Activation map. Key is model name and value is the activation state for
 * model inference.
 * - **backend_map**: Backend map. Key is model name and value is the backend type for inference.
 * - **parallel_inference**: Parallel inference flag.
 * - **infer_on_cpu**: Infer on CPU flag. Always false. Not configurable.
 * - **enable_fp16**: Enable FP16 flag.
 * - **input_on_cuda**: Input on CUDA flag. Always true. Not configurable.
 * - **output_on_cuda**: Output on CUDA flag. Always true. Not configurable.
 * - **is_engine_path**: Is engine path flag.
 * - **use_cuda_graphs**: Use CUDA graphs flag. Always false. Not configurable.
 * - **dla_core**: DLA core index. Default is -1.
 * - **dla_gpu_fallback**: DLA GPU fallback flag.
 * - **dynamic_input_dims**: Dynamic input dimensions flag.
 */
class GPUResidentInferenceOp : public holoscan::GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(GPUResidentInferenceOp, holoscan::GPUResidentOperator)

  /// Default constructor.
  GPUResidentInferenceOp() = default;
  /// Constructor with config file.
  explicit GPUResidentInferenceOp(const std::string& config_file);
  /// Setup the operator.
  void setup(OperatorSpec& spec) override;
  /// Start function of the operator.
  void start() override;
  /// Compute function of the operator.
  void compute([[maybe_unused]] InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override;
  /// Stop function of the operator.
  void stop() override;

 private:
  /// Parameters for inference

  /// Backend to use for inference. Only TensorRT supported. `"trt"` for TensorRT
  std::string backend_ = "trt";
  /// Path to the config file.
  std::string config_file_;
  /// Input tensor names.
  std::vector<std::string> in_tensor_names_;
  /// Output tensor names.
  std::vector<std::string> out_tensor_names_;
  /// Batch sizes per model. Key is model name and value is the batch size.
  std::map<std::string, std::vector<std::string>> batch_sizes_;
  /// Model path map. Key is model name and value is the model path.
  std::map<std::string, std::string> model_path_map_;
  /// Pre processor map. Key is model name and value is the input tensor names to the model.
  std::map<std::string, std::vector<std::string>> pre_processor_map_;
  /// Inference map. Key is model name and value is the output tensor names from the model.
  std::map<std::string, std::vector<std::string>> inference_map_;

  // NOTE 1: Below parameters are not tested in the current release

  /// Device map. Key is model name and value is the GPU ID for inference.
  std::map<std::string, std::string> device_map_;
  /// DLA core map. Key is model name and value is the DLA core index for inference.
  std::map<std::string, std::string> dla_core_map_;
  /// Temporal map. Key is model name and value is the frame delay for model inference.
  std::map<std::string, std::string> temporal_map_;
  /// Activation map. Key is model name and value is the activation state for model inference.
  std::map<std::string, std::string> activation_map_;
  /// Backend map. Key is model name and value is the backend type for inference.
  std::map<std::string, std::string> backend_map_;
  /// Parallel inference flag.
  bool parallel_inference_ = true;
  /// Infer on CPU flag. Always false. Not configurable.
  bool infer_on_cpu_ = false;
  /// Enable FP16 flag.
  bool enable_fp16_ = false;
  /// Input on CUDA flag. Always true. Not configurable.
  bool input_on_cuda_ = true;
  /// Output on CUDA flag. Always true. Not configurable.
  bool output_on_cuda_ = true;
  /// Is engine path flag.
  bool is_engine_path_ = false;
  /// Use CUDA graphs flag. Always false. Not configurable.
  bool use_cuda_graphs_ = false;
  /// DLA core index. Default is -1.
  int32_t dla_core_ = -1;
  /// DLA GPU fallback flag.
  bool dla_gpu_fallback_ = true;
  /// Dynamic input dimensions flag.
  bool dynamic_input_dims_ = false;
  // NOTE 1 ends

  /// Map holding buffer size per tensor. Key is tensor name and value is the buffer size.
  std::map<std::string, size_t> tensor_to_buffersize_;
  /// Map holding data type per tensor. Key is tensor name and value is the data type.
  std::map<std::string, HoloInfer::holoinfer_datatype> tensor_to_datatype_;

  /// Pointer to inference context.
  std::unique_ptr<HoloInfer::InferContext> holoscan_infer_context_;

  /// Pointer to inference specifications
  std::shared_ptr<HoloInfer::InferenceSpecs> inference_specs_;

  /// Map holding dimensions per model. Key is model name and value is a vector with
  /// dimensions.
  std::map<std::string, std::vector<int>> dims_per_tensor_;

  /// Operator Identifier, used in reporting.
  const std::string module_{"GPU Resident Inference Operator"};
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_GPU_RESIDENT_INFERENCE_GPU_RESIDENT_INFERENCE_HPP */

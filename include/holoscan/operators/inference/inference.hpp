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

#ifndef HOLOSCAN_OPERATORS_INFERENCE_INFERENCE_HPP
#define HOLOSCAN_OPERATORS_INFERENCE_INFERENCE_HPP

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "holoscan/core/io_context.hpp"
#include "holoscan/core/io_spec.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"

#include <holoinfer.hpp>

namespace HoloInfer = holoscan::inference;

namespace holoscan::ops {
/**
 * @brief Inference Operator class to perform single/multi model inference.
 *
 * ==Named Inputs==
 *
 * - **receivers** : multi-receiver accepting `nvidia::gxf::Tensor`(s)
 *   - Any number of upstream ports may be connected to this `receivers` port. The operator
 *     will search across all messages for tensors matching those specified in
 *     `in_tensor_names`. These are the set of input tensors used by the models in
 *     `inference_map`.
 *
 * ==Named Outputs==
 *
 * - **transmitter** : `nvidia::gxf::Tensor`(s)
 *   - A message containing tensors corresponding to the inference results from all models
 *     will be emitted. The names of the tensors transmitted correspond to those in
 *     `out_tensor_names`.
 *
 * ==Parameters==
 *
 * For more details on `InferenceOp` parameters, see
 * [Customizing the Inference
 * Operator](https://docs.nvidia.com/holoscan/sdk-user-guide/examples/byom.html#customizing-the-inference-operator)
 * or refer to [Inference](https://docs.nvidia.com/holoscan/sdk-user-guide/inference.html).
 *
 * - **backend**: Backend to use for inference. Set `"trt"` for TensorRT, `"torch"` for LibTorch
 *   and `"onnxrt"` for the ONNX runtime.
 * - **allocator**: Memory allocator to use for the output.
 * - **inference_map**: Tensor to model map.
 * - **model_path_map**: Path to the ONNX model to be loaded.
 * - **pre_processor_map**: Pre processed data to model map.
 * - **device_map**: Mapping of model (`DataMap`) to GPU ID for inference. Optional.
 * - **dla_core_map**: Mapping of model (`DataMap`) to DLA core index for inference. Optional.
 * - **backend_map**: Mapping of model (`DataMap`) to backend type for inference.
 *   Backend options: `"trt"` or `"torch"`. Optional.
 * - **temporal_map**: Mapping of model (`DataMap`) to a frame delay for model inference. Optional.
 * - **activation_map**: Mapping of model (`DataMap`) to a activation state for model inference.
 *   Optional.
 * - **in_tensor_names**: Input tensors (`std::vector<std::string>`). Optional.
 * - **out_tensor_names**: Output tensors (`std::vector<std::string>`). Optional.
 * - **infer_on_cpu**: Whether to run the computation on the CPU instead of GPU. Optional
 *   (default: `false`).
 * - **parallel_inference**: Whether to enable parallel execution. Optional (default: `true`).
 * - **input_on_cuda**: Whether the input buffer is on the GPU. Optional (default: `true`).
 * - **output_on_cuda**: Whether the output buffer is on the GPU. Optional (default: `true`).
 * - **transmit_on_cuda**: Whether to transmit the message on the GPU. Optional (default: `true`).
 * - **enable_fp16**: Use 16-bit floating point computations. Optional (default: `false`).
 * - **enable_cuda_graphs**: Use CUDA Graphs. Optional (default: `true`).
 * - **dla_core**: The DLA core index to execute the engine on, starts at 0. Set to -1 to disable
 *   DLA. Optional (default: `-1`).
 * - **dla_gpu_fallback**: If DLA is enabled, use the GPU if a layer cannot be executed on DLA. If
 *   this is disabled engine creation will fail if a layer cannot executed on DLA. Optional
 *   (default: `true`).
 * - **is_engine_path**: Whether the input model path mapping is for trt engine files. Optional
 *   (default: `false`).
 * - **cuda_stream_pool**: `holoscan::CudaStreamPool` instance to allocate CUDA streams. Optional
 *   (default: `nullptr`).
 *
 * ==Device Memory Requirements==
 *
 * When using this operator with a `BlockMemoryPool`, `num_blocks` must be greater than or equal to
 * the number of output tensors that will be produced. The `block_size` in bytes must be greater
 * than or equal to the largest output tensor (in bytes). If `output_on_cuda` is true, the blocks
 * should be in device memory (`storage_type`=1), otherwise they should be CUDA pinned host memory
 * (`storage_type`=0).
 */
class InferenceOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(InferenceOp)

  InferenceOp() = default;

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void start() override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;
  void stop() override;

  /**
   * DataMap specification
   */
  struct DataMap {
    DataMap() = default;
    explicit operator bool() const noexcept { return !mappings_.empty(); }
    void insert(const std::string& key, const std::string& value) { mappings_[key] = value; }

    std::map<std::string, std::string> get_map() const { return mappings_; }

    std::map<std::string, std::string> mappings_;
  };

  /**
   * DataVecMap specification
   */
  struct DataVecMap {
    DataVecMap() = default;

    explicit operator bool() const noexcept { return !mappings_.empty(); }
    void insert(const std::string& key, const std::vector<std::string>& value) {
      for (auto const& val : value)
        mappings_[key].push_back(val);
    }

    std::map<std::string, std::vector<std::string>> get_map() const { return mappings_; }

    std::map<std::string, std::vector<std::string>> mappings_;
  };

  // Activation input specification
  using ActivationSpec = holoscan::inference::ActivationSpec;

 private:
  ///  @brief Map with key as model name and value as vector of inferred tensor name
  Parameter<DataVecMap> inference_map_;

  ///  @brief Map with key as model name and value as model file path
  Parameter<DataMap> model_path_map_;

  ///  @brief Map with key as model name and value as vector of input tensor names
  Parameter<DataVecMap> pre_processor_map_;

  /// @brief Map with key as model name and value as GPU ID for inference
  Parameter<DataMap> device_map_;

  /// @brief Map with key as model name and value as DLA core ID for inference
  Parameter<DataMap> dla_core_map_;

  /// @brief Map with key as model name and value as frame delay for model inference
  Parameter<DataMap> temporal_map_;

  /// @brief Map with key as model name and value as an activation state for model inference
  Parameter<DataMap> activation_map_;

  ///  @brief Input tensor names
  Parameter<std::vector<std::string>> in_tensor_names_;

  ///  @brief Output tensor names
  Parameter<std::vector<std::string>> out_tensor_names_;

  /// @brief Optimization profile for models with dynamic input shapes
  Parameter<DataVecMap> trt_opt_profile_;

  ///  @brief Memory allocator
  Parameter<std::shared_ptr<Allocator>> allocator_;

  ///  @brief Flag to enable inference on CPU (only supported by ONNX Runtime and LibTorch).
  /// Default is False.
  Parameter<bool> infer_on_cpu_;

  ///  @brief Flag to enable parallel inference. Default is True.
  Parameter<bool> parallel_inference_;

  ///  @brief Flag showing if input buffers are on CUDA. Default is True.
  Parameter<bool> input_on_cuda_;

  ///  @brief Flag showing if output buffers are on CUDA. Default is True.
  Parameter<bool> output_on_cuda_;

  ///  @brief Flag showing if data transmission is on CUDA. Default is True.
  Parameter<bool> transmit_on_cuda_;

  ///  @brief Flag showing if input dimensions are dynamic. Default is False.
  Parameter<bool> dynamic_input_dims_;

  ///  @brief Flag showing if trt engine file conversion will use FP16. Default is False.
  Parameter<bool> enable_fp16_;

  ///  @brief Flag showing if using CUDA Graphs. Default is True.
  Parameter<bool> enable_cuda_graphs_;

  /// @brief The DLA core index to execute the engine on, starts at 0. Set to -1 (the default) to
  /// disable DLA.
  Parameter<int32_t> dla_core_;

  /// @brief If DLA is enabled, use the GPU if a layer cannot be executed on DLA. If the fallback is
  /// disabled, engine creation will fail if a layer cannot executed on DLA.
  Parameter<bool> dla_gpu_fallback_;

  ///  @brief Flag to show if input model path mapping is for cached trt engine files. Default is
  ///  False.
  Parameter<bool> is_engine_path_;

  ///  @brief Backend to do inference on. Supported values: "trt", "torch", "onnxrt".
  Parameter<std::string> backend_;

  ///  @brief Backend map. Multiple backends can be combined in the same application.
  ///  Supported values: "trt" or "torch"
  Parameter<DataMap> backend_map_;

  ///  @brief Optional CUDA stream pool for allocation of an internal CUDA stream if none is
  ///  available in the incoming messages.
  Parameter<std::shared_ptr<CudaStreamPool>> cuda_stream_pool_{};

  // Internal state

  /// Pointer to inference context.
  std::unique_ptr<HoloInfer::InferContext> holoscan_infer_context_;

  /// Pointer to inference specifications
  std::shared_ptr<HoloInfer::InferenceSpecs> inference_specs_;

  /// Map holding dimensions per model. Key is model name and value is a vector with
  /// dimensions.
  std::map<std::string, std::vector<int>> dims_per_tensor_;

  /// Operator Identifier, used in reporting.
  const std::string module_{"Inference Operator"};

  /// @brief Parameter to validate incoming tensor dimensions with model input dimensions
  bool validate_tensor_dimensions_ = true;
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_INFERENCE_INFERENCE_HPP */

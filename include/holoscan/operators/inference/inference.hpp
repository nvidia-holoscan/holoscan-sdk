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

#ifndef HOLOSCAN_OPERATORS_INFERENCE_HPP
#define HOLOSCAN_OPERATORS_INFERENCE_HPP

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "holoscan/core/gxf/gxf_operator.hpp"
#include "holoscan/utils/cuda_stream_handler.hpp"

#include <holoinfer.hpp>
#include <holoinfer_buffer.hpp>
#include <holoinfer_utils.hpp>

namespace HoloInfer = holoscan::inference;

namespace holoscan::ops {
/**
 * @brief Inference Operator class to perform single/multi model inference.
 *
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
      for (auto const& val : value) mappings_[key].push_back(val);
    }

    std::map<std::string, std::vector<std::string>> get_map() const { return mappings_; }

    std::map<std::string, std::vector<std::string>> mappings_;
  };

 private:
  ///  @brief Map with key as model name and value as vector of inferred tensor name
  Parameter<DataVecMap> inference_map_;

  ///  @brief Map with key as model name and value as model file path
  Parameter<DataMap> model_path_map_;

  ///  @brief Map with key as model name and value as vector of input tensor names
  Parameter<DataVecMap> pre_processor_map_;

  /// @brief Map with key as model name and value as GPU ID for inference
  Parameter<DataMap> device_map_;

  ///  @brief Input tensor names
  Parameter<std::vector<std::string>> in_tensor_names_;

  ///  @brief Output tensor names
  Parameter<std::vector<std::string>> out_tensor_names_;

  ///  @brief Memory allocator
  Parameter<std::shared_ptr<Allocator>> allocator_;

  ///  @brief Flag to enable inference on CPU (only supported by onnxruntime).
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

  ///  @brief Flag showing if trt engine file conversion will use FP16. Default is False.
  Parameter<bool> enable_fp16_;

  ///  @brief Flag to show if input model path mapping is for cached trt engine files. Default is
  ///  False.
  Parameter<bool> is_engine_path_;

  ///  @brief Backend to do inference on. Supported values: "trt", "torch", "onnxrt".
  Parameter<std::string> backend_;

  ///  @brief Backend map. Multiple backends can be combined in the same application.
  ///  Supported values: "trt" or "torch"
  Parameter<DataMap> backend_map_;

  ///  @brief Vector of input receivers. Multiple receivers supported.
  Parameter<std::vector<IOSpec*>> receivers_;

  ///  @brief Output transmitter. Single transmitter supported.
  Parameter<std::vector<IOSpec*>> transmitter_;

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

  CudaStreamHandler cuda_stream_handler_;
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_INFERENCE_HPP */

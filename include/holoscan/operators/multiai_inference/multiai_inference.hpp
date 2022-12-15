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

#ifndef HOLOSCAN_OPERATORS_MULTIAI_INFERENCE_HPP
#define HOLOSCAN_OPERATORS_MULTIAI_INFERENCE_HPP

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "../../core/gxf/gxf_operator.hpp"

namespace holoscan::ops {
/**
 * @brief Multi AI Inference Operator class to perform multi model inference.
 *
 * Class wraps a GXF Codelet(`nvidia::holoscan::multiai::MultiAIInference`).
 */
class MultiAIInferenceOp : public holoscan::ops::GXFOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(MultiAIInferenceOp, holoscan::ops::GXFOperator)

  MultiAIInferenceOp() = default;

  const char* gxf_typename() const override {
    return "nvidia::holoscan::multiai::MultiAIInference";
  }

  void setup(OperatorSpec& spec) override;
  void initialize() override;

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
  ///  @brief Map with key as model name and value as inferred tensor name
  Parameter<DataMap> inference_map_;

  ///  @brief Map with key as model name and value as model file path
  Parameter<DataMap> model_path_map_;

  ///  @brief Map with key as model name and value as vector of input tensor names
  Parameter<DataVecMap> pre_processor_map_;

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

  ///  @brief Backend to do inference on. Supported values: "trt", "onnxrt".
  Parameter<std::string> backend_;

  ///  @brief Vector of input receivers. Multiple receivers supported.
  Parameter<std::vector<IOSpec*>> receivers_;

  ///  @brief Output transmitter. Single transmitter supported.
  Parameter<std::vector<IOSpec*>> transmitter_;
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_MULTIAI_INFERENCE_HPP */

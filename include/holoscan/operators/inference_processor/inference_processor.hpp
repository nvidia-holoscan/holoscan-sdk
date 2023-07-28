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

#ifndef HOLOSCAN_OPERATORS_HOLOINFER_PROCESSOR_INFERENCE_PROCESSOR_HPP
#define HOLOSCAN_OPERATORS_HOLOINFER_PROCESSOR_INFERENCE_PROCESSOR_HPP

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "holoscan/core/gxf/gxf_operator.hpp"
#include "holoscan/utils/cuda_stream_handler.hpp"

#include "holoinfer.hpp"
#include "holoinfer_buffer.hpp"
#include "holoinfer_utils.hpp"

namespace HoloInfer = holoscan::inference;

namespace holoscan::ops {
/**
 * @brief Processor Operator class to perform operations per input tensor.
 *
 */
class InferenceProcessorOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(InferenceProcessorOp)

  InferenceProcessorOp() = default;

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void start() override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

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
      for (const auto& val : value) mappings_[key].push_back(val);
    }

    std::map<std::string, std::vector<std::string>> get_map() const { return mappings_; }

    std::map<std::string, std::vector<std::string>> mappings_;
  };

 private:
  ///  @brief Map with key as tensor name and value as vector of supported operations.
  ///  Supported operations: "max_per_channel_scaled"
  Parameter<DataVecMap> process_operations_;

  ///  @brief Map with key as input tensor name and value as vector of processed tensor name
  Parameter<DataVecMap> processed_map_;

  ///  @brief Path to configuration file
  Parameter<std::string> config_path_;

  ///  @brief Vector of input tensor names
  Parameter<std::vector<std::string>> in_tensor_names_;

  ///  @brief Vector of output tensor names
  Parameter<std::vector<std::string>> out_tensor_names_;

  ///  @brief Memory allocator
  Parameter<std::shared_ptr<Allocator>> allocator_;

  ///  @brief Flag showing if input buffers are on CUDA. Default is False.
  ///  Supported value: False
  Parameter<bool> input_on_cuda_;

  ///  @brief Flag showing if output buffers are on CUDA. Default is False.
  ///  Supported value: False
  Parameter<bool> output_on_cuda_;

  ///  @brief Flag showing if data transmission on CUDA. Default is False.
  ///  Supported value: False
  Parameter<bool> transmit_on_cuda_;

  ///  @brief Vector of input receivers. Multiple receivers supported.
  Parameter<std::vector<IOSpec*>> receivers_;

  ///  @brief Output transmitter. Single transmitter supported.
  Parameter<std::vector<IOSpec*>> transmitter_;

  void conditional_disable_output_port(const std::string& name);

  // Internal state

  /// Pointer to Data Processor context.
  std::unique_ptr<HoloInfer::ProcessorContext> holoscan_postprocess_context_;

  /// Map holding data per input tensor.
  HoloInfer::DataMap data_per_tensor_;

  /// Map holding dimensions per model. Key is model name and value is a vector with
  /// dimensions.
  std::map<std::string, std::vector<int>> dims_per_tensor_;

  /// Operator Identifier, used in reporting.
  const std::string module_{"Inference Processor Operator"};

  CudaStreamHandler cuda_stream_handler_;
};

}  // namespace holoscan::ops
#endif /* HOLOSCAN_OPERATORS_HOLOINFER_PROCESSOR_INFERENCE_PROCESSOR_HPP */

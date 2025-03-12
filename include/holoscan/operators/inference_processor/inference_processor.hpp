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

#ifndef HOLOSCAN_OPERATORS_INFERENCE_PROCESSOR_INFERENCE_PROCESSOR_HPP
#define HOLOSCAN_OPERATORS_INFERENCE_PROCESSOR_INFERENCE_PROCESSOR_HPP

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "holoscan/core/io_context.hpp"
#include "holoscan/core/io_spec.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"

#include "holoinfer.hpp"
#include "holoinfer_buffer.hpp"
#include "holoinfer_utils.hpp"

namespace HoloInfer = holoscan::inference;

namespace holoscan::ops {
/**
 * @brief Inference Processor Operator class to perform operations per input tensor.
 *
 * ==Named Inputs==
 *
 * - **receivers** : multi-receiver accepting `nvidia::gxf::Tensor`(s)
 *   - Any number of upstream ports may be connected to this `receivers` port. The operator
 *     will search across all messages for tensors matching those specified in
 *     `in_tensor_names`. These are the set of input tensors used by the processing operations
 *     specified in `process_map`.
 *
 * ==Named Outputs==
 *
 * - **transmitter** : `nvidia::gxf::Tensor`(s)
 *   - A message containing tensors corresponding to the processed results from operations
 *     will be emitted. The names of the tensors transmitted correspond to those in
 *     `out_tensor_names`.
 *
 * ==Parameters==
 *
 * - **allocator**: Memory allocator to use for the output.
 * - **process_operations**: Operations (`DataVecMap`) in sequence on tensors.
 * - **processed_map**: Input-output tensor mapping (`DataVecMap`)
 * - **in_tensor_names**: Names of input tensors (`std::vector<std::string>`) in the order to be fed
 *   into the operator. Optional.
 * - **out_tensor_names**: Names of output tensors (`std::vector<std::string>`) in the order to be
 *   fed into the operator. Optional.
 * - **input_on_cuda**: Whether the input buffer is on the GPU. Optional (default: `false`).
 * - **output_on_cuda**: Whether the output buffer is on the GPU. Optional (default: `false`).
 * - **transmit_on_cuda**: Whether to transmit the message on the GPU. Optional (default: `false`).
 * - **cuda_stream_pool**: `holoscan::CudaStreamPool` instance to allocate CUDA streams.
 *   Optional (default: `nullptr`).
 * - **config_path**: File path to the config file. Optional (default: `""`).
 * - **disable_transmitter**: If `true`, disable the transmitter output port of the operator.
 *   Optional (default: `false`).
 *
 * ==Device Memory Requirements==
 *
 * When using this operator with a `BlockMemoryPool`, `num_blocks` must be greater than or equal to
 * the number of output tensors that will be produced. The `block_size` in bytes must be greater
 * than or equal to the largest output tensor (in bytes). If `output_on_cuda` is true, the blocks
 * should be in device memory (`storage_type`=1), otherwise they should be CUDA pinned host memory
 * (`storage_type`=0).
 */
class InferenceProcessorOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(InferenceProcessorOp)

  InferenceProcessorOp() = default;

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void start() override;
  void stop() override;
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

  ///  @brief Custom cuda kernel
  Parameter<DataMap> custom_kernels_;

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

  ///  @brief Optional CUDA stream pool for allocation of an internal CUDA stream if none is
  ///  available in the incoming messages.
  Parameter<std::shared_ptr<CudaStreamPool>> cuda_stream_pool_{};

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
};

}  // namespace holoscan::ops
#endif /* HOLOSCAN_OPERATORS_INFERENCE_PROCESSOR_INFERENCE_PROCESSOR_HPP */

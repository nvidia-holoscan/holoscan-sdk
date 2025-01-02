/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_OPERATORS_PING_TENSOR_TX_PING_TENSOR_TX_HPP
#define HOLOSCAN_OPERATORS_PING_TENSOR_TX_PING_TENSOR_TX_HPP

#include <memory>
#include <string>

#include <holoscan/holoscan.hpp>

namespace holoscan::ops {

/**
 * @brief Example Tensor transmitter operator.
 *
 * On each tick, it transmits a single tensor on the "out" port.
 *
 * This operator is intended for use in test cases and example applications.
 *
 * ==Named Outputs==
 *
 * - **out** : nvidia::gxf::Tensor
 *   - A generated 1D (H), 2D (HW), 3D (HWC) or 4D (NHWC) tensor (with uninitialized data values).
 *     Depending on the parameters set, this tensor can be in system memory, pinned host memory or
 *     device memory. Setting `batch_size`, `columns` or `channels` to 0 will omit the
 *     corresponding dimension. Notation used: N = batch, H = rows, W = columns, C = channels.
 *
 * ==Parameters==
 *
 * - **allocator**: The memory allocator to use. When not set, a default UnboundedAllocator is used.
 * - **storage_type**: A string indicating where the memory should be allocated. Options are
 *   "system" (system/CPU memory), "host" (CUDA pinned host memory) or "device" (GPU memory). The
 *   `allocator` takes care of allocating memory of the indicated type. The default is "system".
 * - **batch_size**: Size of the batch dimension of the generated tensor. If set to 0, this
 *    dimension is omitted. The default is 0.
 * - **rows**: The number of rows in the generated tensor. This dimension must be >= 1. The default
 *   is 32.
 * - **columns**: The number of columns in the generated tensor. If set to 0, this dimension is
 *    omitted. The default is 64.
 * - **channels**: The number of channels in the generated tensor. If set to 0, this dimension is
 *    omitted. The default is 0.
 * - **data_type_**: A string representing the data type for the generated tensor. Must be one of
 *   "int8_t", "int16_t", "int32_t", "int64_t", "uint8_t", "uint16_t", "uint32_t", "uint64_t",
 *   "float", "double", "complex<float", or "complex<double>". The default is "uint8_t".
 * - **tensor_name**: The name of the generated tensor. The default name is "tensor".
 *
 * ==Device Memory Requirements==
 *
 * When using this operator with a `BlockMemoryPool`, the minimum `block_size` is
 * `(batch_size * rows * columns * channels * element_size_bytes)` where `element_size_bytes` is
 * is the number of bytes for a single element of the specified `data_type`. Only a single memory
 * block is used.
 */
class PingTensorTxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingTensorTxOp)

  PingTensorTxOp() = default;

  void initialize() override;
  void setup(OperatorSpec& spec) override;
  void compute([[maybe_unused]] InputContext& op_input, OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override;

  nvidia::gxf::PrimitiveType element_type() {
    if (element_type_.has_value()) { return element_type_.value(); }
    element_type_ = primitive_type(data_type_.get());
    return element_type_.value();
  }

 private:
  nvidia::gxf::PrimitiveType primitive_type(const std::string& data_type);
  std::optional<nvidia::gxf::PrimitiveType> element_type_;
  size_t count_ = 1;

  Parameter<std::shared_ptr<Allocator>> allocator_{nullptr};
  Parameter<std::string> storage_type_{"system"};
  Parameter<int32_t> batch_size_{0};
  Parameter<int32_t> rows_{32};
  Parameter<int32_t> columns_{64};
  Parameter<int32_t> channels_{0};
  Parameter<std::string> data_type_{"uint8_t"};
  Parameter<std::string> tensor_name_{"tensor"};
  Parameter<std::shared_ptr<CudaStreamPool>> cuda_stream_pool_{};
  Parameter<bool> async_device_allocation_{false};
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_PING_TENSOR_TX_PING_TENSOR_TX_HPP */

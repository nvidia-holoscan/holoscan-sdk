/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <matx.h>

#include <memory>

#include <holoscan/holoscan.hpp>

/**
 * @brief A simple transmitter operator that creates a MatX tensor and sends it downstream.
 *
 * This operator demonstrates how to create a MatX tensor, populate it with data,
 * and convert it to a `holoscan::Tensor` for use within the Holoscan framework.
 * The conversion leverages the DLPack standard for zero-copy data sharing.
 */
class MatXTensorTxOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(MatXTensorTxOp)

  MatXTensorTxOp() = default;

  void setup(holoscan::OperatorSpec& spec) override { spec.output<holoscan::TensorMap>("out"); }

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    // Create a MatX tensor on the GPU
    auto matx_tensor = matx::make_tensor<float>({10});
    matx_tensor.SetVals({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

    // Convert the MatX tensor to a holoscan::Tensor using DLPack.
    // This is a zero-copy operation, meaning both tensors share the same underlying memory.
    auto holoscan_tensor = std::make_shared<holoscan::Tensor>(matx_tensor.ToDlPack());

    // Create a holoscan::TensorMap to hold the tensor
    holoscan::TensorMap out_message;
    out_message.insert({"tensor", holoscan_tensor});

    // Emit the holoscan::TensorMap
    op_output.emit(out_message, "out");
  }
};

/**
 * @brief A simple receiver operator that takes a `holoscan::Tensor` and wraps it as a MatX tensor.
 *
 * This operator demonstrates how to receive a `holoscan::Tensor`, access its data,
 * and perform computations on it using the MatX library. The `holoscan::Tensor`'s data
 * is wrapped in a MatX tensor view without any data copying.
 */
class MatXTensorRxOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(MatXTensorRxOp)

  MatXTensorRxOp() = default;

  void setup(holoscan::OperatorSpec& spec) override { spec.input<holoscan::TensorMap>("in"); }

  void compute(holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    // Receive the holoscan::TensorMap
    auto maybe_tensor_map = op_input.receive<holoscan::TensorMap>("in");

    if (maybe_tensor_map) {
      auto& tensor_map = maybe_tensor_map.value();
      for (const auto& [name, tensor] : tensor_map) {
        HOLOSCAN_LOG_INFO("tensor name: {}", name);
        HOLOSCAN_LOG_INFO("tensor nbytes: {}", tensor->nbytes());

        // Create a MatX tensor view from the holoscan::Tensor's data pointer and shape.
        // This is a zero-copy operation.
        auto matx_tensor =
            matx::make_tensor<float>(static_cast<float*>(tensor->data()), {tensor->shape()[0]});
        HOLOSCAN_LOG_INFO("MatX tensor: ");
        matx::print(matx_tensor);

        // Perform a simple calculation on the GPU: matx_tensor = matx_tensor * 2.0 + 1
        (matx_tensor = matx_tensor * 2.f + matx::ones()).run();

        HOLOSCAN_LOG_INFO("Result of 'matx_tensor * 2 + 1':");
        matx::print(matx_tensor);
      }
    }
  }
};

/**
 * @brief The main application class for the MatX basic example.
 *
 * This application connects a MatX tensor transmitter and receiver
 * to demonstrate basic interoperability.
 */
class MatxBasicApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    // Define the operators
    auto tx = make_operator<MatXTensorTxOp>("tx", make_condition<CountCondition>(1));
    auto rx = make_operator<MatXTensorRxOp>("rx");

    // Define the workflow
    add_flow(tx, rx);
  }
};

int main() {
  auto app = holoscan::make_application<MatxBasicApp>();
  app->run();

  return 0;
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <stdexcept>
#include <string>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/utils/matx_allocator.hpp>

/**
 * @brief A transmitter operator that creates a MatX tensor backed by a Holoscan memory pool.
 *
 * This operator demonstrates how to use `holoscan::MatXAllocator` to create MatX tensors
 * that are backed by a Holoscan allocator (e.g., RMMAllocator), rather than MatX's default
 * CUDA allocator. The tensor is then converted to a `holoscan::Tensor` for downstream use.
 */
class MatXAllocTxOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(MatXAllocTxOp)

  MatXAllocTxOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output<holoscan::TensorMap>("out");
    spec.param(allocator_, "allocator", "Allocator", "Memory allocator for MatX tensors.");
  }

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    // Create a MatXAllocator wrapper around the Holoscan allocator.
    // This enables MatX tensors to be backed by the Holoscan memory pool.
    //
    // The shared_ptr overload accepts allocator_.get() (which returns
    // std::shared_ptr<Allocator>&) directly — no double .get() needed.
    // For stream-ordered allocation, pass a cudaStream_t as the second argument:
    //   holoscan::MatXAllocator matx_alloc(allocator_.get(), cuda_stream);
    holoscan::MatXAllocator matx_alloc(allocator_.get());

    // Create a MatX tensor using the pooled allocator.
    auto matx_tensor = matx::make_tensor<float>({10}, matx_alloc);

    // Populate tensor data via cudaMemcpy (device memory).
    std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto cuda_err = cudaMemcpy(
        matx_tensor.Data(), data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) {
      HOLOSCAN_LOG_ERROR("cudaMemcpy failed: {}", cudaGetErrorString(cuda_err));
      throw std::runtime_error("Failed to copy data to device");
    }

    HOLOSCAN_LOG_INFO("Created MatX tensor with {} elements using pooled allocator",
                      matx_tensor.Size(0));

    // Convert the MatX tensor to a holoscan::Tensor using DLPack (zero-copy).
    auto holoscan_tensor = std::make_shared<holoscan::Tensor>(matx_tensor.ToDlPack());

    // Emit as TensorMap
    holoscan::TensorMap out_message;
    out_message.insert({"tensor", holoscan_tensor});
    op_output.emit(out_message, "out");
  }

 private:
  holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;
};

/**
 * @brief A receiver operator that receives a tensor and performs MatX operations.
 *
 * This operator receives a `holoscan::Tensor` (originally created with pooled allocation),
 * wraps it as a MatX tensor view, and performs GPU-accelerated computation.
 */
class MatXAllocRxOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(MatXAllocRxOp)

  MatXAllocRxOp() = default;

  void setup(holoscan::OperatorSpec& spec) override { spec.input<holoscan::TensorMap>("in"); }

  void compute(holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    auto maybe_tensor_map = op_input.receive<holoscan::TensorMap>("in");

    if (maybe_tensor_map) {
      auto& tensor_map = maybe_tensor_map.value();
      for (const auto& [name, tensor] : tensor_map) {
        HOLOSCAN_LOG_INFO("Received tensor '{}': {} bytes", name, tensor->nbytes());

        // Method 1: Raw pointer — direct data/shape access (no DLPack validation).
        auto matx_tensor_raw =
            matx::make_tensor<float>(static_cast<float*>(tensor->data()), {tensor->shape()[0]});

        HOLOSCAN_LOG_INFO("Input tensor (raw pointer):");
        matx::print(matx_tensor_raw);

        // Method 2: DLPack — MatX native make_tensor overload for type-safe import.
        // to_dlpack() returns DLManagedTensor*; caller owns and must invoke deleter.
        // Pass *dl by value; MatX validates dtype/rank and builds non-owning view.
        matx::tensor_t<float, 1> matx_tensor;
        {
          DLManagedTensor* dl = tensor->to_dlpack();
          auto dl_guard = std::unique_ptr<DLManagedTensor, void (*)(DLManagedTensor*)>(
              dl, [](DLManagedTensor* p) {
                if (p && p->deleter) p->deleter(p);
              });
          matx::make_tensor(matx_tensor, *dl);
        }
        // matx_tensor is valid; view references memory owned by holoscan::Tensor.

        HOLOSCAN_LOG_INFO("Input tensor (DLPack):");
        matx::print(matx_tensor);

        // Perform GPU computation: tensor = tensor * 2 + 1
        (matx_tensor = matx_tensor * 2.f + matx::ones<float>({matx_tensor.Size(0)})).run();

        HOLOSCAN_LOG_INFO("Result of 'tensor * 2 + 1':");
        matx::print(matx_tensor);
      }
    }
  }
};

/**
 * @brief Application demonstrating MatX tensor allocation with Holoscan memory pools.
 *
 * This application creates a two-operator pipeline:
 * 1. MatXAllocTxOp: Creates a MatX tensor using an RMMAllocator-backed MatXAllocator.
 * 2. MatXAllocRxOp: Receives the tensor and performs MatX GPU computation.
 *
 * The key difference from the matx_basic example is that tensor memory comes from
 * Holoscan's RMMAllocator memory pool instead of MatX's default CUDA allocator.
 */
class MatXAllocatorApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    // Create an RMMAllocator for pooled GPU memory management.
    auto rmm_allocator =
        make_resource<RMMAllocator>("rmm_allocator",
                                    Arg("device_memory_initial_size", std::string("16MB")),
                                    Arg("device_memory_max_size", std::string("32MB")),
                                    Arg("host_memory_initial_size", std::string("16MB")),
                                    Arg("host_memory_max_size", std::string("32MB")));

    auto tx = make_operator<MatXAllocTxOp>(
        "tx", make_condition<CountCondition>(1), Arg("allocator", rmm_allocator));
    auto rx = make_operator<MatXAllocRxOp>("rx");

    add_flow(tx, rx);
  }
};

int main() {
  auto app = holoscan::make_application<MatXAllocatorApp>();
  app->run();

  return 0;
}

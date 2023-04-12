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

#include <cuda_runtime.h>

#include <holoscan/holoscan.hpp>

#include <gxf/std/tensor.hpp>
#include <holoscan/core/gxf/gxf_extension_registrar.hpp>
#include <holoscan/core/gxf/gxf_tensor.hpp>

#include "./receive_tensor_gxf.hpp"
#include "./send_tensor_gxf.hpp"

#ifdef CUDA_TRY
#undef CUDA_TRY
#define CUDA_TRY(stmt)                                                                  \
  {                                                                                     \
    cuda_status = stmt;                                                                 \
    if (cudaSuccess != cuda_status) {                                                   \
      HOLOSCAN_LOG_ERROR("Runtime call {} in line {} of file {} failed with '{}' ({})", \
                         #stmt,                                                         \
                         __LINE__,                                                      \
                         __FILE__,                                                      \
                         cudaGetErrorString(cuda_status),                               \
                         cuda_status);                                                  \
    }                                                                                   \
  }
#endif

namespace holoscan::ops {

// This operator is a wrapper around the GXF operator that sends a tensor.
// (`nvidia::gxf::test::SendTensor` class in send_tensor_gxf.hpp)
class GXFSendTensorOp : public holoscan::ops::GXFOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(GXFSendTensorOp, holoscan::ops::GXFOperator)

  GXFSendTensorOp() = default;

  const char* gxf_typename() const override { return "nvidia::gxf::test::SendTensor"; }

  void setup(OperatorSpec& spec) override {
    auto& ping = spec.output<gxf::Entity>("signal");

    spec.param(signal_, "signal", "Signal", "Signal to send", &ping);
    spec.param(pool_, "pool", "Pool", "Allocator instance for output tensors.");
  }

 private:
  Parameter<holoscan::IOSpec*> signal_;
  Parameter<std::shared_ptr<Allocator>> pool_;
};

class ProcessTensorOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ProcessTensorOp)

  ProcessTensorOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<gxf::Entity>("in");
    spec.output<gxf::Entity>("out");
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    // The type of `in_message` is 'holoscan::gxf::Entity'.
    auto in_message = op_input.receive<gxf::Entity>("in");
    // The type of `tensor` is 'std::shared_ptr<holoscan::Tensor>'.
    auto tensor = in_message.get<Tensor>();

    // Process with 'tensor' here.
    cudaError_t cuda_status;

    size_t data_size = tensor->nbytes();
    std::vector<uint8_t> in_data(data_size);
    CUDA_TRY(cudaMemcpy(in_data.data(), tensor->data(), data_size, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < data_size; i++) { in_data[i] *= 2; }

    CUDA_TRY(cudaMemcpy(tensor->data(), in_data.data(), data_size, cudaMemcpyHostToDevice));

    // Create a new message (Entity)
    auto out_message = gxf::Entity::New(&context);
    out_message.add(tensor, "tensor");

    // Send the processed message.
    op_output.emit(out_message);
  };
};

// This operator is a wrapper around the GXF operator that receives a tensor.
// (`nvidia::gxf::test::ReceiveTensor` class in receive_tensor_gxf.hpp)
class GXFReceiveTensorOp : public holoscan::ops::GXFOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(GXFReceiveTensorOp, holoscan::ops::GXFOperator)

  GXFReceiveTensorOp() = default;

  const char* gxf_typename() const override { return "nvidia::gxf::test::ReceiveTensor"; }

  void setup(OperatorSpec& spec) override {
    auto& ping = spec.input<gxf::Entity>("signal");

    spec.param(signal_, "signal", "Signal", "Signal to receive", &ping);
  }

 private:
  Parameter<holoscan::IOSpec*> signal_;
};

}  // namespace holoscan::ops

class App : public holoscan::Application {
 public:
  void register_gxf_codelets() {
    gxf_context_t context = executor().context();

    holoscan::gxf::GXFExtensionRegistrar extension_factory(
        context, "TensorSenderReceiver", "Extension for sending and receiving tensors");

    extension_factory.add_component<nvidia::gxf::test::SendTensor, nvidia::gxf::Codelet>(
        "SendTensor class");
    extension_factory.add_component<nvidia::gxf::test::ReceiveTensor, nvidia::gxf::Codelet>(
        "ReceiveTensor class");

    if (!extension_factory.register_extension()) {
      HOLOSCAN_LOG_ERROR("Failed to register GXF Codelets");
      return;
    }
  }

  void compose() override {
    using namespace holoscan;

    register_gxf_codelets();

    auto tx = make_operator<ops::GXFSendTensorOp>(
        "tx",
        make_condition<CountCondition>(15),
        Arg("pool") = make_resource<UnboundedAllocator>("pool"));

    auto mx = make_operator<ops::ProcessTensorOp>("mx");

    auto rx = make_operator<ops::GXFReceiveTensorOp>("rx");

    add_flow(tx, mx);
    add_flow(mx, rx);
  }
};

int main(int argc, char** argv) {
  holoscan::load_env_log_level();

  auto app = holoscan::make_application<App>();
  app->run();

  return 0;
}

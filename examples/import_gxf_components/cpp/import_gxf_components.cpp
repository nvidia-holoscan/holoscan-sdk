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

#include <cuda_runtime.h>
#include <any>
#include <memory>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gxf/std/tensor.hpp>
#include <holoscan/core/domain/tensor_map.hpp>
#include <holoscan/core/gxf/gxf_extension_registrar.hpp>
#include <holoscan/holoscan.hpp>

#include "./receive_tensor_gxf.hpp"
#include "./send_tensor_gxf.hpp"
// Include the following header files to use GXFCodeletOp and HOLOSCAN_WRAP_GXF_CODELET_AS_OPERATOR
// macro.
#include "holoscan/operators/gxf_codelet/gxf_codelet.hpp"

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
                         static_cast<int>(cuda_status));                                \
    }                                                                                   \
  }
#endif

// Define an operator that wraps the GXF Codelet that sends a tensor
// (`nvidia::gxf::test::SendTensor` class in send_tensor_gxf.hpp)
HOLOSCAN_WRAP_GXF_CODELET_AS_OPERATOR(GXFSendTensorOp, "nvidia::gxf::test::SendTensor")

// Define an operator that wraps the GXF Codelet that receives a tensor, extends the GXFCodeletOp
// (`nvidia::gxf::test::ReceiveTensor` class in receive_tensor_gxf.hpp).
// If there is no need for custom setup or initialize code, the macro
// `HOLOSCAN_WRAP_GXF_CODELET_AS_OPERATOR` can be used (as shown above) to simplify this process.
class GXFReceiveTensorOp : public ::holoscan::ops::GXFCodeletOp {
 public:
  HOLOSCAN_OPERATOR_FORWARD_TEMPLATE()
  explicit GXFReceiveTensorOp(ArgT&& arg, ArgsT&&... args)
      : ::holoscan::ops::GXFCodeletOp("nvidia::gxf::test::ReceiveTensor", std::forward<ArgT>(arg),
                                      std::forward<ArgsT>(args)...) {}
  GXFReceiveTensorOp() : ::holoscan::ops::GXFCodeletOp("nvidia::gxf::test::ReceiveTensor") {}

  void setup(holoscan::OperatorSpec& spec) override {
    using namespace holoscan;
    // Ensure the parent class setup() is called before any additional setup code.
    ops::GXFCodeletOp::setup(spec);

    // You can add any additional setup code here (if needed).
    // You can update conditions of the input/output ports, update the connector types, etc.
    //
    // Example:
    // - `spec.inputs()["signal"]->condition(ConditionType::kNone);`
    //   to update the condition of the input port to 'kNone'.
    //   (assuming that the GXF Codelet has a Receiver component named 'signal'.)
  }

  void initialize() override {
    // You can call any additional initialization code here (if needed).
    //
    // Example:
    // - `register_converter<T>();` to register a converter for a specific type
    // - `register_codec<T>("codec_name", bool_overwrite);` to register a codec for a specific type
    // - `add_arg(holoscan::Arg("arg_name", arg_value));`
    //   or `add_arg(holoscan::Arg("arg_name") = arg_value);` to add an argument to the GXF Operator

    // ...

    // The parent class initialize() call should occur after the argument additions specified above.
    holoscan::ops::GXFCodeletOp::initialize();
  }
};

// Define a resource that wraps the GXF component `nvidia::gxf::BlockMemoryPool`
// (`nvidia::gxf::BlockMemoryPool` class in gxf/std/block_memory_pool.hpp)
// The following class definition can be shortened using
// the `HOLOSCAN_WRAP_GXF_COMPONENT_AS_RESOURCE` macro:
//
//   HOLOSCAN_WRAP_GXF_COMPONENT_AS_RESOURCE(MyBlockMemoryPool, "nvidia::gxf::BlockMemoryPool")
//
// Note that this is illustrated here using `BlockMemoryPool` as a concrete example, but in practice
// applications would just import the existing `holoscan::BlockMemoryPool` resource.
// `GXFComponentResource` would be used to wrap some GXF component not already available via
// the resources in the `holoscan` namespace.
class MyBlockMemoryPool : public ::holoscan::GXFComponentResource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_TEMPLATE()
  explicit MyBlockMemoryPool(ArgT&& arg, ArgsT&&... args)
      : ::holoscan::GXFComponentResource("nvidia::gxf::BlockMemoryPool", std::forward<ArgT>(arg),
                                         std::forward<ArgsT>(args)...) {}
  MyBlockMemoryPool() = default;
};

class ProcessTensorOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ProcessTensorOp)

  ProcessTensorOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("in");
    spec.output<holoscan::TensorMap>("out");
  }

  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override {
    // The type of `in_message` is 'holoscan::TensorMap'.
    auto in_message = op_input.receive<holoscan::TensorMap>("in").value();
    // The type of out_message is TensorMap.
    holoscan::TensorMap out_message;

    for (auto& [key, tensor] : in_message) {  // Process with 'tensor' here.
      cudaError_t cuda_status;
      size_t data_size = tensor->nbytes();
      std::vector<uint8_t> in_data(data_size);
      CUDA_TRY(cudaMemcpy(in_data.data(), tensor->data(), data_size, cudaMemcpyDeviceToHost));
      HOLOSCAN_LOG_INFO("ProcessTensorOp Before key: '{}', shape: ({}), data: [{}]",
                        key,
                        fmt::join(tensor->shape(), ","),
                        fmt::join(in_data, ","));
      for (size_t i = 0; i < data_size; i++) { in_data[i] *= 2; }
      HOLOSCAN_LOG_INFO("ProcessTensorOp After key: '{}', shape: ({}), data: [{}]",
                        key,
                        fmt::join(tensor->shape(), ","),
                        fmt::join(in_data, ","));
      CUDA_TRY(cudaMemcpy(tensor->data(), in_data.data(), data_size, cudaMemcpyHostToDevice));
      out_message.insert({key, tensor});
    }
    // Send the processed message.
    op_output.emit(out_message);
  };
};

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

    auto tx = make_operator<GXFSendTensorOp>("tx",
                                             make_condition<CountCondition>(15),
                                             Arg("pool") = make_resource<MyBlockMemoryPool>(
                                                 "pool",
                                                 Arg("storage_type") = static_cast<int32_t>(1),
                                                 Arg("block_size") = 1024UL,
                                                 Arg("num_blocks") = 2UL));

    // Alternatively, you can use the following code to create the GXFSendTensorOp operator:
    //
    // auto tx = make_operator<ops::GXFCodeletOp>("tx",
    //                                            "nvidia::gxf::test::SendTensor",
    //                                            make_condition<CountCondition>(15),
    //                                            Arg("pool") = make_resource<GXFComponentResource>(
    //                                                "pool",
    //                                                "nvidia::gxf::BlockMemoryPool",
    //                                                Arg("storage_type") = static_cast<int32_t>(1),
    //                                                Arg("block_size") = 1024UL,
    //                                                Arg("num_blocks") = 2UL));

    auto mx = make_operator<ProcessTensorOp>("mx");

    auto rx = make_operator<GXFReceiveTensorOp>("rx");

    add_flow(tx, mx);
    add_flow(mx, rx);
  }
};

int main(int argc, char** argv) {
  auto app = holoscan::make_application<App>();
  app->run();

  return 0;
}

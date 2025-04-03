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

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "distributed_app_fixture.hpp"

#include <gxf/std/tensor.hpp>
#include <holoscan/holoscan.hpp>

namespace holoscan {

namespace {

// Operators
class TensorSource : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TensorSource)
  TensorSource() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output<holoscan::TensorMap>("out_tensor");
  }

  void initialize() override {
    holoscan::Operator::initialize();

    // Initialize counter
    counter = 0;

    // create host tensor
    host_array_buffer_.reserve(4 * 128 * 512);
    for (auto i = 0; i < 4 * 128 * 512; i++) { host_array_buffer_[i] = 0.0; }

    auto tensor_shape = nvidia::gxf::Shape{{4, 128, 512}};
    auto dtype = nvidia::gxf::PrimitiveType::kFloat32;
    const uint64_t bytes_per_element = nvidia::gxf::PrimitiveTypeSize(dtype);
    nvidia::gxf::MemoryStorageType storage_type = nvidia::gxf::MemoryStorageType::kSystem;
    auto strides = nvidia::gxf::ComputeTrivialStrides(tensor_shape, bytes_per_element);

    // wrap this memory as the GXF tensor
    gxf_tensor_->wrapMemory(tensor_shape,
                            dtype,
                            bytes_per_element,
                            strides,
                            storage_type,
                            host_array_buffer_.data(),
                            [](void*) mutable { return nvidia::gxf::Success; });
  }

  void compute(holoscan::InputContext&, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext&) override {
    HOLOSCAN_LOG_INFO("Transmitting value: {}", counter++);

    if (use_tensormap_) {
      // emit TensorMap containing the tensor
      holoscan::TensorMap out_tensormap;
      auto maybe_dl_ctx = gxf_tensor_->toDLManagedTensorContext();
      if (!maybe_dl_ctx) {
        throw std::runtime_error(
            "failed to get std::shared_ptr<DLManagedTensorContext> from nvidia::gxf::Tensor");
      }
      auto out_tensor = std::make_shared<holoscan::Tensor>(maybe_dl_ctx.value());
      out_tensormap.insert({"tensor", out_tensor});
      op_output.emit(out_tensormap, "out_tensor");
    } else {
      // emit std::shared_ptr<holoscan::Tensor>
      auto maybe_dl_ctx = gxf_tensor_->toDLManagedTensorContext();
      if (!maybe_dl_ctx) {
        throw std::runtime_error(
            "failed to get std::shared_ptr<DLManagedTensorContext> from nvidia::gxf::Tensor");
      }
      auto out_tensor = std::make_shared<holoscan::Tensor>(maybe_dl_ctx.value());
      // out_message.insert({"tensor", out_tensor});
      // op_output.emit(out_message, "out_tensor");
      op_output.emit(out_tensor, "out_tensor");
    }
  }

 private:
  bool use_tensormap_ = false;
  std::vector<float> host_array_buffer_;
  std::shared_ptr<nvidia::gxf::Tensor> gxf_tensor_ = std::make_shared<nvidia::gxf::Tensor>();
  uint64_t counter;
};

class TensorSink : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TensorSink)
  TensorSink() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("in_tensor");
  }

  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext&) override {
    if (use_tensormap_) {
      auto in_message = op_input.receive<holoscan::TensorMap>("in_tensor");
      if (!in_message) { throw std::runtime_error("Failed to receive TensorMap"); }
      auto hs_tensor = in_message.value()["tensor"];
    } else {
      auto in_tensor = op_input.receive<std::shared_ptr<holoscan::Tensor>>("in_tensor");
      if (!in_tensor) {
        throw std::runtime_error("Failed to receive std::shared_ptr<holoscan::Tensor>");
      }
      auto hs_tensor = in_tensor.value();
    }
    HOLOSCAN_LOG_INFO("Received tensor");
  }

 private:
  bool use_tensormap_ = false;
};

// Fragments
class SourceFragment : public holoscan::Fragment {
 public:
  void compose() override {
    // Create a PeriodicCondition with a 100ms interval
    auto periodic_condition = make_condition<PeriodicCondition>(
        "periodic-condition", Arg("recess_period") = std::string("10ms"));

    // CountCondition with max number of iterations
    auto count_condition = make_condition<CountCondition>("count-condition", 5);

    // Instantiate operators
    auto source = make_operator<TensorSource>("source", periodic_condition, count_condition);
    add_operator(source);
  }
};

class SinkFragment : public holoscan::Fragment {
 public:
  void compose() override {
    // Instantiate operators
    auto sink = make_operator<TensorSink>("sink");
    add_operator(sink);
  }
};

// Application
class TensorTransmitApp : public holoscan::Application {
 public:
  void compose() override {
    // Instantiate operators
    auto source = make_fragment<SourceFragment>("source");
    auto sink = make_fragment<SinkFragment>("sink");

    // Connect the fragments
    add_flow(source, sink, {{"source.out_tensor", "sink.in_tensor"}});
  }
};

}  // namespace

}  // namespace holoscan

class SingleTensorTransmitDistributedApp : public DistributedApp {};

TEST_F(SingleTensorTransmitDistributedApp, TestTensorTransmitApp) {
  using namespace holoscan;
  auto app = make_application<TensorTransmitApp>();

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  try {
    app->run();
  } catch (const std::exception& e) { HOLOSCAN_LOG_ERROR("Exception: {}", e.what()); }

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("Failed to receive") == std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("Received tensor") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

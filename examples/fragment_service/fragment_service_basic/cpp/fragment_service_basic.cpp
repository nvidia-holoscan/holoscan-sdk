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

#include <memory>

#include <holoscan/holoscan.hpp>

class MyService : public holoscan::DefaultFragmentService {
 public:
  explicit MyService(int value) : value_(value) {}

  int value() const { return value_; }

 private:
  int value_;
};

class MyOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(MyOp)

  MyOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {}

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    HOLOSCAN_LOG_INFO("MyOp::compute() executed");
    auto my_service = service<MyService>();
    HOLOSCAN_LOG_INFO("MyService value: {}", my_service->value());
  }
};

class FragmentServiceApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    auto my_service = std::make_shared<MyService>(10);

    register_service(my_service);

    auto my_op = make_operator<MyOp>("my_op", make_condition<holoscan::CountCondition>(1));
    add_operator(my_op);
  }
};

int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv) {
  auto app = holoscan::make_application<FragmentServiceApp>();
  app->run();

  return 0;
}

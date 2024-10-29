
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cassert>
#include <memory>
#include "holoscan/holoscan.hpp"

namespace holoscan::ops {

class PingTxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingTxOp)

  PingTxOp() = default;

  void setup(OperatorSpec& spec) override { spec.output<int*>("out"); }

  void compute([[maybe_unused]] InputContext& op_input, OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    HOLOSCAN_LOG_INFO("Tx message value: {}", ++index_);
    if (index_ % 2 != 0) {
      int* value = new int{index_};  // NOLINT(*)
      op_output.emit(value, "out");  // emit only odd values
    } else {
      op_output.emit(nullptr, "out");  // emit nullptr for even values
    }
    index_++;
  }  // NOLINT(clang-analyzer-cplusplus.NewDeleteLeaks)

 private:
  int index_ = 0;
};

class PingMxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingMxOp)

  PingMxOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<int*>("in");
    spec.output<int>("out");
    spec.param(multiplier_, "multiplier", "Multiplier", "Multiply the input by this value", 2);
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto* value = op_input.receive<int*>("in").value();

    HOLOSCAN_LOG_INFO("Middle message received (count: {})", count_++);

    if (value != nullptr) {
      HOLOSCAN_LOG_INFO("Middle message value: {}", *value);
      op_output.emit((*value) * multiplier_, "out");
      delete value;  // NOLINT(*)
    }
  };

 private:
  int count_ = 1;
  Parameter<int> multiplier_;
};

class PingRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingRxOp)

  PingRxOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<int>("in").condition(holoscan::ConditionType::kNone);
  }

  void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto received_value = op_input.receive<int>("in");

    HOLOSCAN_LOG_INFO("Rx message received (count: {})", count_++);

    if (received_value) {  // same with 'if (received_value.has_value()) {'
      HOLOSCAN_LOG_INFO("Rx message value: {}", received_value.value());
    } else {
      HOLOSCAN_LOG_INFO("Rx did not receive any value: {}", received_value.error().what());
    }
  };

 private:
  int count_ = 1;
};

}  // namespace holoscan::ops

class MyPingApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    // Define the tx, mx, rx operators, allowing the tx operator to execute 10 times
    auto tx = make_operator<ops::PingTxOp>("tx", make_condition<CountCondition>(10));
    auto mx = make_operator<ops::PingMxOp>("mx", Arg("multiplier", 3));
    auto rx = make_operator<ops::PingRxOp>("rx", make_condition<CountCondition>(20));

    // Define the workflow
    add_flow(tx, mx, {{"out", "in"}});
    add_flow(mx, rx, {{"out", "in"}});
  }
};

int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv) {
  auto app = holoscan::make_application<MyPingApp>();
  app->run();

  return 0;
}

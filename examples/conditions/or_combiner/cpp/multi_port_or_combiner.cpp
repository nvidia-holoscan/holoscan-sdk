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
#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "holoscan/holoscan.hpp"
#include <holoscan/operators/ping_tx/ping_tx.hpp>

namespace holoscan::ops {

class MultiRxOrOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(MultiRxOrOp)

  MultiRxOrOp() = default;

  void setup(OperatorSpec& spec) override {
    // Using size argument to explicitly set the receiver message queue size for each input.
    spec.input<int>("in1");
    spec.input<int>("in2");

    // configure Operator to execute if an input is on "in1" OR "in2"
    // (without this, the default is "in1" AND "in2")
    spec.or_combine_port_conditions({"in1", "in2"});
  }

  void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto in_value1 = op_input.receive<int>("in1");
    if (in_value1) {
      HOLOSCAN_LOG_INFO("count {}, message received on in1", count_);
    }

    auto in_value2 = op_input.receive<int>("in2");
    if (in_value2) {
      HOLOSCAN_LOG_INFO("count {}, message received on in2", count_);
    }
    count_++;
  };

 private:
  size_t count_ = 0;
};

}  // namespace holoscan::ops

class OrCombinerApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    using namespace std::chrono_literals;

    auto tx1 = make_operator<ops::PingTxOp>("tx1",
                                            make_condition<CountCondition>("count", 10),
                                            make_condition<PeriodicCondition>("period", 0.03s));
    auto tx2 = make_operator<ops::PingTxOp>("tx2",
                                            make_condition<CountCondition>("count", 10),
                                            make_condition<PeriodicCondition>("period", 0.08s));

    auto rx_or_combined = make_operator<ops::MultiRxOrOp>("rx_or");

    add_flow(tx1, rx_or_combined, {{"out", "in1"}});
    add_flow(tx2, rx_or_combined, {{"out", "in2"}});
  }
};

int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv) {
  auto app = holoscan::make_application<OrCombinerApp>();
  app->run();
  return 0;
}

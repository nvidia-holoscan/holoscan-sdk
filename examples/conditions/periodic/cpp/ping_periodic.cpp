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
#include <chrono>
#include <memory>
#include <string>

#include "holoscan/holoscan.hpp"

namespace holoscan::ops {

class PingTxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingTxOp)

  PingTxOp() = default;

  void setup(OperatorSpec& spec) override { spec.output<std::shared_ptr<std::string>>("out"); }

  void compute([[maybe_unused]] InputContext& op_input, OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto value = std::make_shared<std::string>("Periodic ping...");
    op_output.emit(value, "out");
  };
};

class PingRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingRxOp)

  PingRxOp() = default;

  void setup(OperatorSpec& spec) override { spec.input<std::shared_ptr<std::string>>("in"); }

  void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto in_value = op_input.receive<std::shared_ptr<std::string>>("in").value();

    HOLOSCAN_LOG_INFO("Rx message received: {}", in_value->c_str());
  };
};

}  // namespace holoscan::ops

class App : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    using namespace std::chrono_literals;

    auto tx =
        make_operator<ops::PingTxOp>("tx",
                                     make_condition<CountCondition>("count-condition", 10),
                                     make_condition<PeriodicCondition>("periodic-condition", 0.2s));
    // Note:: An alternative to using the standard C++ time literals
    // (e.g., `5ns`, `10us`, `1ms`, `0.5s`, `1min`, `0.5h`, etc.)
    // is to use a string format
    // (e.g., `"1000"` (1000ns == 1us), `"1ms"` (1ms), `"2hz"` (0.5s), `"1s"`, etc.)
    // When using the string format, you can specify units using "s" for seconds,
    // "ms" for milliseconds, and "hz" for Hertz.
    // If no unit is specified in the string, the default unit is "ns" (nanoseconds).

    auto rx = make_operator<ops::PingRxOp>("rx");

    add_flow(tx, rx);
  }
};

int main([[maybe_unused]] int argc, char** argv) {
  auto app = holoscan::make_application<App>();

  // Get the configuration
  auto config_path = std::filesystem::canonical(argv[0]).parent_path();
  config_path /= std::filesystem::path("ping_periodic.yaml");
  app->config(config_path);

  app->run();

  return 0;
}

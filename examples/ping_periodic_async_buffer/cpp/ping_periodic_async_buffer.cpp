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
#include <utility>

#include "holoscan/holoscan.hpp"

namespace holoscan::ops {

class PingTxOp : public Operator {
 private:
  int count_ = 0;

 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingTxOp)

  PingTxOp() = default;

  void setup(OperatorSpec& spec) override { spec.output<int>("out"); }

  void compute([[maybe_unused]] InputContext& op_input, OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    count_++;
    op_output.emit(count_, "out");
    HOLOSCAN_LOG_INFO("Tx message sent: {}", count_);
  };
};

class PingRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingRxOp)

  PingRxOp() = default;

  void setup(OperatorSpec& spec) override { spec.input<int>("in"); }

  void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto maybe_value = op_input.receive<int>("in");

    if (!maybe_value) {
      auto error_msg = fmt::format("Operator '{}' did not receive a valid value.", name_);
      HOLOSCAN_LOG_INFO(error_msg);
      return;
    }

    HOLOSCAN_LOG_INFO("Rx message received: {}", maybe_value.value());
  };
};

}  // namespace holoscan::ops

class App : public holoscan::Application {
 private:
  int tx_period_ms_ = 100;
  int rx_period_ms_ = 200;

 public:
  void set_tx_rx_periods(int tx_period_ms, int rx_period_ms) {
    tx_period_ms_ = tx_period_ms;
    rx_period_ms_ = rx_period_ms;
  }

  void compose() override {
    using namespace holoscan;
    using namespace std::chrono_literals;

    auto count_cond_tx = make_condition<CountCondition>("count-condition-tx", 10);
    auto periodic_cond_tx = make_condition<PeriodicCondition>(
        "periodic-condition", tx_period_ms_ * 1ms, PeriodicConditionPolicy::kMinTimeBetweenTicks);

    auto tx = make_operator<ops::PingTxOp>("tx", count_cond_tx, periodic_cond_tx);

    auto count_cond_rx = make_condition<CountCondition>("count-condition-rx", 10);
    auto periodic_cond_rx =
        make_condition<PeriodicCondition>("periodic-condition-rx",
                                          rx_period_ms_ * 1ms,
                                          PeriodicConditionPolicy::kMinTimeBetweenTicks);

    auto rx = make_operator<ops::PingRxOp>("rx", count_cond_rx, periodic_cond_rx);

    add_flow(tx, rx, IOSpec::ConnectorType::kAsyncBuffer);
    // or with port names
    // add_flow(tx, rx, {{"out", "in"}}, IOSpec::ConnectorType::kAsyncBuffer);
  }
};

int main(int argc, char** argv) {
  if (argc == 2 || argc > 4) {
    HOLOSCAN_LOG_ERROR(
        "Usage: {} <tx_period_ms, default 100 ms> <rx_period_ms, default 200 ms> <scheduler "
        "(optional): 1 for event-based "
        "scheduler, anything else for greedy scheduler>",
        argv[0]);
    return -1;
  }

  int tx_period_ms = 0;
  int rx_period_ms = 0;
  bool use_event_based_scheduler = false;

  if (argc >= 3) {
    try {
      tx_period_ms = std::stoi(argv[1]);
      rx_period_ms = std::stoi(argv[2]);
    } catch (const std::invalid_argument& e) {
      HOLOSCAN_LOG_ERROR("Invalid argument: {}", e.what());
      return -1;
    }
    if (tx_period_ms <= 0 || rx_period_ms <= 0) {
      HOLOSCAN_LOG_ERROR(
          "tx_period_ms and rx_period_ms must be greater than 0. Currently provided values are: "
          "tx_period_ms: {}, rx_period_ms: {}",
          tx_period_ms,
          rx_period_ms);
      return -1;
    }
  }

  if (argc == 4) {
    try {
      if (std::stoi(argv[3]) == 1) {
        use_event_based_scheduler = true;
      }
    } catch (const std::invalid_argument& e) {
      HOLOSCAN_LOG_ERROR("Invalid argument: {}", e.what());
      return -1;
    }
  }

  auto app = holoscan::make_application<App>();
  if (tx_period_ms > 0 && rx_period_ms > 0) {
    app->set_tx_rx_periods(tx_period_ms, rx_period_ms);
  }

  if (use_event_based_scheduler) {
    auto scheduler = app->make_scheduler<holoscan::EventBasedScheduler>(
        "event-based-scheduler", holoscan::Arg("worker_thread_number", 2L));
    app->scheduler(scheduler);
  }

  app->run();
  return 0;
}

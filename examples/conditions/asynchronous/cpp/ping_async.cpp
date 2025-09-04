/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <filesystem>
#include <iostream>
#include <string>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/async_ping_rx/async_ping_rx.hpp>
#include <holoscan/operators/async_ping_tx/async_ping_tx.hpp>
#include <holoscan/operators/ping_rx/ping_rx.hpp>
#include <holoscan/operators/ping_tx/ping_tx.hpp>

class App : public holoscan::Application {
 public:
  void set_async_receive(bool async_receive) { async_receive_ = async_receive; }
  void set_async_transmit(bool async_transmit) { async_transmit_ = async_transmit; }

  void compose() override {
    using namespace holoscan;
    uint64_t tx_count = 20UL;

    if (async_receive_) {
      auto rx = make_operator<ops::AsyncPingRxOp>("rx", Arg("delay", 10L));
      if (async_transmit_) {
        auto tx =
            make_operator<ops::AsyncPingTxOp>("tx", Arg("delay", 10L), Arg("count", tx_count));
        add_flow(tx, rx);
      } else {
        auto tx = make_operator<ops::PingTxOp>("tx", make_condition<CountCondition>(tx_count));
        add_flow(tx, rx);
      }
    } else {
      auto rx = make_operator<ops::PingRxOp>("rx");
      if (async_transmit_) {
        auto tx =
            make_operator<ops::AsyncPingTxOp>("tx", Arg("delay", 10L), Arg("count", tx_count));
        add_flow(tx, rx);
      } else {
        auto tx = make_operator<ops::PingTxOp>("tx", make_condition<CountCondition>(tx_count));
        add_flow(tx, rx);
      }
    }
  }

 private:
  bool async_receive_ = true;
  bool async_transmit_ = false;
};

int main([[maybe_unused]] int argc, char** argv) {
  auto app = holoscan::make_application<App>();

  // Get the yaml configuration file
  auto config_path = std::filesystem::canonical(argv[0]).parent_path();
  config_path /= std::filesystem::path("ping_async.yaml");
  if (argc >= 2) {
    config_path = argv[1];
  }
  app->config(config_path);

  // set customizable application parameters via the YAML
  auto async_receive = app->from_config("async_receive").as<bool>();
  auto async_transmit = app->from_config("async_transmit").as<bool>();
  app->set_async_receive(async_receive);
  app->set_async_transmit(async_transmit);

  auto scheduler = app->from_config("scheduler").as<std::string>();
  holoscan::ArgList scheduler_args{holoscan::Arg("stop_on_deadlock", true),
                                   holoscan::Arg("stop_on_deadlock_timeout", 500L)};
  if (scheduler == "multi_thread") {
    // use MultiThreadScheduler instead of the default GreedyScheduler
    app->scheduler(app->make_scheduler<holoscan::MultiThreadScheduler>("MTS", scheduler_args));
  } else if (scheduler == "event_based") {
    // use EventBasedScheduler instead of the default GreedyScheduler
    app->scheduler(app->make_scheduler<holoscan::EventBasedScheduler>("EBS", scheduler_args));
  } else if (scheduler == "greedy") {
    app->scheduler(app->make_scheduler<holoscan::GreedyScheduler>("GS", scheduler_args));
  } else if (scheduler != "default") {
    throw std::runtime_error(fmt::format(
        "unrecognized scheduler option '{}', should be one of {{'multi_thread', 'event_based', "
        "'greedy', 'default'}}",
        scheduler));
  }

  // run the application
  app->run();

  std::cout << "Application has finished running.\n";

  return 0;
}

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
#include <thread>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/ping_rx/ping_rx.hpp>
#include <holoscan/operators/ping_tx/ping_tx.hpp>

class App : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    auto tx = make_operator<ops::PingTxOp>("tx", make_condition<CountCondition>(10));
    auto rx = make_operator<ops::PingRxOp>("rx");

    add_flow(tx, rx);

    // Save a reference to the tx operator so we can access it later
    target_op = tx;
  }
  std::shared_ptr<holoscan::Operator> target_op;
};

int main(int argc, char** argv) {
  auto app = holoscan::make_application<App>();
  auto future = app->run_async();
  HOLOSCAN_LOG_INFO("Application is running asynchronously.");

  auto print_status = std::thread([&app, &future]() {
    // Wait for the application to finish
    while (true) {
      const auto status = future.wait_for(std::chrono::seconds(0));
      if (status == std::future_status::ready) {
        HOLOSCAN_LOG_INFO("# Application finished");
        return;
      } else {
        // Print the current index of the tx operator
        auto tx = std::dynamic_pointer_cast<holoscan::ops::PingTxOp>(app->target_op);
        if (tx) {
          HOLOSCAN_LOG_INFO("# Application still running... PingTxOp index: {}", tx->index());
        } else {
          HOLOSCAN_LOG_INFO("# Application still running... PingTxOp index: {}", "N/A");
        }
      }
      std::this_thread::yield();
    }
  });
  print_status.join();  // print status while application is running

  // Block until application is done and throw any exceptions
  future.get();

  HOLOSCAN_LOG_INFO("Application has finished running.");
  return 0;
}

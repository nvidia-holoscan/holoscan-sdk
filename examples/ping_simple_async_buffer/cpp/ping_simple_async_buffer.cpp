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

#include <holoscan/holoscan.hpp>
#include "ping_rx_async_op.hpp"
#include "ping_tx_async_op.hpp"

class MyPingApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    auto tx = make_operator<ops::PingTxAsyncOp>("tx", make_condition<CountCondition>(20));
    auto rx = make_operator<ops::PingRxAsyncOp>("rx", make_condition<CountCondition>(50));

    add_flow(tx, rx, IOSpec::ConnectorType::kAsyncBuffer);
  }
};

int main() {
  auto app = holoscan::make_application<MyPingApp>();

  // Create and configure the EventBasedScheduler
  auto scheduler = app->make_scheduler<holoscan::EventBasedScheduler>(
      "event-based-scheduler",
      holoscan::Arg("worker_thread_number", 2L) /*Specify 2 worker threads*/);
  app->scheduler(scheduler);

  auto& tracker = app->track(0, 0, 0);

  app->run();

  tracker.print();

  return 0;
}

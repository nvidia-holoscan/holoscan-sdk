/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

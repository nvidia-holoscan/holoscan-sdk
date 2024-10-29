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

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/ping_tx/ping_tx.hpp>
#include <holoscan/operators/ping_rx/ping_rx.hpp>

using namespace std::chrono_literals;

namespace holoscan::ops {

class TimedPingRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TimedPingRxOp)

  TimedPingRxOp() = default;

  void setup(OperatorSpec& spec) override;

  void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override;
};

void TimedPingRxOp::setup(OperatorSpec& spec) {
  spec.input<int>("in");
}

void TimedPingRxOp::compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
                            [[maybe_unused]] ExecutionContext& context) {
  auto value = op_input.receive<int>("in").value();
  HOLOSCAN_LOG_INFO("Rx message value: {}", value);

  // retrieve the scheduler used for this application via it's fragment
  auto scheduler = fragment_->scheduler();

  // To get the clock we currently have to cast the scheduler to gxf::GXFScheduler.
  // TODO: Refactor C++ lib so the clock method is on Scheduler rather than GXFScheduler.
  //       That would allow us to avoid this dynamic_pointer_cast, but might require adding
  //       renaming Clock->GXFClock and then adding a new holoscan::Clock independent of GXF.
  auto gxf_scheduler = std::dynamic_pointer_cast<gxf::GXFScheduler>(scheduler);
  auto clock = gxf_scheduler->clock();

  // # The scheduler's clock is available as a parameter.
  // # The clock object has methods to retrieve timestamps and to sleep
  // # the thread for a specified duration as demonstrated below.
  HOLOSCAN_LOG_INFO("\treceive time (s) = {}", clock->time());
  HOLOSCAN_LOG_INFO("\treceive timestamp (ns) = {}", clock->timestamp());

  HOLOSCAN_LOG_INFO("\tnow pausing for 0.1 s...");
  clock->sleep_for(100'000'000);
  auto ts = clock->timestamp();
  HOLOSCAN_LOG_INFO("\ttimestamp after pause = {}", ts);

  HOLOSCAN_LOG_INFO("\tnow pausing until a target time 0.25 s in the future");
  auto target_ts = ts + 250'000'000;
  clock->sleep_until(target_ts);
  HOLOSCAN_LOG_INFO("\ttimestamp = {}", clock->timestamp());

  HOLOSCAN_LOG_INFO("\tnow pausing 0.125 s via std::chrono::duration");
  clock->sleep_for(0.125s);
  HOLOSCAN_LOG_INFO("\tfinal timestamp = {}", clock->timestamp());

  // The set_time_scale method is on the RealtimeClock class, but not the base Clock class
  // returned by gxf_scheduler->clock so a dynamic_pointer_class is required to use it.
  auto realtime_clock = std::dynamic_pointer_cast<RealtimeClock>(clock);
  HOLOSCAN_LOG_INFO("\tnow adjusting time scale to 4.0 (time runs 4x faster)");
  realtime_clock->set_time_scale(4.0);
  HOLOSCAN_LOG_INFO("\ttimestamp = {}", realtime_clock->timestamp());

  HOLOSCAN_LOG_INFO(
      "\tnow pausing 2.0 s via std::chrono::duration, but real pause will be 0.5 s "
      "due to the adjusted time scale");
  clock->sleep_for(2.0s);
  HOLOSCAN_LOG_INFO("\tfinal timestamp = {} (2.0 s increase will be shown despite scale of 4.0)",
                    clock->timestamp());

  HOLOSCAN_LOG_INFO("\tnow resetting the time scale back to 1.0");
  realtime_clock->set_time_scale(1.0);
}

}  // namespace holoscan::ops

class MyPingApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    // Define the tx and rx operators, allowing the tx operator to execute 3 times
    auto tx = make_operator<ops::PingTxOp>("tx", make_condition<CountCondition>(3));
    auto rx = make_operator<ops::TimedPingRxOp>("rx");

    // Define the workflow:  tx -> rx
    add_flow(tx, rx);
  }
};

int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv) {
  auto app = holoscan::make_application<MyPingApp>();
  app->run();

  return 0;
}

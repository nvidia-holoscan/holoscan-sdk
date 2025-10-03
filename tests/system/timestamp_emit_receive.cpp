/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <string>

#include "holoscan/holoscan.hpp"

namespace holoscan::ops {

class TimestampTxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TimestampTxOp)

  TimestampTxOp() = default;

  void setup(OperatorSpec& spec) override { spec.output<std::shared_ptr<std::string>>("out"); }

  void compute([[maybe_unused]] InputContext& op_input, OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto value = std::make_shared<std::string>(fmt::format("Timestamp Test ping: {}", index_));
    // emitting a timestamp is necessary for this port to be connected to an input port that is
    // using a ExpiringMessageAvailableCondition
    int64_t custom_timestamp = 555666777 + index_;
    op_output.emit(value, "out", custom_timestamp);
    ++index_;
  };

 private:
  int index_ = 1;
};

class TimestampRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TimestampRxOp)

  TimestampRxOp() = default;

  void setup(OperatorSpec& spec) override { spec.input<std::shared_ptr<std::string>>("in"); }

  void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    HOLOSCAN_LOG_INFO("TimestampRxOp::compute() called");
    while (true) {
      auto in_value = op_input.receive<std::shared_ptr<std::string>>("in");
      if (!in_value) {
        break;
      }

      // receive timestamp
      auto in_timestamp = op_input.get_acquisition_timestamp("in");
      ASSERT_TRUE(in_timestamp.has_value());
      HOLOSCAN_LOG_INFO("Rx message acquisition timestamp: {}", in_timestamp.value());
      ASSERT_EQ(in_timestamp.value(), 555666777 + index_);

      // receive all timestamps
      auto in_timestamps = op_input.get_acquisition_timestamps("in");
      ASSERT_EQ(in_timestamps.size(), 1);
      ASSERT_TRUE(in_timestamps[0].has_value());
      ASSERT_EQ(in_timestamps[0].value(), 555666777 + index_);

      auto message = in_value.value();
      HOLOSCAN_LOG_INFO("Rx message received: {}", message->c_str());
    }
    index_++;
  };

 private:
  int index_ = 1;
};

}  // namespace holoscan::ops

class TimestampTestApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    using namespace std::chrono_literals;
    // Configure the operators. Here we use CountCondition to terminate
    // execution after a specific number of messages have been sent.
    // PeriodicCondition is used so that each subsequent message is
    // sent only after a period of 10 milliseconds has elapsed.
    auto tx = make_operator<ops::TimestampTxOp>(
        "tx",
        make_condition<CountCondition>("count-condition", count_),
        make_condition<PeriodicCondition>("periodic-condition", 0.01s));

    auto rx = make_operator<ops::TimestampRxOp>("rx");

    add_flow(tx, rx);
  }

  void set_count(size_t count) { count_ = count; }

 private:
  size_t count_ = 5;
};

TEST(TimestampEmitReceive, TimestampEmitReceiveApp) {
  using namespace holoscan;

  size_t count = 5;
  auto app = make_application<TimestampTestApp>();
  app->set_count(count);

  app->scheduler(app->make_scheduler<holoscan::EventBasedScheduler>(
      "event-based",
      holoscan::Arg("worker_thread_number", static_cast<int64_t>(3)),
      holoscan::Arg("max_duration_ms", static_cast<int64_t>(10000))));

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find(fmt::format("Timestamp Test ping: {}", count)) != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find(fmt::format("Timestamp Test ping: {}", count + 1)) ==
              std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("error") == std::string::npos) << "=== LOG ===\n"
                                                             << log_output << "\n===========\n";
}

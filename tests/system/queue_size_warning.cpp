/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <vector>

#include "holoscan/holoscan.hpp"

#include "holoscan/operators/ping_tx/ping_tx.hpp"

namespace holoscan::ops {


class DefaultMinSizeRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(DefaultMinSizeRxOp)
  DefaultMinSizeRxOp() = default;

  void setup(OperatorSpec& spec) override {
    // queue_size > 1 with no explicit condition triggers the warning and uses min_size=queue_size.
    spec.input<int>("in", IOSpec::IOSize(2));
  }

  void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto values = op_input.receive<std::vector<int>>("in").value();
    EXPECT_EQ(values.size(), 2);
  }
};

class ExplicitMinSizeRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ExplicitMinSizeRxOp)
  ExplicitMinSizeRxOp() = default;

  void setup(OperatorSpec& spec) override {
    // queue_size=2 (buffering) but min_size=1 (no batching)
    spec.input<int>("in", IOSpec::IOSize(2))
        .condition(ConditionType::kMessageAvailable, Arg("min_size", static_cast<uint64_t>(1)));
  }

  void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    // Drain any messages that are currently available.
    while (true) {
      auto maybe_value = op_input.receive<int>("in");
      if (!maybe_value) {
        break;
      }
    }
  }
};

}  // namespace holoscan::ops

class QueueSizeWarningDefaultApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    using namespace std::chrono_literals;

    auto tx = make_operator<ops::PingTxOp>(
        "tx",
        make_condition<CountCondition>("count", 2),
        make_condition<PeriodicCondition>("periodic", 0.01s));
    auto rx = make_operator<ops::DefaultMinSizeRxOp>("rx");
    add_flow(tx, rx);
  }
};

class QueueSizeWarningExplicitMinSizeApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    using namespace std::chrono_literals;

    auto tx = make_operator<ops::PingTxOp>(
        "tx",
        make_condition<CountCondition>("count", 2),
        make_condition<PeriodicCondition>("periodic", 0.01s));
    auto rx = make_operator<ops::ExplicitMinSizeRxOp>("rx");
    add_flow(tx, rx);
  }
};

TEST(QueueSizeWarning, WarnOnQueueSizeGreaterThanOneWithDefaultCondition) {
  auto app = holoscan::make_application<QueueSizeWarningDefaultApp>();
  app->scheduler(app->make_scheduler<holoscan::EventBasedScheduler>(
      "event-based",
      holoscan::Arg("worker_thread_number", static_cast<int64_t>(2)),
      holoscan::Arg("max_duration_ms", static_cast<int64_t>(10000))));

  testing::internal::CaptureStderr();
  app->run();
  std::string log_output = testing::internal::GetCapturedStderr();

  EXPECT_NE(
      log_output.find("Input port 'in' of operator 'rx' is configured with queue_size=2 (> 1)."),
      std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(QueueSizeWarning, NoWarnWhenMinSizeIsExplicitlySet) {
  auto app = holoscan::make_application<QueueSizeWarningExplicitMinSizeApp>();
  app->scheduler(app->make_scheduler<holoscan::EventBasedScheduler>(
      "event-based",
      holoscan::Arg("worker_thread_number", static_cast<int64_t>(2)),
      holoscan::Arg("max_duration_ms", static_cast<int64_t>(10000))));

  testing::internal::CaptureStderr();
  app->run();
  std::string log_output = testing::internal::GetCapturedStderr();

  EXPECT_EQ(
      log_output.find("Input port 'in' of operator 'rx' is configured with queue_size=2 (> 1)."),
      std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

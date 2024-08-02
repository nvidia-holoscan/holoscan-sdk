/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "holoscan/holoscan.hpp"

#include "env_wrapper.hpp"

namespace {

class ValueData {
 public:
  ValueData() = default;
  explicit ValueData(int value) : data_(value) {
    HOLOSCAN_LOG_TRACE("ValueData::ValueData(): {}", data_);
  }
  ~ValueData() { HOLOSCAN_LOG_TRACE("ValueData::~ValueData(): {}", data_); }

  void data(int value) { data_ = value; }

  int data() const { return data_; }

 private:
  int data_;
};

class PingTxOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingTxOp)

  PingTxOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output<std::shared_ptr<ValueData>>("out1");
    spec.output<std::shared_ptr<ValueData>>("out2");
  }

  void compute(holoscan::InputContext&, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext&) override {
    auto value1 = std::make_shared<ValueData>(index_++);
    op_output.emit(value1, "out1");

    auto value2 = std::make_shared<ValueData>(index_++);
    op_output.emit(value2, "out2");
  };
  int index_ = 1;
};

class PingMxOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingMxOp)

  PingMxOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<std::shared_ptr<ValueData>>("in1");
    spec.input<std::shared_ptr<ValueData>>("in2");
    spec.output<std::shared_ptr<ValueData>>("out1");
    spec.output<std::shared_ptr<ValueData>>("out2");
    spec.param(multiplier_, "multiplier", "Multiplier", "Multiply the input by this value", 2);
  }

  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext&) override {
    auto value1 = op_input.receive<std::shared_ptr<ValueData>>("in1").value();
    auto value2 = op_input.receive<std::shared_ptr<ValueData>>("in2").value();

    HOLOSCAN_LOG_INFO("Middle message received (count: {})", count_++);

    HOLOSCAN_LOG_INFO("Middle message value1: {}", value1->data());
    HOLOSCAN_LOG_INFO("Middle message value2: {}", value2->data());

    // Multiply the values by the multiplier parameter
    value1->data(value1->data() * multiplier_);
    value2->data(value2->data() * multiplier_);

    op_output.emit(value1, "out1");
    op_output.emit(value2, "out2");
  };

 private:
  int count_ = 1;
  holoscan::Parameter<int> multiplier_;
};

class PingRxOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_TEMPLATE()
  PingRxOp(bool should_receive_vector, holoscan::IOSpec::IOSize queue_size, ArgT&& arg,
           ArgsT&&... args)
      : should_receive_vector_(should_receive_vector),
        queue_size_(queue_size),
        Operator(std::forward<ArgT>(arg), std::forward<ArgsT>(args)...) {}

  PingRxOp(bool should_receive_vector, holoscan::IOSpec::IOSize queue_size)
      : should_receive_vector_(should_receive_vector), queue_size_(queue_size) {}

  void setup(holoscan::OperatorSpec& spec) override {
    if (queue_size_ == holoscan::IOSpec::kSizeOne) {
      spec.input<std::shared_ptr<ValueData>>("receivers");
    } else {
      spec.input<std::vector<std::shared_ptr<ValueData>>>("receivers", queue_size_);
    }
  }

  void compute(holoscan::InputContext& op_input, holoscan::OutputContext&,
               holoscan::ExecutionContext&) override {
    if (should_receive_vector_) {
      auto value_vector =
          op_input.receive<std::vector<std::shared_ptr<ValueData>>>("receivers").value();

      HOLOSCAN_LOG_INFO("Rx message received (count: {}, size: {})", count_++, value_vector.size());

      std::vector<int> data_vector;
      data_vector.reserve(value_vector.size());
      for (const auto& item : value_vector) { data_vector.push_back(item->data()); }
      HOLOSCAN_LOG_INFO("Rx message values: [{}]", fmt::join(data_vector, ", "));
    } else {
      while (true) {
        auto maybe_value = op_input.receive<std::shared_ptr<ValueData>>("receivers");
        if (!maybe_value) { break; }
        auto value = maybe_value.value();
        HOLOSCAN_LOG_INFO("Rx message received (count: {}, size: 1)", count_++);
        HOLOSCAN_LOG_INFO("Rx message value: {}", value->data());
      }
    }
  };

 private:
  bool should_receive_vector_ = false;
  holoscan::IOSpec::IOSize queue_size_ = holoscan::IOSpec::kSizeOne;
  int count_ = 1;
};

class PingMultiPort : public holoscan::Application {
 public:
  PingMultiPort(bool should_receive_vector, holoscan::IOSpec::IOSize queue_size,
                const std::vector<std::string>& argv = {})
      : should_receive_vector_(should_receive_vector), queue_size_(queue_size), Application(argv) {}
  void compose() override {
    using namespace holoscan;

    // Define the tx, mx, rx operators, allowing the tx operator to execute 10 times
    auto tx = make_operator<PingTxOp>("tx", make_condition<CountCondition>(10));
    auto mx = make_operator<PingMxOp>("mx", Arg("multiplier", 3));
    auto rx = make_operator<PingRxOp>("rx", should_receive_vector_, queue_size_);

    // Define the workflow
    add_flow(tx, mx, {{"out1", "in1"}, {"out2", "in2"}});
    add_flow(mx, rx, {{"out1", "receivers"}, {"out2", "receivers"}});
  }

 private:
  bool should_receive_vector_ = false;
  holoscan::IOSpec::IOSize queue_size_ = holoscan::IOSpec::kSizeOne;
};

};  // namespace

TEST(MultiReceiverOperatorPingApp, TestPingMultiPortMultiAnySize) {
  auto app = holoscan::make_application<PingMultiPort>(true, holoscan::IOSpec::kAnySize);

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("Rx message received (count: 10, size: 2)") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("Rx message values: [57, 60]") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(MultiReceiverOperatorPingApp, TestPingMultiPortSingleAnySize) {
  auto app = holoscan::make_application<PingMultiPort>(false, holoscan::IOSpec::kAnySize);

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  EXPECT_THROW(app->run(), std::runtime_error);

  // it is expected that the run will throw an exception because
  // IOSpec::kAnySize always expects a vector of values when receiving
  // from a multi-receiver port.
  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(
      log_output.find("Unable to receive non-vector data from the input port 'receivers'") !=
      std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(MultiReceiverOperatorPingApp, TestPingMultiPortMultiPrecedingCount) {
  auto app = holoscan::make_application<PingMultiPort>(true, holoscan::IOSpec::kPrecedingCount);

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("Rx message received (count: 10, size: 2)") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  // since PingMxOp's output ports (out1, out2) are handled in std::unordered_map order,
  // the order of output port creation is not deterministic, so the order of the received
  // messages may be different.
  EXPECT_TRUE((log_output.find("Rx message values: [57, 60]") != std::string::npos) ||
              (log_output.find("Rx message values: [60, 57]") != std::string::npos))
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(MultiReceiverOperatorPingApp, TestPingMultiPortSinglePrecedingCount) {
  auto app = holoscan::make_application<PingMultiPort>(false, holoscan::IOSpec::kPrecedingCount);

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("Rx message received (count: 20, size: 1)") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  // since PingMxOp's output ports (out1, out2) are handled in std::unordered_map order,
  // the order of output port creation is not deterministic, so the order of the received
  // messages may be different.
  EXPECT_TRUE((log_output.find("Rx message value: 60") != std::string::npos) ||
              (log_output.find("Rx message value: 57") != std::string::npos))
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(MultiReceiverOperatorPingApp, TestPingMultiPortSingleSizeFive) {
  // make sure that debug messages are logged
  EnvVarWrapper wrapper({
      std::make_pair("HOLOSCAN_LOG_LEVEL", "DEBUG"),
      std::make_pair("HOLOSCAN_EXECUTOR_LOG_LEVEL", "INFO"),  // quiet multi_thread_scheduler.cpp
  });

  auto app = holoscan::make_application<PingMultiPort>(false, holoscan::IOSpec::IOSize(5));

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();

  // since PingMxOp cannot send all messages to PingRxOp (e.g., PingMxOp cannot send the sixth
  // message because the queue size of PingRxOp's input port is 5), PingRxOp will not receive all
  // messages (20 messages).

  EXPECT_TRUE(log_output.find("Push failed on 'receivers'") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  EXPECT_TRUE(log_output.find("No message is received from the input port with name 'receivers'") !=
              std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  EXPECT_TRUE(log_output.find("Rx message received (count: 15, size: 1)") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  EXPECT_TRUE(log_output.find("Rx message received (count: 16, size: 1)") == std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  EXPECT_TRUE((log_output.find("Rx message value: 60") == std::string::npos) &&
              (log_output.find("Rx message value: 57") == std::string::npos))
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(MultiReceiverOperatorPingApp, TestPingMultiPortInvalidQueueSize) {
  auto app = holoscan::make_application<PingMultiPort>(false, holoscan::IOSpec::IOSize(-3));

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  EXPECT_THROW(app->run(), std::runtime_error);

  std::string log_output = testing::internal::GetCapturedStderr();
  // it is expected that the run will throw an exception because
  // the queue size is invalid.
  EXPECT_TRUE((log_output.find("Invalid queue size: -3") != std::string::npos))
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

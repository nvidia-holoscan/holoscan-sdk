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

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "holoscan/holoscan.hpp"

constexpr int kDefaultNumOfIterations = 10;

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

// Operator definitions

class PingTxOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingTxOp)

  PingTxOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output<std::shared_ptr<ValueData>>("out1");
    spec.output<std::shared_ptr<ValueData>>("out2");
  }

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    auto value1 = std::make_shared<ValueData>(index_++);
    op_output.emit(value1, "out1");

    auto value2 = std::make_shared<ValueData>(index_++);
    op_output.emit(value2, "out2");
  };
  int index_ = 1;
};

template <typename ReceiveTypeT = holoscan::gxf::Entity>
class PingEntityRxOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingEntityRxOp)
  HOLOSCAN_OPERATOR_FORWARD_TEMPLATE()
  PingEntityRxOp(bool should_receive_vector, holoscan::IOSpec::IOSize queue_size, ArgT&& arg,
                 ArgsT&&... args)
      : should_receive_vector_(should_receive_vector),
        queue_size_(queue_size),
        Operator(std::forward<ArgT>(arg), std::forward<ArgsT>(args)...) {}

  PingEntityRxOp(bool should_receive_vector, holoscan::IOSpec::IOSize queue_size)
      : should_receive_vector_(should_receive_vector), queue_size_(queue_size) {}

  void setup(holoscan::OperatorSpec& spec) override {
    if (queue_size_ == holoscan::IOSpec::kSizeOne) {
      spec.input<ReceiveTypeT>("receivers");
    } else {
      spec.input<std::vector<ReceiveTypeT>>("receivers", queue_size_);
    }
  }

  void compute(holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    if (should_receive_vector_) {
      auto value_vector = op_input.receive<std::vector<ReceiveTypeT>>("receivers").value();

      if (typeid(value_vector) != typeid(std::vector<ReceiveTypeT>)) {
        HOLOSCAN_LOG_ERROR("Received wrong entity type: {}", typeid(value_vector).name());
        return;
      }

      HOLOSCAN_LOG_INFO("Rx message received (count: {}, size: {})", count_, value_vector.size());
    } else {
      int vector_size = 0;
      while (true) {
        auto maybe_value = op_input.receive<ReceiveTypeT>("receivers");
        if (!maybe_value) { break; }
        if (typeid(maybe_value.value()) != typeid(ReceiveTypeT)) {
          HOLOSCAN_LOG_ERROR("Received wrong entity type: {}", typeid(maybe_value.value()).name());
          break;
        }
        auto value = maybe_value.value();
        vector_size++;
      }
      HOLOSCAN_LOG_INFO("Rx message received (count: {}, size: {})", count_, vector_size);
    }

    count_++;
  };

 private:
  bool should_receive_vector_ = false;
  holoscan::IOSpec::IOSize queue_size_ = holoscan::IOSpec::kSizeOne;
  int count_ = 1;
};

// Application definitions
template <typename SendOp, typename ReceiveOp>
class PingMultiPortDataApp : public holoscan::Application {
 public:
  PingMultiPortDataApp(bool should_receive_vector, holoscan::IOSpec::IOSize queue_size,
                       int count = kDefaultNumOfIterations,
                       const std::vector<std::string>& argv = {})
      : should_receive_vector_(should_receive_vector),
        queue_size_(queue_size),
        count_(count),
        Application(argv) {}

  void compose() override {
    using namespace holoscan;

    // Define the tx, mx, rx operators, allowing the tx operator to execute count_ times
    auto tx = make_operator<SendOp>("tx", make_condition<CountCondition>(count_));
    auto rx = make_operator<ReceiveOp>("rx", should_receive_vector_, queue_size_);

    // Define the workflow
    add_flow(tx, rx, {{"out1", "receivers"}, {"out2", "receivers"}});
  }

 private:
  bool should_receive_vector_ = false;
  holoscan::IOSpec::IOSize queue_size_ = holoscan::IOSpec::kSizeOne;
  int count_ = kDefaultNumOfIterations;
};

};  // namespace

TEST(ReceiveAsEntityApp, TestSendingDataMultiAnySizeReceiveHoloscanEntity) {
  using PingMultiPortEntityApp =
      PingMultiPortDataApp<PingTxOp, PingEntityRxOp<holoscan::gxf::Entity>>;

  auto app = holoscan::make_application<PingMultiPortEntityApp>(true, holoscan::IOSpec::kAnySize);

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("Rx message received (count: 10, size: 2)") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(ReceiveAsEntityApp, TestSendingDataMultiPrecedingCountReceiveHoloscanEntity) {
  using PingMultiPortEntityApp =
      PingMultiPortDataApp<PingTxOp, PingEntityRxOp<holoscan::gxf::Entity>>;

  auto app =
      holoscan::make_application<PingMultiPortEntityApp>(true, holoscan::IOSpec::kPrecedingCount);

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("Rx message received (count: 10, size: 2)") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(ReceiveAsEntityApp, TestSendingDataMultiAnySizeReceiveNvidiaGxfEntity) {
  using PingMultiPortEntityApp =
      PingMultiPortDataApp<PingTxOp, PingEntityRxOp<nvidia::gxf::Entity>>;

  auto app = holoscan::make_application<PingMultiPortEntityApp>(true, holoscan::IOSpec::kAnySize);

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("Rx message received (count: 10, size: 2)") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(ReceiveAsEntityApp, TestSendingDataMultiPrecedingCountReceiveNvidiaGxfEntity) {
  using PingMultiPortEntityApp =
      PingMultiPortDataApp<PingTxOp, PingEntityRxOp<nvidia::gxf::Entity>>;

  auto app =
      holoscan::make_application<PingMultiPortEntityApp>(true, holoscan::IOSpec::kPrecedingCount);

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("Rx message received (count: 10, size: 2)") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(ReceiveAsEntityApp, TestSendingDataMultiAnySizeReceiveIndividualHoloscanEntity) {
  using PingMultiPortEntityApp =
      PingMultiPortDataApp<PingTxOp, PingEntityRxOp<holoscan::gxf::Entity>>;

  auto app = holoscan::make_application<PingMultiPortEntityApp>(false, holoscan::IOSpec::kAnySize);

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  EXPECT_THROW(app->run(), std::invalid_argument);

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

TEST(ReceiveAsEntityApp, TestSendingDataMultiPrecedingCountReceiveIndividualHoloscanEntity) {
  using PingMultiPortEntityApp =
      PingMultiPortDataApp<PingTxOp, PingEntityRxOp<holoscan::gxf::Entity>>;

  auto app =
      holoscan::make_application<PingMultiPortEntityApp>(false, holoscan::IOSpec::kPrecedingCount);

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("Rx message received (count: 10, size: 2)") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(ReceiveAsEntityApp, TestSendingDataMultiAnySizeReceiveIndividualNvidiaGxfEntity) {
  using PingMultiPortEntityApp =
      PingMultiPortDataApp<PingTxOp, PingEntityRxOp<nvidia::gxf::Entity>>;

  auto app = holoscan::make_application<PingMultiPortEntityApp>(false, holoscan::IOSpec::kAnySize);

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  EXPECT_THROW(app->run(), std::invalid_argument);

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

TEST(ReceiveAsEntityApp, TestSendingDataMultiPrecedingCountReceiveIndividualNvidiaGxfEntity) {
  using PingMultiPortEntityApp =
      PingMultiPortDataApp<PingTxOp, PingEntityRxOp<nvidia::gxf::Entity>>;

  auto app =
      holoscan::make_application<PingMultiPortEntityApp>(false, holoscan::IOSpec::kPrecedingCount);

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("Rx message received (count: 10, size: 2)") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

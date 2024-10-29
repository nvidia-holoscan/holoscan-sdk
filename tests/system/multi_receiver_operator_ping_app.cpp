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

class PingNullSharedPtrTxOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingNullSharedPtrTxOp)

  PingNullSharedPtrTxOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output<std::shared_ptr<ValueData>>("out1");
    spec.output<std::shared_ptr<ValueData>>("out2");
  }

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    auto value1 = std::make_shared<ValueData>(index_++);
    op_output.emit(nullptr, "out1");

    auto value2 = std::make_shared<ValueData>(index_++);
    op_output.emit(value2, "out2");
  };
  int index_ = 1;
};

class PingRawNullPtrTxOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingRawNullPtrTxOp)

  PingRawNullPtrTxOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output<const char*>("out1");
    spec.output<const char*>("out2");
  }

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    static const char values[] = {
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
    op_output.emit(nullptr, "out1");

    auto value2 = &values[(index_++) % 16];
    op_output.emit(value2, "out2");
  };
  int index_ = 0;
};

class PingTensorMapTxOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingTensorMapTxOp)

  PingTensorMapTxOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output<holoscan::TensorMap>("out1");
    spec.output<holoscan::TensorMap>("out2");
  }

  void initialize() {
    // Create an allocator for the tensors
    auto frag = fragment();
    allocator_ = frag->make_resource<holoscan::UnboundedAllocator>("allocator");
    add_arg(allocator_.get());

    Operator::initialize();
  }

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output, holoscan::ExecutionContext& context) override {
    const nvidia::gxf::Shape out_shape1{1, 2, 3};
    const nvidia::gxf::Shape out_shape2{3, 2, 1};
    const nvidia::gxf::Shape out_shape3{2, 3, 4};
    const nvidia::gxf::Shape out_shape4{4, 3, 2};

    // Get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
    auto pool = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(),
                                                                    allocator_->gxf_cid());
    const auto maybe_tensormap1 = nvidia::gxf::CreateTensorMap(
        context.context(),
        pool.value(),
        {{"tensor_a",
          nvidia::gxf::MemoryStorageType::kDevice,
          out_shape1,
          nvidia::gxf::PrimitiveType::kUnsigned8,
          0,
          nvidia::gxf::ComputeTrivialStrides(
              out_shape1, nvidia::gxf::PrimitiveTypeSize(nvidia::gxf::PrimitiveType::kUnsigned8))},
         {"tensor_common",
          nvidia::gxf::MemoryStorageType::kDevice,
          out_shape2,
          nvidia::gxf::PrimitiveType::kUnsigned8,
          0,
          nvidia::gxf::ComputeTrivialStrides(
              out_shape2, nvidia::gxf::PrimitiveTypeSize(nvidia::gxf::PrimitiveType::kUnsigned8))}},
        false);

    const auto maybe_tensormap2 = nvidia::gxf::CreateTensorMap(
        context.context(),
        pool.value(),
        {{"tensor_c",
          nvidia::gxf::MemoryStorageType::kDevice,
          out_shape3,
          nvidia::gxf::PrimitiveType::kUnsigned8,
          0,
          nvidia::gxf::ComputeTrivialStrides(
              out_shape3, nvidia::gxf::PrimitiveTypeSize(nvidia::gxf::PrimitiveType::kUnsigned8))},
         {"tensor_common",
          nvidia::gxf::MemoryStorageType::kDevice,
          out_shape4,
          nvidia::gxf::PrimitiveType::kUnsigned8,
          0,
          nvidia::gxf::ComputeTrivialStrides(
              out_shape4, nvidia::gxf::PrimitiveTypeSize(nvidia::gxf::PrimitiveType::kUnsigned8))}},
        false);

    if (!maybe_tensormap1 || !maybe_tensormap2) {
      throw std::runtime_error("Failed to create TensorMap");
    }

    auto gxf_entity1 = holoscan::gxf::Entity(maybe_tensormap1.value());
    auto gxf_entity2 = holoscan::gxf::Entity(maybe_tensormap2.value());

    holoscan::TensorMap tensor_map1;
    tensor_map1.insert({"tensor_a", gxf_entity1.get<holoscan::Tensor>("tensor_a")});
    tensor_map1.insert({"tensor_common", gxf_entity1.get<holoscan::Tensor>("tensor_common")});
    op_output.emit(tensor_map1, "out1");

    holoscan::TensorMap tensor_map2;
    tensor_map2.insert({"tensor_c", gxf_entity2.get<holoscan::Tensor>("tensor_c")});
    tensor_map2.insert({"tensor_common", gxf_entity2.get<holoscan::Tensor>("tensor_common")});
    op_output.emit(tensor_map2, "out2");
  };

 private:
  holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> allocator_{nullptr};
  int index_ = 0;
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
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
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

  void compute(holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    if (should_receive_vector_) {
      auto value_vector =
          op_input.receive<std::vector<std::shared_ptr<ValueData>>>("receivers").value();

      HOLOSCAN_LOG_INFO("Rx message received (count: {}, size: {})", count_++, value_vector.size());

      std::vector<int> data_vector;
      data_vector.reserve(value_vector.size());
      for (const auto& item : value_vector) {
        if (item) {
          data_vector.push_back(item->data());
        } else {
          data_vector.push_back(-1);
        }
      }
      HOLOSCAN_LOG_INFO("Rx message values: [{}]", fmt::join(data_vector, ", "));
    } else {
      while (true) {
        auto maybe_value = op_input.receive<std::shared_ptr<ValueData>>("receivers");
        if (!maybe_value) { break; }
        auto value = maybe_value.value();
        HOLOSCAN_LOG_INFO("Rx message received (count: {}, size: 1)", count_++);
        if (value) {
          HOLOSCAN_LOG_INFO("Rx message value: {}", value->data());
        } else {
          HOLOSCAN_LOG_INFO("Rx message value: -1");
        }
      }
    }
  };

 private:
  bool should_receive_vector_ = false;
  holoscan::IOSpec::IOSize queue_size_ = holoscan::IOSpec::kSizeOne;
  int count_ = 1;
};

class PingRawPtrRxOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingRawPtrRxOp)
  HOLOSCAN_OPERATOR_FORWARD_TEMPLATE()
  PingRawPtrRxOp(bool should_receive_vector, holoscan::IOSpec::IOSize queue_size, ArgT&& arg,
                 ArgsT&&... args)
      : should_receive_vector_(should_receive_vector),
        queue_size_(queue_size),
        Operator(std::forward<ArgT>(arg), std::forward<ArgsT>(args)...) {}

  PingRawPtrRxOp(bool should_receive_vector, holoscan::IOSpec::IOSize queue_size)
      : should_receive_vector_(should_receive_vector), queue_size_(queue_size) {}

  void setup(holoscan::OperatorSpec& spec) override {
    if (queue_size_ == holoscan::IOSpec::kSizeOne) {
      spec.input<const char*>("receivers");
    } else {
      spec.input<std::vector<const char*>>("receivers", queue_size_);
    }
  }

  void compute(holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    if (should_receive_vector_) {
      auto maybe_value_vector = op_input.receive<std::vector<const char*>>("receivers");

      if (!maybe_value_vector) {
        HOLOSCAN_LOG_INFO("Unable to receive vector of raw pointers: {}",
                          maybe_value_vector.error().what());
        return;
      }

      auto& value_vector = maybe_value_vector.value();
      HOLOSCAN_LOG_INFO("Rx message received (count: {}, size: {})", count_++, value_vector.size());

      std::vector<char> data_vector;
      data_vector.reserve(value_vector.size());
      for (const auto& item : value_vector) {
        if (item) {
          data_vector.push_back(*item);
        } else {
          data_vector.push_back('N');
        }
      }
      HOLOSCAN_LOG_INFO("Rx message values: [{}]", fmt::join(data_vector, ", "));
    } else {
      while (true) {
        auto maybe_value = op_input.receive<const char*>("receivers");
        if (!maybe_value) { break; }
        auto value = maybe_value.value();
        HOLOSCAN_LOG_INFO("Rx message received (count: {}, size: 1)", count_++);
        if (value) {
          HOLOSCAN_LOG_INFO("Rx message value: {}", *value);
        } else {
          HOLOSCAN_LOG_INFO("Rx message value: N");
        }
      }
    }
  };

 private:
  bool should_receive_vector_ = false;
  holoscan::IOSpec::IOSize queue_size_ = holoscan::IOSpec::kSizeOne;
  int count_ = 1;
};

class PingTensorMapRxOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingTensorMapRxOp)
  HOLOSCAN_OPERATOR_FORWARD_TEMPLATE()
  PingTensorMapRxOp(bool should_receive_vector, holoscan::IOSpec::IOSize queue_size, ArgT&& arg,
                    ArgsT&&... args)
      : should_receive_vector_(should_receive_vector),
        queue_size_(queue_size),
        Operator(std::forward<ArgT>(arg), std::forward<ArgsT>(args)...) {}

  PingTensorMapRxOp(bool should_receive_vector, holoscan::IOSpec::IOSize queue_size)
      : should_receive_vector_(should_receive_vector), queue_size_(queue_size) {}

  void setup(holoscan::OperatorSpec& spec) override {
    if (queue_size_ == holoscan::IOSpec::kSizeOne) {
      spec.input<holoscan::TensorMap>("receivers");
    } else {
      spec.input<std::vector<holoscan::TensorMap>>("receivers", queue_size_);
    }
  }

  void compute(holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    if (should_receive_vector_) {
      auto value_vector = op_input.receive<std::vector<holoscan::TensorMap>>("receivers").value();

      HOLOSCAN_LOG_INFO("Rx message received (count: {}, size: {})", count_, value_vector.size());

      for (const auto& value : value_vector) {
        for (const auto& [name, tensor] : value) {
          std::vector<int> data_vector;
          if (tensor) {
            for (const auto& shape : tensor->shape()) { data_vector.push_back(shape); }
          }
          HOLOSCAN_LOG_INFO(
              "Rx message values {} (count: {}): [{}]", name, count_, fmt::join(data_vector, ", "));
        }
      }
    } else {
      while (true) {
        auto maybe_value = op_input.receive<holoscan::TensorMap>("receivers");
        if (!maybe_value) { break; }
        auto value = maybe_value.value();
        for (const auto& [name, tensor] : value) {
          std::vector<int> data_vector;
          if (tensor) {
            for (const auto& shape : tensor->shape()) { data_vector.push_back(shape); }
          }
          HOLOSCAN_LOG_INFO(
              "Rx message values {} (count: {}): [{}]", name, count_, fmt::join(data_vector, ", "));
        }
      }
    }

    count_++;
  };

 private:
  bool should_receive_vector_ = false;
  holoscan::IOSpec::IOSize queue_size_ = holoscan::IOSpec::kSizeOne;
  int count_ = 1;
};

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
      spec.input<holoscan::gxf::Entity>("receivers");
    } else {
      spec.input<std::vector<holoscan::gxf::Entity>>("receivers", queue_size_);
    }
  }

  void compute(holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    if (should_receive_vector_) {
      auto value_vector = op_input.receive<std::vector<holoscan::gxf::Entity>>("receivers").value();

      HOLOSCAN_LOG_INFO("Rx message received (count: {}, size: {})", count_, value_vector.size());

      for (const auto& value : value_vector) {
        std::shared_ptr<holoscan::Tensor> tensor_common =
            value.get<holoscan::Tensor>("tensor_common");
        std::vector<int> data_vector;
        if (tensor_common) {
          for (const auto& shape : tensor_common->shape()) { data_vector.push_back(shape); }
        }
        HOLOSCAN_LOG_INFO("Rx message values tensor_common (count: {}): [{}]",
                          count_,
                          fmt::join(data_vector, ", "));
      }
    } else {
      while (true) {
        auto maybe_value = op_input.receive<holoscan::gxf::Entity>("receivers");
        if (!maybe_value) { break; }
        auto value = maybe_value.value();

        std::shared_ptr<holoscan::Tensor> tensor_common =
            value.get<holoscan::Tensor>("tensor_common");
        std::vector<int> data_vector;
        if (tensor_common) {
          for (const auto& shape : tensor_common->shape()) { data_vector.push_back(shape); }
        }
        HOLOSCAN_LOG_INFO("Rx message values tensor_common (count: {}): [{}]",
                          count_,
                          fmt::join(data_vector, ", "));
      }
    }

    count_++;
  };

 private:
  bool should_receive_vector_ = false;
  holoscan::IOSpec::IOSize queue_size_ = holoscan::IOSpec::kSizeOne;
  int count_ = 1;
};

// Application definitions

class PingMultiPortApp : public holoscan::Application {
 public:
  PingMultiPortApp(bool should_receive_vector, holoscan::IOSpec::IOSize queue_size,
                   const std::vector<std::string>& argv = {})
      : should_receive_vector_(should_receive_vector), queue_size_(queue_size), Application(argv) {}
  void compose() override {
    using namespace holoscan;

    // Define the tx, mx, rx operators, allowing the tx operator to execute 10 times
    auto tx =
        make_operator<PingTxOp>("tx", make_condition<CountCondition>(kDefaultNumOfIterations));
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

    // Define the tx, mx, rx operators, allowing the tx operator to execute 10 times
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

TEST(MultiReceiverOperatorPingApp, TestPingMultiPortMultiAnySize) {
  auto app = holoscan::make_application<PingMultiPortApp>(true, holoscan::IOSpec::kAnySize);

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
  auto app = holoscan::make_application<PingMultiPortApp>(false, holoscan::IOSpec::kAnySize);

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

TEST(MultiReceiverOperatorPingApp, TestPingMultiPortMultiPrecedingCount) {
  auto app = holoscan::make_application<PingMultiPortApp>(true, holoscan::IOSpec::kPrecedingCount);

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
  auto app = holoscan::make_application<PingMultiPortApp>(false, holoscan::IOSpec::kPrecedingCount);

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

  auto app = holoscan::make_application<PingMultiPortApp>(false, holoscan::IOSpec::IOSize(5));

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

  EXPECT_TRUE(
      log_output.find(
          "ReceiveError on input port 'receivers': No message received from the input port") !=
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
  auto app = holoscan::make_application<PingMultiPortApp>(false, holoscan::IOSpec::IOSize(-3));

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

TEST(MultiReceiverOperatorPingApp, TestSendingNullSharedPtrMultiAnySize) {
  using PingNullSharedPtrApp = PingMultiPortDataApp<PingNullSharedPtrTxOp, PingRxOp>;

  auto app = holoscan::make_application<PingNullSharedPtrApp>(true, holoscan::IOSpec::kAnySize);

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("Rx message received (count: 10, size: 2)") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("Rx message values: [-1, 20]") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(MultiReceiverOperatorPingApp, TestSendingNullSharedPtrMultiPrecedingCount) {
  using PingNullSharedPtrApp = PingMultiPortDataApp<PingNullSharedPtrTxOp, PingRxOp>;

  auto app =
      holoscan::make_application<PingNullSharedPtrApp>(true, holoscan::IOSpec::kPrecedingCount);

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("Rx message received (count: 10, size: 2)") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("Rx message values: [-1, 20]") != std::string::npos ||
              log_output.find("Rx message values: [20, -1]") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(MultiReceiverOperatorPingApp, TestSendingNullSharedPtrSinglePrecedingCount) {
  using PingNullSharedPtrApp = PingMultiPortDataApp<PingNullSharedPtrTxOp, PingRxOp>;

  auto app =
      holoscan::make_application<PingNullSharedPtrApp>(false, holoscan::IOSpec::kPrecedingCount);

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("Rx message received (count: 20, size: 1)") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("Rx message value: -1") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("Rx message value: 20") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(MultiReceiverOperatorPingApp, TestSendingNullRawPointerMultiAnySize) {
  using PingNullPtrApp = PingMultiPortDataApp<PingRawNullPtrTxOp, PingRawPtrRxOp>;

  auto app = holoscan::make_application<PingNullPtrApp>(true, holoscan::IOSpec::kAnySize);

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("Rx message received (count: 10, size: 2)") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  // since PingRawNullPtrTxOp's output ports (out1, out2) are handled in std::unordered_map order,
  // the order of output port creation is not deterministic, so the order of the received
  // messages may be different.
  EXPECT_TRUE(log_output.find("Rx message values: [N, 9]") != std::string::npos ||
              log_output.find("Rx message values: [9, N]") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(MultiReceiverOperatorPingApp, TestSendingNullRawPointerMultiPrecedingCount) {
  using PingNullPtrApp = PingMultiPortDataApp<PingRawNullPtrTxOp, PingRawPtrRxOp>;

  auto app = holoscan::make_application<PingNullPtrApp>(true, holoscan::IOSpec::kPrecedingCount);

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("Rx message received (count: 10, size: 2)") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  // since PingRawNullPtrTxOp's output ports (out1, out2) are handled in std::unordered_map order,
  // the order of output port creation is not deterministic, so the order of the received
  // messages may be different.
  EXPECT_TRUE(log_output.find("Rx message values: [N, 9]") != std::string::npos ||
              log_output.find("Rx message values: [9, N]") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(MultiReceiverOperatorPingApp, TestSendingNullRawPointerSinglePrecedingCount) {
  using PingNullPtrApp = PingMultiPortDataApp<PingRawNullPtrTxOp, PingRawPtrRxOp>;

  auto app = holoscan::make_application<PingNullPtrApp>(false, holoscan::IOSpec::kPrecedingCount);

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("Rx message received (count: 20, size: 1)") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  // since PingRawNullPtrTxOp's output ports (out1, out2) are handled in std::unordered_map order,
  // the order of output port creation is not deterministic, so the order of the received
  // messages may be different.
  EXPECT_TRUE(log_output.find("Rx message value: N") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  EXPECT_TRUE(log_output.find("Rx message value: 9") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(MultiReceiverOperatorPingApp, TestSendingTensorMapDataMultiAnySize) {
  using PingMultiPortTensorMapApp = PingMultiPortDataApp<PingTensorMapTxOp, PingTensorMapRxOp>;

  auto app =
      holoscan::make_application<PingMultiPortTensorMapApp>(true, holoscan::IOSpec::kAnySize);

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("Rx message received (count: 10, size: 2)") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  EXPECT_TRUE(log_output.find("Rx message values tensor_a (count: 10): [1, 2, 3]") !=
              std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("Rx message values tensor_common (count: 10): [3, 2, 1]") !=
              std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("Rx message values tensor_c (count: 10): [2, 3, 4]") !=
              std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("Rx message values tensor_common (count: 10): [4, 3, 2]") !=
              std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(MultiReceiverOperatorPingApp, TestSendingTensorMapDataMultiPrecedingCount) {
  using PingMultiPortTensorMapApp = PingMultiPortDataApp<PingTensorMapTxOp, PingTensorMapRxOp>;

  auto app = holoscan::make_application<PingMultiPortTensorMapApp>(
      true, holoscan::IOSpec::kPrecedingCount);

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("Rx message received (count: 10, size: 2)") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  EXPECT_TRUE(log_output.find("Rx message values tensor_a (count: 10): [1, 2, 3]") !=
              std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("Rx message values tensor_common (count: 10): [3, 2, 1]") !=
              std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("Rx message values tensor_c (count: 10): [2, 3, 4]") !=
              std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("Rx message values tensor_common (count: 10): [4, 3, 2]") !=
              std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(MultiReceiverOperatorPingApp, TestSendingTensorMapDataSinglePrecedingCount) {
  using PingMultiPortTensorMapApp = PingMultiPortDataApp<PingTensorMapTxOp, PingTensorMapRxOp>;

  auto app = holoscan::make_application<PingMultiPortTensorMapApp>(
      true, holoscan::IOSpec::kPrecedingCount);

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("Rx message received (count: 10, size: 2)") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  EXPECT_TRUE(log_output.find("Rx message values tensor_a (count: 10): [1, 2, 3]") !=
              std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("Rx message values tensor_common (count: 10): [3, 2, 1]") !=
              std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("Rx message values tensor_c (count: 10): [2, 3, 4]") !=
              std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("Rx message values tensor_common (count: 10): [4, 3, 2]") !=
              std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(MultiReceiverOperatorPingApp, TestSendingEntityDataMultiAnySize) {
  using PingMultiPortEntityApp = PingMultiPortDataApp<PingTensorMapTxOp, PingEntityRxOp>;

  auto app = holoscan::make_application<PingMultiPortEntityApp>(true, holoscan::IOSpec::kAnySize);

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("Rx message received (count: 10, size: 2)") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  EXPECT_TRUE(log_output.find("Rx message values tensor_common (count: 10): [3, 2, 1]") !=
              std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("Rx message values tensor_common (count: 10): [4, 3, 2]") !=
              std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(MultiReceiverOperatorPingApp, TestSendingEntityDataMultiPrecedingCount) {
  using PingMultiPortEntityApp = PingMultiPortDataApp<PingTensorMapTxOp, PingEntityRxOp>;

  auto app =
      holoscan::make_application<PingMultiPortEntityApp>(true, holoscan::IOSpec::kPrecedingCount);

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("Rx message received (count: 10, size: 2)") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  EXPECT_TRUE(log_output.find("Rx message values tensor_common (count: 10): [3, 2, 1]") !=
              std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("Rx message values tensor_common (count: 10): [4, 3, 2]") !=
              std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(MultiReceiverOperatorPingApp, TestSendingEntityDataSinglePrecedingCount) {
  using PingMultiPortEntityApp = PingMultiPortDataApp<PingTensorMapTxOp, PingEntityRxOp>;

  auto app =
      holoscan::make_application<PingMultiPortEntityApp>(false, holoscan::IOSpec::kPrecedingCount);

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("Rx message values tensor_common (count: 10): [3, 2, 1]") !=
              std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("Rx message values tensor_common (count: 10): [4, 3, 2]") !=
              std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(MultiReceiverOperatorPingApp, TestSendingTensorMapDataMultiAnySizeIncorrectReceiver) {
  using PingMultiPortTensorMapApp = PingMultiPortDataApp<PingTensorMapTxOp, PingRawPtrRxOp>;

  auto app =
      holoscan::make_application<PingMultiPortTensorMapApp>(true, holoscan::IOSpec::kAnySize);

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(
      log_output.find(
          "Unable to cast the received data to the specified type for input 'receivers:0'") !=
      std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

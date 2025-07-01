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

#include "holoscan/utils/operator_runner.hpp"

#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "holoscan/holoscan.hpp"

using namespace holoscan::ops;

class IntegerGeneratorOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(IntegerGeneratorOp)

  IntegerGeneratorOp() = default;

  void setup(holoscan::OperatorSpec& spec) override { spec.output<int>("out"); }

  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext&) override {
    HOLOSCAN_LOG_INFO("{} - compute() called.", name());
    value_++;
    op_output.emit(value_, "out");
  }

 private:
  int value_ = 0;
};

class ProcessingOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ProcessingOp)

  ProcessingOp() = default;

  explicit ProcessingOp(std::function<int(int)> op_func) : Operator() {
    op_func_ = std::move(op_func);
  }

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<int>("in");
    spec.output<int>("out");
  }

  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext&) override {
    HOLOSCAN_LOG_INFO("{} - compute() called.", name());
    auto in_message = op_input.receive<int>("in").value();

    auto out_message = op_func_(in_message);

    op_output.emit(out_message, "out");
  }

 private:
  std::function<int(int)> op_func_;
};

class ChainingOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ChainingOp)

  ChainingOp() = default;

  void chain_operator(std::shared_ptr<holoscan::Operator> op) { operators_.push_back(op); }

  void initialize() override {
    Operator::initialize();

    for (auto& op : operators_) {
      auto op_runner = std::make_shared<holoscan::ops::OperatorRunner>(op);
      operator_runners_.push_back(op_runner);
    }
  }

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<int>("in");
    spec.output<int>("out");
  }

  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext&) override {
    HOLOSCAN_LOG_INFO("{} - compute() called.", name());
    auto in_message = op_input.receive<int>("in").value();
    auto out_message = in_message;

    for (auto& op_runner : operator_runners_) {
      auto result = op_runner->push_input("in", in_message);
      if (!result) {
        auto error_msg = fmt::format("Failed to push input to operator {} - {}",
                                     op_runner->op()->name(),
                                     result.error().what());
        HOLOSCAN_LOG_ERROR(error_msg);
        throw std::runtime_error(error_msg);
      }

      op_runner->run();

      auto maybe_out_message = op_runner->pop_output<int>("out");

      if (!maybe_out_message) {
        auto error_msg = fmt::format("Failed to pop output from operator {} - {}",
                                     op_runner->op()->name(),
                                     maybe_out_message.error().what());
        HOLOSCAN_LOG_ERROR(error_msg);
        throw std::runtime_error(error_msg);
      }
      out_message = maybe_out_message.value();

      // Set the output message as the input message for the next operator
      in_message = out_message;
    }

    op_output.emit(out_message, "out");
  }

 private:
  std::vector<std::shared_ptr<holoscan::Operator>> operators_;
  std::vector<std::shared_ptr<holoscan::ops::OperatorRunner>> operator_runners_;
};

class PrintResultOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PrintResultOp)

  PrintResultOp() = default;

  void setup(holoscan::OperatorSpec& spec) override { spec.input<int>("in"); }

  void compute(holoscan::InputContext& op_input, holoscan::OutputContext&,
               holoscan::ExecutionContext&) override {
    HOLOSCAN_LOG_INFO("{} - compute() called.", name());

    auto value = op_input.receive<int>("in").value();

    HOLOSCAN_LOG_INFO("Final value: {}", value);
  }
};

class OperatorRunnerTestApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    auto integer_gen_op =
        make_operator<IntegerGeneratorOp>("gen", make_condition<CountCondition>(5));
    auto chainer_op = make_operator<ChainingOp>("chainer_op");
    chainer_op->chain_operator(make_operator<ProcessingOp>("op1", [](int x) { return x + 3; }));
    chainer_op->chain_operator(make_operator<ProcessingOp>("op2", [](int x) { return x * 2; }));
    chainer_op->chain_operator(make_operator<ProcessingOp>("op3", [](int x) { return x - 1; }));
    auto print_op = make_operator<PrintResultOp>("print");
    add_flow(integer_gen_op, chainer_op);
    add_flow(chainer_op, print_op);
  }
};

TEST(OperatorRunnerApp, TestAppWithOperatorRunner) {
  auto app = holoscan::make_application<OperatorRunnerTestApp>();

  // verify that second call to config raises a warning
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();

  // For the last input (5), the output is ((5 + 3) * 2) - 1 = 15
  EXPECT_TRUE(log_output.find("Final value: 15") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

///////////////////////////////////////////////////////////////////////////////
// Test with various input types
///////////////////////////////////////////////////////////////////////////////

// Template operator to verify input/output of different types
template <typename T>
class VerificationOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(VerificationOp)

  VerificationOp() = default;

  explicit VerificationOp(T& expected_value) : expected_value_(expected_value) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<T>("in");
    spec.output<T>("out");
  }

  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override {
    HOLOSCAN_LOG_INFO("{} - compute() called.", name());
    auto value = op_input.receive<T>("in").value();

    // Verify the received value matches expected
    EXPECT_EQ(value, expected_value_);

    // Forward the value
    op_output.emit(value, "out");
  }

 private:
  T expected_value_{};
};

// Template operator that uses OperatorRunner to forward data
template <typename T>
class DataForwardOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(DataForwardOp)

  DataForwardOp() = default;

  explicit DataForwardOp(T& expected_value) : expected_value_(expected_value) {}

  void initialize() override {
    Operator::initialize();
    // Create and initialize the verification operator
    verification_op_ =
        fragment()->template make_operator<VerificationOp<T>>("verify_op", expected_value_);
    op_runner_ = std::make_shared<holoscan::ops::OperatorRunner>(verification_op_);
  }

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<T>("in");
    spec.output<T>("out");
  }

  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override {
    HOLOSCAN_LOG_INFO("{} - compute() called.", name());

    // Get input value
    auto value = op_input.receive<T>("in").value();

    // Use OperatorRunner to process the value
    auto result = op_runner_->push_input("in", value);
    if (!result) {
      HOLOSCAN_LOG_ERROR("Failed to push input to operator {} - {}",
                         op_runner_->op()->name(),
                         result.error().what());
      return;
    }
    op_runner_->run();
    auto output = op_runner_->pop_output<T>("out");

    // Forward the result
    if (output) { op_output.emit(output.value(), "out"); }
  }

 private:
  std::shared_ptr<VerificationOp<T>> verification_op_;
  std::shared_ptr<holoscan::ops::OperatorRunner> op_runner_;
  T expected_value_{};
};

// Generator operator for test data
template <typename T>
class TestDataGeneratorOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TestDataGeneratorOp)

  TestDataGeneratorOp() = default;

  explicit TestDataGeneratorOp(const T& value) : test_value_(value) {}

  void setup(holoscan::OperatorSpec& spec) override { spec.output<T>("out"); }

  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override {
    HOLOSCAN_LOG_INFO("{} - compute() called.", name());
    op_output.emit(test_value_, "out");
  }

 private:
  T test_value_;
};

// Test application template
template <typename T>
class OperatorRunnerTypeTestApp : public holoscan::Application {
 public:
  explicit OperatorRunnerTypeTestApp(const T& test_value) : test_value_(test_value) {}

  void compose() override {
    using namespace holoscan;

    // Create operators
    auto generator = make_operator<TestDataGeneratorOp<T>>("generator", test_value_);
    auto forwarder = make_operator<DataForwardOp<T>>("forwarder", test_value_);

    // Add flow
    add_flow(generator, forwarder);
  }

 private:
  T test_value_{};
};

// Test cases for different types
TEST(OperatorRunnerTypes, TestPrimitiveTypes) {
  // Test int
  {
    auto app = holoscan::make_application<OperatorRunnerTypeTestApp<int>>(42);
    app->run();
  }

  // Test float
  {
    auto app = holoscan::make_application<OperatorRunnerTypeTestApp<float>>(3.14f);
    app->run();
  }

  // Test double
  {
    auto app = holoscan::make_application<OperatorRunnerTypeTestApp<double>>(2.718);
    app->run();
  }
}

TEST(OperatorRunnerTypes, TestStdContainers) {
  // Test vector
  {
    std::vector<int> test_vec{1, 2, 3, 4, 5};
    auto app = holoscan::make_application<OperatorRunnerTypeTestApp<std::vector<int>>>(test_vec);
    app->run();
  }

  // Test string
  {
    std::string test_str{"test_string"};
    auto app = holoscan::make_application<OperatorRunnerTypeTestApp<std::string>>(test_str);
    app->run();
  }
}

TEST(OperatorRunnerTypes, TestSmartPointers) {
  // Test shared_ptr
  {
    auto test_ptr = std::make_shared<int>(42);
    auto app =
        holoscan::make_application<OperatorRunnerTypeTestApp<std::shared_ptr<int>>>(test_ptr);
    app->run();
  }
}

// Test custom type
struct CustomType {
  int value;
  std::string name;
  bool operator==(const CustomType& other) const {
    return value == other.value && name == other.name;
  }
};

TEST(OperatorRunnerTypes, TestCustomType) {
  CustomType test_data{42, "test"};
  auto app = holoscan::make_application<OperatorRunnerTypeTestApp<CustomType>>(test_data);
  app->run();
}

// Specialization for GXF Entity
template <>
class TestDataGeneratorOp<nvidia::gxf::Entity> : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TestDataGeneratorOp)

  TestDataGeneratorOp() = default;

  void setup(holoscan::OperatorSpec& spec) override { spec.output<nvidia::gxf::Entity>("out"); }

  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override {
    HOLOSCAN_LOG_INFO("{} - compute() called.", name());

    // Create a new GXF entity
    auto entity = nvidia::gxf::Entity::New(context.context());

    op_output.emit(std::move(entity), "out");
  }
};

template <>
class VerificationOp<nvidia::gxf::Entity> : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(VerificationOp)

  VerificationOp() = default;

  explicit VerificationOp(nvidia::gxf::Entity& expected_value) : expected_value_(expected_value) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<nvidia::gxf::Entity>("in");
    spec.output<nvidia::gxf::Entity>("out");
  }

  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override {
    HOLOSCAN_LOG_INFO("{} - compute() called.", name());

    auto entity = op_input.receive<nvidia::gxf::Entity>("in").value();

    // Verify entity contents
    EXPECT_NE(entity.eid(), kNullUid);

    // Forward the entity
    op_output.emit(entity, "out");
  }

 private:
  nvidia::gxf::Entity expected_value_;
};

class GXFEntityTestApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    auto generator = make_operator<TestDataGeneratorOp<nvidia::gxf::Entity>>("generator");
    auto forwarder = make_operator<DataForwardOp<nvidia::gxf::Entity>>("forwarder");
    add_flow(generator, forwarder);
  }
};

// Test GXF Entity handling
TEST(OperatorRunnerTypes, TestGXFEntity) {
  auto app = holoscan::make_application<GXFEntityTestApp>();
  app->run();
}

// Test TensorMap handling
TEST(OperatorRunnerTypes, TestTensorMap) {
  holoscan::TensorMap tensor_map;
  auto app = holoscan::make_application<OperatorRunnerTypeTestApp<holoscan::TensorMap>>(tensor_map);
  app->run();
}

// Specialization for std::any
template <>
class TestDataGeneratorOp<std::any> : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TestDataGeneratorOp)

  TestDataGeneratorOp() = default;

  void setup(holoscan::OperatorSpec& spec) override { spec.output<std::any>("out"); }

  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override {
    HOLOSCAN_LOG_INFO("{} - compute() called.", name());

    // Test different types in std::any
    std::vector<std::any> test_data;
    test_data.push_back(42);                         // int
    test_data.push_back(3.14);                       // double
    test_data.push_back(std::string("test"));        // string
    test_data.push_back(std::vector<int>{1, 2, 3});  // vector

    op_output.emit(std::move(test_data), "out");
  }
};

template <>
class VerificationOp<std::any> : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(VerificationOp)

  VerificationOp() = default;

  explicit VerificationOp(std::any& expected_value) : expected_value_(expected_value) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<std::any>("in");
    spec.output<std::any>("out");
  }

  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override {
    HOLOSCAN_LOG_INFO("{} - compute() called.", name());

    auto value = op_input.receive<std::any>("in").value();

    // Verify std::any contents
    auto& data = std::any_cast<const std::vector<std::any>&>(value);

    // Verify each type
    EXPECT_EQ(std::any_cast<int>(data[0]), 42);
    EXPECT_DOUBLE_EQ(std::any_cast<double>(data[1]), 3.14);
    EXPECT_EQ((std::any_cast<std::string>(data[2])), "test");
    EXPECT_EQ((std::any_cast<std::vector<int>>(data[3])), (std::vector<int>{1, 2, 3}));

    // Forward the data
    op_output.emit(std::move(value), "out");
  }

 private:
  std::any expected_value_;
};

class StdAnyTestApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    auto generator = make_operator<TestDataGeneratorOp<std::any>>("generator");
    auto forwarder = make_operator<DataForwardOp<std::any>>("forwarder");
    add_flow(generator, forwarder);
  }
};

// Test std::any handling
TEST(OperatorRunnerTypes, TestStdAny) {
  auto app = holoscan::make_application<StdAnyTestApp>();
  app->run();
}

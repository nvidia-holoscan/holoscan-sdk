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

#pragma once

#include <gtest/gtest.h>

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "holoscan/holoscan.hpp"

namespace holoscan::test {

/**
 * @brief Test data source operator providing predetermined test data
 *
 * This operator emits a sequence of values from a provided vector.
 * It is used to test operators that consume data from a source.
 * For operators with N input ports, the test application should create N source operators.
 *
 * @tparam T The type of the data to emit
 */
template<typename T>
class TestHarnessSourceOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TestHarnessSourceOp)

  TestHarnessSourceOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.output<T>("output");
  }

  void compute(InputContext&, OutputContext& op_output, ExecutionContext&) override {
    if (iteration_ >= test_data_.size()) {
      HOLOSCAN_LOG_DEBUG("[{}] TestHarnessSourceOp reached end of test data", name());
      return;
    }

    // Get the value and convert to actual type for logging (handles std::vector<bool> proxy issue)
    T value = test_data_[iteration_];
    op_output.emit(value, "output");
    iteration_++;
  }

  void set_test_data(const std::vector<T>& data) {
    test_data_ = data;
  }

 private:
  std::vector<T> test_data_;
  size_t iteration_ = 0;
};

/**
 * @brief Simple validation sink operator
 *
 * This operator receives data from the test operator and validates it against a set of
 * provided validation functions.
 * It is used to test operators that produce data to be consumed by another operator.
 * For operators with N output ports, the test application should create N sink operators.
 *
 * @tparam T The type of the data to receive
 */
template<typename T>
class TestHarnessSinkOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TestHarnessSinkOp)

  TestHarnessSinkOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<T>("input");
  }

  void compute(holoscan::InputContext& op_input, holoscan::OutputContext&,
               holoscan::ExecutionContext&) override {
    auto input_data = op_input.receive<T>("input");
    if (!input_data.has_value()) {
      if (!validators_.empty()) {
        FAIL() << "[" << name() << "] TestHarnessSinkOp received no value, but validators are set. "
               << "This indicates a test configuration error.";
      }
      HOLOSCAN_LOG_DEBUG("[{}] TestHarnessSinkOp received no value", name());
      return;
    }

    // Convert to actual type for logging (handles potential proxy types)
    T value = input_data.value();
    received_data_.push_back(value);

    // Run validation functions if provided
    if (!validators_.empty()) {
      for (auto& validator : validators_) {
        validator(value);
      }
    }
  }

  void set_validator(std::function<void(const T&)> validator) {
    validators_.clear();
    validators_.push_back(validator);
  }

  void set_validators(const std::vector<std::function<void(const T&)>>& validators) {
    validators_ = validators;
  }

  size_t get_received_count() const { return received_data_.size(); }

 private:
  std::vector<std::function<void(const T&)>> validators_;
  std::vector<T> received_data_;
};



// Helper function to create operator arguments tuple
template<typename... Args>
std::tuple<Args...> args(Args&&... arguments) {
  return std::make_tuple(std::forward<Args>(arguments)...);
}

// Helper function to create validators vector
template<typename OutputType, typename... Validators>
std::vector<std::function<void(const OutputType&)>> validators(Validators&&... validator_funcs) {
  return {std::function<void(const OutputType&)>(std::forward<Validators>(validator_funcs))...};
}

/**
 * @brief Test application wrapper for testing a Holoscan operator's `compute` method
 *
 * This class provides a test harness for testing a Holoscan operator's `compute` method.
 * It supports an arbitrary number of input and output ports with arbitrary types.
 * It also supports adding conditions to the operator under test.
 *
 * @tparam OperatorType The type of the operator to test
 * @tparam Args The types of the operator's arguments
 *
 */
template<typename OperatorType, typename... Args>
class OperatorTestHarness
    : public holoscan::Application,
      public std::enable_shared_from_this<OperatorTestHarness<OperatorType, Args...>> {
 public:
  explicit OperatorTestHarness(std::tuple<Args...> operator_args)
    : operator_args_(std::move(operator_args)) {}

  // Add input port with data
  template<typename T>
  std::shared_ptr<OperatorTestHarness> add_input_port(const std::string& name,
                                                      const std::vector<T>& data) {
    // Determine the number of data elements
    if (data_count_ == 0) {
      data_count_ = data.size();
    } else if (data.size() != data_count_) {
      throw std::runtime_error("All input ports must have the same number of data elements");
    }

    // Create the source operator, with CountCondition for the number of data elements
    input_port_creators_.emplace_back([this, name, data]() {
      auto source = this->template make_operator<TestHarnessSourceOp<T>>(
          name + "_source", this->template make_condition<holoscan::CountCondition>(data_count_));

      auto typed_source = std::dynamic_pointer_cast<TestHarnessSourceOp<T>>(source);
      if (typed_source) {
        typed_source->set_test_data(data);
      }
      this->add_operator(source);

      return std::make_pair(source, name);
    });
    return this->shared_from_this();
  }

  // Add output port with validators
  template<typename T>
  std::shared_ptr<OperatorTestHarness> add_output_port(const std::string& name,
        const std::vector<std::function<void(const T&)>>& validators = {}) {
    output_port_creators_.emplace_back([this, name, validators]() {
      auto sink = this->template make_operator<TestHarnessSinkOp<T>>(name + "_sink");

      auto typed_sink = std::dynamic_pointer_cast<TestHarnessSinkOp<T>>(sink);
      if (typed_sink && !validators.empty()) {
        typed_sink->set_validators(validators);
      }
      this->add_operator(sink);

      return std::make_pair(sink, name);
    });
    return this->shared_from_this();
  }

  // Add a condition to the operator under test
  template<typename ConditionType, typename... CondArgs>
  std::shared_ptr<OperatorTestHarness> add_condition(const std::string& name, CondArgs&&... args) {
    // Store condition creation for later execution in compose()
    condition_creators_.emplace_back(
        [this, name, args = std::make_tuple(std::forward<CondArgs>(args)...)]() {
          auto condition = std::apply(
              [this, &name](auto&&... captured_args) {
                return this->template make_condition<ConditionType>(
                    name, std::forward<decltype(captured_args)>(captured_args)...);
              },
              args);
          operator_under_test_->add_arg(condition);
        });
    return this->shared_from_this();
  }

  void compose() override {
    using namespace holoscan;

    // Create all input source operators
    for (auto& creator : input_port_creators_) {
      auto [source, port_name] = creator();
      input_sources_[port_name] = source;
    }

    // Create operator under test using just the operator args
    operator_under_test_ = std::apply(
        [this](auto&&... args) {
          return this->template make_operator<OperatorType>("operator_under_test",
                                                            std::forward<decltype(args)>(args)...);
        },
        operator_args_);
    add_operator(operator_under_test_);

    // Add conditions to the operator using add_arg
    for (auto& condition_creator : condition_creators_) {
      condition_creator();
    }

    // Create all output sink operators
    for (auto& creator : output_port_creators_) {
      auto [sink, port_name] = creator();
      output_sinks_[port_name] = sink;
    }

    // Connect all input sources to operator
    for (const auto& [port_name, source] : input_sources_) {
      add_flow(source, operator_under_test_, {{"output", port_name}});
    }

    // Connect operator to all output sinks
    for (const auto& [port_name, sink] : output_sinks_) {
      add_flow(operator_under_test_, sink, {{port_name, "input"}});
    }
  }

  // Get source by name with type casting
  template<typename T>
  std::shared_ptr<TestHarnessSourceOp<T>> get_source(const std::string& port_name) const {
    auto it = input_sources_.find(port_name);
    if (it != input_sources_.end()) {
      return std::dynamic_pointer_cast<TestHarnessSourceOp<T>>(it->second);
    }
    return nullptr;
  }

  // Get the operator under test with type casting
  template<typename T = OperatorType>
  std::shared_ptr<T> get_operator_under_test() const {
    return std::dynamic_pointer_cast<T>(operator_under_test_);
  }

  // Get sink by name with type casting
  template<typename T>
  std::shared_ptr<TestHarnessSinkOp<T>> get_sink(const std::string& port_name) const {
    auto it = output_sinks_.find(port_name);
    if (it != output_sinks_.end()) {
      return std::dynamic_pointer_cast<TestHarnessSinkOp<T>>(it->second);
    }
    return nullptr;
  }

  void run_test() {
    config();
    holoscan::Application::run();
  }

 private:
  std::tuple<Args...> operator_args_;
  size_t data_count_ = 0;

  // Type-erased port creators
  std::vector<std::function<std::pair<std::shared_ptr<holoscan::Operator>,
                                      std::string>()>> input_port_creators_;
  std::vector<std::function<std::pair<std::shared_ptr<holoscan::Operator>,
                                      std::string>()>> output_port_creators_;

  // Runtime storage of operators
  std::map<std::string, std::shared_ptr<holoscan::Operator>> input_sources_;
  std::map<std::string, std::shared_ptr<holoscan::Operator>> output_sinks_;

  std::shared_ptr<OperatorType> operator_under_test_;
  std::vector<std::function<void()>> condition_creators_;
};

// Helper function to create operator tests
template<typename OperatorType, typename... Args>
std::shared_ptr<OperatorTestHarness<OperatorType, std::decay_t<Args>...>>
create_operator_test(Args&&... args) {
  return std::make_shared<OperatorTestHarness<OperatorType, std::decay_t<Args>...>>(
    std::make_tuple(std::forward<Args>(args)...));
}

// Overload without operator arguments
template<typename OperatorType>
std::shared_ptr<OperatorTestHarness<OperatorType>>
create_operator_test() {
  return std::make_shared<OperatorTestHarness<OperatorType>>(
    std::make_tuple());
}

/**
 * @brief Base test fixture for operator unit tests
 *
 * Provides common setup and teardown functionality for operator tests.
 * Use this as a base class instead of directly inheriting from ::testing::Test
 * to ensure consistent test infrastructure across all operator tests.
 */
class OperatorTestBase : public ::testing::Test {
 protected:
  void SetUp() override {
    // Common test setup if needed
  }

  void TearDown() override {
    // Common test cleanup if needed
  }
};

}  // namespace holoscan::test

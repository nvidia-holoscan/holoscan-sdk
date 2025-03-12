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

#include "ping_variant_tx_native_op.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "holoscan/core/executors/gxf/gxf_executor.hpp"

using namespace holoscan;

// Register the custom type with Holoscan.
// NOLINTBEGIN(altera-struct-pack-align)
template <>
struct YAML::convert<void*> {
  static Node encode([[maybe_unused]] const void*& rhs) {
    throw std::runtime_error("void* is unsupported in YAML");
  }

  static bool decode([[maybe_unused]] const Node& node, [[maybe_unused]] void*& rhs) {
    throw std::runtime_error("void* is unsupported in YAML");
  }
};
// NOLINTEND(altera-struct-pack-align)

namespace myops {

class IntegerGeneratorOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(IntegerGeneratorOp)

  IntegerGeneratorOp() = default;

  void setup(holoscan::OperatorSpec& spec) override { spec.output<int>("out"); }

  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
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

  explicit ProcessingOp(std::function<int(int)> op_func)
      : Operator(), op_func_(std::move(op_func)) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<int>("in");
    spec.output<int>("out");
  }

  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    HOLOSCAN_LOG_INFO("{} - compute() called.", name());
    auto in_message = op_input.receive<int>("in").value();

    auto out_message = op_func_(in_message);

    op_output.emit(out_message, "out");
  }

 private:
  std::function<int(int)> op_func_;
};

void PingVarTxNativeOp::initialize() {
  HOLOSCAN_LOG_INFO("PingVarTxNativeOp::initialize() called.");
  // Register the custom type with Holoscan.
  register_converter<void*>();
  // Call the base class initialize.
  holoscan::Operator::initialize();
  HOLOSCAN_LOG_INFO(
      "PingVarTxNativeOp: custom_resource={}, numeric={}, "
      "numeric_array={}, optional_numeric={}, optional_numeric_array={}, "
      "boolean={}, optional_void_ptr={}, string={}, optional_resource={}",
      custom_resource_.get() ? "non-null" : "null",
      numeric_.get(),
      numeric_array_.get(),
      optional_numeric_.has_value() ? optional_numeric_.get() : -1,
      optional_numeric_array_.has_value() ? optional_numeric_array_.get() : std::vector<int>{},
      boolean_.get(),
      optional_void_ptr_.has_value() ? "non-null" : "null",
      string_.get(),
      optional_resource_.has_default_value() ? "non-null" : "null");

  // Setup OperatorRunner objects for the operators.
  auto* frag = fragment();
  auto op_int_generator = frag->make_operator<IntegerGeneratorOp>("int_generator");
  auto op_processing = frag->make_operator<ProcessingOp>("processing", [](int x) { return x * 2; });

  op_int_generator_ = std::make_shared<holoscan::ops::OperatorRunner>(op_int_generator);
  op_processing_ = std::make_shared<holoscan::ops::OperatorRunner>(op_processing);
}

void PingVarTxNativeOp::setup(OperatorSpec& spec) {
  HOLOSCAN_LOG_INFO("PingVarTxNativeOp::setup() called.");
  spec.output<holoscan::gxf::Entity>("out");

  spec.param(custom_resource_,
             "custom_resource",
             "CustomResource",
             "This is a sample parameter for a custom resource.");
  spec.param(numeric_, "numeric", "numeric", "numeric", 0);
  spec.param(numeric_array_,
             "numeric_array",
             "numeric array",
             "numeric array",
             std::vector<float>{0, 1.5, 2.5, 3.0, 4.0});
  spec.param(optional_numeric_,
             "optional_numeric",
             "optional numeric",
             "optional numeric",
             holoscan::ParameterFlag::kOptional);
  spec.param(optional_numeric_array_,
             "optional_numeric_array",
             "optional_numeric array",
             "optional_numeric array",
             holoscan::ParameterFlag::kOptional);
  spec.param(boolean_, "boolean", "boolean", "boolean");
  spec.param(optional_void_ptr_,
             "void_ptr",
             "optional void pointer",
             "optional void pointer",
             holoscan::ParameterFlag::kOptional);
  spec.param(string_, "string", "string", "string", std::string("test text"));
  spec.param(optional_resource_,
             "optional_resource",
             "optional resource",
             "optional resource",
             holoscan::ParameterFlag::kOptional);
}

void PingVarTxNativeOp::compute([[maybe_unused]] InputContext& op_input, OutputContext& op_output,
                                [[maybe_unused]] ExecutionContext& context) {
  HOLOSCAN_LOG_INFO("PingVarTxNativeOp::compute() called.");

  // Check if custom_resource_ is set and print a message.
  if (custom_resource_.get()) {
    HOLOSCAN_LOG_INFO(
        "PingVarTxNativeOp::compute() - custom_resource_ is set - custom_int: {}, float: {}",
        custom_resource_->get_custom_int(),
        custom_resource_->get_float());
  } else {
    HOLOSCAN_LOG_INFO("PingVarTxNativeOp::compute() - custom_resource_ is not set.");
  }

  // Run the int_generator operator.
  op_int_generator_->run();
  auto output_int_generator = op_int_generator_->pop_output("out");
  if (!output_int_generator) {
    HOLOSCAN_LOG_ERROR("PingVarTxNativeOp::compute() - int_generator operator failed to run.");
    auto error = output_int_generator.error();
    throw error;
  }
  auto input_processing = op_processing_->push_input("in", output_int_generator.value());
  if (!input_processing) {
    HOLOSCAN_LOG_ERROR("PingVarTxNativeOp::compute() - processing operator failed to run.");
    auto error = input_processing.error();
    throw error;
  }
  op_processing_->run();
  auto output_processing = op_processing_->pop_output("out");
  if (!output_processing) {
    HOLOSCAN_LOG_ERROR("PingVarTxNativeOp::compute() - processing operator failed to run.");
    auto error = output_processing.error();
    throw error;
  }
  op_output.emit(output_processing.value(), "out");
}

}  // namespace myops

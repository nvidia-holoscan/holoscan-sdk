/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_CORE_OPERATOR_HPP
#define HOLOSCAN_CORE_OPERATOR_HPP

#include "./common.hpp"

#include <stdio.h>
#include <iostream>
#include <memory>
#include <type_traits>

#include "./arg.hpp"
#include "./component.hpp"
#include "./condition.hpp"
#include "./forward_def.hpp"
#include "./operator_spec.hpp"
#include "./resource.hpp"

#define HOLOSCAN_OPERATOR_FORWARD_TEMPLATE()                                            \
  template <typename ArgT,                                                              \
            typename... ArgsT,                                                          \
            typename = std::enable_if_t<                                                \
                !std::is_base_of_v<Operator, std::decay_t<ArgT>> &&                     \
                (std::is_same_v<Arg, std::decay_t<ArgT>> ||                             \
                 std::is_same_v<ArgList, std::decay_t<ArgT>> ||                         \
                 std::is_base_of_v<holoscan::Condition,                                 \
                                   typename holoscan::type_info<ArgT>::element_type> || \
                 std::is_base_of_v<holoscan::Resource,                                  \
                                   typename holoscan::type_info<ArgT>::element_type>)>>

/**
 * @brief Forward the arguments to the super class.
 *
 * This macro is used to forward the arguments of the constructor to the base class. It is used in
 * the constructor of the operator class.
 *
 * Use this macro if the base class is a `holoscan::Operator`.
 *
 * Example:
 *
 * ```cpp
 * class GXFOperator : public holoscan::Operator {
 *  public:
 *   HOLOSCAN_OPERATOR_FORWARD_ARGS(GXFOperator)
 *
 *   GXFOperator() = default;
 *
 *   void initialize() override;
 *
 *   virtual const char* gxf_typename() const = 0;
 * };
 * ```
 *
 * @param class_name The name of the class.
 */
#define HOLOSCAN_OPERATOR_FORWARD_ARGS(class_name) \
  HOLOSCAN_OPERATOR_FORWARD_TEMPLATE()             \
  class_name(ArgT&& arg, ArgsT&&... args)          \
      : Operator(std::forward<ArgT>(arg), std::forward<ArgsT>(args)...) {}

/**
 * @brief Forward the arguments to the super class.
 *
 * This macro is used to forward the arguments of the constructor to the base class. It is used in
 * the constructor of the operator class.
 *
 * Use this macro if the class is derived from `holoscan::Operator` or the base class is derived
 * from `holoscan::Operator`.
 *
 * Example:
 *
 * ```cpp
 * class AJASourceOp : public holoscan::ops::GXFOperator {
 *  public:
 *   HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(AJASourceOp, holoscan::ops::GXFOperator)
 *
 *   AJASourceOp() = default;
 *
 *   const char* gxf_typename() const override { return "nvidia::holoscan::AJASource"; }
 *
 *   void setup(OperatorSpec& spec) override;
 *
 *   void initialize() override;
 * };
 * ```
 *
 * @param class_name The name of the class.
 * @param super_class_name The name of the super class.
 */
#define HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(class_name, super_class_name) \
  HOLOSCAN_OPERATOR_FORWARD_TEMPLATE()                                     \
  class_name(ArgT&& arg, ArgsT&&... args)                                  \
      : super_class_name(std::forward<ArgT>(arg), std::forward<ArgsT>(args)...) {}

namespace holoscan {

/**
 * @brief Base class for all operators.
 *
 * An operator is the most basic unit of work in Holoscan Embedded SDK. An Operator receives
 * streaming data at an input port, processes it, and publishes it to one of its output ports.
 *
 * This class is the base class for all operators. It provides the basic functionality for all
 * operators.
 *
 * @note This class is not intended to be used directly. Inherit from this class to create a new
 * operator.
 */
class Operator : public Component {
 public:
  /**
   * @brief Construct a new Operator object.
   *
   * @param args The arguments to be passed to the operator.
   */
  HOLOSCAN_OPERATOR_FORWARD_TEMPLATE()
  Operator(ArgT&& arg, ArgsT&&... args) {
    add_arg(std::forward<ArgT>(arg));
    (add_arg(std::forward<ArgsT>(args)), ...);
  }

  Operator() = default;

  ~Operator() override = default;

  using Component::name;
  /**
   * @brief Set the name of the operator.
   *
   * @param name The name of the operator.
   * @return The reference to this operator.
   */
  Operator& name(const std::string& name) {
    name_ = name;
    return *this;
  }

  using Component::fragment;
  /**
   * @brief Set the fragment of the operator.
   *
   * @param fragment The pointer to the fragment of the operator.
   * @return The reference to this operator.
   */
  Operator& fragment(Fragment* fragment) {
    fragment_ = fragment;
    return *this;
  }

  /**
   * @brief Set the operator spec.
   *
   * @param spec The operator spec.
   * @return The reference to this operator.
   */
  Operator& spec(std::unique_ptr<OperatorSpec> spec) {
    spec_ = std::move(spec);
    return *this;
  }
  /**
   * @brief Get the operator spec.
   *
   * @return The operator spec.
   */
  OperatorSpec* spec() { return spec_.get(); }

  /**
   * @brief Get the conditions of the operator.
   *
   * @return The conditions of the operator.
   */
  std::unordered_map<std::string, std::shared_ptr<Condition>>& conditions() { return conditions_; }
  /**
   * @brief Get the resources of the operator.
   *
   * @return The resources of the operator.
   */
  std::unordered_map<std::string, std::shared_ptr<Resource>>& resources() { return resources_; }

  using Component::add_arg;

  /**
   * @brief Add a condition to the operator.
   *
   * @param arg The condition to add.
   */
  void add_arg(const std::shared_ptr<Condition>& arg) { conditions_[arg->name()] = arg; }
  /**
   * @brief Add a condition to the operator.
   *
   * @param arg The condition to add.
   */
  void add_arg(std::shared_ptr<Condition>&& arg) { conditions_[arg->name()] = std::move(arg); }

  /**
   * @brief Add a resource to the operator.
   *
   * @param arg The resource to add.
   */
  void add_arg(const std::shared_ptr<Resource>& arg) { resources_[arg->name()] = arg; }
  /**
   * @brief Add a resource to the operator.
   *
   * @param arg The resource to add.
   */
  void add_arg(std::shared_ptr<Resource>&& arg) { resources_[arg->name()] = std::move(arg); }

  /**
   * @brief Define the operator specification.
   *
   * @param spec The reference to the operator specification.
   */
  virtual void setup(OperatorSpec& spec) { (void)spec; }

  /**
   * @brief Implement the startup logic of the operator.
   *
   * This method is called multiple times over the lifecycle of the operator according to the
   * order defined in the lifecycle, and used for heavy initialization tasks such as allocating
   * memory resources.
   */
  virtual void start(){
      // Empty default implementation
  };

  /**
   * @brief Implement the shutdown logic of the operator.
   *
   * This method is called multiple times over the lifecycle of the operator according to the
   * order defined in the lifecycle, and used for heavy deinitialization tasks such as deallocation
   * of all resources previously assigned in start.
   */
  virtual void stop(){
      // Empty default implementation
  };

  /**
   * @brief Implement the compute method.
   *
   * This method is called by the runtime multiple times. The runtime calls this method until
   * the operator is stopped.
   *
   * @param op_input The input context of the operator.
   * @param op_output The output context of the operator.
   * @param context The execution context of the operator.
   */
  virtual void compute(InputContext& op_input, OutputContext& op_output,
                       ExecutionContext& context) {
    (void)op_input;
    (void)op_output;
    (void)context;
  };

 protected:
  std::unique_ptr<OperatorSpec> spec_;  ///< The operator spec of the operator.
  std::unordered_map<std::string, std::shared_ptr<Condition>>
      conditions_;  ///< The conditions of the operator.
  std::unordered_map<std::string, std::shared_ptr<Resource>>
      resources_;  ///< The resources used by the operator.
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_OPERATOR_HPP */

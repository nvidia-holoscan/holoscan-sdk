/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_CORE_OPERATOR_SPEC_HPP
#define HOLOSCAN_CORE_OPERATOR_SPEC_HPP

#include <yaml-cpp/yaml.h>

#include <iostream>
#include <list>
#include <memory>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

#include "./common.hpp"
#include "./component_spec.hpp"
#include "./io_spec.hpp"
namespace holoscan {

struct MultiMessageConditionInfo {
  ConditionType kind;
  std::vector<std::string> port_names;
  ArgList args;
};

/**
 * @brief Class to define the specification of an operator.
 */
class OperatorSpec : public ComponentSpec {
 public:
  /**
   * @brief Construct a new OperatorSpec object.
   *
   * @param fragment The pointer to the fragment that contains this operator.
   */
  explicit OperatorSpec(Fragment* fragment = nullptr) : ComponentSpec(fragment) {}

  /**
   * @brief Get input specifications of this operator.
   *
   * @return The reference to the input specifications of this operator.
   */
  std::unordered_map<std::string, std::shared_ptr<IOSpec>>& inputs() { return inputs_; }

  /**
   * @brief Define an input specification for this operator.
   *
   * @tparam DataT The type of the input data.
   * @return The reference to the input specification.
   */
  template <typename DataT>
  IOSpec& input() {
    return input<DataT>("__iospec_input");
  }

  /**
   * @brief Add a Condition that depends on the status of multiple input ports.
   *
   * @param type The type of multi-message condition (currently only kMultiMessageAvailable)
   * @param port_names The names of the input ports the condition will apply to
   * @param args ArgList of arguments to pass to the MultiMessageAvailableCondition
   */
  void multi_port_condition(ConditionType kind, std::vector<std::string> port_names, ArgList args) {
    multi_port_conditions_.emplace_back(
        MultiMessageConditionInfo{kind, std::move(port_names), std::move(args)});
  }

  std::vector<MultiMessageConditionInfo>& multi_port_conditions() { return multi_port_conditions_; }

  /**
   * @brief Define an input specification for this operator.
   *
   * Note: The 'size' parameter is used for initializing the queue size of the input port. The
   *       queue size can be set by this method or by the 'IOSpec::queue_size(int64_t)' method.
   *       If the queue size is set to 'any size' (IOSpec::kAnySize in C++ or IOSpec.ANY_SIZE
   *       in Python), the connector/condition settings will be ignored.
   *       If the queue size is set to other values, the default connector
   *       (DoubleBufferReceiver/UcxReceiver) and condition (MessageAvailableCondition) will use
   *       the queue size for initialization ('capacity' for the connector and 'min_size' for
   *       the condition) if they are not set.
   *       Please refer to the [Holoscan SDK User
   * Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_create_operator.html#receiving-any-number-of-inputs-c)
   *       to see how to receive any number of inputs in C++.
   *
   * @tparam DataT The type of the input data.
   * @param name The name of the input specification.
   * @param size The size of the queue for the input port.
   * @return The reference to the input specification.
   */
  template <typename DataT>
  IOSpec& input(std::string name, IOSpec::IOSize size = IOSpec::kSizeOne) {
    if (size == IOSpec::kAnySize) {
      // Create receivers object
      receivers_params_.emplace_back();

      // Register parameter
      auto& parameter = receivers_params_.back();
      param(parameter, name.c_str(), "", "", {}, ParameterFlag::kNone);
    }

    auto spec = std::make_shared<IOSpec>(this, name, IOSpec::IOType::kInput, &typeid(DataT), size);
    auto [iter, is_exist] = inputs_.insert_or_assign(name, std::move(spec));
    if (!is_exist) { HOLOSCAN_LOG_ERROR("Input port '{}' already exists", name); }
    return *(iter->second.get());
  }

  /**
   * @brief Get output specifications of this operator.
   *
   * @return The reference to the output specifications of this operator.
   */
  std::unordered_map<std::string, std::shared_ptr<IOSpec>>& outputs() { return outputs_; }

  /**
   * @brief Define an output specification for this operator.
   *
   * @tparam DataT The type of the output data.
   * @return The reference to the output specification.
   */
  template <typename DataT>
  IOSpec& output() {
    return output<DataT>("__iospec_output");
  }

  /**
   * @brief Define an output specification for this operator.
   *
   * @tparam DataT The type of the output data.
   * @param name The name of the output specification.
   * @return The reference to the output specification.
   */
  template <typename DataT>
  IOSpec& output(std::string name) {
    auto spec = std::make_shared<IOSpec>(this, name, IOSpec::IOType::kOutput, &typeid(DataT));
    auto [iter, is_exist] = outputs_.insert_or_assign(name, std::move(spec));
    if (!is_exist) { HOLOSCAN_LOG_ERROR("Output port '{}' already exists", name); }
    return *(iter->second.get());
  }

  using ComponentSpec::param;

  /**
   * @brief Define an IOSpec* parameter for this operator.
   *
   * @param parameter The parameter to define.
   * @param key The key (name) of the parameter.
   * @param headline The headline of the parameter.
   * @param description The description of the parameter.
   * @param flag The flag of the parameter (default: ParameterFlag::kNone).
   */
  void param(Parameter<holoscan::IOSpec*>& parameter, const char* key, const char* headline,
             const char* description, ParameterFlag flag = ParameterFlag::kNone) {
    parameter.key_ = key;
    parameter.headline_ = headline;
    parameter.description_ = description;
    parameter.flag_ = flag;

    auto [_, is_exist] = params_.try_emplace(key, parameter);
    if (!is_exist) { HOLOSCAN_LOG_ERROR("Parameter '{}' already exists", key); }
  }

  /**
   * @brief Define an IOSpec* parameter for this operator.
   *
   * This method is to catch the following case:
   *
   * ```cpp
   * ...
   *     spec.param(iospec1_, "iospec1", "IO Spec", "Example IO Spec.", {});
   * ...
   * private:
   *  Parameter<holoscan::IOSpec*> iospec1_;
   * ```
   *
   * Otherwise, `{}` will be treated as `ParameterFlag::kNone` instead of `std::initializer_list`.
   *
   * @param parameter The parameter to define.
   * @param key The key (name) of the parameter.
   * @param headline The headline of the parameter.
   * @param description The description of the parameter.
   * @param init_list The initializer list of the parameter.
   */
  void param(Parameter<holoscan::IOSpec*>& parameter, const char* key, const char* headline,
             const char* description, [[maybe_unused]] std::initializer_list<void*> init_list) {
    parameter.key_ = key;
    parameter.headline_ = headline;
    parameter.description_ = description;
    // Set default value to nullptr
    parameter.default_value_ = static_cast<holoscan::IOSpec*>(nullptr);

    auto [_, is_exist] = params_.try_emplace(key, parameter);
    if (!is_exist) { HOLOSCAN_LOG_ERROR("Parameter '{}' already exists", key); }
  }

  /**
   * @brief Define an IOSpec* parameter with a default value for this operator.
   *
   * @param parameter The parameter to define.
   * @param key The key (name) of the parameter.
   * @param headline The headline of the parameter.
   * @param description The description of the parameter.
   * @param default_value The default value of the parameter.
   * @param flag The flag of the parameter (default: ParameterFlag::kNone).
   */
  void param(Parameter<holoscan::IOSpec*>& parameter, const char* key, const char* headline,
             const char* description, holoscan::IOSpec* default_value,
             ParameterFlag flag = ParameterFlag::kNone) {
    parameter.default_value_ = default_value;
    param(parameter, key, headline, description, flag);
  }

  /**
   * @brief Define a IOSpec* vector parameter for this operator.
   *
   * @param parameter The parameter to define.
   * @param key The key (name) of the parameter.
   * @param headline The headline of the parameter.
   * @param description The description of the parameter.
   * @param flag The flag of the parameter (default: ParameterFlag::kNone).
   */
  void param(Parameter<std::vector<holoscan::IOSpec*>>& parameter, const char* key,
             const char* headline, const char* description,
             ParameterFlag flag = ParameterFlag::kNone) {
    parameter.key_ = key;
    parameter.headline_ = headline;
    parameter.description_ = description;
    parameter.flag_ = flag;

    auto [_, is_exist] = params_.try_emplace(key, parameter);
    if (!is_exist) { HOLOSCAN_LOG_ERROR("Parameter '{}' already exists", key); }
  }

  /**
   * @brief Define a IOSpec* vector parameter for this operator.
   *
   * This method is to catch the following case:
   *
   * ```cpp
   * ...
   *     spec.param(iospec1_, "iospec1", "IO Spec", "Example IO Spec.", {});
   * ...
   * private:
   *  Parameter<std::vector<holoscan::IOSpec*>> iospec1_;
   * ```
   *
   * Otherwise, `{}` will be treated as `ParameterFlag::kNone` instead of `std::initializer_list`.
   *
   * @param parameter The parameter to define.
   * @param key The key (name) of the parameter.
   * @param headline The headline of the parameter.
   * @param description The description of the parameter.
   * @param init_list The initializer list of the parameter.
   */
  void param(Parameter<std::vector<holoscan::IOSpec*>>& parameter, const char* key,
             const char* headline, const char* description,
             std::initializer_list<holoscan::IOSpec*> init_list) {
    parameter.key_ = key;
    parameter.headline_ = headline;
    parameter.description_ = description;
    parameter.default_value_ = init_list;  // create a vector from initializer list

    auto [_, is_exist] = params_.try_emplace(key, parameter);
    if (!is_exist) { HOLOSCAN_LOG_ERROR("Parameter '{}' already exists", key); }
  }

  /**
   * @brief Define an IOSpec* parameter with a default value for this operator.
   *
   * @param parameter The parameter to define.
   * @param key The key (name) of the parameter.
   * @param headline The headline of the parameter.
   * @param description The description of the parameter.
   * @param default_value The default value of the parameter.
   * @param flag The flag of the parameter (default: ParameterFlag::kNone).
   */
  void param(Parameter<std::vector<holoscan::IOSpec*>>& parameter, const char* key,
             const char* headline, const char* description,
             std::vector<holoscan::IOSpec*> default_value,
             ParameterFlag flag = ParameterFlag::kNone) {
    parameter.default_value_ = default_value;
    param(parameter, key, headline, description, flag);
  }

  /**
   * @brief Get a YAML representation of the operator spec.
   *
   * @return YAML node including the inputs, outputs, and parameters of this operator.
   */
  YAML::Node to_yaml_node() const override;

 protected:
  std::unordered_map<std::string, std::shared_ptr<IOSpec>> inputs_;   ///< Input specs
  std::unordered_map<std::string, std::shared_ptr<IOSpec>> outputs_;  ///< Outputs specs

  // multi-message conditions span multiple IOSpec objects, so store them on OperatorSpec instead
  std::vector<MultiMessageConditionInfo> multi_port_conditions_;

  /// Container for receivers parameters
  std::list<Parameter<std::vector<IOSpec*>>> receivers_params_;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_OPERATOR_SPEC_HPP */

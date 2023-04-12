/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <iostream>
#include <memory>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

#include "./component_spec.hpp"
#include "./io_spec.hpp"
#include "./common.hpp"
namespace holoscan {

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
  std::unordered_map<std::string, std::unique_ptr<IOSpec>>& inputs() { return inputs_; }

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
   * @brief Define an input specification for this operator.
   *
   * @tparam DataT The type of the input data.
   * @param name The name of the input specification.
   * @return The reference to the input specification.
   */
  template <typename DataT>
  IOSpec& input(std::string name) {
    auto spec = std::make_unique<IOSpec>(this, name, IOSpec::IOType::kInput, &typeid(DataT));
    auto [iter, is_exist] = inputs_.insert_or_assign(name, std::move(spec));
    if (!is_exist) { HOLOSCAN_LOG_ERROR("Input port '{}' already exists", name); }
    if (outputs_.find(name) != outputs_.end()) {
      HOLOSCAN_LOG_WARN(
          "Output port name '{}' conflicts with the input port name '{}'", name, name);
    }
    return *(iter->second.get());
  }

  /**
   * @brief Get output specifications of this operator.
   *
   * @return The reference to the output specifications of this operator.
   */
  std::unordered_map<std::string, std::unique_ptr<IOSpec>>& outputs() { return outputs_; }

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
    auto spec = std::make_unique<IOSpec>(this, name, IOSpec::IOType::kOutput, &typeid(DataT));
    auto [iter, is_exist] = outputs_.insert_or_assign(name, std::move(spec));
    if (!is_exist) { HOLOSCAN_LOG_ERROR("Output port '{}' already exists", name); }
    if (inputs_.find(name) != inputs_.end()) {
      HOLOSCAN_LOG_WARN(
          "Input port name '{}' conflicts with the output port name '{}'", name, name);
    }
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
   */
  void param(Parameter<holoscan::IOSpec*>& parameter, const char* key, const char* headline,
             const char* description) {
    parameter.key_ = key;
    parameter.headline_ = headline;
    parameter.description_ = description;

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
   */
  void param(Parameter<holoscan::IOSpec*>& parameter, const char* key, const char* headline,
             const char* description, holoscan::IOSpec* default_value) {
    parameter.default_value_ = default_value;
    param(parameter, key, headline, description);
  }

  /**
   * @brief Define a IOSpec* vector parameter for this operator.
   *
   * @param parameter The parameter to define.
   * @param key The key (name) of the parameter.
   * @param headline The headline of the parameter.
   * @param description The description of the parameter.
   */
  void param(Parameter<std::vector<holoscan::IOSpec*>>& parameter, const char* key,
             const char* headline, const char* description) {
    parameter.key_ = key;
    parameter.headline_ = headline;
    parameter.description_ = description;

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
   */
  void param(Parameter<std::vector<holoscan::IOSpec*>>& parameter, const char* key,
             const char* headline, const char* description,
             std::vector<holoscan::IOSpec*> default_value) {
    parameter.default_value_ = default_value;
    param(parameter, key, headline, description);
  }

  /**
   * @brief Get a YAML representation of the operator spec.
   *
   * @return YAML node including the inputs, outputs, and parameters of this operator.
   */
  YAML::Node to_yaml_node() const override;

 protected:
  std::unordered_map<std::string, std::unique_ptr<IOSpec>> inputs_;   ///< Input specs
  std::unordered_map<std::string, std::unique_ptr<IOSpec>> outputs_;  ///< Outputs specs
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_OPERATOR_SPEC_HPP */

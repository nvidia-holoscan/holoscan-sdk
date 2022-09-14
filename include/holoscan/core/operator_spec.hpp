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

#ifndef HOLOSCAN_CORE_OPERATOR_SPEC_HPP
#define HOLOSCAN_CORE_OPERATOR_SPEC_HPP

#include "./common.hpp"

#include <iostream>
#include <memory>
#include <typeinfo>
#include <unordered_map>
#include <utility>

#include "./component_spec.hpp"
#include "./io_spec.hpp"

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
    if (!is_exist) { HOLOSCAN_LOG_ERROR("Input item '{}' already exists", name); }
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
    // TODO: implement this
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
    if (!is_exist) { HOLOSCAN_LOG_ERROR("Output item '{}' already exists", name); }
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
   * @param default_value The default value of the parameter.
   */
  void param(Parameter<holoscan::IOSpec*>& parameter, const char* key, const char* headline,
             const char* description, holoscan::IOSpec* default_value) {
    parameter.key_ = key;
    parameter.headline_ = headline;
    parameter.description_ = description;

    parameter.default_value_ = default_value;

    auto [_, is_exist] = params_.try_emplace(key, parameter);
    if (!is_exist) { HOLOSCAN_LOG_ERROR("Parameter '{}' already exists", key); }
  }

  /**
   * @brief Define a vector of IOSpec* parameter for this operator.
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
    parameter.key_ = key;
    parameter.headline_ = headline;
    parameter.description_ = description;

    parameter.default_value_ = default_value;

    auto [_, is_exist] = params_.try_emplace(key, parameter);
    if (!is_exist) { HOLOSCAN_LOG_ERROR("Parameter '{}' already exists", key); }
  }

  /**
   * @brief Define a resource that is required by this operator.
   *
   * @tparam ResourceT The type of the resource.
   * @param name The name of the resource.
   * @param args The arguments to construct the resource.
   */
  template <typename ResourceT, typename StringT, typename... ArgsT,
            typename = std::enable_if_t<std::is_constructible_v<std::string, StringT>>>
  void resource(const StringT& name, ArgsT&&... args) {
    resources_.emplace(name, std::make_shared<ResourceT>(std::forward<ArgsT>(args)...));
  }

  /**
   * @brief Define a resource that is required by this operator.
   *
   * @param args The arguments to construct the resource.
   */
  template <typename... ArgsT>
  void resource(ArgsT&&... args) {
    resource("", std::forward<ArgsT>(args)...);
  }

  /**
   * @brief Define a condition that is required by this operator.
   *
   * @tparam ConditionT The type of the condition.
   * @param name The name of the condition.
   * @param args The arguments to construct the condition.
   */
  template <typename ConditionT, typename StringT, typename... ArgsT,
            typename = std::enable_if_t<std::is_constructible_v<std::string, StringT>>>
  void condition(const StringT& name, ArgsT&&... args) {
    conditions_.emplace(name, std::make_shared<ConditionT>(std::forward<ArgsT>(args)...));
  }

  /**
   * @brief Define a condition that is required by this operator.
   *
   * @param args The arguments to construct the condition.
   */
  template <typename ConditionT, typename... ArgsT>
  void condition(ArgsT&&... args) {
    condition("", std::forward<ArgsT>(args)...);
  }

 protected:
  std::unordered_map<std::string, std::unique_ptr<IOSpec>> inputs_;         ///< Input specs
  std::unordered_map<std::string, std::unique_ptr<IOSpec>> outputs_;        ///< Outputs specs
  // TODO(gbae): use these for operator spec.
  std::unordered_map<std::string, std::shared_ptr<Condition>> conditions_;  ///< Conditions
  std::unordered_map<std::string, std::shared_ptr<Resource>> resources_;    ///< Resources
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_OPERATOR_SPEC_HPP */

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

#ifndef HOLOSCAN_CORE_COMPONENT_SPEC_HPP
#define HOLOSCAN_CORE_COMPONENT_SPEC_HPP

#include <yaml-cpp/yaml.h>

#include <any>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <utility>

#include "./common.hpp"
#include "./parameter.hpp"

namespace holoscan {

/**
 * @brief Class to define the specification of a component.
 */
class ComponentSpec {
 public:
  /**
   * @brief Construct a new ComponentSpec object.
   *
   * @param fragment The pointer to the fragment that contains this component.
   */
  explicit ComponentSpec(Fragment* fragment = nullptr) : fragment_(fragment) {}

  /**
   * @brief Set the pointer to the fragment that contains this component.
   *
   * @param fragment The pointer to the fragment that contains this component.
   */
  void fragment(Fragment* fragment) { fragment_ = fragment; }

  /**
   * @brief Get the pointer to the fragment that contains this component.
   *
   * @return The pointer to the fragment that contains this component.
   */
  Fragment* fragment() { return fragment_; }

  /**
   * @brief Define a parameter for this component.
   *
   * @tparam typeT The type of the parameter.
   * @param parameter The parameter to define.
   * @param key The key (name) of the parameter.
   * @param flag The flag of the parameter (default: ParameterFlag::kNone).
   */
  template <typename typeT>
  void param(Parameter<typeT>& parameter, const char* key,
             ParameterFlag flag = ParameterFlag::kNone) {
    param(parameter, key, "N/A", "N/A", flag);
  }

  /**
   * @brief Define a parameter for this component.
   *
   * @tparam typeT The type of the parameter.
   * @param parameter The parameter to define.
   * @param key The key (name) of the parameter.
   * @param headline The headline of the parameter.
   * @param flag The flag of the parameter (default: ParameterFlag::kNone).
   */
  template <typename typeT>
  void param(Parameter<typeT>& parameter, const char* key, const char* headline,
             ParameterFlag flag = ParameterFlag::kNone) {
    param(parameter, key, headline, "N/A", flag);
  }

  /**
   * @brief Define a parameter for this component.
   *
   * @tparam typeT The type of the parameter.
   * @param parameter The parameter to define.
   * @param key The key (name) of the parameter.
   * @param headline The headline of the parameter.
   * @param description The description of the parameter.
   * @param flag The flag of the parameter (default: ParameterFlag::kNone).
   */
  template <typename typeT>
  void param(Parameter<typeT>& parameter, const char* key, const char* headline,
             const char* description, ParameterFlag flag = ParameterFlag::kNone);

  /**
   * @brief Define a parameter for this component.
   *
   * This method is to catch the following case:
   *
   * ```cpp
   * ...
   *     spec.param(int64_value_, "int64_param", "int64_t param", "Example int64_t parameter.", {});
   * ...
   * private:
   *  Parameter<int64_t> int64_param_;
   * ```
   *
   * Otherwise, `{}` will be treated as `ParameterFlag::kNone` instead of `std::initializer_list`.
   *
   * @tparam typeT The type of the parameter.
   * @param parameter The parameter to define.
   * @param key The key (name) of the parameter.
   * @param headline The headline of the parameter.
   * @param description The description of the parameter.
   * @param init_list The initializer list of the parameter.
   */
  template <typename typeT>
  void param(Parameter<typeT>& parameter, const char* key, const char* headline,
             const char* description, std::initializer_list<void*> init_list);

  /**
   * @brief Define a parameter that has a default value.
   *
   * @tparam typeT The type of the parameter.
   * @param parameter The parameter to get.
   * @param key The key (name) of the parameter.
   * @param headline The headline of the parameter.
   * @param description The description of the parameter.
   * @param default_value The default value of the parameter.
   * @param flag The flag of the parameter (default: ParameterFlag::kNone).
   */
  template <typename typeT>
  void param(Parameter<typeT>& parameter, const char* key, const char* headline,
             const char* description, const typeT& default_value,
             ParameterFlag flag = ParameterFlag::kNone);

  /**
   * @brief Define a parameter that has a default value.
   *
   * @tparam typeT The type of the parameter.
   * @param parameter The parameter to get.
   * @param key The key (name) of the parameter.
   * @param headline The headline of the parameter.
   * @param description The description of the parameter.
   * @param default_value The default value of the parameter.
   * @param flag The flag of the parameter (default: ParameterFlag::kNone).
   */
  template <typename typeT>
  void param(Parameter<typeT>& parameter, const char* key, const char* headline,
             const char* description, typeT&& default_value,
             ParameterFlag flag = ParameterFlag::kNone);

  /**
   * @brief Get the parameters of this component.
   *
   * @return The reference to the parameters of this component.
   */
  std::unordered_map<std::string, ParameterWrapper>& params() { return params_; }

  /**
   * @brief Get a YAML representation of the component spec.
   *
   * @return YAML node including the parameters of this component.
   */
  virtual YAML::Node to_yaml_node() const;

  /**
   * @brief Get a description of the component spec.
   *
   * @see to_yaml_node()
   * @return YAML string.
   */
  std::string description() const;

 protected:
  Fragment* fragment_ = nullptr;  ///< The pointer to the fragment that contains this component.
  std::unordered_map<std::string, ParameterWrapper> params_;  ///< The parameters of this component.
};

}  // namespace holoscan

// ------------------------------------------------------------------------------------------------
// Template definitions
//
//   Since the template definitions depends on template methods in other headers, we declare the
//   template methods above, and define them below with the proper header files, so that we don't
//   have circular dependencies.
// ------------------------------------------------------------------------------------------------
#include "./component_spec-inl.hpp"

#endif /* HOLOSCAN_CORE_COMPONENT_SPEC_HPP */

/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_CORE_ARGUMENT_SETTER_HPP
#define HOLOSCAN_CORE_ARGUMENT_SETTER_HPP

#include <yaml-cpp/yaml.h>

#include <any>
#include <complex>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

#include <common/logger.hpp>

#include "../utils/yaml_parser.hpp"
#include "./arg.hpp"
#include "./common.hpp"
#include "./parameter.hpp"
#include "./type_traits.hpp"

namespace holoscan {

// Forward declarations
class Condition;
class Resource;

/**
 * @brief Class to set arguments for components.
 *
 * This class is used to set arguments for components (including Operator, Condition, and Resource)
 * from Arg or ArgList.
 */
class ArgumentSetter {
 public:
  /**
   * @brief Function type for setting an argument to the parameter.
   */
  using SetterFunc = std::function<void(ParameterWrapper&, Arg&)>;

  /**
   * @brief Default @ref SetterFunc for Arg.
   */
  inline static SetterFunc none_argument_setter = [](ParameterWrapper& /*param_wrap*/, Arg& arg) {
    HOLOSCAN_LOG_ERROR("Unable to handle parameter: {}", arg.name());
  };

  /**
   * @brief Get the instance object.
   *
   * @return The reference to the ArgumentSetter instance.
   */
  static ArgumentSetter& get_instance();

  /**
   * @brief Set the param object.
   *
   * @param param_wrap The ParameterWrapper object.
   * @param arg The Arg object to set.
   */
  static void set_param(ParameterWrapper& param_wrap, Arg& arg) {
    auto& instance = get_instance();
    const std::type_index index = std::type_index(param_wrap.type());
    const SetterFunc& func = instance.get_argument_setter(index);
    func(param_wrap, arg);
  }

  /**
   * @brief Register the SetterFunc for the type.
   *
   * @tparam typeT The type of the parameter.
   */
  template <typename typeT>
  static void ensure_type() {
    auto& instance = get_instance();
    instance.add_argument_setter<typeT>();
  }

  /**
   * @brief Get the argument setter function object.
   *
   * @param index The type index of the parameter.
   * @return The reference to the SetterFunc object.
   */
  SetterFunc& get_argument_setter(std::type_index index) {
    if (function_map_.find(index) == function_map_.end()) {
      HOLOSCAN_LOG_WARN("No argument setter for type '{}' exists", index.name());
      return ArgumentSetter::none_argument_setter;
    }

    auto& handler = function_map_[index];
    return handler;
  }

  /**
   * @brief Add the SetterFunc for the type.
   *
   * @tparam typeT typeT The type of the parameter.
   * @param func The SetterFunc object.
   */
  template <typename typeT>
  void add_argument_setter(SetterFunc func) {
    function_map_.try_emplace(std::type_index(typeid(typeT)), func);
  }

  /**
   * @brief Add the SetterFunc for the type.
   *
   * @param index The type index of the parameter.
   * @param func The SetterFunc object.
   */
  void add_argument_setter(std::type_index index, SetterFunc func) {
    function_map_.try_emplace(index, func);
  }

  /**
   * @brief Add the SetterFunc for the type.
   *
   * @tparam typeT The type of the parameter.
   */
  template <typename typeT>
  void add_argument_setter();

 private:
  ArgumentSetter() {
    add_argument_setter<bool>();
    add_argument_setter<int8_t>();
    add_argument_setter<int16_t>();
    add_argument_setter<int32_t>();
    add_argument_setter<int64_t>();
    add_argument_setter<uint8_t>();
    add_argument_setter<uint16_t>();
    add_argument_setter<uint32_t>();
    add_argument_setter<uint64_t>();
    add_argument_setter<float>();
    add_argument_setter<double>();
    add_argument_setter<std::complex<float>>();
    add_argument_setter<std::complex<double>>();
    add_argument_setter<std::string>();
    add_argument_setter<std::vector<bool>>();
    add_argument_setter<std::vector<int8_t>>();
    add_argument_setter<std::vector<int16_t>>();
    add_argument_setter<std::vector<int32_t>>();
    add_argument_setter<std::vector<int64_t>>();
    add_argument_setter<std::vector<uint8_t>>();
    add_argument_setter<std::vector<uint16_t>>();
    add_argument_setter<std::vector<uint32_t>>();
    add_argument_setter<std::vector<uint64_t>>();
    add_argument_setter<std::vector<float>>();
    add_argument_setter<std::vector<double>>();
    add_argument_setter<std::vector<std::complex<float>>>();
    add_argument_setter<std::vector<std::complex<double>>>();
    add_argument_setter<std::vector<std::string>>();
    add_argument_setter<std::vector<std::vector<bool>>>();
    add_argument_setter<std::vector<std::vector<int8_t>>>();
    add_argument_setter<std::vector<std::vector<int16_t>>>();
    add_argument_setter<std::vector<std::vector<int32_t>>>();
    add_argument_setter<std::vector<std::vector<int64_t>>>();
    add_argument_setter<std::vector<std::vector<uint8_t>>>();
    add_argument_setter<std::vector<std::vector<uint16_t>>>();
    add_argument_setter<std::vector<std::vector<uint32_t>>>();
    add_argument_setter<std::vector<std::vector<uint64_t>>>();
    add_argument_setter<std::vector<std::vector<float>>>();
    add_argument_setter<std::vector<std::vector<double>>>();
    add_argument_setter<std::vector<std::vector<std::complex<float>>>>();
    add_argument_setter<std::vector<std::vector<std::complex<double>>>>();
    add_argument_setter<std::vector<std::vector<std::string>>>();

    add_argument_setter<YAML::Node>();
    add_argument_setter<holoscan::IOSpec*>();
    add_argument_setter<std::vector<holoscan::IOSpec*>>();

    add_argument_setter<std::shared_ptr<Resource>>();
    add_argument_setter<std::vector<std::shared_ptr<Resource>>>();

    add_argument_setter<std::shared_ptr<Condition>>();
    add_argument_setter<std::vector<std::shared_ptr<Condition>>>();
  }

  std::unordered_map<std::type_index, SetterFunc> function_map_;  ///< Map of type index to setter
                                                                  ///< function
};

}  // namespace holoscan

// ------------------------------------------------------------------------------------------------
// Template definitions
//
//   Since the template definitions depends on template methods in other headers, we declare the
//   template methods above, and define them below with the proper header files, so that we don't
//   have circular dependencies.
// ------------------------------------------------------------------------------------------------
#include "./argument_setter-inl.hpp"

#endif /* HOLOSCAN_CORE_ARGUMENT_SETTER_HPP */

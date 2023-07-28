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

#ifndef HOLOSCAN_CORE_COMPONENT_HPP
#define HOLOSCAN_CORE_COMPONENT_HPP

#include <stdio.h>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "./parameter.hpp"
#include "./type_traits.hpp"
#include "./arg.hpp"
#include "./forward_def.hpp"

#define HOLOSCAN_COMPONENT_FORWARD_TEMPLATE()                                                \
  template <typename ArgT,                                                                   \
            typename... ArgsT,                                                               \
            typename = std::enable_if_t<!std::is_base_of_v<Component, std::decay_t<ArgT>> && \
                                        (std::is_same_v<Arg, std::decay_t<ArgT>> ||          \
                                         std::is_same_v<ArgList, std::decay_t<ArgT>>)>>
#define HOLOSCAN_COMPONENT_FORWARD_ARGS(class_name) \
  HOLOSCAN_COMPONENT_FORWARD_TEMPLATE()             \
  class_name(ArgT&& arg, ArgsT&&... args)           \
      : Component(std::forward<ArgT>(arg), std::forward<ArgsT>(args)...)

#define HOLOSCAN_COMPONENT_FORWARD_ARGS_SUPER(class_name, super_class_name) \
  HOLOSCAN_COMPONENT_FORWARD_TEMPLATE()                                     \
  class_name(ArgT&& arg, ArgsT&&... args)                                   \
      : super_class_name(std::forward<ArgT>(arg), std::forward<ArgsT>(args)...)

namespace holoscan {

/**
 * @brief Base class for all components.
 *
 * This class is the base class for all components including `holoscan::Operator`,
 * `holoscan::Condition`, and `holoscan::Resource`.
 * It is used to define the common interface for all components.
 */
class Component {
 public:
  Component() = default;

  /**
   * @brief Construct a new Component object.
   *
   * @param args The arguments to be passed to the component.
   */
  HOLOSCAN_COMPONENT_FORWARD_TEMPLATE()
  explicit Component(ArgT&& arg, ArgsT&&... args) {
    add_arg(std::forward<ArgT>(arg));
    (add_arg(std::forward<ArgsT>(args)), ...);
  }

  virtual ~Component() = default;

  /**
   * @brief Get the identifier of the component.
   *
   * By default, the identifier is set to -1.
   * It is set to a valid value when the component is initialized.
   *
   * With the default executor (GXFExecutor), the identifier is set to the GXF component ID.
   *
   * @return The identifier of the component.
   */
  int64_t id() const { return id_; }

  /**
   * @brief Get the name of the component.
   *
   * @return The name of the component.
   */
  const std::string& name() const { return name_; }

  /**
   * @brief Get a pointer to Fragment object.
   *
   * @return The Pointer to Fragment object.
   */
  Fragment* fragment() { return fragment_; }

  /**
   * @brief Add an argument to the component.
   *
   * @param arg The argument to add.
   */
  void add_arg(const Arg& arg) { args_.emplace_back(arg); }
  /**
   * @brief Add an argument to the component.
   *
   * @param arg The argument to add.
   */
  void add_arg(Arg&& arg) { args_.emplace_back(std::move(arg)); }

  /**
   * @brief Add a list of arguments to the component.
   *
   * @param arg The list of arguments to add.
   */
  void add_arg(const ArgList& arg) {
    args_.reserve(args_.size() + arg.size());
    args_.insert(args_.end(), arg.begin(), arg.end());
  }
  /**
   * @brief Add a list of arguments to the component.
   *
   * @param arg The list of arguments to add.
   */
  void add_arg(ArgList&& arg) {
    args_.reserve(args_.size() + arg.size());
    args_.insert(
        args_.end(), std::make_move_iterator(arg.begin()), std::make_move_iterator(arg.end()));
    arg.clear();
  }

  /**
   * @brief Get the list of arguments.
   *
   * @return The vector of arguments.
   */
  std::vector<Arg>& args() { return args_; }

  /**
   * @brief Initialize the component.
   *
   * This method is called only once when the component is created for the first time, and use of
   * light-weight initialization.
   */
  virtual void initialize() {}

  /**
   * @brief Get a YAML representation of the component.
   *
   * @return YAML node including the id, name, fragment name, and arguments of the component.
   */
  virtual YAML::Node to_yaml_node() const;

  /**
   * @brief Get a description of the component.
   *
   * @see to_yaml_node()
   * @return YAML string.
   */
  std::string description() const;

 protected:
  friend class Executor;

  int64_t id_ = -1;               ///< The ID of the component.
  std::string name_ = "";         ///< Name of the component
  Fragment* fragment_ = nullptr;  ///< Pointer to the fragment that owns this component
  std::vector<Arg> args_;         ///< List of arguments
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_COMPONENT_HPP */

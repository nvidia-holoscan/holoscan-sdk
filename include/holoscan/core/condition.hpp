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

#ifndef HOLOSCAN_CORE_CONDITION_HPP
#define HOLOSCAN_CORE_CONDITION_HPP

#include <gxf/core/gxf.h>

#include <any>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <utility>

#include "./common.hpp"
#include "./component.hpp"
#include "./gxf/gxf_component.hpp"
#include "./gxf/gxf_utils.hpp"

#define HOLOSCAN_CONDITION_FORWARD_TEMPLATE()                                                \
  template <typename ArgT,                                                                   \
            typename... ArgsT,                                                               \
            typename = std::enable_if_t<!std::is_base_of_v<Condition, std::decay_t<ArgT>> && \
                                        (std::is_same_v<Arg, std::decay_t<ArgT>> ||          \
                                         std::is_same_v<ArgList, std::decay_t<ArgT>>)>>

/**
 * @brief Forward the arguments to the super class.
 *
 * This macro is used to forward the arguments of the constructor to the base class. It is used in
 * the constructor of the condition class.
 *
 * Use this macro if the base class is a `holoscan::Condition`.
 *
 * @param class_name The name of the class.
 */
#define HOLOSCAN_CONDITION_FORWARD_ARGS(class_name) \
  HOLOSCAN_CONDITION_FORWARD_TEMPLATE()             \
  class_name(ArgT&& arg, ArgsT&&... args)           \
      : Condition(std::forward<ArgT>(arg), std::forward<ArgsT>(args)...) {}

/**
 * @brief Forward the arguments to the super class.
 *
 * This macro is used to forward the arguments of the constructor to the base class. It is used in
 * the constructor of the condition class.
 *
 * Use this macro if the class is derived from `holoscan::Condition` or the base class is derived
 * from `holoscan::Condition`.
 *
 * Example:
 *
 * ```cpp
 * class MessageAvailableCondition : public gxf::GXFCondition {
 *  public:
 *   HOLOSCAN_CONDITION_FORWARD_ARGS_SUPER(MessageAvailableCondition, GXFCondition)
 *   MessageAvailableCondition() = default;
 *   ...
 *   const char* gxf_typename() const override {
 *     return "nvidia::gxf::MessageAvailableSchedulingTerm";
 *   }
 *   ...
 *   void setup(ComponentSpec& spec) override;
 *
 *   void initialize() override { GXFCondition::initialize(); }
 *
 * };
 * ```
 *
 * @param class_name The name of the class.
 * @param super_class_name The name of the super class.
 */
#define HOLOSCAN_CONDITION_FORWARD_ARGS_SUPER(class_name, super_class_name) \
  HOLOSCAN_CONDITION_FORWARD_TEMPLATE()                                     \
  class_name(ArgT&& arg, ArgsT&&... args)                                   \
      : super_class_name(std::forward<ArgT>(arg), std::forward<ArgsT>(args)...) {}

namespace holoscan {

enum class ConditionType {
  kNone,              ///< No condition
  kMessageAvailable,  ///< Default for input port (nvidia::gxf::MessageAvailableSchedulingTerm)
  kDownstreamMessageAffordable,  ///< Default for output port
                                 ///< (nvidia::gxf::DownstreamReceptiveSchedulingTerm)
  kCount,                        ///< nvidia::gxf::CountSchedulingTerm
  kBoolean,                      ///< nvidia::gxf::BooleanSchedulingTerm
  kPeriodic,                     ///< nvidia::gxf::PeriodicSchedulingTerm
  kAsynchronous,                 ///< nvidia::gxf::AsynchronousSchedulingTerm
};

/**
 * @brief Base class for all conditions.
 *
 * A condition is a predicate that can be evaluated at runtime to determine if an operator should
 * execute. This matches the semantics of GXF's Scheduling Term.
 */
class Condition : public Component {
 public:
  Condition() = default;

  Condition(Condition&&) = default;

  /**
   * @brief Construct a new Condition object.
   */
  HOLOSCAN_CONDITION_FORWARD_TEMPLATE()
  explicit Condition(ArgT&& arg, ArgsT&&... args) {
    add_arg(std::forward<ArgT>(arg));
    (add_arg(std::forward<ArgsT>(args)), ...);
  }

  ~Condition() override = default;

  using Component::name;
  /**
   * @brief Set the name of the condition.
   *
   * @param name The name of the condition.
   * @return The reference to the condition.
   */
  Condition& name(const std::string& name) & {
    name_ = name;
    return *this;
  }

  /**
   * @brief Set the name of the condition.
   *
   * @param name The name of the condition.
   * @return The reference to the condition.
   */
  Condition&& name(const std::string& name) && {
    name_ = name;
    return std::move(*this);
  }

  using Component::fragment;

  /**
   * @brief Set the fragment of the condition.
   *
   * @param fragment The pointer to the fragment of the condition.
   * @return The reference to the condition.
   */
  Condition& fragment(Fragment* fragment) {
    fragment_ = fragment;
    return *this;
  }

  /**
   * @brief Set the component specification to the condition.
   *
   * @param spec The component specification.
   * @return The reference to the condition.
   */
  Condition& spec(const std::shared_ptr<ComponentSpec>& spec) {
    spec_ = spec;
    return *this;
  }
  /**
   * @brief Get the component specification of the condition.
   *
   * @return The pointer to the component specification.
   */
  ComponentSpec* spec() { return spec_.get(); }

  /**
   * @brief Get the shared pointer to the component spec.
   *
   * @return The shared pointer to the component spec.
   */
  std::shared_ptr<ComponentSpec> spec_shared() { return spec_; }

  using Component::add_arg;

  /**
   * @brief Define the condition specification.
   *
   * @param spec The reference to the component specification.
   */
  virtual void setup(ComponentSpec& spec) { (void)spec; }

  /**
   * @brief Get a YAML representation of the condition.
   *
   * @return YAML node including spec of the condition in addition to the base component properties.
   */
  YAML::Node to_yaml_node() const override;

 protected:
  std::shared_ptr<ComponentSpec> spec_;  ///< The component specification.
  bool is_initialized_ = false;          ///< Whether the condition is initialized.
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_CONDITION_HPP */

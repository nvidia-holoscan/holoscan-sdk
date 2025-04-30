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

#ifndef HOLOSCAN_CORE_CONDITION_HPP
#define HOLOSCAN_CORE_CONDITION_HPP

#include <gxf/core/gxf.h>
#include <yaml-cpp/yaml.h>

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
#include "gxf/std/scheduling_condition.hpp"
#include "gxf/std/scheduling_terms.hpp"  // for AsynchronousEventState

#define HOLOSCAN_CONDITION_FORWARD_TEMPLATE()                                                     \
  template <typename ArgT,                                                                        \
            typename... ArgsT,                                                                    \
            typename =                                                                            \
                std::enable_if_t<!std::is_base_of_v<::holoscan::Condition, std::decay_t<ArgT>> && \
                                 (std::is_same_v<::holoscan::Arg, std::decay_t<ArgT>> ||          \
                                  std::is_same_v<::holoscan::ArgList, std::decay_t<ArgT>>)>>

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
  explicit class_name(ArgT&& arg, ArgsT&&... args)  \
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
  explicit class_name(ArgT&& arg, ArgsT&&... args)                          \
      : super_class_name(std::forward<ArgT>(arg), std::forward<ArgsT>(args)...) {}

namespace holoscan {

// Forward declarations
class Operator;
class Resource;

/// Make AsynchronousEventState available to the holoscan namespace.
/// This enum is used for controlling and querying the state of asynchronous conditions.
/// Moving it to the holoscan namespace improves usability when working with async operators.
/// See Operator::async_condition() for accessing the internal condition to control execution
/// timing.
using nvidia::gxf::AsynchronousEventState;

// Note: Update `IOSpec::to_yaml_node()` if you add new condition types
enum class ConditionType {
  kNone,              ///< No condition
  kMessageAvailable,  ///< Default for input port (nvidia::gxf::MessageAvailableSchedulingTerm)
  kDownstreamMessageAffordable,   ///< Default for output port
                                  ///< (nvidia::gxf::DownstreamReceptiveSchedulingTerm)
  kCount,                         ///< nvidia::gxf::CountSchedulingTerm
  kBoolean,                       ///< nvidia::gxf::BooleanSchedulingTerm
  kPeriodic,                      ///< nvidia::gxf::PeriodicSchedulingTerm
  kAsynchronous,                  ///< nvidia::gxf::AsynchronousSchedulingTerm
  kExpiringMessageAvailable,      ///< nvidia::gxf::ExpiringMessageAvailableSchedulingTerm
  kMultiMessageAvailable,         ///< nvidia::gxf::MultiMessageAvailableSchedulingTerm
  kMultiMessageAvailableTimeout,  ///< nvidia::gxf::MessageAvailableFrequencyThrottler
};

enum class SchedulingStatusType : int32_t {
  kNever,      ///< Will never execute again
  kReady,      ///< Ready to execute now
  kWait,       ///< Will execute again at some point in the future
  kWaitTime,   ///< Will execute after a certain known time interval. Negative or zero interval will
               ///< result in immediate execution.
  kWaitEvent,  ///< Waiting for an event with unknown interval time. Entity will be put in a waiting
               ///< queue until event done notification is signalled
};

/**
 * @brief Base class for all conditions.
 *
 * A condition is a predicate that can be evaluated at runtime to determine if an operator should
 * execute. This matches the semantics of GXF's Scheduling Term.
 */
class Condition : public Component {
 public:
  /**
   * @brief Resource type used for the initialization of the resource.
   */
  enum class ConditionComponentType {
    kNative,  ///< Native condition.
    kGXF,     ///< GXF condition (scheduling term).
  };

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

  /**
   * @brief Check the condition status before allowing execution.
   *
   * If the condition is waiting for a time event 'target_timestamp' will contain the target
   * timestamp.
   *
   * @param timestamp The current timestamp
   * @param type The status of the condition
   * @param target_timestamp The target timestamp (used if the term is waiting for a time event).
   */
  virtual void check([[maybe_unused]] int64_t timestamp,
                     [[maybe_unused]] SchedulingStatusType* status_type,
                     [[maybe_unused]] int64_t* target_timestamp) const {
    // empty implementation (only used for native Conditions)
    throw std::logic_error("check method not implemented");
  }

  /**
   * @brief Called each time after the entity of this term was executed.
   *
   * @param timestamp The current timestamp
   */
  virtual void on_execute([[maybe_unused]] int64_t timestamp) {
    // empty implementation (only used for native Conditions)
    throw std::logic_error("on_execute method not implemented");
  }

  /**
   * @brief Checks if the state of the condition can be updated and updates it
   *
   * @param timestamp The current timestamp
   */
  virtual void update_state([[maybe_unused]] int64_t timestamp) {
    // empty implementation (only used for native Conditions)
  }

  /**
   * @brief Get the condition type.
   *
   * @return The condition type.
   */
  ConditionComponentType condition_type() const { return condition_type_; }

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
  ComponentSpec* spec() {
    if (!spec_) {
      HOLOSCAN_LOG_WARN("ComponentSpec of Condition '{}' is not initialized, returning nullptr",
                        name_);
      return nullptr;
    }
    return spec_.get();
  }

  /**
   * @brief Get the shared pointer to the component spec.
   *
   * @return The shared pointer to the component spec.
   */
  std::shared_ptr<ComponentSpec> spec_shared() { return spec_; }

  using Component::add_arg;

  /**
   * @brief Add a resource to the condition.
   *
   * @param arg The resource to add.
   */
  void add_arg(const std::shared_ptr<Resource>& arg);

  /**
   * @brief Add a resource to the condition.
   *
   * @param arg The resource to add.
   */
  void add_arg(std::shared_ptr<Resource>&& arg);

  /**
   * @brief Get the resources of the condition.
   *
   * @return The resources of the condition.
   */
  std::unordered_map<std::string, std::shared_ptr<Resource>>& resources() { return resources_; }

  /**
   * @brief Define the condition specification.
   *
   * @param spec The reference to the component specification.
   */
  virtual void setup([[maybe_unused]] ComponentSpec& spec) {}

  void initialize() override;

  /**
   * @brief Get a YAML representation of the condition.
   *
   * @return YAML node including spec of the condition in addition to the base component properties.
   */
  YAML::Node to_yaml_node() const override;

  /**@brief Return the Receiver corresponding to a specific input port of the Operator associated
   * with this condition.
   *
   * @param port_name The name of the input port.
   * @return The Receiver corresponding to the input port, if it exists. Otherwise, return nullopt.
   */
  std::optional<std::shared_ptr<Receiver>> receiver(const std::string& port_name);

  /**@brief Return the Transmitter corresponding to a specific output port of the Operator
   * associated with this condition.
   *
   * @param port_name The name of the output port.
   * @return The Transmitter corresponding to the output port, if it exists. Otherwise, return
   * nullopt.
   */
  std::optional<std::shared_ptr<Transmitter>> transmitter(const std::string& port_name);

  /**
   * Get the GXF component ID of the underlying GXFSchedulingTermWrapper
   *
   * This method is only relevant for native conditions. For conditions
   * inheriting from GXFCondition, please use GXFCondition::gxf_cid() instead.
   *
   * @return The unique GXF component id for this condition.
   */
  gxf_uid_t wrapper_cid() const;

 protected:
  // Add friend classes that can call reset_graph_entites
  friend class holoscan::Operator;

  using Component::reset_graph_entities;

  using ComponentBase::update_params_from_args;

  /**
   * @brief Set the Operator this condition is associated with.
   *
   * @param op The pointer to the Operator object.
   */
  void set_operator(Operator* op) { op_ = op; }

  /**
   * Store GXF component ID of underlying GXFSchedulingTermWrapper
   *
   * @param GXF component id.
   */
  void wrapper_cid(gxf_uid_t cid);

  /// Update parameters based on the specified arguments
  void update_params_from_args();

  /// Set the parameters based on defaults (sets GXF parameters for GXF components)
  virtual void set_parameters();

  bool is_initialized_ = false;  ///< Whether the condition is initialized.

  std::unordered_map<std::string, std::shared_ptr<Resource>>
      resources_;  ///< The resources used by the condition.
  ConditionComponentType condition_type_ =
      ConditionComponentType::kNative;  ///< The type of the component.
  Operator* op_ = nullptr;              ///< The operator this condition is associated with.
  gxf_uid_t wrapper_cid_ = 0;           ///< Component ID of underlying GXFSchedulingTermWrapper
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_CONDITION_HPP */

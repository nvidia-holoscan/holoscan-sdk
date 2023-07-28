/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_CORE_SCHEDULER_HPP
#define HOLOSCAN_CORE_SCHEDULER_HPP

#include <stdio.h>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "./parameter.hpp"
#include "./type_traits.hpp"
#include "./arg.hpp"
#include "./forward_def.hpp"
#include "./component.hpp"
#include "./resource.hpp"

#define HOLOSCAN_SCHEDULER_FORWARD_TEMPLATE()                                                \
  template <typename ArgT,                                                                   \
            typename... ArgsT,                                                               \
            typename = std::enable_if_t<!std::is_base_of_v<Scheduler, std::decay_t<ArgT>> && \
                                        (std::is_same_v<Arg, std::decay_t<ArgT>> ||          \
                                         std::is_same_v<ArgList, std::decay_t<ArgT>>)>>

/**
 * @brief Forward the arguments to the super class.
 *
 * This macro is used to forward the arguments of the constructor to the base class. It is used in
 * the constructor of the scheduler class.
 *
 * Use this macro if the base class is a `holoscan::Scheduler`.
 *
 * @param class_name The name of the class.
 */
#define HOLOSCAN_SCHEDULER_FORWARD_ARGS(class_name) \
  HOLOSCAN_SCHEDULER_FORWARD_TEMPLATE()             \
  class_name(ArgT&& arg, ArgsT&&... args)           \
      : Scheduler(std::forward<ArgT>(arg), std::forward<ArgsT>(args)...) {}

/**
 * @brief Forward the arguments to the super class.
 *
 * This macro is used to forward the arguments of the constructor to the base class. It is used in
 * the constructor of the scheduler class.
 *
 * Use this macro if the class is derived from `holoscan::Scheduler` or the base class is derived
 * from `holoscan::Scheduler`.
 *
 * @param class_name The name of the class.
 * @param super_class_name The name of the super class.
 */
#define HOLOSCAN_SCHEDULER_FORWARD_ARGS_SUPER(class_name, super_class_name) \
  HOLOSCAN_SCHEDULER_FORWARD_TEMPLATE()                                     \
  class_name(ArgT&& arg, ArgsT&&... args)                                   \
      : super_class_name(std::forward<ArgT>(arg), std::forward<ArgsT>(args)...) {}

namespace holoscan {

enum class SchedulerType { kDefault, kGreedy, kMultiThread };

/**
 * @brief Base class for all schedulers.
 *
 * This class is the base class for all schedulers including `holoscan::MultiThreadScheduler` and
 * `holoscan::GreedyScheduler`.
 * It is used to define the common interface for all schedulers.
 */
class Scheduler : public Component {
 public:
  Scheduler() = default;

  Scheduler(Scheduler&&) = default;

  /**
   * @brief Construct a new Scheduler object.
   */
  HOLOSCAN_SCHEDULER_FORWARD_TEMPLATE()
  explicit Scheduler(ArgT&& arg, ArgsT&&... args) {
    add_arg(std::forward<ArgT>(arg));
    (add_arg(std::forward<ArgsT>(args)), ...);
  }

  ~Scheduler() override = default;

  using Component::id;
  /**
   * @brief Set the Scheduler ID.
   *
   * @param id The ID of the scheduler.
   * @return The reference to this scheduler.
   */
  Scheduler& id(int64_t id) {
    id_ = id;
    return *this;
  }

  using holoscan::Component::name;

  /**
   * @brief Set the name of the scheduler.
   *
   * @param name The name of the scheduler.
   * @return The reference to the scheduler.
   */
  Scheduler& name(const std::string& name) & {
    name_ = name;
    return *this;
  }

  /**
   * @brief Set the name of the scheduler.
   *
   * @param name The name of the scheduler.
   * @return The reference to the scheduler.
   */
  Scheduler&& name(const std::string& name) && {
    name_ = name;
    return std::move(*this);
  }

  using holoscan::Component::fragment;

  /**
   * @brief Set the fragment of the scheduler.
   *
   * @param fragment The pointer to the fragment of the scheduler.
   * @return The reference to the scheduler.
   */
  Scheduler& fragment(Fragment* fragment) {
    fragment_ = fragment;
    return *this;
  }

  /**
   * @brief Set the component specification to the scheduler.
   *
   * @param spec The component specification.
   * @return The reference to the scheduler.
   */
  Scheduler& spec(const std::shared_ptr<ComponentSpec>& spec) {
    spec_ = spec;
    return *this;
  }

  /**
   * @brief Get the component specification of the scheduler.
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
   * @brief Add a resource to the scheduler.
   *
   * @param arg The resource to add.
   */
  void add_arg(const std::shared_ptr<Resource>& arg) {
    if (resources_.find(arg->name()) != resources_.end()) {
      HOLOSCAN_LOG_ERROR(
          "Resource '{}' already exists in the scheduler. Please specify a unique "
          "name when creating a Resource instance.",
          arg->name());
    } else {
      resources_[arg->name()] = arg;
    }
  }

  /**
   * @brief Add a resource to the scheduler.
   *
   * @param arg The resource to add.
   */
  void add_arg(std::shared_ptr<Resource>&& arg) {
    if (resources_.find(arg->name()) != resources_.end()) {
      HOLOSCAN_LOG_ERROR(
          "Resource '{}' already exists in the scheduler. Please specify a unique "
          "name when creating a Resource instance.",
          arg->name());
    } else {
      resources_[arg->name()] = std::move(arg);
    }
  }

  /**
   * @brief Get the resources of the scheduler.
   *
   * @return The resources of the scheduler.
   */
  std::unordered_map<std::string, std::shared_ptr<Resource>>& resources() { return resources_; }

  /**
   * @brief Define the scheduler specification.
   *
   * @param spec The reference to the component specification.
   */
  virtual void setup(ComponentSpec& spec) { (void)spec; }

  /**
   * @brief Initialize the scheduler.
   *
   * This function is called after the scheduler is created by
   * holoscan::Fragment::make_scheduler().
   */
  void initialize() override;

 protected:
  std::shared_ptr<ComponentSpec> spec_;  ///< The component specification.

  std::unordered_map<std::string, std::shared_ptr<Resource>>
      resources_;  ///< The resources used by the scheduler.
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_SCHEDULER_HPP */

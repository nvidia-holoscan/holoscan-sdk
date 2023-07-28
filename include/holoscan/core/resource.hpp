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

#ifndef HOLOSCAN_CORE_RESOURCE_HPP
#define HOLOSCAN_CORE_RESOURCE_HPP

#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include "./component.hpp"
#include "./gxf/gxf_component.hpp"
#include "./gxf/gxf_utils.hpp"

#define HOLOSCAN_RESOURCE_FORWARD_TEMPLATE()                                                \
  template <typename ArgT,                                                                  \
            typename... ArgsT,                                                              \
            typename = std::enable_if_t<!std::is_base_of_v<Resource, std::decay_t<ArgT>> && \
                                        (std::is_same_v<Arg, std::decay_t<ArgT>> ||         \
                                         std::is_same_v<ArgList, std::decay_t<ArgT>>)>>

/**
 * @brief Forward the arguments to the super class.
 *
 * This macro is used to forward the arguments of the constructor to the base class. It is used in
 * the constructor of the resource class.
 *
 * Use this macro if the base class is a `holoscan::Resource`.
 *
 * @param class_name The name of the class.
 */
#define HOLOSCAN_RESOURCE_FORWARD_ARGS(class_name) \
  HOLOSCAN_RESOURCE_FORWARD_TEMPLATE()             \
  class_name(ArgT&& arg, ArgsT&&... args)          \
      : Resource(std::forward<ArgT>(arg), std::forward<ArgsT>(args)...) {}

/**
 * @brief Forward the arguments to the super class.
 *
 * This macro is used to forward the arguments of the constructor to the base class. It is used in
 * the constructor of the resource class.
 *
 * Use this macro if the class is derived from `holoscan::Resource` or the base class is derived
 * from `holoscan::Resource`.
 *
 * Example:
 *
 * ```cpp
 * class Allocator : public gxf::GXFResource {
 *  public:
 *   HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(Allocator, GXFResource)
 *   Allocator() = default;
 *
 *   const char* gxf_typename() const override { return "nvidia::gxf::Allocator"; }
 *
 *   ...
 * };
 * ```
 *
 * @param class_name The name of the class.
 * @param super_class_name The name of the super class.
 */
#define HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(class_name, super_class_name) \
  HOLOSCAN_RESOURCE_FORWARD_TEMPLATE()                                     \
  class_name(ArgT&& arg, ArgsT&&... args)                                  \
      : super_class_name(std::forward<ArgT>(arg), std::forward<ArgsT>(args)...) {}

namespace holoscan {

/**
 * @brief Base class for all resources.
 *
 * Resources such as system memory or a GPU memory pool that an Operator needs to perform its job.
 * Resources are allocated during the initialization phase of the application. This matches the
 * semantics of GXF's Memory Allocator or any other components derived from the Component class in
 * GXF.
 */
class Resource : public Component {
 public:
  /**
   * @brief Resource type used for the initialization of the resource.
   */
  enum class ResourceType {
    kNative,  ///< Native resource.
    kGXF,     ///< GXF resource.
  };

  Resource() = default;

  Resource(Resource&&) = default;

  /**
   * @brief Construct a new Resource object.
   *
   * @param args The arguments to be passed to the resource.
   */
  HOLOSCAN_RESOURCE_FORWARD_TEMPLATE()
  explicit Resource(ArgT&& arg, ArgsT&&... args) {
    add_arg(std::forward<ArgT>(arg));
    (add_arg(std::forward<ArgsT>(args)), ...);
  }

  ~Resource() override = default;

  /**
   * @brief Get the resource type.
   *
   * @return The resource type.
   */
  ResourceType resource_type() const { return resource_type_; }

  using Component::name;
  /**
   * @brief Set the name of the resource.
   *
   * @param name The name of the resource.
   * @return The reference to the resource.
   */
  Resource& name(const std::string& name) & {
    name_ = name;
    return *this;
  }

  /**
   * @brief Set the name of the resource.
   *
   * @param name The name of the resource.
   * @return The reference to the resource.
   */
  Resource&& name(const std::string& name) && {
    name_ = name;
    return std::move(*this);
  }

  using Component::fragment;
  /**
   * @brief Set the fragment of the resource.
   *
   * @param fragment The pointer to the fragment of the resource.
   * @return The reference to the resource.
   */
  Resource& fragment(Fragment* fragment) {
    fragment_ = fragment;
    return *this;
  }

  /**
   * @brief Set the component specification to the resource.
   *
   * @param spec The component specification.
   * @return The reference to the resource.
   */
  Resource& spec(const std::shared_ptr<ComponentSpec>& spec) {
    spec_ = spec;
    return *this;
  }
  /**
   * @brief Get the component specification of the resource.
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
   * @brief Define the resource specification.
   *
   * @param spec The reference to the component specification.
   */
  virtual void setup(ComponentSpec& spec) { (void)spec; }

  void initialize() override;

  /**
   * @brief Get a YAML representation of the resource.
   *
   * @return YAML node including spec of the resource in addition to the base component properties.
   */
  YAML::Node to_yaml_node() const override;

 protected:
  ResourceType resource_type_ = ResourceType::kNative;  ///< The type of the resource.
  std::shared_ptr<ComponentSpec> spec_;                 ///< The component specification.
  bool is_initialized_ = false;                         ///< Whether the resource is initialized.
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCE_HPP */

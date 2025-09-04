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

#ifndef HOLOSCAN_CORE_COMPONENT_HPP
#define HOLOSCAN_CORE_COMPONENT_HPP

#include <yaml-cpp/yaml.h>

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

#define HOLOSCAN_COMPONENT_FORWARD_TEMPLATE()                                        \
  template <typename ArgT,                                                           \
            typename... ArgsT,                                                       \
            typename = std::enable_if_t<                                             \
                !std::is_base_of_v<::holoscan::ComponentBase, std::decay_t<ArgT>> && \
                (std::is_same_v<::holoscan::Arg, std::decay_t<ArgT>> ||              \
                 std::is_same_v<::holoscan::ArgList, std::decay_t<ArgT>>)>>
#define HOLOSCAN_COMPONENT_FORWARD_ARGS(class_name) \
  HOLOSCAN_COMPONENT_FORWARD_TEMPLATE()             \
  explicit class_name(ArgT&& arg, ArgsT&&... args)  \
      : Component(std::forward<ArgT>(arg), std::forward<ArgsT>(args)...)

#define HOLOSCAN_COMPONENT_FORWARD_ARGS_SUPER(class_name, super_class_name) \
  HOLOSCAN_COMPONENT_FORWARD_TEMPLATE()                                     \
  explicit class_name(ArgT&& arg, ArgsT&&... args)                          \
      : super_class_name(std::forward<ArgT>(arg), std::forward<ArgsT>(args)...)

namespace holoscan {

namespace gxf {
class GXFExecutor;
}  // namespace gxf

/**
 * @brief Base class for all components.
 *
 * This class is the base class for all components including `holoscan::Operator`,
 * `holoscan::Condition`, and `holoscan::Resource`.
 * It is used to define the common interface for all components.
 */
class ComponentBase {
 public:
  ComponentBase() = default;

  /**
   * @brief Construct a new Component object.
   *
   * @param arg The first argument to be passed to the component.
   * @param args The remaining arguments to be passed to the component.
   */
  HOLOSCAN_COMPONENT_FORWARD_TEMPLATE()
  explicit ComponentBase(ArgT&& arg, ArgsT&&... args) {
    add_arg(std::forward<ArgT>(arg));
    (add_arg(std::forward<ArgsT>(args)), ...);
  }

  virtual ~ComponentBase() = default;

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
   * @brief Get a const pointer to Fragment object.
   *
   * @return The const pointer to Fragment object.
   */
  const Fragment* fragment() const { return fragment_; }

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
   * @brief Register the argument setter for the given type.
   *
   * If an operator or resource has an argument with a custom type, the argument setter must be
   * registered using this method.
   *
   * The argument setter is used to set the value of the argument from the YAML configuration.
   *
   * This method can be called in the initialization phase of the operator/resource
   * (e.g., `initialize()`).
   * The example below shows how to register the argument setter for the custom type (`Vec3`):
   *
   * ```cpp
   * void MyOp::initialize() {
   *   register_converter<Vec3>();
   * }
   * ```
   *
   * It is assumed that `YAML::convert<T>::encode` and `YAML::convert<T>::decode` are implemented
   * for the given type.
   * You need to specialize the `YAML::convert<>` template class.
   *
   * For example, suppose that you had a `Vec3` class with the following members:
   *
   * ```cpp
   * struct Vec3 {
   *   // make sure you have overloaded operator==() for the comparison
   *   double x, y, z;
   * };
   * ```
   *
   * You can define the `YAML::convert<Vec3>` as follows in a '.cpp' file:
   *
   * ```cpp
   * namespace YAML {
   * template<>
   * struct convert<Vec3> {
   *   static Node encode(const Vec3& rhs) {
   *     Node node;
   *     node.push_back(rhs.x);
   *     node.push_back(rhs.y);
   *     node.push_back(rhs.z);
   *     return node;
   *   }
   *
   *   static bool decode(const Node& node, Vec3& rhs) {
   *     if(!node.IsSequence() || node.size() != 3) {
   *       return false;
   *     }
   *
   *     rhs.x = node[0].as<double>();
   *     rhs.y = node[1].as<double>();
   *     rhs.z = node[2].as<double>();
   *     return true;
   *   }
   * };
   * }
   * ```
   *
   * Please refer to the [yaml-cpp
   * documentation](https://github.com/jbeder/yaml-cpp/wiki/Tutorial#converting-tofrom-native-data-types)
   * for more details.
   *
   * @tparam typeT The type of the argument to register.
   */
  template <typename typeT>
  static void register_converter() {
    register_argument_setter<typeT>();
  }

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

  /**
   * @brief Retrieve a registered fragment service or resource.
   *
   * Retrieves a previously registered fragment service or resource by its type and optional
   * identifier. Returns nullptr if no service/resource is found with the specified type and
   * identifier.
   *
   * Note that any changes to the service retrieval logic in this method should be synchronized with
   * the implementation in `Fragment::service()` method to maintain consistency.
   *
   * @tparam ServiceT The type of the service/resource to retrieve. Must inherit from either
   * Resource or FragmentService. Defaults to DefaultFragmentService if
   * not specified.
   * @param id The identifier of the service/resource. If empty, retrieves by type only.
   * @return The shared pointer to the service/resource, or nullptr if not found or if type casting
   * fails.
   */
  template <typename ServiceT = DefaultFragmentService>
  std::shared_ptr<ServiceT> service(std::string_view id = "") const;

  /**
   * @brief Retrieve a registered fragment service or resource for Python bindings.
   *
   * This is a helper method for Python bindings to retrieve a service by its C++ type info.
   *
   * @param service_type The type info of the service/resource to retrieve.
   * @param id The identifier of the service/resource. If empty, retrieves by type only.
   * @return The shared pointer to the base service, or nullptr if not found.
   */
  std::shared_ptr<FragmentService> get_service_by_type_info(const std::type_info& service_type,
                                                            std::string_view id = "") const;

  /// Reset any backend-specific objects (e.g. GXF GraphEntity)
  virtual void reset_backend_objects();

 protected:
  // Make Fragment a friend class so it can call `fragment` and `service_provider`
  friend class holoscan::Fragment;

  /**
   * @brief Register the argument setter for the given type.
   *
   * Please refer to the documentation of `register_converter()` for more details.
   *
   * @tparam typeT The type of the argument to register.
   */
  template <typename typeT>
  static void register_argument_setter();

  /// Update parameters based on the specified arguments
  void update_params_from_args(std::unordered_map<std::string, ParameterWrapper>& params);

  /// Set the fragment that owns this component
  void fragment(Fragment* frag);

  /// Set the service provider that owns this component
  void service_provider(FragmentServiceProvider* provider);

  int64_t id_ = -1;               ///< The ID of the component.
  std::string name_ = "";         ///< Name of the component
  Fragment* fragment_ = nullptr;  ///< Pointer to the fragment that owns this component
  std::vector<Arg> args_;         ///< List of arguments
  FragmentServiceProvider* service_provider_ = nullptr;  ///< Pointer to the service provider
};

/**
 * @brief Common class for all non-Operator components
 *
 * This class is the base class for all non-Operator components including
 * `holoscan::Condition`, `holoscan::Resource`, `holoscan::NetworkContext`, `holoscan::Scheduler`
 * It is used to define the common interface for all components.
 *
 * `holoscan::Operator` does not inherit from this class as it uses `holoscan::OperatorSpec`
 * instead of `holoscan::ComponentSpec`.
 */
class Component : public ComponentBase {
 public:
  /// Set the parameters based on defaults (sets GXF parameters for GXF operators)
  virtual void set_parameters() {}

 protected:
  using ComponentBase::update_params_from_args;

  /// Update parameters based on the specified arguments
  void update_params_from_args();

  std::shared_ptr<ComponentSpec> spec_;  ///< The component specification.
};

}  // namespace holoscan

// ------------------------------------------------------------------------------------------------
// Template definitions
//
//   Since the template definitions depends on template methods in other headers, we declare the
//   template methods above, and define them below with the proper header files, so that we don't
//   have circular dependencies.
// ------------------------------------------------------------------------------------------------
#include "./component-inl.hpp"

#endif /* HOLOSCAN_CORE_COMPONENT_HPP */

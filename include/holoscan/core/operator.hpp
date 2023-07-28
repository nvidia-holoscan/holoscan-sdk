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

#ifndef HOLOSCAN_CORE_OPERATOR_HPP
#define HOLOSCAN_CORE_OPERATOR_HPP

#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "./arg.hpp"
#include "./argument_setter.hpp"
#include "./codec_registry.hpp"
#include "./common.hpp"
#include "./component.hpp"
#include "./condition.hpp"
#include "./forward_def.hpp"
#include "./messagelabel.hpp"
#include "./operator_spec.hpp"
#include "./resource.hpp"

#define HOLOSCAN_OPERATOR_FORWARD_TEMPLATE()                                            \
  template <typename ArgT,                                                              \
            typename... ArgsT,                                                          \
            typename = std::enable_if_t<                                                \
                !std::is_base_of_v<holoscan::Operator, std::decay_t<ArgT>> &&           \
                (std::is_same_v<holoscan::Arg, std::decay_t<ArgT>> ||                   \
                 std::is_same_v<holoscan::ArgList, std::decay_t<ArgT>> ||               \
                 std::is_base_of_v<holoscan::Condition,                                 \
                                   typename holoscan::type_info<ArgT>::derived_type> || \
                 std::is_base_of_v<holoscan::Resource,                                  \
                                   typename holoscan::type_info<ArgT>::derived_type>)>>

/**
 * @brief Forward the arguments to the super class.
 *
 * This macro is used to forward the arguments of the constructor to the base class. It is used in
 * the constructor of the operator class.
 *
 * Use this macro if the base class is a `holoscan::Operator`.
 *
 * Example:
 *
 * ```cpp
 * class GXFOperator : public holoscan::Operator {
 *  public:
 *   HOLOSCAN_OPERATOR_FORWARD_ARGS(GXFOperator)
 *
 *   GXFOperator() = default;
 *
 *   void initialize() override;
 *
 *   virtual const char* gxf_typename() const = 0;
 * };
 * ```
 *
 * @param class_name The name of the class.
 */
#define HOLOSCAN_OPERATOR_FORWARD_ARGS(class_name) \
  HOLOSCAN_OPERATOR_FORWARD_TEMPLATE()             \
  class_name(ArgT&& arg, ArgsT&&... args)          \
      : Operator(std::forward<ArgT>(arg), std::forward<ArgsT>(args)...) {}

/**
 * @brief Forward the arguments to the super class.
 *
 * This macro is used to forward the arguments of the constructor to the base class. It is used in
 * the constructor of the operator class.
 *
 * Use this macro if the class is derived from `holoscan::Operator` or the base class is derived
 * from `holoscan::Operator`.
 *
 * Example:
 *
 * ```cpp
 * class AJASourceOp : public holoscan::ops::GXFOperator {
 *  public:
 *   HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(AJASourceOp, holoscan::ops::GXFOperator)
 *
 *   AJASourceOp() = default;
 *
 *   const char* gxf_typename() const override { return "nvidia::holoscan::AJASource"; }
 *
 *   void setup(OperatorSpec& spec) override;
 *
 *   void initialize() override;
 * };
 * ```
 *
 * @param class_name The name of the class.
 * @param super_class_name The name of the super class.
 */
#define HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(class_name, super_class_name) \
  HOLOSCAN_OPERATOR_FORWARD_TEMPLATE()                                     \
  class_name(ArgT&& arg, ArgsT&&... args)                                  \
      : super_class_name(std::forward<ArgT>(arg), std::forward<ArgsT>(args)...) {}

namespace holoscan {

/**
 * @brief Base class for all operators.
 *
 * An operator is the most basic unit of work in Holoscan SDK. An Operator receives
 * streaming data at an input port, processes it, and publishes it to one of its output ports.
 *
 * This class is the base class for all operators. It provides the basic functionality for all
 * operators.
 *
 * @note This class is not intended to be used directly. Inherit from this class to create a new
 * operator.
 */
class Operator : public Component {
 public:
  /**
   * @brief Operator type used by the executor.
   */
  enum class OperatorType {
    kNative,   ///< Native operator.
    kGXF,      ///< GXF operator.
    kVirtual,  ///< Virtual operator.
  };

  /**
   * @brief Construct a new Operator object.
   *
   * @param args The arguments to be passed to the operator.
   */
  HOLOSCAN_OPERATOR_FORWARD_TEMPLATE()
  explicit Operator(ArgT&& arg, ArgsT&&... args) {
    add_arg(std::forward<ArgT>(arg));
    (add_arg(std::forward<ArgsT>(args)), ...);
  }

  Operator() = default;

  ~Operator() override = default;

  /**
   * @brief Get the operator type.
   *
   * @return The operator type.
   */
  OperatorType operator_type() const { return operator_type_; }

  using Component::id;
  /**
   * @brief Set the Operator ID.
   *
   * @param id The ID of the operator.
   * @return The reference to this operator.
   */
  Operator& id(int64_t id) {
    id_ = id;
    return *this;
  }

  using Component::name;
  /**
   * @brief Set the name of the operator.
   *
   * @param name The name of the operator.
   * @return The reference to this operator.
   */
  Operator& name(const std::string& name) {
    name_ = name;
    return *this;
  }

  using Component::fragment;
  /**
   * @brief Set the fragment of the operator.
   *
   * @param fragment The pointer to the fragment of the operator.
   * @return The reference to this operator.
   */
  Operator& fragment(Fragment* fragment) {
    fragment_ = fragment;
    return *this;
  }

  /**
   * @brief Set the operator spec.
   *
   * @param spec The operator spec.
   * @return The reference to this operator.
   */
  Operator& spec(const std::shared_ptr<OperatorSpec>& spec) {
    spec_ = spec;
    return *this;
  }
  /**
   * @brief Get the operator spec.
   *
   * @return The operator spec.
   */
  OperatorSpec* spec() { return spec_.get(); }

  /**
   * @brief Get the shared pointer to the operator spec.
   *
   * @return The shared pointer to the operator spec.
   */
  std::shared_ptr<OperatorSpec> spec_shared() { return spec_; }

  template <typename ConditionT>
  /**
   * @brief Get a shared pointer to the Condition object.
   *
   * @param name The name of the condition.
   * @return The reference to the Condition object. If the condition does not exist, return the
   * nullptr.
   */
  std::shared_ptr<ConditionT> condition(const std::string& name) {
    if (auto condition = conditions_.find(name); condition != conditions_.end()) {
      return std::dynamic_pointer_cast<ConditionT>(condition->second);
    }
    return nullptr;
  }

  /**
   * @brief Get the conditions of the operator.
   *
   * @return The conditions of the operator.
   */
  std::unordered_map<std::string, std::shared_ptr<Condition>>& conditions() { return conditions_; }

  template <typename ResourceT>
  /**
   * @brief Get a shared pointer to the Resource object.
   *
   * @param name The name of the resource.
   * @return The reference to the Resource object. If the resource does not exist, returns the
   * nullptr.
   */
  std::shared_ptr<ResourceT> resource(const std::string& name) {
    if (auto resource = resources_.find(name); resource != resources_.end()) {
      return std::dynamic_pointer_cast<ResourceT>(resource->second);
    }
    return nullptr;
  }

  /**
   * @brief Get the resources of the operator.
   *
   * @return The resources of the operator.
   */
  std::unordered_map<std::string, std::shared_ptr<Resource>>& resources() { return resources_; }

  using Component::add_arg;

  /**
   * @brief Add a condition to the operator.
   *
   * @param arg The condition to add.
   */
  void add_arg(const std::shared_ptr<Condition>& arg) {
    if (conditions_.find(arg->name()) != conditions_.end()) {
      HOLOSCAN_LOG_ERROR(
          "Condition '{}' already exists in the operator. Please specify a unique "
          "name when creating a Condition instance.",
          arg->name());
    } else {
      conditions_[arg->name()] = arg;
    }
  }

  /**
   * @brief Add a condition to the operator.
   *
   * @param arg The condition to add.
   */
  void add_arg(std::shared_ptr<Condition>&& arg) {
    if (conditions_.find(arg->name()) != conditions_.end()) {
      HOLOSCAN_LOG_ERROR(
          "Condition '{}' already exists in the operator. Please specify a unique "
          "name when creating a Condition instance.",
          arg->name());
    } else {
      conditions_[arg->name()] = std::move(arg);
    }
  }

  /**
   * @brief Add a resource to the operator.
   *
   * @param arg The resource to add.
   */
  void add_arg(const std::shared_ptr<Resource>& arg) {
    if (resources_.find(arg->name()) != resources_.end()) {
      HOLOSCAN_LOG_ERROR(
          "Resource '{}' already exists in the operator. Please specify a unique "
          "name when creating a Resource instance.",
          arg->name());
    } else {
      resources_[arg->name()] = arg;
    }
  }

  /**
   * @brief Add a resource to the operator.
   *
   * @param arg The resource to add.
   */
  void add_arg(std::shared_ptr<Resource>&& arg) {
    if (resources_.find(arg->name()) != resources_.end()) {
      HOLOSCAN_LOG_ERROR(
          "Resource '{}' already exists in the operator. Please specify a unique "
          "name when creating a Resource instance.",
          arg->name());
    } else {
      resources_[arg->name()] = std::move(arg);
    }
  }

  /**
   * @brief Define the operator specification.
   *
   * @param spec The reference to the operator specification.
   */
  virtual void setup(OperatorSpec& spec) { (void)spec; }

  /**
   * @brief Returns whether the operator is a root operator based on its fragment's graph
   *
   * @return True, if the operator is a root operator; false, otherwise
   */
  bool is_root();

  /**
   * @brief Returns whether the operator is a leaf operator based on its fragment's graph
   *
   * @return True, if the operator is a leaf operator; false, otherwise
   */
  bool is_leaf();

  /**
   * @brief Initialize the operator.
   *
   * This function is called when the fragment is initialized by
   * Executor::initialize_fragment().
   */
  void initialize() override;

  /**
   * @brief Implement the startup logic of the operator.
   *
   * This method is called multiple times over the lifecycle of the operator according to the
   * order defined in the lifecycle, and used for heavy initialization tasks such as allocating
   * memory resources.
   */
  virtual void start() {
    // Empty default implementation
  }

  /**
   * @brief Implement the shutdown logic of the operator.
   *
   * This method is called multiple times over the lifecycle of the operator according to the
   * order defined in the lifecycle, and used for heavy deinitialization tasks such as deallocation
   * of all resources previously assigned in start.
   */
  virtual void stop() {
    // Empty default implementation
  }

  /**
   * @brief Implement the compute method.
   *
   * This method is called by the runtime multiple times. The runtime calls this method until
   * the operator is stopped.
   *
   * @param op_input The input context of the operator.
   * @param op_output The output context of the operator.
   * @param context The execution context of the operator.
   */
  virtual void compute(InputContext& op_input, OutputContext& op_output,
                       ExecutionContext& context) {
    (void)op_input;
    (void)op_output;
    (void)context;
  }

  /**
   * @brief Register the argument setter for the given type.
   *
   * If the operator has an argument with a custom type, the argument setter must be registered
   * using this method.
   *
   * The argument setter is used to set the value of the argument from the YAML configuration.
   *
   * This method can be called in the initialization phase of the operator (e.g., `initialize()`).
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

  /// Return operator name and port name from a string in the format of "<op_name>[.<port_name>]".
  static std::pair<std::string, std::string> parse_port_name(const std::string& op_port_name);

  /**
   * @brief Register the codec for serialization/deserialization of a custom type.
   *
   * If the operator has an argument with a custom type, the codec must be registered
   * using this method.
   *
   * For example, suppose we want to emit using the following custom struct type:
   *
   * ```cpp
   * namespace holoscan {
   *   struct Coordinate {
   *     int16_t x;
   *     int16_t y;
   *     int16_t z;
   *   }
   * }  // namespace holoscan
   * ```
   *
   * Then, we can define codec<Coordinate> as follows where the serialize and deserialize methods
   * would be used for serialization and deserialization of this type, respectively.
   *
   * ```cpp
   * namespace holoscan {
   *
   *   template <>
   *   struct codec<Coordinate> {
   *     static expected<size_t, RuntimeError> serialize(const Coordinate& value, Endpoint* endpoint) {
   *       return serialize_trivial_type<Coordinate>(value, endpoint);
   *     }
   *     static expected<Coordinate, RuntimeError> deserialize(Endpoint* endpoint) {
   *       return deserialize_trivial_type<Coordinate>(endpoint);
   *     }
   *   };
   * }  // namespace holoscan
   * ```
   *
   * In this case, since this is a simple struct with a static size, we can use the
   * existing serialize_trivial_type and deserialize_trivial_type implementations.
   *
   * Finally, to register this custom codec at runtime, we need to make the following call
   * within the setup method of our Operator.
   *
   * ```cpp
   * register_codec<Coordinate>("Coordinate");
   * ```
   *
   * @tparam typeT The type of the argument to register.
   * @param codec_name The name of the codec (must be unique unless overwrite is true).
   * @param overwrite If true and codec_name already exists, the codec will be overwritten.
   */
  template <typename typeT>
  static void register_codec(const std::string& codec_name, bool overwrite = true) {
    CodecRegistry::get_instance().add_codec<typeT>(codec_name, overwrite);
  }

  /**
   * @brief Get a YAML representation of the operator.
   *
   * @return YAML node including type, specs, conditions and resources of the operator in addition
   * to the base component properties.
   */
  YAML::Node to_yaml_node() const override;

 protected:
  // Making the following classes as friend classes to allow them to access
  // get_consolidated_input_label, num_published_messages_map, update_input_message_label,
  // reset_input_message_labels and update_published_messages functions, which should only be called
  // externally by them
  friend class AnnotatedDoubleBufferReceiver;
  friend class AnnotatedDoubleBufferTransmitter;
  friend class DFFTCollector;

  /**
   * @brief This function returns a consolidated MessageLabel for all the input ports of an
   * Operator. If there is no input port (root Operator), then a new MessageLabel with the current
   * Operator and default receive timestamp is returned.
   *
   * @return The consolidated MessageLabel
   */

  MessageLabel get_consolidated_input_label();

  /**
   * @brief Update the input_message_labels map with the given MessageLabel a
   * corresponding input_name
   *
   * @param input_name The input port name for which the MessageLabel is updated
   * @param m The new MessageLabel that will be set for the input port
   */
  void update_input_message_label(std::string input_name, MessageLabel m) {
    input_message_labels[input_name] = m;
  }

  /**
   * @brief Reset the input message labels to clear all its contents. This is done for a leaf
   * operator when it finishes its execution as it is assumed that all its inputs are processed.
   */
  void reset_input_message_labels() { input_message_labels.clear(); }

  /**
   * @brief Get the number of published messages for each output port indexed by the output port
   * name.
   *
   * The function is utilized by the DFFTCollector to update the DataFlowTracker with the number of
   * published messages for root operators.
   *
   * @return The map of the number of published messages for every output name.
   */
  std::map<std::string, uint64_t> num_published_messages_map() {
    return num_published_messages_map_;
  }

  /**
   * @brief This function updates the number of published messages for a given output port.
   *
   * @param output_name The name of the output port
   */
  void update_published_messages(std::string output_name);

  /**
   * @brief Register the argument setter for the given type.
   *
   * Please refer to the documentation of `::register_converter()` for more details.
   *
   * @tparam typeT The type of the argument to register.
   */
  template <typename typeT>
  static void register_argument_setter() {
    ArgumentSetter::get_instance().add_argument_setter<typeT>(
        [](ParameterWrapper& param_wrap, Arg& arg) {
          std::any& any_param = param_wrap.value();

          // If arg has no name and value, that indicates that we want to set the default value for
          // the native operator if it is not specified.
          if (arg.name().empty() && !arg.has_value()) {
            auto& param = *std::any_cast<Parameter<typeT>*>(any_param);
            param.set_default_value();
            return;
          }

          std::any& any_arg = arg.value();

          // Note that the type of any_param is Parameter<typeT>*, not Parameter<typeT>.
          auto& param = *std::any_cast<Parameter<typeT>*>(any_param);
          const auto& arg_type = arg.arg_type();
          (void)param;

          auto element_type = arg_type.element_type();
          auto container_type = arg_type.container_type();

          HOLOSCAN_LOG_DEBUG(
              "Registering converter for parameter {} (element_type: {}, container_type: {})",
              arg.name(),
              static_cast<int>(element_type),
              static_cast<int>(container_type));

          if (element_type == ArgElementType::kYAMLNode) {
            auto& arg_value = std::any_cast<YAML::Node&>(any_arg);
            typeT new_value;
            bool parse_ok = YAML::convert<typeT>::decode(arg_value, new_value);
            if (!parse_ok) {
              HOLOSCAN_LOG_ERROR("Unable to parse YAML node for parameter '{}'", arg.name());
            } else {
              param = std::move(new_value);
            }
          } else {
            try {
              auto& arg_value = std::any_cast<typeT&>(any_arg);
              param = arg_value;
            } catch (const std::bad_any_cast& e) {
              HOLOSCAN_LOG_ERROR(
                  "Bad any cast exception caught for argument '{}': {}", arg.name(), e.what());
            }
          }
        });
  }

  OperatorType operator_type_ = OperatorType::kNative;  ///< The type of the operator.
  std::shared_ptr<OperatorSpec> spec_;                  ///< The operator spec of the operator.
  std::unordered_map<std::string, std::shared_ptr<Condition>>
      conditions_;  ///< The conditions of the operator.
  std::unordered_map<std::string, std::shared_ptr<Resource>>
      resources_;  ///< The resources used by the operator.

 private:
  /**
   * @brief Set the operator codelet or any other backend codebase.
   */
  void set_op_backend();

  /// The MessageLabel objects corresponding to the input ports indexed by the input port.
  std::unordered_map<std::string, MessageLabel> input_message_labels;

  /// The number of published messages for each output indexed by output names.
  std::map<std::string, uint64_t> num_published_messages_map_;

  /// The backend Codelet or other codebase pointer. It is used for DFFT.
  void* op_backend_ptr = nullptr;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_OPERATOR_HPP */

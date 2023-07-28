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

#ifndef HOLOSCAN_CORE_GXF_GXF_OPERATOR_HPP
#define HOLOSCAN_CORE_GXF_GXF_OPERATOR_HPP

#include <gxf/core/gxf.h>

#include <iostream>
#include <utility>

#include "../executors/gxf/gxf_parameter_adaptor.hpp"
#include "../operator.hpp"
#include "./gxf_utils.hpp"

namespace holoscan::ops {

class GXFOperator : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_TEMPLATE()
  explicit GXFOperator(ArgT&& arg, ArgsT&&... args)
      : Operator(std::forward<ArgT>(arg), std::forward<ArgsT>(args)...) {
    operator_type_ = holoscan::Operator::OperatorType::kGXF;
  }

  GXFOperator() : Operator() { operator_type_ = holoscan::Operator::OperatorType::kGXF; }

  /**
   * @brief Initialize the GXF operator.
   *
   * This function is called when the fragment is initialized by
   * Executor::initialize_fragment().
   *
   * This sets the operator type to `holoscan::Operator::OperatorType::kGXF`.
   */
  void initialize() override;

  /**
   * @brief Get the type name of the GXF component.
   *
   * The returned string is the type name of the GXF component and is used to
   * create the GXF component.
   *
   * Example: "nvidia::holoscan::AJASource"
   *
   * @return The type name of the GXF component.
   */
  virtual const char* gxf_typename() const = 0;

  /**
   * @brief Get the GXF context object.
   *
   * @return The GXF context object.
   */
  gxf_context_t gxf_context() const { return gxf_context_; }

  /**
   * @brief Set GXF entity ID.
   *
   * @param gxf_eid The GXF entity ID.
   */
  void gxf_eid(gxf_uid_t gxf_eid) { gxf_eid_ = gxf_eid; }
  /**
   * @brief Get the GXF entity ID.
   *
   * @return The GXF entity ID.
   */
  gxf_uid_t gxf_eid() const { return gxf_eid_; }

  /**
   * @brief Set the GXF component ID.
   *
   * @param gxf_cid The GXF component ID.
   */
  void gxf_cid(gxf_uid_t gxf_cid) { gxf_cid_ = gxf_cid; }

  /**
   * @brief Get the GXF component ID.
   *
   * @return The GXF component ID.
   */
  gxf_uid_t gxf_cid() const { return gxf_cid_; }

  /**
   * @brief Register the argument setter and the GXF parameter adaptor for the given type.
   *
   * If the GXF operator has an argument with a custom type, both the argument setter and GXF
   * parameter adaptor must be registered using this method.
   *
   * The argument setter is used to set the value of the argument from the YAML configuration, and
   * the GXF parameter adaptor is used to set the value of the GXF parameter from the argument value
   * in `YAML::Node` object.
   *
   * This method can be called in the initialization phase of the operator (e.g., `initialize()`).
   * The example below shows how to register the argument setter for the custom type (`Vec3`):
   *
   * ```cpp
   * void MyGXFOp::initialize() {
   *   register_converter<Vec3>();
   *
   *   holoscan::ops::GXFOperator::initialize();
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
    ::holoscan::Operator::register_argument_setter<typeT>();
    register_parameter_adaptor<typeT>();
  }

 protected:
  /**
   * @brief Register the GXF parameter adaptor for the given type.
   *
   * Please refer to the documentation of `::register_converter()` for more details.
   *
   * @tparam typeT The type of the argument to register.
   */
  template <typename typeT>
  static void register_parameter_adaptor() {
    ::holoscan::gxf::GXFParameterAdaptor::get_instance().add_param_handler<typeT>(
        [](gxf_context_t context,
           gxf_uid_t uid,
           const char* key,
           const ArgType& arg_type,
           const std::any& any_value) {
          try {
            auto& param = *std::any_cast<Parameter<typeT>*>(any_value);

            param.set_default_value();  // set default value if not set.

            if (param.has_value()) {
              auto& value = param.get();
              switch (arg_type.container_type()) {
                case ArgContainerType::kNative:
                case ArgContainerType::kVector: {
                  if (arg_type.element_type() == ArgElementType::kCustom) {
                    YAML::Node value_node = YAML::convert<typeT>::encode(value);
                    return GxfParameterSetFromYamlNode(context, uid, key, &value_node, "");
                  }
                  break;
                }
                case ArgContainerType::kArray: {
                  HOLOSCAN_LOG_ERROR("Unable to handle ArgContainerType::kArray type for key '{}'",
                                     key);
                  break;
                }
              }

              HOLOSCAN_LOG_WARN(
                  "Unable to get argument for key '{}' with type '{}'", key, typeid(typeT).name());
            }
          } catch (const std::bad_any_cast& e) {
            HOLOSCAN_LOG_ERROR(
                "Bad any cast exception caught for argument '{}': {}", key, e.what());
          }

          return GXF_FAILURE;
        });
  }
  gxf_context_t gxf_context_ = nullptr;  ///< The GXF context.
  gxf_uid_t gxf_eid_ = 0;                ///< GXF entity ID
  gxf_uid_t gxf_cid_ = 0;                ///< The GXF component ID.
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_CORE_GXF_GXF_OPERATOR_HPP */

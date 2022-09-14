/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "../operator.hpp"

#include <iostream>

#include <gxf/core/gxf.h>

#include "../argument_setter.hpp"
#include "../executors/gxf/gxf_parameter_adaptor.hpp"

namespace holoscan::ops {

class GXFOperator : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(GXFOperator)

  GXFOperator() = default;

  void initialize() override;

  virtual const char* gxf_typename() const = 0;

  template <typename typeT>
  static void register_converter() {
    ArgumentSetter::get_instance().add_argument_setter<typeT>([](ParameterWrapper& param_wrap,
                                                                 Arg& arg) {
      std::any& any_param = param_wrap.value();
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
          (int)element_type,
          (int)container_type);

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
        auto& arg_value = std::any_cast<typeT&>(any_arg);
        if (arg_value) {
          param = arg_value;
        } else {
          HOLOSCAN_LOG_ERROR("Unable to handle parameter '{}'", arg.name());
        }
      }
    });

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
                case ArgContainerType::kNative: {
                  if (arg_type.element_type() == ArgElementType::kCustom) {
                    YAML::Node value_node = YAML::convert<typeT>::encode(value);
                    return GxfParameterSetFromYamlNode(context, uid, key, &value_node, "");
                  }
                  break;
                }
                case ArgContainerType::kVector:
                case ArgContainerType::kArray: {
                  HOLOSCAN_LOG_ERROR(
                      "Unable to handle ArgContainerType::kVector/kArray type for key '{}'", key);
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

 protected:
  gxf_context_t gxf_context_ = nullptr;
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_CORE_GXF_GXF_OPERATOR_HPP */

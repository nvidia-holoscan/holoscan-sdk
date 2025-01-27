/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_CORE_COMPONENT_INL_HPP
#define HOLOSCAN_CORE_COMPONENT_INL_HPP

#include <utility>

#include "./argument_setter.hpp"

namespace holoscan {

  template <typename typeT>
  void ComponentBase::register_argument_setter() {
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

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_COMPONENT_INL_HPP */

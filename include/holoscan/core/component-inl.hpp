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

#include <memory>  // For std::shared_ptr, std::static_pointer_cast
#include <string>
#include <string_view>
#include <typeinfo>  // For typeid
#include <utility>

#include "./argument_setter.hpp"
#include "./fragment_service.hpp"
#include "./fragment_service_provider.hpp"

namespace holoscan {

template <typename typeT>
void ComponentBase::register_argument_setter() {
  ArgumentSetter::get_instance().add_argument_setter<typeT>(
      [](ParameterWrapper& param_wrap, Arg& arg) -> bool {
        std::any& any_param = param_wrap.value();

        // If arg has no name and value, that indicates that we want to set the default value for
        // the native operator if it is not specified.
        if (arg.name().empty() && !arg.has_value()) {
          auto& param = *std::any_cast<Parameter<typeT>*>(any_param);
          param.set_default_value();
          return true;
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
            return false;
          } else {
            param = std::move(new_value);
            return true;
          }
        } else {
          try {
            auto& arg_value = std::any_cast<typeT&>(any_arg);
            param = arg_value;
            return true;
          } catch (const std::bad_any_cast& e) {
            // Capture type information for detailed error reporting
            const char* expected = typeid(typeT).name();
            const std::type_info& actual_type = any_arg.type();
            const char* actual = actual_type == typeid(void) ? "<empty>" : actual_type.name();
            std::string error_message =
                fmt::format("Bad any cast while setting argument '{}': expected '{}', got '{}'. {}",
                            arg.name(),
                            expected,
                            actual,
                            e.what());
            HOLOSCAN_LOG_ERROR(error_message);
            return false;
          }
        }
      });
}

template <typename ServiceT>
std::shared_ptr<ServiceT> ComponentBase::service(std::string_view id) const {
  static_assert(holoscan::is_one_of_derived_v<ServiceT, Resource, FragmentService>,
                "ServiceT must inherit from Resource or FragmentService");

  // Early return if no service provider is available
  if (!service_provider_) {
    HOLOSCAN_LOG_DEBUG("Component '{}': No service provider available.", name());
    return nullptr;
  }

  // Get the base service from the provider
  auto base_service = service_provider_->get_service_erased(typeid(ServiceT), id);
  if (!base_service) {
    HOLOSCAN_LOG_DEBUG("Component '{}': Service of type {} with id '{}' not found.",
                       name(),
                       typeid(ServiceT).name(),
                       std::string(id));
    return nullptr;
  }

  // Handle Resource-derived services
  if constexpr (std::is_base_of_v<Resource, ServiceT>) {
    auto resource_ptr = base_service->resource();
    if (!resource_ptr) {
      HOLOSCAN_LOG_DEBUG(
          "Component '{}': No service resource is available for service with id '{}'.",
          name(),
          std::string(id));
      return nullptr;
    }

    // Attempt to cast the resource to the requested type
    auto typed_resource = std::dynamic_pointer_cast<ServiceT>(resource_ptr);
    if (!typed_resource) {
      HOLOSCAN_LOG_DEBUG(
          "Component '{}': Service resource with id '{}' is not type-castable to type '{}'.",
          name(),
          std::string(id),
          typeid(ServiceT).name());
    }
    return typed_resource;
  } else {
    // Handle FragmentService-derived services
    // Since DefaultFragmentService implements FragmentService, we can safely cast
    auto typed_service = std::dynamic_pointer_cast<ServiceT>(base_service);
    if (!typed_service) {
      HOLOSCAN_LOG_DEBUG("Component '{}': Service with id '{}' is not type-castable to type '{}'.",
                         name(),
                         std::string(id),
                         typeid(ServiceT).name());
    }
    return typed_service;
  }
}

inline std::shared_ptr<FragmentService> ComponentBase::get_service_by_type_info(
    const std::type_info& service_type, std::string_view id) const {
  if (!service_provider_) {
    HOLOSCAN_LOG_DEBUG("Component '{}': No service provider available.", name());
    return nullptr;
  }
  return service_provider_->get_service_erased(service_type, id);
}

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_COMPONENT_INL_HPP */

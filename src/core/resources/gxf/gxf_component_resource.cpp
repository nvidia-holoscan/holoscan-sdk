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

#include "holoscan/core/resources/gxf/gxf_component_resource.hpp"

#include <memory>
#include <string>

#include "holoscan/core/fragment.hpp"

namespace holoscan {

const char* GXFComponentResource::gxf_typename() const {
  return gxf_typename_.c_str();
}

void GXFComponentResource::setup([[maybe_unused]] ComponentSpec& spec) {
  // Get the GXF context from the executor
  gxf_context_t gxf_context = fragment()->executor().context();

  // Get the type ID of the component
  gxf_tid_t component_tid;
  gxf_result_t result = GxfComponentTypeId(gxf_context, gxf_typename(), &component_tid);
  if (result != GXF_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Unable to find the GXF type name ('{}') for the component", gxf_typename());
    throw std::runtime_error(
        fmt::format("Unable to find the GXF type name ('{}') for the component", gxf_typename()));
  }

  // Create a new ComponentInfo object for the component
  gxf_component_info_ = std::make_shared<gxf::ComponentInfo>(gxf_context, component_tid);

  // Set fake parameters for the component to be able to show the resource description
  // through the `ComponentSpec::description()` method
  auto& params = spec.params();
  auto& gxf_parameter_map = gxf_component_info_->parameter_info_map();

  for (const auto& key : gxf_component_info_->normal_parameters()) {
    const auto& gxf_param = gxf_parameter_map.at(key);
    ParameterFlag flag = static_cast<ParameterFlag>(gxf_param.flags);

    parameters_.emplace_back(nullptr, key, gxf_param.headline, gxf_param.description, flag);
    auto& parameter = parameters_.back();

    ParameterWrapper&& parameter_wrapper{
        &parameter, &typeid(void), gxf::ComponentInfo::get_arg_type(gxf_param), &parameter};
    params.try_emplace(key, parameter_wrapper);
  }
}

void GXFComponentResource::set_parameters() {
  // Here, we don't call update_params_from_args().
  // Instead, we set the parameters manually using the GXF API.

  // Set parameter values if they are specified in the arguments
  auto& parameter_map = gxf_component_info_->parameter_info_map();
  bool has_dev_id_param = parameter_map.find("dev_id") != parameter_map.end();
  std::optional<int32_t> dev_id_value;
  for (auto& arg : args_) {
    // Issue 4336947: dev_id parameter for allocator needs to be handled manually
    if (arg.name().compare(std::string("dev_id")) == 0 && has_dev_id_param) {
      const auto& arg_type = arg.arg_type();
      if (arg_type.element_type() == holoscan::ArgElementType::kInt32 &&
          arg_type.container_type() == holoscan::ArgContainerType::kNative) {
        try {
          dev_id_value = std::any_cast<int32_t>(arg.value());
          continue;
        } catch (const std::bad_any_cast& e) {
          HOLOSCAN_LOG_ERROR("Cannot cast dev_id argument to int32_t: {}", e.what());
        }
      }
    }
    // Set the parameter if it is found in the parameter map
    if (parameter_map.find(arg.name()) != parameter_map.end()) {
      holoscan::gxf::GXFParameterAdaptor::set_param(
          gxf_context(), gxf_cid(), arg.name().c_str(), arg.arg_type(), arg.value());
    }
  }
  if (has_dev_id_param) {
    if (!gxf_graph_entity_) {
      // only log error if the parameter value does not match the GPU 0 default value
      if (dev_id_value.has_value() && dev_id_value.value() != 0) {
        HOLOSCAN_LOG_ERROR(
            "`dev_id` parameter value '{}' found, but gxf_graph_entity_ was not initialized so it "
            "could not be added to the entity group. This parameter will be ignored and default "
            "GPU device 0 will be used",
            dev_id_value.value());
      }
    } else {
      // Get default value if not set by arguments
      if (!dev_id_value.has_value()) {
        // Get parameter value for dev_id
        auto& parameter_info = parameter_map.at("dev_id");
        dev_id_value = *static_cast<const int32_t*>(parameter_info.default_value);
      }
      handle_dev_id(dev_id_value);
    }
  }
}

}  // namespace holoscan

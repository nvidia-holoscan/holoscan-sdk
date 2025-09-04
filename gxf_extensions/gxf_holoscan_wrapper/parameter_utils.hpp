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

#ifndef GXF_HOLOSCAN_WRAPPER_PARAMETER_UTILS_HPP
#define GXF_HOLOSCAN_WRAPPER_PARAMETER_UTILS_HPP

#include <functional>
#include <list>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "holoscan/core/arg.hpp"
#include "holoscan/core/conditions/gxf/asynchronous.hpp"
#include "holoscan/core/conditions/gxf/boolean.hpp"
#include "holoscan/core/conditions/gxf/periodic.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_utils.hpp"
#include "holoscan/core/io_context.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/parameter.hpp"
#include "holoscan/core/resources/gxf/block_memory_pool.hpp"
#include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"
#include "holoscan/core/resources/gxf/unbounded_allocator.hpp"
#include "holoscan/logger/logger.hpp"

#include "gxf/core/component.hpp"
#include "gxf/core/expected.hpp"
#include "gxf/core/registrar.hpp"
#include "gxf/std/cuda_green_context.hpp"
#include "gxf/std/cuda_green_context_pool.hpp"
#include "gxf/cuda/cuda_stream_pool.hpp"
#include "gxf/std/block_memory_pool.hpp"
#include "gxf/std/scheduling_term.hpp"
#include "gxf/std/unbounded_allocator.hpp"

// Include ResourceWrapper so that ResourceWrapper is known
#include "resource_wrapper.hpp"

namespace holoscan::gxf {

/**
 * @brief Common GXF parameter structure for both OperatorWrapper and ResourceWrapper.
 */
struct CommonGXFParameter {
  nvidia::gxf::Parameter<YAML::Node> param;         ///< The GXF parameter (non-const).
  holoscan::ArgType arg_type;                       ///< The type of the parameter (Holoscan).
  holoscan::Parameter<void*>* param_ptr = nullptr;  ///< The pointer to the parameter (Holoscan).

  CommonGXFParameter(holoscan::ArgType arg_type, holoscan::Parameter<void*>* param_ptr)
      : arg_type(arg_type), param_ptr(param_ptr) {}
};

/**
 * @brief Helper function to find a corresponding GXF parameter flag.
 *
 * @param param_flag The Holoscan parameter flag.
 * @return The corresponding GXF parameter flag.
 */
inline gxf_parameter_flags_t_ find_gxf_param_flag(holoscan::ParameterFlag param_flag) {
  switch (param_flag) {
    case holoscan::ParameterFlag::kNone:
      return GXF_PARAMETER_FLAGS_NONE;
    case holoscan::ParameterFlag::kOptional:
      return GXF_PARAMETER_FLAGS_OPTIONAL;
    case holoscan::ParameterFlag::kDynamic:
      return GXF_PARAMETER_FLAGS_DYNAMIC;
    default:
      return GXF_PARAMETER_FLAGS_NONE;
  }
}

/**
 * @brief Helper function to register parameters using a registrar.
 *
 * This method is used by only OperatorWrapper and ResourceWrapper.
 */
inline gxf_result_t register_parameters(
    nvidia::gxf::Registrar* registrar, std::list<std::shared_ptr<CommonGXFParameter>>& parameters) {
  nvidia::gxf::Expected<void> result;
  for (auto& gxf_param : parameters) {
    auto param_ptr = gxf_param->param_ptr;
    auto param_flag = param_ptr->flag();
    auto gxf_param_flag = find_gxf_param_flag(param_flag);

    // Register parameters as optional so that only arguments from YAML are processed by the
    // initialize_holoscan_object() method
    result &= registrar->parameter(gxf_param->param,
                                   param_ptr->key().c_str(),
                                   param_ptr->headline().c_str(),
                                   param_ptr->description().c_str(),
                                   nvidia::gxf::Registrar::NoDefaultParameter(),
                                   GXF_PARAMETER_FLAGS_OPTIONAL);
  }

  return nvidia::gxf::ToResultCode(result);
}

/**
 * @brief Helper function to process a condition argument.
 */
inline void process_condition_arg(void* gxf_context, uint64_t cid,
                                  holoscan::Parameter<void*>* param_ptr, YAML::Node& param_gxf,
                                  FragmentWrapper& fragment_,
                                  std::function<void(const Arg&)> add_arg_func) {
  std::string tag;
  try {
    tag = param_gxf.as<std::string>();
  } catch (...) {
    std::stringstream ss;
    ss << param_gxf;
    HOLOSCAN_LOG_ERROR("Could not parse parameter {} from {}", param_ptr->key(), ss.str());
    return;
  }

  gxf_uid_t condition_cid = find_component_handle<nvidia::gxf::SchedulingTerm>(
      gxf_context, cid, param_ptr->key().c_str(), tag, "");

  gxf_tid_t condition_tid{};
  gxf_result_t code = GxfComponentType(gxf_context, condition_cid, &condition_tid);
  if (code != GXF_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to get component type for component id '{}': {}",
                       condition_cid,
                       GxfResultStr(code));
    return;
  }

  // Generic lambda to handle the creation of shared pointers and error handling
  auto process_condition = [&](auto* condition_ptr, auto condition_type) {
    code = GxfComponentPointer(
        gxf_context, condition_cid, condition_tid, reinterpret_cast<void**>(&condition_ptr));
    if (code == GXF_SUCCESS && condition_ptr) {
      auto condition = std::make_shared<decltype(condition_type)>(tag, condition_ptr);
      condition->fragment(&fragment_);
      // Setup the condition.
      auto spec = std::make_shared<ComponentSpec>(&fragment_);
      condition->setup(*spec.get());
      condition->spec(spec);
      add_arg_func(Arg(param_ptr->key()) = condition);
    } else {
      HOLOSCAN_LOG_ERROR("Failed to get {} for '{}': {}",
                         typeid(condition_type).name(),
                         condition_cid,
                         GxfResultStr(code));
    }
  };

  // Map of condition type IDs to their corresponding processing functions
  static const std::unordered_map<std::string, std::function<void()>> condition_processors = {
      {"nvidia::gxf::PeriodicSchedulingTerm",
       [&]() {
         process_condition(static_cast<nvidia::gxf::PeriodicSchedulingTerm*>(nullptr),
                           holoscan::PeriodicCondition{});
       }},
      {"nvidia::gxf::BooleanSchedulingTerm",
       [&]() {
         process_condition(static_cast<nvidia::gxf::BooleanSchedulingTerm*>(nullptr),
                           holoscan::BooleanCondition{});
       }},
      {"nvidia::gxf::AsynchronousSchedulingTerm", [&]() {
         process_condition(static_cast<nvidia::gxf::AsynchronousSchedulingTerm*>(nullptr),
                           holoscan::AsynchronousCondition{});
       }}};

  // Get the type name of the condition
  const char* condition_type_name = nullptr;
  code = GxfComponentTypeName(gxf_context, condition_tid, &condition_type_name);
  if (code != GXF_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to get component type name for component id '{}': {}",
                       condition_cid,
                       GxfResultStr(code));
    return;
  }

  // Find and execute the corresponding processing function
  auto it = condition_processors.find(condition_type_name);
  if (it != condition_processors.end()) {
    it->second();
  } else {
    HOLOSCAN_LOG_ERROR("Unsupported condition type for handle: {}", tag);
  }
}

/**
 * @brief Helper function to process a resource argument.
 */
inline void process_resource_arg(void* gxf_context, uint64_t cid,
                                 holoscan::Parameter<void*>* param_ptr, YAML::Node& param_gxf,
                                 FragmentWrapper& fragment_,
                                 std::function<void(const Arg&)> add_arg_func) {
  std::string tag;
  try {
    tag = param_gxf.as<std::string>();
  } catch (...) {
    std::stringstream ss;
    ss << param_gxf;
    HOLOSCAN_LOG_ERROR("Could not parse resource parameter {} from {}", param_ptr->key(), ss.str());
    return;
  }

  gxf_uid_t resource_cid = find_component_handle<nvidia::gxf::Component>(
      gxf_context, cid, param_ptr->key().c_str(), tag, "");

  gxf_tid_t resource_tid{};
  gxf_result_t code = GxfComponentType(gxf_context, resource_cid, &resource_tid);
  if (code != GXF_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to get resource component type for component id '{}': {}",
                       resource_cid,
                       GxfResultStr(code));
    return;
  }

  gxf_tid_t gxf_holoscan_resource_wrapper_tid{};
  code = GxfComponentTypeId(
      gxf_context, "holoscan::gxf::ResourceWrapper", &gxf_holoscan_resource_wrapper_tid);
  if (code != GXF_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to get ResourceWrapper type id: {}", GxfResultStr(code));
    return;
  }
  auto add_resource_arg = [&](auto* ptr, const std::shared_ptr<holoscan::Resource>& resource) {
    if (ptr) {
      add_arg_func(Arg(param_ptr->key()) = resource);
    }
  };

  auto process_resource = [&](auto* resource_ptr, auto resource_type) {
    code = GxfComponentPointer(
        gxf_context, resource_cid, resource_tid, reinterpret_cast<void**>(&resource_ptr));
    if (code == GXF_SUCCESS && resource_ptr) {
      auto resource = std::make_shared<decltype(resource_type)>(tag, resource_ptr);
      resource->fragment(&fragment_);
      // Setup the resource.
      auto spec = std::make_shared<ComponentSpec>(&fragment_);
      resource->setup(*spec.get());
      resource->spec(spec);
      add_resource_arg(resource_ptr, resource);
    } else {
      HOLOSCAN_LOG_ERROR("Failed to get {} for '{}': {}",
                         typeid(resource_type).name(),
                         resource_cid,
                         GxfResultStr(code));
    }
  };

  // Map of resource type IDs to their corresponding processing functions
  static const std::unordered_map<std::string, std::function<void()>> resource_processors = {
      {"nvidia::gxf::UnboundedAllocator",
       [&]() {
         process_resource(static_cast<nvidia::gxf::UnboundedAllocator*>(nullptr),
                          holoscan::UnboundedAllocator{});
       }},
      {"nvidia::gxf::BlockMemoryPool",
       [&]() {
         process_resource(static_cast<nvidia::gxf::BlockMemoryPool*>(nullptr),
                          holoscan::BlockMemoryPool{});
       }},
      {"nvidia::gxf::CudaStreamPool",
       [&]() {
         process_resource(static_cast<nvidia::gxf::CudaStreamPool*>(nullptr),
                          holoscan::CudaStreamPool{});
       }},
      {"nvidia::gxf::CudaGreenContext",
       [&]() {
         process_resource(static_cast<nvidia::gxf::CudaGreenContext*>(nullptr),
                          holoscan::CudaGreenContext(nullptr));
       }},
      {"nvidia::gxf::CudaGreenContextPool", [&]() {
         process_resource(static_cast<nvidia::gxf::CudaGreenContextPool*>(nullptr),
                          holoscan::CudaGreenContextPool{});
       }}};

  // Get the type name of the resource
  const char* resource_type_name;
  code = GxfComponentTypeName(gxf_context, resource_tid, &resource_type_name);
  if (code != GXF_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to get component type name for component id '{}': {}",
                       resource_cid,
                       GxfResultStr(code));
    return;
  }

  // Find and execute the corresponding processing function
  auto it = resource_processors.find(resource_type_name);
  if (it != resource_processors.end()) {
    it->second();
  } else {
    // Check if derived from ResourceWrapper
    bool is_base = false;
    code =
        GxfComponentIsBase(gxf_context, resource_tid, gxf_holoscan_resource_wrapper_tid, &is_base);
    if (code != GXF_SUCCESS) {
      HOLOSCAN_LOG_ERROR("Failed to check if the resource is derived from ResourceWrapper: {}",
                         GxfResultStr(code));
      return;
    }
    if (is_base) {
      holoscan::gxf::ResourceWrapper* resource_wrapper_ptr = nullptr;
      code = GxfComponentPointer(
          gxf_context, resource_cid, resource_tid, reinterpret_cast<void**>(&resource_wrapper_ptr));
      if (code == GXF_SUCCESS && resource_wrapper_ptr) {
        auto resource = resource_wrapper_ptr->resource();
        add_arg_func(Arg(param_ptr->key()) = resource);
      } else {
        HOLOSCAN_LOG_ERROR(
            "Failed to get ResourceWrapper pointer for '{}': {}", resource_cid, GxfResultStr(code));
      }
    } else {
      HOLOSCAN_LOG_ERROR("Unsupported resource type for handle: {}", tag);
    }
  }
}

/**
 * @brief Helper function to process vector of IOSpec receivers (for operators).
 */
inline void process_iospec_vector_arg(
    holoscan::Parameter<void*>* param_ptr, YAML::Node& param_gxf,
    std::function<holoscan::IOSpec&(const std::string&)> input_func) {
  std::vector<std::string> tags;
  try {
    tags = param_gxf.as<std::vector<std::string>>();
  } catch (...) {
    std::stringstream ss;
    ss << param_gxf;
    HOLOSCAN_LOG_ERROR("Could not parse IOSpec parameter {} from {}", param_ptr->key(), ss.str());
    return;
  }

  // Parameter is known to be std::vector<holoscan::IOSpec*>
  auto receivers_param_ptr =
      reinterpret_cast<holoscan::Parameter<std::vector<holoscan::IOSpec*>>*>(param_ptr);
  receivers_param_ptr->set_default_value();
  auto& iospec_vector = receivers_param_ptr->get();
  iospec_vector.reserve(tags.size());
  for (auto& tag : tags) {
    HOLOSCAN_LOG_TRACE("    creating new input port: {}", tag);
    auto& input_port = input_func(tag);
    iospec_vector.push_back(&input_port);
  }
}

/**
 * @brief Initialize a Holoscan object (Operator or Resource) by setting parameters and context.
 *
 * This method is used by both OperatorWrapper and ResourceWrapper to set YAML arguments.
 *
 * @tparam T Holoscan object type (Operator or Resource)
 * @param gxf_context GXF context pointer
 * @param eid entity id
 * @param cid component id
 * @param fragment_ FragmentWrapper
 * @param obj The Holoscan object (Operator or Resource)
 * @param parameters The list of parameters
 * @param input_func Optional: function for IOSpec (only for Operators)
 */
template <typename T>
inline gxf_result_t initialize_holoscan_object(
    void* gxf_context, gxf_uid_t eid, gxf_uid_t cid, FragmentWrapper& fragment_,
    std::shared_ptr<T>& obj, std::list<std::shared_ptr<CommonGXFParameter>>& parameters,
    std::function<holoscan::IOSpec&(const std::string&)> input_func = nullptr) {
  if (!obj) {
    HOLOSCAN_LOG_ERROR("initialize_holoscan_object() - obj is null");
    return GXF_FAILURE;
  }

  if (!fragment_.executor().context()) {
    fragment_.executor().context(gxf_context);
    fragment_.gxf_executor().op_eid(eid);
    fragment_.gxf_executor().op_cid(cid);

    // Get the component name and set it to the object
    const char* cname = "";
    HOLOSCAN_GXF_CALL_FATAL(GxfComponentName(gxf_context, cid, &cname));
    obj->name(cname);
    // Set the fragment to the object
    obj->fragment(&fragment_);
    obj->spec()->fragment(&fragment_);

    // Set parameters from GXF parameter values (YAML::Node)
    for (auto& gxf_param : parameters) {
      const auto& arg_type = gxf_param->arg_type;
      auto param_ptr = gxf_param->param_ptr;
      const auto& arg = gxf_param->param.try_get();
      if (!arg)
        continue;

      auto& param_gxf = const_cast<YAML::Node&>(arg.value());
      auto add_arg_func = [&](const Arg& a) { obj->add_arg(a); };

      switch (arg_type.element_type()) {
        case ArgElementType::kCondition:
          process_condition_arg(gxf_context, cid, param_ptr, param_gxf, fragment_, add_arg_func);
          break;
        case ArgElementType::kResource:
          process_resource_arg(gxf_context, cid, param_ptr, param_gxf, fragment_, add_arg_func);
          break;
        case ArgElementType::kIOSpec:
          if (arg_type.container_type() == ArgContainerType::kVector && input_func) {
            process_iospec_vector_arg(param_ptr, param_gxf, input_func);
          } else {
            HOLOSCAN_LOG_ERROR("Unsupported IOSpec parameter type for {}", param_ptr->key());
          }
          break;
        default:
          // Generic parameter
          obj->add_arg(Arg(param_ptr->key()) = param_gxf);
          break;
      }
    }

    // Initialize the Holoscan object
    obj->initialize();
  }

  return GXF_SUCCESS;
}

}  // namespace holoscan::gxf

#endif /* GXF_HOLOSCAN_WRAPPER_PARAMETER_UTILS_HPP */

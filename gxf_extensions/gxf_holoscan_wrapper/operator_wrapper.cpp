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

#include "operator_wrapper.hpp"

#include "gxf/cuda/cuda_stream_pool.hpp"
#include "gxf/std/scheduling_term.hpp"
#include "gxf/std/unbounded_allocator.hpp"
// #include "gxf/std/block_memory_pool.hpp" // TODO: uncomment when available in GXF SDK package

#include "holoscan/core/common.hpp"
#include "holoscan/core/gxf/gxf_execution_context.hpp"
#include "holoscan/core/gxf/gxf_utils.hpp"
#include "holoscan/core/io_context.hpp"
#include "holoscan/core/resources/gxf/block_memory_pool.hpp"
#include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"
#include "holoscan/core/resources/gxf/unbounded_allocator.hpp"

namespace holoscan::gxf {

// Whether the log level has been set from the environment.
static bool g_operator_wrapper_env_log_level_set = false;

OperatorWrapper::OperatorWrapper() : nvidia::gxf::Codelet() {
  // Check if the environment variable for log level has been set.
  if (!g_operator_wrapper_env_log_level_set) {
    g_operator_wrapper_env_log_level_set = true;
    holoscan::load_env_log_level();
  }
}

gxf_result_t OperatorWrapper::initialize() {
  HOLOSCAN_LOG_TRACE("OperatorWrapper::initialize()");
  if (!op_) {
    HOLOSCAN_LOG_ERROR("OperatorWrapper::initialize() - op_ is null");
    return GXF_FAILURE;
  }
  if (!fragment_.executor().context()) {
    // Set the current GXF context to the fragment's executor.
    fragment_.executor().context(context());
    // Set the executor (GXFExecutor)'s 'op_eid_' to this codelet's eid
    // so that the operator is initialized under this codelet's eid.
    fragment_.gxf_executor().op_eid(eid());
    // Set the executor (GXFExecutor)'s 'op_cid_' to this codelet's cid
    // so that the operator is initialized with this codelet's cid.
    fragment_.gxf_executor().op_cid(cid());

    // Set the operator's name to the typename of the operator.
    // (it doesn't affect anything)
    op_->name(holoscan_typename());
    // Set the operator's fragment to the fragment we created.
    op_->fragment(&fragment_);
    op_->spec()->fragment(&fragment_);  // lazy fragment setup

    // Set parameters.
    for (auto& gxf_param : parameters_) {
      const auto& arg_type = gxf_param.arg_type;
      const auto param_ptr = gxf_param.param_ptr;
      const auto& arg = gxf_param.param.try_get();
      if (arg) {
        auto& param_gxf = const_cast<YAML::Node&>(arg.value());

        // Handle cases where the argument is a handle such as a condition or a resource.
        if (arg_type.element_type() == ArgElementType::kCondition) {
          // Create a condition object based on the handle name.
          HOLOSCAN_LOG_TRACE("  adding condition: {}", param_ptr->key());

          // Parse string from node
          std::string tag;
          try {
            tag = param_gxf.as<std::string>();
          } catch (...) {
            std::stringstream ss;
            ss << param_gxf;
            HOLOSCAN_LOG_ERROR("Could not parse parameter {} from {}", param_ptr->key(), ss.str());
            continue;
          }

          gxf_uid_t condition_cid = find_component_handle<nvidia::gxf::SchedulingTerm>(
              context(), cid(), param_ptr->key().c_str(), tag, "");

          gxf_tid_t condition_tid{};
          gxf_result_t code = GxfComponentType(context(), condition_cid, &condition_tid);

          gxf_tid_t boolean_condition_tid{};
          GxfComponentTypeId(
              context(), "nvidia::gxf::BooleanSchedulingTerm", &boolean_condition_tid);

          if (condition_tid == boolean_condition_tid) {
            nvidia::gxf::BooleanSchedulingTerm* boolean_condition_ptr = nullptr;
            code = GxfComponentPointer(context(),
                                       condition_cid,
                                       condition_tid,
                                       reinterpret_cast<void**>(&boolean_condition_ptr));

            if (boolean_condition_ptr) {
              auto condition =
                  std::make_shared<holoscan::BooleanCondition>(tag, boolean_condition_ptr);
              op_->add_arg(Arg(param_ptr->key()) = condition);
            }
          } else {
            HOLOSCAN_LOG_ERROR("Unsupported condition type for the handle: {}", tag);
          }
        } else if (arg_type.element_type() == ArgElementType::kResource) {
          // Create a resource object based on the handle name.
          HOLOSCAN_LOG_TRACE("  adding resource: {}", param_ptr->key());

          // Parse string from node
          std::string tag;
          try {
            tag = param_gxf.as<std::string>();
          } catch (...) {
            std::stringstream ss;
            ss << param_gxf;
            HOLOSCAN_LOG_ERROR("Could not parse parameter {} from {}", param_ptr->key(), ss.str());
            continue;
          }

          gxf_uid_t resource_cid = find_component_handle<nvidia::gxf::Component>(
              context(), cid(), param_ptr->key().c_str(), tag, "");

          gxf_tid_t resource_tid{};
          gxf_result_t code = GxfComponentType(context(), resource_cid, &resource_tid);

          gxf_tid_t unbounded_allocator_tid{};
          GxfComponentTypeId(
              context(), "nvidia::gxf::UnboundedAllocator", &unbounded_allocator_tid);

          gxf_tid_t block_memory_pool_tid{};
          GxfComponentTypeId(context(), "nvidia::gxf::BlockMemoryPool", &block_memory_pool_tid);

          gxf_tid_t cuda_stream_pool_tid{};
          GxfComponentTypeId(context(), "nvidia::gxf::CudaStreamPool", &cuda_stream_pool_tid);

          if (resource_tid == unbounded_allocator_tid) {
            nvidia::gxf::UnboundedAllocator* unbounded_allocator_ptr = nullptr;
            code = GxfComponentPointer(context(),
                                       resource_cid,
                                       resource_tid,
                                       reinterpret_cast<void**>(&unbounded_allocator_ptr));

            if (unbounded_allocator_ptr) {
              auto resource =
                  std::make_shared<holoscan::UnboundedAllocator>(tag, unbounded_allocator_ptr);
              op_->add_arg(Arg(param_ptr->key()) = resource);
            }
          } else if (resource_tid == block_memory_pool_tid) {
            nvidia::gxf::BlockMemoryPool* block_memory_pool_ptr = nullptr;
            code = GxfComponentPointer(context(),
                                       resource_cid,
                                       resource_tid,
                                       reinterpret_cast<void**>(&block_memory_pool_ptr));

            if (block_memory_pool_ptr) {
              auto resource =
                  std::make_shared<holoscan::BlockMemoryPool>(tag, block_memory_pool_ptr);
              op_->add_arg(Arg(param_ptr->key()) = resource);
            }
          } else if (resource_tid == cuda_stream_pool_tid) {
            nvidia::gxf::CudaStreamPool* cuda_stream_pool_ptr = nullptr;
            code = GxfComponentPointer(context(),
                                       resource_cid,
                                       resource_tid,
                                       reinterpret_cast<void**>(&cuda_stream_pool_ptr));

            if (cuda_stream_pool_ptr) {
              auto resource = std::make_shared<holoscan::CudaStreamPool>(tag, cuda_stream_pool_ptr);
              op_->add_arg(Arg(param_ptr->key()) = resource);
            }
          } else {
            HOLOSCAN_LOG_ERROR("Unsupported resource type for the handle: {}", tag);
          }
        } else if (arg_type.element_type() == ArgElementType::kIOSpec &&
                   arg_type.container_type() == ArgContainerType::kVector) {
          // Create IOSpec objects based on the receivers name
          HOLOSCAN_LOG_TRACE("  adding receivers: {}", param_ptr->key());

          // Parse string from node
          std::vector<std::string> tags;
          try {
            tags = param_gxf.as<std::vector<std::string>>();
          } catch (...) {
            std::stringstream ss;
            ss << param_gxf;
            HOLOSCAN_LOG_ERROR("Could not parse parameter {} from {}", param_ptr->key(), ss.str());
            continue;
          }

          // Get the receivers parameter pointer.
          auto receivers_param_ptr =
              reinterpret_cast<Parameter<std::vector<holoscan::IOSpec*>>*>(param_ptr);
          // Set the default value.
          receivers_param_ptr->set_default_value();
          // Get the vector of IOSpec pointers.
          std::vector<holoscan::IOSpec*>& iospec_vector = receivers_param_ptr->get();
          iospec_vector.reserve(tags.size());
          for (auto& tag : tags) {
            HOLOSCAN_LOG_TRACE("    creating new input port: {}", tag);
            // Create and add the new input port to the vector.
            auto& input_port = op_->spec()->input<gxf::Entity>(tag);
            iospec_vector.push_back(&input_port);
          }
        } else {
          // Add argument to the operator (as YAML::Node object).
          op_->add_arg(Arg(param_ptr->key()) = param_gxf);
        }
      }
    }

    // Initialize the operator.
    op_->initialize();
  }

  return GXF_SUCCESS;
}
gxf_result_t OperatorWrapper::deinitialize() {
  HOLOSCAN_LOG_TRACE("OperatorWrapper::deinitialize()");
  return GXF_SUCCESS;
}

gxf_result_t OperatorWrapper::registerInterface(nvidia::gxf::Registrar* registrar) {
  HOLOSCAN_LOG_TRACE("OperatorWrapper::registerInterface()");
  nvidia::gxf::Expected<void> result;
  if (!op_) {
    HOLOSCAN_LOG_ERROR("OperatorWrapper::registerInterface() - op_ is null");
    return GXF_FAILURE;
  }

  // This method (registerInterface()) is called before initialize() multiple times.
  if (!op_->spec()) {
    // Setup the operator.
    auto spec = std::make_shared<OperatorSpec>(nullptr);
    op_->setup(*spec.get());
    op_->spec(spec);

    // Initialize the list of GXFParameter objects
    for (auto& param : op_->spec()->params()) {
      HOLOSCAN_LOG_TRACE("  adding param: {}", param.first);
      // Cast the storage pointer to a Parameter<void*> pointer, to access metadata.
      // (Accessing the value is illegal, because the type is unknown.)
      auto storage_ptr = static_cast<holoscan::Parameter<void*>*>(param.second.storage_ptr());
      if (!storage_ptr) {
        HOLOSCAN_LOG_ERROR("OperatorWrapper::registerInterface() - storage_ptr is null");
        return GXF_FAILURE;
      }
      parameters_.emplace_back(GXFParameter{{}, param.second.arg_type(), storage_ptr});
    }
  }

  // Register the operator's parameters.
  for (auto& gxf_param : parameters_) {
    const auto param_ptr = gxf_param.param_ptr;
    HOLOSCAN_LOG_TRACE("  registering param: {}", param_ptr->key());
    // TODO(gbae): support parameter flags
    result &= registrar->parameter(gxf_param.param,
                                   param_ptr->key().c_str(),
                                   param_ptr->headline().c_str(),
                                   param_ptr->description().c_str(),
                                   nvidia::gxf::Unexpected(),
                                   GXF_PARAMETER_FLAGS_OPTIONAL);
  }

  return nvidia::gxf::ToResultCode(result);
}

gxf_result_t OperatorWrapper::start() {
  HOLOSCAN_LOG_TRACE("OperatorWrapper::start()");
  if (op_ == nullptr) {
    HOLOSCAN_LOG_ERROR("OperatorWrapper::start() - Operator is not set");
    return GXF_FAILURE;
  }
  op_->start();
  return GXF_SUCCESS;
}

gxf_result_t OperatorWrapper::tick() {
  HOLOSCAN_LOG_TRACE("OperatorWrapper::tick()");
  if (!op_) {
    HOLOSCAN_LOG_ERROR("OperatorWrapper::tick() - Operator is not set");
    return GXF_FAILURE;
  }

  HOLOSCAN_LOG_TRACE("Calling operator: {}", op_->name());

  GXFExecutionContext exec_context(context(), op_.get());
  InputContext* op_input = exec_context.input();
  OutputContext* op_output = exec_context.output();
  op_->compute(*op_input, *op_output, exec_context);

  return GXF_SUCCESS;
}

gxf_result_t OperatorWrapper::stop() {
  HOLOSCAN_LOG_TRACE("OperatorWrapper::stop()");
  if (!op_) {
    HOLOSCAN_LOG_ERROR("OperatorWrapper::stop() - Operator is not set");
    return GXF_FAILURE;
  }
  op_->stop();
  return GXF_SUCCESS;
}

}  // namespace holoscan::gxf

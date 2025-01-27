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

#include "operator_wrapper.hpp"

#include <memory>
#include <string>

#include "holoscan/core/gxf/gxf_execution_context.hpp"
#include "holoscan/core/io_context.hpp"

#include "parameter_utils.hpp"  // include the new utility header

namespace holoscan::gxf {

OperatorWrapper::OperatorWrapper() : nvidia::gxf::Codelet() {
  // Set the log level from the environment variable if it exists.
  // Or, set the default log level to INFO if it hasn't been set by the user.
  if (!Logger::log_level_set_by_user) { holoscan::set_log_level(LogLevel::INFO); }
  // Set the log format from the environment variable if it exists.
  // Or, set the default log format depending on the log level if it hasn't been set by the user.
  holoscan::set_log_pattern();
}

gxf_result_t OperatorWrapper::initialize() {
  HOLOSCAN_LOG_TRACE("OperatorWrapper::initialize()");
  if (!op_) {
    HOLOSCAN_LOG_ERROR("OperatorWrapper::initialize() - op_ is null");
    return GXF_FAILURE;
  }

  // Use the utility function. We pass a lambda for input_func to handle IOSpec.
  auto input_func = [&](const std::string& tag) -> holoscan::IOSpec& {
    return op_->spec()->input<gxf::Entity>(tag);
  };

  op_->enable_metadata(true);  // enable metadata by default
  return initialize_holoscan_object(
      context(), eid(), cid(), fragment_, op_, parameters_, input_func);
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
  // Setup spec if not already done.
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
      parameters_.push_back(
          std::make_shared<CommonGXFParameter>(param.second.arg_type(), storage_ptr));
    }
  }

  // Call registerParameterlessComponent() regardless of whether there are parameters or not.
  // This will ensure that the component parameter information for this type (codelet) is available.
  result &= registrar->registerParameterlessComponent();
  auto code = register_parameters(registrar, parameters_);
  if (code != GXF_SUCCESS) return code;

  return nvidia::gxf::ToResultCode(result);
}

gxf_result_t OperatorWrapper::start() {
  HOLOSCAN_LOG_TRACE("OperatorWrapper::start()");
  if (op_ == nullptr) {
    HOLOSCAN_LOG_ERROR("OperatorWrapper::start() - Operator is not set");
    return GXF_FAILURE;
  }

  op_->start();

  exec_context_ = std::make_unique<GXFExecutionContext>(context(), op_.get());
  exec_context_->init_cuda_object_handler(op_.get());
  HOLOSCAN_LOG_TRACE("GXFWrapper: exec_context_->cuda_object_handler() for op '{}' is {}null",
                     op_->name(),
                     exec_context_->cuda_object_handler() == nullptr ? "" : "not ");
  op_input_ = exec_context_->input();
  op_input_->cuda_object_handler(exec_context_->cuda_object_handler());
  op_output_ = exec_context_->output();
  op_output_->cuda_object_handler(exec_context_->cuda_object_handler());
  return GXF_SUCCESS;
}

gxf_result_t OperatorWrapper::tick() {
  HOLOSCAN_LOG_TRACE("OperatorWrapper::tick()");
  if (!op_) {
    HOLOSCAN_LOG_ERROR("OperatorWrapper::tick() - Operator is not set");
    return GXF_FAILURE;
  }

  // clear any existing values from a previous compute call
  op_->metadata()->clear();

  // clear any received streams from previous compute call
  exec_context_->clear_received_streams();

  HOLOSCAN_LOG_TRACE("Calling operator: {}", op_->name());
  op_->compute(*op_input_, *op_output_, *exec_context_);
  return GXF_SUCCESS;
}

gxf_result_t OperatorWrapper::stop() {
  HOLOSCAN_LOG_TRACE("OperatorWrapper::stop()");
  if (!op_) {
    HOLOSCAN_LOG_ERROR("OperatorWrapper::stop() - Operator is not set");
    return GXF_FAILURE;
  }

  op_->stop();

  exec_context_->release_internal_cuda_streams();
  return GXF_SUCCESS;
}

}  // namespace holoscan::gxf

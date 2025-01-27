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

#include <memory>

#include "resource_wrapper.hpp"
#include "parameter_utils.hpp"  // include the new utility header

namespace holoscan::gxf {

ResourceWrapper::ResourceWrapper() : nvidia::gxf::Component() {
  // Set the log level from the environment variable if it exists.
  // Or, set the default log level to INFO if it hasn't been set by the user.
  if (!Logger::log_level_set_by_user) { holoscan::set_log_level(LogLevel::INFO); }
  // Set the log format from the environment variable if it exists.
  // Or, set the default log format depending on the log level if it hasn't been set by the user.
  holoscan::set_log_pattern();
}

std::shared_ptr<Resource> ResourceWrapper::resource() const {
  return res_;
}

gxf_result_t ResourceWrapper::initialize() {
  HOLOSCAN_LOG_TRACE("ResourceWrapper::initialize()");
  if (!res_) {
    HOLOSCAN_LOG_ERROR("ResourceWrapper::initialize() - res_ is null");
    return GXF_FAILURE;
  }

  // For resources, we do not provide an input_func since IOSpec is not used.
  return initialize_holoscan_object(context(), eid(), cid(), fragment_, res_, parameters_);
}

gxf_result_t ResourceWrapper::deinitialize() {
  HOLOSCAN_LOG_TRACE("ResourceWrapper::deinitialize()");
  return GXF_SUCCESS;
}

gxf_result_t ResourceWrapper::registerInterface(nvidia::gxf::Registrar* registrar) {
  HOLOSCAN_LOG_TRACE("ResourceWrapper::registerInterface()");
  nvidia::gxf::Expected<void> result;
  if (!res_) {
    HOLOSCAN_LOG_ERROR("ResourceWrapper::registerInterface() - res_ is null");
    return GXF_FAILURE;
  }

  // This method (registerInterface()) is called before initialize() multiple times.
  // Setup spec if not already done.
  if (!res_->spec()) {
    // Setup the resource.
    auto spec = std::make_shared<ComponentSpec>(nullptr);
    res_->setup(*spec.get());
    res_->spec(spec);

    // Initialize the list of GXFParameter objects
    for (auto& param : res_->spec()->params()) {
      HOLOSCAN_LOG_TRACE("  adding param: {}", param.first);
      // Cast the storage pointer to a Parameter<void*> pointer, to access metadata.
      // (Accessing the value is illegal, because the type is unknown.)
      auto storage_ptr = static_cast<holoscan::Parameter<void*>*>(param.second.storage_ptr());
      if (!storage_ptr) {
        HOLOSCAN_LOG_ERROR("ResourceWrapper::registerInterface() - storage_ptr is null");
        return GXF_FAILURE;
      }
      parameters_.push_back(
          std::make_shared<CommonGXFParameter>(param.second.arg_type(), storage_ptr));
    }
  }

  // Call registerParameterlessComponent() regardless of whether there are parameters or not.
  // This will ensure that the component parameter information for this type (component) is
  // available.
  result &= registrar->registerParameterlessComponent();
  auto code = register_parameters(registrar, parameters_);
  if (code != GXF_SUCCESS) return code;

  return nvidia::gxf::ToResultCode(result);
}

}  // namespace holoscan::gxf

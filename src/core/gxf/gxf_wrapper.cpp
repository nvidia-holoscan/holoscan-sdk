/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/core/gxf/gxf_wrapper.hpp"

#include "holoscan/core/common.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_execution_context.hpp"
#include "holoscan/core/io_context.hpp"

#include "gxf/std/transmitter.hpp"

namespace holoscan::gxf {

gxf_result_t GXFWrapper::initialize() {
  HOLOSCAN_LOG_TRACE("GXFWrapper::initialize()");
  return GXF_SUCCESS;
}
gxf_result_t GXFWrapper::deinitialize() {
  HOLOSCAN_LOG_TRACE("GXFWrapper::deinitialize()");
  return GXF_SUCCESS;
}

gxf_result_t GXFWrapper::registerInterface(nvidia::gxf::Registrar* registrar) {
  HOLOSCAN_LOG_TRACE("GXFWrapper::registerInterface()");
  (void)registrar;
  return GXF_SUCCESS;
}

gxf_result_t GXFWrapper::start() {
  HOLOSCAN_LOG_TRACE("GXFWrapper::start()");
  if (op_ == nullptr) {
    HOLOSCAN_LOG_ERROR("GXFWrapper::start() - Operator is not set");
    return GXF_FAILURE;
  }

  HOLOSCAN_LOG_TRACE("Starting operator: {}", op_->name());

  try {
    op_->start();
  } catch (const std::exception& e) {
    store_exception();
    HOLOSCAN_LOG_ERROR(
        "Exception occurred when starting operator: '{}' - {}", op_->name(), e.what());
    return GXF_FAILURE;
  }

  return GXF_SUCCESS;
}

gxf_result_t GXFWrapper::tick() {
  HOLOSCAN_LOG_TRACE("GXFWrapper::tick()");
  if (op_ == nullptr) {
    HOLOSCAN_LOG_ERROR("GXFWrapper::tick() - Operator is not set");
    return GXF_FAILURE;
  }

  HOLOSCAN_LOG_TRACE("Calling operator: {}", op_->name());
  GXFExecutionContext exec_context(context(), op_);
  InputContext* op_input = exec_context.input();
  OutputContext* op_output = exec_context.output();
  try {
    // clear any existing values from a previous compute call
    op_->metadata()->clear();

    op_->compute(*op_input, *op_output, exec_context);
    // Note: output metadata is inserted via op_output.emit() rather than here
  } catch (const std::exception& e) {
    // Note: Rethrowing the exception (using `throw;`) would cause the Python interpreter to exit.
    //       To avoid this, we store the exception and return GXF_FAILURE.
    //       The exception is then rethrown in GXFExecutor::run_gxf_graph().
    store_exception();
    HOLOSCAN_LOG_ERROR("Exception occurred for operator: '{}' - {}", op_->name(), e.what());
    return GXF_FAILURE;
  }

  return GXF_SUCCESS;
}

gxf_result_t GXFWrapper::stop() {
  HOLOSCAN_LOG_TRACE("GXFWrapper::stop()");
  if (op_ == nullptr) {
    HOLOSCAN_LOG_ERROR("GXFWrapper::stop() - Operator is not set");
    return GXF_FAILURE;
  }

  HOLOSCAN_LOG_TRACE("Stopping operator: {}", op_->name());

  try {
    op_->stop();
  } catch (const std::exception& e) {
    store_exception();
    HOLOSCAN_LOG_ERROR(
        "Exception occurred when stopping operator: '{}' - {}", op_->name(), e.what());
    return GXF_FAILURE;
  }

  return GXF_SUCCESS;
}

void GXFWrapper::store_exception() {
  auto stored_exception = std::current_exception();
  if (stored_exception != nullptr) { op_->fragment()->executor().exception(stored_exception); }
}

}  // namespace holoscan::gxf

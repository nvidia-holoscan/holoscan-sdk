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

#include "holoscan/core/gxf/gxf_scheduling_term_wrapper.hpp"

#include "holoscan/core/common.hpp"
#include "holoscan/core/condition.hpp"
#include "holoscan/core/fragment.hpp"

namespace holoscan::gxf {

gxf_result_t GXFSchedulingTermWrapper::initialize() {
  HOLOSCAN_LOG_TRACE("GXFSchedulingTermWrapper::initialize()");
  return GXF_SUCCESS;
}
gxf_result_t GXFSchedulingTermWrapper::deinitialize() {
  HOLOSCAN_LOG_TRACE("GXFSchedulingTermWrapper::deinitialize()");
  return GXF_SUCCESS;
}

gxf_result_t GXFSchedulingTermWrapper::registerInterface(nvidia::gxf::Registrar* registrar) {
  HOLOSCAN_LOG_TRACE("GXFSchedulingTermWrapper::registerInterface()");
  (void)registrar;
  return GXF_SUCCESS;
}

namespace {

/// @brief Convert a Holoscan scheduling status enum to the corresponding GXF one
nvidia::gxf::SchedulingConditionType to_gxf_scheduling_type(
    holoscan::SchedulingStatusType status_type) {
  switch (status_type) {
    case holoscan::SchedulingStatusType::kNever:
      return nvidia::gxf::SchedulingConditionType::NEVER;
    case holoscan::SchedulingStatusType::kReady:
      return nvidia::gxf::SchedulingConditionType::READY;
    case holoscan::SchedulingStatusType::kWait:
      return nvidia::gxf::SchedulingConditionType::WAIT;
    case holoscan::SchedulingStatusType::kWaitTime:
      return nvidia::gxf::SchedulingConditionType::WAIT_TIME;
    case holoscan::SchedulingStatusType::kWaitEvent:
      return nvidia::gxf::SchedulingConditionType::WAIT_EVENT;
    default:
      throw std::runtime_error(
          fmt::format("Unknown holoscan::SchedulingStatusType: {}", static_cast<int>(status_type)));
  }
}

}  // namespace

gxf_result_t GXFSchedulingTermWrapper::check_abi(int64_t timestamp,
                                                 nvidia::gxf::SchedulingConditionType* status_type,
                                                 int64_t* target_timestamp) const {
  HOLOSCAN_LOG_TRACE("GXFSchedulingTermWrapper::start()");
  if (condition_ == nullptr) {
    HOLOSCAN_LOG_ERROR("GXFSchedulingTermWrapper::start() - Operator is not set");
    return GXF_FAILURE;
  }
  if (status_type == nullptr) {
    HOLOSCAN_LOG_ERROR("GXFSchedulingTermWrapper::check_abi() - status_type is nullptr");
    return GXF_FAILURE;
  }
  if (target_timestamp == nullptr) {
    HOLOSCAN_LOG_ERROR("GXFSchedulingTermWrapper::check_abi() - target_timestamp is nullptr");
    return GXF_FAILURE;
  }

  HOLOSCAN_LOG_TRACE("Starting check of condition: {}", condition_->name());
  try {
    holoscan::SchedulingStatusType holoscan_type{};
    condition_->check(timestamp, &holoscan_type, target_timestamp);
    *status_type = to_gxf_scheduling_type(holoscan_type);
  } catch (const std::exception& e) {
    store_exception();
    HOLOSCAN_LOG_ERROR(
        "Exception occurred when checking condition: '{}' - {}", condition_->name(), e.what());
    return GXF_FAILURE;
  }
  return GXF_SUCCESS;
}

gxf_result_t GXFSchedulingTermWrapper::update_state_abi(int64_t timestamp) {
  HOLOSCAN_LOG_TRACE("GXFSchedulingTermWrapper::update_state_abi()");
  if (condition_ == nullptr) {
    HOLOSCAN_LOG_ERROR("GXFSchedulingTermWrapper::update_state_abi() - Condition is not set");
    return GXF_FAILURE;
  }

  HOLOSCAN_LOG_TRACE("Updating state of condition: {}", condition_->name());
  try {
    condition_->update_state(timestamp);
  } catch (const std::exception& e) {
    store_exception();
    HOLOSCAN_LOG_ERROR("Exception occurred when checking state of condition: '{}' - {}",
                       condition_->name(),
                       e.what());
    return GXF_FAILURE;
  }
  return GXF_SUCCESS;
}

gxf_result_t GXFSchedulingTermWrapper::onExecute_abi(int64_t timestamp) {
  HOLOSCAN_LOG_TRACE("GXFSchedulingTermWrapper::onExecute_abi()");
  if (condition_ == nullptr) {
    HOLOSCAN_LOG_ERROR("GXFSchedulingTermWrapper::onExecute_abi() - Condition is not set");
    return GXF_FAILURE;
  }

  HOLOSCAN_LOG_TRACE("Calling on_execute of condition: {}", condition_->name());
  try {
    condition_->on_execute(timestamp);
  } catch (const std::exception& e) {
    store_exception();
    HOLOSCAN_LOG_ERROR("Exception occurred during on_execute for condition: '{}' - {}",
                       condition_->name(),
                       e.what());
    return GXF_FAILURE;
  }
  return GXF_SUCCESS;
}

void GXFSchedulingTermWrapper::store_exception() const {
  auto stored_exception = std::current_exception();
  if (stored_exception != nullptr) {
    condition_->fragment()->executor().exception(stored_exception);
  }
}

}  // namespace holoscan::gxf

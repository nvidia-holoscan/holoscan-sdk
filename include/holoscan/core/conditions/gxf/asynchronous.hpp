/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_CORE_CONDITIONS_GXF_ASYNCHRONOUS_HPP
#define HOLOSCAN_CORE_CONDITIONS_GXF_ASYNCHRONOUS_HPP

#include <string>

#include <gxf/std/scheduling_terms.hpp>

#include "../../gxf/gxf_condition.hpp"

namespace holoscan {

/**
 * @brief Condition class to support asynchronous execution of operators.
 *
 * This condition waits on an asynchronous event which can happen outside of the regular compute
 * function of an operator.
 *
 * The method `event_state()` method is used to get or set the asynchronous condition's state.
 * The possible states are:
 *   - AsynchronousEventState::READY          ///< Initial state, first compute call is pending
 *   - AsynchronousEventState::WAIT           ///< Request to async service yet to be sent,
 *                                                 nothing to do but wait
 *   - AsynchronousEventState::EVENT_WAITING  ///< Request sent to an async service, pending event
 *                                                 done notification
 *   - AsynchronousEventState::EVENT_DONE     ///< Event done notification received, entity ready
 *                                                 to compute
 *   - AsynchronousEventState::EVENT_NEVER    ///< Entity will not call compute again, end of
 *                                                 execution
 *
 * This class wraps GXF SchedulingTerm(`nvidia::gxf::AsynchronousSchedulingTerm`). The event used
 * corresponds to `gxf_event_t` enum value `GXF_EVENT_EXTERNAL` which is supported by all
 * schedulers.
 *
 * ==Parameters==
 *
 * None
 */
class AsynchronousCondition : public gxf::GXFCondition {
 public:
  HOLOSCAN_CONDITION_FORWARD_ARGS_SUPER(AsynchronousCondition, GXFCondition)

  AsynchronousCondition() = default;
  AsynchronousCondition(const std::string& name, nvidia::gxf::AsynchronousSchedulingTerm* term);

  const char* gxf_typename() const override { return "nvidia::gxf::AsynchronousSchedulingTerm"; }

  /**
   * @brief Set the condition's asynchronous event state.
   *
   * @param state The state to which the condition should be set.
   */
  void event_state(AsynchronousEventState state);

  /**
   * @brief Get the asynchronous event state
   *
   * @return The current state of the condition.
   */
  AsynchronousEventState event_state() const;

  nvidia::gxf::AsynchronousSchedulingTerm* get() const;

 private:
  AsynchronousEventState event_state_{AsynchronousEventState::READY};
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_CONDITIONS_GXF_ASYNCHRONOUS_HPP */

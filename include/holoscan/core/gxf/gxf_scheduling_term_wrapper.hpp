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

#ifndef HOLOSCAN_CORE_GXF_GXF_SCHEDULING_TERM_WRAPPER_HPP
#define HOLOSCAN_CORE_GXF_GXF_SCHEDULING_TERM_WRAPPER_HPP

#include "holoscan/core/gxf/gxf_condition.hpp"

#include <memory>

#include "../condition.hpp"
#include "./gxf_execution_context.hpp"
#include "gxf/core/parameter_parser_std.hpp"
#include "gxf/core/registrar.hpp"
#include "gxf/std/scheduling_condition.hpp"
#include "gxf/std/scheduling_term.hpp"

namespace holoscan::gxf {

/**
 * @brief Class to wrap a native Condition into a GXF SchedulingTerm.
 *
 */
class GXFSchedulingTermWrapper : public nvidia::gxf::SchedulingTerm {
 public:
  virtual ~GXFSchedulingTermWrapper() = default;

  gxf_result_t initialize() override;
  gxf_result_t deinitialize() override;

  gxf_result_t registerInterface(nvidia::gxf::Registrar* registrar) override;

  /**
   * @brief Get the condition on which the scheduling waits before allowing execution.
   *
   * If the term is waiting for a time event 'target_timestamp' will contain the target timestamp.
   *
   * @param timestamp The current timestamp
   * @param status_type The status of the scheduling condition
   * @param target_timestamp The target timestamp (used if the term is waiting for a time event).
   */
  gxf_result_t check_abi(int64_t timestamp, nvidia::gxf::SchedulingConditionType* status_type,
                         int64_t* target_timestamp) const override;

  /**
   * @brief Called each time after the entity of this term was executed.
   *
   * @param timestamp The current timestamp
   */
  gxf_result_t onExecute_abi(int64_t timestamp) override;

  /**
   * @brief Checks if the state of the scheduling term can be updated and updates it
   *
   * @param timestamp The current timestamp
   */
  gxf_result_t update_state_abi(int64_t timestamp) override;

  /**
   * @brief Set the Condition object to be wrapped.
   *
   * @param condition The pointer to the native Condition object.
   */
  void set_condition(std::shared_ptr<Condition> condition) { condition_ = condition; }

 private:
  void store_exception() const;

  std::shared_ptr<Condition> condition_{};
};

}  // namespace holoscan::gxf

#endif /* HOLOSCAN_CORE_GXF_GXF_SCHEDULING_TERM_WRAPPER_HPP */

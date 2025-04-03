/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_CORE_OPERATOR_STATUS_HPP
#define HOLOSCAN_CORE_OPERATOR_STATUS_HPP

#include <cstdint>

namespace holoscan {

/**
 * @brief Enum representing the status of an operator.
 *
 * This enum defines the possible statuses of an operator in the Holoscan SDK.
 * It maps the GXF entity status values to human-readable operator statuses.
 */
enum class OperatorStatus {
  kNotStarted = 0,  ///< Operator is created but not started (GXF_ENTITY_STATUS_NOT_STARTED)
  kStartPending,    ///< Operator is pending to start (GXF_ENTITY_STATUS_START_PENDING)
  kStarted,         ///< Operator is started (GXF_ENTITY_STATUS_STARTED)
  kTickPending,     ///< Operator is pending to tick (compute) (GXF_ENTITY_STATUS_TICK_PENDING)
  kTicking,         ///< Operator is currently ticking (in compute) (GXF_ENTITY_STATUS_TICKING)
  kIdle,            ///< Operator is idle (GXF_ENTITY_STATUS_IDLE)
  kStopPending,     ///< Operator is pending to stop (GXF_ENTITY_STATUS_STOP_PENDING)
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_OPERATOR_STATUS_HPP */

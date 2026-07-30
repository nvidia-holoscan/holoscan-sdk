/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

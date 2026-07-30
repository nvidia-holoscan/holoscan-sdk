/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef HOLOSCAN_CORE_EXECUTORS_GPU_RESIDENT_CONTROLCOMMAND_HPP_
#define HOLOSCAN_CORE_EXECUTORS_GPU_RESIDENT_CONTROLCOMMAND_HPP_

namespace holoscan {

enum class ControlCommand : unsigned int {
  INVALID = 0,
  DATA_NOT_READY,
  DATA_READY,
  RESULT_READY,
  RESULT_NOT_READY,
  TEAR_DOWN,
  MAX_COMMANDS
};

}  // namespace holoscan

#endif  // HOLOSCAN_CORE_EXECUTORS_GPU_RESIDENT_CONTROLCOMMAND_HPP_

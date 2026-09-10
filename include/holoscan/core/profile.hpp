/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace holoscan {

/// @brief Identify one closed baseline profile request.
enum class Profile : std::uint8_t {
  kStandard = 0,   ///< General-purpose default capability policy.
  kRobotics = 1,   ///< Robotics-oriented capability and diagnostic policy.
  kRealTime = 2,   ///< Bounded real-time-oriented capability policy.
  kRegulated = 3,  ///< Regulated-domain capability and evidence policy.
  kResearch = 4,   ///< Research-oriented capability policy.
};

}  // namespace holoscan

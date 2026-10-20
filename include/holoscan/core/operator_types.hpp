/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace holoscan {

/// @brief Classify the implementation boundary hosting one Operator callback surface.
enum class OperatorHostKind : std::uint8_t {
  kNative = 0,   ///< Ordinary native C++ Operator implementation.
  kForeign = 1,  ///< Language-neutral callback-table host.
};

}  // namespace holoscan

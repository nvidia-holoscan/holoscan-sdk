/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>

namespace holoscan::channel_internal {

/// Maximum payload carried by the fixed HoloMQ inline-payload endpoint family.
inline constexpr std::size_t kMaximumInlinePayloadBytes = 32U * 1024U;

}  // namespace holoscan::channel_internal

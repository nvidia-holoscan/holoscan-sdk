// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace holoviz {

/// How a rendered frame is presented to the display.
///
/// \c kAuto selects the best tear-free mode the platform supports: \c kFIFOLatestReady if
/// available, otherwise \c kFIFO. \c kAuto never selects \c kImmediate; that mode must be
/// requested explicitly. See \c PresentBackend::create_swapchain().
enum class PresentMode : uint8_t {
  kAuto,             ///< backend picks the best available tear-free mode (never kImmediate)
  kFIFO,             ///< tear-free; waits for the next vertical blank
  kFIFOLatestReady,  ///< tear-free; drops stale queued frames, presents the newest one
  kMailbox,    ///< tear-free; drops stale queued frames, presents the newest one (May skip present
               ///< IDs)
  kImmediate,  ///< no vsync wait; may tear — benchmarking / explicit opt-in only
};

}  // namespace holoviz

/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_VIZ_HOLOVIZ_INIT_FLAGS_HPP
#define HOLOSCAN_VIZ_HOLOVIZ_INIT_FLAGS_HPP

#include <cstdint>

namespace holoscan::viz {

namespace init_flags {

/// Flags passed to the init function
typedef enum {
  NONE = 0x00000000,        ///< none
  FULLSCREEN = 0x00000001,  ///< switch the app to full screen mode
  HEADLESS = 0x00000002     ///< run in headless mode
} InitFlags;

}  // namespace init_flags

using InitFlags = init_flags::InitFlags;

}  // namespace holoscan::viz

#endif /* HOLOSCAN_VIZ_HOLOVIZ_INIT_FLAGS_HPP */

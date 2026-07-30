/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef MODULES_HOLOVIZ_SRC_HOLOVIZ_DISPLAY_EVENT_TYPE_HPP
#define MODULES_HOLOVIZ_SRC_HOLOVIZ_DISPLAY_EVENT_TYPE_HPP

namespace holoscan::viz {

enum class DisplayEventType {
  FIRST_PIXEL_OUT,  // signaled when the first pixel of the next display refresh cycle leaves the
                    // display engine for the display
};

}  // namespace holoscan::viz

#endif /* MODULES_HOLOVIZ_SRC_HOLOVIZ_DISPLAY_EVENT_TYPE_HPP */

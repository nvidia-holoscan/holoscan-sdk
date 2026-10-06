/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Pure WindowSinkDesc -> holoviz::WindowDesc mapping, kept free of Holoscan runtime types so the
// operator's window policy is unit-testable without a GPU or window system.

#pragma once

#include <holoscan/visualization/sink_desc.hpp>
#include <holoviz/window.hpp>

namespace holoscan::visualization::detail {

inline constexpr uint32_t kDefaultWindowWidth = 1280U;
inline constexpr uint32_t kDefaultWindowHeight = 720U;

/// Return the HoloViz window description for \p sink. A decorated window defaults each zero
/// dimension independently (1280x720); a fullscreen window with a zero width or height takes
/// the monitor's current video mode ({0, 0} to HoloViz). HoloViz's window layer does the
/// monitor lookup, mode selection, and mode restoration.
inline holoviz::WindowDesc window_desc_from_sink(const WindowSinkDesc& sink) {
  const bool monitor_sized = sink.fullscreen && (sink.width == 0U || sink.height == 0U);
  return holoviz::WindowDesc{
      .title = sink.title,
      .size = monitor_sized
                  ? holoviz::Extent2D{0U, 0U}
                  : holoviz::Extent2D{sink.width != 0U ? sink.width : kDefaultWindowWidth,
                                      sink.height != 0U ? sink.height : kDefaultWindowHeight},
      .resizable = !sink.fullscreen,
      .fullscreen = sink.fullscreen,
      .display_name = sink.display_name,
      .refresh_rate_mhz = sink.refresh_hz * 1000U,
      .close_on_escape = sink.close_on_escape,
  };
}

}  // namespace holoscan::visualization::detail

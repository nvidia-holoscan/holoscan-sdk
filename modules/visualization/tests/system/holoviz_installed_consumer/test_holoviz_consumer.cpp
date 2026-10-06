// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <holoviz/holoviz.hpp>
#include <holoviz/imgui/ui.hpp>
#include <holoviz/logger.hpp>
#include <holoviz/text.hpp>

#ifdef HOLOVIZ_STATIC_DEFINE
#error "Shared HoloViz must not propagate HOLOVIZ_STATIC_DEFINE"
#endif

int main() {
  // Empty handles exercise exported HoloViz symbols and expected results without initializing a
  // GPU.
  holoviz::Buffer buffer;
  const auto slice = buffer.whole();
  if (slice || slice.error().code != holoviz::ErrorCode::kInvalidState) {
    return 1;
  }
  if (!buffer.destroy()) {
    return 2;
  }
  holoviz::set_log_level(holoviz::LogLevel::kOff);
  return holoviz::log_level() == holoviz::LogLevel::kOff ? 0 : 3;
}

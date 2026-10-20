// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Native close request for windowed tests: sends WM_DELETE_WINDOW to a top-level X11 window
// through the test's own Xlib connection, as a window manager does when the user clicks the
// close button. The window is found by its exact WM_NAME, so the test never hands libholoviz's
// private GLFWwindow* to another GLFW instance. Xlib headers stay in the .cpp: their macros
// (None, Status, Bool, ...) collide with gtest and HoloViz names.

#pragma once

#include <chrono>
#include <string>

namespace holoviz::test_util {

enum class X11CloseRequest {
  kSent,            ///< WM_DELETE_WINDOW was sent and flushed
  kNoDisplay,       ///< XOpenDisplay failed (no X server reachable)
  kWindowNotFound,  ///< no window with that title appeared within the timeout
};

/// Send WM_DELETE_WINDOW to the X11 window titled \p title, searching the window tree until
/// \p timeout expires. A GLFW window on a native Wayland connection is never found. Windows
/// that disappear during the search are skipped. Installs a process-global Xlib error handler
/// for the duration of the call, so no other thread may install or rely on one concurrently
/// (in practice: no concurrent GLFW calls on X11).
X11CloseRequest send_x11_close_request(const std::string& title, std::chrono::milliseconds timeout);

}  // namespace holoviz::test_util

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>

#include <memory>

#include <holoviz/window.hpp>

#include "window_systems/window_util.hpp"  // window_systems::NativeMonitorInfo

namespace holoviz::detail {

class SurfaceImpl;

/// Backend interface for Window; see the Window handle for API documentation.
class WindowImpl {
 public:
  using CloseCallback = Window::CloseCallback;

  WindowImpl(const WindowImpl&) = delete;
  WindowImpl& operator=(const WindowImpl&) = delete;
  WindowImpl(WindowImpl&&) = delete;
  WindowImpl& operator=(WindowImpl&&) = delete;
  virtual ~WindowImpl() = default;

  /// Release all owned backend resources now. \p timeout bounds how long to
  /// wait for the object to stop being in use: zero (the default) fails
  /// immediately with ErrorCode::kBackpressured if the object is in use, and a
  /// non-zero timeout waits up to that duration before returning
  /// ErrorCode::kBackpressured. Backends without in-use tracking ignore the
  /// timeout and destroy immediately. Idempotent: repeated calls are no-ops
  /// returning success. Called by the destructor if not already destroyed.
  [[nodiscard]] virtual expected<void> destroy(
      std::chrono::nanoseconds timeout = std::chrono::nanoseconds::zero()) = 0;

  virtual WindowSystem system() const = 0;
  virtual Extent2D extent() const = 0;
  virtual Extent2D window_size() const = 0;
  virtual float content_scale() const = 0;
  virtual bool minimized() const = 0;
  virtual bool close_requested() const = 0;
  virtual void set_close_callback(void* user_data, CloseCallback callback) = 0;
  [[nodiscard]] virtual expected<void> set_input_callbacks(
      const WindowInputCallbacks& callbacks) = 0;
  [[nodiscard]] virtual expected<void> set_event_callbacks(
      const WindowEventCallbacks& callbacks) = 0;
  [[nodiscard]] virtual expected<NativeObject> native_object(NativeObjectType type) const = 0;

  /// Create a presentable Surface for this window. The caller wraps the
  /// returned SurfaceImpl in a Surface handle.
  [[nodiscard]] virtual expected<std::unique_ptr<SurfaceImpl>> create_surface() const = 0;

  /// The window system's view of the monitor this window occupies (fullscreen monitor, or
  /// the one with the largest overlap), for holoviz::display_for_window()'s join against
  /// enumerate_displays(). Default: kNotImplemented for window systems without monitor
  /// geometry.
  [[nodiscard]] virtual expected<window_systems::NativeMonitorInfo> current_monitor() const {
    return make_unexpected(
        Error{ErrorCode::kNotImplemented,
              "This window system does not report monitor geometry for windows."});
  }

 protected:
  WindowImpl() = default;
};

}  // namespace holoviz::detail

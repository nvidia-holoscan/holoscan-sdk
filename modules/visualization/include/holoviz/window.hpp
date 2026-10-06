/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Window: device-independent OS window. Windows are created before, and
// independently of, any Device — that ordering is what lets
// Adapter::supports_surface() and create_device_for_surface() pick a GPU that
// can actually present to the surface.
//
// Typical bring-up sequence:
//   auto window  = create_window(desc).value();
//   auto surface = window.create_surface().value();
//   auto device  = create_device_for_surface(device_desc, surface).value();
//   auto sc      = device.create_swapchain({.surface = surface, ...}).value();

#pragma once

#include <chrono>
#include <memory>
#include <string>

#include <holoviz/holoviz_export.h>

#include "common.hpp"
#include "surface.hpp"

namespace holoviz {

enum class WindowSystem : uint8_t {
  kAuto,  ///< 5.0: resolves to kGlfw.
  kGlfw,
  // Reserved: kWin32, kWayland, kX11, kQnxScreen.
};

struct WindowDesc {
  std::string title;
  /// Window size. For fullscreen windows, {0, 0} selects the monitor's current video mode.
  Extent2D size{1280, 720};
  /// Resizable windows report their live framebuffer size through extent(); the caller
  /// owns swapchain recreation (react to kOutOfDate from acquire()/present(), or compare
  /// extent() against the swapchain per frame).
  bool resizable = false;
  WindowSystem provider = WindowSystem::kAuto;
  /// Create the window fullscreen on the monitor named by \ref display_name. Fullscreen
  /// windows are created undecorated (required for flip-model presentation / G-Sync) and
  /// are not user-resizable: \ref resizable is ignored. Swapchains built on the window still
  /// recreate themselves automatically when the surface goes out of date (video-mode or
  /// focus changes), as for a resizable window. The original video mode is restored when
  /// the window is destroyed. Creation fails with ErrorCode::kFailure when no monitor is
  /// available or its current video mode cannot be queried.
  bool fullscreen = false;
  /// Monitor for fullscreen placement (GLFW monitor name, matched exactly). Empty selects
  /// the primary monitor; an unknown name logs the available monitors and falls back to
  /// the primary. Ignored unless \ref fullscreen is set.
  std::string display_name;
  /// Refresh rate in millihertz for the fullscreen video mode; 0 keeps the monitor's
  /// current rate. The window system picks the closest supported mode for \ref size and
  /// this rate. Ignored unless \ref fullscreen is set.
  uint32_t refresh_rate_mhz = 0;
  /// Request close when ESC is pressed (close_requested() then reports true and the close
  /// callback fires). Off by default: applications receive the raw key stream through
  /// WindowInputCallbacks and may consume ESC themselves; enable for HoloViz 4.x behavior.
  bool close_on_escape = false;
};

/// Raw input callbacks, delivered during poll_window_events() on the main thread.
/// Codes and actions are the window system's native values (GLFW_* constants for
/// WindowSystem::kGlfw); consumers translate. One consumer per window: registering a
/// second non-empty set fails with kInvalidState; pass a default-constructed struct to
/// clear. Null members are simply not delivered.
struct WindowInputCallbacks {
  void* user_data = nullptr;
  void (*cursor_pos)(void* user_data, double x, double y) = nullptr;  ///< window coordinates
  void (*mouse_button)(void* user_data, int button, int action, int mods) = nullptr;
  void (*scroll)(void* user_data, double dx, double dy) = nullptr;
  void (*key)(void* user_data, int key, int scancode, int action, int mods) = nullptr;
  void (*char_input)(void* user_data, unsigned int codepoint) = nullptr;
  void (*focus)(void* user_data, int focused) = nullptr;

  [[nodiscard]] bool empty() const {
    return cursor_pos == nullptr && mouse_button == nullptr && scroll == nullptr &&
           key == nullptr && char_input == nullptr && focus == nullptr;
  }
};

/// Window-geometry callbacks, delivered during poll_window_events() on the main thread.
/// Each non-null member is additionally invoked once, synchronously, at registration with
/// the current value, so callers need no separate initial query. One consumer per window:
/// registering a second non-empty set fails with kInvalidState; pass a default-constructed
/// struct to clear. Null members are simply not delivered.
struct WindowEventCallbacks {
  void* user_data = nullptr;
  /// Framebuffer size in pixels (what swapchains are sized to).
  void (*framebuffer_size)(void* user_data, uint32_t width, uint32_t height) = nullptr;
  /// Window size in screen coordinates (differs from framebuffer_size on HiDPI).
  void (*window_size)(void* user_data, uint32_t width, uint32_t height) = nullptr;
  /// HiDPI content scale (pixels per UI unit) per axis.
  void (*content_scale)(void* user_data, float x_scale, float y_scale) = nullptr;

  [[nodiscard]] bool empty() const {
    return framebuffer_size == nullptr && window_size == nullptr && content_scale == nullptr;
  }
};

namespace detail {
class WindowImpl;
}

/// A native OS window. Not thread-safe: create, poll, and destroy windows on
/// the main thread (a GLFW requirement on some platforms).
class Window {
 public:
  using CloseCallback = void (*)(void* user_data);

  HOLOVIZ_EXPORT explicit Window(std::unique_ptr<detail::WindowImpl> impl);
  HOLOVIZ_EXPORT Window();
  HOLOVIZ_EXPORT Window(Window&&) noexcept;
  HOLOVIZ_EXPORT Window& operator=(Window&&) noexcept;
  Window(const Window&) = delete;
  Window& operator=(const Window&) = delete;
  HOLOVIZ_EXPORT ~Window();

  explicit operator bool() const { return impl_ != nullptr; }

  /// Release the underlying backend resources now instead of at destruction.
  /// \p timeout bounds how long to wait for the object to stop being in use:
  /// zero (the default) fails immediately with ErrorCode::kBackpressured if
  /// the object is in use; a non-zero timeout waits up to that duration before
  /// returning ErrorCode::kBackpressured. Only CommandList currently tracks
  /// in-use state; other objects destroy immediately. On success (and on any
  /// error other than kBackpressured) the handle becomes empty. Safe to call
  /// on an empty handle (no-op success). The destructor calls this
  /// automatically and logs any error.
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> destroy(
      std::chrono::nanoseconds timeout = std::chrono::nanoseconds::zero());

  [[nodiscard]] HOLOVIZ_EXPORT expected<WindowSystem> system() const;

  /// Return the framebuffer size in pixels (what swapchains are sized to).
  [[nodiscard]] HOLOVIZ_EXPORT expected<Extent2D> extent() const;

  /// Return the window size in screen coordinates (differs from extent() on HiDPI).
  [[nodiscard]] HOLOVIZ_EXPORT expected<Extent2D> window_size() const;

  /// Return the HiDPI scale factor (pixels per UI unit).
  [[nodiscard]] HOLOVIZ_EXPORT expected<float> content_scale() const;

  /// Return whether the window is iconified/minimized (zero presentable area). Poll after
  /// poll_window_events(); pacing while minimized is the caller's policy (swapchain
  /// acquire reports kBackpressured on a zero-extent surface).
  [[nodiscard]] HOLOVIZ_EXPORT expected<bool> minimized() const;

  /// Return whether the user asked to close the window (e.g. clicked the
  /// close button). Poll this as the frame-loop condition; the request is
  /// only observed after poll_window_events().
  [[nodiscard]] HOLOVIZ_EXPORT expected<bool> close_requested() const;
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> set_close_callback(void* user_data,
                                                                 CloseCallback callback);

  /// Register raw input callbacks (see WindowInputCallbacks). kInvalidState when a
  /// non-empty registration already exists and \p callbacks is also non-empty.
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> set_input_callbacks(
      const WindowInputCallbacks& callbacks);

  /// Register window-geometry callbacks (see WindowEventCallbacks). Each non-null member
  /// is invoked once, synchronously, with the current value before this call returns.
  /// kInvalidState when a non-empty registration already exists and \p callbacks is also
  /// non-empty.
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> set_event_callbacks(
      const WindowEventCallbacks& callbacks);

  /// Create a presentable Surface from this window. The window must outlive
  /// the returned Surface, and the Surface must outlive any swapchain built
  /// from it.
  [[nodiscard]] HOLOVIZ_EXPORT expected<Surface> create_surface() const;

  /// GLFW: answers kGLFWwindow.
  ///
  /// The returned `GLFWwindow*` belongs to HoloViz's private GLFW instance, which is linked
  /// statically into libholoviz with hidden symbols. It identifies the window for interop
  /// bookkeeping only. Passing it to any GLFW entry point the application links itself is
  /// undefined behaviour: that GLFW has separate global state and its own struct layout. Deliver
  /// input through \ref set_input_callbacks or imgui::Renderer::install_input_callbacks, and pump
  /// events with \ref poll_window_events.
  [[nodiscard]] HOLOVIZ_EXPORT expected<NativeObject> native_object(NativeObjectType type) const;

  /// Backend-internal access; not part of the stable API surface.
  /// Precondition: the handle is non-empty.
  detail::WindowImpl& impl() { return *impl_; }
  const detail::WindowImpl& impl() const { return *impl_; }

 private:
  std::unique_ptr<detail::WindowImpl> impl_;
};

[[nodiscard]] HOLOVIZ_EXPORT expected<Window> create_window(const WindowDesc& desc);

/// Poll the window system's event queue for all windows. Call once per frame
/// from the main thread; input and close events are only delivered here.
HOLOVIZ_EXPORT void poll_window_events();

}  // namespace holoviz

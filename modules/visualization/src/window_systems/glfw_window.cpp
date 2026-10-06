/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// vk_impls.hpp must be the very first include so that VK_VERSION_1_0 is
// defined before glfw_window.hpp pulls in <GLFW/glfw3.h>. GLFW only
// declares glfwCreateWindowSurface when that macro is already visible.
#include "detail/destroy_timeout.hpp"
#include "vulkan/vk_impls.hpp"  // IWYU pragma: keep

#include "glfw_window.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "vulkan/vk_adapter.hpp"
#include "vulkan/vk_errors.hpp"
#include "vulkan/vk_surface.hpp"
#include "window_util.hpp"

namespace holoviz::glfw {

static uint32_t glfw_window_count = 0;

namespace {

/// Initialize GLFW for the first window-system user (no-op while glfw_window_count > 0). On
/// Linux the platform is chosen from the environment (window_util.hpp, choose_glfw_platform)
/// instead of letting GLFW probe blindly: GLFW 3.4 tries Wayland before X11, and that probe makes
/// libwayland-client print "XDG_RUNTIME_DIR is invalid or not set" to stderr whenever no Wayland
/// route exists. The environment only says which connection routes are *possible*; GLFW performs
/// the actual connection attempt. With no route to any window system glfwInit() is skipped and
/// kNotFound tells the caller why (headless deployments only need direct-display surfaces). Does
/// not touch glfw_window_count; callers keep their own refcounting.
expected<void> glfw_init_shared() {
  if (glfw_window_count > 0) {
    return {};
  }
#if defined(__linux__)
  const window_systems::WindowSystemEnv env{
      .wayland_display = std::getenv("WAYLAND_DISPLAY") != nullptr,
      .wayland_socket = std::getenv("WAYLAND_SOCKET") != nullptr,
      .xdg_runtime_dir = std::getenv("XDG_RUNTIME_DIR") != nullptr,
      .x11_display = std::getenv("DISPLAY") != nullptr,
  };
  switch (window_systems::choose_glfw_platform(env)) {
    case window_systems::GLFWPlatformChoice::kNone:
      return make_unexpected(
          Error{ErrorCode::kNotFound,
                "[holoviz] No window system route: DISPLAY, WAYLAND_DISPLAY, WAYLAND_SOCKET and "
                "XDG_RUNTIME_DIR are all unset."});
    case window_systems::GLFWPlatformChoice::kX11Only:
      // Only X11 reachable: say so, otherwise GLFW would probe Wayland first.
      glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11);
      break;
    case window_systems::GLFWPlatformChoice::kAny:
      // A Wayland route may exist; GLFW chooses (it honors XDG_SESSION_TYPE when both are
      // reachable) and falls back to X11 when the Wayland connection fails.
      glfwInitHint(GLFW_PLATFORM, GLFW_ANY_PLATFORM);
      break;
  }
#endif
  if (!glfwInit()) {
    return make_unexpected(Error{ErrorCode::kFailure, "Failed to initialize GLFW"});
  }
  return {};
}

/// Resolve the fullscreen monitor for \p desired_name: exact name match, else the primary
/// monitor (with a warning listing the available monitor names). Empty name = primary.
GLFWmonitor* resolve_monitor(const std::string& desired_name) {
  GLFWmonitor* primary = glfwGetPrimaryMonitor();
  if (desired_name.empty()) {
    return primary;
  }

  int monitor_count = 0;
  GLFWmonitor** monitors = glfwGetMonitors(&monitor_count);
  std::vector<std::string_view> names;
  names.reserve(static_cast<size_t>(monitor_count));
  for (int i = 0; i < monitor_count; ++i) {
    const char* name = glfwGetMonitorName(monitors[i]);
    names.emplace_back(name != nullptr ? name : "");
  }

  if (const auto index = window_systems::find_monitor_by_name(names, desired_name)) {
    return monitors[*index];
  }

  HOLOVIZ_LOG_WARN("[holoviz] Display \"{}\" not found, using the primary display instead",
                   desired_name);
  HOLOVIZ_LOG_WARN("[holoviz] Available displays:");
  for (const std::string_view name : names) {
    HOLOVIZ_LOG_WARN("[holoviz]  \"{}\"", name);
  }
  return primary;
}

}  // namespace

expected<void> GlfwWindow::initialize(const WindowDesc& desc) {
  VIZ_FORWARD_ERROR(glfw_init_shared());

  glfw_window_count++;
  glfw_refcounted_ = true;

  fullscreen_ = desc.fullscreen;
  close_on_escape_ = desc.close_on_escape;

  GLFWmonitor* monitor = nullptr;
  Extent2D size = desc.size;
  int refresh_rate = GLFW_DONT_CARE;
  if (desc.fullscreen) {
    // Both failures below return before any window exists; the destructor's destroy() drops
    // the GLFW refcount taken above.
    monitor = resolve_monitor(desc.display_name);
    if (monitor == nullptr) {
      return make_unexpected(
          Error{ErrorCode::kFailure, "[holoviz] No monitor is available for a fullscreen window"});
    }
    const GLFWvidmode* current_mode = glfwGetVideoMode(monitor);
    if (current_mode == nullptr) {
      const char* monitor_name = glfwGetMonitorName(monitor);
      return make_unexpected(
          Error{ErrorCode::kFailure,
                std::string("[holoviz] Could not query the current video mode of monitor \"") +
                    (monitor_name != nullptr ? monitor_name : "") + "\" for a fullscreen window"});
    }
    // A zero size takes the monitor's current mode; a zero refresh rate keeps the current
    // one (GLFW_DONT_CARE would pick the highest rate at the size). GLFW then selects the
    // closest supported mode for the size and rate.
    size = window_systems::resolve_window_size(
        desc.size,
        /*fullscreen=*/true,
        {static_cast<uint32_t>(current_mode->width), static_cast<uint32_t>(current_mode->height)});
    refresh_rate =
        window_systems::select_refresh_rate_hz(desc.refresh_rate_mhz, current_mode->refreshRate);
  }
  // Fullscreen windows must be undecorated to switch to flip-model presentation and
  // enable features like G-Sync. Window hints are sticky, so set it both ways.
  glfwWindowHint(GLFW_DECORATED, desc.fullscreen ? GLFW_FALSE : GLFW_TRUE);

  // Fullscreen windows are fixed-size from the API's perspective (WindowDesc documents
  // `resizable` as ignored there), and resizable_ hands that contract to the surface. The
  // GLFW hint must still be TRUE for fullscreen windows: GLFW_RESIZABLE=FALSE writes fixed
  // min==max WM_NORMAL_HINTS, and Mutter (GNOME) refuses _NET_WM_STATE_FULLSCREEN for such
  // windows, leaving an undecorated window auto-placed on the primary monitor next to the
  // dock instead of covering the requested monitor. GLFW ignores resizability for monitor
  // windows anyway: the window system sizes them to the monitor.
  resizable_ = desc.resizable && !desc.fullscreen;
  glfwWindowHint(GLFW_RESIZABLE, (resizable_ || desc.fullscreen) ? GLFW_TRUE : GLFW_FALSE);
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  // Fullscreen video-mode refresh rate (resolved above); sticky hint, so set it both ways.
  glfwWindowHint(GLFW_REFRESH_RATE, refresh_rate);
  window_ = glfwCreateWindow(size.width, size.height, desc.title.c_str(), monitor, nullptr);

  if (!window_) {
    return make_unexpected(Error{ErrorCode::kFailure, "Failed to create GLFW window"});
  }

  extent_ = size;
  float y_scale = 1.0F;
  glfwGetWindowContentScale(window_, &content_scale_, &y_scale);

  install_callbacks();

  return {};
}

void GlfwWindow::install_callbacks() {
  // This window owns the GLFW user pointer. The trampolines are installed once and no-op
  // while no consumer is registered; events are delivered during poll_events() on the
  // main thread.
  glfwSetWindowUserPointer(window_, this);
  glfwSetWindowCloseCallback(window_, [](GLFWwindow* w) {
    auto* self = static_cast<GlfwWindow*>(glfwGetWindowUserPointer(w));
    if (self != nullptr) {
      if (self->close_callback_ != nullptr) {
        self->close_callback_(self->close_user_data_);
      }
    }
  });
  glfwSetCursorPosCallback(window_, [](GLFWwindow* w, double x, double y) {
    auto* self = static_cast<GlfwWindow*>(glfwGetWindowUserPointer(w));
    if (self != nullptr) {
      if (self->input_callbacks_.cursor_pos != nullptr) {
        self->input_callbacks_.cursor_pos(self->input_callbacks_.user_data, x, y);
      }
    }
  });
  glfwSetMouseButtonCallback(window_, [](GLFWwindow* w, int button, int action, int mods) {
    auto* self = static_cast<GlfwWindow*>(glfwGetWindowUserPointer(w));
    if (self != nullptr) {
      if (self->input_callbacks_.mouse_button != nullptr) {
        self->input_callbacks_.mouse_button(self->input_callbacks_.user_data, button, action, mods);
      }
    }
  });
  glfwSetScrollCallback(window_, [](GLFWwindow* w, double dx, double dy) {
    auto* self = static_cast<GlfwWindow*>(glfwGetWindowUserPointer(w));
    if (self != nullptr) {
      if (self->input_callbacks_.scroll != nullptr) {
        self->input_callbacks_.scroll(self->input_callbacks_.user_data, dx, dy);
      }
    }
  });
  glfwSetKeyCallback(window_, [](GLFWwindow* w, int key, int scancode, int action, int mods) {
    auto* self = static_cast<GlfwWindow*>(glfwGetWindowUserPointer(w));
    if (self != nullptr) {
      if (self->close_on_escape_ && key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        // glfwSetWindowShouldClose does not fire the close callback (that is only for
        // window-system close attempts), so deliver it here for a uniform contract.
        glfwSetWindowShouldClose(w, GLFW_TRUE);
        if (self->close_callback_ != nullptr) {
          self->close_callback_(self->close_user_data_);
        }
      }
      if (self->input_callbacks_.key != nullptr) {
        self->input_callbacks_.key(self->input_callbacks_.user_data, key, scancode, action, mods);
      }
    }
  });
  glfwSetCharCallback(window_, [](GLFWwindow* w, unsigned int codepoint) {
    auto* self = static_cast<GlfwWindow*>(glfwGetWindowUserPointer(w));
    if (self != nullptr) {
      if (self->input_callbacks_.char_input != nullptr) {
        self->input_callbacks_.char_input(self->input_callbacks_.user_data, codepoint);
      }
    }
  });
  glfwSetWindowFocusCallback(window_, [](GLFWwindow* w, int focused) {
    auto* self = static_cast<GlfwWindow*>(glfwGetWindowUserPointer(w));
    if (self != nullptr) {
      if (self->input_callbacks_.focus != nullptr) {
        self->input_callbacks_.focus(self->input_callbacks_.user_data, focused);
      }
    }
  });
  glfwSetFramebufferSizeCallback(window_, [](GLFWwindow* w, int width, int height) {
    auto* self = static_cast<GlfwWindow*>(glfwGetWindowUserPointer(w));
    if (self != nullptr) {
      if (width > 0 && height > 0) {
        self->extent_ = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
      }
      if (self->event_callbacks_.framebuffer_size != nullptr) {
        self->event_callbacks_.framebuffer_size(self->event_callbacks_.user_data,
                                                static_cast<uint32_t>(width),
                                                static_cast<uint32_t>(height));
      }
    }
  });
  glfwSetWindowSizeCallback(window_, [](GLFWwindow* w, int width, int height) {
    auto* self = static_cast<GlfwWindow*>(glfwGetWindowUserPointer(w));
    if (self != nullptr) {
      if (self->event_callbacks_.window_size != nullptr) {
        self->event_callbacks_.window_size(self->event_callbacks_.user_data,
                                           static_cast<uint32_t>(width),
                                           static_cast<uint32_t>(height));
      }
    }
  });
  glfwSetWindowContentScaleCallback(window_, [](GLFWwindow* w, float x_scale, float y_scale) {
    auto* self = static_cast<GlfwWindow*>(glfwGetWindowUserPointer(w));
    if (self != nullptr) {
      self->content_scale_ = x_scale;
      if (self->event_callbacks_.content_scale != nullptr) {
        self->event_callbacks_.content_scale(self->event_callbacks_.user_data, x_scale, y_scale);
      }
    }
  });
}

GlfwWindow::~GlfwWindow() {
  if (expected<void> result = destroy(detail::kDestructorDestroyTimeout); !result) {
    try {
      HOLOVIZ_LOG_ERROR("[holoviz] GlfwWindow: destroy failed in destructor: {}",
                        result.error().message);
    } catch (...) {
    }
  }
}

expected<void> GlfwWindow::destroy([[maybe_unused]] std::chrono::nanoseconds timeout) {
  if (window_) {
    // Detach the trampolines' back-pointer before destruction so a late event
    // delivered during teardown cannot reach a dying object.
    glfwSetWindowUserPointer(window_, nullptr);
    if (fullscreen_) {
      // GLFW does not restore the original video mode when a fullscreen window is
      // destroyed; switch back to windowed first.
      int width = 0;
      int height = 0;
      glfwGetFramebufferSize(window_, &width, &height);
      glfwSetWindowMonitor(window_, nullptr, 0, 0, width, height, GLFW_DONT_CARE);
    }
    glfwDestroyWindow(window_);
    window_ = nullptr;
  }
  input_callbacks_ = {};
  event_callbacks_ = {};

  // Drop this window's reference on the process-wide GLFW refcount exactly
  // once, even when destroy() runs again from the destructor.
  if (glfw_refcounted_) {
    glfw_refcounted_ = false;

    if (glfw_window_count > 0) {
      glfw_window_count--;
    }

    if (glfw_window_count == 0) {
      glfwTerminate();
    }
  }

  return {};
}

Extent2D GlfwWindow::extent() const {
  // Live framebuffer size: a resizable window's extent changes after creation, and the
  // swapchain-recreation path sizes from it.
  if (window_ != nullptr) {
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window_, &width, &height);
    if (width > 0 && height > 0) {
      return {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
    }
  }
  return extent_;
}

Extent2D GlfwWindow::window_size() const {
  if (window_ != nullptr) {
    int width = 0;
    int height = 0;
    glfwGetWindowSize(window_, &width, &height);
    if (width > 0 && height > 0) {
      return {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
    }
  }
  return extent_;
}

bool GlfwWindow::minimized() const {
  return window_ != nullptr && glfwGetWindowAttrib(window_, GLFW_ICONIFIED) == GLFW_TRUE;
}

bool GlfwWindow::close_requested() const {
  return glfwWindowShouldClose(window_);
}

void GlfwWindow::set_close_callback(void* user_data, Window::CloseCallback callback) {
  close_user_data_ = user_data;
  close_callback_ = callback;
}

expected<void> GlfwWindow::set_input_callbacks(const WindowInputCallbacks& callbacks) {
  if (!input_callbacks_.empty() && !callbacks.empty()) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidState,
              .message = "[holoviz] Window::set_input_callbacks: input callbacks are already "
                         "registered for this window; clear them first"});
  }
  input_callbacks_ = callbacks;
  return {};
}

expected<void> GlfwWindow::set_event_callbacks(const WindowEventCallbacks& callbacks) {
  if (!event_callbacks_.empty() && !callbacks.empty()) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidState,
              .message = "[holoviz] Window::set_event_callbacks: event callbacks are already "
                         "registered for this window; clear them first"});
  }
  event_callbacks_ = callbacks;

  // Deliver the current values once so consumers need no separate initial query.
  if (window_ != nullptr) {
    if (callbacks.framebuffer_size != nullptr) {
      int width = 0;
      int height = 0;
      glfwGetFramebufferSize(window_, &width, &height);
      callbacks.framebuffer_size(
          callbacks.user_data, static_cast<uint32_t>(width), static_cast<uint32_t>(height));
    }
    if (callbacks.window_size != nullptr) {
      int width = 0;
      int height = 0;
      glfwGetWindowSize(window_, &width, &height);
      callbacks.window_size(
          callbacks.user_data, static_cast<uint32_t>(width), static_cast<uint32_t>(height));
    }
    if (callbacks.content_scale != nullptr) {
      float x_scale = 1.0F;
      float y_scale = 1.0F;
      glfwGetWindowContentScale(window_, &x_scale, &y_scale);
      callbacks.content_scale(callbacks.user_data, x_scale, y_scale);
    }
  }
  return {};
}

expected<NativeObject> GlfwWindow::native_object(NativeObjectType type) const {
  if (type == NativeObjectType::kGLFWwindow) {
    return NativeObject{type, reinterpret_cast<uint64_t>(window_)};
  }
  return make_unexpected(
      Error{ErrorCode::kInvalidArgument, "GLFW window backend only supports kGLFWwindow"});
}

expected<window_systems::NativeMonitorInfo> GlfwWindow::current_monitor() const {
  if (window_ == nullptr) {
    return make_unexpected(
        Error{ErrorCode::kInvalidState, "current_monitor() on a destroyed window"});
  }

  GLFWmonitor* monitor = glfwGetWindowMonitor(window_);  // non-null only for fullscreen windows
  if (monitor == nullptr) {
    // Windowed: pick the monitor with the largest overlap with the window rectangle. That
    // needs the window's global desktop position, which GLFW's Wayland platform cannot report
    // (glfwGetWindowPos() raises GLFW_FEATURE_UNAVAILABLE and leaves its outputs at 0, which
    // would silently select the output at the origin); Wayland hands out no output identity
    // for a window either, so the windowed route is unsupported there.
    if (glfwGetPlatform() == GLFW_PLATFORM_WAYLAND) {
      return make_unexpected(
          Error{ErrorCode::kNotImplemented,
                "[holoviz] Windowed display lookup needs the window's global position, which "
                "Wayland does not expose; use a fullscreen window."});
    }
    int window_x = 0;
    int window_y = 0;
    glfwGetWindowPos(window_, &window_x, &window_y);
    int window_w = 0;
    int window_h = 0;
    glfwGetWindowSize(window_, &window_w, &window_h);
    const Rect2D window_rect{.x = window_x,
                             .y = window_y,
                             .width = static_cast<uint32_t>(std::max(window_w, 0)),
                             .height = static_cast<uint32_t>(std::max(window_h, 0))};

    int monitor_count = 0;
    GLFWmonitor** monitors = glfwGetMonitors(&monitor_count);
    std::vector<Rect2D> monitor_rects;
    monitor_rects.reserve(static_cast<size_t>(std::max(monitor_count, 0)));
    for (int i = 0; i < monitor_count; ++i) {
      Rect2D rect{};  // a monitor without a video mode has no area and can never overlap
      if (const GLFWvidmode* mode = glfwGetVideoMode(monitors[i]); mode != nullptr) {
        glfwGetMonitorPos(monitors[i], &rect.x, &rect.y);
        rect.width = static_cast<uint32_t>(std::max(mode->width, 0));
        rect.height = static_cast<uint32_t>(std::max(mode->height, 0));
      }
      monitor_rects.push_back(rect);
    }
    if (const auto best = window_systems::select_monitor_by_overlap(window_rect, monitor_rects)) {
      monitor = monitors[*best];
    }
  }
  if (monitor == nullptr) {
    return make_unexpected(
        Error{ErrorCode::kNotFound, "No monitor overlaps this window (is a monitor connected?)"});
  }

  window_systems::NativeMonitorInfo info{};
  if (const char* name = glfwGetMonitorName(monitor); name != nullptr) {
    // Bounded copy of the NUL-terminated GLFW string; info.name is zero-initialized.
    std::snprintf(info.name.data(), info.name.size(), "%s", name);
  }
  int physical_w = 0;
  int physical_h = 0;
  glfwGetMonitorPhysicalSize(monitor, &physical_w, &physical_h);
  info.physical_size_mm = {static_cast<uint32_t>(std::max(physical_w, 0)),
                           static_cast<uint32_t>(std::max(physical_h, 0))};
  if (const GLFWvidmode* mode = glfwGetVideoMode(monitor); mode != nullptr) {
    info.current_resolution = {static_cast<uint32_t>(mode->width),
                               static_cast<uint32_t>(mode->height)};
    info.refresh_rate_mhz = static_cast<uint32_t>(std::max(mode->refreshRate, 0)) * 1000U;
  }
  return info;
}

expected<std::unique_ptr<detail::SurfaceImpl>> GlfwWindow::create_surface() const {
  VIZ_UNWRAP(vulkan::VulkanInstance, instance, vulkan::GetVulkanInstance());

  VkSurfaceKHR vk_surface = VK_NULL_HANDLE;
  VIZ_VK_FORWARD_ERROR(
      glfwCreateWindowSurface(instance.vk_instance_, window_, instance.vk_allocator_, &vk_surface));

  // A fullscreen window is not user-resizable, but its presentable area still changes with
  // the video mode (and the driver may invalidate the swapchain on focus changes), so it opts
  // into the swapchain's automatic recovery just like a resizable window. Its surface reports
  // a fixed currentExtent, which is what that recovery sizes the new swapchain from.
  auto surface = std::make_unique<vulkan::VulkanSurface>();
  VIZ_FORWARD_ERROR(surface->init(instance.vk_instance_,
                                  instance.vk_allocator_,
                                  vk_surface,
                                  extent_,
                                  /*resizable=*/resizable_ || fullscreen_));

  return surface;
}

expected<void> faux_window_create() {
  VIZ_FORWARD_ERROR(glfw_init_shared());

  glfw_window_count++;

  return {};
}

void faux_window_destroy() {
  if (glfw_window_count > 0) {
    glfw_window_count--;
  }

  if (glfw_window_count == 0) {
    glfwTerminate();
  }
}

void poll_events() {
  if (glfw_window_count > 0) {
    glfwPollEvents();
  }
}

}  // namespace holoviz::glfw

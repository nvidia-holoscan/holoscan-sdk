/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "detail/window_impl.hpp"

#include <mutex>

#include <GLFW/glfw3.h>

namespace holoviz::glfw {

class GlfwWindow final : public detail::WindowImpl {
 public:
  GlfwWindow() = default;
  ~GlfwWindow() override;

  expected<void> initialize(const WindowDesc& desc);

  [[nodiscard]] expected<void> destroy(std::chrono::nanoseconds timeout) override;

  WindowSystem system() const override { return WindowSystem::kGlfw; }
  Extent2D extent() const override;
  Extent2D window_size() const override;
  float content_scale() const override;
  bool minimized() const override;

  bool close_requested() const override;
  using CloseCallback = void (*)(void* user_data);
  void set_close_callback(void* user_data, CloseCallback callback) override;
  [[nodiscard]] expected<void> set_input_callbacks(const WindowInputCallbacks& callbacks) override;
  [[nodiscard]] expected<void> set_event_callbacks(const WindowEventCallbacks& callbacks) override;

  expected<NativeObject> native_object(NativeObjectType type) const override;
  expected<std::unique_ptr<detail::SurfaceImpl>> create_surface() const override;
  expected<window_systems::NativeMonitorInfo> current_monitor() const override;

 private:
  /// Install this module's GLFW callbacks on window_ (called once from initialize()).
  void install_callbacks();

  GLFWwindow* window_ = nullptr;
  bool glfw_refcounted_ = false;  ///< True while this window holds a glfw_window_count reference.
  Extent2D extent_{};
  bool resizable_ = false;
  bool fullscreen_ = false;
  bool close_on_escape_ = false;
  float content_scale_ = 1.0F;
  void* close_user_data_ = nullptr;
  Window::CloseCallback close_callback_ = nullptr;
  /// This window owns the GLFW user pointer (set to `this` at initialize()); the close
  /// callback and the raw input/event callbacks all route through it.
  WindowInputCallbacks input_callbacks_{};
  WindowEventCallbacks event_callbacks_{};
};

expected<void> faux_window_create();
void faux_window_destroy();

/// The lock serializing every GLFW call and callback-mutated window state (see its definition
/// in glfw_window.cpp). Acquired before one-time Vulkan instance initialization, never after.
std::recursive_mutex& window_system_mutex();

void poll_events();

}  // namespace holoviz::glfw

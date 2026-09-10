/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "rhi/detail/window_impl.hpp"

#include <GLFW/glfw3.h>

namespace holoviz::rhi::glfw {

class GlfwWindow final : public detail::WindowImpl {
 public:
  GlfwWindow() = default;
  ~GlfwWindow() override;

  expected<void> initialize(const WindowDesc& desc);

  [[nodiscard]] expected<void> destroy(std::chrono::nanoseconds timeout) override;

  WindowSystem system() const override { return WindowSystem::kGlfw; }
  Extent2D extent() const override;
  float content_scale() const override { return content_scale_; }

  bool close_requested() const override;
  using CloseCallback = void (*)(void* user_data);
  void set_close_callback(void* user_data, CloseCallback callback) override;
  [[nodiscard]] expected<void> set_input_callbacks(const WindowInputCallbacks& callbacks) override;

  expected<NativeObject> native_object(NativeObjectType type) const override;
  expected<std::unique_ptr<detail::SurfaceImpl>> create_surface() const override;

 private:
  GLFWwindow* window_ = nullptr;
  bool glfw_refcounted_ = false;  ///< True while this window holds a glfw_window_count reference.
  Extent2D extent_{};
  bool resizable_ = false;
  float content_scale_ = 1.0F;
  bool close_requested_ = false;
  void* close_user_data_ = nullptr;
  Window::CloseCallback close_callback_ = nullptr;
  /// This window owns the GLFW user pointer (set to `this` at initialize()); both the
  /// close callback and the raw input callbacks route through it.
  WindowInputCallbacks input_callbacks_{};
};

expected<void> faux_window_create();
void faux_window_destroy();

void poll_events();

}  // namespace holoviz::rhi::glfw

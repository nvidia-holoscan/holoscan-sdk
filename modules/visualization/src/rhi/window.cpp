/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "holoviz/rhi/window.hpp"

#include <memory>
#include <utility>

#include "detail/handle_errors.hpp"
#include "detail/surface_impl.hpp"
#include "detail/window_impl.hpp"
#include "holoviz/rhi/surface.hpp"
#include "window_systems/glfw_window.hpp"

namespace holoviz::rhi {

Window::Window(std::unique_ptr<detail::WindowImpl> impl) : impl_(std::move(impl)) {}
Window::Window() = default;
Window::Window(Window&&) noexcept = default;
Window& Window::operator=(Window&&) noexcept = default;
Window::~Window() = default;

expected<void> Window::destroy(std::chrono::nanoseconds timeout) {
  if (!impl_) {
    return {};  // empty handle: nothing to free
  }
  expected<void> result = impl_->destroy(timeout);
  if (result || result.error().code != ErrorCode::kBackpressured) {
    impl_.reset();
  }
  return result;
}

expected<WindowSystem> Window::system() const {
  if (!impl_) {
    return detail::null_impl_error("Window::system");
  }
  return impl_->system();
}

expected<Extent2D> Window::extent() const {
  if (!impl_) {
    return detail::null_impl_error("Window::extent");
  }
  return impl_->extent();
}

expected<float> Window::content_scale() const {
  if (!impl_) {
    return detail::null_impl_error("Window::content_scale");
  }
  return impl_->content_scale();
}

expected<bool> Window::close_requested() const {
  if (!impl_) {
    return detail::null_impl_error("Window::close_requested");
  }
  return impl_->close_requested();
}

expected<void> Window::set_close_callback(void* user_data, CloseCallback callback) {
  if (!impl_) {
    return detail::null_impl_error("Window::set_close_callback");
  }
  impl_->set_close_callback(user_data, callback);
  return {};
}

expected<void> Window::set_input_callbacks(const WindowInputCallbacks& callbacks) {
  if (!impl_) {
    return detail::null_impl_error("Window::set_input_callbacks");
  }
  return impl_->set_input_callbacks(callbacks);
}

expected<Surface> Window::create_surface() const {
  if (!impl_) {
    return detail::null_impl_error("Window::create_surface");
  }
  VIZ_UNWRAP(std::unique_ptr<detail::SurfaceImpl>, surface_impl, impl_->create_surface());
  return Surface{std::move(surface_impl)};
}

expected<NativeObject> Window::native_object(NativeObjectType type) const {
  if (!impl_) {
    return detail::null_impl_error("Window::native_object");
  }
  return impl_->native_object(type);
}

expected<Window> create_window(const WindowDesc& desc) {
  switch (desc.provider) {
    case WindowSystem::kAuto:
    case WindowSystem::kGlfw: {
      auto impl = std::make_unique<glfw::GlfwWindow>();
      VIZ_FORWARD_ERROR(impl->initialize(desc));
      return Window{std::move(impl)};
    }
    default:
      return make_unexpected(Error{ErrorCode::kInvalidArgument, "Unsupported window provider"});
  }
}

void poll_window_events() {
  glfw::poll_events();
}

}  // namespace holoviz::rhi

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// vk_impls.hpp must be the very first include so that VK_VERSION_1_0 is
// defined before glfw_window.hpp pulls in <GLFW/glfw3.h>. GLFW only
// declares glfwCreateWindowSurface when that macro is already visible.
#include "rhi/detail/destroy_timeout.hpp"
#include "rhi/vulkan/vk_impls.hpp"  // IWYU pragma: keep

#include "glfw_window.hpp"

#include <memory>

#include "rhi/vulkan/vk_adapter.hpp"
#include "rhi/vulkan/vk_errors.hpp"
#include "rhi/vulkan/vk_surface.hpp"

namespace holoviz::rhi::glfw {

static uint32_t glfw_window_count = 0;

expected<void> GlfwWindow::initialize(const WindowDesc& desc) {
  if (glfw_window_count == 0) {
    if (!glfwInit()) {
      return make_unexpected(Error{ErrorCode::kFailure, "Failed to initialize GLFW"});
    }
  }

  glfw_window_count++;
  glfw_refcounted_ = true;

  glfwWindowHint(GLFW_RESIZABLE, desc.resizable ? GLFW_TRUE : GLFW_FALSE);
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  window_ =
      glfwCreateWindow(desc.size.width, desc.size.height, desc.title.c_str(), nullptr, nullptr);

  if (!window_) {
    return make_unexpected(Error{ErrorCode::kFailure, "Failed to create GLFW window"});
  }

  extent_ = {desc.size.width, desc.size.height};
  resizable_ = desc.resizable;

  // This window owns the GLFW user pointer. The trampolines are installed once and
  // no-op while no consumer is registered; events are delivered during poll_events()
  // on the main thread.
  glfwSetWindowUserPointer(window_, this);
  glfwSetWindowCloseCallback(window_, [](GLFWwindow* w) {
    auto* self = static_cast<GlfwWindow*>(glfwGetWindowUserPointer(w));
    if (self != nullptr && self->close_callback_ != nullptr) {
      self->close_callback_(self->close_user_data_);
    }
  });
  glfwSetCursorPosCallback(window_, [](GLFWwindow* w, double x, double y) {
    auto* self = static_cast<GlfwWindow*>(glfwGetWindowUserPointer(w));
    if (self != nullptr && self->input_callbacks_.cursor_pos != nullptr) {
      self->input_callbacks_.cursor_pos(self->input_callbacks_.user_data, x, y);
    }
  });
  glfwSetMouseButtonCallback(window_, [](GLFWwindow* w, int button, int action, int mods) {
    auto* self = static_cast<GlfwWindow*>(glfwGetWindowUserPointer(w));
    if (self != nullptr && self->input_callbacks_.mouse_button != nullptr) {
      self->input_callbacks_.mouse_button(self->input_callbacks_.user_data, button, action, mods);
    }
  });
  glfwSetScrollCallback(window_, [](GLFWwindow* w, double dx, double dy) {
    auto* self = static_cast<GlfwWindow*>(glfwGetWindowUserPointer(w));
    if (self != nullptr && self->input_callbacks_.scroll != nullptr) {
      self->input_callbacks_.scroll(self->input_callbacks_.user_data, dx, dy);
    }
  });
  glfwSetKeyCallback(window_, [](GLFWwindow* w, int key, int scancode, int action, int mods) {
    auto* self = static_cast<GlfwWindow*>(glfwGetWindowUserPointer(w));
    if (self != nullptr && self->input_callbacks_.key != nullptr) {
      self->input_callbacks_.key(self->input_callbacks_.user_data, key, scancode, action, mods);
    }
  });
  glfwSetCharCallback(window_, [](GLFWwindow* w, unsigned int codepoint) {
    auto* self = static_cast<GlfwWindow*>(glfwGetWindowUserPointer(w));
    if (self != nullptr && self->input_callbacks_.char_input != nullptr) {
      self->input_callbacks_.char_input(self->input_callbacks_.user_data, codepoint);
    }
  });
  glfwSetWindowFocusCallback(window_, [](GLFWwindow* w, int focused) {
    auto* self = static_cast<GlfwWindow*>(glfwGetWindowUserPointer(w));
    if (self != nullptr && self->input_callbacks_.focus != nullptr) {
      self->input_callbacks_.focus(self->input_callbacks_.user_data, focused);
    }
  });

  return {};
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
    glfwDestroyWindow(window_);
    window_ = nullptr;
  }
  input_callbacks_ = {};

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

expected<NativeObject> GlfwWindow::native_object(NativeObjectType type) const {
  if (type == NativeObjectType::kGLFWwindow) {
    return NativeObject{type, reinterpret_cast<uint64_t>(window_)};
  }
  return make_unexpected(
      Error{ErrorCode::kInvalidArgument, "GLFW window backend only supports kGLFWwindow"});
}

expected<std::unique_ptr<detail::SurfaceImpl>> GlfwWindow::create_surface() const {
  VIZ_UNWRAP(vulkan::VulkanInstance, instance, vulkan::GetVulkanInstance());

  VkSurfaceKHR vk_surface = VK_NULL_HANDLE;
  VIZ_VK_FORWARD_ERROR(
      glfwCreateWindowSurface(instance.vk_instance_, window_, instance.vk_allocator_, &vk_surface));

  auto surface = std::make_unique<vulkan::VulkanSurface>();
  VIZ_FORWARD_ERROR(surface->init(
      instance.vk_instance_, instance.vk_allocator_, vk_surface, extent_, resizable_));

  return surface;
}

expected<void> faux_window_create() {
  if (glfw_window_count == 0) {
    if (!glfwInit()) {
      return make_unexpected(Error{ErrorCode::kFailure, "Failed to initialize GLFW"});
    }
  }

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

}  // namespace holoviz::rhi::glfw

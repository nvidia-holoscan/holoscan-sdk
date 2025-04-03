/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "context.hpp"

#include <imgui.h>

#include <list>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/logger/logger.hpp>
#include <nvh/nvprint.hpp>

#include "exclusive_window.hpp"
#include "glfw_window.hpp"
#include "headless_window.hpp"
#include "layers/geometry_layer.hpp"
#include "layers/im_gui_layer.hpp"
#include "layers/image_layer.hpp"
#include "vulkan/vulkan_app.hpp"

namespace {

void nvprint_callback(int level, const char* fmt) {
  // nvpro_core requires a new-line, our logging is automatically adding it.
  // Remove the trailing newline if there is one.
  std::string str(fmt);
  if (str.back() == '\n') { str.pop_back(); }

  switch (level) {
    case LOGLEVEL_INFO:
    case LOGLEVEL_OK:
      HOLOSCAN_LOG_INFO(str.c_str());
      break;
    case LOGLEVEL_WARNING:
      HOLOSCAN_LOG_WARN(str.c_str());
      break;
    case LOGLEVEL_ERROR:
      HOLOSCAN_LOG_ERROR(str.c_str());
      break;
    case LOGLEVEL_DEBUG:
      HOLOSCAN_LOG_DEBUG(str.c_str());
      break;
    default:
    case LOGLEVEL_STATS:
      HOLOSCAN_LOG_TRACE(str.c_str());
      break;
  }

  // nvpro_core prints the log message even if a callback is set. Set the string to '\0' to avoid
  // the extra print.
  char* buffer = const_cast<char*>(fmt);
  *buffer = '\0';
}

}  // namespace

namespace holoscan::viz {

/// thread local context
static thread_local Context* g_context = nullptr;

/// thread local ImGui context
thread_local ImGuiContext* g_im_gui_context = nullptr;

class Context::Impl {
 public:
  void setup() {
    im_gui_context_ = ImGui::CreateContext();
    vulkan_.reset(new Vulkan);
    if (surface_format_.has_value()) { vulkan_->set_surface_format(surface_format_.value()); }
    vulkan_->set_present_mode(present_mode_);
    vulkan_->setup(window_.get(), font_path_, font_size_in_pixels_);

    // setup the window callbacks, they call the user callbacks if present
    key_callback_handle_ = window_->add_key_callback(
        [this](Key key, KeyAndButtonAction action, KeyModifiers modifiers) {
          if (key_callback_) { key_callback_(key_user_pointer_, key, action, modifiers); }
        });
    unicode_char_callback_handle_ = window_->add_unicode_char_callback([this](uint32_t code_point) {
      if (unicode_char_callback_) {
        unicode_char_callback_(unicode_char_user_pointer_, code_point);
      }
    });
    mouse_button_callback_handle_ = window_->add_mouse_button_callback(
        [this](MouseButton button, KeyAndButtonAction action, KeyModifiers modifiers) {
          if (mouse_button_callback_) {
            mouse_button_callback_(mouse_button_user_pointer_, button, action, modifiers);
          }
        });
    scroll_callback_handle_ =
        window_->add_scroll_callback([this](double x_offset, double y_offset) {
          if (scroll_callback_) { scroll_callback_(scroll_user_pointer_, x_offset, y_offset); }
        });
    cursor_pos_callback_handle_ =
        window_->add_cursor_pos_callback([this](double x_pos, double y_pos) {
          if (cursor_pos_callback_) {
            cursor_pos_callback_(cursor_pos_user_pointer_, x_pos, y_pos);
          }
        });
    framebuffer_size_callback_handle_ =
        window_->add_framebuffer_size_callback([this](int width, int height) {
          if (framebuffer_size_callback_) {
            framebuffer_size_callback_(framebuffer_size_user_pointer_, width, height);
          }
        });
    window_size_callback_handle_ = window_->add_window_size_callback([this](int width, int height) {
      if (window_size_callback_) {
        window_size_callback_(window_size_user_pointer_, width, height);
      }
    });

    if (framebuffer_size_callback_ || window_size_callback_) {
      // call with initial framebuffer and window size
      uint32_t width, height;
      if (framebuffer_size_callback_) {
        window_->get_framebuffer_size(&width, &height);
        framebuffer_size_callback_(framebuffer_size_user_pointer_, width, height);
      }
      if (window_size_callback_) {
        window_->get_window_size(&width, &height);
        window_size_callback_(window_size_user_pointer_, width, height);
      }
    }
  }

  ImGuiContext* im_gui_context_ = nullptr;

  InitFlags flags_ = InitFlags::NONE;
  std::optional<SurfaceFormat> surface_format_;
  CUstream cuda_stream_ = 0;
  std::string font_path_;
  float font_size_in_pixels_ = 0.F;
  PresentMode present_mode_ = PresentMode::AUTO;

  std::unique_ptr<Window> window_;

  void* key_user_pointer_ = nullptr;
  KeyCallbackFunction key_callback_ = nullptr;
  Window::CallbackHandle key_callback_handle_;

  void* unicode_char_user_pointer_ = nullptr;
  UnicodeCharCallbackFunction unicode_char_callback_ = nullptr;
  Window::CallbackHandle unicode_char_callback_handle_;

  void* mouse_button_user_pointer_ = nullptr;
  MouseButtonCallbackFunction mouse_button_callback_ = nullptr;
  Window::CallbackHandle mouse_button_callback_handle_;

  void* scroll_user_pointer_ = nullptr;
  ScrollCallbackFunction scroll_callback_ = nullptr;
  Window::CallbackHandle scroll_callback_handle_;

  void* cursor_pos_user_pointer_ = nullptr;
  CursorPosCallbackFunction cursor_pos_callback_ = nullptr;
  Window::CallbackHandle cursor_pos_callback_handle_;

  void* framebuffer_size_user_pointer_ = nullptr;
  FramebufferSizeCallbackFunction framebuffer_size_callback_ = nullptr;
  Window::CallbackHandle framebuffer_size_callback_handle_;

  void* window_size_user_pointer_ = nullptr;
  WindowSizeCallbackFunction window_size_callback_ = nullptr;
  Window::CallbackHandle window_size_callback_handle_;

  std::unique_ptr<Vulkan> vulkan_;

  /**
   * We need to call ImGui::NewFrame() once for the first ImGUILayer, this is set to 'false' at
   *  begin in to 'true' when the first ImGUI layer had been created.
   */
  bool imgui_new_frame_ = false;

  std::unique_ptr<Layer> active_layer_;  ///< currently active layer

  std::list<std::unique_ptr<Layer>> layers_;  ///< the list of the layers of the current frame

  std::list<std::unique_ptr<Layer>> layer_cache_;  ///< layers of the previous frame,
                                                   ///    most likely they can be reused
};

Context::Context() : impl_(new Impl) {
  // disable nvpro_core file logging
  nvprintSetFileLogging(false);
  // and use the callback to use our logging functions
  nvprintSetCallback(nvprint_callback);
}

Context::~Context() {
  ImGuiContext* prev_context = nullptr;
  if ((g_context != this) && (impl_->im_gui_context_)) {
    // make ImGui context current while shutting down
    prev_context = ImGui::GetCurrentContext();
    ImGui::SetCurrentContext(impl_->im_gui_context_);
  }

  // explicitly destroy objects which can have references to vulkan or use ImGui while the ImGui
  // context is current
  impl_->active_layer_.reset();
  impl_->layers_.clear();
  impl_->layer_cache_.clear();
  impl_->vulkan_.reset();
  impl_->window_size_callback_handle_.reset();
  impl_->framebuffer_size_callback_handle_.reset();
  impl_->cursor_pos_callback_handle_.reset();
  impl_->scroll_callback_handle_.reset();
  impl_->mouse_button_callback_handle_.reset();
  impl_->unicode_char_callback_handle_.reset();
  impl_->key_callback_handle_.reset();
  impl_->window_.reset();

  if (impl_->im_gui_context_) { ImGui::DestroyContext(impl_->im_gui_context_); }

  if (prev_context) {
    // make the previous ImGui context current
    ImGui::SetCurrentContext(prev_context);
  }

  // if the context is current make it not current
  if (g_context == this) { set(nullptr); }
}

void Context::set(Context* context) {
  g_context = context;
  if (context) {
    // also make the ImGui context current
    ImGui::SetCurrentContext(context->impl_->im_gui_context_);
  } else {
    ImGui::SetCurrentContext(nullptr);
  }
}

Context* Context::get_current() {
  return g_context;
}

Context& Context::get() {
  if (!g_context) {
    // if no context is current, create one and make it current
    Context* context = new Context();
    set(context);
  }
  return *g_context;
}

void Context::init(GLFWwindow* window, InitFlags flags) {
  impl_->window_.reset(new GLFWWindow(window));
  impl_->flags_ = flags;
  impl_->setup();
}

void Context::init(uint32_t width, uint32_t height, const char* title, InitFlags flags,
                   const char* display_name) {
  if (flags & InitFlags::HEADLESS) {
    impl_->window_.reset(new HeadlessWindow(width, height, flags));
  } else {
    impl_->window_.reset(new GLFWWindow(width, height, title, flags, display_name));
  }
  impl_->flags_ = flags;
  impl_->setup();
}

void Context::init(const char* display_name, uint32_t width, uint32_t height, uint32_t refresh_rate,
                   InitFlags flags) {
  impl_->window_.reset(new ExclusiveWindow(display_name, width, height, refresh_rate, flags));
  impl_->flags_ = flags;
  impl_->setup();
}

void Context::set_key_callback(void* user_pointer, KeyCallbackFunction callback) {
  impl_->key_user_pointer_ = user_pointer;
  impl_->key_callback_ = callback;
}

void Context::set_unicode_char_callback(void* user_pointer, UnicodeCharCallbackFunction callback) {
  impl_->unicode_char_user_pointer_ = user_pointer;
  impl_->unicode_char_callback_ = callback;
}

void Context::set_mouse_button_callback(void* user_pointer, MouseButtonCallbackFunction callback) {
  impl_->mouse_button_user_pointer_ = user_pointer;
  impl_->mouse_button_callback_ = callback;
}

void Context::set_scroll_callback(void* user_pointer, ScrollCallbackFunction callback) {
  impl_->scroll_user_pointer_ = user_pointer;
  impl_->scroll_callback_ = callback;
}

void Context::set_cursor_pos_callback(void* user_pointer, CursorPosCallbackFunction callback) {
  impl_->cursor_pos_user_pointer_ = user_pointer;
  impl_->cursor_pos_callback_ = callback;
}

void Context::set_framebuffer_size_callback(void* user_pointer,
                                            FramebufferSizeCallbackFunction callback) {
  impl_->framebuffer_size_user_pointer_ = user_pointer;
  impl_->framebuffer_size_callback_ = callback;
  if (impl_->window_ && callback) {
    // call with initial framebuffer size
    uint32_t width, height;
    impl_->window_->get_framebuffer_size(&width, &height);
    callback(user_pointer, width, height);
  }
}

void Context::set_window_size_callback(void* user_pointer, WindowSizeCallbackFunction callback) {
  impl_->window_size_user_pointer_ = user_pointer;
  impl_->window_size_callback_ = callback;
  if (impl_->window_ && callback) {
    // call with initial window size
    uint32_t width, height;
    impl_->window_->get_window_size(&width, &height);
    callback(user_pointer, width, height);
  }
}

std::vector<SurfaceFormat> Context::get_surface_formats() const {
  if (!impl_->vulkan_) {
    throw std::runtime_error("There is no window, please call viz::Init() first.");
  }
  return impl_->vulkan_->get_surface_formats();
}

void Context::set_surface_format(SurfaceFormat surface_format) {
  impl_->surface_format_ = surface_format;
  // update Vulkan surface format if Vulkan has already been initialized
  if (impl_->vulkan_) { impl_->vulkan_->set_surface_format(impl_->surface_format_.value()); }
}

Window* Context::get_window() const {
  if (!impl_->window_) { throw std::runtime_error("There is no window set."); }

  return impl_->window_.get();
}

std::vector<PresentMode> Context::get_present_modes() const {
  if (!impl_->vulkan_) {
    throw std::runtime_error("There is no window, please call viz::Init() first.");
  }
  return impl_->vulkan_->get_present_modes();
}

void Context::set_present_mode(PresentMode present_mode) {
  impl_->present_mode_ = present_mode;
  // update Vulkan present mode if Vulkan has already been initialized
  if (impl_->vulkan_) { impl_->vulkan_->set_present_mode(impl_->present_mode_); }
}

std::vector<ImageFormat> Context::get_image_formats() const {
  if (!impl_->vulkan_) {
    throw std::runtime_error("There is no window, please call viz::Init() first.");
  }
  return impl_->vulkan_->get_image_formats();
}

void Context::set_cuda_stream(CUstream stream) {
  impl_->cuda_stream_ = stream;
}

CUstream Context::get_cuda_stream() const {
  return impl_->cuda_stream_;
}

void Context::set_font(const char* path, float size_in_pixels) {
  if (impl_->vulkan_) {
    throw std::runtime_error("The font has to be set before Init() is called");
  }
  impl_->font_path_ = path;
  impl_->font_size_in_pixels_ = size_in_pixels;
}

void Context::begin() {
  if (!impl_->window_) {
    throw std::runtime_error("There is no window, please call viz::Init() first.");
  }

  impl_->imgui_new_frame_ = false;
  impl_->window_->begin();

  // start the transfer pass, layers transfer their data on EndLayer(), layers are drawn on End()
  impl_->vulkan_->begin_transfer_pass();
}

void Context::end() {
  if (!impl_->window_) {
    throw std::runtime_error("There is no window, please call viz::Init() first.");
  }

  impl_->window_->end();

  // end the transfer pass
  impl_->vulkan_->end_transfer_pass();

  // draw the layers
  impl_->vulkan_->begin_render_pass();

  // sort layers (inverse because highest priority is drawn last)
  std::list<Layer*> sorted_layers;
  for (auto&& item : impl_->layers_) { sorted_layers.emplace_back(item.get()); }
  sorted_layers.sort([](Layer* a, Layer* b) { return a->get_priority() < b->get_priority(); });

  // render
  for (auto&& layer : sorted_layers) { layer->render(impl_->vulkan_.get()); }

  // rendering is done
  impl_->vulkan_->end_render_pass();

  // if the call sequence changed, then unused items remained in the cache, delete them
  impl_->layer_cache_.clear();

  // move the items to the layer cache for reuse in the next frame
  impl_->layer_cache_.splice(impl_->layer_cache_.begin(), impl_->layers_);
}

void Context::begin_image_layer() {
  if (impl_->active_layer_) { throw std::runtime_error("There already is an active layer."); }

  impl_->active_layer_.reset(new ImageLayer());
}

void Context::begin_geometry_layer() {
  if (impl_->active_layer_) { throw std::runtime_error("There already is an active layer."); }

  impl_->active_layer_.reset(new GeometryLayer());
}

void Context::begin_im_gui_layer() {
  if (!ImGui::GetCurrentContext()) {
    throw std::runtime_error("ImGui had not been setup, please call holoscan::viz::Init().");
  }

  if (impl_->active_layer_) { throw std::runtime_error("There already is an active layer."); }

  if (!impl_->imgui_new_frame_) {
    // Start the Dear ImGui frame
    impl_->window_->im_gui_new_frame();
    impl_->imgui_new_frame_ = true;
  } else {
    throw std::runtime_error("Multiple ImGui layers are not supported");
  }

  impl_->active_layer_.reset(new ImGuiLayer());
}

void Context::end_layer() {
  if (!impl_->active_layer_) { throw std::runtime_error("There is no active layer."); }

  // scan the layer cache to check if the active layer already had been seen before
  for (auto it = impl_->layer_cache_.begin(); it != impl_->layer_cache_.end(); ++it) {
    if (impl_->active_layer_->can_be_reused(*it->get())) {
      // set the 'soft' parameters
      /// @todo this is error prone and needs to be resolved in a better way,
      ///       maybe have different categories (sections) of parameters and then copy
      ///       the 'soft' parameters to the re-used layer
      (*it)->set_opacity(impl_->active_layer_->get_opacity());
      (*it)->set_priority(impl_->active_layer_->get_priority());
      (*it)->set_views(impl_->active_layer_->get_views());

      // replace the current active layer with the cached item
      impl_->active_layer_ = std::move(*it);

      // remove from the cache
      impl_->layer_cache_.erase(it);
      break;
    }
  }

  impl_->active_layer_->end(impl_->vulkan_.get());
  impl_->layers_.push_back(std::move(impl_->active_layer_));
}

void Context::read_framebuffer(ImageFormat fmt, uint32_t width, uint32_t height, size_t buffer_size,
                               CUdeviceptr device_ptr, size_t row_pitch) {
  if (!impl_->vulkan_) {
    throw std::runtime_error("There is no window, please call viz::Init() first.");
  }

  impl_->vulkan_->read_framebuffer(
      fmt, width, height, buffer_size, device_ptr, impl_->cuda_stream_, row_pitch);
}

Layer* Context::get_active_layer() const {
  if (!impl_->active_layer_) { throw std::runtime_error("There is no active layer."); }
  return impl_->active_layer_.get();
}

ImageLayer* Context::get_active_image_layer() const {
  if (!impl_->active_layer_) { throw std::runtime_error("There is no active layer."); }

  if (impl_->active_layer_->get_type() != Layer::Type::Image) {
    throw std::runtime_error("The active layer is not an image layer.");
  }

  return static_cast<ImageLayer*>(impl_->active_layer_.get());
}

GeometryLayer* Context::get_active_geometry_layer() const {
  if (!impl_->active_layer_) { throw std::runtime_error("There is no active layer."); }

  if (impl_->active_layer_->get_type() != Layer::Type::Geometry) {
    throw std::runtime_error("The active layer is not a geometry layer.");
  }

  return static_cast<GeometryLayer*>(impl_->active_layer_.get());
}

}  // namespace holoscan::viz

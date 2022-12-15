/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <memory>
#include <stdexcept>

#include <nvh/nvprint.hpp>
#include "exclusive_window.hpp"
#include "glfw_window.hpp"
#include "headless_window.hpp"
#include "layers/geometry_layer.hpp"
#include "layers/im_gui_layer.hpp"
#include "layers/image_layer.hpp"
#include "vulkan/vulkan.hpp"

namespace holoscan::viz {

struct Context::Impl {
    InitFlags flags_      = InitFlags::NONE;
    CUstream cuda_stream_ = 0;

    std::unique_ptr<Window> window_;

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

Context &Context::get() {
    // since C++11 static variables a thread-safe
    static Context instance;

    return instance;
}

Context::Context()
    : impl_(new Impl) {
    // disable nvpro_core file logging
    nvprintSetFileLogging(false);
}

void Context::init(GLFWwindow *window, InitFlags flags) {
    impl_->window_.reset(new GLFWWindow(window));
    impl_->vulkan_.reset(new Vulkan);
    impl_->vulkan_->setup(impl_->window_.get());
    impl_->flags_ = flags;
}

void Context::init(uint32_t width, uint32_t height, const char *title, InitFlags flags) {
    if (flags & InitFlags::HEADLESS) {
        impl_->window_.reset(new HeadlessWindow(width, height, flags));
    } else {
        impl_->window_.reset(new GLFWWindow(width, height, title, flags));
    }
    impl_->vulkan_.reset(new Vulkan);
    impl_->vulkan_->setup(impl_->window_.get());
    impl_->flags_ = flags;
}

void Context::init(const char *display_name, uint32_t width, uint32_t height,
                                     uint32_t refresh_rate, InitFlags flags) {
    impl_->window_.reset(new ExclusiveWindow(display_name, width, height, refresh_rate, flags));
    impl_->vulkan_.reset(new Vulkan);
    impl_->vulkan_->setup(impl_->window_.get());
    impl_->flags_ = flags;
}

void Context::shutdown() {
    impl_->active_layer_.reset();
    impl_->layers_.clear();
    impl_->layer_cache_.clear();
    impl_->vulkan_.reset();
    impl_->window_.reset();
}

Window *Context::get_window() const {
    if (!impl_->window_) {
        throw std::runtime_error("There is no window set.");
    }

    return impl_->window_.get();
}

void Context::im_gui_set_current_context(ImGuiContext *context) {
    ImGui::SetCurrentContext(context);
}

void Context::set_cuda_stream(CUstream stream) {
    impl_->cuda_stream_ = stream;
}

CUstream Context::get_cuda_stream() const {
    return impl_->cuda_stream_;
}

void Context::begin() {
    impl_->imgui_new_frame_ = false;
    impl_->window_->begin();

    // start the transfer pass, layers transfer their data on EndLayer(), layers are drawn on End()
    impl_->vulkan_->begin_transfer_pass();
}

void Context::end() {
    impl_->window_->end();

    // end the transfer pass
    impl_->vulkan_->end_transfer_pass();

    // draw the layers
    impl_->vulkan_->begin_render_pass();

    // sort layers (inverse because highest priority is drawn last)
    std::list<Layer *> sorted_layers;
    for (auto &&item : impl_->layers_) {
        sorted_layers.emplace_back(item.get());
    }
    sorted_layers.sort([](Layer *a, Layer *b) { return a->get_priority() < b->get_priority(); });

    // render
    for (auto &&layer : sorted_layers) {
        layer->render(impl_->vulkan_.get());
    }

    // rendering is done
    impl_->vulkan_->end_render_pass();

    // if the call sequence changed, then unused items remained in the cache, delete them
    impl_->layer_cache_.clear();

    // move the items to the layer cache for reuse in the next frame
    impl_->layer_cache_.splice(impl_->layer_cache_.begin(), impl_->layers_);
}

void Context::begin_image_layer() {
    if (impl_->active_layer_) {
        throw std::runtime_error("There already is an active layer.");
    }

    impl_->active_layer_.reset(new ImageLayer());
}

void Context::begin_geometry_layer() {
    if (impl_->active_layer_) {
        throw std::runtime_error("There already is an active layer.");
    }

    impl_->active_layer_.reset(new GeometryLayer());
}

void Context::begin_im_gui_layer() {
    if (!ImGui::GetCurrentContext()) {
        throw std::runtime_error(
         "ImGui had not been setup, please call ImGuiSetCurrentContext() before calling Init().");
    }

    if (impl_->active_layer_) {
        throw std::runtime_error("There already is an active layer.");
    }

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
    if (!impl_->active_layer_) {
        throw std::runtime_error("There is no active layer.");
    }

    // scan the layer cache to check if the active layer already had been seen before
    for (auto it = impl_->layer_cache_.begin(); it != impl_->layer_cache_.end(); ++it) {
        if (impl_->active_layer_->can_be_reused(*it->get())) {
            // set the 'soft' parameters
            /// @todo this is error prone and needs to be resolved in a better way,
            ///       maybe have different categories (sections) of parameters and then copy
            ///       the 'soft' parameters to the re-used layer
            (*it)->set_opacity(impl_->active_layer_->get_opacity());
            (*it)->set_priority(impl_->active_layer_->get_priority());

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

void Context::read_framebuffer(ImageFormat fmt, size_t buffer_size, CUdeviceptr device_ptr) {
    impl_->vulkan_->read_framebuffer(fmt, buffer_size, device_ptr, impl_->cuda_stream_);
}

Layer *Context::get_active_layer() const {
    if (!impl_->active_layer_) {
        throw std::runtime_error("There is no active layer.");
    }
    return impl_->active_layer_.get();
}

ImageLayer *Context::get_active_image_layer() const {
    if (!impl_->active_layer_) {
        throw std::runtime_error("There is no active layer.");
    }

    if (impl_->active_layer_->get_type() != Layer::Type::Image) {
        throw std::runtime_error("The active layer is not an image layer.");
    }

    return static_cast<ImageLayer *>(impl_->active_layer_.get());
}

GeometryLayer *Context::get_active_geometry_layer() const {
    if (!impl_->active_layer_) {
        throw std::runtime_error("There is no active layer.");
    }

    if (impl_->active_layer_->get_type() != Layer::Type::Geometry) {
        throw std::runtime_error("The active layer is not a geometry layer.");
    }

    return static_cast<GeometryLayer *>(impl_->active_layer_.get());
}

}  // namespace holoscan::viz

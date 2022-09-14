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

#include "Context.h"

#include "layers/ImageLayer.h"
#include "layers/GeometryLayer.h"
#include "layers/ImGuiLayer.h"
#include "vulkan/Vulkan.h"
#include "GLFWWindow.h"
#include "ExclusiveWindow.h"

#include <nvh/nvprint.hpp>

#include <imgui.h>

#include <memory>

namespace clara::holoviz
{

static std::unique_ptr<Context> g_context;

struct Context::Impl
{
    InitFlags flags_ = InitFlags::NONE;

    std::unique_ptr<Window> window_;

    Vulkan vulkan_;

    /**
     * We need to call ImGui::NewFrame() once for the first ImGUILayer, this is set to 'false' at begin in to 'true' when the
     * first ImGUI layer had been created.
     */
    bool imgui_new_frame_ = false;

    std::unique_ptr<Layer> active_layer_; ///< currently active layer

    std::list<std::unique_ptr<Layer>> layers_; ///< the list of the layers of the current frame

    std::list<std::unique_ptr<Layer>> layer_cache_; ///< layers of the previous frame, most likely they can be reused
};

Context::Context()
    : impl_(new Impl)
{
    // disable nvpro_core file logging
    nvprintSetFileLogging(false);
}

Context::~Context() {}

void Context::Init(GLFWwindow *window, InitFlags flags)
{
    impl_->window_.reset(new GLFWWindow(window));
    impl_->vulkan_.Setup(impl_->window_.get());
    impl_->flags_ = flags;
}

void Context::Init(uint32_t width, uint32_t height, const char *title, InitFlags flags)
{
    impl_->window_.reset(new GLFWWindow(width, height, title, flags));
    impl_->vulkan_.Setup(impl_->window_.get());
    impl_->flags_ = flags;
}

void Context::Init(const char *display_name, uint32_t width, uint32_t height, uint32_t refresh_rate, InitFlags flags)
{
    impl_->window_.reset(new ExclusiveWindow(display_name, width, height, refresh_rate, flags));
    impl_->vulkan_.Setup(impl_->window_.get());
    impl_->flags_ = flags;
}

Context &Context::Get()
{
    if (!g_context)
    {
        g_context.reset(new Context());
    }
    return *g_context;
}

void Context::Shutdown()
{
    g_context.release();
}

Window *Context::GetWindow() const
{
    return impl_->window_.get();
}

void Context::ImGuiSetCurrentContext(ImGuiContext *context)
{
    ImGui::SetCurrentContext(context);
}

void Context::Begin()
{
    impl_->imgui_new_frame_ = false;
    impl_->window_->Begin();
}

void Context::End()
{
    impl_->window_->End();

    // this will daraw the layers
    impl_->vulkan_.SubmitFrame(impl_->layers_);

    // if the call sequence changed, then unused items remained in the cache, delete them
    impl_->layer_cache_.clear();

    // move the items to the layer cache for reuse in the next frame
    impl_->layer_cache_.splice(impl_->layer_cache_.begin(), impl_->layers_);
}

void Context::BeginImageLayer()
{
    if (impl_->active_layer_)
    {
        throw std::runtime_error("There already is an active layer.");
    }

    impl_->active_layer_.reset(new ImageLayer());
}

void Context::BeginGeometryLayer()
{
    if (impl_->active_layer_)
    {
        throw std::runtime_error("There already is an active layer.");
    }

    impl_->active_layer_.reset(new GeometryLayer());
}

void Context::BeginImGuiLayer()
{
    if (!ImGui::GetCurrentContext())
    {
        throw std::runtime_error(
            "ImGui had not been setup, please call ImGuiSetCurrentContext() before calling Init().");
    }

    if (impl_->active_layer_)
    {
        throw std::runtime_error("There already is an active layer.");
    }

    if (!impl_->imgui_new_frame_)
    {
        // Start the Dear ImGui frame
        impl_->window_->ImGuiNewFrame();
        impl_->imgui_new_frame_ = true;
    }
    else
    {
        throw std::runtime_error("Multiple ImGui layers are not supported");
    }

    impl_->active_layer_.reset(new ImGuiLayer());
}

void Context::EndLayer()
{
    if (!impl_->active_layer_)
    {
        throw std::runtime_error("There is no active layer.");
    }

    // scan the layer cache to check if the active layer already had been seen before
    for (auto it = impl_->layer_cache_.begin(); it != impl_->layer_cache_.end(); ++it)
    {
        if (impl_->active_layer_->CanBeReused(*it->get()))
        {
            // set the 'soft' parameters
            /// @todo this is error prone and needs to be resolved in a better way,
            ///       maybe have different categories (sections) of parameters and then copy
            ///       the 'soft' parameters to the re-used layer
            (*it)->SetOpacity(impl_->active_layer_->GetOpacity());
            (*it)->SetPriority(impl_->active_layer_->GetPriority());

            // replace the current active layer with the cached item
            impl_->active_layer_ = std::move(*it);

            // remove from the cache
            impl_->layer_cache_.erase(it);
            break;
        }
    }

    // if not found in the cache
    if (impl_->active_layer_)
    {
        impl_->layers_.push_back(std::move(impl_->active_layer_));
    }
}

Layer *Context::GetActiveLayer() const
{
    if (!impl_->active_layer_)
    {
        throw std::runtime_error("There is no active layer.");
    }
    return impl_->active_layer_.get();
}

ImageLayer *Context::GetActiveImageLayer() const
{
    if (!impl_->active_layer_)
    {
        throw std::runtime_error("There is no active layer.");
    }

    if (impl_->active_layer_->GetType() != Layer::Type::Image)
    {
        throw std::runtime_error("The active layer is not an image layer.");
    }

    return static_cast<ImageLayer *>(impl_->active_layer_.get());
}

GeometryLayer *Context::GetActiveGeometryLayer() const
{
    if (!impl_->active_layer_)
    {
        throw std::runtime_error("There is no active layer.");
    }

    if (impl_->active_layer_->GetType() != Layer::Type::Geometry)
    {
        throw std::runtime_error("The active layer is not a geometry layer.");
    }

    return static_cast<GeometryLayer *>(impl_->active_layer_.get());
}

} // namespace clara::holoviz

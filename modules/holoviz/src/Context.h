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
#pragma once

#include "holoviz/InitFlags.h"

#include <memory>
#include <list>

struct ImGuiContext;
typedef struct GLFWwindow GLFWwindow;

namespace clara::holoviz
{

class Window;
class ImageLayer;
class GeometryLayer;
class Layer;

/**
 * The Context object is the central object for Holoviz to store all the state. It's a global unique object.
 */
class Context
{
public:
    /**
     * Destroy the context object.
     */
    ~Context();

    /**
     * Initialize the context with an existing GLFWwindow object
     *
     * @param window    existing GLFWwindow window
     * @param flags     init flags
     */
    void Init(GLFWwindow *window, InitFlags flags);

    /**
     * Initialize the context and create a window with the given properties.
     *
     * @param width, height     window size
     * @param title             window title
     * @param flags             init flags
     */
    void Init(uint32_t width, uint32_t height, const char *title, InitFlags flags);

    /**
     * Initialize the context and create a exclusive window with the given properties.
     *
     * @param display_name  name of the display, this can either be the EDID name as displayed in the NVIDIA Settings, or
     *                      the output name used by xrandr, if nullptr then the first display is selected.
     * @param width         desired width, ignored if 0
     * @param height        desired height, ignored if 0
     * @param refresh_rate  desired refresh rate (number of times the display is refreshed each second multiplied by 1000), ignored if 0
     * @param flags         init flags
     */
    void Init(const char *display_name, uint32_t width, uint32_t height, uint32_t refresh_rate, InitFlags flags);

    /**
     * Get the global context, if there is no global context yet it will be created.
     *
     * @return global context
     */
    static Context &Get();

    /**
     * Shutdown and cleanup the global context.
     */
    static void Shutdown();

    /**
     * Get the window object.
     *
     * @return window object
     */
    Window *GetWindow() const;

    /**
     * If using ImGui, create a context and pass it to Holoviz, do this before calling viz::Init().
     *
     * Background: The ImGui context is a global variable and global variables are not shared across so/DLL
     * boundaries. Therefore the app needs to create the ImGui context first and then provides the pointer
     * to Holoviz like this:
     * @code{.cpp}
     * ImGui::CreateContext();
     * clara::holoviz::ImGuiSetCurrentContext(ImGui::GetCurrentContext());
     * @endcode
     *
     * @param context   ImGui context
     */
    void ImGuiSetCurrentContext(ImGuiContext *context);

    /**
     * Start recording layer definitions.
     */
    void Begin();

    /**
     * End recording and output the composited layers.
     */
    void End();

    /**
     * Begin an image layer definition.
     */
    void BeginImageLayer();

    /**
     * Begin a geometry layer definition.
     */
    void BeginGeometryLayer();

    /**
     * Begin an ImGui layer definition.
     */
    void BeginImGuiLayer();

    /**
     * End the current layer.
     */
    void EndLayer();

    /**
     * @returns the active layer
     */
    Layer *GetActiveLayer() const;

    /**
     * @returns the active image layer
     */
    ImageLayer *GetActiveImageLayer() const;

    /**
     * @returns the active geometry layer
     */
    GeometryLayer *GetActiveGeometryLayer() const;

private:
    Context();

    struct Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace clara::holoviz

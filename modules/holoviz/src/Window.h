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

#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <functional>

namespace clara::holoviz
{

/**
 * Base class for all window backends.
 */
class Window
{
public:
    /**
     * Construct a new Window object.
     */
    Window(){};

    /**
     * Destroy the Window object.
     */
    virtual ~Window(){};

    /**
     * Initialize ImGui to be used with that window (mouse, keyboard, ...).
     */
    virtual void InitImGui() = 0;

    /**
     * Setup call backs.
     *
     * @param framebuffersize_cb    Called when the frame buffer size changes
     */
    virtual void SetupCallbacks(std::function<void(int width, int height)> frame_buffer_size_cb) = 0;

    /**
     * Get the required instance extensions for vulkan.
     *
     * @param [out] count   returns the extension count
     * @return array with required extensions
     */
    virtual const char **GetRequiredInstanceExtensions(uint32_t *count) = 0;

    /**
     * Get the current frame buffer size.
     *
     * @param [out] width, height   framebuffer size
     */
    virtual void GetFramebufferSize(uint32_t *width, uint32_t *height) = 0;

    /**
     * Create a Vulkan surface.
     *
     * @param pysical_device    Vulkan device
     * @param instance          Vulkan instance
     * @return vulkan surface
     */
    virtual VkSurfaceKHR CreateSurface(VkPhysicalDevice pysical_device, VkInstance instance) = 0;

    /**
     * @returns true if the window should be closed
     */
    virtual bool ShouldClose() = 0;

    /**
     * @returns true if the window is minimized
     */
    virtual bool IsMinimized() = 0;

    /**
     * Start a new ImGui frame.
     */
    virtual void ImGuiNewFrame() = 0;

    /**
     * Start a new frame.
     */
    virtual void Begin() = 0;

    /**
     * End the current frame.
     */
    virtual void End() = 0;
};

} // namespace clara::holoviz

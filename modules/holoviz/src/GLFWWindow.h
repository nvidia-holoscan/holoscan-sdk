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

#include "Window.h"

#include "holoviz/InitFlags.h"

#include <memory>
#include <cstdint>

typedef struct GLFWwindow GLFWwindow;

namespace clara::holoviz
{

/**
 * Specialization of the Window class handling a GLFWwindow.
 */
class GLFWWindow : public Window
{
public:
    /**
     * Construct a new GLFWWindow object with an existing GLFWwindow object.
     *
     * @param window    existing GLFWwindow window
     */
    GLFWWindow(GLFWwindow *window);

    /**
     * Construct a new GLFWWindow object of a given size.
     *
     * @param width, height     window size
     * @param title             window tile
     * @param flags             init flags
     */
    GLFWWindow(uint32_t width, uint32_t height, const char *title, InitFlags flags);

    /**
     * Delete the standard constructor, always need parameters to construct.
     */
    GLFWWindow() = delete;

    /**
     * Destroy the GLFWWindow object.
     */
    virtual ~GLFWWindow();

    /// clara::holoviz::Window virtual members
    ///@{
    void InitImGui() override;
    void SetupCallbacks(std::function<void(int width, int height)> frame_buffer_size_cb) override;

    const char **GetRequiredInstanceExtensions(uint32_t *count) override;
    void GetFramebufferSize(uint32_t *width, uint32_t *height) override;

    VkSurfaceKHR CreateSurface(VkPhysicalDevice pysical_device, VkInstance instance) override;

    bool ShouldClose() override;
    bool IsMinimized() override;

    void ImGuiNewFrame() override;

    void Begin() override;
    void End() override;
    ///@}

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace clara::holoviz

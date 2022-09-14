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

#include "ExclusiveWindow.h"

#include <nvvk/error_vk.hpp>
#include <nvh/nvprint.hpp>

#include <X11/extensions/Xrandr.h>
#include <vulkan/vulkan_xlib_xrandr.h>

#include <imgui.h>

#include <array>
#include <vector>
#include <string>

namespace clara::holoviz
{

/**
 * ExclusiveWindow implementation details
 */
struct ExclusiveWindow::Impl
{
    std::string display_name_;
    uint32_t desired_width_        = 0;
    uint32_t desired_height_       = 0;
    uint32_t desired_refresh_rate_ = 0;

    uint32_t width_        = 0;
    uint32_t height_       = 0;
    uint32_t refresh_rate_ = 0;

    Display *dpy_ = nullptr;
};

ExclusiveWindow::~ExclusiveWindow()
{
    if (impl_->dpy_)
    {
        XCloseDisplay(impl_->dpy_);
    }
}

ExclusiveWindow::ExclusiveWindow(const char *display_name, uint32_t width, uint32_t height, uint32_t refresh_rate,
                                 InitFlags flags)
    : impl_(new Impl)
{
    impl_->display_name_         = display_name;
    impl_->desired_width_        = width;
    impl_->desired_height_       = height;
    impl_->desired_refresh_rate_ = refresh_rate;
}

void ExclusiveWindow::InitImGui() {}

void ExclusiveWindow::SetupCallbacks(std::function<void(int width, int height)> frame_buffer_size_cb) {}

const char **ExclusiveWindow::GetRequiredInstanceExtensions(uint32_t *count)
{
    static char const *extensions[]{VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_DISPLAY_EXTENSION_NAME,
                                    VK_EXT_ACQUIRE_XLIB_DISPLAY_EXTENSION_NAME,
                                    VK_EXT_DIRECT_MODE_DISPLAY_EXTENSION_NAME};

    *count = sizeof(extensions) / sizeof(extensions[0]);
    return extensions;
}

void ExclusiveWindow::GetFramebufferSize(uint32_t *width, uint32_t *height)
{
    *width  = impl_->width_;
    *height = impl_->height_;
}

VkSurfaceKHR ExclusiveWindow::CreateSurface(VkPhysicalDevice pysical_device, VkInstance instance)
{
    uint32_t display_count = 0;
    NVVK_CHECK(vkGetPhysicalDeviceDisplayPropertiesKHR(pysical_device, &display_count, nullptr));
    std::vector<VkDisplayPropertiesKHR> display_properties(display_count);
    NVVK_CHECK(vkGetPhysicalDeviceDisplayPropertiesKHR(pysical_device, &display_count, display_properties.data()));

    // pick the display
    LOGI("____________________\n");
    LOGI("Available displays :\n");
    VkDisplayPropertiesKHR selected_display = display_properties[0];
    bool found_display                      = false;
    for (auto &&displayProperty : display_properties)
    {
        LOGI("%s\n", displayProperty.displayName);
        if (std::string(displayProperty.displayName).find(impl_->display_name_) != std::string::npos)
        {
            selected_display = displayProperty;
            found_display    = true;
            break;
        }
    }
    LOGI("\n");

    if (!found_display)
    {
        LOGW("Display \"%s\" not found, using the first available display instead\n", impl_->display_name_.c_str());
    }
    LOGI("Using display \"%s\"\n", selected_display.displayName);

    const VkDisplayKHR display = selected_display.display;

    // If the X11 server is running, acquire permission from the X-Server to directly access the display in Vulkan
    impl_->dpy_ = XOpenDisplay(NULL);
    if (impl_->dpy_)
    {
        const PFN_vkAcquireXlibDisplayEXT vkAcquireXlibDisplayEXT =
            PFN_vkAcquireXlibDisplayEXT(vkGetInstanceProcAddr(instance, "vkAcquireXlibDisplayEXT"));
        if (!vkAcquireXlibDisplayEXT)
        {
            throw std::runtime_error("Could not get proc address of vkAcquireXlibDisplayEXT");
        }
        LOGI("X server is running, trying to aquire display\n");
        VkResult result = vkAcquireXlibDisplayEXT(pysical_device, impl_->dpy_, display);
        if (result < 0)
        {
            nvvk::checkResult(result);
            throw std::runtime_error("Failed to aquire display from X-Server.");
        }
    }

    // pick highest available resolution
    uint32_t mode_count = 0;
    NVVK_CHECK(vkGetDisplayModePropertiesKHR(pysical_device, display, &mode_count, nullptr));
    std::vector<VkDisplayModePropertiesKHR> modes(mode_count);
    NVVK_CHECK(vkGetDisplayModePropertiesKHR(pysical_device, display, &mode_count, modes.data()));
    VkDisplayModePropertiesKHR mode_properties = modes[0];
    // find the mode
    for (const auto &m : modes)
    {
        if (((impl_->desired_width_ > 0) && (m.parameters.visibleRegion.width >= impl_->desired_width_)) &&
            ((impl_->desired_height_ > 0) && (m.parameters.visibleRegion.height >= impl_->desired_height_)) &&
            ((impl_->desired_refresh_rate_ > 0) && (m.parameters.refreshRate >= impl_->desired_refresh_rate_)))
        {
            mode_properties = m;
        }
    }

    if (((impl_->desired_width_ > 0) && (mode_properties.parameters.visibleRegion.width != impl_->desired_width_)) ||
        ((impl_->desired_height_ > 0) && (mode_properties.parameters.visibleRegion.height != impl_->desired_height_)) ||
        ((impl_->desired_refresh_rate_ > 0) &&
         (mode_properties.parameters.refreshRate != impl_->desired_refresh_rate_)))
    {
        LOGW("Did not find a display mode with the desired properties %dx%d %.3f Hz\n", impl_->desired_width_,
             impl_->desired_height_, static_cast<float>(impl_->desired_refresh_rate_) / 1000.f);
    }
    LOGW("Using display mode %dx%d %.3f Hz\n", mode_properties.parameters.visibleRegion.width,
         mode_properties.parameters.visibleRegion.height,
         static_cast<float>(mode_properties.parameters.refreshRate) / 1000.f);

    impl_->width_        = mode_properties.parameters.visibleRegion.width;
    impl_->height_       = mode_properties.parameters.visibleRegion.height;
    impl_->refresh_rate_ = mode_properties.parameters.refreshRate;

    // pick first compatible plane
    uint32_t plane_count = 0;
    NVVK_CHECK(vkGetPhysicalDeviceDisplayPlanePropertiesKHR(pysical_device, &plane_count, nullptr));
    std::vector<VkDisplayPlanePropertiesKHR> planes(plane_count);
    NVVK_CHECK(vkGetPhysicalDeviceDisplayPlanePropertiesKHR(pysical_device, &plane_count, planes.data()));
    uint32_t plane_index;
    bool found_plane = false;
    for (uint32_t i = 0; i < planes.size(); ++i)
    {
        auto p = planes[i];

        // skip planes bound to different display
        if (p.currentDisplay && (p.currentDisplay != display))
        {
            continue;
        }

        uint32_t display_count = 0;
        NVVK_CHECK(vkGetDisplayPlaneSupportedDisplaysKHR(pysical_device, i, &display_count, nullptr));
        std::vector<VkDisplayKHR> displays(display_count);
        NVVK_CHECK(vkGetDisplayPlaneSupportedDisplaysKHR(pysical_device, i, &display_count, displays.data()));
        for (auto &d : displays)
        {
            if (d == display)
            {
                found_plane = true;
                plane_index = i;
                break;
            }
        }

        if (found_plane)
        {
            break;
        }
    }

    if (!found_plane)
    {
        throw std::runtime_error("Could not find a compatible display plane!");
    }

    // find alpha mode bit
    VkDisplayPlaneCapabilitiesKHR plane_capabilities;
    NVVK_CHECK(vkGetDisplayPlaneCapabilitiesKHR(pysical_device, mode_properties.displayMode, plane_index,
                                                &plane_capabilities));
    VkDisplayPlaneAlphaFlagBitsKHR selected_alpha_mode =
        VkDisplayPlaneAlphaFlagBitsKHR::VK_DISPLAY_PLANE_ALPHA_OPAQUE_BIT_KHR;
    const std::array<VkDisplayPlaneAlphaFlagBitsKHR, 4> available_alpha_modes{
        VkDisplayPlaneAlphaFlagBitsKHR::VK_DISPLAY_PLANE_ALPHA_OPAQUE_BIT_KHR,
        VkDisplayPlaneAlphaFlagBitsKHR::VK_DISPLAY_PLANE_ALPHA_GLOBAL_BIT_KHR,
        VkDisplayPlaneAlphaFlagBitsKHR::VK_DISPLAY_PLANE_ALPHA_PER_PIXEL_BIT_KHR,
        VkDisplayPlaneAlphaFlagBitsKHR::VK_DISPLAY_PLANE_ALPHA_PER_PIXEL_PREMULTIPLIED_BIT_KHR};
    for (auto &&alpha_mode : available_alpha_modes)
    {
        if (plane_capabilities.supportedAlpha & alpha_mode)
        {
            selected_alpha_mode = alpha_mode;
            break;
        }
    }

    VkDisplaySurfaceCreateInfoKHR surface_create_info{VK_STRUCTURE_TYPE_DISPLAY_SURFACE_CREATE_INFO_KHR};
    surface_create_info.displayMode     = mode_properties.displayMode;
    surface_create_info.planeIndex      = plane_index;
    surface_create_info.planeStackIndex = planes[plane_index].currentStackIndex;
    surface_create_info.transform       = VkSurfaceTransformFlagBitsKHR::VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    surface_create_info.globalAlpha     = 1.0f;
    surface_create_info.alphaMode       = selected_alpha_mode;
    surface_create_info.imageExtent =
        VkExtent2D{mode_properties.parameters.visibleRegion.width, mode_properties.parameters.visibleRegion.height};

    VkSurfaceKHR surface;
    NVVK_CHECK(vkCreateDisplayPlaneSurfaceKHR(instance, &surface_create_info, nullptr, &surface));

    return surface;
}

bool ExclusiveWindow::ShouldClose()
{
    return false;
}

bool ExclusiveWindow::IsMinimized()
{
    return false;
}

void ExclusiveWindow::ImGuiNewFrame()
{
    ImGuiIO &io                = ImGui::GetIO();
    io.DisplaySize             = ImVec2((float)impl_->width_, (float)impl_->height_);
    io.DisplayFramebufferScale = ImVec2(1.f, 1.f);

    ImGui::NewFrame();
}

void ExclusiveWindow::Begin() {}

void ExclusiveWindow::End() {}

} // namespace clara::holoviz
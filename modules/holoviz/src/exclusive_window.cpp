/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "exclusive_window.hpp"

#include <X11/extensions/Xrandr.h>
#include <imgui.h>
#include <vulkan/vulkan_xlib_xrandr.h>

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

#include <holoscan/logger/logger.hpp>
#include <nvh/nvprint.hpp>
#include <nvvk/error_vk.hpp>

namespace holoscan::viz {

/**
 * ExclusiveWindow implementation details
 */
struct ExclusiveWindow::Impl {
  std::string display_name_;
  uint32_t desired_width_ = 0;
  uint32_t desired_height_ = 0;
  uint32_t desired_refresh_rate_ = 0;

  uint32_t width_ = 0;
  uint32_t height_ = 0;
  uint32_t refresh_rate_ = 0;

  Display* dpy_ = nullptr;
};

ExclusiveWindow::~ExclusiveWindow() {
  if (impl_->dpy_) { XCloseDisplay(impl_->dpy_); }
}

ExclusiveWindow::ExclusiveWindow(const char* display_name, uint32_t width, uint32_t height,
                                 uint32_t refresh_rate, InitFlags flags)
    : impl_(new Impl) {
  impl_->display_name_ = display_name;
  impl_->desired_width_ = width;
  impl_->desired_height_ = height;
  impl_->desired_refresh_rate_ = refresh_rate;
}

void ExclusiveWindow::init_im_gui() {}

void ExclusiveWindow::setup_callbacks(
    std::function<void(int width, int height)> frame_buffer_size_cb) {}

const char** ExclusiveWindow::get_required_instance_extensions(uint32_t* count) {
  static char const* extensions[]{VK_KHR_SURFACE_EXTENSION_NAME,
                                  VK_KHR_DISPLAY_EXTENSION_NAME,
                                  VK_EXT_ACQUIRE_XLIB_DISPLAY_EXTENSION_NAME,
                                  VK_EXT_DIRECT_MODE_DISPLAY_EXTENSION_NAME};

  *count = sizeof(extensions) / sizeof(extensions[0]);
  return extensions;
}

const char** ExclusiveWindow::get_required_device_extensions(uint32_t* count) {
  static char const* extensions[]{VK_KHR_SWAPCHAIN_EXTENSION_NAME};

  *count = sizeof(extensions) / sizeof(extensions[0]);
  return extensions;
}

uint32_t ExclusiveWindow::select_device(vk::Instance instance,
                                        const std::vector<vk::PhysicalDevice>& physical_devices) {
  std::string first_display;
  uint32_t first_device_index;
  for (uint32_t index = 0; index < physical_devices.size(); ++index) {
    const std::vector<vk::DisplayPropertiesKHR> display_properties =
        physical_devices[index].getDisplayPropertiesKHR();
    for (auto&& displayProperty : display_properties) {
      if (std::string(displayProperty.displayName).find(impl_->display_name_) !=
          std::string::npos) {
        return index;
      }

      if (first_display.empty()) {
        first_display = displayProperty.displayName;
        first_device_index = index;
      }
    }
  }
  if (first_display.empty()) {
    throw std::runtime_error("No device with a connected display found");
  }
  HOLOSCAN_LOG_WARN("Display \"{}\" not found, using the first available display \"{}\" instead",
                    impl_->display_name_.c_str(),
                    first_display.c_str());
  return first_device_index;
}

void ExclusiveWindow::get_framebuffer_size(uint32_t* width, uint32_t* height) {
  *width = impl_->width_;
  *height = impl_->height_;
}

vk::SurfaceKHR ExclusiveWindow::create_surface(vk::PhysicalDevice physical_device,
                                               vk::Instance instance) {
  const std::vector<vk::DisplayPropertiesKHR> display_properties =
      physical_device.getDisplayPropertiesKHR();

  // pick the display
  HOLOSCAN_LOG_INFO("____________________");
  HOLOSCAN_LOG_INFO("Available displays :");
  vk::DisplayPropertiesKHR selected_display = display_properties[0];
  for (auto&& displayProperty : display_properties) {
    HOLOSCAN_LOG_INFO("{}", displayProperty.displayName);
    if (std::string(displayProperty.displayName).find(impl_->display_name_) !=
                           std::string::npos) {
      selected_display = displayProperty;
    }
  }
  HOLOSCAN_LOG_INFO("");
  HOLOSCAN_LOG_INFO("Using display \"{}\"", selected_display.displayName);

  const vk::DisplayKHR display = selected_display.display;

  // If the X11 server is running, acquire permission from the X-Server to directly
  //                                                   access the display in Vulkan
  impl_->dpy_ = XOpenDisplay(NULL);
  if (impl_->dpy_) {
    const PFN_vkAcquireXlibDisplayEXT vkAcquireXlibDisplayEXT =
        PFN_vkAcquireXlibDisplayEXT(vkGetInstanceProcAddr(instance, "vkAcquireXlibDisplayEXT"));
    if (!vkAcquireXlibDisplayEXT) {
      throw std::runtime_error("Could not get proc address of vkAcquireXlibDisplayEXT");
    }
    HOLOSCAN_LOG_INFO("X server is running, trying to acquire display");
    VkResult result = vkAcquireXlibDisplayEXT(physical_device, impl_->dpy_, display);
    if (result < 0) {
      nvvk::checkResult(result);
      throw std::runtime_error("Failed to acquire display from X-Server.");
    }
  }

  // pick highest available resolution
  const std::vector<vk::DisplayModePropertiesKHR> modes =
      physical_device.getDisplayModePropertiesKHR(display);
  vk::DisplayModePropertiesKHR mode_properties = modes[0];
  // find the mode
  for (const auto& m : modes) {
    if (((impl_->desired_width_ > 0) &&
         (m.parameters.visibleRegion.width >= impl_->desired_width_)) &&
        ((impl_->desired_height_ > 0) &&
         (m.parameters.visibleRegion.height >= impl_->desired_height_)) &&
        ((impl_->desired_refresh_rate_ > 0) &&
         (m.parameters.refreshRate >= impl_->desired_refresh_rate_))) {
      mode_properties = m;
    }
  }

  if (((impl_->desired_width_ > 0) &&
       (mode_properties.parameters.visibleRegion.width != impl_->desired_width_)) ||
      ((impl_->desired_height_ > 0) &&
       (mode_properties.parameters.visibleRegion.height != impl_->desired_height_)) ||
      ((impl_->desired_refresh_rate_ > 0) &&
       (mode_properties.parameters.refreshRate != impl_->desired_refresh_rate_))) {
    HOLOSCAN_LOG_WARN("Did not find a display mode with the desired properties {}x{} {:.3f} Hz",
                      impl_->desired_width_,
                      impl_->desired_height_,
                      static_cast<float>(impl_->desired_refresh_rate_) / 1000.f);
  }
  HOLOSCAN_LOG_INFO("Using display mode {}x{} {:.3f} Hz",
                    mode_properties.parameters.visibleRegion.width,
                    mode_properties.parameters.visibleRegion.height,
                    static_cast<float>(mode_properties.parameters.refreshRate) / 1000.f);

  impl_->width_ = mode_properties.parameters.visibleRegion.width;
  impl_->height_ = mode_properties.parameters.visibleRegion.height;
  impl_->refresh_rate_ = mode_properties.parameters.refreshRate;

  // pick first compatible plane
  const std::vector<vk::DisplayPlanePropertiesKHR> planes =
      physical_device.getDisplayPlanePropertiesKHR();
  uint32_t plane_index;
  bool found_plane = false;
  for (uint32_t i = 0; i < planes.size(); ++i) {
    auto p = planes[i];

    // skip planes bound to different display
    if (p.currentDisplay && (p.currentDisplay != display)) { continue; }

    const std::vector<vk::DisplayKHR> displays =
        physical_device.getDisplayPlaneSupportedDisplaysKHR(i);
    for (auto& d : displays) {
      if (d == display) {
        found_plane = true;
        plane_index = i;
        break;
      }
    }

    if (found_plane) { break; }
  }

  if (!found_plane) { throw std::runtime_error("Could not find a compatible display plane!"); }

  // find alpha mode bit
  const vk::DisplayPlaneCapabilitiesKHR plane_capabilities =
      physical_device.getDisplayPlaneCapabilitiesKHR(mode_properties.displayMode, plane_index);
  vk::DisplayPlaneAlphaFlagBitsKHR selected_alpha_mode = vk::DisplayPlaneAlphaFlagBitsKHR::eOpaque;
  const std::array<vk::DisplayPlaneAlphaFlagBitsKHR, 4> available_alpha_modes{
      vk::DisplayPlaneAlphaFlagBitsKHR::eOpaque,
      vk::DisplayPlaneAlphaFlagBitsKHR::eGlobal,
      vk::DisplayPlaneAlphaFlagBitsKHR::ePerPixel,
      vk::DisplayPlaneAlphaFlagBitsKHR::ePerPixelPremultiplied};
  for (auto&& alpha_mode : available_alpha_modes) {
    if (plane_capabilities.supportedAlpha & alpha_mode) {
      selected_alpha_mode = alpha_mode;
      break;
    }
  }

  vk::DisplaySurfaceCreateInfoKHR surface_create_info;
  surface_create_info.displayMode = mode_properties.displayMode;
  surface_create_info.planeIndex = plane_index;
  surface_create_info.planeStackIndex = planes[plane_index].currentStackIndex;
  surface_create_info.transform = vk::SurfaceTransformFlagBitsKHR::eIdentity;
  surface_create_info.globalAlpha = 1.0f;
  surface_create_info.alphaMode = selected_alpha_mode;
  surface_create_info.imageExtent = vk::Extent2D{mode_properties.parameters.visibleRegion.width,
                                                 mode_properties.parameters.visibleRegion.height};

  return instance.createDisplayPlaneSurfaceKHR(surface_create_info);
}

bool ExclusiveWindow::should_close() {
  return false;
}

bool ExclusiveWindow::is_minimized() {
  return false;
}

void ExclusiveWindow::im_gui_new_frame() {
  ImGuiIO& io = ImGui::GetIO();
  io.DisplaySize = ImVec2(static_cast<float>(impl_->width_), static_cast<float>(impl_->height_));
  io.DisplayFramebufferScale = ImVec2(1.f, 1.f);

  ImGui::NewFrame();
}

void ExclusiveWindow::begin() {}

void ExclusiveWindow::end() {}

float ExclusiveWindow::get_aspect_ratio() {
  return float(impl_->width_) / float(impl_->height_);
}

}  // namespace holoscan::viz

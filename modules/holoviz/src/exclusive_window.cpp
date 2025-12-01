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

#include "exclusive_window.hpp"

#include <imgui.h>

#include <array>
#include <cstdint>
#include <limits>
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
  vk::DisplayKHR display_;
  uint32_t desired_width_ = 0;
  uint32_t desired_height_ = 0;
  uint32_t desired_refresh_rate_ = 0;

  uint32_t width_ = 0;
  uint32_t height_ = 0;
  uint32_t refresh_rate_ = 0;
  vk::PhysicalDevicePresentIdFeaturesKHR present_id_feature_;
  vk::PhysicalDevicePresentWaitFeaturesKHR present_wait_feature_;
};

ExclusiveWindow::~ExclusiveWindow() {}

ExclusiveWindow::ExclusiveWindow(const char* display_name, uint32_t width, uint32_t height,
                                 uint32_t refresh_rate, InitFlags flags)
    : impl_(new Impl) {
  impl_->display_name_ = display_name;
  impl_->desired_width_ = width;
  impl_->desired_height_ = height;
  impl_->desired_refresh_rate_ = refresh_rate;
  impl_->present_id_feature_.presentId = true;
  impl_->present_wait_feature_.presentWait = true;
}

void ExclusiveWindow::init_im_gui() {}

std::vector<Window::InstanceExtensionInfo> ExclusiveWindow::get_required_instance_extensions() {
  return {{VK_KHR_SURFACE_EXTENSION_NAME},
          {VK_KHR_DISPLAY_EXTENSION_NAME},
          {VK_EXT_DISPLAY_SURFACE_COUNTER_EXTENSION_NAME}};
}

std::vector<Window::DeviceExtensionInfo> ExclusiveWindow::get_required_device_extensions() {
  std::vector<Window::DeviceExtensionInfo> device_extensions = {
      {VK_KHR_SWAPCHAIN_EXTENSION_NAME},
      {VK_EXT_DISPLAY_CONTROL_EXTENSION_NAME},
      {VK_KHR_PRESENT_ID_EXTENSION_NAME, true /*optional*/, &impl_->present_id_feature_},
      {VK_KHR_PRESENT_WAIT_EXTENSION_NAME, true /*optional*/, &impl_->present_wait_feature_},
  };
  return device_extensions;
}

uint32_t ExclusiveWindow::select_device(vk::Instance instance,
                                        const std::vector<vk::PhysicalDevice>& physical_devices) {
  std::string first_display;
  bool found_display = false;
  uint32_t display_device_index;
  uint32_t first_device_index = 0;

  for (uint32_t index = 0; (index < physical_devices.size()) && (!found_display); ++index) {
    const std::vector<vk::DisplayPropertiesKHR> display_properties =
        physical_devices[index].getDisplayPropertiesKHR();
    for (auto&& displayProperty : display_properties) {
      if (std::string(displayProperty.displayName).find(impl_->display_name_) !=
          std::string::npos) {
        impl_->display_ = displayProperty.display;
        display_device_index = index;
        found_display = true;
        break;
      }

      if (first_display.empty()) {
        first_display = displayProperty.displayName;
        impl_->display_ = displayProperty.display;
        first_device_index = index;
      }
    }
  }
  if (!found_display) {
    if (first_display.empty()) {
      throw std::runtime_error("No device with a connected display found");
    }

    HOLOSCAN_LOG_WARN("Display \"{}\" not found, using the first available display \"{}\" instead",
                      impl_->display_name_,
                      first_display);
    HOLOSCAN_LOG_INFO("____________________");
    HOLOSCAN_LOG_INFO("Available displays :");
    for (uint32_t index = 0; index < physical_devices.size(); ++index) {
      const std::vector<vk::DisplayPropertiesKHR> display_properties =
          physical_devices[index].getDisplayPropertiesKHR();
      for (auto&& displayProperty : display_properties) {
        HOLOSCAN_LOG_INFO(" {}", displayProperty.displayName);
      }
    }
    return first_device_index;
  }

  HOLOSCAN_LOG_INFO("Using display \"{}\"", impl_->display_name_);
  return display_device_index;
}

void ExclusiveWindow::get_framebuffer_size(uint32_t* width, uint32_t* height) {
  *width = impl_->width_;
  *height = impl_->height_;
}

void ExclusiveWindow::get_window_size(uint32_t* width, uint32_t* height) {
  get_framebuffer_size(width, height);
}

vk::SurfaceKHR ExclusiveWindow::create_surface(vk::PhysicalDevice physical_device,
                                               vk::Instance instance) {
  // find the best mode that meets all requirements
  const std::vector<vk::DisplayModePropertiesKHR> modes =
      physical_device.getDisplayModePropertiesKHR(impl_->display_);

  uint32_t least_width_diff = std::numeric_limits<uint32_t>::max();
  uint32_t least_height_diff = std::numeric_limits<uint32_t>::max();
  uint32_t least_rate_diff = std::numeric_limits<uint32_t>::max();
  vk::DisplayModePropertiesKHR mode_properties = modes[0];

  for (const auto& m : modes) {
    uint32_t width_diff;
    if (impl_->desired_width_ > 0) {
      width_diff = std::abs(static_cast<int32_t>(m.parameters.visibleRegion.width) -
                            static_cast<int32_t>(impl_->desired_width_));
    } else {
      width_diff = std::numeric_limits<uint32_t>::max() - m.parameters.visibleRegion.width;
    }
    uint32_t height_diff;
    if (impl_->desired_height_ > 0) {
      height_diff = std::abs(static_cast<int32_t>(m.parameters.visibleRegion.height) -
                             static_cast<int32_t>(impl_->desired_height_));
    } else {
      height_diff = std::numeric_limits<uint32_t>::max() - m.parameters.visibleRegion.height;
    }
    uint32_t rate_diff;
    if (impl_->desired_refresh_rate_ > 0) {
      rate_diff = std::abs(static_cast<int32_t>(m.parameters.refreshRate) -
                           static_cast<int32_t>(impl_->desired_refresh_rate_));
    } else {
      rate_diff = std::numeric_limits<uint32_t>::max() - m.parameters.refreshRate;
    }

    if ((width_diff < least_width_diff) ||
        ((width_diff == least_width_diff) && (height_diff < least_height_diff)) ||
        ((width_diff == least_width_diff) && (height_diff == least_height_diff) &&
         (rate_diff < least_rate_diff))) {
      mode_properties = m;
      least_width_diff = width_diff;
      least_height_diff = height_diff;
      least_rate_diff = rate_diff;
    }
  }

  if (((impl_->desired_width_ > 0) && (least_width_diff != 0)) ||
      ((impl_->desired_height_ > 0) && (least_height_diff != 0)) ||
      ((impl_->desired_refresh_rate_ > 0) && (least_rate_diff != 0))) {
    HOLOSCAN_LOG_WARN(
        "Did not find a display mode with the desired properties {}x{} {:.3f} Hz, using closest "
        "match",
        impl_->desired_width_,
        impl_->desired_height_,
        static_cast<float>(impl_->desired_refresh_rate_) / 1000.F);
    HOLOSCAN_LOG_INFO("Available display modes:");
    for (const auto& m : modes) {
      HOLOSCAN_LOG_INFO(" {}x{} {:.3f} Hz",
                        m.parameters.visibleRegion.width,
                        m.parameters.visibleRegion.height,
                        static_cast<float>(m.parameters.refreshRate) / 1000.F);
    }
  }
  HOLOSCAN_LOG_INFO("Using display mode {}x{} {:.3f} Hz",
                    mode_properties.parameters.visibleRegion.width,
                    mode_properties.parameters.visibleRegion.height,
                    static_cast<float>(mode_properties.parameters.refreshRate) / 1000.F);

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
    if (p.currentDisplay && (p.currentDisplay != impl_->display_)) {
      continue;
    }

    const std::vector<vk::DisplayKHR> displays =
        physical_device.getDisplayPlaneSupportedDisplaysKHR(i);
    for (auto& d : displays) {
      if (d == impl_->display_) {
        found_plane = true;
        plane_index = i;
        break;
      }
    }

    if (found_plane) {
      break;
    }
  }

  if (!found_plane) {
    throw std::runtime_error("Could not find a compatible display plane!");
  }

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
  surface_create_info.globalAlpha = 1.0F;
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
  io.DisplayFramebufferScale = ImVec2(1.F, 1.F);

  ImGui::NewFrame();
}

void ExclusiveWindow::begin() {}

void ExclusiveWindow::end() {
  // call the base class
  Window::end();
}

float ExclusiveWindow::get_aspect_ratio() {
  return float(impl_->width_) / float(impl_->height_);
}

vk::DisplayKHR ExclusiveWindow::get_display() {
  return impl_->display_;
}

}  // namespace holoscan::viz

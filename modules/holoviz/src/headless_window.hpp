/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_VIZ_HEADLESSWINDOW_HPP
#define HOLOSCAN_VIZ_HEADLESSWINDOW_HPP

#include <cstdint>
#include <memory>
#include <vector>

#include "holoviz/init_flags.hpp"
#include "window.hpp"

namespace holoscan::viz {

/**
 * Specialization of the Window class handling a headless display.
 */
class HeadlessWindow : public Window {
 public:
  /**
   * Construct a new headless window.
   *
   * @param width         desired width, ignored if 0
   * @param height        desired height, ignored if 0
   * @param flags         init flags
   */
  HeadlessWindow(uint32_t width, uint32_t height, InitFlags flags);

  /**
   * Delete the standard constructor, always need parameters to construct.
   */
  HeadlessWindow() = delete;

  /**
   * Destroy the headless window object.
   */
  virtual ~HeadlessWindow();

  /// holoscan::viz::Window virtual members
  ///@{
  void init_im_gui() override;

  std::vector<Window::InstanceExtensionInfo> get_required_instance_extensions() override;
  std::vector<Window::DeviceExtensionInfo> get_required_device_extensions() override;
  uint32_t select_device(vk::Instance instance,
                         const std::vector<vk::PhysicalDevice>& physical_devices) override;
  void get_framebuffer_size(uint32_t* width, uint32_t* height) override;
  void get_window_size(uint32_t* width, uint32_t* height) override;

  vk::SurfaceKHR create_surface(vk::PhysicalDevice physical_device, vk::Instance instance) override;

  bool should_close() override;
  bool is_minimized() override;

  void im_gui_new_frame() override;

  void begin() override;
  void end() override;

  float get_aspect_ratio() override;
  ///@}

 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace holoscan::viz

#endif /* HOLOSCAN_VIZ_HEADLESSWINDOW_HPP */

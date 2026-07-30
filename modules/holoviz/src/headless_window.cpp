/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "headless_window.hpp"

#include <imgui.h>

#include <vector>

namespace holoscan::viz {

/**
 * HeadlessWindow implementation details
 */
struct HeadlessWindow::Impl {
  uint32_t width_ = 0;
  uint32_t height_ = 0;
};

HeadlessWindow::~HeadlessWindow() {}

HeadlessWindow::HeadlessWindow(uint32_t width, uint32_t height, InitFlags flags) : impl_(new Impl) {
  impl_->width_ = width;
  impl_->height_ = height;
}

void HeadlessWindow::init_im_gui() {}

std::vector<Window::InstanceExtensionInfo> HeadlessWindow::get_required_instance_extensions() {
  return {};
}

std::vector<Window::DeviceExtensionInfo> HeadlessWindow::get_required_device_extensions() {
  return {};
}

uint32_t HeadlessWindow::select_device(vk::Instance instance,
                                       const std::vector<vk::PhysicalDevice>& physical_devices) {
  // headless can be on any device so select the first one
  return 0;
}

void HeadlessWindow::get_framebuffer_size(uint32_t* width, uint32_t* height) {
  *width = impl_->width_;
  *height = impl_->height_;
}

void HeadlessWindow::get_window_size(uint32_t* width, uint32_t* height) {
  get_framebuffer_size(width, height);
}

vk::SurfaceKHR HeadlessWindow::create_surface(vk::PhysicalDevice physical_device,
                                              vk::Instance instance) {
  return VK_NULL_HANDLE;
}

bool HeadlessWindow::should_close() {
  return false;
}

bool HeadlessWindow::is_minimized() {
  return false;
}

void HeadlessWindow::im_gui_new_frame() {
  ImGuiIO& io = ImGui::GetIO();
  io.DisplaySize = ImVec2(static_cast<float>(impl_->width_), static_cast<float>(impl_->height_));
  io.DisplayFramebufferScale = ImVec2(1.F, 1.F);

  ImGui::NewFrame();
}

void HeadlessWindow::begin() {}

void HeadlessWindow::end() {
  // call the base class
  Window::end();
}

float HeadlessWindow::get_aspect_ratio() {
  return float(impl_->width_) / float(impl_->height_);
}

}  // namespace holoscan::viz

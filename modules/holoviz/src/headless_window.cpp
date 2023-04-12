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

#include "headless_window.hpp"

#include <imgui.h>

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

void HeadlessWindow::setup_callbacks(
    std::function<void(int width, int height)> frame_buffer_size_cb) {}

const char** HeadlessWindow::get_required_instance_extensions(uint32_t* count) {
  static char const* extensions[]{};

  *count = sizeof(extensions) / sizeof(extensions[0]);
  return extensions;
}

const char** HeadlessWindow::get_required_device_extensions(uint32_t* count) {
  *count = 0;
  return nullptr;
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
  io.DisplayFramebufferScale = ImVec2(1.f, 1.f);

  ImGui::NewFrame();
}

void HeadlessWindow::begin() {}

void HeadlessWindow::end() {}

float HeadlessWindow::get_aspect_ratio() {
  return float(impl_->width_) / float(impl_->height_);
}

}  // namespace holoscan::viz

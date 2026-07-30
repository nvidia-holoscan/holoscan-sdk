/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "window.hpp"

#include <nvh/cameramanipulator.hpp>

namespace holoscan::viz {

Window::Window() {
  // setup camera
  CameraManip.setLookat(
      nvmath::vec3f(0.F, 0.F, 1.F), nvmath::vec3f(0.F, 0.F, 0.F), nvmath::vec3f(0.F, 1.F, 0.F));
}

void Window::end() {
  // update the camera
  CameraManip.updateAnim();
}

void Window::set_camera(const nvmath::vec3f& eye, const nvmath::vec3f& look_at,
                        const nvmath::vec3f& up, bool anim) {
  CameraManip.setLookat(eye, look_at, up, !anim);
}

void Window::get_view_matrix(nvmath::mat4f* view_matrix) {
  *view_matrix = nvmath::perspectiveVK(CameraManip.getFov(), 1.F /*aspectRatio*/, 0.1F, 1000.0F) *
                 CameraManip.getMatrix();
}

void Window::get_camera_matrix(nvmath::mat4f* camera_matrix) {
  *camera_matrix = CameraManip.getMatrix();
}

}  // namespace holoscan::viz

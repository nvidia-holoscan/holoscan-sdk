/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

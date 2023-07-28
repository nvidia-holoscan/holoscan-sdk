/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gtest/gtest.h>

#include <holoviz/holoviz.hpp>
#include "headless_fixture.hpp"

namespace viz = holoscan::viz;

class CameraPose : public TestHeadless {};

TEST_F(CameraPose, Get) {
  std::array<float, 16> pose;

  // it's an error to specify a size less than 16
  EXPECT_THROW(viz::GetCameraPose(15, pose.data()), std::invalid_argument);
  // it's an error to specify a null pointer for matrix
  EXPECT_THROW(viz::GetCameraPose(16, nullptr), std::invalid_argument);

  // in headless mode the camera matrix is the identity matrix
  EXPECT_NO_THROW(viz::GetCameraPose(pose.size(), pose.data()));
  for (uint32_t row = 0; row < 4; ++row) {
    for (uint32_t col = 0; col < 4; ++col) {
      EXPECT_TRUE(pose[row * 4 + col] == ((row == col) ? 1.f : 0.f));
    }
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <thread>

#include <holoviz/holoviz.hpp>
#include "test_fixture.hpp"

namespace viz = holoscan::viz;

class CameraPose : public TestHeadless {};

TEST_F(CameraPose, Set) {
  EXPECT_NO_THROW(viz::SetCamera(1.F, 2.F, 3.F, 4.F, 5.F, 6.F, 0.F, 1.F, 0.F));

  float rotation[9];
  float translation[3];
  EXPECT_NO_THROW(viz::GetCameraPose(rotation, translation));

  // There are test errors on some systems when using EXPECT_FLOAT_EQ() (includes a error margin of
  // 4 ULP, see https://google.github.io/googletest/reference/assertions.html#floating-point).
  // Use EXPECT_NEAR() with a higher epsilon.
  constexpr float epsilon = 1e-6F;
  EXPECT_NEAR(rotation[0], -0.707106769F, epsilon);
  EXPECT_NEAR(rotation[1], 0.F, epsilon);
  EXPECT_NEAR(rotation[2], 0.707106769F, epsilon);
  EXPECT_NEAR(rotation[3], -0.408248335F, epsilon);
  EXPECT_NEAR(rotation[4], 0.81649667F, epsilon);
  EXPECT_NEAR(rotation[5], -0.408248335F, epsilon);
  EXPECT_NEAR(rotation[6], -0.577350259F, epsilon);
  EXPECT_NEAR(rotation[7], -0.577350259F, epsilon);
  EXPECT_NEAR(rotation[8], -0.577350259F, epsilon);
  EXPECT_NEAR(translation[0], -1.41421342F, epsilon);
  EXPECT_NEAR(translation[1], 0.F, epsilon);
  EXPECT_NEAR(translation[2], 3.46410155F, epsilon);
}

TEST_F(CameraPose, GetDefault) {
  float rotation[9] = {0.F};
  float translation[3] = {0.F, 0.F, 0.F};

  EXPECT_NO_THROW(viz::GetCameraPose(rotation, translation));
  for (uint32_t row = 0; row < 3; ++row) {
    for (uint32_t col = 0; col < 3; ++col) {
      EXPECT_FLOAT_EQ(rotation[row * 3 + col], ((row == col) ? 1.F : 0.F));
    }
  }
  EXPECT_FLOAT_EQ(translation[0], 0.F);
  EXPECT_FLOAT_EQ(translation[1], 0.F);
  EXPECT_FLOAT_EQ(translation[2], -1.F);

  std::array<float, 16> pose;
  // it's an error to specify a size less than 16
  EXPECT_THROW(viz::GetCameraPose(15, pose.data()), std::invalid_argument);
  // it's an error to specify a null pointer for matrix
  EXPECT_THROW(viz::GetCameraPose(16, nullptr), std::invalid_argument);

  // this is the default setup for the matrix, see Window class constructor
  std::array<float, 16> expected_pose{1.73205066F,
                                      0.F,
                                      0.F,
                                      0.F,
                                      0.F,
                                      -1.73205066F,
                                      0.F,
                                      0.F,
                                      0.F,
                                      0.F,
                                      -1.00010002F,
                                      0.900090039F,
                                      0.F,
                                      0.F,
                                      -1.F,
                                      1.F};
  EXPECT_NO_THROW(viz::GetCameraPose(pose.size(), pose.data()));
  for (int i = 0; i < 16; ++i) {
    EXPECT_FLOAT_EQ(pose[i], expected_pose[i]);
  }
}

TEST_F(CameraPose, Anim) {
  // move the camera in x direction
  EXPECT_NO_THROW(viz::SetCamera(100.F, 0.F, 1.F, 100.F, 0.F, 0.F, 0.F, 1.F, 0.F, true));

  // start animation (duration default is 500 ms)
  EXPECT_NO_THROW(viz::Begin());
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_NO_THROW(viz::End());

  float rotation[9] = {0.F};
  float translation[3] = {0.F, 0.F, 0.F};

  EXPECT_NO_THROW(viz::GetCameraPose(rotation, translation));
  // translation has changed
  EXPECT_NE(translation[0], 0.F);
  EXPECT_FLOAT_EQ(translation[1], 0.F);
  EXPECT_NE(translation[2], -1.F);

  // wait for the end
  EXPECT_NO_THROW(viz::Begin());
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  EXPECT_NO_THROW(viz::End());

  EXPECT_NO_THROW(viz::GetCameraPose(rotation, translation));
  for (uint32_t row = 0; row < 3; ++row) {
    for (uint32_t col = 0; col < 3; ++col) {
      EXPECT_FLOAT_EQ(rotation[row * 3 + col], ((row == col) ? 1.F : 0.F));
    }
  }

  EXPECT_FLOAT_EQ(translation[0], -100.F);
  EXPECT_FLOAT_EQ(translation[1], 0.F);
  EXPECT_FLOAT_EQ(translation[2], -1.F);
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "holoscan/pose_tree/pose_tree_history.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <random>
#include <thread>

#include "holoscan/pose_tree/math/pose3.hpp"

namespace holoscan {

TEST(PoseTreeEdgeHistory, NotInitialized) {
  PoseTreeEdgeHistory history;
  EXPECT_FALSE(history.set(1.0, Pose3d::identity(), 42));
  EXPECT_FALSE(history.at(0));
  EXPECT_FALSE(history.get(1.0, PoseTreeEdgeHistory::AccessMethod::kNearest, 42));
  history.reset();  // Check no crash.
}

TEST(PoseTreeEdgeHistory, EmptyBuffer) {
  PoseTreeEdgeHistory history(0, 1, 512, nullptr);
  EXPECT_FALSE(history.set(1.0, Pose3d::identity(), 42));
  EXPECT_FALSE(history.at(0));
  EXPECT_FALSE(history.get(1.0, PoseTreeEdgeHistory::AccessMethod::kNearest, 42));
  history.reset();  // Check no crash.
}

TEST(PoseTreeEdgeHistory, set) {
  PoseTreeEdgeHistory::TimedPose buffer[16];
  PoseTreeEdgeHistory history(0, 1, 16, buffer);
  EXPECT_TRUE(history.set(1.0, Pose3d::identity(), 1));
  // Version is in the past
  EXPECT_FALSE(history.set(3.0, Pose3d::identity(), 1));
  EXPECT_FALSE(history.set(3.0, Pose3d::identity(), 0));
  // Time is in the past
  EXPECT_FALSE(history.set(1.0, Pose3d::identity(), 2));
  EXPECT_FALSE(history.set(0.5, Pose3d::identity(), 2));

  for (int it = 2; it < 128; it++) {
    EXPECT_TRUE(history.set(static_cast<double>(it), Pose3d::identity(), it));
  }
}

TEST(PoseTreeEdgeHistory, at) {
  PoseTreeEdgeHistory::TimedPose buffer[16];
  PoseTreeEdgeHistory history(0, 1, 16, buffer);
  history.set(1.0, Pose3d::identity(), 1);
  history.set(2.0, Pose3d::identity(), 2);

  EXPECT_FALSE(history.at(-1));
  EXPECT_FALSE(history.at(2));

  EXPECT_EQ(history.at(0).value().time, 1.0);
  EXPECT_EQ(history.at(1).value().time, 2.0);

  for (int it = 3; it < 128; it++) {
    history.set(static_cast<double>(it), Pose3d::identity(), it);
  }
  EXPECT_FALSE(history.at(16));
  EXPECT_EQ(history.at(15).value().time, 127.0);
  EXPECT_EQ(history.at(0).value().time, 112.0);
}

TEST(PoseTreeEdgeHistory, latest) {
  PoseTreeEdgeHistory::TimedPose buffer[16];
  PoseTreeEdgeHistory history(0, 1, 16, buffer);
  EXPECT_FALSE(history.latest());
  for (int i = 1; i <= 32; i++) {
    history.set(static_cast<double>(i), Pose3d::identity(), i);
    EXPECT_EQ(history.latest().value().time, static_cast<double>(i));
  }
}

TEST(PoseTreeEdgeHistory, get_kNearest) {
  const auto method = PoseTreeEdgeHistory::AccessMethod::kNearest;
  PoseTreeEdgeHistory::TimedPose buffer[16];
  PoseTreeEdgeHistory history(0, 1, 16, buffer);
  history.set(1.0, Pose3d::from_translation(Vector3d(1.0, 0.0, 0.0)), 1);
  history.set(2.0, Pose3d::from_translation(Vector3d(2.0, 0.0, 0.0)), 2);

  EXPECT_FALSE(history.get(-1.0, method, 1024));
  EXPECT_FALSE(history.get(1.5, method, 0));

  EXPECT_EQ(history.get(1.0, method, 1024).value().translation.x(), 1.0);
  EXPECT_EQ(history.get(1.49, method, 1024).value().translation.x(), 1.0);
  EXPECT_EQ(history.get(1.51, method, 1024).value().translation.x(), 2.0);
  EXPECT_EQ(history.get(1.51, method, 1).value().translation.x(), 1.0);
  EXPECT_EQ(history.get(11.51, method, 1024).value().translation.x(), 2.0);

  for (int it = 3; it < 128; it++) {
    history.set(static_cast<double>(it),
                Pose3d::from_translation(Vector3d(static_cast<double>(it), 0.0, 0.0)),
                it);
  }
  EXPECT_FALSE(history.get(111.99, method, 1024));
  EXPECT_EQ(history.get(115.51, method, 1024).value().translation.x(), 116.0);
  EXPECT_EQ(history.get(112.0, method, 1024).value().translation.x(), 112.0);
}

TEST(PoseTreeEdgeHistory, get_kPrevious) {
  const auto method = PoseTreeEdgeHistory::AccessMethod::kPrevious;
  PoseTreeEdgeHistory::TimedPose buffer[16];
  PoseTreeEdgeHistory history(0, 1, 16, buffer);
  history.set(1.0, Pose3d::from_translation(Vector3d(1.0, 0.0, 0.0)), 1);
  history.set(2.0, Pose3d::from_translation(Vector3d(2.0, 0.0, 0.0)), 2);

  EXPECT_FALSE(history.get(-1.0, method, 1024));
  EXPECT_FALSE(history.get(1.5, method, 0));

  EXPECT_EQ(history.get(1.0, method, 1024).value().translation.x(), 1.0);
  EXPECT_EQ(history.get(1.49, method, 1024).value().translation.x(), 1.0);
  EXPECT_EQ(history.get(1.51, method, 1024).value().translation.x(), 1.0);
  EXPECT_EQ(history.get(1.51, method, 1).value().translation.x(), 1.0);
  EXPECT_EQ(history.get(11.51, method, 1024).value().translation.x(), 2.0);

  for (int it = 3; it < 128; it++) {
    history.set(static_cast<double>(it),
                Pose3d::from_translation(Vector3d(static_cast<double>(it), 0.0, 0.0)),
                it);
  }
  EXPECT_FALSE(history.get(111.99, method, 1024));
  EXPECT_EQ(history.get(115.51, method, 1024).value().translation.x(), 115.0);
  EXPECT_EQ(history.get(112.0, method, 1024).value().translation.x(), 112.0);
}

TEST(PoseTreeEdgeHistory, get_kInterpolateLinearly) {
  const auto method = PoseTreeEdgeHistory::AccessMethod::kInterpolateLinearly;
  PoseTreeEdgeHistory::TimedPose buffer[16];
  PoseTreeEdgeHistory history(0, 1, 16, buffer);
  history.set(1.0, Pose3d::from_translation(Vector3d(1.0, 0.0, 0.0)), 1);
  history.set(2.0, Pose3d::from_translation(Vector3d(2.0, 0.0, 0.0)), 2);

  EXPECT_FALSE(history.get(-1.0, method, 1024));
  EXPECT_FALSE(history.get(1.5, method, 0));

  EXPECT_EQ(history.get(1.0, method, 1024).value().translation.x(), 1.0);
  EXPECT_EQ(history.get(1.49, method, 1024).value().translation.x(), 1.49);
  EXPECT_EQ(history.get(1.51, method, 1024).value().translation.x(), 1.51);
  EXPECT_EQ(history.get(1.51, method, 1).value().translation.x(), 1.0);
  EXPECT_EQ(history.get(11.51, method, 1024).value().translation.x(), 2.0);

  for (int it = 3; it < 128; it++) {
    history.set(static_cast<double>(it),
                Pose3d::from_translation(Vector3d(static_cast<double>(it), 0.0, 0.0)),
                it);
  }
  EXPECT_FALSE(history.get(111.99, method, 1024));
  EXPECT_EQ(history.get(115.51, method, 1024).value().translation.x(), 115.51);
  EXPECT_EQ(history.get(112.0, method, 1024).value().translation.x(), 112.0);
}

TEST(PoseTreeEdgeHistory, get_kExtrapolateLinearly) {
  const auto method = PoseTreeEdgeHistory::AccessMethod::kExtrapolateLinearly;
  PoseTreeEdgeHistory::TimedPose buffer[16];
  PoseTreeEdgeHistory history(0, 1, 16, buffer);
  history.set(1.0, Pose3d::from_translation(Vector3d(1.0, 0.0, 0.0)), 1);
  history.set(2.0, Pose3d::from_translation(Vector3d(2.0, 0.0, 0.0)), 2);
  history.set(3.0, Pose3d::from_translation(Vector3d(3.0, 0.0, 0.0)), 3);

  EXPECT_FALSE(history.get(-1.0, method, 1024));
  EXPECT_FALSE(history.get(1.5, method, 0));
  EXPECT_FALSE(history.get(1.5, method, 1));

  EXPECT_EQ(history.get(1.0, method, 1024).value().translation.x(), 1.0);
  EXPECT_EQ(history.get(1.49, method, 1024).value().translation.x(), 1.49);
  EXPECT_EQ(history.get(1.51, method, 1024).value().translation.x(), 1.51);
  EXPECT_EQ(history.get(11.51, method, 1024).value().translation.x(), 11.51);

  for (int it = 3; it < 128; it++) {
    history.set(static_cast<double>(it),
                Pose3d::from_translation(Vector3d(static_cast<double>(it), 0.0, 0.0)),
                it);
  }
  EXPECT_FALSE(history.get(111.99, method, 1024));
  EXPECT_EQ(history.get(115.51, method, 1024).value().translation.x(), 115.51);
  EXPECT_EQ(history.get(112.0, method, 1024).value().translation.x(), 112.0);
}

TEST(PoseTreeEdgeHistory, get_kInterpolateSlerp) {
  const auto method = PoseTreeEdgeHistory::AccessMethod::kInterpolateSlerp;
  PoseTreeEdgeHistory::TimedPose buffer[16];
  PoseTreeEdgeHistory history(0, 1, 16, buffer);
  history.set(1.0, Pose3d::from_translation(Vector3d(1.0, 0.0, 0.0)), 1);
  history.set(2.0, Pose3d::from_translation(Vector3d(2.0, 0.0, 0.0)), 2);

  EXPECT_FALSE(history.get(-1.0, method, 1024));
  EXPECT_FALSE(history.get(1.5, method, 0));

  EXPECT_EQ(history.get(1.0, method, 1024).value().translation.x(), 1.0);
  EXPECT_EQ(history.get(1.49, method, 1024).value().translation.x(), 1.49);
  EXPECT_EQ(history.get(1.51, method, 1024).value().translation.x(), 1.51);
  EXPECT_EQ(history.get(1.51, method, 1).value().translation.x(), 1.0);
  EXPECT_EQ(history.get(11.51, method, 1024).value().translation.x(), 2.0);

  for (int it = 3; it < 128; it++) {
    history.set(static_cast<double>(it),
                Pose3d::from_translation(Vector3d(static_cast<double>(it), 0.0, 0.0)),
                it);
  }
  EXPECT_FALSE(history.get(111.99, method, 1024));
  EXPECT_EQ(history.get(115.51, method, 1024).value().translation.x(), 115.51);
  EXPECT_EQ(history.get(112.0, method, 1024).value().translation.x(), 112.0);

  history.set(1000.0, Pose3d::identity(), 1000);
  history.set(
      1002.0,
      Pose3d{SO3d::from_angle_axis(M_PI * 0.5, Vector3d(0.0, 1.0, 0.0)), Vector3d(3.0, 3.0, 3.0)},
      1001);
  EXPECT_NEAR(history.get(1001.0, method, 1024).value().rotation.angle(), M_PI * 0.25, 1.0e-8);
  EXPECT_NEAR(history.get(1001.0, method, 1024).value().translation.x(),
              (2.0 - std::sqrt(2.0)) * 1.5,
              1.0e-8);
  EXPECT_NEAR(history.get(1001.0, method, 1024).value().translation.y(), 1.5, 1e-8);
  EXPECT_NEAR(
      history.get(1001.0, method, 1024).value().translation.z(), std::sqrt(2.0) * 1.5, 1.0e-8);
}

TEST(PoseTreeEdgeHistory, get_kExtrapolateSlerp) {
  const auto method = PoseTreeEdgeHistory::AccessMethod::kExtrapolateSlerp;
  PoseTreeEdgeHistory::TimedPose buffer[16];
  PoseTreeEdgeHistory history(0, 1, 16, buffer);
  history.set(1.0, Pose3d::from_translation(Vector3d(1.0, 0.0, 0.0)), 1);
  history.set(2.0, Pose3d::from_translation(Vector3d(2.0, 0.0, 0.0)), 2);
  history.set(3.0, Pose3d::from_translation(Vector3d(3.0, 0.0, 0.0)), 3);

  EXPECT_FALSE(history.get(-1.0, method, 1024));
  EXPECT_FALSE(history.get(1.5, method, 0));
  EXPECT_FALSE(history.get(1.5, method, 1));

  EXPECT_EQ(history.get(1.0, method, 1024).value().translation.x(), 1.0);
  EXPECT_EQ(history.get(1.49, method, 1024).value().translation.x(), 1.49);
  EXPECT_EQ(history.get(1.51, method, 1024).value().translation.x(), 1.51);
  EXPECT_EQ(history.get(11.51, method, 1024).value().translation.x(), 11.51);

  for (int it = 3; it < 128; it++) {
    history.set(static_cast<double>(it),
                Pose3d::from_translation(Vector3d(static_cast<double>(it), 0.0, 0.0)),
                it);
  }
  EXPECT_FALSE(history.get(111.99, method, 1024));
  EXPECT_EQ(history.get(115.51, method, 1024).value().translation.x(), 115.51);
  EXPECT_EQ(history.get(112.0, method, 1024).value().translation.x(), 112.0);

  history.set(1000.0, Pose3d::identity(), 1000);
  history.set(
      1002.0,
      Pose3d{SO3d::from_angle_axis(M_PI * 0.5, Vector3d(0.0, 1.0, 0.0)), Vector3d(3.0, 3.0, 3.0)},
      1001);
  EXPECT_NEAR(history.get(1001.0, method, 1024).value().rotation.angle(), M_PI * 0.25, 1.0e-8);
  EXPECT_NEAR(history.get(1001.0, method, 1024).value().translation.x(),
              (2.0 - std::sqrt(2.0)) * 1.5,
              1.0e-8);
  EXPECT_NEAR(history.get(1001.0, method, 1024).value().translation.y(), 1.5, 1e-8);
  EXPECT_NEAR(
      history.get(1001.0, method, 1024).value().translation.z(), std::sqrt(2.0) * 1.5, 1.0e-8);
}

TEST(PoseTreeEdgeHistory, disconnect) {
  const auto method_nn = PoseTreeEdgeHistory::AccessMethod::kNearest;
  const auto method_ex = PoseTreeEdgeHistory::AccessMethod::kExtrapolateLinearly;
  const auto method_in = PoseTreeEdgeHistory::AccessMethod::kInterpolateLinearly;
  PoseTreeEdgeHistory::TimedPose buffer[16];
  PoseTreeEdgeHistory history(0, 1, 16, buffer);
  history.set(1.0, Pose3d::from_translation(Vector3d(1.0, 0.0, 0.0)), 1);
  history.set(2.0, Pose3d::from_translation(Vector3d(2.0, 0.0, 0.0)), 2);
  EXPECT_TRUE(history.connected());
  EXPECT_FALSE(history.disconnect(1.5, 3));
  EXPECT_FALSE(history.disconnect(3.0, 1));
  EXPECT_TRUE(history.disconnect(3.0, 3));
  EXPECT_FALSE(history.connected());
  history.set(4.0, Pose3d::from_translation(Vector3d(4.0, 0.0, 0.0)), 4);

  EXPECT_EQ(history.get(1.51, method_nn, 1024).value().translation.x(), 2.0);
  EXPECT_EQ(history.get(1.51, method_in, 1024).value().translation.x(), 1.51);
  EXPECT_EQ(history.get(1.51, method_ex, 1024).value().translation.x(), 1.51);

  EXPECT_EQ(history.get(2.51, method_nn, 1024).value().translation.x(), 2.0);
  EXPECT_EQ(history.get(2.51, method_in, 1024).value().translation.x(), 2.0);
  EXPECT_EQ(history.get(2.51, method_ex, 1024).value().translation.x(), 2.51);

  EXPECT_FALSE(history.get(3.51, method_nn, 1024));
  EXPECT_FALSE(history.get(3.51, method_in, 1024));
  EXPECT_FALSE(history.get(3.51, method_ex, 1024));

  EXPECT_EQ(history.get(3.51, method_nn, 2).value().translation.x(), 2.0);
  EXPECT_EQ(history.get(3.51, method_in, 2).value().translation.x(), 2.0);
  EXPECT_EQ(history.get(3.51, method_ex, 2).value().translation.x(), 3.51);

  EXPECT_EQ(history.get(4.51, method_nn, 1024).value().translation.x(), 4.0);
  EXPECT_EQ(history.get(4.51, method_in, 1024).value().translation.x(), 4.0);
  EXPECT_FALSE(history.get(4.51, method_ex, 1024));
}

}  // namespace holoscan

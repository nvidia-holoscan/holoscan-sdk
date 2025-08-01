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
#include "holoscan/pose_tree/pose_tree.hpp"

#include <sys/time.h>

#include <gtest/gtest.h>

#include <random>
#include <string>
#include <utility>
#include <vector>

#include "holoscan/pose_tree/math/interpolation.hpp"
#include "holoscan/pose_tree/math/pose3.hpp"

static inline double get_time() {
  timeval tv;
  gettimeofday(&tv, 0);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

#define EXPECT_VEC_EQ(A, B)                 \
  {                                         \
    auto _a = (A);                          \
    auto _b = (B);                          \
    ASSERT_EQ(_a.size(), _b.size());        \
    const size_t a_length = _a.size();      \
    for (size_t i = 0; i < a_length; i++) { \
      EXPECT_EQ(_a[i], _b[i]);              \
    }                                       \
  }

#define EXPECT_VEC_NEAR(A, B, T)            \
  {                                         \
    auto _a = (A);                          \
    auto _b = (B);                          \
    ASSERT_EQ(_a.size(), _b.size());        \
    const size_t a_length = _a.size();      \
    for (size_t i = 0; i < a_length; i++) { \
      EXPECT_NEAR(_a[i], _b[i], T);         \
    }                                       \
  }

#define EXPECT_MAT_NEAR(A, B, T)            \
  {                                         \
    auto _a = (A);                          \
    auto _b = (B);                          \
    ASSERT_EQ(_a.rows(), _b.rows());        \
    ASSERT_EQ(_a.cols(), _b.cols());        \
    for (int i = 0; i < _a.rows(); i++) {   \
      for (int j = 0; j < _a.cols(); j++) { \
        EXPECT_NEAR(_a(i, j), _b(i, j), T); \
      }                                     \
    }                                       \
  }

#define ASSERT_VEC_NEAR(A, B, T)            \
  {                                         \
    auto _a = (A);                          \
    auto _b = (B);                          \
    ASSERT_EQ(_a.size(), _b.size());        \
    const size_t a_length = _a.size();      \
    for (size_t i = 0; i < a_length; i++) { \
      ASSERT_NEAR(_a[i], _b[i], T);         \
    }                                       \
  }

#define EXPECT_VEC_NEAR_ZERO(A, T)          \
  {                                         \
    auto _a = (A);                          \
    const size_t a_length = _a.size();      \
    for (size_t i = 0; i < a_length; i++) { \
      EXPECT_NEAR(_a[i], 0, T);             \
    }                                       \
  }

#define EXPECT_SO_NEAR_ID(A, T) EXPECT_NEAR((A).angle(), 0.0, T);

#define EXPECT_POSE_NEAR_ID(A, T)             \
  {                                           \
    EXPECT_SO_NEAR_ID((A).rotation, T);       \
    EXPECT_VEC_NEAR_ZERO((A).translation, T); \
  }

#define EXPECT_SO_NEAR(A, B, T) EXPECT_SO_NEAR_ID((A) * (B).inverse(), T);

#define EXPECT_POSE_NEAR(A, B, T) EXPECT_POSE_NEAR_ID((A) * (B).inverse(), T);

namespace holoscan {
namespace {
// Used to generate random poses.
std::default_random_engine s_rng;
Vector4d sigma(1.0, 1.0, 1.0, 0.4);

template <typename Derived, typename RandomEngine>
auto vector_normal_distribution(const Eigen::MatrixBase<Derived>& sigma, RandomEngine& rng) {
  std::normal_distribution<typename Derived::Scalar> normal;
  auto result = sigma.eval();
  for (int i = 0; i < result.size(); i++) {
    result[i] *= normal(rng);
  }
  return result;
}

// Picks a random point on a N-dimensional sphere (uniformly distributed)
template <typename K, int N, typename RandomEngine>
Vector<K, N> random_point_on_sphere(RandomEngine& rng, size_t n = N) {
  // TODO This theoretically has problems because we could sample all zeros many times, but most
  // likely this will happen extremely rarely.
  while (true) {
    auto result = vector_normal_distribution(Vector<K, N>::Constant(n), rng);
    K length = result.norm();
    if (is_almost_zero(length)) {
      continue;
    }
    return result / length;
  }
}

template <typename K, typename RandomEngine>
Pose3<K> pose_normal_distribution(const Vector<K, 4>& sigma, RandomEngine& rng) {
  std::normal_distribution<K> normal;
  Pose3<K> delta{
      SO3<K>::from_axis_angle(random_point_on_sphere<double, 3>(rng), sigma[3] * normal(rng)),
      vector_normal_distribution(sigma.template head<3>(), rng)};
  return delta;
}

}  // namespace

TEST(error_to_str, errors) {
  ASSERT_EQ(PoseTree::error_to_str(PoseTree::Error::kInvalidArgument),
            std::string("Invalid Argument"));
  EXPECT_EQ(PoseTree::error_to_str(PoseTree::Error::kOutOfMemory), std::string("Out of Memory"));
  EXPECT_EQ(PoseTree::error_to_str(PoseTree::Error::kFrameNotFound),
            std::string("Frame not found"));
  EXPECT_EQ(PoseTree::error_to_str(PoseTree::Error::kAlreadyExists),
            std::string("Edge already exists"));
  EXPECT_EQ(PoseTree::error_to_str(PoseTree::Error::kCyclingDependency),
            std::string("Cycling dependency"));
  EXPECT_EQ(PoseTree::error_to_str(PoseTree::Error::kFramesNotLinked),
            std::string("Frames are not linked"));
  EXPECT_EQ(PoseTree::error_to_str(PoseTree::Error::kPoseOutOfOrder),
            std::string("Pose updated out of order"));
  EXPECT_EQ(PoseTree::error_to_str(PoseTree::Error::kLogicError), std::string("Logic Error"));
  EXPECT_EQ(PoseTree::error_to_str(static_cast<PoseTree::Error>(-1)),
            std::string("Invalid Error Code"));
}

TEST(PoseTree, init) {
  EXPECT_EQ(PoseTree().init(0, 256, 1024, 32, 32, 4, 4).error(), PoseTree::Error::kInvalidArgument);
  EXPECT_EQ(PoseTree().init(16, 0, 1024, 32, 32, 4, 4).error(), PoseTree::Error::kInvalidArgument);
  EXPECT_EQ(PoseTree().init(16, 256, 0, 32, 32, 4, 4).error(), PoseTree::Error::kInvalidArgument);
  EXPECT_EQ(PoseTree().init(16, 256, 1024, 0, 32, 4, 4).error(), PoseTree::Error::kInvalidArgument);
  EXPECT_EQ(PoseTree().init(16, 256, 1024, 32, 0, 4, 4).error(), PoseTree::Error::kInvalidArgument);
  EXPECT_EQ(PoseTree().init(16, 256, 1024, 32, 32, 0, 4).error(),
            PoseTree::Error::kInvalidArgument);
  EXPECT_EQ(PoseTree().init(16, 256, 1024, 32, 32, 4, 0).error(),
            PoseTree::Error::kInvalidArgument);
  EXPECT_TRUE(PoseTree().init(16, 256, 1024, 32, 32, 4, 4));
}

TEST(PoseTree, Id) {
  for (int i = 0; i < 10; i++) {
    PoseTree pg;
    ASSERT_TRUE(pg.init(16, 256, 1024, 32, 32, 4, 4));
    const PoseTree::version_t version = pg.get_pose_tree_version();
    const PoseTree::frame_t lhs = pg.create_frame("a").value();
    const PoseTree::frame_t rhs = pg.create_frame("b").value();
    EXPECT_FALSE(pg.create_frame("a"));
    EXPECT_FALSE(pg.create_frame("b"));
    ASSERT_TRUE(pg.find_frame("a"));
    ASSERT_TRUE(pg.find_frame("b"));
    EXPECT_EQ(pg.find_frame("a").value(), lhs);
    EXPECT_EQ(pg.find_frame("b").value(), rhs);
    EXPECT_FALSE(pg.find_frame("aa"));
    EXPECT_TRUE(pg.create_frame("aa"));
    const Pose3d actual = pose_normal_distribution(sigma, s_rng);
    EXPECT_EQ(pg.get_pose_tree_version(), version);
    ASSERT_TRUE(pg.set("a", "b", 0.0, actual));
    EXPECT_GT(pg.get_pose_tree_version(), version);
    auto maybe = pg.get(rhs, rhs, 0.0);
    ASSERT_TRUE(maybe);
    EXPECT_POSE_NEAR(maybe.value(), Pose3d::identity(), 1e-12);
    maybe = pg.get(lhs, lhs, 0.0);
    ASSERT_TRUE(maybe);
    EXPECT_POSE_NEAR(maybe.value(), Pose3d::identity(), 1e-12);
  }
}

TEST(PoseTree, create_frames) {
  PoseTree pg;
  ASSERT_TRUE(pg.init(8, 256, 1024, 4, 4, 4, 4));
  ASSERT_TRUE(pg.create_frame("a"));
  ASSERT_EQ(pg.create_frame("a").error(), PoseTree::Error::kAlreadyExists);
  ASSERT_EQ(pg.create_frame("_a").error(), PoseTree::Error::kInvalidArgument);
  const std::string long_name =
      (std::string("abcdefghijklmnopqrstuvwxyz1234567890abcdefghijklmnopqrstuvwxyz01234569") +
       std::string("abcdefghijklmnopqrstuvwxyz1234567890abcdefghijklmnopqrstuvwxyz01234569") +
       std::string("abcdefghijklmnopqrstuvwxyz1234567890abcdefghijklmnopqrstuvwxyz01234569"));
  ASSERT_EQ(pg.create_frame(long_name.c_str()).error(), PoseTree::Error::kInvalidArgument);
  auto focb = pg.find_or_create_frame("b");
  ASSERT_TRUE(focb);
  ASSERT_TRUE(pg.find_or_create_frame("b").value() == focb.value());
  ASSERT_TRUE(pg.find_or_create_frame("b", 8).value() == focb.value());
  ASSERT_TRUE(pg.create_frame("c", 8));
  // Adds unnamed variable, should always work (assuming enough memory)
  ASSERT_TRUE(pg.create_frame(8));
  ASSERT_TRUE(pg.create_frame(8));
  ASSERT_TRUE(pg.create_frame());
  ASSERT_TRUE(pg.create_frame());
  // Check run out of memory
  ASSERT_EQ(pg.create_frame(256).error(), PoseTree::Error::kOutOfMemory);
  ASSERT_TRUE(pg.create_frame(8));

  ASSERT_EQ(pg.create_frame().error(), PoseTree::Error::kOutOfMemory);
}

TEST(PoseTree, create_edges) {
  PoseTree pg;
  ASSERT_TRUE(pg.init(8, 256, 1024, 4, 4, 1, 1));
  const PoseTree::frame_t a = pg.create_frame("a", 1).value();
  const PoseTree::frame_t b = pg.create_frame("b").value();
  const PoseTree::frame_t c = pg.create_frame("c").value();
  const PoseTree::frame_t d = pg.create_frame("d").value();
  const PoseTree::frame_t e = pg.create_frame("e").value();
  pg.delete_frame(e);
  ASSERT_TRUE(pg.create_edges(a, b));
  ASSERT_EQ(pg.create_edges(a, b).error(), PoseTree::Error::kAlreadyExists);
  ASSERT_EQ(pg.create_edges(b, a).error(), PoseTree::Error::kAlreadyExists);
  ASSERT_TRUE(pg.create_edges(b, c, 16));
  ASSERT_EQ(pg.create_edges(c, d, 1024).error(), PoseTree::Error::kOutOfMemory);
  ASSERT_EQ(pg.create_edges(d, e).error(), PoseTree::Error::kFrameNotFound);
  ASSERT_EQ(pg.create_edges(d, a).error(), PoseTree::Error::kOutOfMemory);
  ASSERT_TRUE(pg.create_edges(d, c, 16));
}

TEST(PoseTree, TwoNodes) {
  for (int i = 0; i < 10; i++) {
    PoseTree pg;
    ASSERT_TRUE(pg.init(16, 256, 1024, 32, 32, 4, 4));
    const PoseTree::frame_t lhs = pg.create_frame().value();
    const PoseTree::frame_t rhs = pg.create_frame().value();
    const Pose3d actual = pose_normal_distribution(sigma, s_rng);
    ASSERT_TRUE(pg.set(lhs, rhs, 0.0, actual));
    auto maybe = pg.get(lhs, rhs, 0.0);
    ASSERT_TRUE(maybe);
    EXPECT_POSE_NEAR(maybe.value(), actual, 1e-12);
    maybe = pg.get(rhs, lhs, 0.0);
    ASSERT_TRUE(maybe);
    EXPECT_POSE_NEAR(maybe.value(), actual.inverse(), 1e-12);
  }
}

TEST(PoseTree, ThreeNodes) {
  for (int i = 0; i < 10; i++) {
    PoseTree pg;
    ASSERT_TRUE(pg.init(16, 256, 1024, 32, 32, 4, 4));
    const PoseTree::frame_t a = pg.create_frame().value();
    const PoseTree::frame_t b = pg.create_frame().value();
    const PoseTree::frame_t c = pg.create_frame().value();
    const Pose3d aTb = pose_normal_distribution(sigma, s_rng);
    const Pose3d bTc = pose_normal_distribution(sigma, s_rng);
    const Pose3d aTc = aTb * bTc;
    ASSERT_TRUE(pg.set(a, b, 0.0, aTb));
    ASSERT_TRUE(pg.set(b, c, 0.0, bTc));
    auto maybe = pg.get(a, c, 0.0);
    ASSERT_TRUE(maybe);
    EXPECT_POSE_NEAR(maybe.value(), aTc, 1e-12);
    maybe = pg.get(c, a, 0.0);
    ASSERT_TRUE(maybe);
    EXPECT_POSE_NEAR(maybe.value(), aTc.inverse(), 1e-12);
  }
}

TEST(PoseTree, Tree) {
  // Create a pose graph which looks like a tree and go from top to bottom.
  constexpr int kRepetitions = 10;
  constexpr int kDepth = 10;
  constexpr int kBranching = 10;
  for (int i = 0; i < kRepetitions; i++) {
    PoseTree pg;
    ASSERT_TRUE(pg.init(1024, 1024 * 16, 1024 * 1024, 16, 16, 4, 4));
    Pose3d actual = Pose3d::identity();
    std::vector<std::vector<PoseTree::frame_t>> nodes;
    nodes.push_back({pg.create_frame().value()});
    for (int d = 1; d < kDepth; d++) {
      nodes.push_back({});
      for (size_t b = 0; b < kBranching; b++) {
        const Pose3d pose = pose_normal_distribution(sigma, s_rng);
        nodes[d].push_back(pg.create_frame().value());
        ASSERT_TRUE(pg.set(nodes[d - 1][0], nodes[d][b], 0.0, pose));
        if (b == 0) {
          actual = actual * pose;
        }
      }
    }
    auto maybe = pg.get(nodes[0][0], nodes[kDepth - 1][0], 0.0);
    ASSERT_TRUE(maybe);
    EXPECT_POSE_NEAR(maybe.value(), actual, 1e-12);
    maybe = pg.get(nodes[kDepth - 1][0], nodes[0][0], 0.0);
    ASSERT_TRUE(maybe);
    EXPECT_POSE_NEAR(maybe.value(), actual.inverse(), 1e-12);
  }
}

TEST(PoseTree, getPose2XY) {
  PoseTree pg;
  ASSERT_TRUE(pg.init(16, 256, 1024, 32, 32, 4, 4));
  const PoseTree::frame_t a = pg.create_frame("a").value();
  const PoseTree::frame_t b = pg.create_frame("b").value();
  EXPECT_TRUE(pg.set(a, b, 0.0, Pose3d::from_translation(Vector3d(1.0, 2.0, 3.0))));
  EXPECT_POSE_NEAR(pg.get_pose2_xy(a, b, 1.0).value(), Pose2d::from_translation(1.0, 2.0), 1e-12);
  EXPECT_POSE_NEAR(
      pg.get_pose2_xy("a", "b", 1.0).value(), Pose2d::from_translation(1.0, 2.0), 1e-12);
}

TEST(PoseTree, CheckCycle) {
  PoseTree pg;
  ASSERT_TRUE(pg.init(16, 256, 1024, 32, 32, 4, 4));
  const PoseTree::frame_t a = pg.create_frame().value();
  const PoseTree::frame_t b = pg.create_frame().value();
  const PoseTree::frame_t c = pg.create_frame().value();
  EXPECT_TRUE(pg.set(a, b, 0.0, Pose3d::identity()));
  EXPECT_TRUE(pg.set(b, c, 0.0, Pose3d::identity()));
  EXPECT_FALSE(pg.set(a, c, 0.0, Pose3d::identity()));
}

TEST(PoseTree, CheckInterpolation) {
  PoseTree pg;
  ASSERT_TRUE(pg.init(16, 256, 1024, 32, 32, 4, 4));
  const PoseTree::frame_t a = pg.create_frame("a").value();
  const PoseTree::frame_t b = pg.create_frame("b").value();
  EXPECT_TRUE(pg.set(a, b, 0.0, Pose3d::from_translation(Vector3d(1.0, 0.0, 0.0))));
  EXPECT_TRUE(pg.set(a, b, 1.0, Pose3d::from_translation(Vector3d(3.0, 0.0, 0.0))));
  EXPECT_POSE_NEAR(
      pg.get(a, b, 0.5).value(), Pose3d::from_translation(Vector3d(2.0, 0.0, 0.0)), 1e-12);
  EXPECT_POSE_NEAR(
      pg.get("a", "b", 0.5).value(), Pose3d::from_translation(Vector3d(2.0, 0.0, 0.0)), 1e-12);
  EXPECT_POSE_NEAR(
      pg.get(a, b, 0.5, PoseTreeEdgeHistory::AccessMethod::kInterpolateLinearly).value(),
      Pose3d::from_translation(Vector3d(2.0, 0.0, 0.0)),
      1e-12);
  EXPECT_POSE_NEAR(
      pg.get("a", "b", 0.5, PoseTreeEdgeHistory::AccessMethod::kInterpolateLinearly).value(),
      Pose3d::from_translation(Vector3d(2.0, 0.0, 0.0)),
      1e-12);
  EXPECT_POSE_NEAR(
      pg.get(a, b, 1.5, PoseTreeEdgeHistory::AccessMethod::kInterpolateLinearly).value(),
      Pose3d::from_translation(Vector3d(3.0, 0.0, 0.0)),
      1e-12);
  EXPECT_POSE_NEAR(
      pg.get("a", "b", 1.5, PoseTreeEdgeHistory::AccessMethod::kInterpolateLinearly).value(),
      Pose3d::from_translation(Vector3d(3.0, 0.0, 0.0)),
      1e-12);
  EXPECT_POSE_NEAR(
      pg.get(a, b, 0.5, PoseTreeEdgeHistory::AccessMethod::kExtrapolateLinearly).value(),
      Pose3d::from_translation(Vector3d(2.0, 0.0, 0.0)),
      1e-12);
  EXPECT_POSE_NEAR(
      pg.get("a", "b", 0.5, PoseTreeEdgeHistory::AccessMethod::kExtrapolateLinearly).value(),
      Pose3d::from_translation(Vector3d(2.0, 0.0, 0.0)),
      1e-12);
  EXPECT_POSE_NEAR(
      pg.get(a, b, 1.5, PoseTreeEdgeHistory::AccessMethod::kExtrapolateLinearly).value(),
      Pose3d::from_translation(Vector3d(4.0, 0.0, 0.0)),
      1e-12);
  EXPECT_POSE_NEAR(
      pg.get("a", "b", 1.5, PoseTreeEdgeHistory::AccessMethod::kExtrapolateLinearly).value(),
      Pose3d::from_translation(Vector3d(4.0, 0.0, 0.0)),
      1e-12);
  EXPECT_POSE_NEAR(pg.get(a, b, 0.4, PoseTreeEdgeHistory::AccessMethod::kNearest).value(),
                   Pose3d::from_translation(Vector3d(1.0, 0.0, 0.0)),
                   1e-12);
  EXPECT_POSE_NEAR(pg.get("a", "b", 0.4, PoseTreeEdgeHistory::AccessMethod::kNearest).value(),
                   Pose3d::from_translation(Vector3d(1.0, 0.0, 0.0)),
                   1e-12);
  EXPECT_POSE_NEAR(pg.get(a, b, 0.6, PoseTreeEdgeHistory::AccessMethod::kNearest).value(),
                   Pose3d::from_translation(Vector3d(3.0, 0.0, 0.0)),
                   1e-12);
  EXPECT_POSE_NEAR(pg.get("a", "b", 0.6, PoseTreeEdgeHistory::AccessMethod::kNearest).value(),
                   Pose3d::from_translation(Vector3d(3.0, 0.0, 0.0)),
                   1e-12);
}

TEST(PoseTree, CheckVersioning) {
  PoseTree pg;
  ASSERT_TRUE(pg.init(16, 256, 1024, 32, 32, 4, 4));
  const PoseTree::frame_t a = pg.create_frame().value();
  const PoseTree::frame_t b = pg.create_frame().value();
  const auto v1 = pg.get_pose_tree_version();
  EXPECT_TRUE(pg.set(a, b, 0.0, Pose3d::from_translation(Vector3d(1.0, 0.0, 0.0))));
  const auto v2 = pg.get_pose_tree_version();
  EXPECT_TRUE(pg.set(a, b, 1.0, Pose3d::from_translation(Vector3d(3.0, 0.0, 0.0))));
  EXPECT_FALSE(pg.get(a, b, 0.0, v1));
  EXPECT_POSE_NEAR(
      pg.get(a, b, 1.0).value(), Pose3d::from_translation(Vector3d(3.0, 0.0, 0.0)), 1e-12);
  EXPECT_POSE_NEAR(
      pg.get(a, b, 1.0, v2).value(), Pose3d::from_translation(Vector3d(1.0, 0.0, 0.0)), 1e-12);
}

TEST(PoseTree, CheckOrderedUpdates) {
  PoseTree pg;
  ASSERT_TRUE(pg.init(16, 256, 1024, 32, 32, 4, 4));
  const PoseTree::frame_t a = pg.create_frame().value();
  const PoseTree::frame_t b = pg.create_frame().value();
  EXPECT_TRUE(pg.set(a, b, 0.0, Pose3d::from_translation(Vector3d(1.0, 0.0, 0.0))));
  EXPECT_TRUE(pg.set(a, b, 1.0, Pose3d::from_translation(Vector3d(3.0, 0.0, 0.0))));
  EXPECT_FALSE(pg.set(a, b, 1.0, Pose3d::from_translation(Vector3d(2.0, 0.0, 0.0))));
  EXPECT_POSE_NEAR(
      pg.get(a, b, 1.0).value(), Pose3d::from_translation(Vector3d(3.0, 0.0, 0.0)), 1e-12);
  EXPECT_POSE_NEAR(
      pg.get(b, a, 1.0).value(), Pose3d::from_translation(Vector3d(-3.0, 0.0, 0.0)), 1e-12);
  EXPECT_TRUE(pg.set(a, b, 2.0, Pose3d::from_translation(Vector3d(4.0, 0.0, 0.0))));
  EXPECT_FALSE(pg.set(a, b, 1.5, Pose3d::identity()));
}

TEST(PoseTree, TemporalDirect) {
  PoseTree pg;
  ASSERT_TRUE(pg.init(16, 256, 1024, 32, 32, 4, 4));
  const PoseTree::frame_t a = pg.create_frame().value();
  const PoseTree::frame_t b = pg.create_frame().value();
  const Pose3d aTb0 = Pose3d::from_translation({0.7, -1.3, 2.0});
  const Pose3d aTb1 = Pose3d::from_translation({-0.7, -1.6, 2.4});
  EXPECT_TRUE(pg.set(a, b, 0.0, aTb0));
  EXPECT_TRUE(pg.set(a, b, 1.0, aTb1));
  EXPECT_POSE_NEAR(pg.get(a, b, 0.0).value(), aTb0, 1e-12);
  EXPECT_POSE_NEAR(pg.get(a, b, 1.0).value(), aTb1, 1e-12);
  EXPECT_POSE_NEAR(pg.get(a, b, 12.6).value(), aTb1, 1e-12);
  EXPECT_POSE_NEAR(pg.get(a, b, 0.5).value(), Pose3d::from_translation({0.0, -1.45, 2.2}), 1e-12);
  EXPECT_POSE_NEAR(pg.get(a, b, 0.3).value(), interpolate(0.3, aTb0, aTb1), 1e-12);
  EXPECT_POSE_NEAR(pg.get(a, b, 0.6).value(), interpolate(0.6, aTb0, aTb1), 1e-12);
  EXPECT_POSE_NEAR(pg.get(a, b, 0.9).value(), interpolate(0.9, aTb0, aTb1), 1e-12);
}

TEST(PoseTree, TemporalIndirect) {
  PoseTree pg;
  ASSERT_TRUE(pg.init(16, 256, 1024, 32, 32, 4, 4));
  const PoseTree::frame_t a = pg.create_frame().value();
  const PoseTree::frame_t b = pg.create_frame().value();
  const PoseTree::frame_t c = pg.create_frame().value();
  const Pose3d aTb0 = pose_normal_distribution(sigma, s_rng);
  const Pose3d aTb1 = pose_normal_distribution(sigma, s_rng);
  const Pose3d bTc0 = pose_normal_distribution(sigma, s_rng);
  const Pose3d bTc1 = pose_normal_distribution(sigma, s_rng);
  EXPECT_TRUE(pg.set(a, b, 0.0, aTb0));
  EXPECT_TRUE(pg.set(a, b, 1.0, aTb1));
  EXPECT_TRUE(pg.set(b, c, 0.0, bTc0));
  EXPECT_TRUE(pg.set(b, c, 1.0, bTc1));
  EXPECT_POSE_NEAR(pg.get(a, b, 0.0).value(), aTb0, 1e-12);
  EXPECT_POSE_NEAR(pg.get(a, b, 1.0).value(), aTb1, 1e-12);
  EXPECT_POSE_NEAR(pg.get(a, b, 12.6).value(), aTb1, 1e-12);
  EXPECT_POSE_NEAR(pg.get(b, c, 0.0).value(), bTc0, 1e-12);
  EXPECT_POSE_NEAR(pg.get(b, c, 1.0).value(), bTc1, 1e-12);
  EXPECT_POSE_NEAR(pg.get(b, c, 12.6).value(), bTc1, 1e-12);
  EXPECT_POSE_NEAR(pg.get(a, b, 0.3).value(), interpolate(0.3, aTb0, aTb1), 1e-12);
  EXPECT_POSE_NEAR(pg.get(a, b, 0.5).value(), interpolate(0.5, aTb0, aTb1), 1e-12);
  EXPECT_POSE_NEAR(pg.get(a, b, 0.9).value(), interpolate(0.9, aTb0, aTb1), 1e-12);
  EXPECT_POSE_NEAR(pg.get(b, c, 0.3).value(), interpolate(0.3, bTc0, bTc1), 1e-12);
  EXPECT_POSE_NEAR(pg.get(b, c, 0.5).value(), interpolate(0.5, bTc0, bTc1), 1e-12);
  EXPECT_POSE_NEAR(pg.get(b, c, 0.9).value(), interpolate(0.9, bTc0, bTc1), 1e-12);
  EXPECT_POSE_NEAR(pg.get(a, c, 0.3).value(),
                   interpolate(0.3, aTb0, aTb1) * interpolate(0.3, bTc0, bTc1),
                   1e-12);
  EXPECT_POSE_NEAR(pg.get(a, c, 0.5).value(),
                   interpolate(0.5, aTb0, aTb1) * interpolate(0.5, bTc0, bTc1),
                   1e-12);
  EXPECT_POSE_NEAR(pg.get(a, c, 0.9).value(),
                   interpolate(0.9, aTb0, aTb1) * interpolate(0.9, bTc0, bTc1),
                   1e-12);
}

TEST(PoseTree, get_latest) {
  const char* a = "a";
  const char* b = "b";
  const char* c = "c";
  PoseTree pg;
  ASSERT_TRUE(pg.init(16, 256, 1024, 32, 32, 4, 4));
  ASSERT_TRUE(pg.create_frame(a));
  ASSERT_TRUE(pg.create_frame(b));
  ASSERT_TRUE(pg.create_frame(c));
  const Pose3d aTb0 = pose_normal_distribution(sigma, s_rng);
  const Pose3d aTb1 = pose_normal_distribution(sigma, s_rng);
  const Pose3d bTc0 = pose_normal_distribution(sigma, s_rng);
  const Pose3d bTc1 = pose_normal_distribution(sigma, s_rng);
  pg.set(a, b, 0.0, aTb0);
  pg.set(a, b, 2.0, aTb1);
  pg.set(b, c, 0.0, bTc0);
  pg.set(b, c, 2.5, bTc1);

  EXPECT_NEAR(pg.get_latest(a, b)->second, 2.0, 1e-12);
  EXPECT_POSE_NEAR(pg.get_latest(a, b)->first, aTb1, 1e-12);
  EXPECT_POSE_NEAR(pg.get_latest(b, a)->first, aTb1.inverse(), 1e-12);

  EXPECT_NEAR(pg.get_latest(b, c)->second, 2.5, 1e-12);
  EXPECT_POSE_NEAR(pg.get_latest(b, c)->first, bTc1, 1e-12);
  EXPECT_POSE_NEAR(pg.get_latest(c, b)->first, bTc1.inverse(), 1e-12);

  EXPECT_FALSE(pg.get_latest(a, c));
  EXPECT_FALSE(pg.get_latest(a, a));
  EXPECT_FALSE(pg.get_latest(a, "e"));
  EXPECT_FALSE(pg.get_latest("e", b));
  for (int i = 10; i <= 2048; i++) {
    ASSERT_TRUE(pg.set(a, b, static_cast<double>(i), pose_normal_distribution(sigma, s_rng)));
    ASSERT_NEAR(pg.get_latest(a, b)->second, static_cast<double>(i), 1e-12);
  }
}

TEST(PoseTree, get_latest_edge) {
  const char* a = "a";
  const char* b = "b";
  const char* c = "c";
  PoseTree pg;
  ASSERT_TRUE(pg.init(16, 256, 1024, 32, 32, 4, 4));
  ASSERT_TRUE(pg.create_frame(a));
  ASSERT_TRUE(pg.create_frame(b));
  ASSERT_TRUE(pg.create_frame(c));
  const Pose3d aTb0 = pose_normal_distribution(sigma, s_rng);
  const Pose3d aTb1 = pose_normal_distribution(sigma, s_rng);
  const Pose3d bTc0 = pose_normal_distribution(sigma, s_rng);
  const Pose3d bTc1 = pose_normal_distribution(sigma, s_rng);
  pg.set(a, b, 0.0, aTb0);
  pg.set(a, b, 2.5, aTb1);
  pg.set(b, c, 0.0, bTc0);
  auto version = pg.get_pose_tree_version();
  pg.set(b, c, 2.0, bTc1);

  EXPECT_POSE_NEAR(pg.get(a, b).value(), aTb1, 1e-12);
  EXPECT_POSE_NEAR(pg.get(b, a).value(), aTb1.inverse(), 1e-12);

  EXPECT_POSE_NEAR(pg.get(b, c).value(), bTc1, 1e-12);
  EXPECT_POSE_NEAR(pg.get(c, b).value(), bTc1.inverse(), 1e-12);

  EXPECT_POSE_NEAR(pg.get(a, c).value(), aTb1 * bTc1, 1e-12);
  EXPECT_POSE_NEAR(pg.get(a, c, version).value(), aTb1 * bTc0, 1e-12);
  EXPECT_POSE_NEAR(pg.get(a, a).value(), Pose3d::identity(), 1e-12);
  EXPECT_FALSE(pg.get(a, "e"));
  EXPECT_FALSE(pg.get("e", b));
}

TEST(PoseTree, disconnect_edge) {
  {
    PoseTree pg;
    ASSERT_TRUE(pg.init(16, 256, 1024, 32, 32, 4, 4));
    const PoseTree::frame_t a = pg.create_frame().value();
    const PoseTree::frame_t b = pg.create_frame().value();
    const PoseTree::frame_t c = pg.create_frame().value();
    EXPECT_TRUE(pg.set(a, b, 0.0, Pose3d::identity()));
    EXPECT_TRUE(pg.set(b, c, 0.0, Pose3d::identity()));
    EXPECT_FALSE(pg.set(a, c, 0.0, Pose3d::identity()));
    EXPECT_FALSE(pg.disconnect_edge(a, b, 0.0));
    EXPECT_TRUE(pg.disconnect_edge(a, b, 1.0));
    EXPECT_FALSE(pg.disconnect_edge(a, b, 1.0));
    EXPECT_TRUE(pg.get(a, c, 0.0));
    EXPECT_TRUE(pg.get(c, a, 0.0));
    EXPECT_FALSE(pg.get(a, c, 1.0));
    EXPECT_FALSE(pg.get(c, a, 1.0));
    EXPECT_FALSE(pg.set(a, c, 0.99, Pose3d::identity()));
    EXPECT_TRUE(pg.set(a, c, 1.0, Pose3d::identity()));
  }
  {
    PoseTree pg;
    ASSERT_TRUE(pg.init(16, 256, 1024, 32, 32, 4, 4));
    const PoseTree::frame_t a = pg.create_frame().value();
    const PoseTree::frame_t b = pg.create_frame().value();
    const PoseTree::frame_t c = pg.create_frame().value();
    EXPECT_TRUE(pg.set(a, b, 0.0, Pose3d::identity()));
    EXPECT_TRUE(pg.set(b, c, 0.0, Pose3d::identity()));
    EXPECT_FALSE(pg.disconnect_edge(a, c, 1.0));
    EXPECT_FALSE(pg.set(a, c, 0.0, Pose3d::identity()));
    EXPECT_TRUE(pg.disconnect_edge(c, b, 1.0));
    EXPECT_TRUE(pg.get(a, c, 0.0));
    EXPECT_TRUE(pg.get(c, a, 0.0));
    EXPECT_FALSE(pg.get(a, c, 1.0));
    EXPECT_FALSE(pg.get(c, a, 1.0));
    EXPECT_FALSE(pg.set(a, c, 0.99, Pose3d::identity()));
    EXPECT_TRUE(pg.set(a, c, 1.0, Pose3d::identity()));
  }
}

TEST(PoseTree, disconnect_frame) {
  PoseTree pg;
  ASSERT_TRUE(pg.init(16, 256, 1024, 32, 32, 4, 4));
  const PoseTree::frame_t a = pg.create_frame().value();
  const PoseTree::frame_t b = pg.create_frame().value();
  const PoseTree::frame_t c = pg.create_frame().value();
  EXPECT_TRUE(pg.set(a, b, 0.0, Pose3d::identity()));
  EXPECT_TRUE(pg.set(b, c, 0.0, Pose3d::identity()));
  EXPECT_FALSE(pg.set(a, c, 0.0, Pose3d::identity()));
  EXPECT_FALSE(pg.disconnect_frame(b, 0.0));
  EXPECT_TRUE(pg.disconnect_frame(b, 1.0));
  EXPECT_TRUE(pg.get(a, c, 0.0));
  EXPECT_FALSE(pg.get(a, c, 1.0));
  EXPECT_TRUE(pg.get(c, a, 0.0));
  EXPECT_FALSE(pg.get(c, a, 1.0));

  EXPECT_TRUE(pg.get(a, b, 0.0));
  EXPECT_FALSE(pg.get(a, b, 1.0));
  EXPECT_TRUE(pg.get(b, a, 0.0));
  EXPECT_FALSE(pg.get(b, a, 1.0));
  EXPECT_TRUE(pg.set(a, c, 1.01, Pose3d::identity()));
  EXPECT_TRUE(pg.set(a, b, 1.01, Pose3d::identity()));
}

TEST(PoseTree, delete_frame) {
  PoseTree pg;
  ASSERT_TRUE(pg.init(16, 256, 1024, 32, 32, 4, 4));
  const PoseTree::frame_t a = pg.create_frame().value();
  const PoseTree::frame_t b = pg.create_frame("b").value();
  const PoseTree::frame_t c = pg.create_frame().value();
  EXPECT_TRUE(pg.set(a, b, 1.0, Pose3d::identity()));
  EXPECT_TRUE(pg.set(b, c, 1.0, Pose3d::identity()));
  EXPECT_FALSE(pg.set(a, c, 1.0, Pose3d::identity()));
  EXPECT_TRUE(pg.get(a, c, 1.0));
  EXPECT_TRUE(pg.delete_frame(b));
  EXPECT_FALSE(pg.find_frame("b"));
  EXPECT_EQ(pg.set(a, b, 1.0, Pose3d::identity()).error(), PoseTree::Error::kFrameNotFound);
  EXPECT_FALSE(pg.get(a, c, 1.0));
  EXPECT_TRUE(pg.set(a, c, 1.0, Pose3d::identity()));
}

TEST(PoseTree, delete_edge) {
  PoseTree pg;
  ASSERT_TRUE(pg.init(16, 256, 1024, 32, 32, 4, 4));
  const PoseTree::frame_t a = pg.create_frame().value();
  const PoseTree::frame_t b = pg.create_frame().value();
  const PoseTree::frame_t c = pg.create_frame().value();
  EXPECT_TRUE(pg.set(a, b, 1.0, Pose3d::identity()));
  EXPECT_TRUE(pg.set(b, c, 1.0, Pose3d::identity()));
  // For now a and c are connected and would create a loop, after we delete a-b
  // a-c are not connected and we can connect them
  EXPECT_FALSE(pg.set(a, c, 1.0, Pose3d::identity()));
  EXPECT_TRUE(pg.get(a, c, 1.0));
  EXPECT_TRUE(pg.delete_edge(a, b));
  EXPECT_TRUE(pg.get(b, c, 1.0));
  EXPECT_FALSE(pg.get(a, b, 1.0));
  EXPECT_FALSE(pg.get(a, c, 1.0));
  EXPECT_TRUE(pg.set(a, c, 1.0, Pose3d::identity()));
  // Deleting an edge reset the history, we can update in the past.
  EXPECT_FALSE(pg.set(b, c, 0.0, Pose3d::identity()));
  EXPECT_TRUE(pg.delete_edge(c, b));
  EXPECT_TRUE(pg.set(b, c, 0.0, Pose3d::identity()));
}

TEST(PoseTree, disconnect_edge_and_queries) {
  PoseTree pg;
  ASSERT_TRUE(pg.init(16, 256, 1024, 32, 32, 4, 4));
  const PoseTree::frame_t a = pg.create_frame().value();
  const PoseTree::frame_t b = pg.create_frame().value();
  const PoseTree::frame_t c = pg.create_frame().value();
  const PoseTree::frame_t d = pg.create_frame().value();
  EXPECT_TRUE(pg.set(a, b, 0.0, Pose3d::from_translation(Vector3d(1.0, 0.0, 0.0))));
  EXPECT_TRUE(pg.set(b, c, 0.0, Pose3d::from_translation(Vector3d(0.0, 1.0, 0.0))));
  EXPECT_TRUE(pg.set(c, d, 0.0, Pose3d::from_translation(Vector3d(0.0, 0.0, 1.0))));
  EXPECT_POSE_NEAR(
      pg.get(a, d, 1.0).value(), Pose3d::from_translation(Vector3d(1.0, 1.0, 1.0)), 1e-12);
  EXPECT_TRUE(pg.disconnect_edge(b, c, 1.0));
  EXPECT_FALSE(pg.get(a, d, 1.0));
  EXPECT_TRUE(pg.set(b, c, 2.0, Pose3d::from_translation(Vector3d(0.0, 2.0, 0.0))));
  EXPECT_TRUE(pg.disconnect_edge(b, c, 3.0));
  EXPECT_TRUE(pg.set(b, c, 4.0, Pose3d::from_translation(Vector3d(0.0, 4.0, 0.0))));
  EXPECT_TRUE(pg.set(b, c, 5.0, Pose3d::from_translation(Vector3d(0.0, 5.0, 0.0))));
  EXPECT_TRUE(pg.disconnect_edge(b, c, 6.0));
  EXPECT_TRUE(pg.set(b, c, 7.0, Pose3d::from_translation(Vector3d(0.0, 7.0, 0.0))));
  EXPECT_TRUE(pg.disconnect_edge(b, c, 8.0));
  EXPECT_TRUE(pg.set(b, c, 8.01, Pose3d::from_translation(Vector3d(0.0, 8.0, 0.0))));

  EXPECT_POSE_NEAR(
      pg.get(a, d, 2.5).value(), Pose3d::from_translation(Vector3d(1.0, 2.0, 1.0)), 1e-12);
  EXPECT_FALSE(pg.get(a, d, 3.5));
  EXPECT_POSE_NEAR(
      pg.get(a, d, 4.5).value(), Pose3d::from_translation(Vector3d(1.0, 4.5, 1.0)), 1e-12);
  EXPECT_POSE_NEAR(
      pg.get(a, d, 5.5).value(), Pose3d::from_translation(Vector3d(1.0, 5.0, 1.0)), 1e-12);
  EXPECT_FALSE(pg.get(a, d, 6.5));
  EXPECT_POSE_NEAR(
      pg.get(a, d, 7.99).value(), Pose3d::from_translation(Vector3d(1.0, 7.0, 1.0)), 1e-12);
  EXPECT_POSE_NEAR(
      pg.get(a, d, 8.01).value(), Pose3d::from_translation(Vector3d(1.0, 8.0, 1.0)), 1e-12);
}

TEST(PoseTree, AutoGeneratedName) {
  PoseTree pg;
  ASSERT_TRUE(pg.init(16, 256, 1024, 32, 32, 4, 4));
  const auto maybe_frame = pg.create_frame();
  ASSERT_TRUE(maybe_frame);
  const auto maybe_name = pg.get_frame_name(maybe_frame.value());
  ASSERT_TRUE(maybe_name);
  const std::string expected_name = "_frame_" + std::to_string(maybe_frame.value());
  EXPECT_EQ(maybe_name.value(), expected_name);

  // Disallowed name
  EXPECT_FALSE(pg.find_or_create_frame("_lorem"));
}

TEST(PoseTree, DefaultAccess) {
  PoseTree pg;
  ASSERT_TRUE(pg.init(16, 256, 1024, 32, 32, 4, 4));
  const PoseTree::frame_t a = pg.create_frame().value();
  const PoseTree::frame_t b = pg.create_frame().value();
  const PoseTree::frame_t c = pg.create_frame().value();
  ASSERT_TRUE(pg.create_edges(a, b, 16, PoseTreeEdgeHistory::AccessMethod::kNearest));
  ASSERT_TRUE(pg.create_edges(b, c, PoseTreeEdgeHistory::AccessMethod::kPrevious));
  EXPECT_TRUE(pg.set(a, b, 0.0, Pose3d::from_translation(Vector3d(1.0, 0.0, 0.0))));
  EXPECT_TRUE(pg.set(a, b, 1.0, Pose3d::from_translation(Vector3d(2.0, 0.0, 0.0))));
  EXPECT_TRUE(pg.set(b, c, 0.0, Pose3d::from_translation(Vector3d(1.0, 0.0, 0.0))));
  EXPECT_TRUE(pg.set(b, c, 1.0, Pose3d::from_translation(Vector3d(2.0, 0.0, 0.0))));
  EXPECT_POSE_NEAR(
      pg.get(a, b, 0.6).value(), Pose3d::from_translation(Vector3d(2.0, 0.0, 0.0)), 1e-12);
  EXPECT_POSE_NEAR(
      pg.get(b, c, 0.6).value(), Pose3d::from_translation(Vector3d(1.0, 0.0, 0.0)), 1e-12);
}

TEST(PoseTree, get_content) {
  PoseTree pg;
  ASSERT_TRUE(pg.init(16, 256, 1024, 32, 32, 4, 4));
  const PoseTree::frame_t a = pg.create_frame().value();
  const PoseTree::frame_t b = pg.create_frame().value();
  const PoseTree::frame_t c = pg.create_frame().value();
  ASSERT_TRUE(pg.create_edges(a, b, 16, PoseTreeEdgeHistory::AccessMethod::kNearest));
  ASSERT_TRUE(pg.create_edges(b, c, PoseTreeEdgeHistory::AccessMethod::kPrevious));
  EXPECT_TRUE(pg.set(a, b, 1.0, Pose3d::from_translation(Vector3d(2.0, 0.0, 0.0))));
  EXPECT_TRUE(pg.set(b, c, 0.0, Pose3d::from_translation(Vector3d(1.0, 0.0, 0.0))));

  std::vector<PoseTree::frame_t> frames;
  std::vector<std::string_view> frame_names;
  std::vector<std::pair<PoseTree::frame_t, PoseTree::frame_t>> edges;
  std::vector<std::pair<std::string_view, std::string_view>> edge_names;

  ASSERT_FALSE(pg.get_frame_uids(frames));
  ASSERT_FALSE(pg.get_frame_names(frame_names));
  ASSERT_FALSE(pg.get_edge_uids(edges));
  ASSERT_FALSE(pg.get_edge_names(edge_names));

  frames.resize(3);
  frame_names.resize(3);
  edges.resize(2);
  edge_names.resize(2);

  ASSERT_TRUE(pg.get_frame_uids(frames));
  ASSERT_TRUE(pg.get_frame_names(frame_names));
  ASSERT_TRUE(pg.get_edge_uids(edges));
  ASSERT_TRUE(pg.get_edge_names(edge_names));

  EXPECT_EQ(frames.size(), 3);
  EXPECT_EQ(frame_names.size(), 3);
  EXPECT_EQ(edges.size(), 2);
  EXPECT_EQ(edge_names.size(), 2);
}

}  // namespace holoscan

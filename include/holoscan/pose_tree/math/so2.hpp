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
#ifndef HOLOSCAN_POSE_TREE_MATH_SO2_HPP
#define HOLOSCAN_POSE_TREE_MATH_SO2_HPP

#include <cmath>

#include "holoscan/pose_tree/math/types.hpp"

namespace holoscan {

/**
 * @brief Class representing 2D rotations using the SO(2) group.
 *
 * This class represents rotations in 2D space as elements of the special orthogonal group SO(2).
 * Internally, rotations are stored as normalized direction vectors (cos, sin) to avoid
 * trigonometric function calls during composition and transformation operations.
 *
 * @tparam K Scalar type (typically float or double).
 */
template <typename K>
class SO2 {
 public:
  using Scalar = K;
  static constexpr int kDimension = 2;

  /**
   * @brief Default constructor creates an undefined rotation.
   *
   * @note Use identity() to create the identity rotation.
   */
  SO2() {}

  /**
   * @brief Create the identity rotation.
   *
   * @return Identity rotation (no rotation).
   */
  static SO2 identity() {
    SO2 q;
    q.cos_sin_ = Vector2<K>(K(1), K(0));
    return q;
  }

  /**
   * @brief Create rotation from an angle.
   *
   * @note This uses calls to trigonometric functions.
   * @param angle Rotation angle in radians.
   * @return Rotation representing the given angle.
   */
  static SO2 from_angle(K angle) {
    SO2 q;
    q.cos_sin_ = Vector2<K>(std::cos(angle), std::sin(angle));
    return q;
  }

  /**
   * @brief Create rotation from a not necessarily normalized direction vector.
   *
   * @param direction Direction vector (will be normalized internally).
   * @return Rotation aligned with the given direction.
   */
  static SO2 from_direction(const Vector2<K>& direction) {
    SO2 q;
    const K norm = direction.norm();
    q.cos_sin_ = direction / norm;
    return q;
  }

  /**
   * @brief Create rotation from direction components.
   *
   * @param dx X component of direction vector.
   * @param dy Y component of direction vector.
   * @return Rotation aligned with the given direction.
   */
  static SO2 from_direction(K dx, K dy) { return from_direction(Vector2<K>(dx, dy)); }

  /**
   * @brief Create rotation from a normalized cos/sin direction vector.
   *
   * @param cos_sin Normalized direction vector (cos, sin).
   * @return Rotation represented by the given cos/sin values.
   */
  static SO2 from_normalized(const Vector2<K>& cos_sin) {
    SO2 q;
    q.cos_sin_ = cos_sin;
    return q;
  }

  /**
   * @brief Create rotation from normalized cos and sin values.
   *
   * @param cos_angle Cosine of the rotation angle.
   * @param sin_angle Sine of the rotation angle.
   * @return Rotation represented by the given cos/sin values.
   */
  static SO2 from_normalized(K cos_angle, K sin_angle) {
    return from_normalized(Vector2<K>(cos_angle, sin_angle));
  }

  /**
   * @brief Get the cosine of the rotation angle.
   *
   * @note This is a simple getter and does not call trigonometric functions.
   * @return Cosine of the rotation angle.
   */
  K cos() const { return cos_sin_[0]; }

  /**
   * @brief Get the sine of the rotation angle.
   *
   * @note This is a simple getter and does not call trigonometric functions.
   * @return Sine of the rotation angle.
   */
  K sin() const { return cos_sin_[1]; }

  /**
   * @brief Get the cos and sin of the rotation angle as a direction vector.
   *
   * @return Direction vector (cos, sin).
   */
  const Vector2<K>& as_direction() const { return cos_sin_; }

  /**
   * @brief Get the rotation angle in range [-π, π].
   *
   * @note This uses a call to a trigonometric function.
   * @return Rotation angle in radians.
   */
  K angle() const { return std::atan2(sin(), cos()); }

  /**
   * @brief Get the 2×2 rotation matrix representation.
   *
   * @return 2×2 rotation matrix.
   */
  Matrix2<K> matrix() const {
    Matrix2<K> m;
    m(0, 0) = cos();
    m(0, 1) = -sin();
    m(1, 0) = sin();
    m(1, 1) = cos();
    return m;
  }

  /**
   * @brief Get the inverse rotation.
   *
   * @return Inverse rotation.
   */
  SO2 inverse() const { return SO2::from_normalized(cos(), -sin()); }

  /**
   * @brief Cast to a different scalar type.
   *
   * @tparam S Target scalar type.
   * @return Rotation cast to the target type.
   */
  template <typename S, typename std::enable_if_t<!std::is_same<S, K>::value, int> = 0>
  SO2<S> cast() const {
    // We need to re-normalize in the new type
    return SO2<S>::from_direction(as_direction().template cast<S>());
  }

  /**
   * @brief Cast to the same scalar type (no-op).
   *
   * @tparam S Target scalar type (same as K).
   * @return Reference to this rotation.
   */
  template <typename S, typename std::enable_if_t<std::is_same<S, K>::value, int> = 0>
  const SO2& cast() const {
    // Nothing to do as the type does not change
    return *this;
  }

  /**
   * @brief Compose two rotations.
   *
   * @param lhs Left rotation.
   * @param rhs Right rotation.
   * @return Composed rotation lhs * rhs.
   */
  friend SO2 operator*(const SO2& lhs, const SO2& rhs) {
    return from_direction(lhs.cos() * rhs.cos() - lhs.sin() * rhs.sin(),
                          lhs.sin() * rhs.cos() + lhs.cos() * rhs.sin());
  }

  /**
   * @brief Rotate a 2D vector.
   *
   * @param lhs Rotation to apply.
   * @param vec Vector to rotate.
   * @return Rotated vector.
   */
  friend Vector2<K> operator*(const SO2& lhs, const Vector2<K>& vec) {
    return Vector2<K>(lhs.cos() * vec[0] - lhs.sin() * vec[1],
                      lhs.sin() * vec[0] + lhs.cos() * vec[1]);
  }

 private:
  /// Internal representation as (cos, sin) direction vector.
  Vector2<K> cos_sin_;
};

/// SO2 with double precision.
using SO2d = SO2<double>;
/// SO2 with single precision.
using SO2f = SO2<float>;

}  // namespace holoscan

#endif /* HOLOSCAN_POSE_TREE_MATH_SO2_HPP */

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
#ifndef HOLOSCAN_POSE_TREE_MATH_POSE2_HPP
#define HOLOSCAN_POSE_TREE_MATH_POSE2_HPP

#include <cmath>

#include "holoscan/pose_tree/math/so2.hpp"
#include "holoscan/pose_tree/math/types.hpp"

namespace holoscan {

/**
 * @brief Class representing 2D transformations (rigid body motion in 2D).
 *
 * This class represents elements of the SE(2) group, which combines 2D rotations and
 * translations. Each pose consists of a rotation component (SO2) and a translation
 * vector in 2D space.
 *
 * @tparam K Scalar type (typically float or double).
 */
template <typename K>
struct Pose2 {
  using Scalar = K;
  static constexpr int kDimension = 2;

  /// Rotation component.
  SO2<K> rotation;
  /// Translation component.
  Vector2<K> translation;

  /**
   * @brief Create the identity transformation.
   *
   * @return Identity transformation (no rotation or translation).
   */
  static Pose2 identity() { return Pose2{SO2<K>::identity(), Vector2<K>::Zero()}; }

  /**
   * @brief Create a pure translation transformation.
   *
   * @param translation Translation vector.
   * @return Translation-only transformation.
   */
  static Pose2 from_translation(const Vector2<K>& translation) {
    return Pose2{SO2<K>::identity(), translation};
  }

  /**
   * @brief Create a pure translation transformation from components.
   *
   * @param x X component of translation.
   * @param y Y component of translation.
   * @return Translation-only transformation.
   */
  static Pose2 from_translation(K x, K y) { return Pose2{SO2<K>::identity(), Vector2<K>{x, y}}; }

  /**
   * @brief Create a pure rotation transformation.
   *
   * @param angle Rotation angle in radians.
   * @return Rotation-only transformation.
   */
  static Pose2 from_rotation(const K angle) {
    return Pose2{SO2<K>::from_angle(angle), Vector2<K>::Zero()};
  }

  /**
   * @brief Create a pose from position and angle components.
   *
   * @param px X component of translation.
   * @param py Y component of translation.
   * @param angle Rotation angle in radians.
   * @return Pose with specified translation and rotation.
   */
  static Pose2 from_xy_a(K px, K py, K angle) {
    return Pose2{SO2<K>::from_angle(angle), Vector2<K>{px, py}};
  }

  /**
   * @brief Create a pose from a 3×3 transformation matrix.
   *
   * @param matrix 3×3 homogeneous transformation matrix.
   * @return Pose represented by the matrix.
   */
  static Pose2 from_matrix(const Matrix3<K>& matrix) {
    return Pose2{SO2<K>::from_normalized(matrix(0, 0), matrix(1, 0)),
                 Vector2<K>(matrix(0, 2), matrix(1, 2))};
  }

  /**
   * @brief Get the inverse transformation.
   *
   * @return Inverse transformation.
   */
  Pose2 inverse() const {
    const SO2<K> inv = rotation.inverse();
    return Pose2{inv, -(inv * translation)};
  }

  /**
   * @brief Get the 3×3 homogeneous transformation matrix representation.
   *
   * @return 3×3 transformation matrix.
   */
  Matrix3<K> matrix() const {
    Matrix3<K> ret;
    ret << rotation.cos(), -rotation.sin(), translation.x(), rotation.sin(), rotation.cos(),
        translation.y(), K(0), K(0), K(1);
    return ret;
  }

  /**
   * @brief Cast to a different scalar type.
   *
   * @tparam S Target scalar type.
   * @return Pose cast to the target type.
   */
  template <typename S, typename std::enable_if_t<!std::is_same<S, K>::value, int> = 0>
  Pose2<S> cast() const {
    return Pose2<S>{rotation.template cast<S>(), translation.template cast<S>()};
  }

  /**
   * @brief Cast to the same scalar type (no-op).
   *
   * @tparam S Target scalar type (same as K).
   * @return Reference to this pose.
   */
  template <typename S, typename std::enable_if_t<std::is_same<S, K>::value, int> = 0>
  const Pose2& cast() const {
    // Nothing to do as the type does not change
    return *this;
  }

  /**
   * @brief Compose two poses.
   *
   * @param lhs Left pose.
   * @param rhs Right pose.
   * @return Composed pose lhs * rhs.
   */
  friend Pose2 operator*(const Pose2& lhs, const Pose2& rhs) {
    return Pose2{lhs.rotation * rhs.rotation, lhs.rotation * rhs.translation + lhs.translation};
  }

  /**
   * @brief Transform a 2D vector with the given transformation.
   *
   * @param pose Transformation to apply.
   * @param vec Vector to transform.
   * @return Transformed vector.
   */
  friend Vector2<K> operator*(const Pose2& pose, const Vector2<K>& vec) {
    return pose.rotation * vec + pose.translation;
  }

  /**
   * @brief Compute the power of the transformation.
   *
   * This computes the transformation raised to the given exponent using
   * exponential coordinates and matrix exponentiation.
   *
   * @param exponent Power to raise the transformation to.
   * @return Transformation raised to the given power.
   */
  Pose2 pow(K exponent) const {
    const K angle = rotation.angle();
    const K half_angle = angle / K(2);
    const K csc_sin =
        is_almost_zero(angle)
            ? exponent * (K(1) + (K(1) - exponent * exponent) * half_angle * half_angle / K(6))
            : std::sin(half_angle * exponent) / std::sin(half_angle);
    const SO2<K> rot = SO2<K>::from_angle(half_angle * (exponent - K(1)));
    return Pose2{SO2<K>::from_angle(angle * exponent), csc_sin * (rot * translation)};
  }
};

/// Pose2 with double precision.
using Pose2d = Pose2<double>;
/// Pose2 with single precision.
using Pose2f = Pose2<float>;

/**
 * @brief Exponential map from three-dimensional tangent space to SE(2) manifold space.
 *
 * For SE(2) this function encodes the tangent space as a three-dimensional vector (tx, ty, a)
 * where (tx, ty) is the translation component and a is the angle.
 *
 * @tparam K Scalar type.
 * @param tangent Tangent space vector (tx, ty, angle).
 * @return Pose on the SE(2) manifold.
 */
template <typename K>
Pose2<K> pose2_exp(const Vector3<K>& tangent) {
  return Pose2<K>{SO2<K>::from_angle(tangent[2]), tangent.template head<2>()};
}

/**
 * @brief Logarithmic map from manifold to tangent space.
 *
 * This computes the tangent for a pose relative to the identity pose.
 * Log and exp are inverse to each other.
 *
 * @tparam K Scalar type.
 * @param pose Pose on the SE(2) manifold.
 * @return Tangent space vector (tx, ty, angle).
 */
template <typename K>
Vector3<K> pose2_log(const Pose2<K>& pose) {
  return Vector3<K>{pose.translation.x(), pose.translation.y(), pose.rotation.angle()};
}

/**
 * @brief Check if given pose is almost identity.
 *
 * @tparam K Scalar type.
 * @param pose Pose to check.
 * @return True if the pose is very close to the identity transformation.
 */
template <typename K>
bool is_pose_almost_identity(const Pose2<K>& pose) {
  return is_almost_zero(pose.translation.x()) && is_almost_zero(pose.translation.y()) &&
         is_almost_one(pose.rotation.cos());
}

/**
 * @brief Get distance to identity pose in position and angle.
 *
 * @tparam K Scalar type.
 * @param pose Pose to measure.
 * @return Vector where first element is position magnitude, second is angle magnitude.
 */
template <typename K>
Vector2<K> pose2_magnitude(const Pose2<K>& pose) {
  return {pose.translation.norm(), std::abs(pose.rotation.angle())};
}

}  // namespace holoscan

#endif /* HOLOSCAN_POSE_TREE_MATH_POSE2_HPP */

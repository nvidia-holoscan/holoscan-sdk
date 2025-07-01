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
#ifndef HOLOSCAN_POSE_TREE_MATH_POSE3_HPP
#define HOLOSCAN_POSE_TREE_MATH_POSE3_HPP

#include <cmath>

#include "holoscan/pose_tree/math/pose2.hpp"
#include "holoscan/pose_tree/math/so3.hpp"
#include "holoscan/pose_tree/math/types.hpp"

namespace holoscan {

/**
 * @brief Class representing 3D transformations (rigid body motion in 3D).
 *
 * This class represents elements of the SE(3) group, which combines 3D rotations and
 * translations. Each pose consists of a rotation component (SO3) and a translation
 * vector in 3D space.
 *
 * @tparam K Scalar type (typically float or double).
 */
template <typename K>
struct Pose3 {
  using Scalar = K;
  static constexpr int kDimension = 3;

  /// Rotation component.
  SO3<K> rotation;
  /// Translation component.
  Vector3<K> translation;

  /**
   * @brief Default constructor creates identity pose.
   */
  Pose3() : rotation(SO3<K>::identity()), translation(Vector3<K>::Zero()) {}

  /**
   * @brief Constructor from rotation and translation components.
   *
   * @param rotation Rotation component.
   * @param translation Translation component.
   */
  Pose3(const SO3<K>& rotation, const Vector3<K>& translation)
      : rotation(rotation), translation(translation) {}

  /**
   * @brief Create the identity transformation.
   *
   * @return Identity transformation (no rotation or translation).
   */
  static Pose3 identity() { return Pose3{SO3<K>::identity(), Vector3<K>::Zero()}; }

  /**
   * @brief Create a pure translation transformation.
   *
   * @param translation Translation vector.
   * @return Translation-only transformation.
   */
  static Pose3 from_translation(const Vector3<K>& translation) {
    return Pose3{SO3<K>::identity(), translation};
  }

  /**
   * @brief Create a pure translation transformation from components.
   *
   * @param x X component of translation.
   * @param y Y component of translation.
   * @param z Z component of translation.
   * @return Translation-only transformation.
   */
  static Pose3 from_translation(K x, K y, K z) {
    return Pose3{SO3<K>::identity(), Vector3<K>{x, y, z}};
  }

  /**
   * @brief Create a pure rotation transformation.
   *
   * @param axis Rotation axis (will be normalized internally).
   * @param angle Rotation angle in radians.
   * @return Rotation-only transformation.
   */
  static Pose3 from_rotation(const Vector3<K>& axis, K angle) {
    return Pose3{SO3<K>::from_axis_angle(axis, angle), Vector3<K>::Zero()};
  }

  /**
   * @brief Create a 3D pose from a 2D pose in the XY plane.
   *
   * @param pose 2D pose in the XY plane.
   * @return 3D pose with Z translation = 0 and rotation around Z-axis.
   */
  static Pose3 from_pose2_xy(const Pose2<K>& pose) {
    return Pose3{SO3<K>::from_so2_xy(pose.rotation),
                 Vector3<K>{pose.translation.x(), pose.translation.y(), K(0)}};
  }

  /**
   * @brief Create a pose from a 4×4 transformation matrix.
   *
   * @param matrix 4×4 homogeneous transformation matrix.
   * @return Pose represented by the matrix.
   */
  static Pose3 from_matrix(const Matrix4<K>& matrix) {
    return Pose3{
        SO3<K>::from_normalized_quaternion(Quaternion<K>(matrix.template topLeftCorner<3, 3>())),
        matrix.template topRightCorner<3, 1>()};
  }

  /**
   * @brief Get the inverse transformation.
   *
   * @return Inverse transformation.
   */
  Pose3 inverse() const {
    const auto inv = rotation.inverse();
    return Pose3{inv, -(inv * translation)};
  }

  /**
   * @brief Get the 4×4 homogeneous transformation matrix representation.
   *
   * @return 4×4 transformation matrix.
   */
  Matrix4<K> matrix() const {
    Matrix4<K> ret;
    ret.template topLeftCorner<3, 3>() = rotation.matrix();
    ret.template topRightCorner<3, 1>() = translation;
    ret.template bottomLeftCorner<1, 3>().setZero();
    ret(3, 3) = K(1);
    return ret;
  }

  /**
   * @brief Cast to a different scalar type.
   *
   * @tparam S Target scalar type.
   * @return Pose cast to the target type.
   */
  template <typename S, typename std::enable_if_t<!std::is_same<S, K>::value, int> = 0>
  Pose3<S> cast() const {
    return Pose3<S>{rotation.template cast<S>(), translation.template cast<S>()};
  }

  /**
   * @brief Cast to the same scalar type (no-op).
   *
   * @tparam S Target scalar type (same as K).
   * @return Reference to this pose.
   */
  template <typename S, typename std::enable_if_t<std::is_same<S, K>::value, int> = 0>
  const Pose3& cast() const {
    // Nothing to do as the type does not change
    return *this;
  }

  /**
   * @brief Convert to a 2D pose in the XY plane.
   *
   * @return 2D pose representing the XY components of this 3D pose.
   */
  Pose2<K> to_pose2_xy() const {
    return Pose2<K>{rotation.to_so2_xy(), translation.template head<2>()};
  }

  /**
   * @brief Compose two poses.
   *
   * @param lhs Left pose.
   * @param rhs Right pose.
   * @return Composed pose lhs * rhs.
   */
  friend Pose3 operator*(const Pose3& lhs, const Pose3& rhs) {
    return Pose3{lhs.rotation * rhs.rotation, lhs.rotation * rhs.translation + lhs.translation};
  }

  /**
   * @brief Transform a 3D vector with the given transformation.
   *
   * @param pose Transformation to apply.
   * @param vec Vector to transform.
   * @return Transformed vector.
   */
  friend Vector3<K> operator*(const Pose3& pose, const Vector3<K>& vec) {
    return pose.rotation * vec + pose.translation;
  }

  /**
   * @brief Compute the power of the transformation.
   *
   * This computes the transformation raised to the given exponent using
   * exponential coordinates and matrix exponentiation. The implementation
   * handles the general case by aligning the rotation axis with the Z-axis.
   *
   * @param exponent Power to raise the transformation to.
   * @return Transformation raised to the given power.
   */
  Pose3 pow(K exponent) const {
    // First step: align the rotation vector with the z axis.
    const K angle = rotation.angle();
    if (is_almost_zero(angle)) {
      // TODO(bbutin): Use Taylor expansion in that case?
      return Pose3{rotation, exponent * translation};
    }
    const Vector3<K> axis = rotation.axis();
    const Vector3<K> cross = axis.cross(Vector3<K>(K(0), K(0), K(1)));
    const K cos = axis.z();
    const K sin = cross.norm();

    Matrix3<K> rot = Matrix3<K>::Identity() * cos;
    // If axis is align in the Z direction, the matrix above will do the trick, if not we compute
    // rotation matrix to transform the axis in the Z axis.
    if (!is_almost_zero(sin)) {
      // TODO(bbutin): Use Taylor expansion in case sin ~ 0?
      const Vector3<K> unit = cross / sin;
      rot += (1.0 - cos) * unit * unit.transpose();
      rot(0, 1) += -cross.z();
      rot(0, 2) += cross.y();
      rot(1, 2) += -cross.x();
      rot(1, 0) += cross.z();
      rot(2, 0) += -cross.y();
      rot(2, 1) += cross.x();
    }
    // Now we can compute the exponentiation the same way as for the 2d for x and y, while z will
    // just be linearly FromScaledAxis
    const K half_angle = angle / K(2);
    const K csc_sin =
        is_almost_zero(angle)
            ? exponent * (K(1) + (K(1) - exponent * exponent) * half_angle * half_angle / K(6))
            : std::sin(half_angle * exponent) / std::sin(half_angle);
    const SO2<K> rot2 = SO2<K>::from_angle(half_angle * (exponent - K(1)));
    const Vector3<K> t = rot * translation;
    const Vector2<K> xy = csc_sin * (rot2 * t.template head<2>());
    const SO3<K> final_rotation = SO3<K>::from_angle_axis(angle * exponent, axis);
    return Pose3{final_rotation, rot.inverse() * Vector3<K>(xy.x(), xy.y(), t.z() * exponent)};
  }
};

/// Pose3 with double precision.
using Pose3d = Pose3<double>;
/// Pose3 with single precision.
using Pose3f = Pose3<float>;

/**
 * @brief Exponential map of SE(3) which maps a tangent space element to the manifold space.
 *
 * For SE(3) this function encodes the tangent space as a six-dimensional vector
 * (px, py, pz, rx, ry, rz) where (px, py, pz) is the translation component and (rx, ry, rz) is the
 * scaled rotation axis.
 *
 * @tparam K Scalar type.
 * @param tangent Tangent space vector (tx, ty, tz, rx, ry, rz).
 * @return Pose on the SE(3) manifold.
 */
template <typename K>
Pose3<K> pose3_exp(const Vector6<K>& tangent) {
  return Pose3<K>{SO3<K>::from_scaled_axis(tangent.template tail<3>()), tangent.template head<3>()};
}

/**
 * @brief Logarithmic map of SE(3) which maps a manifold space element to the tangent space.
 *
 * This computes the tangent for a pose relative to the identity pose.
 * Log and exp are inverse to each other.
 *
 * @tparam K Scalar type.
 * @param pose Pose on the SE(3) manifold.
 * @return Tangent space vector (tx, ty, tz, rx, ry, rz).
 */
template <typename K>
Vector6<K> pose3_log(const Pose3<K>& pose) {
  Vector6<K> result;
  result.template head<3>() = pose.translation;
  result.template tail<3>() = pose.rotation.angle() * pose.rotation.axis();
  return result;
}

/**
 * @brief Check if given pose is almost identity.
 *
 * @tparam K Scalar type.
 * @param pose Pose to check.
 * @return True if the pose is very close to the identity transformation.
 */
template <typename K>
bool is_pose_almost_identity(const Pose3<K>& pose) {
  return is_almost_zero(pose.translation.norm()) && is_almost_zero(pose.rotation.angle());
}

/**
 * @brief Get distance to identity pose in position and angle.
 *
 * @tparam K Scalar type.
 * @param pose Pose to measure.
 * @return Vector where first element is position magnitude, second is angle magnitude.
 */
template <typename K>
Vector2<K> pose3_magnitude(const Pose3<K>& pose) {
  return {pose.translation.norm(), static_cast<K>(std::abs(pose.rotation.angle()))};
}

}  // namespace holoscan

#endif /* HOLOSCAN_POSE_TREE_MATH_POSE3_HPP */

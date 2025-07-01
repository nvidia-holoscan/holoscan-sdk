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
#ifndef HOLOSCAN_POSE_TREE_MATH_SO3_HPP
#define HOLOSCAN_POSE_TREE_MATH_SO3_HPP

#include <cmath>

#include "holoscan/pose_tree/math/so2.hpp"
#include "holoscan/pose_tree/math/types.hpp"

namespace holoscan {

/**
 * @brief Class representing 3D rotations using the SO(3) group.
 *
 * This class represents rotations in 3D space as elements of the special orthogonal group SO(3).
 * Internally, rotations are stored as unit quaternions to ensure numerical stability and
 * efficient composition operations.
 *
 * @tparam K Scalar type (typically float or double).
 */
template <typename K>
class SO3 {
 public:
  using Scalar = K;
  static constexpr int kDimension = 3;

  /**
   * @brief Default constructor creates an uninitialized quaternion.
   *
   * @note Use identity() to create an identity rotation.
   */
  SO3() {}

  /**
   * @brief Create the identity rotation.
   *
   * @return Identity rotation (no rotation).
   */
  static SO3 identity() { return SO3(Quaternion<K>::Identity()); }

  /**
   * @brief Create rotation which rotates around the given axis by the magnitude of the axis.
   *
   * @param axis_angle Scaled axis vector where the magnitude represents the rotation angle.
   * @return Rotation around the scaled axis.
   */
  static SO3 from_scaled_axis(const Vector3<K>& axis_angle) {
    const K norm = axis_angle.norm();
    if (is_almost_zero(norm)) {
      return SO3::identity();
    } else {
      return SO3(Quaternion<K>(Eigen::AngleAxis<K>(norm, axis_angle / norm)));
    }
  }

  /**
   * @brief Create rotation which rotates by an angle around a given axis.
   *
   * @param axis Rotation axis (will be normalized internally).
   * @param angle Rotation angle in radians.
   * @return Rotation around the given axis by the specified angle.
   */
  static SO3 from_axis_angle(const Vector3<K>& axis, K angle) {
    return SO3(Quaternion<K>(Eigen::AngleAxis<K>(angle, axis.normalized())));
  }

  /**
   * @brief Create rotation from angle and axis (alternative parameter order).
   *
   * @param angle Rotation angle in radians.
   * @param axis Rotation axis (will be normalized internally).
   * @return Rotation around the given axis by the specified angle.
   */
  static SO3 from_angle_axis(K angle, const Vector3<K>& axis) {
    return SO3(Quaternion<K>(Eigen::AngleAxis<K>(angle, axis.normalized())));
  }

  /**
   * @brief Create rotation from a (not necessarily normalized) quaternion.
   *
   * @param quaternion Quaternion representation (will be normalized internally).
   * @return Rotation represented by the normalized quaternion.
   */
  static SO3 from_quaternion(const Quaternion<K>& quaternion) {
    return SO3(quaternion.normalized());
  }

  /**
   * @brief Create rotation from a normalized quaternion.
   *
   * @note This will assert if the quaternion does not have unit length.
   * @param quaternion Normalized quaternion representation.
   * @return Rotation represented by the quaternion.
   */
  static SO3 from_normalized_quaternion(const Quaternion<K>& quaternion) { return SO3(quaternion); }

  /**
   * @brief Create a 3D rotation from a 2D rotation in the XY plane.
   *
   * @param rotation 2D rotation in the XY plane.
   * @return 3D rotation equivalent to the 2D rotation around the Z-axis.
   */
  static SO3 from_so2_xy(const SO2<K>& rotation) {
    return from_axis_angle({K(0), K(0), K(1)}, rotation.angle());
  }

  /**
   * @brief Create rotation from a 3×3 rotation matrix.
   *
   * @param matrix 3×3 rotation matrix.
   * @return Rotation represented by the matrix.
   */
  static SO3 from_matrix(const Matrix<K, 3, 3>& matrix) {
    return SO3(Quaternion<K>(matrix).normalized());
  }

  /**
   * @brief Get the rotation axis.
   *
   * @return Normalized rotation axis.
   */
  Vector3<K> axis() const { return quaternion_.coeffs().head(3).normalized(); }

  /**
   * @brief Get the angle of rotation around the axis.
   *
   * @note This calls a trigonometric function.
   * @return Rotation angle in radians.
   */
  K angle() const {
    return K(2) * std::atan2(quaternion_.coeffs().head(3).norm(), quaternion_.coeffs().w());
  }

  /**
   * @brief Get the quaternion representation of the rotation.
   *
   * @return Unit quaternion representing the rotation.
   */
  const Quaternion<K>& quaternion() const { return quaternion_; }

  /**
   * @brief Get the 3×3 rotation matrix representation.
   *
   * @return 3×3 rotation matrix.
   */
  Matrix3<K> matrix() const { return quaternion_.toRotationMatrix(); }

  /**
   * @brief Get the roll, pitch, yaw Euler angles of the rotation.
   *
   * @return Vector containing (roll, pitch, yaw) angles in radians.
   */
  Vector3<K> euler_angles_rpy() const {
    Vector3d euler_angles = quaternion_.toRotationMatrix().eulerAngles(0, 1, 2);
    // Make sure the roll is in range [-Pi/2, Pi/2]
    if (std::abs(euler_angles[0]) > M_PI * K(0.5)) {
      euler_angles[0] -= M_PI;
      euler_angles[1] = M_PI - euler_angles[1];
      euler_angles[2] += M_PI;
      // TODO(bbutin): make sure we are in the range [-Pi, Pi]
    }
    return euler_angles;
  }

  /**
   * @brief Get the inverse rotation.
   *
   * @return Inverse rotation.
   */
  SO3 inverse() const {
    // conjugate == inverse iff quaternion_.norm() == 1
    return SO3(quaternion_.conjugate());
  }

  /**
   * @brief Cast to a different scalar type.
   *
   * @tparam S Target scalar type.
   * @return Rotation cast to the target type.
   */
  template <typename S, typename std::enable_if_t<!std::is_same<S, K>::value, int> = 0>
  SO3<S> cast() const {
    // We need to re-normalize in the new type
    return SO3<S>::from_quaternion(quaternion().template cast<S>());
  }

  /**
   * @brief Cast to the same scalar type (no-op).
   *
   * @tparam S Target scalar type (same as K).
   * @return Reference to this rotation.
   */
  template <typename S, typename std::enable_if_t<std::is_same<S, K>::value, int> = 0>
  const SO3& cast() const {
    // Nothing to do as the type does not change
    return *this;
  }

  /**
   * @brief Convert to a 2D rotation in the XY plane.
   *
   * @return 2D rotation representing the Z-axis component of this 3D rotation.
   */
  SO2<K> to_so2_xy() const {
    // 2D rotation matrix:
    //   cos(a)   -sin(a)
    //   sin(a)    cos(a)
    // Quaternion to 3D rotation matrix:
    //   1 - 2*(qy^2 + qz^2)      2*(qx*qy - qz*qw)   ...
    //     2*(qx*qy + qz*qw)    1 - 2*(qx^2 + qz^2)   ...
    // It follows (modulo re-normalization):
    //   cos(a) = 1 - (qx^2 + qy^2 + 2*qz^2)
    //   sin(a) = 2*qz*qw
    // These formulas correspond to the half-angle formulas for sin/cos.
    const K qx = quaternion_.x();
    const K qy = quaternion_.y();
    const K qz = quaternion_.z();
    const K qw = quaternion_.w();
    const K cos_a = K(1) - (qx * qx + qy * qy + K(2) * qz * qz);
    const K sin_a = K(2) * qz * qw;
    return SO2<K>::from_direction(cos_a, sin_a);
  }

  /**
   * @brief Compose two rotations.
   *
   * @param lhs Left rotation.
   * @param rhs Right rotation.
   * @return Composed rotation lhs * rhs.
   */
  friend SO3 operator*(const SO3& lhs, const SO3& rhs) {
    return from_quaternion(lhs.quaternion_ * rhs.quaternion_);
  }

  /**
   * @brief Rotate a 3D vector by the given rotation.
   *
   * @param rot Rotation to apply.
   * @param vec Vector to rotate.
   * @return Rotated vector.
   */
  friend Vector3<K> operator*(const SO3& rot, const Vector3<K>& vec) {
    // TODO: faster implementation
    return (rot.quaternion_ * Quaternion<K>(K(0), vec.x(), vec.y(), vec.z()) *
            rot.quaternion_.conjugate())
        .coeffs()
        .head(3);
  }

  /**
   * @brief Create rotation from roll/pitch/yaw Euler angles.
   *
   * @param roll_angle Roll angle in radians (rotation around X-axis).
   * @param pitch_angle Pitch angle in radians (rotation around Y-axis).
   * @param yaw_angle Yaw angle in radians (rotation around Z-axis).
   * @return Rotation representing the given Euler angles.
   */
  static SO3 from_euler_angles_rpy(K roll_angle, K pitch_angle, K yaw_angle) {
    SO3 roll = SO3::from_angle_axis(roll_angle, {K(1), K(0), K(0)});
    SO3 pitch = SO3::from_angle_axis(pitch_angle, {K(0), K(1), K(0)});
    SO3 yaw = SO3::from_angle_axis(yaw_angle, {K(0), K(0), K(1)});
    return roll * pitch * yaw;
  }

  /**
   * @brief Create rotation from Euler angles vector.
   *
   * @param roll_pitch_yaw Vector containing (roll, pitch, yaw) angles in radians.
   * @return Rotation representing the given Euler angles.
   */
  static SO3 from_euler_angles_rpy(const Vector3d& roll_pitch_yaw) {
    return from_euler_angles_rpy(roll_pitch_yaw[0], roll_pitch_yaw[1], roll_pitch_yaw[2]);
  }

  /**
   * @brief Compute the Jacobian of the rotation of a normal vector.
   *
   * Plane normals only have rotation components.
   *
   * @param n Normal vector to compute Jacobian for.
   * @return 3×4 Jacobian matrix.
   */
  Matrix<K, 3, 4> vector_rotation_jacobian(const Vector3<K>& n) const {
    // Ref: https://www.weizmann.ac.il/sci-tea/benari/sites/sci-tea.benari/files/uploads/
    // softwareAndLearningMaterials/quaternion-tutorial-2-0-1.pdf
    // Rotation Matrix in Quaternion (R):
    //     [w^2 + x^2 - y^2 - z^2    2xy - 2wz                 2wy + 2xz]
    //     [2wz + 2xy                w^2 - x^2 + y^2 - z^2     2yz - 2wx]
    //     [2xz - 2wy                2wx + 2yz                 w^2 - x^2 - y^2 + z^2]
    const K qx = quaternion_.x();
    const K qy = quaternion_.y();
    const K qz = quaternion_.z();
    const K qw = quaternion_.w();
    Matrix<K, 3, 4> result;
    result << K(2) * (qw * n[0] - qz * n[1] + qy * n[2]),
        K(2) * (qx * n[0] + qy * n[1] + qz * n[2]), K(-2) * (qy * n[0] - qx * n[1] - qw * n[2]),
        K(-2) * (qz * n[0] + qw * n[1] - qx * n[2]),

        K(2) * (qz * n[0] + qw * n[1] - qx * n[2]), K(2) * (qy * n[0] - qx * n[1] - qw * n[2]),
        K(2) * (qx * n[0] + qy * n[1] + qz * n[2]), K(2) * (qw * n[0] - qz * n[1] + qy * n[2]),

        K(-2) * (qy * n[0] - qx * n[1] - qw * n[2]), K(2) * (qz * n[0] + qw * n[1] - qx * n[2]),
        K(-2) * (qw * n[0] - qz * n[1] + qy * n[2]), K(2) * (qx * n[0] + qy * n[1] + qz * n[2]);
    return result;
  }

 private:
  /**
   * @brief Private constructor from quaternion.
   *
   * @param quaternion Unit quaternion representing the rotation.
   */
  explicit SO3(const Quaternion<K>& quaternion) : quaternion_(quaternion) {}

  /// Internal quaternion representation.
  Quaternion<K> quaternion_;
};

/// SO3 with double precision.
using SO3d = SO3<double>;
/// SO3 with single precision.
using SO3f = SO3<float>;

}  // namespace holoscan

#endif /* HOLOSCAN_POSE_TREE_MATH_SO3_HPP */

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
#ifndef HOLOSCAN_POSE_TREE_MATH_TYPES_HPP
#define HOLOSCAN_POSE_TREE_MATH_TYPES_HPP

#include <Eigen/Eigen>

#include <cmath>
#include <limits>

namespace holoscan {

/**
 * @brief Check if a value is almost zero within numerical precision.
 *
 * @tparam K Scalar type.
 * @param value Value to check.
 * @return True if the value is within 100 * epsilon of zero.
 */
template <typename K>
bool is_almost_zero(K value) {
  return std::abs(value) < std::numeric_limits<K>::epsilon() * K(100);
}

/**
 * @brief Check if a value is almost one within numerical precision.
 *
 * @tparam K Scalar type.
 * @param value Value to check.
 * @return True if the value is within 100 * epsilon of one.
 */
template <typename K>
bool is_almost_one(K value) {
  return std::abs(1.0 - value) < std::numeric_limits<K>::epsilon() * K(100);
}

/// Generic matrix type alias.
template <typename K, int N, int M>
using Matrix = Eigen::Matrix<K, N, M>;

/// Generic vector type alias (column vector).
template <typename K, int N>
using Vector = Eigen::Matrix<K, N, 1>;

/// Generic row vector type alias.
template <typename K, int N>
using RowVector = Eigen::Matrix<K, 1, N>;

/// Generic column vector type alias.
template <typename K, int N>
using ColVector = Eigen::Matrix<K, N, 1>;

/// Dynamic-sized vector types.
template <typename K>
using VectorX = Vector<K, Eigen::Dynamic>;
using VectorXd = VectorX<double>;
using VectorXf = VectorX<float>;
using VectorXi = VectorX<int>;
using VectorXub = VectorX<uint8_t>;

/// Dynamic-sized row vector types.
template <typename K>
using RowVectorX = RowVector<K, Eigen::Dynamic>;
using RowVectorXf = RowVectorX<float>;

/// Dynamic-sized column vector types.
template <typename K>
using ColVectorX = ColVector<K, Eigen::Dynamic>;
using ColVectorXf = ColVectorX<float>;

/// Dynamic-sized matrix types.
template <typename K>
using MatrixX = Matrix<K, Eigen::Dynamic, Eigen::Dynamic>;
using MatrixXd = MatrixX<double>;
using MatrixXf = MatrixX<float>;
using MatrixXi = MatrixX<int>;

/**
 * @brief Macro to define fixed-size square matrix types.
 *
 * Creates template aliases for square matrices of size N×N and common scalar types.
 */
#define DEFINE_MATRIX_TYPES(N)            \
  template <typename K>                   \
  using Matrix##N = Matrix<K, N, N>;      \
  using Matrix##N##d = Matrix##N<double>; \
  using Matrix##N##f = Matrix##N<float>;  \
  using Matrix##N##i = Matrix##N<int>;

DEFINE_MATRIX_TYPES(2)
DEFINE_MATRIX_TYPES(3)
DEFINE_MATRIX_TYPES(4)
DEFINE_MATRIX_TYPES(5)
DEFINE_MATRIX_TYPES(6)
DEFINE_MATRIX_TYPES(7)
DEFINE_MATRIX_TYPES(8)

#undef DEFINE_MATRIX_TYPES

/**
 * @brief Macro to define matrix types with fixed rows and dynamic columns.
 *
 * Creates template aliases for matrices with N fixed rows and dynamic columns.
 * These are useful for storing and efficiently manipulating sets of geometric entities:
 * - 2×N: 2D points in Euclidean coordinates
 * - 3×N: 2D points in homogeneous coordinates or 3D points in Euclidean coordinates
 * - 4×N: planes or 3D points in homogeneous coordinates
 */
#define DEFINE_FIXED_ROWS_MATRIX_TYPES(N)            \
  template <typename K>                              \
  using Matrix##N##X = Matrix<K, N, Eigen::Dynamic>; \
  using Matrix##N##Xd = Matrix##N##X<double>;        \
  using Matrix##N##Xf = Matrix##N##X<float>;         \
  using Matrix##N##Xi = Matrix##N##X<int>;

DEFINE_FIXED_ROWS_MATRIX_TYPES(2)
DEFINE_FIXED_ROWS_MATRIX_TYPES(3)
DEFINE_FIXED_ROWS_MATRIX_TYPES(4)
DEFINE_FIXED_ROWS_MATRIX_TYPES(5)
DEFINE_FIXED_ROWS_MATRIX_TYPES(6)
DEFINE_FIXED_ROWS_MATRIX_TYPES(7)
DEFINE_FIXED_ROWS_MATRIX_TYPES(8)

#undef DEFINE_FIXED_ROWS_MATRIX_TYPES

/// 3×4 matrix types (commonly used for camera projection matrices).
template <typename K>
using Matrix34 = Matrix<K, 3, 4>;
using Matrix34d = Matrix34<double>;
using Matrix34f = Matrix34<float>;
using Matrix34i = Matrix34<int>;

/// 4×3 matrix types.
template <typename K>
using Matrix43 = Matrix<K, 4, 3>;
using Matrix43d = Matrix43<double>;
using Matrix43f = Matrix43<float>;
using Matrix43i = Matrix43<int>;

/**
 * @brief Macro to define fixed-size vector types.
 *
 * Creates template aliases for vectors of size N and common scalar types.
 */
#define DEFINE_VECTOR_TYPES(N)            \
  template <typename K>                   \
  using Vector##N = Vector<K, N>;         \
  using Vector##N##d = Vector##N<double>; \
  using Vector##N##f = Vector##N<float>;  \
  using Vector##N##i = Vector##N<int>;    \
  using Vector##N##ub = Vector##N<uint8_t>;

DEFINE_VECTOR_TYPES(2)
DEFINE_VECTOR_TYPES(3)
DEFINE_VECTOR_TYPES(4)
DEFINE_VECTOR_TYPES(5)
DEFINE_VECTOR_TYPES(6)
DEFINE_VECTOR_TYPES(7)
DEFINE_VECTOR_TYPES(8)

#undef DEFINE_VECTOR_TYPES

/// Quaternion type aliases.
template <typename K>
using Quaternion = Eigen::Quaternion<K>;
using Quaterniond = Quaternion<double>;
using Quaternionf = Quaternion<float>;

/**
 * @brief Add two quaternions.
 *
 * @tparam K Scalar type.
 * @param lhs Left operand.
 * @param rhs Right operand.
 * @return Sum of the two quaternions.
 */
template <typename K>
Quaternion<K> operator+(const Quaternion<K>& lhs, const Quaternion<K>& rhs) {
  return Quaternion<K>(lhs.coeffs() + rhs.coeffs());
}

/**
 * @brief Subtract two quaternions.
 *
 * @tparam K Scalar type.
 * @param lhs Left operand.
 * @param rhs Right operand.
 * @return Difference of the two quaternions.
 */
template <typename K>
Quaternion<K> operator-(const Quaternion<K>& lhs, const Quaternion<K>& rhs) {
  return Quaternion<K>(lhs.coeffs() - rhs.coeffs());
}

/**
 * @brief Negate a quaternion.
 *
 * @tparam K Scalar type.
 * @param q Quaternion to negate.
 * @return Negated quaternion.
 */
template <typename K>
Quaternion<K> operator-(const Quaternion<K>& q) {
  return Quaternion<K>(-q.coeffs());
}

}  // namespace holoscan

#endif /* HOLOSCAN_POSE_TREE_MATH_TYPES_HPP */

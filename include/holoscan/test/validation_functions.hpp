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

#pragma once

#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace holoscan::test {

/**
 * @brief Creates a validation function that compares each output against expected values
 *        using exact equality
 *
 * @tparam T The data type to validate (must support operator==)
 * @param expected_values Vector of expected values to compare against
 * @return Validation function that can be used with TestHarnessSinkOp
 */
template<typename T>
std::function<void(const T&)> create_exact_equality_validator(
    const std::vector<T>& expected_values) {
  static_assert(std::is_arithmetic_v<T> || std::is_same_v<T, std::string>,
                "Type must be arithmetic or string for exact equality validation");

  // Use shared_ptr to ensure the index persists across lambda calls
  auto output_index = std::make_shared<size_t>(0);

  return [expected_values, output_index](const T& output) {
    EXPECT_LT(*output_index, expected_values.size()) << "Received more outputs than expected";
    if (*output_index < expected_values.size()) {
      EXPECT_EQ(output, expected_values[*output_index])
        << "Output " << *output_index << " should be " << expected_values[*output_index]
        << " but got " << output;
      (*output_index)++;
    }
  };
}

/**
 * @brief Creates a validation function that compares floating-point values using approximate equality
 *
 * @tparam T The floating-point data type (float, double, etc.)
 * @param expected_values Vector of expected values to compare against
 * @param tolerance Optional tolerance for floating-point comparison (default uses gtest default)
 * @return Validation function that can be used with TestHarnessSinkOp
 */
template<typename T>
std::function<void(const T&)> create_float_equality_validator(
    const std::vector<T>& expected_values, T tolerance = T{}) {
  static_assert(std::is_floating_point_v<T>,
                "Type must be floating point for float equality validation");

  // Use shared_ptr to ensure the index persists across lambda calls
  auto output_index = std::make_shared<size_t>(0);
  bool use_tolerance = (tolerance != T{});

  return [expected_values, output_index, tolerance, use_tolerance](const T& output) {
    EXPECT_LT(*output_index, expected_values.size()) << "Received more outputs than expected";
    if (*output_index < expected_values.size()) {
      if (use_tolerance) {
        EXPECT_NEAR(output, expected_values[*output_index], tolerance)
          << "Output " << *output_index << " should be approximately "
          << expected_values[*output_index] << " (tolerance: " << tolerance
          << ") but got " << output;
      } else {
        EXPECT_FLOAT_EQ(output, expected_values[*output_index])
          << "Output " << *output_index << " should be approximately "
          << expected_values[*output_index]
          << " but got " << output;
      }
      (*output_index)++;
    }
  };
}

/**
 * @brief Creates a validation function that applies a transformation and then compares results
 *
 * @tparam InputT The input data type
 * @tparam OutputT The transformed data type for comparison
 * @param expected_values Vector of expected transformed values
 * @param transform_func Function to transform input before comparison
 * @return Validation function that can be used with TestHarnessSinkOp
 */
template<typename InputT, typename OutputT>
std::function<void(const InputT&)> create_transform_equality_validator(
    const std::vector<OutputT>& expected_values,
    std::function<OutputT(const InputT&)> transform_func) {
  // Use shared_ptr to ensure the index persists across lambda calls
  auto output_index = std::make_shared<size_t>(0);

  return [expected_values, output_index, transform_func](const InputT& output) {
    EXPECT_LT(*output_index, expected_values.size()) << "Received more outputs than expected";
    if (*output_index < expected_values.size()) {
      auto transformed = transform_func(output);
      EXPECT_EQ(transformed, expected_values[*output_index])
        << "Transformed output " << *output_index << " should be " << expected_values[*output_index]
        << " but got " << transformed << " (original: " << output << ")";
      (*output_index)++;
    }
  };
}

}  // namespace holoscan::test

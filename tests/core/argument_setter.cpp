/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <yaml-cpp/yaml.h>

#include <complex>
#include <cstdint>
#include <limits>
#include <string>
#include <typeinfo>

// clang-format off
#include "holoscan/core/component_spec.hpp"  // must be before argument_setter import
#include "holoscan/core/argument_setter.hpp"
#include "holoscan/core/parameter.hpp"
#include "holoscan/core/arg.hpp"
// clang-format on

namespace holoscan {

TEST(ArgumentSetter, TestArgumentSetterInstance) {
  ArgumentSetter instance = ArgumentSetter::get_instance();

  // call static ensure_type method
  ArgumentSetter::ensure_type<float>;  // NOLINT(clang-diagnostic-unused-value)

  // get the setter corresponding to float
  float f = 1.0;
  auto func = instance.get_argument_setter(typeid(f));
}

TEST(ArgumentSetter, SetParamThrowsOnBadAnyCast) {
  // Prepare an int parameter
  Parameter<int32_t> param_int;
  ParameterWrapper param_wrap(param_int);

  // Provide a mismatched native type (std::string) to trigger bad_any_cast path
  Arg arg("beta");
  arg = std::string("not_an_int");

  // Expect ArgumentSetter::set_param to throw with a descriptive message
  EXPECT_THROW(
      {
        try {
          ArgumentSetter::set_param(param_wrap, arg);
        } catch (const std::runtime_error& e) {
          std::string msg(e.what());
          EXPECT_NE(msg.find("Failed to set parameter 'beta'"), std::string::npos);
          EXPECT_NE(msg.find("arg_type"), std::string::npos);
          throw;
        }
      },
      std::runtime_error);
}

// ============================================================================
// Integer Type Conversion Tests
// ============================================================================

// Test widening conversions (always safe)
TEST(ArgumentSetter, IntegerWideningConversions) {
  // int8_t to int16_t
  {
    Parameter<int16_t> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<int8_t>(42);
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_EQ(param.get(), 42);
  }

  // int8_t to int32_t
  {
    Parameter<int32_t> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<int8_t>(-100);
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_EQ(param.get(), -100);
  }

  // int32_t to int64_t
  {
    Parameter<int64_t> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<int32_t>(1000000);
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_EQ(param.get(), 1000000);
  }

  // uint8_t to uint16_t
  {
    Parameter<uint16_t> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<uint8_t>(200);
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_EQ(param.get(), 200);
  }

  // uint16_t to uint32_t
  {
    Parameter<uint32_t> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<uint16_t>(50000);
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_EQ(param.get(), 50000);
  }

  // uint32_t to uint64_t
  {
    Parameter<uint64_t> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<uint32_t>(3000000000);
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_EQ(param.get(), 3000000000);
  }
}

// Test signed to unsigned conversions
TEST(ArgumentSetter, SignedToUnsignedConversions) {
  // Positive int64_t to uint32_t (should succeed)
  {
    Parameter<uint32_t> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<int64_t>(1000000);
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_EQ(param.get(), 1000000u);
  }

  // Negative int64_t to uint32_t (should fail)
  {
    Parameter<uint32_t> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<int64_t>(-100);
    EXPECT_THROW(ArgumentSetter::set_param(param_wrap, arg), std::runtime_error);
  }

  // Positive int32_t to uint16_t (value fits)
  {
    Parameter<uint16_t> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<int32_t>(30000);
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_EQ(param.get(), 30000u);
  }

  // Negative int32_t to uint16_t (should fail)
  {
    Parameter<uint16_t> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<int32_t>(-1);
    EXPECT_THROW(ArgumentSetter::set_param(param_wrap, arg), std::runtime_error);
  }

  // Positive int8_t to uint8_t
  {
    Parameter<uint8_t> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<int8_t>(100);
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_EQ(param.get(), 100u);
  }

  // Negative int8_t to uint8_t (should fail)
  {
    Parameter<uint8_t> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<int8_t>(-50);
    EXPECT_THROW(ArgumentSetter::set_param(param_wrap, arg), std::runtime_error);
  }

  // Zero int64_t to uint64_t (should succeed)
  {
    Parameter<uint64_t> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<int64_t>(0);
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_EQ(param.get(), 0u);
  }
}

// Test unsigned to signed conversions
TEST(ArgumentSetter, UnsignedToSignedConversions) {
  // uint8_t to int8_t (value fits in signed range)
  {
    Parameter<int8_t> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<uint8_t>(100);
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_EQ(param.get(), 100);
  }

  // uint8_t to int8_t (value too large for signed)
  {
    Parameter<int8_t> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<uint8_t>(200);  // > INT8_MAX (127)
    EXPECT_THROW(ArgumentSetter::set_param(param_wrap, arg), std::runtime_error);
  }

  // uint16_t to int16_t (value fits)
  {
    Parameter<int16_t> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<uint16_t>(30000);
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_EQ(param.get(), 30000);
  }

  // uint16_t to int16_t (value too large)
  {
    Parameter<int16_t> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<uint16_t>(40000);  // > INT16_MAX (32767)
    EXPECT_THROW(ArgumentSetter::set_param(param_wrap, arg), std::runtime_error);
  }

  // uint32_t to int32_t (value fits)
  {
    Parameter<int32_t> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<uint32_t>(1000000000);
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_EQ(param.get(), 1000000000);
  }

  // uint32_t to int32_t (value too large)
  {
    Parameter<int32_t> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<uint32_t>(3000000000u);  // > INT32_MAX (2147483647)
    EXPECT_THROW(ArgumentSetter::set_param(param_wrap, arg), std::runtime_error);
  }

  // uint64_t to int64_t (value fits)
  {
    Parameter<int64_t> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<uint64_t>(1000000000000LL);
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_EQ(param.get(), 1000000000000LL);
  }

  // uint64_t to int64_t (value too large)
  {
    Parameter<int64_t> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<uint64_t>(10000000000000000000ULL);  // > INT64_MAX
    EXPECT_THROW(ArgumentSetter::set_param(param_wrap, arg), std::runtime_error);
  }
}

// ============================================================================
// Parameterized Edge Case Tests for all Integer Types
// ============================================================================

// Helper template to get a wider type for boundary testing
template <typename T>
struct WiderType;

template <>
struct WiderType<int8_t> {
  using type = int16_t;
};
template <>
struct WiderType<int16_t> {
  using type = int32_t;
};
template <>
struct WiderType<int32_t> {
  using type = int64_t;
};
template <>
struct WiderType<uint8_t> {
  using type = uint16_t;
};
template <>
struct WiderType<uint16_t> {
  using type = uint32_t;
};
template <>
struct WiderType<uint32_t> {
  using type = uint64_t;
};

template <typename T>
using WiderType_t = typename WiderType<T>::type;

// Test fixture for typed tests
template <typename T>
class ArgumentSetterIntegerBoundaryTest : public ::testing::Test {};

// Define the types we want to test (excluding int64_t/uint64_t as they have no wider type)
using IntegerTypesToTest = ::testing::Types<int8_t, int16_t, int32_t, uint8_t, uint16_t, uint32_t>;

TYPED_TEST_SUITE(ArgumentSetterIntegerBoundaryTest, IntegerTypesToTest);

// Test that values at min and max boundaries convert successfully from wider types
TYPED_TEST(ArgumentSetterIntegerBoundaryTest, MinMaxValuesFromWiderType) {
  using TargetType = TypeParam;
  using SourceType = WiderType_t<TargetType>;

  // Test minimum value
  {
    Parameter<TargetType> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<SourceType>(std::numeric_limits<TargetType>::min());
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_EQ(param.get(), std::numeric_limits<TargetType>::min());
  }

  // Test maximum value
  {
    Parameter<TargetType> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<SourceType>(std::numeric_limits<TargetType>::max());
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_EQ(param.get(), std::numeric_limits<TargetType>::max());
  }
}

// Test that values just beyond min/max boundaries fail when converting from wider types
TYPED_TEST(ArgumentSetterIntegerBoundaryTest, BeyondMinMaxValuesFail) {
  using TargetType = TypeParam;
  using SourceType = WiderType_t<TargetType>;

  // Test one less than minimum (for signed types)
  if constexpr (std::is_signed_v<TargetType>) {
    Parameter<TargetType> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<SourceType>(static_cast<SourceType>(std::numeric_limits<TargetType>::min()) -
                                  1);
    EXPECT_THROW(ArgumentSetter::set_param(param_wrap, arg), std::runtime_error);
  }

  // Test one more than maximum
  {
    Parameter<TargetType> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<SourceType>(static_cast<SourceType>(std::numeric_limits<TargetType>::max()) +
                                  1);
    EXPECT_THROW(ArgumentSetter::set_param(param_wrap, arg), std::runtime_error);
  }
}

// Additional edge case tests for int64_t and uint64_t (which don't have wider types)
TEST(ArgumentSetter, Int64EdgeCases) {
  // Test INT64_MIN and INT64_MAX can be assigned
  {
    Parameter<int64_t> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = std::numeric_limits<int64_t>::min();
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_EQ(param.get(), std::numeric_limits<int64_t>::min());
  }

  {
    Parameter<int64_t> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = std::numeric_limits<int64_t>::max();
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_EQ(param.get(), std::numeric_limits<int64_t>::max());
  }
}

TEST(ArgumentSetter, UInt64EdgeCases) {
  // Test UINT64_MAX can be assigned
  {
    Parameter<uint64_t> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = std::numeric_limits<uint64_t>::max();
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_EQ(param.get(), std::numeric_limits<uint64_t>::max());
  }

  // Test 0 (min value) can be assigned
  {
    Parameter<uint64_t> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = std::numeric_limits<uint64_t>::min();
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_EQ(param.get(), 0ULL);
  }
}

// Test conversions to/from long and long long (platform-dependent sizes)
TEST(ArgumentSetter, IntegerLongTypeConversions) {
  // long to int64_t
  {
    Parameter<int64_t> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<long>(1234567L);  // NOLINT
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_EQ(param.get(), 1234567LL);
  }

  // long long to int64_t
  {
    Parameter<int64_t> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<long long>(123456789012LL);  // NOLINT
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_EQ(param.get(), 123456789012LL);
  }

  // int64_t to long
  {
    Parameter<long> param;  // NOLINT
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<int64_t>(7654321LL);
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_EQ(param.get(), 7654321L);
  }

  // int64_t to long long
  {
    Parameter<long long> param;  // NOLINT
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<int64_t>(7654321LL);
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_EQ(param.get(), 7654321LL);
  }

  // uint64_t to unsigned long
  {
    Parameter<unsigned long> param;  // NOLINT
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<uint64_t>(9876543456789012345ULL);
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_EQ(param.get(), 9876543456789012345ULL);
  }
  // uint64_t to unsigned long long
  {
    Parameter<unsigned long long> param;  // NOLINT
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<uint64_t>(9876543456789012345ULL);
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_EQ(param.get(), 9876543456789012345ULL);
  }

  // unsigned long to uint64_t
  {
    Parameter<uint64_t> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<unsigned long>(9876543UL);  // NOLINT
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_EQ(param.get(), 9876543ULL);
  }

  // unsigned long long to uint64_t
  {
    Parameter<uint64_t> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<unsigned long long>(9876543456789012345ULL);  // NOLINT
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_EQ(param.get(), 9876543456789012345ULL);
  }
}

// ============================================================================
// Floating Point Type Conversion Tests
// ============================================================================

// Test float to double conversion (always safe - widening)
TEST(ArgumentSetter, FloatToDoubleConversion) {
  // Normal positive value
  {
    Parameter<double> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = 3.14159f;
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_FLOAT_EQ(static_cast<float>(param.get()), 3.14159f);
  }

  // Normal negative value
  {
    Parameter<double> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = -2.71828f;
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_FLOAT_EQ(static_cast<float>(param.get()), -2.71828f);
  }

  // Zero
  {
    Parameter<double> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = 0.0f;
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_DOUBLE_EQ(param.get(), 0.0);
  }

  // Very small value
  {
    Parameter<double> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = 1.0e-30f;
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_FLOAT_EQ(static_cast<float>(param.get()), 1.0e-30f);
  }

  // FLT_MAX (maximum float value)
  {
    Parameter<double> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = std::numeric_limits<float>::max();
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_FLOAT_EQ(static_cast<float>(param.get()), std::numeric_limits<float>::max());
  }

  // FLT_MIN (smallest positive normalized float)
  {
    Parameter<double> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = std::numeric_limits<float>::min();
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_FLOAT_EQ(static_cast<float>(param.get()), std::numeric_limits<float>::min());
  }
}

// Test double to float conversion (narrowing - may lose precision or overflow)
TEST(ArgumentSetter, DoubleToFloatConversion) {
  // Normal value within float range
  {
    Parameter<float> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = 3.14159265358979;
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    // Note: precision is lost but value is representable
    EXPECT_FLOAT_EQ(param.get(), 3.14159265358979f);
  }

  // Negative value within float range
  {
    Parameter<float> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = -2.718281828459045;
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_FLOAT_EQ(param.get(), -2.718281828459045f);
  }

  // Zero
  {
    Parameter<float> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = 0.0;
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_FLOAT_EQ(param.get(), 0.0f);
  }

  // Value at float max boundary (should succeed)
  {
    Parameter<float> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<double>(std::numeric_limits<float>::max());
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_FLOAT_EQ(param.get(), std::numeric_limits<float>::max());
  }

  // Value at float lowest boundary (should succeed)
  {
    Parameter<float> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<double>(std::numeric_limits<float>::lowest());
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_FLOAT_EQ(param.get(), std::numeric_limits<float>::lowest());
  }

  // Value larger than float max (should fail - overflow)
  {
    Parameter<float> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<double>(std::numeric_limits<float>::max()) * 2.0;
    EXPECT_THROW(ArgumentSetter::set_param(param_wrap, arg), std::runtime_error);
  }

  // Value smaller than float lowest (should fail - underflow)
  {
    Parameter<float> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<double>(std::numeric_limits<float>::lowest()) * 2.0;
    EXPECT_THROW(ArgumentSetter::set_param(param_wrap, arg), std::runtime_error);
  }

  // Very large positive double (beyond float range)
  {
    Parameter<float> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = 1.0e100;  // Way beyond float max (~3.4e38)
    EXPECT_THROW(ArgumentSetter::set_param(param_wrap, arg), std::runtime_error);
  }

  // Very large negative double (beyond float range)
  {
    Parameter<float> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = -1.0e100;  // Way beyond float lowest (~-3.4e38)
    EXPECT_THROW(ArgumentSetter::set_param(param_wrap, arg), std::runtime_error);
  }
}

// Test same-type floating point assignments
TEST(ArgumentSetter, FloatingSameTypeConversions) {
  // float to float
  {
    Parameter<float> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = 1.23456f;
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_FLOAT_EQ(param.get(), 1.23456f);
  }

  // double to double
  {
    Parameter<double> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = 1.23456789012345;
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_DOUBLE_EQ(param.get(), 1.23456789012345);
  }
}

// Test integer to floating-point conversions
TEST(ArgumentSetter, IntegerToFloatingPointConversions) {
  // int8_t to float
  {
    Parameter<float> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<int8_t>(-42);
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_FLOAT_EQ(param.get(), -42.0f);
  }

  // int8_t to double
  {
    Parameter<double> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<int8_t>(127);
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_DOUBLE_EQ(param.get(), 127.0);
  }

  // int16_t to float
  {
    Parameter<float> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<int16_t>(-12345);
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_FLOAT_EQ(param.get(), -12345.0f);
  }

  // int16_t to double
  {
    Parameter<double> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<int16_t>(32767);
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_DOUBLE_EQ(param.get(), 32767.0);
  }

  // int32_t to float
  {
    Parameter<float> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<int32_t>(1000000);
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_FLOAT_EQ(param.get(), 1000000.0f);
  }

  // int32_t to double
  {
    Parameter<double> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<int32_t>(-2147483648);
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_DOUBLE_EQ(param.get(), -2147483648.0);
  }

  // int64_t to float (within exact representation range)
  {
    Parameter<float> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<int64_t>(16777216);  // 2^24, exactly representable in float
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_FLOAT_EQ(param.get(), 16777216.0f);
  }

  // int64_t to double
  {
    Parameter<double> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<int64_t>(9007199254740992LL);  // 2^53, exactly representable in double
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_DOUBLE_EQ(param.get(), 9007199254740992.0);
  }

  // uint8_t to float
  {
    Parameter<float> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<uint8_t>(255);
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_FLOAT_EQ(param.get(), 255.0f);
  }

  // uint8_t to double
  {
    Parameter<double> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<uint8_t>(128);
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_DOUBLE_EQ(param.get(), 128.0);
  }

  // uint16_t to float
  {
    Parameter<float> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<uint16_t>(65535);
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_FLOAT_EQ(param.get(), 65535.0f);
  }

  // uint16_t to double
  {
    Parameter<double> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<uint16_t>(32768);
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_DOUBLE_EQ(param.get(), 32768.0);
  }

  // uint32_t to float
  {
    Parameter<float> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<uint32_t>(16777216);  // 2^24, exactly representable in float
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_FLOAT_EQ(param.get(), 16777216.0f);
  }

  // uint32_t to double
  {
    Parameter<double> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<uint32_t>(4294967295U);
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_DOUBLE_EQ(param.get(), 4294967295.0);
  }

  // uint64_t to float
  {
    Parameter<float> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<uint64_t>(1048576);
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_FLOAT_EQ(param.get(), 1048576.0f);
  }

  // uint64_t to double
  {
    Parameter<double> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<uint64_t>(9007199254740992ULL);  // 2^53, exactly representable in double
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_DOUBLE_EQ(param.get(), 9007199254740992.0);
  }

  // Zero conversions
  {
    Parameter<float> param_float;
    ParameterWrapper param_wrap_float(param_float);
    Arg arg_zero("value");
    arg_zero = static_cast<int32_t>(0);
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap_float, arg_zero));
    EXPECT_FLOAT_EQ(param_float.get(), 0.0f);

    Parameter<double> param_double;
    ParameterWrapper param_wrap_double(param_double);
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap_double, arg_zero));
    EXPECT_DOUBLE_EQ(param_double.get(), 0.0);
  }

  // Negative int32_t to float
  {
    Parameter<float> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<int32_t>(-999999);
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_FLOAT_EQ(param.get(), -999999.0f);
  }

  // Negative int64_t to double
  {
    Parameter<double> param;
    ParameterWrapper param_wrap(param);
    Arg arg("value");
    arg = static_cast<int64_t>(-1234567890123LL);
    EXPECT_NO_THROW(ArgumentSetter::set_param(param_wrap, arg));
    EXPECT_DOUBLE_EQ(param.get(), -1234567890123.0);
  }
}

}  // namespace holoscan
